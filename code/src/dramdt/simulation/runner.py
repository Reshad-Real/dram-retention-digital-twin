"""NGSpice deck execution and raw-output parsing.

Two backends are provided:

``SpiceRunner(backend="subprocess")``
    Spawns ``ngspice -b -o <log> <deck>`` in a private scratch directory.  This
    is the production backend: NGSpice keeps global state in process memory, so
    only process isolation makes a 50k-sample campaign safely parallel.

``SpiceRunner(backend="pyspice-shared")``
    Drives the ``ngspice`` shared library through
    :class:`PySpice.Spice.NgSpice.Shared.NgSpiceShared`.  Convenient for
    interactive work; used by the cross-backend equivalence check so that the
    two paths are demonstrated to agree rather than assumed to.

Parsing is deliberately tolerant.  A failed ``.meas`` in NGSpice does not abort
the deck -- it prints a diagnostic and moves on -- so the parser records which
measurements succeeded, which failed, and why, and never silently substitutes a
value.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..env import SpiceEnvironment, resolve_environment
from ..logging_utils import get_logger
from ..models.cell import MEAS_BEGIN, MEAS_END

__all__ = ["SpiceRunner", "SimulationResult", "SpiceError",
           "parse_measurements", "parse_leakage_file"]

log = get_logger(__name__)


class SpiceError(RuntimeError):
    """Raised when NGSpice cannot be executed at all (not for failed measures)."""


# --------------------------------------------------------------------------
# output parsing
# --------------------------------------------------------------------------
# `meas` echoes e.g.  "t_read              =  4.96638e-09 targ= ... trig= ..."
# `print` echoes e.g. "i(vvdd) = -3.707400e-12"
_VALUE_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_()\[\]#.]*)\s*=\s*"
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*)$")
_FAIL_RE = re.compile(r"^\s*(?:Error:\s*)?meas(?:ure)?\b.*?\b([A-Za-z_]\w*)\b.*?(?:failed|out of interval)",
                      re.IGNORECASE)
_FAILED_STMT_RE = re.compile(r"^\s*meas\s+\w+\s+(\w+)\b.*failed!\s*$", re.IGNORECASE)

_CONVERGENCE_MARKERS = (
    "no convergence", "singular matrix", "timestep too small",
    "iteration limit reached", "transient analysis failed", "doAnalyses: ",
)


def _normalise_key(key: str) -> str:
    """``i(vvdd)`` -> ``i_vvdd``; ``vvdd#branch`` -> ``vvdd_branch``."""
    k = key.strip().lower()
    k = k.replace("#", "_").replace("(", "_").replace(")", "").replace("[", "_").replace("]", "")
    k = re.sub(r"[^a-z0-9_]+", "_", k)
    return re.sub(r"_+", "_", k).strip("_")


def parse_measurements(text: str, strict_region: bool = True) -> tuple[dict[str, float], dict[str, str]]:
    """Extract ``name -> value`` pairs and ``name -> reason`` failures from a log.

    Parameters
    ----------
    strict_region
        When true (default) only lines between the ``MEAS_BEGIN``/``MEAS_END``
        markers plus explicit ``meas`` echoes are considered, which keeps the
        operating-point node dump from polluting the measurement namespace.
    """
    values: dict[str, float] = {}
    failures: dict[str, str] = {}

    in_region = not strict_region
    for line in text.splitlines():
        stripped = line.strip()
        if MEAS_BEGIN in stripped:
            in_region = True
            continue
        if MEAS_END in stripped:
            in_region = False
            continue

        low = stripped.lower()
        if low.startswith("error: measure") or "failed!" in low:
            m = _FAILED_STMT_RE.match(stripped) or _FAIL_RE.match(stripped)
            name = _normalise_key(m.group(1)) if m else "<unknown>"
            failures.setdefault(name, stripped)
            continue

        # `meas` echoes appear outside the marker region too; accept them always.
        m = _VALUE_RE.match(line)
        if not m:
            continue
        key, raw, _rest = m.groups()
        if not in_region and not _looks_like_meas_echo(line):
            continue
        try:
            values[_normalise_key(key)] = float(raw)
        except ValueError:                                  # pragma: no cover
            continue

    return values, failures


def _looks_like_meas_echo(line: str) -> bool:
    """A `.meas` echo is padded and often carries targ=/trig=/from=/to= tails."""
    return bool(re.search(r"\b(targ|trig|from|to)\s*=", line)) or \
        bool(re.match(r"^\s*[a-z_]\w*\s{2,}=", line, re.IGNORECASE))


def parse_leakage_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read an NGSpice ``wrdata`` file holding ``V_SN, I(Vsnp)`` columns.

    Returns ``(v_sn, i_out)`` where ``i_out`` is the current *leaving* the
    storage node (positive == the node is discharging).  NGSpice reports the
    current flowing into the probe source's positive terminal, so the sign is
    inverted here once, in one place.
    """
    p = Path(path)
    if not p.exists():
        return np.empty(0), np.empty(0)
    rows: list[tuple[float, float]] = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            rows.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    if not rows:
        return np.empty(0), np.empty(0)
    arr = np.asarray(rows, dtype=float)
    v = arr[:, 0]
    i_into_source = arr[:, 1]
    return v, -i_into_source


# --------------------------------------------------------------------------
@dataclass
class SimulationResult:
    """Everything one NGSpice invocation produced."""

    ok: bool
    measurements: dict[str, float] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    leakage_v: np.ndarray = field(default_factory=lambda: np.empty(0))
    leakage_i: np.ndarray = field(default_factory=lambda: np.empty(0))
    runtime_s: float = 0.0
    returncode: int = 0
    backend: str = "subprocess"
    log_excerpt: str = ""
    error: str | None = None
    converged: bool = True
    deck_path: str | None = None

    def get(self, key: str, default: float = float("nan")) -> float:
        return self.measurements.get(key, default)

    @property
    def has_leakage_curve(self) -> bool:
        return self.leakage_v.size > 1 and self.leakage_i.size == self.leakage_v.size


# --------------------------------------------------------------------------
class SpiceRunner:
    """Executes NGSpice decks and returns parsed :class:`SimulationResult`."""

    def __init__(self,
                 environment: SpiceEnvironment | None = None,
                 backend: str = "subprocess",
                 timeout_s: float = 300.0,
                 keep_artifacts: bool = False,
                 scratch_root: str | Path | None = None,
                 reuse_workdir: bool = True):
        self.env = environment or resolve_environment()
        self.backend = backend
        self.timeout_s = float(timeout_s)
        self.keep_artifacts = bool(keep_artifacts)
        self.scratch_root = Path(scratch_root) if scratch_root else None
        if self.scratch_root:
            self.scratch_root.mkdir(parents=True, exist_ok=True)
        # Creating and deleting a directory per simulation dominates the
        # wall-clock cost on Windows (NTFS metadata + on-access AV scanning) and
        # caps campaign throughput well below CPU saturation.  Reusing one
        # directory per process removes that churn entirely.
        self.reuse_workdir = bool(reuse_workdir) and not keep_artifacts
        self._persistent: Path | None = None
        self._shared = None

        if backend == "subprocess" and not self.env.has_exe:
            raise SpiceError("subprocess backend requires an ngspice executable")
        if backend == "pyspice-shared" and not self.env.has_dll:
            raise SpiceError("pyspice-shared backend requires the ngspice shared library")

    # ------------------------------------------------------------------
    def run(self, netlist: str, leak_file: str = "leak.csv",
            tag: str = "run") -> SimulationResult:
        if self.backend == "subprocess":
            return self._run_subprocess(netlist, leak_file, tag)
        if self.backend == "pyspice-shared":
            return self._run_shared(netlist, tag)
        raise SpiceError(f"Unknown backend {self.backend!r}")

    # ------------------------------------------------------------------
    def _workdir_for(self, tag: str) -> Path:
        if not self.reuse_workdir:
            return Path(tempfile.mkdtemp(prefix=f"dramdt_{tag}_", dir=self.scratch_root))
        if self._persistent is None:
            self._persistent = Path(tempfile.mkdtemp(
                prefix=f"dramdt_w{os.getpid()}_", dir=self.scratch_root))
        return self._persistent

    def _run_subprocess(self, netlist: str, leak_file: str, tag: str) -> SimulationResult:
        workdir = self._workdir_for(tag)
        deck = workdir / "deck.cir"
        logf = workdir / "deck.log"
        leak = workdir / leak_file
        # A reused directory must never let one run read the previous run's
        # artefacts, so stale outputs are removed before every invocation.
        for stale in (logf, leak):
            try:
                stale.unlink()
            except FileNotFoundError:
                pass
            except OSError:                                 # pragma: no cover
                pass
        deck.write_text(netlist, encoding="utf-8")

        t0 = time.perf_counter()
        rc, out, err, timed_out = 0, "", "", False
        try:
            proc = subprocess.run(
                [str(self.env.exe), "-b", "-o", str(logf), str(deck)],
                cwd=str(workdir), capture_output=True, text=True,
                # Without an explicit DEVNULL, ngspice can block forever on a
                # prompt when the parent's stdin is not a console (which is the
                # case for a detached/batch campaign run).
                stdin=subprocess.DEVNULL,
                timeout=self.timeout_s, env=self.env.subprocess_env())
            rc, out, err = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            timed_out = True
            rc = -9
        except OSError as exc:
            if not self.reuse_workdir:
                shutil.rmtree(workdir, ignore_errors=True)
            raise SpiceError(f"Failed to launch ngspice: {exc}") from exc
        runtime = time.perf_counter() - t0

        text = ""
        if logf.exists():
            text = logf.read_text(encoding="utf-8", errors="replace")
        text = text + "\n" + out + "\n" + err

        values, failures = parse_measurements(text)
        lv, li = parse_leakage_file(workdir / leak_file)
        converged = not any(m in text.lower() for m in _CONVERGENCE_MARKERS)

        result = SimulationResult(
            ok=(not timed_out) and (rc == 0) and bool(values),
            measurements=values, failures=failures,
            leakage_v=lv, leakage_i=li,
            runtime_s=runtime, returncode=rc, backend="subprocess",
            log_excerpt=text[-4000:], converged=converged,
            error=("timeout" if timed_out else (None if rc == 0 else f"ngspice rc={rc}")),
            deck_path=str(deck) if self.keep_artifacts else None,
        )
        if not self.keep_artifacts and not self.reuse_workdir:
            shutil.rmtree(workdir, ignore_errors=True)
        return result

    # ------------------------------------------------------------------
    def _ensure_shared(self):
        if self._shared is None:
            self.env.activate_shared()
            from PySpice.Spice.NgSpice.Shared import NgSpiceShared  # noqa: PLC0415
            self._shared = NgSpiceShared.new_instance()
        return self._shared

    @staticmethod
    def split_control_block(netlist: str) -> tuple[str, list[str]]:
        """Separate a deck into its circuit body and its ``.control`` commands.

        The shared library does **not** execute a ``.control`` block on ``run``
        -- it reports "No job (tran, ac, op etc.) defined" -- so the commands
        have to be issued one at a time through the command interpreter.
        """
        body: list[str] = []
        commands: list[str] = []
        in_control = False
        for raw in netlist.splitlines():
            s = raw.strip()
            low = s.lower()
            if low.startswith(".control"):
                in_control = True
                continue
            if low.startswith(".endc"):
                in_control = False
                continue
            if in_control:
                if s and not s.startswith("*"):
                    commands.append(s)
            else:
                if not low.startswith(".end"):
                    body.append(raw)
        body.append(".end")
        return "\n".join(body) + "\n", commands

    def _run_shared(self, netlist: str, tag: str) -> SimulationResult:
        """Run through PySpice's shared-library binding.

        The shared library writes ``wrdata`` files relative to the process CWD
        and keeps one global circuit, so this backend is single-threaded by
        construction and used for validation rather than for the campaign.
        """
        ng = self._ensure_shared()
        workdir = Path(tempfile.mkdtemp(prefix=f"dramdt_sh_{tag}_", dir=self.scratch_root))
        cwd = os.getcwd()
        t0 = time.perf_counter()
        error: str | None = None
        chunks: list[str] = []
        skip_leak = False
        try:
            os.chdir(workdir)
            try:
                ng.remove_circuit()
            except Exception:
                pass
            body, commands = self.split_control_block(netlist)
            ng.load_circuit(body)
            dc_ok = True
            for cmd in commands:
                try:
                    out = ng.exec_command(cmd)
                    if out:
                        chunks.append(out if isinstance(out, str)
                                      else "\n".join(map(str, out)))
                except Exception as exc:
                    # A failing `meas` is a normal outcome, not a fatal error.
                    chunks.append(f"command failed: {cmd} -> {exc}")
                    if cmd.lower().startswith("dc "):
                        dc_ok = False
            if not dc_ok:
                # Known limitation of PySpice 1.5 + ngspice 46: a `dc` sweep
                # issued through the shared library's command interface fails
                # even on a trivial circuit.  Suppress the leakage sweep rather
                # than parse whatever plot `wrdata` last wrote, which would be
                # the transient waveform masquerading as a leakage curve.
                skip_leak = True
                error = ("dc sweep unsupported by the shared-library backend; "
                         "leakage characteristic not produced")
            stdout = ng.stdout or ""
            text = "\n".join(chunks) + "\n" + stdout
        except Exception as exc:
            text, error = "\n".join(chunks), f"{type(exc).__name__}: {exc}"
        finally:
            os.chdir(cwd)
        runtime = time.perf_counter() - t0

        values, failures = parse_measurements(text)
        lv, li = ((np.empty(0), np.empty(0)) if skip_leak
                  else parse_leakage_file(workdir / "leak.csv"))
        if not self.keep_artifacts:
            shutil.rmtree(workdir, ignore_errors=True)

        return SimulationResult(
            # The transient/OP measurements are valid even when the DC sweep is
            # unavailable, so a missing leakage curve does not make the run bad.
            ok=bool(values), measurements=values, failures=failures,
            leakage_v=lv, leakage_i=li, runtime_s=runtime, backend="pyspice-shared",
            log_excerpt=text[-4000:], error=error,
        )

    # ------------------------------------------------------------------
    def close(self) -> None:
        if self._persistent is not None and not self.keep_artifacts:
            shutil.rmtree(self._persistent, ignore_errors=True)
            self._persistent = None
        if self._shared is not None:
            try:
                self._shared.remove_circuit()
            except Exception:                               # pragma: no cover
                pass
            self._shared = None

    def __enter__(self) -> "SpiceRunner":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
