"""Discovery and configuration of the NGSpice / PySpice execution environment.

Two execution backends are supported and both are exercised by the framework:

``subprocess``
    Launches ``ngspice -b`` as an isolated child process.  This is the
    production backend for the large sampling campaign because it is
    process-safe and therefore trivially parallelisable.

``pyspice-shared``
    Uses :class:`PySpice.Spice.NgSpice.Shared.NgSpiceShared`, i.e. the
    ``ngspice`` shared library bound through PySpice.  A single NGSpice
    library instance is a per-process singleton, so this backend is used for
    interactive work and for the cross-backend validation study.

Resolution order for the NGSpice binaries:
    1. explicit argument / config value,
    2. ``DRAMDT_NGSPICE_EXE`` / ``DRAMDT_NGSPICE_DLL`` environment variables,
    3. the vendored copy under ``<project>/tools/ng``,
    4. whatever is on ``PATH``.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .paths import TOOLS_DIR

__all__ = ["SpiceEnvironment", "resolve_environment", "NgSpiceNotFound"]


class NgSpiceNotFound(RuntimeError):
    """Raised when no usable NGSpice installation can be located."""


# ``ngspice_con.exe`` is preferred over ``ngspice.exe`` on Windows: the latter
# is linked against the GDI plotting front-end and blocks indefinitely when it
# is started without an interactive window station, which is exactly the case
# for a detached batch campaign.  The console build has no such dependency.
_VENDOR_EXE_CANDIDATES = (
    TOOLS_DIR / "ng" / "Spice64" / "bin" / "ngspice_con.exe",
    TOOLS_DIR / "ng" / "Spice64" / "bin" / "ngspice.exe",
    TOOLS_DIR / "ng" / "bin" / "ngspice",
)
_VENDOR_DLL_CANDIDATES = (
    TOOLS_DIR / "ng" / "Spice64_dll" / "dll-vs" / "ngspice.dll",
    TOOLS_DIR / "ng" / "Spice64_dll" / "bin" / "ngspice.dll",
    TOOLS_DIR / "ng" / "lib" / "libngspice.so",
    TOOLS_DIR / "ng" / "lib" / "libngspice.dylib",
)


@dataclass
class SpiceEnvironment:
    """Everything needed to invoke NGSpice, in either backend."""

    exe: Path | None = None
    dll: Path | None = None
    lib_dir: Path | None = None          # SPICE_LIB_DIR for the shared library
    version: str = "unknown"
    extra_env: dict[str, str] = field(default_factory=dict)

    # -- introspection ----------------------------------------------------
    @property
    def has_exe(self) -> bool:
        return self.exe is not None and Path(self.exe).exists()

    @property
    def has_dll(self) -> bool:
        return self.dll is not None and Path(self.dll).exists()

    def describe(self) -> dict[str, str]:
        return {
            "ngspice_exe": str(self.exe) if self.exe else "<none>",
            "ngspice_dll": str(self.dll) if self.dll else "<none>",
            "spice_lib_dir": str(self.lib_dir) if self.lib_dir else "<none>",
            "ngspice_version": self.version,
            "platform": platform.platform(),
        }

    # -- backends ---------------------------------------------------------
    def subprocess_env(self) -> dict[str, str]:
        """Environment mapping for ``ngspice -b`` child processes."""
        env = dict(os.environ)
        if self.lib_dir is not None:
            env["SPICE_LIB_DIR"] = str(self.lib_dir)
        env.update(self.extra_env)
        return env

    def activate_shared(self) -> None:
        """Prepare the current process for the PySpice shared-library backend.

        Must be called *before* ``PySpice.Spice.NgSpice.Shared`` is imported.
        """
        if not self.has_dll:
            raise NgSpiceNotFound("ngspice shared library not available")
        dll = Path(self.dll)
        os.environ["NGSPICE_LIBRARY_PATH"] = str(dll)
        # Prefer the shared library's *own* share/ngspice tree: its spinit
        # references code-model paths relative to that distribution, and
        # pointing it at the executable's tree produces a stream of
        # "code model couldn't be loaded" errors.
        own_lib = _lib_dir_for(dll)
        lib = own_lib or self.lib_dir
        if lib is not None:
            os.environ["SPICE_LIB_DIR"] = str(lib)
        if hasattr(os, "add_dll_directory") and dll.parent.exists():
            try:
                os.add_dll_directory(str(dll.parent))
            except (OSError, AttributeError):      # pragma: no cover
                pass


_VERSION_RE = re.compile(r"ngspice[- ]?(\d+[\w.]*)", re.IGNORECASE)


def _probe_version(exe: Path) -> str:
    """Read the simulator's version banner.

    ``--version`` prints the banner directly; a batch run does not, so the
    banner has to come from the dedicated flag rather than from a run log
    (whose last lines are memory statistics, not a version).
    """
    try:
        proc = subprocess.run([str(exe), "--version"], capture_output=True,
                              text=True, timeout=60)
        for line in (proc.stdout + "\n" + proc.stderr).splitlines():
            m = _VERSION_RE.search(line)
            if m:
                return line.strip(" *\t")
    except Exception:                               # pragma: no cover
        pass
    # Fall back to confirming the binary can actually run a deck.
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as td:
            cir = Path(td) / "probe.cir"
            log = Path(td) / "probe.log"
            cir.write_text("* ngspice probe\nV1 a 0 DC 1\nR1 a 0 1k\n.op\n.end\n",
                           encoding="utf-8")
            subprocess.run([str(exe), "-b", "-o", str(log), str(cir)],
                           capture_output=True, text=True, timeout=120, cwd=td,
                           stdin=subprocess.DEVNULL)
            if log.exists() and "TEMP" in log.read_text(encoding="utf-8",
                                                        errors="replace"):
                return "ngspice (runs; version banner not reported)"
    except Exception as exc:                        # pragma: no cover
        return f"probe failed: {exc}"
    return "ngspice (version unknown)"


def _first_existing(candidates) -> Path | None:
    for c in candidates:
        if c is not None and Path(c).exists():
            return Path(c)
    return None


def _lib_dir_for(binary: Path | None) -> Path | None:
    """Locate the ``share/ngspice`` directory belonging to a binary/DLL."""
    if binary is None:
        return None
    for parent in Path(binary).resolve().parents:
        cand = parent / "share" / "ngspice"
        if cand.is_dir():
            return cand
    return None


def resolve_environment(exe: str | os.PathLike | None = None,
                        dll: str | os.PathLike | None = None,
                        probe: bool = True) -> SpiceEnvironment:
    """Locate NGSpice and return a ready-to-use :class:`SpiceEnvironment`."""
    exe_path = (Path(exe) if exe else None)
    if exe_path is None or not exe_path.exists():
        env_exe = os.environ.get("DRAMDT_NGSPICE_EXE")
        exe_path = Path(env_exe) if env_exe and Path(env_exe).exists() else None
    if exe_path is None:
        exe_path = _first_existing(_VENDOR_EXE_CANDIDATES)
    if exe_path is None:
        which = shutil.which("ngspice_con") or shutil.which("ngspice")
        exe_path = Path(which) if which else None

    dll_path = (Path(dll) if dll else None)
    if dll_path is None or not dll_path.exists():
        env_dll = os.environ.get("DRAMDT_NGSPICE_DLL")
        dll_path = Path(env_dll) if env_dll and Path(env_dll).exists() else None
    if dll_path is None:
        dll_path = _first_existing(_VENDOR_DLL_CANDIDATES)

    if exe_path is None and dll_path is None:
        raise NgSpiceNotFound(
            "No NGSpice installation found.  Run `python scripts/00_setup_environment.py` "
            "to download a vendored copy, install ngspice from your package manager, or "
            "set DRAMDT_NGSPICE_EXE / DRAMDT_NGSPICE_DLL."
        )

    lib_dir = _lib_dir_for(exe_path) or _lib_dir_for(dll_path)
    version = _probe_version(exe_path) if (probe and exe_path) else "not probed"

    return SpiceEnvironment(exe=exe_path, dll=dll_path, lib_dir=lib_dir, version=version)
