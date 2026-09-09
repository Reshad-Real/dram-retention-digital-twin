"""Stage 2/3 -- the parallel NGSpice sampling campaign.

Design points are decoded from a space-filling design, simulated in worker
*processes* (NGSpice keeps global state, so threads are not an option) and
written to disk in chunks.  A campaign is restartable: completed chunks are
detected on disk and skipped, so a 50,000-point run survives interruption.

Determinism
-----------
The design matrix is a pure function of the master seed, and every per-sample
random draw (mismatch offsets) is derived from ``(master_seed, label, index)``.
Results therefore do not depend on the number of workers or on completion
order, which is asserted by ``tests/test_reproducibility.py``.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..config import Config
from ..doe.sampling import DesignMatrix, generate_design
from ..env import SpiceEnvironment, resolve_environment
from ..logging_utils import get_logger
from ..models.cell import CellParameters, DramCellBuilder
from ..models.technology import Technology
from ..paths import INTERIM_DIR, RAW_DIR
from ..seeds import derive_seed, rng_for
from ..simulation.measure import MetricExtractor
from ..simulation.runner import SpiceRunner, parse_measurements

__all__ = ["SimulationCampaign", "CampaignResult", "simulate_one",
           "RECORD_PROVENANCE_FIELDS"]

log = get_logger(__name__)

RECORD_PROVENANCE_FIELDS = (
    "sample_id", "assist_config", "corner", "config_hash", "model_fingerprint",
    "backend", "worker_pid",
)

# --------------------------------------------------------------------------
# per-worker state (created once per process by the pool initialiser)
# --------------------------------------------------------------------------
_W: dict[str, Any] = {}


def _worker_init(cfg_dict: dict[str, Any],
                 env_desc: dict[str, str],
                 keep_artifacts: bool,
                 scratch_root: str | None) -> None:
    """Build the per-process simulation stack exactly once."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    cfg = Config(cfg_dict)
    env = resolve_environment(exe=env_desc.get("ngspice_exe"),
                              dll=env_desc.get("ngspice_dll"),
                              probe=False)
    # Corner cards are generated once in the parent; workers only read them.
    tech = Technology.from_config(cfg, generate=False)
    tech.corner_cards = {k: Path(v) for k, v in (env_desc.get("corner_cards") or {}).items()}

    sim_cfg = cfg.get("simulation", {})
    _W.update(
        cfg=cfg,
        tech=tech,
        builder=DramCellBuilder(cfg, tech),
        extractor=MetricExtractor(cfg),
        runner=SpiceRunner(env, backend=str(sim_cfg.get("backend", "subprocess")),
                           timeout_s=float(sim_cfg.get("timeout_s", 300)),
                           keep_artifacts=keep_artifacts,
                           scratch_root=scratch_root),
        config_hash=cfg.hash,
        model_fingerprint=tech.fingerprint(),
    )
    # Each worker keeps one persistent scratch directory; release it when the
    # process exits so a long campaign does not leave hundreds behind.
    import atexit
    atexit.register(lambda: _W["runner"].close() if "runner" in _W else None)


def simulate_one(sample: Mapping[str, Any], sample_id: int,
                 mismatch: Mapping[str, float] | None = None,
                 state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Simulate one decoded design point and return a flat record.

    Importable and callable outside the pool (``state`` overrides the worker
    globals), which is what the unit tests use.
    """
    st = state or _W
    builder: DramCellBuilder = st["builder"]
    extractor: MetricExtractor = st["extractor"]
    runner: SpiceRunner = st["runner"]

    p: CellParameters = builder.parameters_from_sample(sample, sample_id=sample_id)
    if mismatch:
        p.delvto = dict(mismatch)

    deck = builder.build_characterization_netlist(p)
    result = runner.run(deck, tag=f"s{sample_id}")
    metrics = extractor.extract(result, p)

    record: dict[str, Any] = {"sample_id": int(sample_id)}
    # design-space values (as decoded, including derived quantities)
    for k, v in sample.items():
        record[k] = v
    # resolved physical parameters that are not already in the sample
    for k, v in p.to_dict().items():
        record.setdefault(k, v)
    if mismatch:
        for k, v in mismatch.items():
            record[f"delvto_{k}"] = v
    record.update(metrics.to_record())
    record.update({
        "config_hash": st.get("config_hash", ""),
        "model_fingerprint": st.get("model_fingerprint", ""),
        "backend": result.backend,
        "worker_pid": os.getpid(),
        "n_measurements": len(result.measurements),
        "n_failed_measurements": len(result.failures),
    })
    return record


def _simulate_chunk(args: tuple[int, list[tuple[int, dict[str, Any], dict[str, float] | None]]]
                    ) -> tuple[int, list[dict[str, Any]]]:
    """Worker entry point: simulate a contiguous block of samples."""
    chunk_id, items = args
    out: list[dict[str, Any]] = []
    for sample_id, sample, mismatch in items:
        try:
            out.append(simulate_one(sample, sample_id, mismatch))
        except Exception as exc:                            # keep the campaign alive
            rec = {"sample_id": int(sample_id), "simulation_ok": False,
                   "notes": f"worker exception: {type(exc).__name__}: {exc}"}
            rec.update({k: v for k, v in sample.items()})
            out.append(rec)
    return chunk_id, out


# --------------------------------------------------------------------------
@dataclass
class CampaignResult:
    """Outcome of a completed campaign."""

    frame: pd.DataFrame
    design: DesignMatrix
    n_requested: int
    n_completed: int
    n_ok: int
    duration_s: float
    chunk_dir: Path
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.n_ok / max(self.n_completed, 1)

    def describe(self) -> dict[str, Any]:
        return {
            "n_requested": self.n_requested,
            "n_completed": self.n_completed,
            "n_simulations_ok": self.n_ok,
            "success_rate": round(self.success_rate, 6),
            "duration_s": round(self.duration_s, 2),
            "samples_per_second": round(self.n_completed / max(self.duration_s, 1e-9), 3),
            **self.design.describe(),
            **self.diagnostics,
        }


class SimulationCampaign:
    """Runs a space-filling sampling campaign over the configured design space."""

    def __init__(self, config: Config,
                 environment: SpiceEnvironment | None = None,
                 chunk_dir: str | Path | None = None,
                 keep_artifacts: bool = False,
                 scratch_root: str | Path | None = None):
        self.cfg = config
        self.env = environment or resolve_environment(probe=False)
        self.exp = dict(config.get("experiment", {}))
        self.chunk_dir = Path(chunk_dir) if chunk_dir else INTERIM_DIR / "chunks"
        self.chunk_dir.mkdir(parents=True, exist_ok=True)
        self.keep_artifacts = keep_artifacts
        self.scratch_root = str(scratch_root) if scratch_root else None
        # Generate corner model cards once, in the parent process.
        self.tech = Technology.from_config(config, generate=True)

    # ------------------------------------------------------------------
    @property
    def n_workers(self) -> int:
        n = int(self.exp.get("n_workers", 0) or 0)
        if n > 0:
            return n
        return max((os.cpu_count() or 4) - 2, 1)

    def _design(self, n_samples: int) -> DesignMatrix:
        ds = self.cfg.design_space
        return generate_design(
            n=n_samples, d=ds.n_dim,
            sampler=str(self.exp.get("sampler", "lhs")),
            seed=int(self.exp.get("seed", 0)),
            label="main_campaign",
            criterion=str(self.exp.get("lhs_criterion", "maximin")),
            iterations=int(self.exp.get("lhs_iterations", 20)))

    def _mismatch_for(self, p_like: Mapping[str, Any], index: int) -> dict[str, float] | None:
        """Zero-mean mismatch offsets; ``None`` when mismatch is disabled."""
        if not bool(self.exp.get("apply_nominal_mismatch", False)):
            return None
        rng = rng_for(int(self.exp.get("seed", 0)), "mismatch", index)
        wsan = float(p_like.get("wsan", 180e-9))
        lsa = float(p_like.get("lsa", 45e-9))
        devices = {
            "access": (float(p_like.get("wacc", 90e-9)), float(p_like.get("lacc", 45e-9))),
            "sa_n1": (wsan, lsa), "sa_n2": (wsan, lsa),
            "sa_p1": (float(p_like.get("wsap", wsan)), lsa),
            "sa_p2": (float(p_like.get("wsap", wsan)), lsa),
        }
        return self.tech.sample_mismatch(devices, rng)

    # ------------------------------------------------------------------
    def run(self, n_samples: int | None = None,
            resume: bool = True,
            progress_every: int = 20) -> CampaignResult:
        """Execute the campaign, returning the assembled dataframe."""
        n = int(n_samples if n_samples is not None else self.exp.get("n_samples", 1000))
        design = self._design(n)
        ds = self.cfg.design_space

        chunk_size = max(int(self.exp.get("chunk_size", 250)), 1)
        n_chunks = math.ceil(n / chunk_size)
        workers = self.n_workers

        log.info("Campaign: %d samples, %d dimensions, %d chunks of %d, %d workers",
                 n, ds.n_dim, n_chunks, chunk_size, workers)
        log.info("NGSpice: %s", self.env.exe)

        env_desc = {
            "ngspice_exe": str(self.env.exe) if self.env.exe else "",
            "ngspice_dll": str(self.env.dll) if self.env.dll else "",
            "corner_cards": {k: str(v) for k, v in self.tech.corner_cards.items()},
        }

        # --- assemble the work list, skipping chunks already on disk -----
        pending: list[tuple[int, list[tuple[int, dict[str, Any], dict[str, float] | None]]]] = []
        done_chunks: list[Path] = []
        for c in range(n_chunks):
            path = self.chunk_dir / f"chunk_{c:05d}.parquet"
            if resume and path.exists():
                done_chunks.append(path)
                continue
            lo, hi = c * chunk_size, min((c + 1) * chunk_size, n)
            items = []
            for i in range(lo, hi):
                sample = ds.decode_row(design.points[i])
                items.append((i, sample, self._mismatch_for(sample, i)))
            pending.append((c, items))

        if done_chunks:
            log.info("Resuming: %d/%d chunks already complete", len(done_chunks), n_chunks)

        t0 = time.perf_counter()
        completed = len(done_chunks)

        if pending:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_worker_init,
                initargs=(dict(self.cfg), env_desc, self.keep_artifacts, self.scratch_root),
            ) as pool:
                futures = {pool.submit(_simulate_chunk, item): item[0] for item in pending}
                for k, fut in enumerate(as_completed(futures), start=1):
                    chunk_id, records = fut.result()
                    path = self.chunk_dir / f"chunk_{chunk_id:05d}.parquet"
                    df_chunk = pd.DataFrame.from_records(records)
                    try:
                        df_chunk.to_parquet(path, index=False)
                    except Exception:                       # no parquet engine
                        path = path.with_suffix(".csv")
                        df_chunk.to_csv(path, index=False)
                    done_chunks.append(path)
                    completed += 1
                    if k % progress_every == 0 or k == len(pending):
                        el = time.perf_counter() - t0
                        rate = (completed * chunk_size) / max(el, 1e-9)
                        remaining = (n_chunks - completed) * chunk_size / max(rate, 1e-9)
                        log.info("  chunk %d/%d | %.0f samples/s | elapsed %.0fs | eta %.0fs",
                                 completed, n_chunks, rate, el, remaining)

        duration = time.perf_counter() - t0

        # --- assemble ----------------------------------------------------
        frames = []
        for path in sorted(set(done_chunks)):
            frames.append(pd.read_parquet(path) if path.suffix == ".parquet"
                          else pd.read_csv(path))
        frame = (pd.concat(frames, ignore_index=True).sort_values("sample_id")
                 .reset_index(drop=True)) if frames else pd.DataFrame()

        n_ok = int(frame.get("simulation_ok", pd.Series(dtype=bool)).sum()) if len(frame) else 0
        log.info("Campaign complete: %d records, %d successful (%.2f %%) in %.1f s",
                 len(frame), n_ok, 100.0 * n_ok / max(len(frame), 1), duration)

        return CampaignResult(
            frame=frame, design=design, n_requested=n, n_completed=len(frame),
            n_ok=n_ok, duration_s=duration, chunk_dir=self.chunk_dir,
            diagnostics={"n_workers": workers, "chunk_size": chunk_size,
                         "config_hash": self.cfg.hash,
                         "model_fingerprint": self.tech.fingerprint(),
                         "ngspice": str(self.env.exe)})

    # ------------------------------------------------------------------
    def run_retention_validation(self, frame: pd.DataFrame,
                                 fraction: float | None = None,
                                 window_s: float | None = None,
                                 max_points: int = 400) -> pd.DataFrame:
        """Re-simulate a subset with a *direct* long transient.

        This is the empirical check on the quasi-static retention model: it
        produces the agreement statistics reported in the results, rather than
        asserting that the model is adequate.
        """
        frac = float(fraction if fraction is not None
                     else self.exp.get("retention_validation_fraction", 0.004))
        cap = float(window_s if window_s is not None
                    else self.exp.get("retention_validation_window_s", 0.05))

        usable = frame[(frame.get("simulation_ok", False))
                       & np.isfinite(frame.get("retention_time_s", np.nan))
                       & (~frame.get("retention_censored", False).astype(bool))
                       & (frame.get("retention_time_s", 0) > 0)
                       & (frame.get("retention_time_s", 0) <= cap)]
        if usable.empty:
            log.warning("No samples eligible for direct retention validation "
                        "(window cap %.4g s)", cap)
            return pd.DataFrame()

        n_pick = min(max(int(len(frame) * frac), 10), max_points, len(usable))
        rng = np.random.default_rng(derive_seed(int(self.exp.get("seed", 0)), "ret_valid"))
        picks = usable.iloc[rng.choice(len(usable), n_pick, replace=False)]

        builder = DramCellBuilder(self.cfg, self.tech)
        sim_cfg = self.cfg.get("simulation", {})
        runner = SpiceRunner(self.env, timeout_s=float(sim_cfg.get("timeout_s", 300)) * 4)

        rows: list[dict[str, Any]] = []
        log.info("Direct-transient retention validation on %d design points", n_pick)
        for k, (_, row) in enumerate(picks.iterrows(), start=1):
            p = builder.parameters_from_sample(row.to_dict(), sample_id=int(row["sample_id"]))
            t_ret = float(row["retention_time_s"])
            v_init = float(row["v_sn_written_v"])
            v_fail = float(row["v_fail_v"])
            if not (np.isfinite(v_init) and np.isfinite(v_fail) and v_init > v_fail):
                continue
            deck = builder.build_retention_netlist(
                p, t_stop=t_ret * 2.5, v_fail=v_fail, n_points=4000, v_init=v_init)
            res = runner.run(deck, tag=f"ret{int(row['sample_id'])}")
            vals, _ = parse_measurements(res.log_excerpt)
            rows.append({
                "sample_id": int(row["sample_id"]),
                "retention_quasistatic_s": t_ret,
                "retention_transient_s": vals.get("t_ret_tran", float("nan")),
                "v_sn_end_v": vals.get("v_sn_end", float("nan")),
                "v_init_v": v_init, "v_fail_v": v_fail,
                "temperature_c": float(row.get("temperature_c", float("nan"))),
                "corner": row.get("corner", ""),
                "sim_runtime_s": res.runtime_s,
            })
            if k % 50 == 0:
                log.info("  %d/%d validated", k, n_pick)
        return pd.DataFrame(rows)
