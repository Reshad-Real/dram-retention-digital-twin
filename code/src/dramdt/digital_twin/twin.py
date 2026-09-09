"""The DRAM digital twin.

A :class:`DigitalTwin` bundles

* the configuration (so derived quantities are computed identically to training),
* one fitted surrogate per response,
* the feature-engineering transform,
* the inverse of any target transform,

behind a single ``predict`` call that maps *raw design variables* to *physical
responses*.  That is what makes it usable both interactively ("what if I raise
VPP by 100 mV?") and as the inner loop of the multi-objective search, where
several million evaluations would be impossible with a circuit simulator.

The twin is deliberately strict about provenance: it records the dataset hash,
the configuration hash and the per-target model identity it was built from, and
refuses to load a twin whose configuration hash does not match the one
requested unless that check is explicitly waived.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..config import Config
from ..dataset.preprocess import engineer_features
from ..logging_utils import get_logger

__all__ = ["DigitalTwin", "TwinPrediction", "load_twin"]

log = get_logger(__name__)


@dataclass
class TwinPrediction:
    """Predictions for a batch of designs, in physical units."""

    frame: pd.DataFrame
    latency_us_per_design: float
    n_designs: int
    targets: list[str]

    def __getitem__(self, key: str) -> pd.Series:
        return self.frame[key]

    def to_dict(self) -> list[dict[str, Any]]:
        return self.frame.to_dict(orient="records")


class DigitalTwin:
    """Deployed surrogate ensemble with the full input-preparation pipeline."""

    def __init__(self, config: Config,
                 models: Mapping[str, Any],
                 metadata: Mapping[str, Any] | None = None):
        self.cfg = config
        self.models: dict[str, Any] = dict(models)
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.metadata.setdefault("config_hash", config.hash)
        self._ds = config.design_space

        # Map each stored (possibly transformed) target back to a physical name.
        self.inverse: dict[str, tuple[str, str]] = {}
        for obj in config.get("objectives", []):
            name, tf = obj["name"], obj.get("transform")
            if tf == "log10":
                self.inverse[f"log10_{name}"] = (name, "log10")

    # ------------------------------------------------------------------
    @property
    def targets(self) -> list[str]:
        return list(self.models)

    @property
    def physical_targets(self) -> list[str]:
        return [self.inverse.get(t, (t, ""))[0] for t in self.models]

    # ------------------------------------------------------------------
    def prepare(self, designs: Sequence[Mapping[str, Any]] | pd.DataFrame) -> pd.DataFrame:
        """Raw design variables -> the exact feature frame the models expect.

        Missing variables are filled from the design-space constants and the
        ``architecture`` defaults, so a caller may specify only the knobs it
        cares about.
        """
        if isinstance(designs, pd.DataFrame):
            rows = designs.to_dict(orient="records")
        else:
            rows = [dict(d) for d in designs]

        arch = dict(self.cfg.get("architecture", {}))
        prepared: list[dict[str, Any]] = []
        for row in rows:
            params: dict[str, Any] = dict(self._ds.constants)
            for var in self._ds.variables:
                if var.name in row:
                    params[var.name] = row[var.name]
                elif var.name in arch:
                    params[var.name] = arch[var.name]
                else:
                    params[var.name] = var.decode(0.5)      # midpoint fallback
            params.update({k: v for k, v in row.items() if k not in params})
            prepared.append(self._ds.apply_derived(params))

        frame = pd.DataFrame(prepared)
        frame, _created = engineer_features(frame)
        return frame

    # ------------------------------------------------------------------
    def predict(self, designs: Sequence[Mapping[str, Any]] | pd.DataFrame,
                physical: bool = True,
                prepared: pd.DataFrame | None = None) -> TwinPrediction:
        """Predict every response for a batch of designs."""
        frame = prepared if prepared is not None else self.prepare(designs)
        n = len(frame)
        out = pd.DataFrame(index=frame.index)

        t0 = time.perf_counter()
        for target, surrogate in self.models.items():
            try:
                pred = surrogate.predict(frame)
            except Exception as exc:                        # pragma: no cover
                log.error("Twin prediction failed for %r: %s", target, exc)
                pred = np.full(n, np.nan)
            out[target] = pred
            if physical and target in self.inverse:
                name, kind = self.inverse[target]
                if kind == "log10":
                    out[name] = np.power(10.0, np.asarray(pred, dtype=float))
        elapsed = time.perf_counter() - t0

        for col in frame.columns:
            if col not in out.columns:
                out[col] = frame[col].to_numpy()

        return TwinPrediction(frame=out,
                              latency_us_per_design=1e6 * elapsed / max(n, 1),
                              n_designs=n, targets=list(self.models))

    def predict_one(self, **design: Any) -> dict[str, float]:
        """Convenience single-design call, returning a plain dict."""
        pred = self.predict([design])
        return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in pred.frame.iloc[0].items()}

    # ------------------------------------------------------------------
    def what_if(self, base: Mapping[str, Any], variable: str,
                values: Sequence[float]) -> pd.DataFrame:
        """Sweep one variable around a base design (the twin's what-if mode)."""
        designs = []
        for v in values:
            d = dict(base)
            d[variable] = v
            designs.append(d)
        frame = self.predict(designs).frame.copy()
        # `predict` already carries the prepared features through, so the swept
        # variable is usually present; overwrite it and move it to the front
        # rather than inserting a duplicate column.
        frame[variable] = list(values)
        cols = [variable] + [c for c in frame.columns if c != variable]
        return frame[cols]

    def benchmark(self, n: int = 10000, seed: int = 0,
                  batch_sizes: Sequence[int] = (1, 10, 100, 1000, 10000)
                  ) -> pd.DataFrame:
        """Measure prediction latency and the speed-up over SPICE.

        The SPICE reference is the *measured* mean simulation time recorded in
        the training dataset, not an assumed figure.
        """
        rng = np.random.default_rng(seed)
        rows: list[dict[str, Any]] = []
        spice_ms = float(self.metadata.get("mean_spice_runtime_s", float("nan"))) * 1e3

        for bs in batch_sizes:
            if bs > n:
                continue
            u = rng.random((bs, self._ds.n_dim))
            designs = [self._ds.decode_row(row) for row in u]
            frame = self.prepare(designs)
            t0 = time.perf_counter()
            self.predict(designs, prepared=frame)
            dt = time.perf_counter() - t0
            per = 1e3 * dt / bs
            rows.append({
                "batch_size": bs,
                "total_ms": round(dt * 1e3, 4),
                "ms_per_design": round(per, 6),
                "designs_per_second": round(bs / max(dt, 1e-12), 1),
                "spice_ms_per_design": round(spice_ms, 3) if np.isfinite(spice_ms) else "",
                "speedup_vs_spice": round(spice_ms / per, 1) if np.isfinite(spice_ms) and per > 0 else "",
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        import joblib
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"config": dict(self.cfg), "models": self.models,
                     "metadata": self.metadata}, p, compress=3)
        log.info("Digital twin saved -> %s (%.1f MB)", p.name,
                 p.stat().st_size / 1024 / 1024)
        return p

    @classmethod
    def load(cls, path: str | Path, expect_config_hash: str | None = None,
             strict: bool = True) -> "DigitalTwin":
        import joblib
        blob = joblib.load(Path(path))
        cfg = Config(blob["config"])
        twin = cls(cfg, blob["models"], blob.get("metadata", {}))
        got = twin.metadata.get("config_hash")
        if expect_config_hash and got != expect_config_hash:
            msg = (f"Digital twin was built from configuration {got!r} but "
                   f"{expect_config_hash!r} was requested")
            if strict:
                raise ValueError(msg)
            log.warning(msg)
        return twin

    def describe(self) -> dict[str, Any]:
        return {
            "targets": self.targets,
            "physical_targets": self.physical_targets,
            "models": {t: getattr(s, "model_name", "?") for t, s in self.models.items()},
            **self.metadata,
        }


def load_twin(path: str | Path, **kw: Any) -> DigitalTwin:
    return DigitalTwin.load(path, **kw)
