"""Uniform logging + run-manifest helpers.

Every pipeline stage writes a timestamped log to ``logs/`` and a JSON manifest
recording the configuration hash, package versions, seeds and wall-clock time.
The manifests are what make a completed run auditable after the fact.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .paths import LOGS_DIR, PROJECT_ROOT

__all__ = ["setup_logging", "get_logger", "RunManifest", "stage", "package_versions"]

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_TRACKED_PACKAGES = (
    "numpy", "scipy", "pandas", "matplotlib", "sklearn", "xgboost", "lightgbm",
    "catboost", "optuna", "shap", "pymoo", "PySpice", "openpyxl", "joblib",
)


def setup_logging(name: str,
                  level: int = logging.INFO,
                  to_file: bool = True,
                  filename: str | None = None) -> logging.Logger:
    """Configure the root logger once and return a named child logger."""
    root = logging.getLogger()
    root.setLevel(level)

    if not any(isinstance(h, logging.StreamHandler) and getattr(h, "_dramdt", False)
               for h in root.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        sh.setLevel(level)
        sh._dramdt = True                                   # type: ignore[attr-defined]
        root.addHandler(sh)

    if to_file:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        fn = filename or f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        already = any(getattr(h, "_dramdt_file", None) == fn for h in root.handlers)
        if not already:
            fh = logging.FileHandler(LOGS_DIR / fn, encoding="utf-8")
            fh.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
            fh.setLevel(logging.DEBUG)
            fh._dramdt_file = fn                            # type: ignore[attr-defined]
            root.addHandler(fh)

    # Third-party chatter we never want in our logs.
    for noisy in ("matplotlib", "PIL", "numexpr", "optuna", "shap", "PySpice"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger(name)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def package_versions() -> dict[str, str]:
    """Version string for each tracked third-party package (best effort)."""
    from importlib.metadata import version, PackageNotFoundError
    out: dict[str, str] = {"python": sys.version.split()[0]}
    for pkg in _TRACKED_PACKAGES:
        dist = {"sklearn": "scikit-learn"}.get(pkg, pkg)
        try:
            out[pkg] = version(dist)
        except PackageNotFoundError:
            out[pkg] = "<not installed>"
        except Exception:                                   # pragma: no cover
            out[pkg] = "<unknown>"
    return out


def _git_revision() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "<not a git repository>"
    except Exception:
        return "<git unavailable>"


@dataclass
class RunManifest:
    """Machine-readable record of one pipeline-stage execution."""

    stage: str
    started_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_utc: str | None = None
    duration_s: float | None = None
    status: str = "running"
    config_hash: str | None = None
    seeds: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.environment:
            self.environment = {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python_implementation": platform.python_implementation(),
                "git_revision": _git_revision(),
                "packages": package_versions(),
            }

    def save(self, path: str | Path | None = None) -> Path:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        p = Path(path) if path else LOGS_DIR / f"manifest_{self.stage}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, default=str), encoding="utf-8")
        return p


@contextmanager
def stage(name: str,
          logger: logging.Logger | None = None,
          manifest: RunManifest | None = None,
          manifest_path: str | Path | None = None) -> Iterator[RunManifest]:
    """Context manager wrapping a pipeline stage with timing + manifest output."""
    log = logger or setup_logging(name)
    man = manifest or RunManifest(stage=name)
    t0 = time.perf_counter()
    log.info("=" * 78)
    log.info("STAGE START : %s", name)
    log.info("=" * 78)
    try:
        yield man
        man.status = "success"
    except BaseException as exc:
        man.status = "failed"
        man.error = f"{type(exc).__name__}: {exc}"
        log.exception("STAGE FAILED: %s", name)
        raise
    finally:
        man.duration_s = time.perf_counter() - t0
        man.finished_utc = datetime.now(timezone.utc).isoformat()
        p = man.save(manifest_path)
        log.info("STAGE END   : %s | status=%s | %.2f s | manifest=%s",
                 name, man.status, man.duration_s, p.name)
