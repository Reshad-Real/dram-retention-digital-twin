"""Deterministic seeding.

Reproducibility rule used throughout the framework: *every* stochastic
component derives its seed from the single master seed in the experiment
configuration via :func:`derive_seed`, which is a pure function of
``(master_seed, label, index)``.  Nothing consumes global RNG state, so
results are independent of execution order and of the number of worker
processes.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any

import numpy as np

__all__ = ["derive_seed", "seed_everything", "rng_for", "SEED_MAX"]

SEED_MAX = 2 ** 31 - 1


def derive_seed(master_seed: int, label: str = "", index: int | None = None) -> int:
    """Derive a stable child seed from ``(master_seed, label, index)``.

    Uses BLAKE2b so the mapping is stable across Python versions and platforms
    (unlike :func:`hash`, which is randomised per interpreter run).
    """
    key = f"{int(master_seed)}|{label}|{'' if index is None else int(index)}"
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % SEED_MAX


def rng_for(master_seed: int, label: str = "", index: int | None = None) -> np.random.Generator:
    """Return an independent NumPy :class:`Generator` for a named stream."""
    return np.random.default_rng(derive_seed(master_seed, label, index))


def seed_everything(master_seed: int, deterministic_hash: bool = True) -> dict[str, Any]:
    """Seed Python, NumPy and (if present) the ML libraries.

    Returns a dict describing what was seeded, for logging into the run manifest.
    """
    seeded: dict[str, Any] = {"master_seed": int(master_seed)}

    random.seed(master_seed)
    seeded["python_random"] = True

    np.random.seed(master_seed % SEED_MAX)
    seeded["numpy_legacy"] = True

    if deterministic_hash:
        os.environ.setdefault("PYTHONHASHSEED", str(master_seed))
        seeded["PYTHONHASHSEED"] = os.environ.get("PYTHONHASHSEED")

    # Keep BLAS deterministic-ish and avoid oversubscription when we fan out
    # across worker processes.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    try:                                            # pragma: no cover - optional
        import torch                                # type: ignore
        torch.manual_seed(master_seed)
        seeded["torch"] = True
    except Exception:
        pass

    return seeded
