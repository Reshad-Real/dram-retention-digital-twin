"""DRAM retention time from SPICE-measured storage-node leakage.

Why a quasi-static model
------------------------
Retention times of a DRAM cell span milliseconds to seconds, while the
electrical events that define read/write margin happen in nanoseconds.  Running
a direct transient over a full retention interval at the time resolution needed
for the rest of the characterisation is computationally prohibitive for a
50,000-point campaign.

The storage node is, however, an *isolated capacitor*: once the wordline is
low, the only elements attached to ``SN`` are the storage capacitance and the
leakage paths of the access device.  Charge conservation on that node is exact,

.. math::

    C_\\mathrm{node}\\,\\frac{\\mathrm{d}V_{SN}}{\\mathrm{d}t} = -I_\\mathrm{leak}(V_{SN})

so the retention time follows by separation of variables:

.. math::

    t_\\mathrm{ret} = \\int_{V_\\mathrm{fail}}^{V_\\mathrm{init}}
                      \\frac{C_\\mathrm{node}}{I_\\mathrm{leak}(V)}\\,\\mathrm{d}V

The one modelling assumption is that the leakage is quasi-static, i.e. that
:math:`I_\\mathrm{leak}` depends on the instantaneous node voltage only.  That
holds here because the discharge is many orders of magnitude slower than every
device time constant.

Crucially, :math:`I_\\mathrm{leak}(V)` is **not** assumed: it is measured by a
DC sweep of the leakage-replica probe source in the very same NGSpice run, at
the sample's own temperature, corner and bias, and therefore includes
subthreshold conduction, GIDL, junction and gate-tunnelling leakage exactly as
the BSIM4 model card describes them.

:func:`dramdt.simulation.retention.validate_against_transient` cross-checks the
model against direct long-window transients; the agreement is reported in the
results rather than asserted here.

Failure criterion
-----------------
A cell has *failed* when charge sharing can no longer produce a bitline
differential that the sense amplifier can resolve:

.. math::

    \\Delta V_{BL} = (V_{SN} - V_\\mathrm{BLpre})\\,\\frac{C_s}{C_s + C_{BL}}
                    \\;\\ge\\; \\Delta V_\\mathrm{min}

which inverts to the failure level implemented in :func:`retention_fail_level`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "RetentionResult", "retention_fail_level", "quasistatic_retention",
    "refresh_interval_from_retention", "validate_against_transient",
]


@dataclass
class RetentionResult:
    """Outcome of the quasi-static retention integration."""

    retention_s: float
    censored: bool                  # True == node equilibrates above V_fail
    v_init: float
    v_fail: float
    v_equilibrium: float | None
    i_at_init: float
    i_min_in_range: float
    method: str = "quasistatic-integration"
    note: str = ""

    @property
    def valid(self) -> bool:
        return math.isfinite(self.retention_s) and self.retention_s > 0.0


# --------------------------------------------------------------------------
def retention_fail_level(v_blpre: float, cs: float, cbl: float,
                         delta_v_min: float) -> float:
    """Storage-node voltage at which the read margin drops to ``delta_v_min``.

    ``delta_v_min`` is the minimum bitline differential the sense amplifier can
    resolve (its input-referred offset plus a design guard band), in volts.
    """
    if cs <= 0:
        raise ValueError("cs must be positive")
    return v_blpre + delta_v_min * (cs + cbl) / cs


def refresh_interval_from_retention(retention_s: float, guard_band: float = 2.0) -> float:
    """Refresh interval implied by a retention time and a policy guard band."""
    if not math.isfinite(retention_s) or retention_s <= 0:
        return float("nan")
    return retention_s / max(guard_band, 1e-9)


# --------------------------------------------------------------------------
def quasistatic_retention(v_sn: Sequence[float] | np.ndarray,
                          i_leak: Sequence[float] | np.ndarray,
                          c_node: float,
                          v_init: float,
                          v_fail: float,
                          n_grid: int = 2001,
                          i_floor: float = 1e-21) -> RetentionResult:
    """Integrate the charge-loss ODE over a SPICE-measured leakage curve.

    Parameters
    ----------
    v_sn, i_leak
        The measured characteristic.  ``i_leak`` must be the current *leaving*
        the storage node (positive == discharging); this is what
        :func:`dramdt.simulation.runner.parse_leakage_file` returns.
    c_node
        Total capacitance on the storage node (F).
    v_init, v_fail
        Integration limits: the written '1' level and the failure level.
    """
    v = np.asarray(v_sn, dtype=float)
    i = np.asarray(i_leak, dtype=float)

    if v.size < 2 or i.size != v.size:
        return RetentionResult(float("nan"), False, v_init, v_fail, None,
                               float("nan"), float("nan"),
                               note="leakage characteristic unavailable")
    order = np.argsort(v)
    v, i = v[order], i[order]

    if not (v_init > v_fail):
        # The level actually written is already below the level at which the
        # read fails, so the cell has no retention at all.  Zero is the correct
        # physical answer here, not a missing value: reporting NaN would delete
        # the entire infeasible region from the dataset.
        return RetentionResult(0.0, False, v_init, v_fail, None,
                               float(np.interp(v_init, v, i)), float(np.min(i)),
                               note="written level is at or below the failure level: "
                                    "zero retention")

    lo = max(v_fail, float(v[0]))
    hi = min(v_init, float(v[-1]))
    if not (hi > lo):
        return RetentionResult(float("nan"), False, v_init, v_fail, None,
                               float("nan"), float("nan"),
                               note="failure level outside the swept range")

    grid = np.linspace(lo, hi, int(n_grid))
    i_grid = np.interp(grid, v, i)
    i_at_init = float(np.interp(min(v_init, v[-1]), v, i))
    i_min = float(np.min(i_grid))

    # The node discharges only while I_leak > 0.  If the current changes sign
    # inside the interval the node equilibrates there and never reaches V_fail.
    if i_min <= i_floor:
        bad = np.nonzero(i_grid <= i_floor)[0]
        v_eq = float(grid[bad[-1]])
        return RetentionResult(
            float("inf"), True, v_init, v_fail, v_eq, i_at_init, i_min,
            note=("storage node equilibrates at "
                  f"{v_eq:.4f} V, above the failure level: retention is "
                  "right-censored (no failure within this bias condition)"))

    integrand = c_node / i_grid
    t_ret = float(np.trapezoid(integrand, grid))
    return RetentionResult(t_ret, False, v_init, v_fail, None, i_at_init, i_min)


# --------------------------------------------------------------------------
def validate_against_transient(quasistatic_s: Sequence[float],
                               transient_s: Sequence[float]) -> dict[str, float]:
    """Agreement statistics between the two retention models.

    Both are compared in log10 space because retention spans several decades.
    Returns Pearson/Spearman correlation, the median absolute log10 error and
    the geometric-mean ratio, over the pairs where both models produced a
    finite positive value.
    """
    from scipy import stats

    q = np.asarray(quasistatic_s, dtype=float)
    t = np.asarray(transient_s, dtype=float)
    m = np.isfinite(q) & np.isfinite(t) & (q > 0) & (t > 0)
    n = int(m.sum())
    if n < 3:
        return {"n_pairs": float(n)}

    lq, lt = np.log10(q[m]), np.log10(t[m])
    err = lq - lt
    out = {
        "n_pairs": float(n),
        "pearson_r_log10": float(stats.pearsonr(lq, lt)[0]),
        "spearman_rho": float(stats.spearmanr(lq, lt)[0]),
        "median_abs_log10_error": float(np.median(np.abs(err))),
        "mean_log10_error": float(np.mean(err)),
        "geometric_mean_ratio": float(10.0 ** np.mean(err)),
        "p90_abs_log10_error": float(np.percentile(np.abs(err), 90)),
        "within_factor_2_pct": float(100.0 * np.mean(np.abs(err) <= math.log10(2.0))),
        "within_factor_10_pct": float(100.0 * np.mean(np.abs(err) <= 1.0)),
    }
    return out
