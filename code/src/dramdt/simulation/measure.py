"""Raw NGSpice output -> physical DRAM metrics.

:class:`MetricExtractor` is the single place where measured quantities become
the response variables used by every downstream stage (surrogates, optimisation,
XAI).  It never invents a value: when a measurement is missing the corresponding
metric is ``NaN`` and a reason is recorded in ``DramMetrics.notes``.

Power decomposition
-------------------
The simulated slice is *one cell together with its column periphery* (sense
amplifier, precharge/equalise devices and write driver).  All power figures are
reported at that granularity and are therefore directly comparable across
design points:

``standby_power_w``   quiescent draw from both supplies at the idle operating
                      point (precharge active, wordline low, SA off)
``active_power_w``    ``energy_per_access_j * access_rate_hz``
``refresh_power_w``   ``read_energy_j / refresh_interval_s`` -- a refresh is a
                      read followed by a full restore
``total_power_w``     the sum, and the quantity minimised by the optimiser

``access_rate_hz`` is a configuration value (``metrics.access_rate_hz``), not a
hidden constant: all three components are also reported separately so results
can be reweighted for a different access profile without re-simulating.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Mapping

import numpy as np

from ..models.cell import CellParameters
from .retention import (RetentionResult, quasistatic_retention,
                        refresh_interval_from_retention, retention_fail_level)
from .runner import SimulationResult

__all__ = ["DramMetrics", "MetricExtractor"]

NAN = float("nan")


@dataclass
class DramMetrics:
    """The response vector of one simulated design point."""

    # --- write ---------------------------------------------------------
    v_sn_written: float = NAN          # V, level actually stored after a '1' write
    write_delay_s: float = NAN
    write_efficiency: float = NAN      # v_sn_written / vdd
    write_success: bool = False

    # --- read ----------------------------------------------------------
    read_margin_v: float = NAN         # bitline differential at SA fire time
    read_delay_s: float = NAN
    read_success: bool = False
    v_bl_final: float = NAN
    v_blb_final: float = NAN
    v_sn_restored: float = NAN

    # --- retention -----------------------------------------------------
    retention_s: float = NAN
    retention_censored: bool = False
    v_fail: float = NAN
    refresh_interval_s: float = NAN
    leakage_current_a: float = NAN     # storage-node leakage at V_SN = VDD

    # --- energy / power ------------------------------------------------
    read_energy_j: float = NAN
    write_energy_j: float = NAN
    energy_per_access_j: float = NAN
    standby_power_w: float = NAN
    active_power_w: float = NAN
    refresh_power_w: float = NAN
    total_power_w: float = NAN

    # --- bookkeeping ---------------------------------------------------
    converged: bool = True
    simulation_ok: bool = False
    runtime_s: float = NAN
    notes: list[str] = field(default_factory=list)

    # -- convenience units used in tables/figures ------------------------
    def to_record(self) -> dict[str, Any]:
        """Flat record in publication units."""
        return {
            "v_sn_written_v": self.v_sn_written,
            "write_delay_ns": self.write_delay_s * 1e9,
            "write_efficiency": self.write_efficiency,
            "write_success": bool(self.write_success),
            "read_margin_mv": self.read_margin_v * 1e3,
            "read_delay_ns": self.read_delay_s * 1e9,
            "read_success": bool(self.read_success),
            "v_bl_final_v": self.v_bl_final,
            "v_blb_final_v": self.v_blb_final,
            "v_sn_restored_v": self.v_sn_restored,
            "retention_time_s": self.retention_s,
            "retention_time_ms": self.retention_s * 1e3,
            "retention_censored": bool(self.retention_censored),
            "v_fail_v": self.v_fail,
            "refresh_interval_ms": self.refresh_interval_s * 1e3,
            "leakage_current_fa": self.leakage_current_a * 1e15,
            "read_energy_fj": self.read_energy_j * 1e15,
            "write_energy_fj": self.write_energy_j * 1e15,
            "energy_per_access_fj": self.energy_per_access_j * 1e15,
            "standby_power_nw": self.standby_power_w * 1e9,
            "active_power_nw": self.active_power_w * 1e9,
            "refresh_power_nw": self.refresh_power_w * 1e9,
            "total_power_nw": self.total_power_w * 1e9,
            "converged": bool(self.converged),
            "simulation_ok": bool(self.simulation_ok),
            "sim_runtime_s": self.runtime_s,
            "notes": "; ".join(self.notes),
        }


class MetricExtractor:
    """Turns a :class:`SimulationResult` into :class:`DramMetrics`."""

    def __init__(self, config: Mapping[str, Any]):
        m = dict(config.get("metrics", {}))
        self.delta_v_min = float(m.get("sense_margin_min_v", 0.025))
        self.write_threshold = float(m.get("write_success_fraction", 0.85))
        self.read_resolve_fraction = float(m.get("read_resolve_fraction", 0.9))
        self.access_rate_hz = float(m.get("access_rate_hz", 1e5))
        self.refresh_guard_band = float(m.get("refresh_guard_band", 2.0))
        self.node_cap_overhead = float(m.get("node_cap_overhead_f", 0.0))
        self.retention_grid = int(m.get("retention_grid_points", 2001))
        self.max_retention_s = float(m.get("retention_censor_cap_s", 1e3))

    # ------------------------------------------------------------------
    def extract(self, result: SimulationResult, p: CellParameters) -> DramMetrics:
        met = DramMetrics(converged=result.converged,
                          simulation_ok=result.ok,
                          runtime_s=result.runtime_s)
        if not result.ok:
            met.notes.append(f"simulation failed: {result.error or 'no measurements parsed'}")
            return met

        g = result.get

        # ---------------- write ----------------------------------------
        met.v_sn_written = g("v_sn_hold")
        met.write_delay_s = g("t_write")
        if math.isfinite(met.v_sn_written) and p.vdd > 0:
            met.write_efficiency = met.v_sn_written / p.vdd
            met.write_success = met.write_efficiency >= self.write_threshold
        if not math.isfinite(met.write_delay_s):
            met.notes.append("write-delay measurement did not trigger")
        elif met.write_delay_s <= 0:
            met.notes.append("non-positive write delay (target reached before WL edge)")
            met.write_delay_s = NAN

        # ---------------- read -----------------------------------------
        v_bl, v_blb = g("v_bl_cs"), g("v_blb_cs")
        if math.isfinite(v_bl) and math.isfinite(v_blb):
            met.read_margin_v = v_bl - v_blb
        met.read_delay_s = g("t_read")
        met.v_bl_final, met.v_blb_final = g("v_bl_final"), g("v_blb_final")
        met.v_sn_restored = g("v_sn_restore")
        if math.isfinite(met.v_bl_final) and math.isfinite(met.v_blb_final):
            differential = met.v_bl_final - met.v_blb_final
            met.read_success = bool(
                differential >= self.read_resolve_fraction * p.vdd)
            if not met.read_success:
                met.notes.append("sense amplifier did not resolve a correct '1'")
        if not math.isfinite(met.read_delay_s) or met.read_delay_s <= 0:
            if math.isfinite(met.read_delay_s):
                met.notes.append("non-positive read delay")
            else:
                met.notes.append("read-delay measurement did not trigger")
            met.read_delay_s = NAN

        # ---------------- retention ------------------------------------
        met.v_fail = retention_fail_level(p.vblpre, p.cs, p.cbl, self.delta_v_min)
        v_init = met.v_sn_written if math.isfinite(met.v_sn_written) else p.vdd
        if result.has_leakage_curve:
            c_node = p.cs + self.node_cap_overhead
            rr: RetentionResult = quasistatic_retention(
                result.leakage_v, result.leakage_i, c_node,
                v_init=v_init, v_fail=met.v_fail, n_grid=self.retention_grid)
            met.retention_censored = rr.censored
            if rr.censored:
                met.retention_s = self.max_retention_s
                met.notes.append(rr.note)
            else:
                met.retention_s = rr.retention_s
                if rr.note:
                    met.notes.append(rr.note)
            # storage-node leakage at the written level
            met.leakage_current_a = float(
                np.interp(v_init, result.leakage_v, result.leakage_i))
        else:
            met.notes.append("leakage characteristic missing: retention not computed")

        met.refresh_interval_s = refresh_interval_from_retention(
            met.retention_s, self.refresh_guard_band)

        # ---------------- energy / power --------------------------------
        q_read, q_write = g("q_read"), g("q_write")
        qp_read = g("qp_read")
        if math.isfinite(q_read):
            e = abs(q_read) * p.vdd
            if math.isfinite(qp_read):
                e += abs(qp_read) * p.vblpre
            met.read_energy_j = e
        if math.isfinite(q_write):
            met.write_energy_j = abs(q_write) * p.vdd
        if math.isfinite(met.read_energy_j) and math.isfinite(met.write_energy_j):
            met.energy_per_access_j = 0.5 * (met.read_energy_j + met.write_energy_j)
        elif math.isfinite(met.read_energy_j):
            met.energy_per_access_j = met.read_energy_j

        i_vdd, i_pre = g("i_vvdd"), g("i_vpre_s")
        standby = 0.0
        have_standby = False
        if math.isfinite(i_vdd):
            standby += abs(i_vdd) * p.vdd
            have_standby = True
        if math.isfinite(i_pre):
            standby += abs(i_pre) * p.vblpre
            have_standby = True
        met.standby_power_w = standby if have_standby else NAN

        if math.isfinite(met.energy_per_access_j):
            met.active_power_w = met.energy_per_access_j * self.access_rate_hz
        if math.isfinite(met.read_energy_j) and math.isfinite(met.refresh_interval_s) \
                and met.refresh_interval_s > 0:
            met.refresh_power_w = met.read_energy_j / met.refresh_interval_s

        parts = [met.standby_power_w, met.active_power_w, met.refresh_power_w]
        if all(math.isfinite(x) for x in parts):
            met.total_power_w = float(sum(parts))
        else:
            finite = [x for x in parts if math.isfinite(x)]
            if finite:
                met.total_power_w = float(sum(finite))
                met.notes.append("total power omits a non-finite component")

        return met
