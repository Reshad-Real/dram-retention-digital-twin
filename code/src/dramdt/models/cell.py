"""Stage 1 -- the parameterised 1T1C DRAM cell.

Topology built by :class:`DramCellBuilder` (see ``Proposed Method``, block 1):

* **Cell** -- one NMOS access transistor ``Maccess`` (BL / WL / SN) and the
  storage capacitor ``Cstore`` between the storage node ``SN`` and the cell
  plate.  ``Rdiel`` models capacitor dielectric leakage (used by the high-k /
  high-C assist study).
* **Bitline pair** -- ``Cbl`` / ``Cblb`` lumped bitline capacitances, sized
  from the number of cells per bitline.
* **Precharge / equalise** -- three NMOS pass devices driven by a boosted gate
  (VPP).  A VDD-level gate cannot pull BL to VDD/2 because ``Vgs = VDD/2 < Vth``;
  boosted precharge is what real parts use and what the netlist models.
* **Sense amplifier** -- a cross-coupled NMOS/PMOS latch with separate N and P
  enable devices, so SA sizing is a genuine design variable.
* **Write driver** -- a full CMOS transmission gate from the data node to BL.
* **Leakage replica** -- an *identical* access device whose storage node is
  driven by a probe source, so the storage-node leakage characteristic
  ``I_leak(V_SN)`` can be measured by a DC sweep in the same run.  This is what
  makes the quasi-static retention model in :mod:`dramdt.simulation.retention`
  a measurement rather than an assumption.

PySpice is used for the circuit container, unit handling and netlist emission;
the BSIM4 instance lines are emitted through :attr:`Circuit.raw_spice` because
they carry instance parameters (``AS``/``AD``/``PS``/``PD``/``delvto``) that the
generic PySpice ``Mosfet`` element does not expose.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import Config, ConfigError
from .technology import Technology

__all__ = ["CellParameters", "TimingSchedule", "DramCellBuilder", "MEAS_BEGIN", "MEAS_END"]

MEAS_BEGIN = "---DRAMDT-MEAS-BEGIN---"
MEAS_END = "---DRAMDT-MEAS-END---"


# --------------------------------------------------------------------------
@dataclass
class TimingSchedule:
    """Absolute waveform times (seconds) for one characterisation run."""

    edge: float
    pre1_end: float
    wr_on: float
    wr_off: float
    pre2_on: float
    pre2_end: float
    rd_wl: float
    sa: float
    stop: float

    @classmethod
    def build(cls, p: "CellParameters", timing: Mapping[str, Any]) -> "TimingSchedule":
        edge = float(timing.get("edge_time_s", 100e-12))
        gap = float(timing.get("gap_s", 1e-9))
        settle = float(timing.get("initial_precharge_s", 5e-9))
        prech = float(timing.get("precharge_s", 10e-9))
        window = float(timing.get("sense_window_s", 13e-9))

        pre1_end = settle
        wr_on = pre1_end + gap
        wr_off = wr_on + p.t_write_pulse
        pre2_on = wr_off + gap
        pre2_end = pre2_on + prech
        rd_wl = pre2_end + gap
        sa = rd_wl + p.t_sense_delay
        stop = sa + window
        return cls(edge=edge, pre1_end=pre1_end, wr_on=wr_on, wr_off=wr_off,
                   pre2_on=pre2_on, pre2_end=pre2_end, rd_wl=rd_wl, sa=sa, stop=stop)


# --------------------------------------------------------------------------
@dataclass
class CellParameters:
    """Fully-resolved physical parameters of one DRAM cell instance.

    Instances are produced by :meth:`DramCellBuilder.parameters_from_sample`
    from a decoded design-space row, so nothing here is ever hard-coded.
    """

    # supplies / biases (V)
    vdd: float = 1.1
    vwl_high: float = 1.9
    vwl_low: float = 0.0
    vblpre: float = 0.55
    vplate: float = 0.55

    # capacitances (F)
    cs: float = 20e-15
    cbl: float = 100e-15

    # access device (m)
    wacc: float = 90e-9
    lacc: float = 45e-9

    # periphery devices (m)
    wsan: float = 180e-9
    wsap: float = 270e-9
    wen_n: float = 360e-9
    wen_p: float = 720e-9
    wpc: float = 180e-9
    wwr: float = 360e-9
    lsa: float = 45e-9

    # storage-capacitor dielectric leakage (ohm); huge == ideal dielectric
    rdiel: float = 1e15

    # environment / process
    temperature_c: float = 27.0
    corner: str = "TT"

    # timing knobs that are design variables (s)
    t_write_pulse: float = 6e-9
    t_sense_delay: float = 6e-9

    # geometry helper: diffusion extension used for AS/AD/PS/PD (m)
    ldiff: float = 112.5e-9

    # per-instance threshold offsets for Monte-Carlo mismatch (V)
    delvto: dict[str, float] = field(default_factory=dict)

    # bookkeeping
    sample_id: int = -1
    assist_config: str = "baseline"

    # -- derived ----------------------------------------------------------
    @property
    def vwl_swing(self) -> float:
        return self.vwl_high - self.vwl_low

    @property
    def vwl_boost(self) -> float:
        return self.vwl_high - self.vdd

    @property
    def charge_share_ratio(self) -> float:
        """Cs / (Cs + Cbl): the ideal charge-sharing transfer ratio."""
        return self.cs / (self.cs + self.cbl)

    def diffusion_geometry(self, w: float) -> tuple[float, float]:
        """Return ``(area, perimeter)`` of a source/drain diffusion of width *w*."""
        return w * self.ldiff, 2.0 * (w + self.ldiff)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("delvto", None)
        d["vwl_boost"] = self.vwl_boost
        d["charge_share_ratio"] = self.charge_share_ratio
        return d


# --------------------------------------------------------------------------
def _pwl(pairs: Sequence[tuple[float, float]]) -> str:
    """Format a PWL source argument list."""
    return " ".join(f"{t:.9e} {v:.9e}" for t, v in pairs)


def _mos(name: str, d: str, g: str, s: str, b: str, model: str,
         w: float, l: float, *, area: tuple[float, float] | None = None,
         delvto: float = 0.0, m: int = 1) -> str:
    """Emit one BSIM4 instance line."""
    parts = [f"M{name} {d} {g} {s} {b} {model} W={w:.6e} L={l:.6e}"]
    if area is not None:
        a, p = area
        parts.append(f"AS={a:.6e} AD={a:.6e} PS={p:.6e} PD={p:.6e}")
    if m != 1:
        parts.append(f"M={m}")
    if delvto:
        parts.append(f"delvto={delvto:.6e}")
    return " ".join(parts)


class DramCellBuilder:
    """Turns a :class:`CellParameters` into a runnable NGSpice deck."""

    def __init__(self, config: Config, technology: Technology | None = None):
        self.cfg = config
        self.tech = technology or Technology.from_config(config)
        self.arch = dict(config.get("architecture", {}))
        self.timing_cfg = dict(config.get("timing", {}))
        self.sim_cfg = dict(config.get("simulation", {}))
        self.metrics_cfg = dict(config.get("metrics", {}))

    # ------------------------------------------------------------------
    # design-space row -> physical parameters
    # ------------------------------------------------------------------
    def parameters_from_sample(self, sample: Mapping[str, Any],
                               sample_id: int = -1) -> CellParameters:
        """Build :class:`CellParameters` from a decoded design-space row.

        Every field falls back to the ``architecture`` section of the config, so
        a configuration may sample a variable or pin it -- without code changes.
        """
        g = lambda key, default: sample.get(key, self.arch.get(key, default))  # noqa: E731

        vdd = float(g("vdd", 1.1))
        vwl_high = float(sample["vwl_high"]) if "vwl_high" in sample else \
            vdd + float(g("vwl_boost", 0.8))
        vblpre = float(sample["vblpre"]) if "vblpre" in sample else \
            vdd * float(g("vblpre_ratio", 0.5))
        vplate = float(sample["vplate"]) if "vplate" in sample else \
            vdd * float(g("vplate_ratio", 0.5))

        cs = float(g("cs", 20e-15))
        cbl = float(sample["cbl"]) if "cbl" in sample else \
            cs * float(g("cbl_ratio", 5.0))

        wsan = float(g("wsan", 180e-9))

        return CellParameters(
            vdd=vdd,
            vwl_high=vwl_high,
            vwl_low=float(g("vwl_low", 0.0)),
            vblpre=vblpre,
            vplate=vplate,
            cs=cs,
            cbl=cbl,
            wacc=float(g("wacc", 90e-9)),
            lacc=float(g("lacc", 45e-9)),
            wsan=wsan,
            wsap=float(sample["wsap"]) if "wsap" in sample else
                wsan * float(g("wsap_ratio", 1.5)),
            wen_n=float(sample["wen_n"]) if "wen_n" in sample else
                wsan * float(g("wen_ratio", 2.0)),
            wen_p=float(sample["wen_p"]) if "wen_p" in sample else
                wsan * float(g("wen_ratio", 2.0)) * 2.0,
            wpc=float(g("wpc", 180e-9)),
            wwr=float(g("wwr", 360e-9)),
            lsa=float(g("lsa", 45e-9)),
            rdiel=float(g("rdiel", 1e15)),
            temperature_c=float(g("temperature_c", 27.0)),
            corner=str(g("corner", "TT")),
            t_write_pulse=float(g("t_write_pulse", 6e-9)),
            t_sense_delay=float(g("t_sense_delay", 6e-9)),
            ldiff=float(self.arch.get("ldiff", 112.5e-9)),
            sample_id=sample_id,
            assist_config=str(sample.get("assist_config", "baseline")),
        )

    # ------------------------------------------------------------------
    # netlist construction
    # ------------------------------------------------------------------
    def _header(self, p: CellParameters, title: str) -> str:
        card = self.tech.card_for(p.corner)
        opts = self.sim_cfg.get("options", {})
        opt_line = " ".join(f"{k.upper()}={v}" for k, v in opts.items())
        return (
            f"* {title}\n"
            f"* dramdt generated deck | corner={p.corner} | T={p.temperature_c:.2f} C\n"
            f".include \"{Path(card).as_posix()}\"\n"
            f".options TEMP={p.temperature_c:.6g} TNOM={self.sim_cfg.get('tnom', 27)}"
            f"{(' ' + opt_line) if opt_line else ''}\n"
        )

    def _core_devices(self, p: CellParameters, *, include_replica: bool = True) -> str:
        nm, pm = self.tech.nmos_model, self.tech.pmos_model
        acc_area = p.diffusion_geometry(p.wacc)
        pc_area = p.diffusion_geometry(p.wpc)
        dv = p.delvto

        lines = [
            "",
            "* ---------------- supplies ----------------",
            f"Vvdd   vddn  0 DC {p.vdd:.9g}",
            f"Vpre_s vpre  0 DC {p.vblpre:.9g}",
            f"Vplate plate 0 DC {p.vplate:.9g}",
            f"Vwdata wdata_s 0 DC {p.vdd:.9g}",
            "* 1 ohm of series damping: an ideal source feeding the write-driver",
            "* transmission gate makes its branch current stiff enough to collapse",
            "* the transient timestep once the sense amplifier slams the bitline.",
            "Rwdata wdata_s wdata 1.0",
            "",
            "* ---------------- 1T1C cell under test ----------------",
            _mos("access", "bl", "wl", "sn", "0", nm, p.wacc, p.lacc,
                 area=acc_area, delvto=dv.get("access", 0.0)),
            f"Cstore sn plate {p.cs:.9e}",
            f"Rdiel  sn plate {p.rdiel:.9e}",
            "",
            "* ---------------- bitline pair ----------------",
            f"Cbl  bl  0 {p.cbl:.9e}",
            f"Cblb blb 0 {p.cbl:.9e}",
            "",
            "* ---------------- precharge / equalise (boosted gate) ----------------",
            _mos("pc1", "bl", "preg", "vpre", "0", nm, p.wpc, p.lsa, area=pc_area),
            _mos("pc2", "blb", "preg", "vpre", "0", nm, p.wpc, p.lsa, area=pc_area),
            _mos("eq", "bl", "preg", "blb", "0", nm, p.wpc, p.lsa, area=pc_area),
            "",
            "* ---------------- write driver (CMOS transmission gate) ----------------",
            _mos("wn", "bl", "wrn", "wdata", "0", nm, p.wwr, p.lsa),
            _mos("wp", "bl", "wrp", "wdata", "vddn", pm, 2.0 * p.wwr, p.lsa),
            "",
            "* ---------------- sense amplifier (cross-coupled latch) ----------------",
            _mos("n1", "bl", "blb", "san", "0", nm, p.wsan, p.lsa, delvto=dv.get("sa_n1", 0.0)),
            _mos("n2", "blb", "bl", "san", "0", nm, p.wsan, p.lsa, delvto=dv.get("sa_n2", 0.0)),
            _mos("p1", "bl", "blb", "sap", "vddn", pm, p.wsap, p.lsa, delvto=dv.get("sa_p1", 0.0)),
            _mos("p2", "blb", "bl", "sap", "vddn", pm, p.wsap, p.lsa, delvto=dv.get("sa_p2", 0.0)),
            _mos("ne", "san", "saen", "0", "0", nm, p.wen_n, p.lsa),
            _mos("pe", "sap", "saep", "vddn", "vddn", pm, p.wen_p, p.lsa),
        ]

        if include_replica:
            lines += [
                "",
                "* ------- storage-node leakage replica (identical access device) -------",
                "* SN is driven by a probe source so I_leak(V_SN) can be swept in DC.",
                _mos("lk", "blref", "wloff", "snp", "0", nm, p.wacc, p.lacc,
                     area=acc_area, delvto=dv.get("access", 0.0)),
                f"Vblref blref 0 DC {p.vblpre:.9g}",
                f"Vwloff wloff 0 DC {p.vwl_low:.9g}",
                "Vsnp   snp   0 DC 0",
                f"Rdielp snp plate {p.rdiel:.9e}",
            ]
        return "\n".join(lines)

    def _stimuli(self, p: CellParameters, t: TimingSchedule) -> str:
        e = t.edge
        vpp = p.vwl_high                    # boosted rail also drives the precharge gates
        wl = [(0.0, p.vwl_low),
              (t.wr_on, p.vwl_low), (t.wr_on + e, p.vwl_high),
              (t.wr_off, p.vwl_high), (t.wr_off + e, p.vwl_low),
              (t.rd_wl, p.vwl_low), (t.rd_wl + e, p.vwl_high),
              (t.stop, p.vwl_high)]
        preg = [(0.0, vpp),
                (t.pre1_end, vpp), (t.pre1_end + e, 0.0),
                (t.pre2_on, 0.0), (t.pre2_on + e, vpp),
                (t.pre2_end, vpp), (t.pre2_end + e, 0.0),
                (t.stop, 0.0)]
        wrn = [(0.0, 0.0), (t.wr_on, 0.0), (t.wr_on + e, p.vdd),
               (t.wr_off, p.vdd), (t.wr_off + e, 0.0), (t.stop, 0.0)]
        wrp = [(0.0, p.vdd), (t.wr_on, p.vdd), (t.wr_on + e, 0.0),
               (t.wr_off, 0.0), (t.wr_off + e, p.vdd), (t.stop, p.vdd)]
        saen = [(0.0, 0.0), (t.sa, 0.0), (t.sa + e, p.vdd), (t.stop, p.vdd)]
        saep = [(0.0, p.vdd), (t.sa, p.vdd), (t.sa + e, 0.0), (t.stop, 0.0)]
        return "\n".join([
            "",
            "* ---------------- stimuli ----------------",
            f"Vwl   wl   0 PWL({_pwl(wl)})",
            f"Vpreg preg 0 PWL({_pwl(preg)})",
            f"Vwrn  wrn  0 PWL({_pwl(wrn)})",
            f"Vwrp  wrp  0 PWL({_pwl(wrp)})",
            f"Vsaen saen 0 PWL({_pwl(saen)})",
            f"Vsaep saep 0 PWL({_pwl(saep)})",
        ])

    def _control(self, p: CellParameters, t: TimingSchedule,
                 leak_file: str = "leak.csv") -> str:
        tstep = float(self.sim_cfg.get("tran_step_s", 20e-12))
        wl_mid = 0.5 * (p.vwl_high + p.vwl_low)
        n_leak = int(self.sim_cfg.get("leakage_sweep_points", 51))
        dv = p.vdd / max(n_leak - 1, 1)
        eps = 1e-13

        return "\n".join([
            "",
            ".control",
            "set noaskquit",
            "set numdgt=10",
            "",
            "* Tolerances are set per analysis.  The transient resolves micro-amp",
            "* switching currents, whereas the leakage sweep resolves femto-amps;",
            "* one global ABSTOL cannot serve both without collapsing the",
            "* transient timestep.",
            f"option abstol={self.sim_cfg.get('tran_abstol', 1e-13):g} "
            f"vntol={self.sim_cfg.get('tran_vntol', 1e-6):g} "
            f"gmin={self.sim_cfg.get('tran_gmin', 1e-13):g}",
            "",
            "* ---- (1) quiescent operating point: standby leakage ----",
            "op",
            f"echo {MEAS_BEGIN}",
            "print i(Vvdd)",
            "print i(Vpre_s)",
            "print v(sn)",
            "",
            "* ---- (2) write / read transient ----",
            f"tran {tstep:.6e} {t.stop:.6e}",
            f"meas tran v_sn_hold FIND v(sn) AT={t.rd_wl - eps:.9e}",
            "let vwtarg = 0.9 * v_sn_hold",
            f"meas tran t_write TRIG v(wl) VAL={wl_mid:.6g} RISE=1 "
            f"TARG v(sn) VAL=$&vwtarg RISE=1",
            f"meas tran v_bl_cs  FIND v(bl)  AT={t.sa - eps:.9e}",
            f"meas tran v_blb_cs FIND v(blb) AT={t.sa - eps:.9e}",
            "let vdiff = v(bl)-v(blb)",
            f"meas tran t_read TRIG v(saen) VAL={0.5 * p.vdd:.6g} RISE=1 "
            f"TARG vdiff VAL={0.9 * p.vdd:.6g} RISE=1",
            f"meas tran q_read  INTEG i(Vvdd) FROM={t.pre2_on:.9e} TO={t.stop:.9e}",
            f"meas tran q_write INTEG i(Vvdd) FROM=0 TO={t.pre2_on:.9e}",
            f"meas tran qp_read INTEG i(Vpre_s) FROM={t.pre2_on:.9e} TO={t.stop:.9e}",
            f"meas tran v_bl_final  FIND v(bl)  AT={t.stop - eps:.9e}",
            f"meas tran v_blb_final FIND v(blb) AT={t.stop - eps:.9e}",
            f"meas tran v_sn_restore FIND v(sn) AT={t.stop - eps:.9e}",
            f"meas tran v_bl_pre FIND v(bl) AT={t.pre2_end - eps:.9e}",
            "",
            "* ---- (3) storage-node leakage characteristic I_leak(V_SN) ----",
            f"option abstol={self.sim_cfg.get('dc_abstol', 1e-18):g} "
            f"vntol={self.sim_cfg.get('dc_vntol', 1e-9):g} "
            f"gmin={self.sim_cfg.get('dc_gmin', 1e-15):g} "
            f"reltol={self.sim_cfg.get('dc_reltol', 1e-5):g}",
            f"dc Vsnp 0 {p.vdd:.9g} {dv:.9e}",
            f"wrdata {leak_file} i(Vsnp)",
            f"echo {MEAS_END}",
            ".endc",
            ".end",
            "",
        ])

    # -- public -----------------------------------------------------------
    def schedule(self, p: CellParameters) -> TimingSchedule:
        return TimingSchedule.build(p, self.timing_cfg)

    def build_characterization_netlist(self, p: CellParameters,
                                       leak_file: str = "leak.csv") -> str:
        """The main deck: OP + write/read transient + leakage DC sweep."""
        t = self.schedule(p)
        return (self._header(p, "DRAM 1T1C characterisation")
                + self._core_devices(p)
                + self._stimuli(p, t)
                + self._control(p, t, leak_file))

    def build_retention_netlist(self, p: CellParameters,
                                t_stop: float,
                                v_fail: float,
                                n_points: int = 2000,
                                v_init: float | None = None) -> str:
        """A long direct transient watching the storage node discharge.

        Used to validate the quasi-static retention model on a subset of
        designs.  The cell is initialised to the level actually written
        (``v_init``, defaulting to VDD) and the wordline is held at its low
        level for the whole window, exactly as during data retention.
        """
        v_init = p.vdd if v_init is None else float(v_init)
        nm = self.tech.nmos_model
        acc_area = p.diffusion_geometry(p.wacc)
        step = max(t_stop / max(n_points, 10), 1e-12)
        return "\n".join([
            self._header(p, "DRAM storage-node retention transient"),
            "",
            f"Vplate plate 0 DC {p.vplate:.9g}",
            f"Vbl    bl    0 DC {p.vblpre:.9g}",
            f"Vwl    wl    0 DC {p.vwl_low:.9g}",
            _mos("access", "bl", "wl", "sn", "0", nm, p.wacc, p.lacc,
                 area=acc_area, delvto=p.delvto.get("access", 0.0)),
            f"Cstore sn plate {p.cs:.9e}",
            f"Rdiel  sn plate {p.rdiel:.9e}",
            "",
            "* `.ic` must be a netlist-level directive: inside a .control block it",
            "* is not honoured and the node would start at 0 V.",
            f".ic v(sn)={v_init:.9g}",
            "",
            ".control",
            "set noaskquit",
            "set numdgt=10",
            "* This transient resolves the SAME femto-ampere leakage as the DC sweep,",
            "* so it needs the DC tolerances -- in particular GMIN.  At NGSpice's",
            "* default GMIN (1e-12 S) the shunt conductance across the storage node's",
            "* junctions injects hundreds of femto-amps, which is orders of magnitude",
            "* above the real leakage and makes the cell appear to discharge far too",
            "* fast.",
            f"option abstol={self.sim_cfg.get('dc_abstol', 1e-18):g} "
            f"vntol={self.sim_cfg.get('dc_vntol', 1e-9):g} "
            f"gmin={self.sim_cfg.get('dc_gmin', 1e-15):g} "
            f"reltol={self.sim_cfg.get('dc_reltol', 1e-5):g}",
            "* No `uic`: the operating point is solved with v(sn) constrained to the",
            "* written level, then released.  With `uic` the cell-plate source steps",
            "* from 0 V at t=0 and couples straight through Cstore into the storage",
            "* node, offsetting the whole trajectory by V_plate.",
            f"tran {step:.6e} {t_stop:.6e}",
            f"echo {MEAS_BEGIN}",
            f"meas tran t_ret_tran WHEN v(sn)={v_fail:.9g} FALL=1",
            f"meas tran v_sn_end FIND v(sn) AT={t_stop * (1.0 - 1e-9):.9e}",
            f"echo {MEAS_END}",
            ".endc",
            ".end",
            "",
        ])

    # -- PySpice interoperability ----------------------------------------
    def to_pyspice_circuit(self, p: CellParameters):
        """Return an equivalent :class:`PySpice.Spice.Netlist.Circuit`.

        The heavy sampling campaign runs NGSpice in batch mode for process
        safety, but the identical topology is also expressible as a PySpice
        ``Circuit`` -- this is what the ``pyspice-shared`` backend and the
        cross-backend equivalence test in ``tests/`` consume.
        """
        from PySpice.Spice.Netlist import Circuit           # local: optional at import time

        t = self.schedule(p)
        circuit = Circuit(f"dram_1t1c_sample_{p.sample_id}")
        circuit.raw_spice += f'.include "{Path(self.tech.card_for(p.corner)).as_posix()}"\n'
        circuit.raw_spice += (
            f".options TEMP={p.temperature_c:.6g} "
            f"TNOM={self.sim_cfg.get('tnom', 27)}\n")
        circuit.raw_spice += self._core_devices(p) + "\n"
        circuit.raw_spice += self._stimuli(p, t) + "\n"
        return circuit
