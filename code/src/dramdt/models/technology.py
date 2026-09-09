"""SPICE technology handling: model cards, process corners and mismatch.

The framework ships with the *Predictive Technology Model* (PTM) BSIM4
(``level=54``) bulk-CMOS cards for the 45/32/22 nm HP and LP nodes.  Those
cards enable ``igcmod``/``igbmod`` (gate tunnelling), ``diomod`` (junction
leakage) and non-zero ``agidl``/``bgidl``/``cgidl`` (gate-induced drain
leakage) -- the three mechanisms that dominate DRAM storage-node charge loss --
so retention is simulated from device physics rather than assumed.

Process corners
---------------
PTM publishes a single typical card per node.  The classical SS/FF/SF/FS
corners are *synthesised* from it by applying the shifts declared under
``technology.corners`` in the configuration:

    vth0  <- vth0 + sign * dvth0        (magnitude shift; sign-aware for PMOS)
    u0    <- u0  * u0_scale
    toxe  <- toxe * tox_scale           (also toxp/toxm, kept consistent)

This is the standard first-order corner-emulation recipe.  The shifts are
configuration values, not hidden constants, and the generated cards are written
to disk so that every simulation is traceable to an explicit model file.

Local mismatch
--------------
Random device-to-device mismatch uses the Pelgrom model,

    sigma_vth = AVT / sqrt(W * L)

applied through the BSIM4 *instance* parameter ``delvto``, so each transistor
in a Monte-Carlo sample carries its own threshold offset.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config import Config, ConfigError
from ..paths import SPICE_MODEL_DIR

__all__ = [
    "CornerDefinition", "Technology", "generate_corner_cards",
    "ModelCard", "pelgrom_sigma_vth",
]

_MODEL_RE = re.compile(r"^\s*\.model\s+(\S+)\s+(\S+)", re.IGNORECASE)


# --------------------------------------------------------------------------
# model-card parsing / rewriting
# --------------------------------------------------------------------------
@dataclass
class ModelCard:
    """A parsed BSIM model card file (one or more ``.model`` blocks)."""

    path: Path
    text: str
    blocks: dict[str, tuple[int, int]] = field(default_factory=dict)  # name -> (start, end) line idx

    @classmethod
    def load(cls, path: str | Path) -> "ModelCard":
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"SPICE model card not found: {p}")
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        starts: list[tuple[str, int]] = []
        for i, line in enumerate(lines):
            m = _MODEL_RE.match(line)
            if m:
                starts.append((m.group(1).lower(), i))
        blocks: dict[str, tuple[int, int]] = {}
        for k, (name, s) in enumerate(starts):
            e = starts[k + 1][1] if k + 1 < len(starts) else len(lines)
            blocks[name] = (s, e)
        return cls(path=p, text=text, blocks=blocks)

    @property
    def device_names(self) -> list[str]:
        return list(self.blocks)

    def get_param(self, device: str, param: str) -> float | None:
        """Read a parameter value from one ``.model`` block."""
        if device.lower() not in self.blocks:
            return None
        s, e = self.blocks[device.lower()]
        chunk = "\n".join(self.text.splitlines()[s:e])
        m = re.search(rf"\b{re.escape(param)}\s*=\s*([-+0-9.eE]+)", chunk, re.IGNORECASE)
        return float(m.group(1)) if m else None

    def with_scaled(self, edits: Mapping[str, Mapping[str, float]]) -> str:
        """Return the card text with per-device parameter edits applied.

        ``edits`` maps ``device -> {param: new_value}``.  Only parameters that
        already exist in the block are rewritten (we never invent parameters,
        which would silently change the model's meaning).
        """
        lines = self.text.splitlines()
        for device, params in edits.items():
            key = device.lower()
            if key not in self.blocks:
                continue
            s, e = self.blocks[key]
            for param, value in params.items():
                pat = re.compile(rf"(\b{re.escape(param)}\s*=\s*)([-+0-9.eE]+)", re.IGNORECASE)
                for i in range(s, e):
                    if pat.search(lines[i]):
                        lines[i] = pat.sub(
                            lambda m: f"{m.group(1)}{value:.6g}", lines[i], count=1)
                        break
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# corners
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CornerDefinition:
    """First-order shifts that synthesise a process corner from a typical card."""

    name: str
    nmos_dvth0: float = 0.0      # volts, added to |vth0|
    pmos_dvth0: float = 0.0
    nmos_u0_scale: float = 1.0
    pmos_u0_scale: float = 1.0
    nmos_tox_scale: float = 1.0
    pmos_tox_scale: float = 1.0
    description: str = ""

    @classmethod
    def from_mapping(cls, name: str, spec: Mapping[str, Any]) -> "CornerDefinition":
        return cls(
            name=name,
            nmos_dvth0=float(spec.get("nmos_dvth0", 0.0)),
            pmos_dvth0=float(spec.get("pmos_dvth0", 0.0)),
            nmos_u0_scale=float(spec.get("nmos_u0_scale", 1.0)),
            pmos_u0_scale=float(spec.get("pmos_u0_scale", 1.0)),
            nmos_tox_scale=float(spec.get("nmos_tox_scale", 1.0)),
            pmos_tox_scale=float(spec.get("pmos_tox_scale", 1.0)),
            description=str(spec.get("description", "")),
        )


def _corner_edits(card: ModelCard, corner: CornerDefinition) -> dict[str, dict[str, float]]:
    """Compute the concrete parameter values implementing *corner*."""
    edits: dict[str, dict[str, float]] = {}
    for dev, dvth, uscale, tscale in (
        ("nmos", corner.nmos_dvth0, corner.nmos_u0_scale, corner.nmos_tox_scale),
        ("pmos", corner.pmos_dvth0, corner.pmos_u0_scale, corner.pmos_tox_scale),
    ):
        if dev not in card.blocks:
            continue
        e: dict[str, float] = {}
        vth0 = card.get_param(dev, "vth0")
        if vth0 is not None and dvth:
            # dvth0 is a *magnitude* shift whose own sign carries the meaning:
            # positive == slower (larger |vth0|), negative == faster.  The
            # device polarity only supplies the direction in which "larger
            # magnitude" points, so it must multiply dvth rather than replace
            # its sign -- using copysign(dvth, vth0) here would silently turn
            # every fast corner into a slow one.
            polarity = math.copysign(1.0, vth0 if vth0 != 0 else 1.0)
            e["vth0"] = vth0 + polarity * dvth
        u0 = card.get_param(dev, "u0")
        if u0 is not None and uscale != 1.0:
            e["u0"] = u0 * uscale
        if tscale != 1.0:
            for tp in ("toxe", "toxp", "toxm"):
                val = card.get_param(dev, tp)
                if val is not None:
                    e[tp] = val * tscale
        if e:
            edits[dev] = e
    return edits


def generate_corner_cards(base_card: str | Path,
                          corners: Mapping[str, CornerDefinition],
                          out_dir: str | Path | None = None,
                          overwrite: bool = False) -> dict[str, Path]:
    """Materialise one model-card file per process corner.

    Returns a mapping ``corner name -> path``.  Files are written only when
    absent (or when ``overwrite``), so repeated runs are idempotent.
    """
    card = ModelCard.load(base_card)
    out = Path(out_dir) if out_dir else Path(base_card).parent / "corners"
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(base_card).stem

    paths: dict[str, Path] = {}
    for name, corner in corners.items():
        target = out / f"{stem}_{name.lower()}.pm"
        if overwrite or not target.exists():
            edits = _corner_edits(card, corner)
            body = card.with_scaled(edits)
            header = (
                f"* ---------------------------------------------------------------\n"
                f"* Process corner '{name}' synthesised by dramdt from {Path(base_card).name}\n"
                f"* {corner.description}\n"
                f"* Applied shifts: {edits if edits else 'none (typical corner)'}\n"
                f"* Corner emulation is first-order (vth0 / u0 / tox); it is NOT a\n"
                f"* foundry-signed-off corner model.\n"
                f"* ---------------------------------------------------------------\n"
            )
            target.write_text(header + body, encoding="utf-8")
        paths[name] = target
    return paths


# --------------------------------------------------------------------------
# mismatch
# --------------------------------------------------------------------------
def pelgrom_sigma_vth(avt_v_um: float, w_m: float, l_m: float) -> float:
    """Pelgrom threshold-mismatch sigma.

    Parameters
    ----------
    avt_v_um : Pelgrom coefficient in V*um (e.g. 3.5e-3 for 45 nm).
    w_m, l_m : device width/length in metres.
    """
    w_um = max(w_m, 1e-12) * 1e6
    l_um = max(l_m, 1e-12) * 1e6
    return avt_v_um / math.sqrt(w_um * l_um)


# --------------------------------------------------------------------------
# facade
# --------------------------------------------------------------------------
@dataclass
class Technology:
    """Everything the netlist builder needs to know about the process."""

    node_nm: float
    flavour: str
    base_card: Path
    corner_cards: dict[str, Path]
    corners: dict[str, CornerDefinition]
    nmos_model: str = "nmos"
    pmos_model: str = "pmos"
    vdd_nominal: float = 1.1
    avt_v_um: float = 3.5e-3
    au0_pct_um: float = 1.0
    citation: str = ""

    # -- construction -----------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Config | Mapping[str, Any],
                    generate: bool = True) -> "Technology":
        tech = dict(cfg.get("technology", {}))
        if not tech:
            raise ConfigError("Configuration has no 'technology' section")

        raw_card = tech.get("model_card")
        if not raw_card:
            raise ConfigError("technology.model_card is required")
        card_path = Path(raw_card)
        if not card_path.is_absolute():
            cand = SPICE_MODEL_DIR / card_path
            card_path = cand if cand.exists() else card_path
        if not card_path.exists():
            raise ConfigError(
                f"SPICE model card {raw_card!r} not found (looked at {card_path}).  "
                "Run scripts/00_setup_environment.py to fetch the PTM cards.")

        corner_specs = tech.get("corners") or {"TT": {}}
        corners = {k: CornerDefinition.from_mapping(k, v or {}) for k, v in corner_specs.items()}

        corner_dir = tech.get("corner_dir")
        corner_dir = Path(corner_dir) if corner_dir else (SPICE_MODEL_DIR / "corners")
        if not Path(corner_dir).is_absolute():
            corner_dir = SPICE_MODEL_DIR / corner_dir

        corner_cards = (generate_corner_cards(card_path, corners, corner_dir)
                        if generate else {})

        return cls(
            node_nm=float(tech.get("node_nm", 45)),
            flavour=str(tech.get("flavour", "LP")),
            base_card=card_path,
            corner_cards=corner_cards,
            corners=corners,
            nmos_model=str(tech.get("nmos_model", "nmos")),
            pmos_model=str(tech.get("pmos_model", "pmos")),
            vdd_nominal=float(tech.get("vdd_nominal", 1.1)),
            avt_v_um=float(tech.get("mismatch", {}).get("avt_v_um", 3.5e-3)),
            au0_pct_um=float(tech.get("mismatch", {}).get("au0_pct_um", 1.0)),
            citation=str(tech.get("citation", "")),
        )

    # -- accessors --------------------------------------------------------
    @property
    def corner_names(self) -> list[str]:
        return list(self.corners)

    def card_for(self, corner: str) -> Path:
        """Path to the model card implementing *corner* (falls back to typical)."""
        if corner in self.corner_cards:
            return self.corner_cards[corner]
        if corner.upper() in self.corner_cards:
            return self.corner_cards[corner.upper()]
        return self.base_card

    def sigma_vth(self, w_m: float, l_m: float) -> float:
        return pelgrom_sigma_vth(self.avt_v_um, w_m, l_m)

    def sample_mismatch(self, devices: Mapping[str, tuple[float, float]],
                        rng: np.random.Generator) -> dict[str, float]:
        """Draw a ``delvto`` offset for each ``name -> (W, L)`` device."""
        return {name: float(rng.normal(0.0, self.sigma_vth(w, l)))
                for name, (w, l) in devices.items()}

    def fingerprint(self) -> str:
        """Hash of the actual model-card contents actually used (provenance)."""
        h = hashlib.sha256()
        for p in [self.base_card] + sorted(self.corner_cards.values()):
            if Path(p).exists():
                h.update(Path(p).read_bytes())
        return h.hexdigest()[:16]

    def describe(self) -> dict[str, Any]:
        return {
            "node_nm": self.node_nm,
            "flavour": self.flavour,
            "base_card": str(self.base_card),
            "corners": {k: str(v) for k, v in self.corner_cards.items()},
            "vdd_nominal": self.vdd_nominal,
            "avt_v_um": self.avt_v_um,
            "citation": self.citation,
            "model_fingerprint": self.fingerprint(),
        }
