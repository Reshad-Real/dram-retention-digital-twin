"""Stage 7 -- DRAM assist techniques.

An *assist technique* is a named, configuration-declared transformation of a
design point.  Each technique may ``set`` a parameter to an absolute value,
``scale`` it multiplicatively, or ``offset`` it additively, and may declare
*side effects* -- the physical price paid for the assist.  For example a
high-k / high-C storage capacitor raises ``cs`` but also degrades the dielectric
leakage resistance ``rdiel``; encoding that coupling is what makes the assist
comparison meaningful instead of a free lunch.

The techniques compared here mirror block 7 of the proposed method:

===========================  ==========================================
``high_k_capacitor``         high-k / high-C storage capacitor
``wordline_boost``           elevated VPP on the wordline
``negative_wordline``        negative WL low level (retention assist)
``bitline_precharge_opt``    tuned VBL,pre away from VDD/2
``sense_amp_upsize``         larger / rebalanced sense-amplifier latch
``refresh_interval_opt``     retention-aware refresh scheduling
``combined_low_power``       all of the above applied together
===========================  ==========================================

``refresh_interval_opt`` deliberately has no electrical override: it acts at the
*metric* level, through the refresh-policy guard band applied in
:mod:`dramdt.simulation.measure`.  It is represented here so that the assist
comparison covers it uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping

from ..config import Config, ConfigError

__all__ = ["AssistTechnique", "AssistLibrary", "apply_assist"]


@dataclass(frozen=True)
class AssistTechnique:
    """A declarative transformation of a design point."""

    name: str
    description: str = ""
    set_values: Mapping[str, float] = field(default_factory=dict)
    scale_values: Mapping[str, float] = field(default_factory=dict)
    offset_values: Mapping[str, float] = field(default_factory=dict)
    side_effects: Mapping[str, Any] = field(default_factory=dict)
    refresh_policy: str | None = None
    category: str = "assist"

    @classmethod
    def from_mapping(cls, name: str, spec: Mapping[str, Any]) -> "AssistTechnique":
        if not isinstance(spec, Mapping):
            raise ConfigError(f"assist technique {name!r} must be a mapping")
        return cls(
            name=name,
            description=str(spec.get("description", "")),
            set_values=dict(spec.get("set", {}) or {}),
            scale_values=dict(spec.get("scale", {}) or {}),
            offset_values=dict(spec.get("offset", {}) or {}),
            side_effects=dict(spec.get("side_effects", {}) or {}),
            refresh_policy=spec.get("refresh_policy"),
            category=str(spec.get("category", "assist")),
        )

    # ------------------------------------------------------------------
    def apply(self, sample: Mapping[str, Any],
              clip: Mapping[str, tuple[float, float]] | None = None) -> dict[str, Any]:
        """Return a copy of *sample* with this technique applied.

        Order is ``scale`` -> ``offset`` -> ``set`` -> ``side_effects`` so that
        an absolute ``set`` always wins over a relative adjustment, and side
        effects (the physical cost) are applied last and cannot be undone by
        the technique's own knobs.
        """
        out: dict[str, Any] = dict(sample)

        for key, factor in self.scale_values.items():
            if key in out and isinstance(out[key], (int, float)):
                out[key] = float(out[key]) * float(factor)
        for key, delta in self.offset_values.items():
            if key in out and isinstance(out[key], (int, float)):
                out[key] = float(out[key]) + float(delta)
            elif isinstance(delta, (int, float)):
                out[key] = float(delta)
        for key, value in self.set_values.items():
            out[key] = value
        for key, value in self.side_effects.items():
            out[key] = value

        if clip:
            for key, (lo, hi) in clip.items():
                if key in out and isinstance(out[key], (int, float)):
                    out[key] = min(max(float(out[key]), float(lo)), float(hi))

        out["assist_config"] = self.name
        if self.refresh_policy:
            out["refresh_policy"] = self.refresh_policy
        return out

    def touched_parameters(self) -> list[str]:
        keys = set(self.set_values) | set(self.scale_values) | set(self.offset_values)
        keys |= set(self.side_effects)
        return sorted(keys)

    def describe(self) -> dict[str, Any]:
        return {
            "technique": self.name,
            "category": self.category,
            "description": self.description,
            "parameters_modified": ", ".join(self.touched_parameters()) or "-",
            "refresh_policy": self.refresh_policy or "default",
        }


class AssistLibrary(dict):
    """Named collection of :class:`AssistTechnique`, loaded from configuration."""

    @classmethod
    def from_config(cls, cfg: Config | Mapping[str, Any]) -> "AssistLibrary":
        section = cfg.get("assist_techniques") or {}
        lib = cls()
        if "baseline" not in section:
            lib["baseline"] = AssistTechnique(
                name="baseline",
                description="Unassisted reference design point.",
                category="reference")
        for name, spec in section.items():
            lib[name] = AssistTechnique.from_mapping(name, spec or {})
        if not lib:
            raise ConfigError("No assist techniques defined and no baseline available")
        return lib

    @property
    def names(self) -> list[str]:
        return list(self)

    def describe(self) -> list[dict[str, Any]]:
        return [t.describe() for t in self.values()]


def apply_assist(sample: Mapping[str, Any],
                 technique: AssistTechnique | str,
                 library: AssistLibrary | None = None,
                 clip: Mapping[str, tuple[float, float]] | None = None) -> dict[str, Any]:
    """Convenience wrapper: apply *technique* (or its name) to *sample*."""
    if isinstance(technique, str):
        if library is None:
            raise ConfigError("A library is required when addressing a technique by name")
        if technique not in library:
            raise ConfigError(f"Unknown assist technique {technique!r}; "
                              f"available: {', '.join(library.names)}")
        technique = library[technique]
    return technique.apply(sample, clip=clip)
