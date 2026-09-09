"""Configuration loading, merging, validation and hashing.

The framework is *entirely* configuration driven.  A run is described by one or
more YAML files that are deep-merged in order, so a new DRAM architecture is
introduced by adding a YAML file -- never by editing Python.

Key sections of a configuration document
----------------------------------------
``meta``          free-form provenance (name, node, citation of the model card)
``technology``    SPICE model card, corner definitions, mismatch parameters
``architecture``  fixed topology parameters of the DRAM array/cell
``timing``        the read/write/precharge waveform schedule
``design_space``  the sampled variables (bounds, scale, units, derived exprs)
``metrics``       definitions/thresholds used by the measurement extractor
``simulation``    solver options, analyses to run, timeouts
``experiment``    sample count, sampler, seeds, parallelism
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from .paths import CONFIG_DIR, PROJECT_ROOT

__all__ = [
    "Config", "load_config", "deep_merge", "config_hash", "ConfigError",
    "DesignVariable", "DesignSpace",
]


class ConfigError(ValueError):
    """Raised when a configuration document is missing or inconsistent."""


# --------------------------------------------------------------------------
# generic dict utilities
# --------------------------------------------------------------------------
def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (``override`` wins)."""
    out: dict[str, Any] = dict(copy.deepcopy(dict(base)))
    for k, v in override.items():
        if k in out and isinstance(out[k], Mapping) and isinstance(v, Mapping):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def config_hash(cfg: Mapping[str, Any]) -> str:
    """Stable SHA-256 over the canonicalised configuration."""
    blob = json.dumps(cfg, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# safe arithmetic evaluation for `derived:` expressions
# --------------------------------------------------------------------------
_SAFE_NAMES = {
    "abs": abs, "min": min, "max": max, "round": round, "pow": pow,
    "exp": math.exp, "log": math.log, "log10": math.log10, "sqrt": math.sqrt,
    "floor": math.floor, "ceil": math.ceil, "pi": math.pi, "e": math.e,
}
_EXPR_OK = re.compile(r"^[0-9A-Za-z_+\-*/%.,()\s<>=!&|?:']+$")

#: Compiled derived expressions, keyed by source.  The digital twin evaluates
#: these on every prediction, so re-compiling the string each time is a
#: measurable cost inside an optimisation loop.
_EXPR_CACHE: dict[str, Any] = {}


def _compile_expression(expr: str):
    code = _EXPR_CACHE.get(expr)
    if code is None:
        if not _EXPR_OK.match(expr):
            raise ConfigError(f"Unsafe or malformed expression: {expr!r}")
        code = compile(expr, "<dramdt-derived>", "eval")
        _EXPR_CACHE[expr] = code
    return code


def evaluate_expression(expr: str, variables: Mapping[str, Any]) -> float:
    """Evaluate a small arithmetic expression against ``variables``.

    Only arithmetic, comparisons and the whitelisted maths functions above are
    reachable: ``__builtins__`` is emptied and the source is pattern-checked
    before it is compiled.
    """
    if not isinstance(expr, str):
        return float(expr)
    code = _compile_expression(expr)
    env = dict(_SAFE_NAMES)
    env.update(variables)
    try:
        return eval(code, {"__builtins__": {}}, env)      # noqa: S307 - guarded above
    except Exception as exc:
        raise ConfigError(f"Failed to evaluate {expr!r}: {exc}") from exc


# --------------------------------------------------------------------------
# design space
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DesignVariable:
    """One sampled dimension of the design space."""

    name: str
    low: float
    high: float
    unit: str = ""
    scale: str = "linear"           # 'linear' | 'log'
    kind: str = "continuous"        # 'continuous' | 'integer' | 'categorical'
    categories: tuple[Any, ...] = ()
    description: str = ""
    group: str = "design"           # design | environment | process | assist

    def __post_init__(self) -> None:
        if self.kind == "categorical":
            if not self.categories:
                raise ConfigError(f"Categorical variable {self.name!r} has no categories")
        else:
            if not (self.high > self.low):
                raise ConfigError(
                    f"Variable {self.name!r}: high ({self.high}) must exceed low ({self.low})")
            if self.scale == "log" and self.low <= 0:
                raise ConfigError(f"Variable {self.name!r}: log scale requires low > 0")

    # -- unit-hypercube <-> physical mapping ------------------------------
    def decode(self, u: float) -> Any:
        """Map ``u`` in [0, 1] onto the physical domain of this variable."""
        u = min(max(float(u), 0.0), 1.0)
        if self.kind == "categorical":
            idx = min(int(u * len(self.categories)), len(self.categories) - 1)
            return self.categories[idx]
        if self.scale == "log":
            v = math.exp(math.log(self.low) + u * (math.log(self.high) - math.log(self.low)))
        else:
            v = self.low + u * (self.high - self.low)
        if self.kind == "integer":
            return int(round(v))
        return float(v)

    def encode(self, value: Any) -> float:
        """Inverse of :meth:`decode` (used to place known designs in the cube)."""
        if self.kind == "categorical":
            try:
                i = list(self.categories).index(value)
            except ValueError as exc:
                raise ConfigError(f"{value!r} not a category of {self.name!r}") from exc
            return (i + 0.5) / len(self.categories)
        v = float(value)
        if self.scale == "log":
            return (math.log(v) - math.log(self.low)) / (math.log(self.high) - math.log(self.low))
        return (v - self.low) / (self.high - self.low)


@dataclass
class DesignSpace:
    """An ordered collection of :class:`DesignVariable` plus derived quantities."""

    variables: tuple[DesignVariable, ...]
    derived: dict[str, str]
    constants: dict[str, float]

    # -- construction -----------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "DesignSpace":
        section = cfg.get("design_space")
        if not section:
            raise ConfigError("Configuration has no 'design_space' section")
        raw_vars = section.get("variables") or {}
        if not raw_vars:
            raise ConfigError("'design_space.variables' is empty")

        variables: list[DesignVariable] = []
        for name, spec in raw_vars.items():
            if not isinstance(spec, Mapping):
                raise ConfigError(f"design_space.variables.{name} must be a mapping")
            kind = spec.get("kind", "continuous")
            cats = tuple(spec.get("categories", ()) or ())
            variables.append(DesignVariable(
                name=name,
                low=float(spec.get("low", 0.0)),
                high=float(spec.get("high", 1.0)),
                unit=str(spec.get("unit", "")),
                scale=str(spec.get("scale", "linear")),
                kind=str(kind),
                categories=cats,
                description=str(spec.get("description", "")),
                group=str(spec.get("group", "design")),
            ))
        return cls(
            variables=tuple(variables),
            derived=dict(section.get("derived") or {}),
            constants={k: float(v) for k, v in (section.get("constants") or {}).items()},
        )

    # -- accessors --------------------------------------------------------
    @property
    def names(self) -> list[str]:
        return [v.name for v in self.variables]

    @property
    def n_dim(self) -> int:
        return len(self.variables)

    def __getitem__(self, name: str) -> DesignVariable:
        for v in self.variables:
            if v.name == name:
                return v
        raise KeyError(name)

    def continuous_variables(self) -> list[DesignVariable]:
        return [v for v in self.variables if v.kind != "categorical"]

    def bounds(self) -> tuple[list[float], list[float]]:
        """Physical lower/upper bounds of the non-categorical variables."""
        cv = self.continuous_variables()
        return [v.low for v in cv], [v.high for v in cv]

    # -- sample decoding --------------------------------------------------
    def decode_row(self, u: Sequence[float]) -> dict[str, Any]:
        """Decode a unit-hypercube row into a full physical parameter dict."""
        if len(u) != self.n_dim:
            raise ConfigError(f"Expected {self.n_dim} coordinates, got {len(u)}")
        params: dict[str, Any] = dict(self.constants)
        for var, ui in zip(self.variables, u):
            params[var.name] = var.decode(ui)
        return self.apply_derived(params)

    def apply_derived(self, params: dict[str, Any]) -> dict[str, Any]:
        """Evaluate the ``derived:`` expressions in declaration order."""
        out = dict(params)
        for name, expr in self.derived.items():
            numeric = {k: v for k, v in out.items() if isinstance(v, (int, float))}
            out[name] = evaluate_expression(expr, numeric)
        return out

    def describe(self) -> list[dict[str, Any]]:
        """Tabular description, used to emit the design-space table in the report."""
        rows = []
        for v in self.variables:
            rows.append({
                "variable": v.name,
                "group": v.group,
                "kind": v.kind,
                "low": v.low if v.kind != "categorical" else "",
                "high": v.high if v.kind != "categorical" else "",
                "categories": ", ".join(map(str, v.categories)) if v.categories else "",
                "unit": v.unit,
                "scale": v.scale,
                "description": v.description,
            })
        return rows


# --------------------------------------------------------------------------
# top-level config object
# --------------------------------------------------------------------------
class Config(dict):
    """A validated configuration document with convenience accessors."""

    def __init__(self, data: Mapping[str, Any], sources: Sequence[Path] = ()):
        super().__init__(copy.deepcopy(dict(data)))
        self.sources: tuple[Path, ...] = tuple(Path(s) for s in sources)
        self._design_space: DesignSpace | None = None

    # -- dotted access ----------------------------------------------------
    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        sentinel = object()
        val = self.get_path(dotted, sentinel)
        if val is sentinel:
            raise ConfigError(f"Required configuration key missing: {dotted!r}")
        return val

    # -- derived objects --------------------------------------------------
    @property
    def design_space(self) -> DesignSpace:
        if self._design_space is None:
            self._design_space = DesignSpace.from_config(self)
        return self._design_space

    @property
    def hash(self) -> str:
        return config_hash(self)

    @property
    def name(self) -> str:
        return str(self.get_path("meta.name", "unnamed"))

    def resolve_path(self, dotted: str, default: str | None = None) -> Path:
        """Resolve a path-valued config entry relative to the project root."""
        raw = self.get_path(dotted, default)
        if raw is None:
            raise ConfigError(f"Path key missing: {dotted!r}")
        p = Path(str(raw))
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(dict(self), sort_keys=False, default_flow_style=False),
                     encoding="utf-8")
        return p

    def summary(self) -> str:
        ds = self.design_space
        return (f"Config '{self.name}' [hash={self.hash}] "
                f"{ds.n_dim} design variables, "
                f"{len(ds.derived)} derived, sources={[s.name for s in self.sources]}")


# --------------------------------------------------------------------------
# loading + validation
# --------------------------------------------------------------------------
_REQUIRED_SECTIONS = ("meta", "technology", "architecture", "timing",
                      "design_space", "metrics", "simulation", "experiment")


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path} does not contain a YAML mapping")
    return dict(data)


def _resolve(spec: str | Path) -> Path:
    p = Path(spec)
    if p.is_absolute() and p.exists():
        return p
    for cand in (PROJECT_ROOT / p, CONFIG_DIR / p, CONFIG_DIR / f"{p}.yaml", Path.cwd() / p):
        if cand.exists():
            return cand
    raise ConfigError(f"Cannot locate configuration {spec!r}")


def validate(cfg: Mapping[str, Any]) -> list[str]:
    """Return a list of human-readable validation problems (empty == valid)."""
    problems: list[str] = []
    for sec in _REQUIRED_SECTIONS:
        if sec not in cfg:
            problems.append(f"missing required section '{sec}'")

    try:
        ds = DesignSpace.from_config(cfg)
    except ConfigError as exc:
        problems.append(str(exc))
        ds = None

    if ds is not None:
        # derived expressions must be evaluable at the design-space midpoint
        try:
            ds.decode_row([0.5] * ds.n_dim)
        except ConfigError as exc:
            problems.append(f"derived expression failure at midpoint: {exc}")

    n = cfg.get("experiment", {}).get("n_samples")
    if n is not None and (not isinstance(n, int) or n < 1):
        problems.append(f"experiment.n_samples must be a positive integer, got {n!r}")

    sampler = cfg.get("experiment", {}).get("sampler")
    if sampler is not None and sampler not in ("lhs", "sobol", "random", "grid"):
        problems.append(f"unknown experiment.sampler {sampler!r}")

    corners = cfg.get("technology", {}).get("corners")
    if corners is not None and not isinstance(corners, Mapping):
        problems.append("technology.corners must be a mapping of name -> shifts")

    return problems


def load_config(*specs: str | Path,
                overrides: Mapping[str, Any] | None = None,
                strict: bool = True) -> Config:
    """Load and deep-merge one or more YAML configuration documents.

    Parameters
    ----------
    *specs
        Paths (or bare names resolved inside ``configs/``).  Later documents
        override earlier ones.  A document may itself declare ``extends: <name>``
        which is loaded first.
    overrides
        Final mapping merged on top of everything (e.g. CLI switches).
    strict
        Raise on validation problems instead of only warning.
    """
    if not specs:
        specs = ("dram_1t1c_45nm_lp.yaml",)

    merged: dict[str, Any] = {}
    used: list[Path] = []

    def _load_one(spec: str | Path, seen: set[Path]) -> dict[str, Any]:
        path = _resolve(spec)
        if path in seen:
            raise ConfigError(f"Circular 'extends' chain at {path}")
        seen.add(path)
        doc = _read_yaml(path)
        parent = doc.pop("extends", None)
        base: dict[str, Any] = {}
        if parent:
            for pspec in ([parent] if isinstance(parent, (str, Path)) else list(parent)):
                base = deep_merge(base, _load_one(pspec, seen))
        used.append(path)
        return deep_merge(base, doc)

    for spec in specs:
        merged = deep_merge(merged, _load_one(spec, set()))

    if overrides:
        merged = deep_merge(merged, overrides)

    problems = validate(merged)
    if problems:
        msg = "Configuration validation problems:\n  - " + "\n  - ".join(problems)
        if strict:
            raise ConfigError(msg)
        import warnings
        warnings.warn(msg, stacklevel=2)

    return Config(merged, sources=used)
