"""The framework's algorithms as IEEE-style pseudocode.

Each entry renders to three forms:

``.tex``   ``algorithmic``/``algorithm`` float, ready for an IEEE template
``.md``    a fenced code block for the README and the Markdown report
``.txt``   plain text

The pseudocode is written against the *actual implementation*, so each spec
records the module and function it describes; if the code moves, the reference
moves with it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

__all__ = ["AlgorithmSpec", "ALGORITHM_SPECS", "render_algorithm",
           "write_all_algorithms"]


@dataclass
class AlgorithmSpec:
    number: int
    name: str
    implemented_by: str
    require: list[str] = field(default_factory=list)
    ensure: list[str] = field(default_factory=list)
    body: list[str] = field(default_factory=list)     # '>' prefixes add indent
    note: str = ""

    @property
    def label(self) -> str:
        return f"alg:{re.sub(r'[^a-z0-9]+', '_', self.name.lower()).strip('_')}"


# --------------------------------------------------------------------------
ALGORITHM_SPECS: list[AlgorithmSpec] = [
    AlgorithmSpec(
        number=1,
        name="Configuration-driven DRAM design-space sampling",
        implemented_by="dramdt.doe.sampling.generate_design + "
                       "dramdt.config.DesignSpace.decode_row",
        require=[
            r"design-space specification $\mathcal{D}=\{(v_i,\ell_i,u_i,s_i,k_i)\}_{i=1}^{d}$",
            r"derived-quantity expressions $\mathcal{E}$, constants $\mathcal{K}$",
            r"sample count $N$, master seed $\sigma$",
        ],
        ensure=[r"physical design points $\{\mathbf{p}^{(n)}\}_{n=1}^{N}$"],
        body=[
            r"$\sigma_{\text{doe}} \leftarrow \textsc{DeriveSeed}(\sigma, \text{`design'})$",
            r"$U \leftarrow \textsc{LatinHypercube}(N, d, \sigma_{\text{doe}})$"
            r"\Comment{one sample per stratum in every 1-D projection}",
            r"record $\textsc{Discrepancy}(U)$ and $\min_{a\neq b}\lVert u_a-u_b\rVert$"
            r"\Comment{design quality is measured, not assumed}",
            r"\For{$n \leftarrow 1$ \KwTo $N$}",
            r"> $\mathbf{p}^{(n)} \leftarrow \mathcal{K}$",
            r"> \For{$i \leftarrow 1$ \KwTo $d$}",
            r"> > \uIf{$k_i$ is categorical}",
            r"> > > $p^{(n)}_i \leftarrow c_{\lfloor U_{ni}\,|C_i| \rfloor}$",
            r"> > \uElseIf{$s_i = \log$}",
            r"> > > $p^{(n)}_i \leftarrow \exp\!\big(\ln \ell_i + U_{ni}(\ln u_i - \ln \ell_i)\big)$",
            r"> > \Else",
            r"> > > $p^{(n)}_i \leftarrow \ell_i + U_{ni}(u_i-\ell_i)$",
            r"> > \EndIf",
            r"> \EndFor",
            r"> \For{$e \in \mathcal{E}$ \textbf{in declaration order}}",
            r"> > $\mathbf{p}^{(n)}[\text{lhs}(e)] \leftarrow \textsc{Eval}(e, \mathbf{p}^{(n)})$",
            r"> \EndFor",
            r"\EndFor",
            r"\Return $\{\mathbf{p}^{(n)}\}$",
        ],
        note="Nothing about the DRAM architecture is hard-coded: retargeting the "
             "framework means supplying a different $\\mathcal{D}$.",
    ),

    AlgorithmSpec(
        number=2,
        name="SPICE characterisation of one DRAM design point",
        implemented_by="dramdt.models.cell.DramCellBuilder + "
                       "dramdt.simulation.runner.SpiceRunner",
        require=[r"design point $\mathbf{p}$, technology $\mathcal{T}$, timing schedule $\mathcal{S}$"],
        ensure=[r"raw measurement set $\mathcal{M}$, leakage characteristic $I_{\mathrm{leak}}(V)$"],
        body=[
            r"$\mathcal{C} \leftarrow \mathcal{T}.\textsc{CardFor}(\mathbf{p}.\text{corner})$"
            r"\Comment{corner-specific BSIM4 card}",
            r"build 1T1C cell, bitline pair, boosted precharge, write driver, sense latch",
            r"append \emph{leakage replica}: an identical access device whose storage node"
            r" is driven by probe source $V_{\mathrm{snp}}$",
            r"$\mathcal{S} \leftarrow \textsc{Schedule}(\mathbf{p})$"
            r"\Comment{write $\rightarrow$ precharge $\rightarrow$ read $\rightarrow$ sense}",
            r"\textbf{analysis 1:} \texttt{op} $\Rightarrow$ quiescent supply currents",
            r"\textbf{analysis 2:} \texttt{tran} over $\mathcal{S}$ with loose ABSTOL"
            r"\Comment{$\mu$A-scale switching}",
            r"> measure $V_{SN}^{\mathrm{hold}}$, $t_{\mathrm{write}}$, "
            r"$V_{BL},V_{\overline{BL}}$ at $t_{\mathrm{SA}}$, $t_{\mathrm{read}}$, "
            r"$\int i_{\mathrm{dd}}\,\mathrm{d}t$",
            r"\textbf{analysis 3:} \texttt{dc} sweep $V_{\mathrm{snp}}: 0 \to V_{DD}$"
            r" with tight ABSTOL\Comment{fA-scale leakage}",
            r"> $I_{\mathrm{leak}}(V) \leftarrow -\,i(V_{\mathrm{snp}})$"
            r"\Comment{current leaving the storage node}",
            r"\Return $\mathcal{M}, I_{\mathrm{leak}}(\cdot)$",
        ],
        note="Tolerances are set per analysis: one global ABSTOL cannot resolve "
             "femto-ampere leakage and micro-ampere switching in the same deck.",
    ),

    AlgorithmSpec(
        number=3,
        name="Quasi-static retention time from measured leakage",
        implemented_by="dramdt.simulation.retention.quasistatic_retention",
        require=[
            r"leakage characteristic $I_{\mathrm{leak}}(V)$ (measured, Algorithm~2)",
            r"node capacitance $C_{\mathrm{node}}$, written level $V_{\mathrm{init}}$",
            r"precharge level $V_{\mathrm{BLpre}}$, capacitances $C_s, C_{BL}$, "
            r"sense-amplifier offset $\Delta V_{\min}$",
        ],
        ensure=[r"retention time $t_{\mathrm{ret}}$ and a censoring flag"],
        body=[
            r"$V_{\mathrm{fail}} \leftarrow V_{\mathrm{BLpre}} + "
            r"\Delta V_{\min}\,\dfrac{C_s+C_{BL}}{C_s}$"
            r"\Comment{charge sharing no longer resolvable}",
            r"\If{$V_{\mathrm{init}} \le V_{\mathrm{fail}}$}",
            r"> \Return $t_{\mathrm{ret}} \leftarrow 0$"
            r"\Comment{a real observation, not a missing value}",
            r"\EndIf",
            r"$\mathcal{V} \leftarrow$ uniform grid on $[V_{\mathrm{fail}}, V_{\mathrm{init}}]$",
            r"$I \leftarrow \textsc{Interp}(I_{\mathrm{leak}}, \mathcal{V})$",
            r"\If{$\min I \le \epsilon$}",
            r"> \Return $t_{\mathrm{ret}} \leftarrow \infty$ \textbf{(censored)}"
            r"\Comment{node equilibrates above $V_{\mathrm{fail}}$}",
            r"\EndIf",
            r"$t_{\mathrm{ret}} \leftarrow \displaystyle\int_{V_{\mathrm{fail}}}"
            r"^{V_{\mathrm{init}}} \frac{C_{\mathrm{node}}}{I_{\mathrm{leak}}(V)}\,\mathrm{d}V$"
            r"\Comment{trapezoidal quadrature}",
            r"\Return $t_{\mathrm{ret}}$",
        ],
        note="Exact for an isolated capacitor under quasi-static leakage; validated "
             "against direct long-window transient simulation.",
    ),

    AlgorithmSpec(
        number=4,
        name="Cross-validated surrogate selection",
        implemented_by="dramdt.surrogate.train.SurrogateTrainer",
        require=[r"dataset $\mathcal{D}$ with a fixed train/val/test split",
                 r"model zoo $\mathcal{Z}$, folds $K$, primary metric $\mu$"],
        ensure=[r"per-target best surrogate $f^\star_y$ and its held-out score"],
        body=[
            r"\ForEach{target $y$}",
            r"> $\mathcal{D}_y \leftarrow \{(\mathbf{x},y) \in \mathcal{D} : y "
            r"\text{ observed}\}$\Comment{drop per target, never globally}",
            r"> \ForEach{model $m \in \mathcal{Z}$}",
            r"> > \For{$k \leftarrow 1$ \KwTo $K$}",
            r"> > > fit $m$ on $\mathcal{D}_y^{\mathrm{dev}}\setminus \mathrm{fold}_k$; "
            r"score on $\mathrm{fold}_k$",
            r"> > > retain the fold score\Comment{needed by the statistical tests}",
            r"> > \EndFor",
            r"> > refit $m$ on all of $\mathcal{D}_y^{\mathrm{dev}}$; "
            r"score once on $\mathcal{D}_y^{\mathrm{test}}$",
            r"> \EndFor",
            r"> $f^\star_y \leftarrow \arg\max_m \overline{\mu}_{\mathrm{cv}}(m)$",
            r"\EndFor",
            r"\Return $\{f^\star_y\}$",
        ],
        note="The test split is touched exactly once per model and never "
             "participates in selection or tuning.",
    ),

    AlgorithmSpec(
        number=5,
        name="Robust surrogate-driven multi-objective optimisation",
        implemented_by="dramdt.optimization.problem.DramDesignProblem + "
                       "dramdt.optimization.runner.OptimizationRunner",
        require=[r"digital twin $\mathcal{F}$, decision variables $\mathbf{x}$",
                 r"objectives $\{f_j\}$ with senses, constraints $\{g_c\}$",
                 r"operating conditions $\mathcal{Q}$, budget $B$, repetitions $R$"],
        ensure=[r"Pareto set $\mathcal{P}$ and indicator distributions"],
        body=[
            r"\ForEach{algorithm $a$, repetition $r \in 1..R$}",
            r"> seed $\leftarrow \textsc{DeriveSeed}(\sigma, a, r)$; budget $B$ "
            r"\Comment{identical for every algorithm}",
            r"> \While{budget remains}",
            r"> > \ForEach{candidate population $X$}",
            r"> > > \ForEach{condition $q \in \mathcal{Q}$}",
            r"> > > > $\mathbf{R}_q \leftarrow \mathcal{F}(\textsc{Decode}(X, q))$",
            r"> > > \EndFor",
            r"> > > $f_j \leftarrow \min_q$ if $f_j$ maximised else $\max_q$"
            r"\Comment{worst case over PVT}",
            r"> > > $g_c \leftarrow$ normalised violation, $g_c \le 0$ feasible",
            r"> > \EndFor",
            r"> \EndWhile",
            r"\EndFor",
            r"$\mathcal{P}^{\mathrm{ref}} \leftarrow \textsc{NonDominated}"
            r"\big(\bigcup_{a,r}\mathcal{P}_{a,r}\big)$",
            r"compute HV, IGD$^+$, spacing, spread per run against "
            r"$\mathcal{P}^{\mathrm{ref}}$",
            r"\Return $\mathcal{P}^{\mathrm{ref}}$, indicator distributions",
        ],
        note="Worst-case reduction over $\\mathcal{Q}$ turns the search into a robust "
             "design problem; equal budgets and $R$ repetitions make the comparison "
             "statistically testable.",
    ),

    AlgorithmSpec(
        number=6,
        name="Digital-twin robustness and yield estimation",
        implemented_by="dramdt.robustness.analysis.RobustnessAnalyzer",
        require=[r"design $\mathbf{x}^\star$, process sigmas $\{\varsigma_p\}$",
                 r"specification limits $\{(y_s, \tau_s, \text{sense})\}$, trials $M$"],
        ensure=[r"joint yield $Y$, per-spec yields, worst-case corner, SPICE check"],
        body=[
            r"\For{$m \leftarrow 1$ \KwTo $M$}",
            r"> $\mathbf{x}^{(m)} \leftarrow \mathbf{x}^\star \odot "
            r"(1 + \boldsymbol{\varepsilon}),\ \varepsilon_p\sim\mathcal{N}(0,\varsigma_p^2)$",
            r"\EndFor",
            r"$\mathbf{Y} \leftarrow \mathcal{F}(\{\mathbf{x}^{(m)}\})$"
            r"\Comment{twin: $M$ evaluations in milliseconds}",
            r"$Y \leftarrow \frac{1}{M}\big|\{m : \forall s,\ y_s^{(m)} "
            r"\text{ meets } \tau_s\}\big|$"
            r"\Comment{joint, not the product of marginals}",
            r"$\mathcal{G} \leftarrow$ evaluate $\mathbf{x}^\star$ over "
            r"corner $\times$ temperature grid; record worst case per spec",
            r"re-simulate a random subset of $\{\mathbf{x}^{(m)}\}$ in NGSpice; "
            r"report $|Y_{\text{twin}} - Y_{\text{SPICE}}|$",
            r"\Return $Y$, per-spec yields, $\mathcal{G}$, agreement",
        ],
        note="A robustness claim that has not been checked against the simulator is "
             "not reported as a result.",
    ),
]


# --------------------------------------------------------------------------
def _body_to_latex(body: Sequence[str]) -> list[str]:
    out = []
    for raw in body:
        depth = 0
        line = raw
        while line.startswith(">"):
            depth += 1
            line = line[1:].lstrip()
        prefix = ""
        if not line.startswith("\\"):
            prefix = r"\State "
        out.append("    " * (depth + 1) + prefix + line)
    return out


def render_algorithm(spec: AlgorithmSpec, fmt: str = "tex") -> str:
    """Render one algorithm to ``tex``, ``md`` or ``txt``."""
    if fmt == "tex":
        lines = [
            r"\begin{algorithm}[!t]",
            rf"\caption{{{spec.name}}}",
            rf"\label{{{spec.label}}}",
            r"\begin{algorithmic}[1]",
        ]
        for r in spec.require:
            lines.append(rf"\Require {r}")
        for e in spec.ensure:
            lines.append(rf"\Ensure {e}")
        lines += _body_to_latex(spec.body)
        lines += [r"\end{algorithmic}", r"\end{algorithm}"]
        if spec.note:
            lines.append(rf"% Note: {spec.note}")
        return "\n".join(lines) + "\n"

    # ---- plain text / markdown ----
    _KEYWORDS = {
        "If": "if", "uIf": "if", "ElseIf": "else if", "uElseIf": "else if",
        "Else": "else", "EndIf": "end if", "For": "for", "EndFor": "end for",
        "While": "while", "EndWhile": "end while", "ForEach": "for each",
        "EndForEach": "end for", "Return": "return", "KwTo": "to", "State": "",
    }
    _SYMBOLS = {
        r"\leftarrow": "<-", r"\rightarrow": "->", r"\Rightarrow": "=>",
        r"\le": "<=", r"\ge": ">=", r"\neq": "!=", r"\times": "x",
        r"\infty": "inf", r"\epsilon": "eps", r"\sigma": "sigma",
        r"\Delta": "Delta", r"\varepsilon": "eps", r"\varsigma": "sigma",
        r"\mu": "mu", r"\tau": "tau", r"\odot": "*", r"\forall": "for all",
        r"\in": "in", r"\cup": "U", r"\setminus": "\\", r"\sim": "~",
        r"\min": "min", r"\max": "max", r"\arg\max": "argmax",
        r"\ln": "ln", r"\exp": "exp", r"\lfloor": "floor(", r"\rfloor": ")",
        r"\lVert": "||", r"\rVert": "||", r"\big|": "|", r"\big\|": "||",
        r"\displaystyle": "", r"\int": "integral", r"\mathrm{d}": "d",
        r"\overline": "mean ", r"\star": "*", r"\prime": "'",
        r"\ell": "lo", r"\log": "log", r"\sum": "sum", r"\cdot": ".",
        r"\{": "{", r"\}": "}", r"\|": "||",
        r"\,": " ", r"\;": " ", r"\!": "", r"\quad": "  ",
    }

    def strip_tex(s: str) -> str:
        s = re.sub(r"\\Comment\{(.*?)\}", r"   // \1", s)
        # keywords first, before generic command stripping
        s = re.sub(r"\\(" + "|".join(_KEYWORDS) + r")\b",
                   lambda m: (_KEYWORDS[m.group(1)] + " "
                              if _KEYWORDS[m.group(1)] else ""), s)
        # fractions -> (a)/(b); subscript/superscript braces -> _x / ^x
        s = re.sub(r"\\d?frac\{(.*?)\}\{(.*?)\}", r"(\1)/(\2)", s)
        s = re.sub(r"\\(?:text|mathrm|mathcal|mathbf|textsc|textbf|emph|texttt|"
                   r"boldsymbol|mathbb)\{(.*?)\}", r"\1", s)
        s = re.sub(r"\\(?:text|mathrm|mathcal|mathbf|textsc|textbf|emph|texttt|"
                   r"boldsymbol|mathbb)\{(.*?)\}", r"\1", s)   # one nested level
        for tex, plain in _SYMBOLS.items():
            s = s.replace(tex, plain)
        s = re.sub(r"_\{(.*?)\}", r"_\1", s)
        s = re.sub(r"\^\{(.*?)\}", r"^\1", s)
        s = s.replace("$", "")
        s = re.sub(r"\\[a-zA-Z]+", "", s)          # any remaining commands
        s = s.replace("{", "").replace("}", "").replace("~", " ")
        return re.sub(r"\s+", " ", s).strip()

    lines = [f"Algorithm {spec.number}: {spec.name}",
             f"Implemented by: {spec.implemented_by}", ""]
    for r in spec.require:
        lines.append(f"Require: {strip_tex(r)}")
    for e in spec.ensure:
        lines.append(f"Ensure:  {strip_tex(e)}")
    lines.append("")
    n = 0
    for raw in spec.body:
        depth = 0
        line = raw
        while line.startswith(">"):
            depth += 1
            line = line[1:].lstrip()
        n += 1
        lines.append(f"{n:3d}: " + "  " * depth + strip_tex(line))
    if spec.note:
        lines += ["", f"Note: {spec.note}"]
    text = "\n".join(lines) + "\n"
    if fmt == "md":
        return f"### Algorithm {spec.number}: {spec.name}\n\n```\n{text}```\n"
    return text


def write_all_algorithms(out_dir: str | Path,
                         formats: Sequence[str] = ("tex", "md", "txt")
                         ) -> dict[str, list[Path]]:
    """Write every algorithm in every format, plus combined documents."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[Path]] = {f: [] for f in formats}

    for spec in ALGORITHM_SPECS:
        stem = f"algorithm_{spec.number:02d}_" + \
               re.sub(r"[^a-z0-9]+", "_", spec.name.lower()).strip("_")[:48]
        for fmt in formats:
            p = d / f"{stem}.{fmt}"
            p.write_text(render_algorithm(spec, fmt), encoding="utf-8")
            written[fmt].append(p)

    for fmt in formats:
        combined = d / f"all_algorithms.{fmt}"
        sep = "\n\n" if fmt != "md" else "\n\n---\n\n"
        combined.write_text(
            sep.join(render_algorithm(s, fmt) for s in ALGORITHM_SPECS),
            encoding="utf-8")
        written[fmt].append(combined)
    return written
