"""Render main.tex to a Word document, keeping every formula in LaTeX notation.

Numbering (sections, equations, tables, figures, algorithms, citations) is taken
from LaTeX's own main.aux, so the DOCX and the PDF cannot disagree.

Usage:  python build_docx.py
"""
from __future__ import annotations

import copy
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from lxml import etree

HERE = Path(__file__).resolve().parent
TEX = HERE / "main.tex"
BIB = HERE / "bibliography.tex"
AUX = HERE / "main.aux"
OUT = HERE / "manuscript.docx"

BODY_PT, TAB_PT, MATH_PT = 10.5, 8.0, 10.0
MATH_FONT, MONO_FONT = "Cambria Math", "Consolas"

# --------------------------------------------------------------------------
# native Word equations (OMML) via pandoc
#
# Every math snippet the manuscript uses is converted, once, to Office Math
# (OMML) by pandoc, so equations render as real Word equations rather than as
# LaTeX text.  Conversion is batched: all unique snippets are written to one
# markdown file, converted in a single pandoc call, and the <m:oMath> element is
# pulled out of each paragraph in order.
# --------------------------------------------------------------------------

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
_OMATH_CACHE: dict[str, object] = {}      # processed latex -> lxml <m:oMath>
_COLLECT: set | None = None               # not-None while collecting snippets


def request_omath(latex: str):
    """Return a deep copy of the cached OMML element for ``latex`` or None.

    In collect mode the string is only recorded (returns None); after the cache
    is built it returns a fresh element ready to append to a paragraph.
    """
    key = norm_math(latex.strip())
    if not key:
        return None
    if _COLLECT is not None:
        _COLLECT.add(key)
        return None
    el = _OMATH_CACHE.get(key)
    return copy.deepcopy(el) if el is not None else None


def _pandoc_omath(snippets: list[str]) -> list:
    """Convert display-math snippets to OMML elements, in order (one per input)."""
    md = "\n".join(f"$${s}$$\n" for s in snippets)
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "m.md"
        dst = Path(td) / "m.docx"
        src.write_text(md, encoding="utf-8")
        subprocess.run(["pandoc", str(src), "-o", str(dst)], check=True,
                       capture_output=True, text=True)
        with zipfile.ZipFile(dst) as z:
            root = etree.fromstring(z.read("word/document.xml"))
    return root.findall(f".//{{{M_NS}}}oMath")


def build_math_cache(strings: set) -> None:
    keys = sorted(strings)
    if not keys:
        return
    try:
        elems = _pandoc_omath(keys)
        if len(elems) == len(keys):
            for k, e in zip(keys, elems):
                _OMATH_CACHE[k] = e
            return
        print(f"  batch OMML count mismatch ({len(elems)} vs {len(keys)}); "
              "falling back to per-snippet")
    except Exception as exc:                                  # pragma: no cover
        print(f"  batch OMML failed ({exc}); per-snippet fallback")
    # robust fallback: one pandoc call per snippet
    for k in keys:
        try:
            e = _pandoc_omath([k])
            if e:
                _OMATH_CACHE[k] = e[0]
        except Exception:
            pass


def append_omath(par, latex: str, math_pt: float = MATH_PT) -> bool:
    """Append a native OMML equation to a paragraph. Return True on success."""
    el = request_omath(latex)
    if el is None:
        return _COLLECT is not None      # in collect mode, treat as handled
    # normalise run sizes inside the equation to the requested point size
    sz = str(int(math_pt * 2))
    for r in el.findall(f".//{{{M_NS}}}r"):
        rpr = r.find(f"{{{M_NS}}}rPr")
        # also set a w:rPr size so Word honours the font size
        w_rpr = r.find(qn("w:rPr"))
        if w_rpr is None:
            w_rpr = OxmlElement("w:rPr")
            r.insert(0, w_rpr)
        for tag in ("w:sz", "w:szCs"):
            e = w_rpr.find(qn(tag))
            if e is None:
                e = OxmlElement(tag)
                w_rpr.append(e)
            e.set(qn("w:val"), sz)
    par._p.append(el)
    return True

# --------------------------------------------------------------------------
# numbering harvested from main.aux
# --------------------------------------------------------------------------


def load_aux() -> tuple[dict[str, str], dict[str, str]]:
    txt = AUX.read_text(encoding="utf-8", errors="replace")
    labels = {}
    for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^{}]*)\}\{", txt):
        labels[m.group(1)] = m.group(2)
    cites = {m.group(1): m.group(2)
             for m in re.finditer(r"\\bibcite\{([^}]+)\}\{\{(\d+)\}", txt)}
    return labels, cites


LABELS, CITES = load_aux()

# --------------------------------------------------------------------------
# LaTeX -> portable LaTeX (expand this paper's private macros)
# --------------------------------------------------------------------------

MACROS = {
    r"\Vsn": r"V_{\mathrm{SN}}",
    r"\Vfail": r"V_{\mathrm{fail}}",
    r"\Vblpre": r"V_{\mathrm{BLpre}}",
    r"\Ileak": r"I_{\mathrm{leak}}",
    r"\Cnode": r"C_{\mathrm{node}}",
    r"\Cbl": r"C_{\mathrm{BL}}",
    r"\Cs": r"C_{s}",
    r"\tret": r"t_{\mathrm{ret}}",
    r"\dVbl": r"\Delta V_{\mathrm{BL}}",
    r"\dVmin": r"\Delta V_{\min}",
}


def expand_macros(s: str) -> str:
    # longest first so \Vblpre is not eaten by \Vbl-like prefixes; pad the
    # replacement with spaces so an adjacent control word (e.g. \kappa\Cs) does
    # not fuse into an undefined one (\kappaC_{s}) after expansion
    for name in sorted(MACROS, key=len, reverse=True):
        rep = " " + MACROS[name] + " "
        s = re.sub(re.escape(name) + r"(?![A-Za-z])", lambda _m, r=rep: r, s)
    return s


def norm_math(s: str) -> str:
    """Rewrite math that pandoc's LaTeX reader rejects into supported forms."""
    # \textsc{...} (small caps) is not valid in math mode; render upright
    s = re.sub(r"\\textsc\{([^{}]*)\}", r"\\mathrm{\1}", s)
    return s


# --------------------------------------------------------------------------
# siunitx
# --------------------------------------------------------------------------

UNITS = {
    r"\volt": "V", r"\milli\volt": "mV", r"\micro\volt": "uV",
    r"\ampere": "A", r"\femto\ampere": "fA", r"\micro\ampere": "uA",
    r"\second": "s", r"\milli\second": "ms", r"\micro\second": "\u00b5s",
    r"\nano\second": "ns", r"\pico\second": "ps",
    r"\hertz": "Hz", r"\kilo\hertz": "kHz",
    r"\celsius": "\u00b0C", r"\electronvolt": "eV",
    r"\siemens": "S", r"\ohm": "\u03a9",
    r"\farad": "F", r"\femto\farad": "fF",
    r"\metre": "m", r"\nano\metre": "nm", r"\percent": "%",
    r"\watt": "W", r"\nano\watt": "nW", r"\joule": "J", r"\femto\joule": "fJ",
}


def unit_text(u: str) -> str:
    # collapse whitespace so a line-wrapped unit (\micro\n\second) still matches
    u = re.sub(r"\s+", "", u.strip())
    if u in UNITS:
        return UNITS[u]
    out = u
    for k in sorted(UNITS, key=len, reverse=True):
        out = out.replace(k, UNITS[k])
    return out.replace("\\", "")


def braced(s: str, i: int) -> tuple[str, int]:
    """Read a {...} group starting at s[i] == '{'; return (inner, index after)."""
    assert s[i] == "{"
    depth, j = 0, i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:j], j + 1
        j += 1
    return s[i + 1:], len(s)


def apply_si(s: str) -> str:
    out, i = [], 0
    while i < len(s):
        if s.startswith(r"\SI{", i):
            val, j = braced(s, i + 3)
            if j < len(s) and s[j] == "{":
                unit, j = braced(s, j)
                out.append(f"{val.strip()}\u2009{unit_text(unit)}")
                i = j
                continue
        if s.startswith(r"\si{", i):
            unit, j = braced(s, i + 3)
            out.append(unit_text(unit))
            i = j
            continue
        out.append(s[i])
        i += 1
    return "".join(out)


# --------------------------------------------------------------------------
# text-mode cleanup
# --------------------------------------------------------------------------

SIMPLE = [
    (r"\\textendash\b", "\u2013"),
    (r"\\ldots\b", "\u2026"), (r"\\dots\b", "\u2026"),
    (r"\\times\b", "\u00d7"), (r"\\approx\b", "\u2248"),
    (r"\\alpha\b", "\u03b1"), (r"\\beta\b", "\u03b2"),
    (r"\\gamma\b", "\u03b3"), (r"\\rho\b", "\u03c1"),
    (r"\\kappa\b", "\u03ba"), (r"\\sigma\b", "\u03c3"),
    (r"\\delta\b", "\u03b4"), (r"\\Delta\b", "\u0394"),
    (r"\\varrho\b", "\u03f1"), (r"\\chi\b", "\u03c7"),
    (r"\\emph\{([^{}]*)\}", r"\1"),
    (r"\\relax\b", ""),
    (r"\\(?:quad|qquad|noindent|centering|footnotesize|small|hfill"
     r"|raggedright|raggedleft|bfseries|itshape|scriptsize|normalsize)\b", " "),
    (r"\\[,;:!]", "\u2009"),
    (r"\\ ", " "),
]


def detex(s: str, keep_edges: bool = False) -> str:
    """LaTeX text -> plain text (math is handled separately, before this).

    With keep_edges, a single leading/trailing space is preserved so that word
    boundaries survive when a paragraph is split across formatting runs.
    """
    if keep_edges:
        lead = " " if s[:1].isspace() else ""
        trail = " " if s[-1:].isspace() else ""
        core = detex(s)
        return (lead + core + trail) if core else (lead or trail)
    s = apply_si(s)
    for pat, rep in SIMPLE:
        s = re.sub(pat, rep, s)
    s = s.replace("---", "\u2014").replace("--", "\u2013")
    s = s.replace("``", "\u201c").replace("''", "\u201d")
    s = s.replace("{,}", ",")          # LaTeX thousands separator, text mode
    s = re.sub(r"\\%", "%", s)
    s = re.sub(r"\\&", "&", s)
    s = re.sub(r"\\_", "_", s)
    s = re.sub(r"\\#", "#", s)
    s = re.sub(r"\\\$", "$", s)
    s = s.replace(r"\{", "{").replace(r"\}", "}")
    s = s.replace("~", "\u00a0")
    s = re.sub(r"\{\}", "", s)
    # collapse newlines too: LaTeX source wraps freely, Word must not inherit
    # those wraps as hard line breaks inside captions, cells and algorithm lines
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _compress_runs(nums: list[int]) -> str:
    """[1,2,3,4] -> '1-4'; [1,3,4,5,8] -> '1,3-5,8' (matches natbib sort&compress)."""
    out, i, n = [], 0, len(nums)
    while i < n:
        j = i
        while j + 1 < n and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if j == i else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ",".join(out)


def resolve_refs(s: str) -> str:
    def cite(m):
        keys = [k.strip() for k in m.group(1).split(",")]
        nums = sorted({int(CITES[k]) for k in keys if k in CITES})
        return "[" + _compress_runs(nums) + "]"

    s = re.sub(r"\\cite\{([^}]+)\}", cite, s)
    s = re.sub(r"\\ref\{([^}]+)\}", lambda m: LABELS.get(m.group(1), "?"), s)
    s = re.sub(r"\\href\{([^}]+)\}\{[^}]*\}", r"\1", s)
    return s


# --------------------------------------------------------------------------
# rich runs: bold / italic / mono / inline math kept as LaTeX
# --------------------------------------------------------------------------

TOKEN = re.compile(
    r"\\textbf\{|\\textit\{|\\texttt\{|\\emph\{|\\textsc\{|\$")


def add_rich(par, text: str, bold: bool = False, italic: bool = False,
             mono: bool = False, _top: bool = True) -> None:
    """Emit runs, preserving $...$ verbatim as LaTeX and honouring font macros.

    Recursive, so that maths nested inside \\textbf{...} is still emitted as a
    LaTeX math run rather than being flattened to plain text.
    """
    if _top:
        text = resolve_refs(expand_macros(text))
    i = 0
    while i < len(text):
        m = TOKEN.search(text, i)
        if not m:
            _emit(par, text[i:], bold, italic, mono)
            return
        if m.start() > i:
            _emit(par, text[i:m.start()], bold, italic, mono)
        tok = m.group(0)
        if tok == "$":
            end = text.find("$", m.start() + 1)
            if end == -1:
                end = len(text)
            body = text[m.start() + 1:end].strip()
            if not append_omath(par, body):
                # graceful fallback: LaTeX text set in the math font
                r = par.add_run(f"${body}$")
                r.font.name = MATH_FONT
                r.font.size = Pt(MATH_PT)
                r.bold, r.italic = bold, italic
            i = end + 1
            continue
        inner, j = braced(text, m.end() - 1)
        if tok == r"\texttt{":
            add_rich(par, inner, bold, italic, True, False)
        elif tok == r"\textbf{":
            add_rich(par, inner, True, italic, mono, False)
        elif tok == r"\textsc{":
            _emit(par, detex(inner, True).upper(), bold, italic, mono)
        else:  # \textit \emph
            add_rich(par, inner, bold, True, mono, False)
        i = j


def _emit(par, raw: str, bold: bool, italic: bool, mono: bool):
    txt = detex(raw, keep_edges=True)
    if not txt:
        return None
    r = par.add_run(txt)
    r.bold, r.italic = bold, italic
    if mono:
        r.font.name = MONO_FONT
        r.font.size = Pt(BODY_PT - 1)
    return r


# --------------------------------------------------------------------------
# document helpers
# --------------------------------------------------------------------------


def shade(cell, hexcolor="F2F2F2"):
    tcPr = cell._tc.get_or_add_tcPr()
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:fill"), hexcolor)
    tcPr.append(el)


def caption(doc, text, style_bold_prefix):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run(style_bold_prefix)
    r.bold = True
    r.font.size = Pt(BODY_PT - 1)
    add_rich(p, " " + text)
    for r in p.runs[1:]:
        r.font.size = Pt(BODY_PT - 1)
    return p


# --------------------------------------------------------------------------
# environment renderers
# --------------------------------------------------------------------------

ROW_SPLIT = re.compile(r"\\\\(?:\s*\[[^\]]*\])?")

# environments whose internal \\ and & are structural and must survive
NESTED_ENVS = ("cases", "pmatrix", "bmatrix", "vmatrix", "Vmatrix", "matrix",
               "array", "aligned", "split", "gathered", "subequations")
NESTED_RE = re.compile(
    r"\\begin\{(" + "|".join(NESTED_ENVS) + r")\}.*?\\end\{\1\}", re.S)


def mask_nested(s: str) -> tuple[str, dict[str, str]]:
    holes: dict[str, str] = {}

    def take(m):
        key = f"\x00{len(holes)}\x00"
        holes[key] = m.group(0)
        return key

    return NESTED_RE.sub(take, s), holes


def unmask(s: str, holes: dict[str, str]) -> str:
    for k, v in holes.items():
        s = s.replace(k, v)
    return s


def unwrap_boxed(s: str) -> str:
    """Drop a \\boxed{...} wrapper, brace-aware, keeping its contents."""
    while True:
        i = s.find(r"\boxed{")
        if i == -1:
            return s
        inner, j = braced(s, i + len(r"\boxed"))
        s = s[:i] + inner + s[j:]


def render_equation(doc, body: str, labels: list[str], numbered: bool) -> None:
    """Display maths, kept verbatim in LaTeX, with the LaTeX number."""
    body = expand_macros(body)
    body = re.sub(r"\\label\{[^}]*\}", "", body)
    body = re.sub(r"\\notag\b|\\nonumber\b", "", body)
    body = unwrap_boxed(body).strip()

    masked, holes = mask_nested(body)
    raw_lines = [ln for ln in ROW_SPLIT.split(masked) if ln.strip()]
    lines = []
    for ln in raw_lines:
        # '&' is alignment scaffolding of the align environment, not maths;
        # strip it so each line is valid standalone LaTeX.
        ln = ln.replace("&&", r" \quad ")
        ln = re.sub(r"(?<!\\)&", " ", ln)
        lines.append(unmask(ln, holes).strip())

    for k, line in enumerate(lines):
        line = line.rstrip("\\").strip()
        # source line-wraps are not part of the maths
        line = re.sub(r"\s+", " ", line)
        line = re.sub(r"^(?:\\[,;:! ])+|(?:\\[,;:! ])+$", "", line).strip()
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = Pt(6)
        pf.space_after = Pt(6)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        # centre the equation with a right-aligned number, journal style
        tab = p.paragraph_format.tab_stops
        tab.add_tab_stop(Inches(3.15), WD_TAB_ALIGNMENT.CENTER)
        tab.add_tab_stop(Inches(6.4), WD_TAB_ALIGNMENT.RIGHT)
        num = ""
        if numbered and k < len(labels) and labels[k]:
            num = LABELS.get(labels[k], "")
        p.add_run("\t")
        if not append_omath(p, line):
            r = p.add_run(line)
            r.font.name = MATH_FONT
            r.font.size = Pt(MATH_PT)
        if num:
            rn = p.add_run(f"\t({num})")
            rn.font.size = Pt(BODY_PT)


def skip_group(s: str, i: int) -> int:
    """Index just past a balanced {...} at s[i] (tolerates leading whitespace)."""
    while i < len(s) and s[i].isspace():
        i += 1
    if i < len(s) and s[i] == "{":
        return braced(s, i)[1]
    return i


def parse_tabular(block: str) -> list[list[tuple[str, int]]]:
    """Return rows of (cell_text, span). Rules and spacing commands are dropped.

    The column specification must be skipped brace-aware: specs such as
    ``{@{}p{0.16\\linewidth}l@{}}`` contain nested braces, so a ``[^}]*`` scan
    would stop early and leak the remainder into the first cell.
    """
    m = re.search(r"\\begin\{tabular\}", block)
    if not m:
        return []
    i = m.end()
    if i < len(block) and block[i] == "[":          # optional [pos]
        i = block.index("]", i) + 1
    i = skip_group(block, i)                        # the column spec
    end = block.find(r"\end{tabular}", i)
    body = block[i:end if end != -1 else len(block)]
    body = re.sub(r"\\(?:toprule|midrule|bottomrule|hline)\b", "", body)
    body = re.sub(r"\\addlinespace(\[[^\]]*\])?", "", body)
    body = re.sub(r"\\cmidrule(\([^)]*\))?(\[[^\]]*\])?\{[^}]*\}", "", body)
    rows = []
    for raw in ROW_SPLIT.split(body):
        if not raw.strip():
            continue
        cells = []
        for c in split_cells(raw):
            span, c = unwrap_span(c)
            cells.append((c.strip(), span))
        if any(t for t, _ in cells):
            rows.append(cells)
    return rows


def unwrap_span(c: str) -> tuple[int, str]:
    """Strip \\multicolumn / \\multirow wrappers brace-aware, returning (span, text)."""
    span = 1
    for cmd, takes_span in ((r"\multicolumn", True), (r"\multirow", False)):
        s = c.lstrip()
        if not s.startswith(cmd):
            continue
        i = s.index("{", len(cmd))
        n_txt, i = braced(s, i)                     # {N} or {nrows}
        i = skip_group(s, i)                        # {colspec} or {width}
        if i < len(s) and s[i] == "{":
            inner, _ = braced(s, i)
            if takes_span and n_txt.strip().isdigit():
                span = int(n_txt.strip())
            c = inner
    return span, c


def split_cells(row: str) -> list[str]:
    """Split on & that are not escaped and not inside braces."""
    out, depth, cur = [], 0, []
    i = 0
    while i < len(row):
        ch = row[i]
        if ch == "\\" and i + 1 < len(row) and row[i + 1] == "&":
            cur.append("\\&")
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "&" and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    out.append("".join(cur))
    return out


def render_table(doc, block: str) -> None:
    lab = re.search(r"\\label\{([^}]+)\}", block)
    cap = re.search(r"\\caption\{(.*?)\}\s*\n\s*\\label", block, re.S)
    if cap is None:
        cap = re.search(r"\\caption\{(.*)\}", block, re.S)
    num = LABELS.get(lab.group(1), "?") if lab else "?"
    caption(doc, cap.group(1) if cap else "", f"Table {num}.")

    rows = parse_tabular(block)
    if not rows:
        return
    ncol = max(sum(s for _, s in r) for r in rows)
    t = doc.add_table(rows=0, cols=ncol)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        ci = 0
        for text, span in row:
            if ci >= ncol:
                break
            target = cells[ci]
            if span > 1 and ci + span - 1 < ncol:
                target = target.merge(cells[ci + span - 1])
            target.text = ""
            p = target.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            add_rich(p, text)
            for r in p.runs:
                r.font.size = Pt(TAB_PT)
                if ri == 0:
                    r.bold = True
            if ri == 0:
                shade(target)
            ci += span
    doc.add_paragraph().paragraph_format.space_after = Pt(6)


PAGE_W_IN = 6.3


def _place_image(doc, block: str) -> None:
    img = re.search(r"\\includegraphics\[([^\]]*)\]\{([^}]+)\}", block)
    if not img:
        return
    path = HERE / img.group(2)
    if not path.exists():
        return
    sub = re.search(r"\\begin\{subfigure\}\{([\d.]+)\\linewidth\}", block)
    subfrac = float(sub.group(1)) if sub else 1.0
    mw = re.search(r"([\d.]+)\\linewidth", img.group(1))
    imgfrac = float(mw.group(1)) if mw else 1.0
    w = PAGE_W_IN * subfrac * imgfrac
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(w))


def render_figure(doc, block: str) -> None:
    subs = re.findall(r"\\begin\{subfigure\}.*?\\end\{subfigure\}", block, re.S)
    outer = re.sub(r"\\begin\{subfigure\}.*?\\end\{subfigure\}", "", block, flags=re.S)
    lab = re.search(r"\\label\{([^}]+)\}", outer)
    cap = (re.search(r"\\caption\{(.*?)\}\s*\n\s*\\label", outer, re.S)
           or re.search(r"\\caption\{(.*)\}", outer, re.S))
    num = LABELS.get(lab.group(1), "?") if lab else "?"

    if subs:
        letters = "abcdefgh"
        for k, sub in enumerate(subs):
            _place_image(doc, sub)
            sc = (re.search(r"\\caption\{(.*?)\}\s*\\label", sub, re.S)
                  or re.search(r"\\caption\{(.*)\}", sub, re.S))
            if sc:
                p = caption(doc, sc.group(1), f"({letters[k]})")
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        _place_image(doc, block)
    p = caption(doc, cap.group(1) if cap else "", f"Figure {num}.")
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT


ALGO_LINE = [
    (r"\\Require\b", "Require:", 0, "kw"),
    (r"\\Ensure\b", "Ensure:", 0, "kw"),
    (r"\\Statex\b", "", 0, "plain"),
    (r"\\State\b", "", 0, "num"),
    (r"\\ForAll\b", "for all", +1, "block"),
    (r"\\For\b", "for", +1, "block"),
    (r"\\While\b", "while", +1, "block"),
    (r"\\If\b", "if", +1, "block"),
    (r"\\ElsIf\b", "else if", 0, "block"),
    (r"\\Else\b", "else", 0, "block"),
    (r"\\EndFor\b", "end for", -1, "end"),
    (r"\\EndWhile\b", "end while", -1, "end"),
    (r"\\EndIf\b", "end if", -1, "end"),
    (r"\\Return\b", "return", 0, "num"),
]


def _table_borders(tbl, color="666666", sz="6"):
    tblPr = tbl._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), sz)
        e.set(qn("w:space"), "0")
        e.set(qn("w:color"), color)
        borders.append(e)
    tblPr.append(borders)


def _para_bottom_rule(p, color="999999", sz="6"):
    pPr = p._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), sz)
    bottom.set(qn("w:space"), "2")
    bottom.set(qn("w:color"), color)
    pbdr.append(bottom)
    pPr.append(pbdr)


def render_algorithm(doc, block: str) -> None:
    """Render as a framed algorithm block (a bordered single-cell table)."""
    lab = re.search(r"\\label\{([^}]+)\}", block)
    cap = re.search(r"\\caption\{(.*?)\}", block, re.S)
    num = LABELS.get(lab.group(1), "?") if lab else "?"

    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    _table_borders(tbl)
    box = tbl.cell(0, 0)

    p = box.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(f"Algorithm {num}.  ")
    r.bold = True
    add_rich(p, cap.group(1) if cap else "")
    _para_bottom_rule(p)

    inner = re.search(r"\\begin\{algorithmic\}(?:\[\d*\])?(.*)\\end\{algorithmic\}",
                      block, re.S)
    if not inner:
        return
    body = inner.group(1)
    # split into logical lines on the leading commands
    starts = [m.start() for m in re.finditer(
        r"\\(?:Require|Ensure|Statex|State|ForAll|For|While|If|ElsIf|Else|"
        r"EndFor|EndWhile|EndIf|Return)\b", body)]
    chunks = [body[a:b] for a, b in zip(starts, starts[1:] + [len(body)])]

    indent, counter = 0, 0
    for chunk in chunks:
        kind, word, delta = "num", "", 0
        for pat, w, d, k in ALGO_LINE:
            if re.match(r"\s*" + pat, chunk):
                kind, word, delta = k, w, d
                chunk = re.sub(r"\s*" + pat, "", chunk, count=1)
                break
        if delta < 0:
            indent = max(0, indent + delta)

        comment = ""
        mc = re.search(r"\\Comment\{(.*)\}\s*$", chunk, re.S)
        if mc:
            comment = mc.group(1)
            chunk = chunk[:mc.start()]
        # {cond} argument of block commands
        cond = ""
        chunk = chunk.strip()
        if kind == "block" and chunk.startswith("{"):
            cond, rest = braced(chunk, 0)
            chunk = chunk[rest:]
            cond = cond.strip()

        text = chunk.strip().rstrip("\\").strip()
        line = box.add_paragraph()
        pf = line.paragraph_format
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        pf.left_indent = Inches(0.30 + 0.22 * indent)
        pf.first_line_indent = Inches(-0.30)

        if kind in ("num", "block", "end") and (text or cond or word):
            counter += 1
            rn = line.add_run(f"{counter}: ")
            rn.font.size = Pt(BODY_PT - 1.5)
            rn.font.color.rgb = RGBColor(0x80, 0x80, 0x80)
        if word:
            rw = line.add_run(word + " ")
            rw.bold = True
            rw.font.size = Pt(BODY_PT - 0.5)
        if cond:
            add_rich(line, cond)
        if kind == "block" and cond:
            rt = line.add_run(" then" if word in ("if", "else if") else " do")
            rt.bold = True
            rt.font.size = Pt(BODY_PT - 0.5)
        if text:
            if cond or word:
                line.add_run(" ")
            add_rich(line, text)
        if comment:
            rc = line.add_run("   \u25b7 ")
            rc.font.color.rgb = RGBColor(0x70, 0x70, 0x70)
            start = len(line.runs)
            add_rich(line, comment)
            for r in line.runs[start:]:
                r.italic = True
                r.font.color.rgb = RGBColor(0x70, 0x70, 0x70)
        for r in line.runs:
            if r.font.size is None:
                r.font.size = Pt(BODY_PT - 0.5)
        if delta > 0:
            indent += delta
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


# --------------------------------------------------------------------------
# main walk
# --------------------------------------------------------------------------

# NB: "algorithm" is deliberately absent -- it has its own renderer below, and
# listing it here would let the generic handler consume the block silently.
ENVS = ("equation", "align", "table", "figure", "itemize",
        "enumerate", "abstract")


def strip_comments(tex: str) -> str:
    out = []
    for line in tex.split("\n"):
        m = re.match(r"^(.*?)(?<!\\)%.*$", line)
        out.append(m.group(1) if m else line)
    return "\n".join(out)


def build(emit: bool = True) -> None:
    tex = strip_comments(TEX.read_text(encoding="utf-8"))
    body = tex.split(r"\begin{document}", 1)[1].split(r"\end{document}")[0]

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Cambria"
    st.font.size = Pt(BODY_PT)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.line_spacing = 1.12
    for s in doc.sections:
        s.left_margin = s.right_margin = Inches(1.0)
        s.top_margin = s.bottom_margin = Inches(0.9)

    # ---- title -------------------------------------------------------
    mt = re.search(r"\\title\{(.*?)\n*\}\s*\n\s*\\author", tex, re.S)
    title = mt.group(1) if mt else "Manuscript"
    title = re.sub(r"\\bfseries|\\Large|\\huge", "", title)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(detex(title))
    r.bold = True
    r.font.size = Pt(17)
    p.paragraph_format.space_after = Pt(14)

    note = doc.add_paragraph()
    note.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rn = note.add_run(
        "All equations in this document are native, editable Word (Office Math) "
        "objects; numbering matches the LaTeX/PDF version exactly.")
    rn.italic = True
    rn.font.size = Pt(8.5)
    rn.font.color.rgb = RGBColor(0x60, 0x60, 0x60)
    note.paragraph_format.space_after = Pt(14)

    pos, n_fig, n_tab, n_alg, n_eq = 0, 0, 0, 0, 0
    pending_par: list[str] = []

    def flush(align=WD_ALIGN_PARAGRAPH.JUSTIFY):
        nonlocal pending_par
        text = " ".join(pending_par).strip()
        pending_par = []
        if not text:
            return
        p = doc.add_paragraph()
        p.alignment = align
        add_rich(p, text)

    lines = body.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()

        m_env = re.match(r"\\begin\{(" + "|".join(ENVS) + r")\}", s)
        m_sec = re.match(r"\\(section|subsection|subsubsection)\*?\{(.*)\}", s)

        if m_sec:
            flush()
            kind, txt = m_sec.group(1), m_sec.group(2)
            # find the label that follows, to get the LaTeX number
            num = ""
            look = "\n".join(lines[i:i + 3])
            ml = re.search(r"\\label\{([^}]+)\}", look)
            if ml and ml.group(1) in LABELS and kind == "section":
                num = LABELS[ml.group(1)] + ". "
            lvl = {"section": 1, "subsection": 2, "subsubsection": 3}[kind]
            h = doc.add_heading("", level=lvl)
            rh = h.add_run(num + detex(txt))
            rh.font.color.rgb = RGBColor(0, 0, 0)
            rh.font.name = "Cambria"
            rh.bold = True
            rh.font.size = Pt({1: 14, 2: 11.5, 3: 10.5}[lvl])
            i += 1
            continue

        if m_env:
            flush()
            env = m_env.group(1)
            depth, j = 0, i
            while j < len(lines):
                depth += len(re.findall(r"\\begin\{" + env + r"\}", lines[j]))
                depth -= len(re.findall(r"\\end\{" + env + r"\}", lines[j]))
                if depth == 0:
                    break
                j += 1
            block = "\n".join(lines[i:j + 1])

            if env == "abstract":
                inner = re.search(r"\\begin\{abstract\}(.*)\\end\{abstract\}",
                                  block, re.S).group(1)
                h = doc.add_heading("", level=1)
                rh = h.add_run("Abstract")
                rh.font.color.rgb = RGBColor(0, 0, 0)
                rh.font.name = "Cambria"
                rh.bold = True
                rh.font.size = Pt(13)
                for para in re.split(r"\n\s*\n", inner):
                    para = para.replace(r"\noindent", "").strip()
                    if para:
                        p = doc.add_paragraph()
                        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                        add_rich(p, para)
            elif env in ("equation", "align"):
                inner = re.search(
                    r"\\begin\{" + env + r"\}(.*)\\end\{" + env + r"\}",
                    block, re.S).group(1)
                labs = re.findall(r"\\label\{([^}]+)\}", inner)
                # align: one label per row -- split the same way render_equation
                # does (nested cases/matrix masked) so labels stay aligned
                if env == "align":
                    masked_inner, _ = mask_nested(inner)
                    per = []
                    for seg in ROW_SPLIT.split(masked_inner):
                        if not seg.strip():
                            continue
                        ml = re.search(r"\\label\{([^}]+)\}", seg)
                        per.append(ml.group(1) if ml else "")
                    labs = per
                render_equation(doc, inner, labs, True)
                n_eq += len(labs)
            elif env == "table":
                render_table(doc, block)
                n_tab += 1
            elif env == "figure":
                render_figure(doc, block)
                n_fig += 1
            elif env in ("itemize", "enumerate"):
                inner = re.search(
                    r"\\begin\{" + env + r"\}(.*)\\end\{" + env + r"\}",
                    block, re.S).group(1)
                style = "List Bullet" if env == "itemize" else "List Number"
                for item in inner.split(r"\item")[1:]:
                    p = doc.add_paragraph(style=style)
                    p.paragraph_format.space_after = Pt(5)
                    add_rich(p, " ".join(item.split()))
            i = j + 1
            continue

        if s.startswith(r"\begin{algorithm}"):
            flush()
            j = i
            while j < len(lines) and r"\end{algorithm}" not in lines[j]:
                j += 1
            render_algorithm(doc, "\n".join(lines[i:j + 1]))
            n_alg += 1
            i = j + 1
            continue

        if s.startswith(r"\begin{thebibliography}"):
            flush()
            break

        if s.startswith("\\") and re.match(
                r"\\(maketitle|label|captionsetup|clearpage|newpage|"
                r"bibliographystyle|input|sisetup)", s):
            i += 1
            continue

        if s.startswith(r"\noindent"):
            s = s.replace(r"\noindent", "").strip()

        if s.startswith(r"\textbf{Keywords:}"):
            flush()
            p = doc.add_paragraph()
            add_rich(p, s)
            i += 1
            continue

        if not s:
            flush()
        else:
            pending_par.append(s)
        i += 1

    flush()

    # ---- references --------------------------------------------------
    h = doc.add_heading("", level=1)
    rh = h.add_run("References")
    rh.font.color.rgb = RGBColor(0, 0, 0)
    rh.font.name = "Cambria"
    rh.bold = True
    rh.font.size = Pt(14)

    bib = BIB.read_text(encoding="utf-8")
    entries = re.split(r"\\bibitem\{([^}]+)\}", bib)[1:]
    pairs = list(zip(entries[0::2], entries[1::2]))
    pairs.sort(key=lambda kv: int(CITES.get(kv[0], "999")))
    for key, text in pairs:
        num = CITES.get(key, "?")
        url = ""
        mh = re.search(r"\\href\{([^}]+)\}", text)
        if mh:
            url = mh.group(1)
        body_txt = re.sub(r"\\href\{[^}]+\}\{[^}]*\}", "", text).strip()
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.left_indent = Inches(0.38)
        pf.first_line_indent = Inches(-0.38)
        pf.space_after = Pt(4)
        r = p.add_run(f"[{num}] ")
        r.bold = True
        r.font.size = Pt(BODY_PT - 1)
        add_rich(p, detex(body_txt))
        for r in p.runs[1:]:
            r.font.size = Pt(BODY_PT - 1)
        if url:
            p.add_run(" ").font.size = Pt(BODY_PT - 1)
            add_hyperlink(p, url, url)

    if not emit:
        return
    doc.save(OUT)
    print(f"wrote {OUT}")
    print(f"  figures {n_fig} | tables {n_tab} | algorithms {n_alg} | "
          f"numbered equations {n_eq} | references {len(pairs)} | "
          f"OMML equations cached {len(_OMATH_CACHE)}")


def add_hyperlink(par, url, text):
    part = par.part
    r_id = part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True)
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), r_id)
    new_run = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    col = OxmlElement("w:color")
    col.set(qn("w:val"), "1155CC")
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), str(int((BODY_PT - 1) * 2)))
    rPr.append(col)
    rPr.append(u)
    rPr.append(sz)
    new_run.append(rPr)
    t = OxmlElement("w:t")
    t.text = text
    new_run.append(t)
    link.append(new_run)
    par._p.append(link)
    return link


def main() -> None:
    global _COLLECT
    # pass 1: walk the document, collecting every math snippet
    _COLLECT = set()
    build(emit=False)
    snippets = _COLLECT
    _COLLECT = None
    print(f"collected {len(snippets)} unique math snippets; converting via pandoc")
    build_math_cache(snippets)
    print(f"cached {len(_OMATH_CACHE)} OMML equations")
    # pass 2: emit with native equations
    build(emit=True)


if __name__ == "__main__":
    main()
