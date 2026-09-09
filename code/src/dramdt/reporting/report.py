"""Assembly of the final technical report.

:class:`ReportBuilder` collects sections, figures, tables and references and
emits a Markdown report plus a LaTeX skeleton.  Figures and tables are
referenced by the files the pipeline actually produced, so a section can never
cite an artefact that does not exist -- :meth:`ReportBuilder.validate` reports
any dangling reference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..logging_utils import get_logger
from ..paths import PROJECT_ROOT

__all__ = ["ReportBuilder", "Section", "Reference"]

log = get_logger(__name__)


@dataclass
class Reference:
    key: str
    text: str
    year: int | None = None
    url: str = ""

    def render_md(self, n: int) -> str:
        u = f" <{self.url}>" if self.url else ""
        return f"[{n}] {self.text}{u}"


@dataclass
class Section:
    title: str
    body: str = ""
    level: int = 2
    figures: list[tuple[str, str]] = field(default_factory=list)   # (path, caption)
    tables: list[tuple[str, str]] = field(default_factory=list)    # (path, caption)
    subsections: list["Section"] = field(default_factory=list)


class ReportBuilder:
    """Incremental report assembly."""

    def __init__(self, title: str, subtitle: str = "", config: Mapping[str, Any] | None = None):
        self.title = title
        self.subtitle = subtitle
        self.config = dict(config or {})
        self.sections: list[Section] = []
        self.references: list[Reference] = []
        self.metadata: dict[str, Any] = {
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._fig_counter = 0
        self._tab_counter = 0

    # ------------------------------------------------------------------
    def add_section(self, title: str, body: str = "", level: int = 2,
                    figures: Sequence[tuple[str, str]] = (),
                    tables: Sequence[tuple[str, str]] = ()) -> Section:
        s = Section(title=title, body=body.strip(), level=level,
                    figures=list(figures), tables=list(tables))
        self.sections.append(s)
        return s

    def add_reference(self, key: str, text: str, year: int | None = None,
                      url: str = "") -> None:
        if any(r.key == key for r in self.references):
            return
        self.references.append(Reference(key=key, text=text, year=year, url=url))

    def add_references(self, refs: Iterable[Mapping[str, Any]]) -> None:
        for r in refs:
            self.add_reference(str(r.get("key", "")), str(r.get("text", "")),
                               r.get("year"), str(r.get("url", "")))

    # ------------------------------------------------------------------
    def validate(self) -> list[str]:
        """Return a list of referenced artefacts that are missing on disk."""
        missing: list[str] = []
        for s in self._walk():
            for p, _c in list(s.figures) + list(s.tables):
                path = Path(p)
                if not path.is_absolute():
                    path = PROJECT_ROOT / p
                if not path.exists():
                    missing.append(str(p))
        return missing

    def _walk(self) -> Iterable[Section]:
        def rec(sections: Sequence[Section]):
            for s in sections:
                yield s
                yield from rec(s.subsections)
        return rec(self.sections)

    def _rel(self, p: str, base: Path) -> str:
        path = Path(p)
        if not path.is_absolute():
            path = PROJECT_ROOT / p
        try:
            return str(path.relative_to(base.parent)).replace("\\", "/")
        except ValueError:
            try:
                return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            except ValueError:
                return str(path).replace("\\", "/")

    # ------------------------------------------------------------------
    def render_markdown(self, out_path: str | Path) -> Path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self._fig_counter = self._tab_counter = 0

        lines: list[str] = [f"# {self.title}", ""]
        if self.subtitle:
            lines += [f"*{self.subtitle}*", ""]
        lines += [f"_Generated {self.metadata['generated_utc']}_", ""]

        # table of contents
        lines += ["## Contents", ""]
        for s in self.sections:
            anchor = re.sub(r"[^a-z0-9\- ]", "", s.title.lower()).replace(" ", "-")
            lines.append(f"- [{s.title}](#{anchor})")
        lines.append("")

        for s in self.sections:
            lines.extend(self._render_section_md(s, out))
        if self.references:
            lines += ["", "## References", ""]
            for i, r in enumerate(sorted(self.references,
                                         key=lambda x: (-(x.year or 0), x.text)), 1):
                lines.append(r.render_md(i))
                lines.append("")

        out.write_text("\n".join(lines), encoding="utf-8")
        log.info("Report written -> %s (%d sections, %d references)",
                 out.name, len(self.sections), len(self.references))
        return out

    def _render_section_md(self, s: Section, out: Path) -> list[str]:
        lines = ["", "#" * s.level + f" {s.title}", ""]
        if s.body:
            lines += [s.body, ""]
        for p, caption in s.figures:
            self._fig_counter += 1
            rel = self._rel(p, out)
            lines += [f"![Figure {self._fig_counter}]({rel})", "",
                      f"**Figure {self._fig_counter}.** {caption}", ""]
        for p, caption in s.tables:
            self._tab_counter += 1
            path = Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p
            lines += [f"**Table {self._tab_counter}.** {caption}", ""]
            md = path.with_suffix(".md")
            if md.exists():
                content = md.read_text(encoding="utf-8")
                content = "\n".join(l for l in content.splitlines()
                                    if not l.startswith("**"))
                lines += [content, ""]
            else:
                lines += [f"_(see `{self._rel(str(path), out)}`)_", ""]
        for sub in s.subsections:
            lines.extend(self._render_section_md(sub, out))
        return lines

    # ------------------------------------------------------------------
    def render_latex(self, out_path: str | Path) -> Path:
        """Emit an IEEE-style LaTeX skeleton that includes the generated assets."""
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        def esc(t: str) -> str:
            for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
                t = t.replace(a, b)
            return t

        lines = [
            r"% Generated by dramdt -- IEEE-style skeleton.",
            r"% Compile with pdflatex; requires IEEEtran, graphicx, booktabs,",
            r"% algorithm/algorithmic, amsmath.",
            r"\documentclass[conference]{IEEEtran}",
            r"\usepackage{graphicx,booktabs,amsmath,amssymb,algorithm,algorithmic}",
            r"\usepackage[caption=false]{subfig}",
            r"\begin{document}",
            rf"\title{{{esc(self.title)}}}",
            r"\maketitle",
            "",
        ]
        if self.subtitle:
            lines += [r"\begin{abstract}", esc(self.subtitle), r"\end{abstract}", ""]

        for s in self.sections:
            lines.append(rf"\section{{{esc(s.title)}}}")
            if s.body:
                body = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s.body)
                body = re.sub(r"`(.+?)`", r"\\texttt{\1}", body)
                lines += [esc(body), ""]
            for p, caption in s.figures:
                stem = str(Path(p).with_suffix("")).replace("\\", "/")
                lines += [
                    r"\begin{figure}[!t]", r"\centering",
                    rf"\includegraphics[width=\columnwidth]{{{stem}}}",
                    rf"\caption{{{esc(caption)}}}",
                    r"\end{figure}", "",
                ]
            for p, caption in s.tables:
                tex = Path(p).with_suffix(".tex")
                rel = str(tex).replace("\\", "/")
                lines += [rf"\input{{{rel}}}", ""]

        if self.references:
            lines += [r"\begin{thebibliography}{99}"]
            for r in sorted(self.references, key=lambda x: (-(x.year or 0), x.text)):
                lines.append(rf"\bibitem{{{r.key}}} {esc(r.text)}")
            lines += [r"\end{thebibliography}"]
        lines += [r"\end{document}", ""]

        out.write_text("\n".join(lines), encoding="utf-8")
        log.info("LaTeX skeleton written -> %s", out.name)
        return out
