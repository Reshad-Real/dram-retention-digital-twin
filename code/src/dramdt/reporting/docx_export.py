"""Render the generated Markdown report as a Word document, and then as a PDF.

The report is authored once, in Markdown, by :mod:`dramdt.reporting.report`.
This module is a *renderer*: it re-reads that file and produces
``.docx``/``.pdf`` without re-deriving any number, so the three formats cannot
disagree with each other.

What it understands
-------------------
``# / ## / ###`` headings, paragraphs with ``**bold**`` / ``*italic*`` /
``` `code` ``` / ``[text](url)`` runs, bullet and numbered lists, fenced code
blocks, horizontal rules, GitHub-style tables, and ``![alt](path)`` images with
the ``**Figure N.** caption`` line that follows them.

PDF conversion prefers Microsoft Word (via ``docx2pdf``) because it preserves
the layout exactly; if Word is unavailable it falls back to a ReportLab
renderer that reproduces the structure rather than the styling, and says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..logging_utils import get_logger

__all__ = ["markdown_to_docx", "docx_to_pdf", "MarkdownDocxRenderer"]

log = get_logger(__name__)

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_IMAGE = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<path>[^)]+)\)\s*$")
_FIGCAP = re.compile(r"^\*\*(Figure\s+\d+\.)\*\*\s*(.*)$")
_TABCAP = re.compile(r"^\*\*(Table\s+\d+\.)\*\*\s*(.*)$")
_TABLE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")
_TABLE_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_BULLET = re.compile(r"^\s*[-*+]\s+(.*)$")
_NUMBERED = re.compile(r"^\s*(\d+)\.\s+(.*)$")
_HRULE = re.compile(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$")
_FENCE = re.compile(r"^\s*```")

# inline: bold, italic, inline code, links
_INLINE = re.compile(
    r"(\*\*.+?\*\*)"          # bold
    r"|(\*[^*\n]+?\*)"        # italic
    r"|(_[^_\n]+?_)"          # italic (underscore)
    r"|(`[^`\n]+?`)"          # code
    r"|(\[[^\]]+\]\([^)]+\))" # link
)


@dataclass
class DocxReport:
    docx_path: Path | None = None
    pdf_path: Path | None = None
    n_figures: int = 0
    n_tables: int = 0
    n_headings: int = 0
    pdf_engine: str = ""
    warnings: list[str] = field(default_factory=list)


class MarkdownDocxRenderer:
    """Converts one Markdown report into a styled Word document."""

    def __init__(self, base_dir: Path, title: str | None = None,
                 max_image_width_in: float = 6.3):
        self.base_dir = Path(base_dir)
        self.title = title
        self.max_image_width_in = float(max_image_width_in)
        self.report = DocxReport()

    # ------------------------------------------------------------------
    def _styles(self, doc) -> None:
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Pt, RGBColor

        normal = doc.styles["Normal"]
        normal.font.name = "Calibri"
        normal.font.size = Pt(10.5)
        normal.paragraph_format.space_after = Pt(6)
        normal.paragraph_format.line_spacing = 1.15

        for name, size, colour in (("Heading 1", 18, "1F3864"),
                                   ("Heading 2", 14, "1F4E79"),
                                   ("Heading 3", 12, "2E5C8A")):
            try:
                st = doc.styles[name]
                st.font.name = "Calibri Light"
                st.font.size = Pt(size)
                st.font.bold = True
                st.font.color.rgb = RGBColor.from_string(colour)
                st.paragraph_format.space_before = Pt(12)
                st.paragraph_format.space_after = Pt(4)
            except KeyError:                                  # pragma: no cover
                pass

        try:
            cap = doc.styles.add_style("FigureCaption", 1)     # WD_STYLE_TYPE.PARAGRAPH
            cap.font.name = "Calibri"
            cap.font.size = Pt(9)
            cap.font.italic = True
            cap.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.paragraph_format.space_after = Pt(10)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _add_runs(self, paragraph, text: str) -> None:
        """Add inline-formatted runs to a paragraph."""
        pos = 0
        for m in _INLINE.finditer(text):
            if m.start() > pos:
                paragraph.add_run(text[pos:m.start()])
            tok = m.group(0)
            if tok.startswith("**"):
                paragraph.add_run(tok[2:-2]).bold = True
            elif tok.startswith("`"):
                r = paragraph.add_run(tok[1:-1])
                r.font.name = "Consolas"
                from docx.shared import Pt
                r.font.size = Pt(9.5)
            elif tok.startswith("["):
                label, _url = re.match(r"\[([^\]]+)\]\(([^)]+)\)", tok).groups()
                r = paragraph.add_run(label)
                r.underline = True
            else:                                   # *italic* or _italic_
                paragraph.add_run(tok[1:-1]).italic = True
            pos = m.end()
        if pos < len(text):
            paragraph.add_run(text[pos:])

    # ------------------------------------------------------------------
    def _resolve(self, rel: str) -> Path | None:
        p = Path(rel)
        for cand in (p, self.base_dir / p, self.base_dir.parent / p):
            if cand.exists():
                return cand
        return None

    def _add_image(self, doc, rel_path: str, alt: str) -> None:
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Inches

        src = self._resolve(rel_path)
        if src is None:
            self.report.warnings.append(f"image not found: {rel_path}")
            p = doc.add_paragraph()
            self._add_runs(p, f"[missing figure: {rel_path}]")
            return

        # Scale to the text column while preserving aspect ratio.
        width = Inches(self.max_image_width_in)
        try:
            from PIL import Image
            with Image.open(src) as im:
                w_px, h_px = im.size
            # Very tall figures are height-limited so one figure never spills
            # across two pages.
            aspect = h_px / max(w_px, 1)
            if aspect * self.max_image_width_in > 8.2:
                width = Inches(8.2 / aspect)
        except Exception:
            pass

        para = doc.add_paragraph()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.add_run().add_picture(str(src), width=width)
        self.report.n_figures += 1

    # ------------------------------------------------------------------
    def _add_table(self, doc, rows: list[list[str]]) -> None:
        from docx.shared import Pt

        if not rows:
            return
        n_cols = max(len(r) for r in rows)
        table = doc.add_table(rows=len(rows), cols=n_cols)
        try:
            table.style = "Light Grid Accent 1"
        except Exception:
            table.style = "Table Grid"
        table.autofit = True

        for i, row in enumerate(rows):
            for j in range(n_cols):
                cell = table.cell(i, j)
                cell.text = ""
                para = cell.paragraphs[0]
                para.paragraph_format.space_after = Pt(0)
                text = row[j] if j < len(row) else ""
                self._add_runs(para, text)
                for run in para.runs:
                    run.font.size = Pt(8.5)
                    if i == 0:
                        run.bold = True
        doc.add_paragraph()
        self.report.n_tables += 1

    # ------------------------------------------------------------------
    def render(self, markdown_path: str | Path, out_path: str | Path) -> DocxReport:
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.shared import Pt, RGBColor

        md_path = Path(markdown_path)
        lines = md_path.read_text(encoding="utf-8").splitlines()

        doc = Document()
        for section in doc.sections:
            from docx.shared import Inches
            section.left_margin = section.right_margin = Inches(1.0)
            section.top_margin = section.bottom_margin = Inches(0.9)
        self._styles(doc)

        i = 0
        in_code = False
        code_buf: list[str] = []
        pending_table: list[list[str]] = []

        def flush_table() -> None:
            nonlocal pending_table
            if pending_table:
                self._add_table(doc, pending_table)
                pending_table = []

        while i < len(lines):
            raw = lines[i]
            line = raw.rstrip()

            # ---- fenced code -------------------------------------------
            if _FENCE.match(line):
                if in_code:
                    para = doc.add_paragraph()
                    run = para.add_run("\n".join(code_buf))
                    run.font.name = "Consolas"
                    run.font.size = Pt(8.5)
                    para.paragraph_format.left_indent = Pt(14)
                    para.paragraph_format.space_after = Pt(8)
                    code_buf = []
                in_code = not in_code
                i += 1
                continue
            if in_code:
                code_buf.append(raw)
                i += 1
                continue

            # ---- tables -------------------------------------------------
            if _TABLE_ROW.match(line):
                if _TABLE_SEP.match(line):
                    i += 1
                    continue
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                pending_table.append(cells)
                i += 1
                continue
            flush_table()

            if not line.strip():
                i += 1
                continue

            # ---- headings ------------------------------------------------
            m = _HEADING.match(line)
            if m:
                level, text = len(m.group(1)), m.group(2).strip()
                if level == 1 and self.report.n_headings == 0:
                    h = doc.add_heading(text, 0)
                    for run in h.runs:
                        run.font.color.rgb = RGBColor.from_string("1F3864")
                else:
                    doc.add_heading(text, min(level, 4))
                self.report.n_headings += 1
                i += 1
                continue

            # ---- horizontal rule ----------------------------------------
            if _HRULE.match(line):
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(2)
                p.add_run("_" * 78).font.color.rgb = RGBColor.from_string("BFBFBF")
                i += 1
                continue

            # ---- image + its caption -------------------------------------
            m = _IMAGE.match(line.strip())
            if m:
                self._add_image(doc, m.group("path"), m.group("alt"))
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    cm = _FIGCAP.match(lines[j].strip())
                    if cm:
                        para = doc.add_paragraph()
                        try:
                            para.style = doc.styles["FigureCaption"]
                        except Exception:
                            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        run = para.add_run(f"{cm.group(1)} ")
                        run.bold = True
                        run.font.size = Pt(9)
                        self._add_runs(para, cm.group(2))
                        for r in para.runs[1:]:
                            r.font.size = Pt(9)
                            r.italic = True
                        i = j + 1
                        continue
                i += 1
                continue

            # ---- table caption -------------------------------------------
            cm = _TABCAP.match(line.strip())
            if cm:
                para = doc.add_paragraph()
                run = para.add_run(f"{cm.group(1)} ")
                run.bold = True
                run.font.size = Pt(9)
                self._add_runs(para, cm.group(2))
                for r in para.runs[1:]:
                    r.font.size = Pt(9)
                    r.italic = True
                i += 1
                continue

            # ---- lists ---------------------------------------------------
            m = _BULLET.match(line)
            if m:
                para = doc.add_paragraph(style="List Bullet")
                self._add_runs(para, m.group(1))
                i += 1
                continue
            m = _NUMBERED.match(line)
            if m:
                para = doc.add_paragraph(style="List Number")
                self._add_runs(para, m.group(2))
                i += 1
                continue

            # ---- paragraph (join soft-wrapped lines) ----------------------
            buf = [line.strip()]
            j = i + 1
            while j < len(lines):
                nxt = lines[j].rstrip()
                if (not nxt.strip() or _HEADING.match(nxt) or _IMAGE.match(nxt.strip())
                        or _TABLE_ROW.match(nxt) or _BULLET.match(nxt)
                        or _NUMBERED.match(nxt) or _HRULE.match(nxt)
                        or _FENCE.match(nxt) or _FIGCAP.match(nxt.strip())
                        or _TABCAP.match(nxt.strip())):
                    break
                buf.append(nxt.strip())
                j += 1
            para = doc.add_paragraph()
            self._add_runs(para, " ".join(buf))
            i = j

        flush_table()

        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        doc.save(out)
        self.report.docx_path = out
        log.info("DOCX written -> %s (%d headings, %d figures, %d tables)",
                 out.name, self.report.n_headings, self.report.n_figures,
                 self.report.n_tables)
        return self.report


# --------------------------------------------------------------------------
def markdown_to_docx(markdown_path: str | Path, out_path: str | Path,
                     base_dir: str | Path | None = None) -> DocxReport:
    md = Path(markdown_path)
    renderer = MarkdownDocxRenderer(base_dir or md.parent)
    return renderer.render(md, out_path)


def docx_to_pdf(docx_path: str | Path, pdf_path: str | Path | None = None
                ) -> tuple[Path | None, str]:
    """Convert a DOCX to PDF.  Returns ``(path, engine)``; path is None on failure."""
    src = Path(docx_path)
    dst = Path(pdf_path) if pdf_path else src.with_suffix(".pdf")
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        import docx2pdf
        docx2pdf.convert(str(src), str(dst))
        if dst.exists() and dst.stat().st_size > 0:
            log.info("PDF written via Microsoft Word -> %s (%.1f MB)",
                     dst.name, dst.stat().st_size / 1024 / 1024)
            return dst, "Microsoft Word (docx2pdf)"
    except Exception as exc:
        log.warning("Word conversion unavailable (%s); falling back to ReportLab", exc)

    ok = _reportlab_pdf(src, dst)
    if ok:
        log.info("PDF written via ReportLab -> %s", dst.name)
        return dst, "ReportLab (structure preserved, Word styling not applied)"
    return None, "unavailable"


def _reportlab_pdf(docx_src: Path, dst: Path) -> bool:
    """Structure-preserving PDF fallback when Word is not available."""
    try:
        from docx import Document
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (Image, PageBreak, Paragraph,
                                        SimpleDocTemplate, Spacer, Table,
                                        TableStyle)
        from reportlab.lib import colors
    except Exception as exc:                                   # pragma: no cover
        log.error("ReportLab fallback unavailable: %s", exc)
        return False

    doc_in = Document(str(docx_src))
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.5,
                          leading=13, spaceAfter=5)
    out = SimpleDocTemplate(str(dst), pagesize=A4,
                            leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.7 * inch)
    flow: list[Any] = []

    def esc(t: str) -> str:
        return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    for para in doc_in.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style_name = (para.style.name or "").lower()
        if style_name.startswith("title"):
            flow.append(Paragraph(esc(text), styles["Title"]))
        elif style_name.startswith("heading"):
            lvl = "".join(ch for ch in style_name if ch.isdigit()) or "1"
            key = f"Heading{min(int(lvl), 4)}"
            flow.append(Spacer(1, 6))
            flow.append(Paragraph(esc(text), styles.get(key, styles["Heading2"])))
        else:
            flow.append(Paragraph(esc(text), body))

    for table in doc_in.tables:
        data = [[esc(c.text)[:70] for c in row.cells] for row in table.rows]
        if not data:
            continue
        t = Table(data, repeatRows=1)
        t.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 6.5),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DDE7F0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        flow.append(Spacer(1, 6))
        flow.append(t)

    out.build(flow)
    return dst.exists()
