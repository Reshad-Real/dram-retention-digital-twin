# Manuscript

**Quasi-Static Retention Modeling from SPICE-Measured Storage-Node Leakage:
Digital-Twin Simulation for Large-Scale DRAM Design-Space Exploration**

## Contents

| File | What it is |
|---|---|
| `main.tex` | Manuscript source. Compiles standalone, no `.bib` and no bibtex step. |
| `bibliography.tex` | The 65 `\bibitem` entries (Vancouver, with DOI links). |
| `figures/` | The 11 manuscript figures (PNG, 300 dpi). |
| `main.pdf` | Compiled LaTeX output (24 pages). |
| `manuscript.docx` | Word version with **native, editable equations** (Office Math). |
| `manuscript.pdf` | PDF export of the DOCX, for checking the Word rendering. |
| `build_docx.py` | Converts `main.tex` to `manuscript.docx`. |
| `overleaf_project.zip` | Upload straight to Overleaf. |

## Compiling on Overleaf

1. New Project, Upload Project, choose `overleaf_project.zip`.
2. Set the main document to `main.tex`.
3. Compiler pdfLaTeX; run it twice so cross-references resolve.

Only standard TeX Live packages are used (`amsmath`, `graphicx`, `booktabs`,
`algorithm`, `algpseudocode`, `siunitx`, `natbib`, `hyperref`, `subcaption`,
`microtype`). The bibliography is a literal `thebibliography`, so there is no
`.bib` to keep in sync.

Verified with Tectonic: 24 pages, 0 errors, 0 undefined references, 0 undefined
citations.

## Structure

8 sections, 21 numbered equations, 9 tables, 11 figures, 5 algorithms, 65
references. Citations follow Elsevier Vancouver style; every reference carries a
resolvable DOI. All 65 DOIs were verified against Crossref, and author lists,
journals, years, volumes and pages come from Crossref metadata.

| Section | |
|---|---|
| 1 | Introduction (motivation, timeliness, contributions in bullets) |
| 2 | Literature Review (2021 to 2026, with a gap-analysis table) |
| 3 | Proposed Framework and Methodology |
| 4 | Quasi-Static Retention Modelling |
| 5 | Experimental Setup |
| 6 | Results and Analysis |
| 7 | Discussion (superiority over prior work, and limitations) |
| 8 | Conclusion (no citations, by design) |

### Figures (consolidated, multi-panel, large font)

1. Framework overview.
2. Transistor-level 1T1C circuit schematic (a) and leakage replica (b), drawn in
   CircuiTikz with American source symbols (sources in `tikz/`).
3. Characterization waveforms: (a) control signals, (b) storage node, (c)
   bitlines.
4. Retention: (a) measured leakage, (b) Arrhenius, (c) integrand, (d) model vs
   transient, (e) error distribution.
5. Dataset: (a) coverage, (b) distributions, (c) correlation.
6. Parity: (a) to (f), one per response.
7. Digital twin: (a) model ranking, (b) selected twin, (c) cost.
8. Explainability: (a) SHAP, (b) attribution heatmap, (c) PDP vs ALE.
9. Optimization: (a) trade-off, (b) convergence, (c) hypervolume, (d)
   critical-difference diagram.
10. Robustness: (a) corner sweep, (b) Monte-Carlo, (c) Sobol'.
11. Assist techniques.

Every multi-panel figure labels each subplot as "(a) Name" and each subplot is
explained in the text.

## Equations in the DOCX

`manuscript.docx` contains **native Word (Office Math) equations**, not LaTeX
text. Display equations are centred with the equation number right-aligned, and
their numbering matches `main.pdf` exactly. The conversion is done by pandoc:
each LaTeX math snippet is converted once to Office Math (OMML) and inserted as a
real, editable equation object. Algorithms are rendered as framed algorithm
blocks with numbered steps and native math.

Typographic notes: the manuscript uses American English throughout, only ordinary
hyphens (no en-dashes or em-dashes), and colon-terminated run-in headings
("Name:" rather than "Name.").

## Regenerating

- **Circuit figures:** `python scripts/build_circuits.py` compiles the CircuiTikz
  sources in `MANUSCRIPT/tikz/` (via Tectonic) and exports the PNGs.
- **Other figures:** `python scripts/13_manuscript_figures.py`.
- **DOCX:** `python build_docx.py`. This requires **pandoc** on the PATH (used to
  produce native equations) and reads `main.aux`, so `main.tex` must be compiled
  first and recompiled after any edit. Numbering is taken from `main.aux`, so the
  DOCX cannot drift out of step with the PDF.

## Provenance

Every number quoted was extracted programmatically from the run artefacts in
`results/`, `data/` and `logs/`; none was transcribed by hand. Where the
framework's limits show (surrogate error at the Pareto frontier, the unreachable
64 ms retention budget at 125 C, and the disagreement between the two post-hoc
statistical procedures), the manuscript reports them rather than omitting them.
