# Deliverables package

**Quasi-Static Retention Modeling from SPICE-Measured Storage-Node Leakage: Digital-Twin Simulation for Large-Scale DRAM Design-Space Exploration**

> This is the artifact inventory for the `01_`–`08_` folders. See the top-level `README.md` for the repository overview and the `code/` framework.

Generated 2026-07-23T13:47:54+00:00 · 531 files · 156.9 MB

Run tag `main` · configuration hash `e22061fdb47e16e7` · master seed `20260723`

---

## Contents

| Folder | What is in it |
|---|---|
| `01_dataset/` | The published dataset: Excel workbook (README, data dictionary, full table, design space, summary statistics, provenance, retention validation), a gzipped CSV mirror, and the Parquet files for programmatic use. |
| `02_report/` | The technical report as **PDF**, **Word**, Markdown and an IEEE LaTeX skeleton, plus `algorithms/` with the six core algorithms as IEEE-style pseudocode in three formats. |
| `03_figures/` | Every figure as PNG (300 dpi), PDF and SVG. |
| `04_tables/` | Every table as CSV, LaTeX (`booktabs`) and Markdown. |
| `05_results/` | The underlying result data: optimisation fronts and indicators, surrogate leaderboards and per-fold scores, SHAP/PDP/ALE outputs, robustness and yield data, statistical test outputs. |
| `06_models/` | The fitted digital twin and the surrogate archive. |
| `07_reproducibility/` | Configurations, per-stage run manifests, the literature database, the environment lock and the README / REPRODUCIBILITY guides. |
| `CHECKSUMS.json` | SHA-256 for every file in this package. |

## Inventory

| Item | Count |
|---|---|
| dataset files | 4 |
| report files | 5 |
| algorithm files | 21 |
| figures (png) | 61 |
| figures (pdf) | 61 |
| figures (svg) | 61 |
| tables (csv) | 52 |
| tables (tex) | 50 |
| tables (md) | 50 |
| results/optimization | 6 |
| results/surrogate | 5 |
| results/explainability | 37 |
| results/robustness | 11 |
| results/statistics | 80 |
| models | 2 |
| configs | 6 |
| run manifests | 11 |
| literature | 1 |

## Headline results

- **50,000 SPICE simulations**, 100.00 % successful, (this export resumed from the checkpointed chunk files, so the recorded duration is the reload time, not the simulation time).
- **Retention model validated** against direct transient simulation on 200 designs: Pearson r(log10) = 0.99974, median error 0.0107 decades, 100 % within a factor of two.
- **Selected surrogate per response** (chosen by cross-validation from 14 model families; full leaderboard in `04_tables/`):
    - `write_efficiency` -> mlp_deep (CV R2 = 0.9931, test R2 = 0.9986)
    - `energy_per_access_fj` -> mlp_deep (CV R2 = 0.9896, test R2 = 0.9961)
    - `log10_total_power_nw` -> mlp_deep (CV R2 = 0.9736, test R2 = 0.9827)
    - `read_delay_ns` -> mlp_deep (CV R2 = 0.9711, test R2 = 0.9878)
    - `read_margin_mv` -> mlp (CV R2 = 0.9614, test R2 = 0.9897)
    - `log10_retention_time_s` -> mlp_deep (CV R2 = 0.9114, test R2 = 0.9510)
- **Digital twin**: 0.03248 ms per design at batch 10000, a 37,009x speed-up over the measured SPICE evaluation time.
- **Best optimiser** by median hypervolume: NSGA-II (2851), over 7 independent runs at an identical budget.

## Reading order

1. `02_report/technical_report_main.pdf` — the full study.
2. `01_dataset/dram_dataset_main.xlsx` — start at the README sheet, then the data dictionary.
3. `04_tables/csv/` — every number in the report, machine-readable.
4. `07_reproducibility/REPRODUCIBILITY.md` — how to re-run it.

## Notes

- The PDF was produced by Microsoft Word (docx2pdf) from the Word rendering, which is itself generated from the Markdown report, so all four formats carry identical numbers.
- Device models are the public PTM BSIM4 cards for a low-power **logic** process, not a DRAM process; absolute retention times are longer than a production part would show. Section 9 of the report states the limitations in full.
- Design points that fail to write or to sense are **retained and labelled** in the dataset, not discarded.
