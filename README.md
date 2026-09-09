# Quasi-Static Retention Modeling from SPICE-Measured Storage-Node Leakage

**Digital-Twin Simulation for Large-Scale DRAM Design-Space Exploration**

This repository is the complete reproducibility package for the paper above: the
released dataset, the trained digital twin, every figure and table, all analysis
outputs and run manifests, the manuscript, and the framework source code needed to
regenerate them.

> **Summary.** Retention is the DRAM response a circuit simulator cannot afford to
> produce in quantity, because a faithful transient spans up to ten decades of time.
> Instead of simulating the discharge, the framework *measures what causes it*: an
> in-deck leakage replica makes the storage-node leakage characteristic
> $I_\mathrm{leak}(V_\mathrm{SN})$ observable in a single DC sweep, and closed-form
> integration turns it into retention. A 50,000-point NGSpice campaign then trains a
> digital twin that makes worst-case-over-PVT multi-objective optimization,
> variance-based sensitivity analysis and mismatch-based yield estimation affordable,
> with every optimizer output returned to the simulator for verification.

---

## Repository structure

| Path | Contents |
|---|---|
| `01_dataset/` | The released dataset — Excel workbook (data dictionary, full table, design space, summary statistics, provenance, retention validation), a gzipped CSV mirror, and Parquet files (`main_raw`, `main_processed`) for programmatic use. |
| `02_report/` | The technical report (PDF, Word, Markdown) and the six core algorithms as pseudocode. |
| `03_figures/` | Every manuscript figure as PNG (300 dpi), PDF and SVG. |
| `04_tables/` | Every table as CSV, LaTeX (`booktabs`) and Markdown. |
| `05_results/` | Underlying result data: optimization fronts and indicators, surrogate leaderboards and per-fold scores, SHAP/PDP/ALE outputs, robustness/yield data, statistical tests. |
| `06_models/` | The fitted digital twin (`digital_twin_main.joblib`) and the surrogate archive. |
| `07_reproducibility/` | Configurations, per-stage run manifests, the literature database, the environment lock, and reproducibility guides. |
| `08_manuscript/` | The manuscript sources (LaTeX + bibliography), the compiled PDF, the Word version, figures and CircuiTikz sources. |
| `code/` | The **`dramdt`** framework (`src/`), the numbered pipeline `scripts/`, the `configs/`, and `requirements.txt` / `pyproject.toml`. |
| `CHECKSUMS.json` | SHA-256 for every file in the released artifact folders (`01_`–`08_`). |

## The dataset

`01_dataset/` holds 49,987 characterized 1T1C DRAM cell designs (13 design
variables + 2 operating conditions, 6 responses). Failing designs are **retained and
labeled**, so the dataset covers the infeasible region; 57.9 % of points are
feasible. The Excel workbook's first sheets document the schema and design space.

## Reproducing the results

```bash
cd code
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The pipeline is a numbered sequence of stages under `code/scripts/` (each writes to
`results/` and a run manifest); all randomness derives from a single master seed via
a pure `(seed, label, index)` hash, so results are independent of worker count and
completion order:

```
02_generate_dataset  →  04_train_surrogates  →  05_optimize
   →  06_explain  →  07_robustness  →  08_statistics  →  09_figures_tables
```

Stage 02 (the SPICE campaign) needs NGSpice 46 on the path; the later stages run on
the released dataset alone. Configuration lives in `code/configs/` (base, technology,
surrogate, optimization); no DRAM constant is hard-coded, so retargeting to another
node or topology is a configuration edit.

## How to cite

If you use this dataset, twin or framework, please cite the paper (update on
publication) and this repository. A machine-readable `CITATION.cff` is included, so
GitHub's **"Cite this repository"** button will render a formatted citation.

## License

- **Code** (`code/`): MIT License (see `LICENSE`).
- **Dataset, figures, tables and manuscript**: released for reuse with attribution
  (CC-BY-4.0 recommended — state your final choice here).

## Persistent archive / DOI

GitHub is not itself a citable long-term archive. To obtain the persistent DOI the
paper's data-availability statement refers to, create a GitHub **Release** and link
this repository to **[Zenodo](https://zenodo.org)** (Zenodo mints a DOI for each
release automatically). Put that DOI here once minted.
