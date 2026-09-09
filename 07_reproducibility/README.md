# A Digital Twin-Driven Framework for Low-Power DRAM Optimization Using PySpice and Machine Learning

A configuration-driven, end-to-end research framework that takes a parameterised
DRAM bit cell from netlist generation, through automated NGSpice
characterisation, machine-learning surrogate modelling, digital-twin
deployment, robust multi-objective optimisation, explainable-AI analysis and
statistical validation, to publication-ready figures, tables and a technical
report.

Every number the framework reports comes from simulations it actually ran.
Nothing is illustrative unless it says so.

![Proposed method](Proposed%20Method.png)

---

## What it does

| Stage | Module | What happens |
|---|---|---|
| 1. Cell design & parameterisation | `dramdt.models` | 1T1C cell, bitline pair, boosted precharge/equalise, CMOS write driver, cross-coupled sense amplifier — all emitted from YAML |
| 2. Automated simulation | `dramdt.simulation` | One NGSpice invocation per design point runs an operating point, a write/read transient and a storage-node leakage DC sweep |
| 3. Dataset generation | `dramdt.doe`, `dramdt.dataset` | Latin-hypercube design, parallel campaign with checkpointing, cleaning, feature engineering, Excel/CSV release |
| 4. Surrogate models | `dramdt.surrogate` | 14 model families compared under 5-fold cross-validation, Optuna tuning, bootstrap confidence intervals |
| 5. Multi-objective optimisation | `dramdt.optimization` | NSGA-II, NSGA-III, MOEA/D, differential evolution, PSO and Bayesian optimisation under an identical budget, repeated for statistics |
| 6. Digital twin | `dramdt.digital_twin` | The best surrogate per response, packaged with the full input pipeline behind one `predict` call |
| 7. Analysis | `dramdt.xai`, `dramdt.robustness`, `dramdt.statistics` | SHAP/ALE explainability, assist-technique comparison, Monte-Carlo yield, PVT corner sweeps, Sobol' sensitivity, Friedman/Nemenyi testing |

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate elsewhere
pip install -e .

python scripts/00_setup_environment.py    # fetch NGSpice + PTM models, self-check
python scripts/run_all.py --quick         # ~15 min smoke run of every stage
python scripts/run_all.py                 # the full study
```

`--quick` runs every stage with small sample counts and budgets, which is the
fastest way to confirm the installation works end to end.

### Running one stage at a time

```bash
python scripts/00_setup_environment.py            # NGSpice + PTM device models
python scripts/01_validate_spice.py               # physics + solver validation
python scripts/02_generate_dataset.py             # the 50,000-point campaign
python scripts/03_assist_study.py                 # assist-technique comparison
python scripts/04_train_surrogates.py             # surrogate zoo + digital twin
python scripts/05_optimize.py                     # multi-objective optimisation
python scripts/06_explain.py                      # SHAP / PDP / ALE
python scripts/07_robustness.py                   # yield, corners, sensitivity
python scripts/08_statistics.py                   # significance testing
python scripts/09_figures_tables.py               # publication figures
python scripts/10_report.py                       # the technical report
```

Stages resume: `python scripts/run_all.py --from 04 --to 10`.

---

## Retargeting to a different DRAM architecture

No DRAM-specific constant is hard-coded. To model a different cell or node,
add a YAML file and point the scripts at it:

```yaml
# configs/my_dram.yaml
extends: base.yaml
meta:      { name: my_dram, cell_type: 1T1C }
technology:
  node_nm: 22
  model_card: 22nm_LP.pm
  corners: { TT: {}, SS: { nmos_dvth0: 0.04, pmos_dvth0: 0.04 } }
design_space:
  variables:
    vdd: { low: 0.6, high: 1.0, unit: V }
    cs:  { low: 3.0e-15, high: 25.0e-15, unit: F, scale: log }
    # ...
  derived:
    vwl_high: "vdd + vwl_boost"
```

```bash
python scripts/run_all.py --tag my_dram   # after pointing --config at the file
```

Configurations for 45 nm, 32 nm and 22 nm PTM low-power nodes ship with the
framework.

---

## Repository layout

```
configs/           architecture, surrogate and optimisation configurations
src/dramdt/        the framework package
  models/          cell netlist, technology/corners, assist techniques
  simulation/      NGSpice execution, measurement extraction, retention model
  doe/             Latin hypercube / Sobol designs
  dataset/         campaign, preprocessing, repository export
  surrogate/       model zoo, cross-validated training, metrics
  digital_twin/    the deployed twin
  optimization/    problem, algorithms, indicators, runner
  xai/             SHAP, permutation importance, PDP, ALE
  robustness/      Monte-Carlo yield, corner sweeps, Sobol' indices
  statistics/      Friedman, Nemenyi, Wilcoxon, effect sizes, bootstrap
  viz/             validated palette, publication style, every figure
  reporting/       tables, IEEE algorithms, report assembly
scripts/           one executable per pipeline stage
tests/             unit tests + NGSpice integration tests
data/              spice_models/, raw/, interim/, processed/
models/            fitted surrogates and the packaged digital twin
results/           figures/, tables/, surrogate/, optimization/, xai/,
                   robustness/, statistics/
reports/           technical report (Markdown + LaTeX) and IEEE algorithms
logs/              per-stage logs and JSON run manifests
docs/literature/   the reference database
notebooks/         exploratory notebooks
supplementary/     environment lock and supporting material
```

---

## Two things worth knowing about the physics

**Retention is measured, not assumed.** Every deck contains a *leakage replica*
— an access device identical to the cell's, whose storage node is driven by a
probe source. A DC sweep of that source yields the storage-node leakage
characteristic `I_leak(V)` at the sample's own temperature and corner, including
subthreshold, GIDL, junction and gate-tunnelling components. Retention follows
by integrating the exact charge-loss relation

```
t_ret = ∫ C_node / I_leak(V) dV     from V_fail to V_written
```

and the result is cross-checked against direct long-window transient simulation
(`scripts/01_validate_spice.py`).

**The failure level is derived, not picked.** A cell has failed when charge
sharing can no longer produce a resolvable bitline differential, so

```
V_fail = V_BLpre + ΔV_min · (Cs + Cbl) / Cs
```

with `ΔV_min` the sense amplifier's input-referred offset from the configuration.

---

## Requirements

* Python 3.12+
* NGSpice 44+ (`scripts/00_setup_environment.py` vendors a copy on Windows; use
  `apt install ngspice` / `brew install ngspice` elsewhere)
* The Python packages in `requirements.txt`

On Windows the framework prefers `ngspice_con.exe` over `ngspice.exe`: the
latter links the GDI plotting front-end and blocks indefinitely when started
without an interactive window station, which is exactly what happens in a
detached batch run.

### The two execution backends

`subprocess` (default) runs `ngspice -b` as an isolated child process and is
what the campaign uses — NGSpice keeps global state, so only process isolation
makes 50,000 runs safely parallel. `pyspice-shared` drives the NGSpice shared
library through PySpice; it reproduces the operating point and the transient
*exactly* (verified in `results/tables/validation_backend_equivalence.csv`) but,
with PySpice 1.5 + NGSpice 46, cannot execute a `dc` sweep through the shared
command interface. The leakage characteristic therefore comes from the
subprocess backend. See `REPRODUCIBILITY.md` §8.

---

## Reproducibility

Every stage writes a JSON manifest to `logs/` recording its seeds, configuration
hash, inputs, outputs, package versions and wall-clock time. All randomness
derives from one master seed via a pure `(seed, label, index)` hash, so results
do not depend on worker count or completion order. See `REPRODUCIBILITY.md`.

## Dataset

`scripts/02_generate_dataset.py` writes a repository-ready Excel workbook to
`data/processed/` with README, data-dictionary, dataset, design-space,
summary-statistics and provenance sheets, plus a gzipped CSV mirror. It is
intended for deposit in a data repository such as Mendeley Data or Zenodo.

## Limitations

Stated in full in Section 9 of the generated report. In short: the device models
are public PTM BSIM4 cards for a general-purpose low-power **logic** process,
not a DRAM process, so absolute retention times are longer than a production
part would show; process corners are first-order syntheses, not foundry sign-off
models; the bitline is lumped; and the optimiser searches the surrogate, which is
why its solutions are returned to NGSpice for verification.

## Licence

MIT (code). The PTM model cards are redistributed under the PTM project's terms;
the generated dataset is released CC BY 4.0.
