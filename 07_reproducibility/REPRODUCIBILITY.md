# Reproducibility guide

This document states exactly what has to be true for a re-run to reproduce the
reported results, and what will legitimately differ.

---

## 1. Determinism model

There is **one** source of randomness: `experiment.seed` in the configuration.
Every stochastic component derives its own stream from it through a pure
function

```python
derive_seed(master_seed, label, index) -> int      # BLAKE2b, not Python's hash()
```

implemented in `src/dramdt/seeds.py`. Consequences:

* **Order independence.** A sample's mismatch draw depends on its index, not on
  when it is simulated, so the dataset is identical whether the campaign runs on
  1 worker or 32. `tests/test_sampling_and_seeds.py::test_sample_order_does_not_affect_per_sample_draws` asserts this.
* **Cross-platform stability.** BLAKE2b is used instead of `hash()`, which is
  randomised per interpreter process unless `PYTHONHASHSEED` is pinned.
* **Independent streams.** Different labels give independent sequences, so
  adding a new stochastic stage cannot perturb an existing one.

`seed_everything()` additionally seeds Python's `random`, NumPy's legacy global
state and (if present) PyTorch, and pins BLAS thread counts to 1 so that
floating-point reduction order does not vary with the thread pool.

---

## 2. What is pinned

| Layer | Where it is recorded |
|---|---|
| Configuration | SHA-256 (first 16 hex) of the canonicalised merged document, in every manifest |
| Device models | SHA-256 of the actual model-card bytes (`model_fingerprint`) |
| NGSpice | executable path and version banner, in `logs/manifest_setup.json` |
| Python packages | `supplementary/environment_lock.txt` and every manifest |
| Seeds | `seeds` block of every manifest |
| Wall-clock | `duration_s` of every stage |

The configuration hash covers **everything** that affects the physics, including
solver tolerances and the assist-technique definitions. If you edit any
configuration file, the hash changes and the manifests will show it — that is
intentional.

---

## 3. Exact reproduction

```bash
git clone <repository> && cd <repository>
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

python scripts/00_setup_environment.py     # NGSpice + PTM cards + self-check
python scripts/run_all.py --tag main
```

Then compare:

```bash
# configuration and model provenance must match exactly
python -c "from dramdt.config import load_config; print(load_config('dram_1t1c_45nm_lp.yaml').hash)"
python -c "from dramdt.config import load_config; from dramdt.models import Technology; \
           print(Technology.from_config(load_config('dram_1t1c_45nm_lp.yaml')).fingerprint())"
```

If both match the values in `reports/results_summary_main.json`, you are running
the same experiment.

---

## 4. What will reproduce bit-for-bit, and what will not

**Bit-for-bit identical**

* the sampling design (a pure function of the seed);
* the train/validation/test partition;
* every derived quantity and engineered feature;
* the corner model cards.

**Numerically identical in practice, but not guaranteed bit-for-bit**

* SPICE measurements. NGSpice's adaptive time-stepping is deterministic for a
  fixed binary and deck, but a different NGSpice build, a different BLAS or a
  different CPU's floating-point behaviour can move the last digits. Metric
  differences of order 1e-9 relative are expected and harmless; the solver
  convergence study in `results/tables/solver_convergence_study.csv` quantifies
  the sensitivity that actually matters.
* Model fits. Scikit-learn, XGBoost, LightGBM and CatBoost are seeded, but
  multi-threaded histogram construction can reorder floating-point reductions.
  Scores typically agree to ~1e-6; rankings do not change.

**Legitimately different**

* wall-clock timings and the `samples_per_second` figures;
* worker PIDs and scratch paths;
* the tie-breaking order of exactly-equal Pareto solutions.

---

## 5. Cost

Measured on a 14-core / 20-thread laptop CPU:

| Stage | Approximate cost |
|---|---|
| 00 setup | 1-3 min (downloads) |
| 01 validation | 3-6 min |
| 02 dataset, 50,000 samples | 45-70 min on 18 workers |
| 03 assist study | 10-25 min |
| 04 surrogates (14 models x 6 responses) | 20-60 min |
| 05 optimisation (6 algorithms x 11 runs) | 20-60 min |
| 06 explainability | 5-20 min |
| 07 robustness | 5-15 min |
| 08 statistics | < 1 min |
| 09 figures | 1-3 min |
| 10 report | < 1 min |

`python scripts/run_all.py --quick` completes every stage in roughly 15 minutes
and is the recommended first run.

---

## 6. Restartability

The campaign writes one Parquet file per chunk to `data/interim/chunks_<tag>/`
and skips chunks that already exist. An interrupted 50,000-sample run resumes
where it stopped:

```bash
python scripts/02_generate_dataset.py --tag main      # resumes by default
python scripts/02_generate_dataset.py --tag main --no-resume   # start clean
```

Because chunk contents are a pure function of `(seed, index)`, a resumed run is
identical to an uninterrupted one.

---

## 7. Verifying an installation

```bash
pytest -q                      # full suite, including NGSpice integration tests
pytest -q -m "not spice"       # unit tests only, no simulator needed
pytest -q -m "not slow"        # skip the long physics checks
```

The integration tests are the meaningful check: they assert that charge sharing
matches the analytical law, that the quasi-static retention model tracks a direct
transient, that retention falls with temperature, that a strongly negative
wordline switches on GIDL, and that both execution backends agree.

---

## 8. Known environment pitfalls

* **Windows, detached runs.** Use `ngspice_con.exe`, not `ngspice.exe`. The
  latter links the GDI front-end and blocks forever without an interactive
  window station. `dramdt.env` prefers the console build automatically.
* **`stdin`.** Child NGSpice processes are given `DEVNULL`; without it a prompt
  can hang a batch run indefinitely.
* **Solver tolerances are per analysis.** The transient resolves micro-amp
  switching; the leakage sweep resolves femto-amps. A single global `ABSTOL`
  cannot serve both, and a single global `GMIN` is worse — NGSpice's default
  `GMIN` of 1e-12 S injects hundreds of femto-amps into the storage node and
  makes retention look an order of magnitude too short.
* **Parquet.** `pyarrow` is used for chunk checkpoints; without it the campaign
  transparently falls back to CSV.
* **Memory.** 18 workers peak at roughly 4 GB total.
* **YAML numeric literals.** PyYAML implements YAML 1.1, whose float pattern
  requires a *signed* exponent: `1.0e-15` is a float but `1.0e15` is the string
  `"1.0e15"`. Every consumer in this framework coerces with `float()`, so such
  values remain numerically correct, and dataset writers pass frames through
  `sanitize_for_parquet` (which logs any column it had to coerce). Prefer
  `1.0e+15` in new configuration keys.
* **The PySpice shared-library backend cannot run a DC sweep.** With
  PySpice 1.5 and NGSpice 46, a `dc` command issued through the shared
  library's command interface fails with *"Voltage source ... is not in the
  circuit"* even on a trivial two-resistor deck, so the storage-node leakage
  characteristic — and therefore retention and leakage current — is available
  only from the `subprocess` backend. The operating point and the transient
  reproduce **exactly** across the two backends (see
  `results/tables/validation_backend_equivalence.csv`), which is why the
  equivalence check compares the transient-derived metrics and the campaign
  uses the subprocess backend throughout. The shared backend also does not
  execute `.control` blocks on `run()`; `SpiceRunner.split_control_block`
  replays those commands one at a time through `exec_command`.

---

## 9. File-level provenance

| Artefact | Produced by | Manifest |
|---|---|---|
| `data/processed/<tag>_processed.parquet` | `02_generate_dataset.py` | `logs/manifest_dataset_<tag>.json` |
| `data/processed/dram_dataset_<tag>.xlsx` | `02_generate_dataset.py` | same |
| `models/digital_twin_<tag>.joblib` | `04_train_surrogates.py` | `logs/manifest_surrogate_<tag>.json` |
| `results/optimization/pareto_<tag>.csv` | `05_optimize.py` | `logs/manifest_optimize_<tag>.json` |
| `results/figures/*` | `09_figures_tables.py` | `logs/manifest_figures_<tag>.json` |
| `reports/technical_report_<tag>.md` | `10_report.py` | `logs/manifest_report_<tag>.json` |

Every manifest records the configuration hash it was produced under. If two
artefacts disagree on that hash, they came from different experiments.
