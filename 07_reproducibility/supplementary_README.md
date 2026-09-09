# Supplementary material

| File | What it contains |
|---|---|
| `environment_lock.txt` | Exact package versions, NGSpice build and device-model fingerprint used for the archived run. Written by `scripts/00_setup_environment.py`. |
| `../docs/literature/references.yaml` | The reference database, split into 2021-2026 positioning works and foundational methodology, each with the URL that was retrieved. |
| `../reports/algorithms/` | The six core algorithms as IEEE-style pseudocode in LaTeX, Markdown and plain text. |
| `../results/tables/` | Every table in CSV, LaTeX (`booktabs`) and Markdown. |
| `../results/figures/` | Every figure as PNG (300 dpi), PDF and SVG. |
| `../logs/manifest_*.json` | Per-stage run manifests: seeds, configuration hash, inputs, outputs, package versions, timings. |
| `../data/processed/dram_dataset_main.xlsx` | The repository-ready dataset workbook. |

## Notes on interpreting the archived results

**Device models.** All simulations use the public PTM BSIM4 (level 54) cards for
a low-power *logic* process. These enable gate tunnelling (`igcmod`, `igbmod`),
junction leakage (`diomod`) and non-zero GIDL coefficients, which is why
retention can be simulated from device physics at all — but they are not DRAM
process models. Absolute retention times are correspondingly longer than a
production DRAM cell would exhibit.

**Process corners.** SS/FF/SF/FS cards are synthesised from the typical card by
first-order `vth0` / `u0` / `tox` shifts whose magnitudes are declared in
`configs/dram_1t1c_45nm_lp.yaml`. Each generated card carries a header stating
the shifts that were applied. They are a defensible spread, not foundry
sign-off corners.

**Censoring.** Rows with `retention_censored = TRUE` are design points whose
storage node reaches equilibrium *above* the read-failure level, so they never
lose the stored '1' under that bias condition. `retention_time_s` then holds the
configured censoring cap (`metrics.retention_censor_cap_s`), not a measured
value. Treat these as right-censored observations.

**Zero retention.** `retention_time_s = 0` is a real observation, not a missing
value: it means the write left the storage node at or below the level at which
the read already fails.

**Infeasible designs are included.** Points that fail to write or fail to sense
are retained and flagged (`write_success`, `read_success`, `design_feasible`).
They are part of the dataset on purpose: a surrogate that has only seen working
designs cannot warn an optimiser away from a broken one.

**The assist study is a controlled, paired comparison.** Every technique is
applied *on top of the baseline configuration*, not to the raw sampled design,
and every technique sees the identical set of 300 base designs. Without that
composition a technique would inherit whatever the Latin hypercube happened to
draw for the knobs the baseline pins, and a technique with no electrical effect
at all would still appear to move the metrics. The check that the design is
sound is `refresh_interval_opt`: it is a pure refresh-scheduling policy, so it
must show exactly 0 % change on every electrical metric — and it does.

**Power granularity.** Power is reported for the simulated slice — one cell plus
its column periphery (sense amplifier, precharge/equalise devices, write
driver). The active component is `energy_per_access x access_rate_hz` with
`access_rate_hz` taken from configuration; standby, active and refresh
components are reported separately so results can be reweighted for a different
access profile without re-simulating.
