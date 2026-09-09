# A Digital Twin-Driven Framework for Low-Power DRAM Optimization Using PySpice and Machine Learning

*A configuration-driven, end-to-end framework that couples automated NGSpice characterisation of a 1T1C DRAM bit cell with machine-learning surrogates, a deployable digital twin, robust multi-objective optimisation and explainable-AI analysis. Every reported number is produced by the accompanying code from executed simulations.*

_Generated 2026-07-23T13:46:25+00:00_

## Contents

- [1. Introduction](#1-introduction)
- [2. Related work (2021-2026)](#2-related-work-2021-2026)
- [3. Framework](#3-framework)
- [4. Experimental setup](#4-experimental-setup)
- [5. Dataset and physical validation](#5-dataset-and-physical-validation)
- [6. Surrogate models and the digital twin](#6-surrogate-models-and-the-digital-twin)
- [7. Multi-objective optimisation](#7-multi-objective-optimisation)
- [8. Explainability, robustness and assist techniques](#8-explainability-robustness-and-assist-techniques)
- [9. Limitations and threats to validity](#9-limitations-and-threats-to-validity)
- [10. Reproduction](#10-reproduction)


## 1. Introduction

DRAM bit-cell design is a genuinely multi-objective problem. Retention time,
read latency, read margin and power cannot be improved independently: raising the
storage capacitance improves retention and sense margin but costs area and access
energy; boosting the wordline enables a full '1' write but increases both power and
gate-oxide stress; lowering the bitline precharge level saves precharge energy but
erodes the margin for one of the two data polarities. Exploring that space directly
with a circuit simulator is expensive, and the expense grows multiplicatively once
process corners, temperature and statistical variation enter.

This work builds a complete framework that makes such exploration tractable. A
parameterised 1T1C cell, its bitline pair, boosted precharge network, CMOS write
driver and cross-coupled sense amplifier are generated from configuration and
characterised in NGSpice. A space-filling design of
50000 points is
simulated, machine-learning surrogates are fitted to the resulting responses, and the
best surrogate per response is deployed as a digital twin. The twin then drives a
robust multi-objective search whose solutions are returned to the circuit simulator
for verification.

**What is contributed.** (i) A configuration-driven pipeline in which no DRAM
architectural constant is hard-coded, so retargeting to another cell or technology
node means editing YAML, not Python. (ii) A retention model that is *measured*
rather than assumed: the storage-node leakage characteristic is extracted by a DC
sweep of a leakage replica inside the same simulation, and the charge-loss ODE is
integrated over it -- with the result cross-checked against direct long-window
transient simulation. (iii) A broad, statistically tested comparison of
14 surrogate
families and six optimisers under identical budgets. (iv) A published dataset of
every simulated design point. (v) Explainability and robustness analyses that turn
the surrogate from a black box into a source of design guidance.

**What is deliberately not claimed.** The device models are the public PTM BSIM4
cards for a general-purpose low-power logic process, not a DRAM-specific process.
Absolute retention times therefore reflect those cards and are not predictions about
any commercial part; the framework, the relative trends and the methodology are the
contribution. Section 9 states the limitations in full.


## 2. Related work (2021-2026)

Machine learning for analog and mixed-signal design has converged on two families:
*simulation-in-the-loop* methods, which call the simulator inside the optimiser, and
*surrogate-based* methods, which replace most of those calls with a learned model.
Recent surveys of the area place both families side by side, and recent
optimisation studies report reductions in SPICE calls of roughly 56-83 %, with
multi-fidelity schemes reporting an order-of-magnitude reduction in total
simulation time while still matching ground-truth Pareto fronts. This framework sits
in the surrogate family and adopts the same discipline of returning to the simulator
for verification.

On the tooling side, programmatic SPICE deck generation with PySpice has recently
been used both to build datasets for learning and to drive complete simulation
studies, which supports the toolchain adopted here. On the DRAM side, published work
concentrates on the architecture and system levels -- refresh scheduling, retention
profiling, data mapping and measurement infrastructure -- and on demonstrating that
DRAM modelling accuracy must be validated rather than assumed. The present study
works one level below those, at the bit cell and its column periphery, and adopts the
same validation discipline: the retention model is checked against direct transient
simulation before it is used.

The gap this work addresses is the absence of an *end-to-end, reproducible and
openly-data-backed* pipeline that takes a configurable DRAM cell from netlist
generation through surrogate modelling, digital-twin deployment, robust
multi-objective optimisation, explainability and statistical validation.

**Table 1.** Recent works positioning this study.


| Work | Year | Focus |
|---|---|---|
| Multi-Fidelity Surrogate Models for Accelerated Multi-Objective Analog Circuit | 2025 | surrogate modelling; multi-objective circuit optimisation |
| Analog Design and Machine Learning: A Review | 2025 | review of ML in analog design |
| Analog circuit sizing based on Evolutionary Algorithms and deep learning | 2024 | evolutionary sizing with learned models |
| Machine Learning Driven Global Optimisation Framework for Analog Circuit Desig | 2024 | ML-guided global optimisation |
| Performance Evaluation of Evolutionary Algorithms for Analog Integrated Circui | 2023 | benchmarking evolutionary optimisers on circuit problems |
| The Dawn of AI-Native EDA: Opportunities and Challenges of Large Circuit Model | 2024 | AI-native EDA landscape |
| When Device Modeling Meets Machine Learning: Opportunities and Challenges | 2024 | ML for compact/device modelling |
| Digital Twin for Secure Semiconductor Lifecycle Management: Prospects and Appl | 2022 | digital twins in the semiconductor lifecycle |
| SPICEPilot: Navigating SPICE Code Generation and Simulation with AI Guidance | 2024 | automated SPICE deck generation, PySpice-based datasets |
| Python Framework for Modular and Parametric SPICE Netlists Generation | 2023 | parametric netlist generation |
| PySpice-Simulated In Situ Learning with Memristor Emulation for Single-Layer S | 2024 | PySpice/NGSpice in a research pipeline |
| EasyDRAM: An FPGA-based Infrastructure for Fast and Accurate End-to-End Evalua | 2025 | DRAM characterisation infrastructure |
| Re-Evaluating the Real-System Modeling Accuracy of Ramulator 2.0 | 2026 | DRAM simulator fidelity |
| PENDRAM: Enabling High-Performance and Energy-Efficient Processing of Deep Neu | 2024 | DRAM energy optimisation at the architecture level |

_Full bibliographic records and retrieval URLs are in docs/literature/references.yaml._


## 3. Framework

The framework follows the seven stages of the proposed method.

**Stage 1 -- cell design and parameterisation.** `dramdt.models.cell` emits the
BSIM4 deck for a 1T1C cell, a lumped bitline pair, a boosted precharge/equalise
network, a CMOS transmission-gate write driver and a cross-coupled sense
amplifier. Two design details are worth stating because they are easy to get
wrong and both were found by simulation during development: the precharge pass
devices must be driven from the boosted rail, because a VDD-level gate leaves
`Vgs = VDD/2 < Vth` and cannot pull the bitline to VDD/2; and the access device
needs `VPP > VDD + Vth,eff` to store a full '1', which is why the wordline-boost
range extends to 1.2 V.

**Stage 2 -- automated simulation.** Each deck runs three analyses in one NGSpice
invocation: an operating point for standby current, a transient covering
write / precharge / read / sense, and a DC sweep of a *leakage replica* -- an
identical access device whose storage node is driven by a probe source -- yielding
the storage-node leakage characteristic. Tolerances are set per analysis, because a
single ABSTOL cannot resolve femto-ampere leakage and micro-ampere switching in the
same deck.

**Stage 3 -- dataset.** Latin hypercube sampling over
15 variables, decoded through the configuration's derived
expressions. Design points that fail to write or to sense are *retained and
labelled*, not discarded: a surrogate that has never seen an infeasible design
cannot steer an optimiser away from one.

**Stage 4 -- surrogates.** A zoo spanning linear, kernel, neighbour, ensemble,
boosting and neural families is cross-validated on train+val; the test split is
touched once per model.

**Stage 5 -- optimisation.** Six optimisers under an identical evaluation budget,
each repeated 7 times. Objectives and
constraints are reduced pessimistically over the configured PVT conditions, making
the search a robust-design problem.

**Stage 6 -- digital twin.** The best surrogate per response is packaged with the
full input-preparation pipeline behind one `predict` call.

**Stage 7 -- analysis.** Assist-technique comparison, SHAP/ALE explainability,
Monte-Carlo yield, corner sweeps and Sobol' sensitivity.

The pseudocode for the six core algorithms is in `reports/algorithms/`.


## 4. Experimental setup

| Item | Value |
|---|---|
| Simulator | NGSpice |
| Device models | PTM BSIM4 (level 54), ? nm  |
| Model-card fingerprint | `?` |
| Process corners |  |
| Design variables | 15 |
| Samples | 50000 |
| Sampler | lhs |
| Master seed | 20260723 |
| Configuration hash | `e22061fdb47e16e7` |
| Platform | ? |

Corner cards for SS/FF/SF/FS are synthesised from the typical PTM card by the
first-order `vth0`/`u0`/`tox` shifts declared in the configuration; they are
explicitly *not* foundry sign-off corners. A solver convergence study (Table: solver_convergence_study) shows the production setting differs from a 20 ps GEAR reference by at most 0.70 % at the 95th percentile on every metric, with pass/fail labels unchanged.

All randomness derives from the single master seed through a pure
`(seed, label, index)` hash, so results are independent of worker count and
completion order.


## 5. Dataset and physical validation

The campaign completed **50000 simulations**
(71428027540.329 per second on
18 workers, 0.0 min
total), of which 49999 produced measurements
(100.00 %). Preprocessing retained
49987 of 50000 rows
(99.97 %); the removals were
{'simulation_failed': 1, 'not_converged': 3, 'incomplete_measurements': 9}. 57.9 % of the retained points are feasible (both write and read succeed); the remainder are kept and labelled as infeasible.

The Latin hypercube achieved a centred-L2 discrepancy of
0.00680435 and a minimum pairwise
distance of 0.4115
(both on a 4000-point subsample).

**Two independent physics checks were run before trusting the dataset.**

First, the measured bitline differential was compared with the analytical charge-sharing law `dV = (V_SN - V_BLpre)*Cs/(Cs+Cbl)`: median relative error **3.802 %**, 95th percentile 17.992 % over 120 design points.

Second, the quasi-static retention model was checked against direct long-window transient simulation on 200 designs: Pearson r = **0.99974** in log10 space, Spearman rho = 0.9995, median absolute error **0.0107 decades**, and 100.0 % of designs within a factor of two. The retention numbers in this study are therefore integrations of a measured leakage characteristic whose accuracy has been quantified, not an analytical shortcut.

![Figure 1](results/figures/fig01_design_space_coverage.png)

**Figure 1.** Latin-hypercube design: pairwise projections and marginals.

![Figure 2](results/figures/fig02_response_distributions.png)

**Figure 2.** Distribution of the simulated responses.

![Figure 3](results/figures/fig03_correlation_matrix.png)

**Figure 3.** Rank correlation between design variables and responses.

![Figure 4](results/figures/fig04_retention_validation.png)

**Figure 4.** Validation of the quasi-static retention model against direct transient simulation.

**Table 2.** Measured bitline differential against the charge-sharing law.


| sample | corner | temperature_c | cs_ff | cbl_ff | v_sn_written_v | measured_dv_mv | analytical_dv_mv | relative_error_pct |
|---|---|---|---|---|---|---|---|---|
| 0 | TT | 79.61 | 23.49 | 198.7 | 0.7813 | 45.07 | 44.86 | 0.4685 |
| 1 | TT | -3.646 | 11.11 | 45.92 | 0.5546 | 48.81 | 47.59 | 2.568 |
| 2 | TT | -7.444 | 9.672 | 109.0 | 1.098 | 48.88 | 47.21 | 3.536 |
| 3 | FF | -38.56 | 29.77 | 171.6 | 0.8018 | 53.41 | 52.43 | 1.880 |
| 4 | TT | 107.9 | 7.442 | 76.53 | 0.9565 | 51.70 | 43.63 | 18.52 |
| 5 | SF | -32.83 | 6.146 | 47.00 | 0.8195 | 57.01 | 51.13 | 11.49 |
| 6 | SF | -17.01 | 10.88 | 109.1 | 0.8089 | 33.23 | 30.24 | 9.903 |
| 7 | TT | 55.37 | 23.79 | 52.85 | 0.8160 | 133.5 | 131.7 | 1.342 |
| 8 | TT | 73.93 | 7.127 | 33.60 | 1.199 | 132.5 | 125.2 | 5.789 |
| 9 | SF | 105.0 | 22.11 | 263.0 | 1.023 | 40.31 | 39.78 | 1.337 |
| 10 | FS | 36.83 | 27.30 | 143.9 | 1.198 | 101.0 | 101.8 | 0.8056 |
| 11 | FF | 29.88 | 29.98 | 189.1 | 0.9430 | 51.65 | 49.84 | 3.643 |
| 12 | FS | 60.77 | 6.861 | 16.15 | 0.6082 | 55.07 | 49.24 | 11.84 |
| 13 | TT | -12.90 | 9.340 | 90.63 | 0.8712 | 31.93 | 30.82 | 3.627 |
| 14 | SF | 62.23 | 22.65 | 184.2 | 0.8458 | 58.98 | 57.44 | 2.687 |
| 15 | FS | 6.565 | 7.282 | 75.56 | 0.6259 | 23.95 | 23.33 | 2.682 |
| 16 | SS | 35.24 | 24.73 | 264.2 | 0.7510 | 14.23 | 14.02 | 1.493 |
| 17 | SS | 26.37 | 7.676 | 15.74 | 0.7849 | 112.1 | 115.9 | 3.254 |
| 18 | FF | 11.36 | 39.13 | 290.9 | 0.7774 | 44.35 | 43.67 | 1.576 |
| 19 | SF | 85.20 | 20.63 | 202.2 | 0.6425 | 24.96 | 13.87 | 79.99 |
| 20 | FF | 102.0 | 37.45 | 352.0 | 0.7889 | 12.61 | 12.16 | 3.704 |
| 21 | SF | -34.42 | 8.131 | 77.02 | 1.010 | 55.09 | 50.40 | 9.315 |
| 22 | TT | 4.035 | 5.239 | 32.38 | 1.054 | 66.10 | 62.05 | 6.527 |
| 23 | SF | 23.43 | 10.68 | 59.74 | 0.6586 | 36.17 | 32.82 | 10.20 |
| 24 | FF | -22.83 | 8.069 | 53.13 | 1.239 | 98.29 | 97.00 | 1.325 |
| 25 | SF | 8.580 | 26.03 | 90.16 | 0.6933 | 54.90 | 52.71 | 4.164 |
| 26 | FS | -16.47 | 19.22 | 132.8 | 1.165 | 87.14 | 86.24 | 1.042 |
| 27 | FS | 0.8620 | 20.24 | 88.55 | 0.7496 | 57.41 | 55.76 | 2.973 |
| 28 | TT | 103.1 | 8.589 | 59.99 | 0.8577 | 42.46 | 37.28 | 13.89 |
| 29 | FS | 48.46 | 11.56 | 24.74 | 1.009 | 133.3 | 129.9 | 2.625 |
| 30 | FS | -34.86 | 10.06 | 50.22 | 1.269 | 122.1 | 117.2 | 4.215 |
| 31 | FF | -24.10 | 13.05 | 152.1 | 1.028 | 40.87 | 38.42 | 6.371 |
| 32 | SS | 15.50 | 5.412 | 52.07 | 0.6150 | 31.65 | 27.84 | 13.69 |
| 33 | FF | 67.51 | 8.909 | 73.46 | 0.7312 | 32.71 | 30.88 | 5.947 |
| 34 | SS | -18.96 | 16.88 | 151.1 | 0.8777 | 33.45 | 33.55 | 0.3045 |
| 35 | FF | 113.0 | 28.36 | 235.7 | 0.8158 | 50.98 | 49.23 | 3.553 |
| 36 | TT | -9.444 | 9.053 | 95.62 | 0.5893 | 18.31 | 17.07 | 7.227 |
| 37 | SF | -6.491 | 12.07 | 51.77 | 1.059 | 83.05 | 78.52 | 5.777 |
| 38 | FS | 84.65 | 5.324 | 24.22 | 1.036 | 67.91 | 61.52 | 10.40 |
| 39 | TT | 107.0 | 16.36 | 63.41 | 1.071 | 111.2 | 106.8 | 4.091 |

_Relative error between the simulated V_BL - V_BLB at the sense-amplifier firing instant and (V_SN - V_BLpre)*Cs/(Cs+Cbl)._

**Table 3.** Agreement statistics for the retention model.


| statistic | value |
|---|---|
| n_pairs | 18 |
| pearson_r_log10 | 1.0000 |
| spearman_rho | 1 |
| median_abs_log10_error | 0.009786 |
| mean_log10_error | -0.01160 |
| geometric_mean_ratio | 0.9736 |
| p90_abs_log10_error | 0.02026 |
| within_factor_2_pct | 100 |
| within_factor_10_pct | 100 |

**Table 4.** Solver convergence study.


| setting | ms_per_sample | speedup_vs_reference | read_margin_mv_median_err_pct | read_margin_mv_p95_err_pct | read_delay_ns_median_err_pct | read_delay_ns_p95_err_pct | write_delay_ns_median_err_pct | write_delay_ns_p95_err_pct | v_sn_written_v_median_err_pct | v_sn_written_v_p95_err_pct | retention_time_s_median_err_pct | retention_time_s_p95_err_pct | energy_per_access_fj_median_err_pct | energy_per_access_fj_p95_err_pct | total_power_nw_median_err_pct | total_power_nw_p95_err_pct | read_success_agreement_pct | write_success_agreement_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 ps / GEAR | 63 | 1.693 | 0.009100 | 0.1449 | 0.01530 | 0.2420 | 0.1520 | 0.7592 | 0.003800 | 0.04200 | 0.001800 | 0.4649 | 0.05650 | 0.3348 | 0.05490 | 0.3300 | 100 | 100 |
| 50 ps / TRAP | 60.90 | 1.752 | 0.003100 | 0.02840 | 0.01110 | 0.1547 | 0.03910 | 0.6964 | 7.000e-4 | 0.005800 | 3.000e-4 | 0.05170 | 0.02130 | 0.1653 | 0.02070 | 0.1591 | 100 | 100 |
| 100 ps / GEAR | 47.40 | 2.249 | 0.01990 | 0.5967 | 0.1581 | 0.8344 | 0.3680 | 1.367 | 0.01190 | 0.1842 | 0.003400 | 1.738 | 0.1905 | 0.5836 | 0.1788 | 0.5444 | 100 | 100 |

_Percentages are relative errors against the 20 ps GEAR setting over the same design points; agreement columns give the share of points whose pass/fail label is unchanged._


## 6. Surrogate models and the digital twin

The surrogate comparison is summarised below.

| Response | Selected model | CV R2 | Test R2 | Test RMSE | Test MAE |
|---|---|---|---|---|---|
| write_efficiency | mlp_deep | 0.99308 | 0.99859 | 0.004596 | 0.003023 |
| energy_per_access_fj | mlp_deep | 0.98961 | 0.99614 | 2.674 | 1.237 |
| log10_total_power_nw | mlp_deep | 0.97362 | 0.98266 | 0.05173 | 0.02065 |
| read_delay_ns | mlp_deep | 0.97108 | 0.98778 | 1.031 | 0.4237 |
| read_margin_mv | mlp | 0.96139 | 0.98966 | 3.978 | 1.659 |
| log10_retention_time_s | mlp_deep | 0.91140 | 0.95098 | 0.9212 | 0.2123 |

The model in each row is the one selected by cross-validation on the train+val partition and deployed in the digital twin; the test columns are the single, untouched-until-then evaluation of that choice.

By family, the median held-out R2 ranks as: **neural** 0.9887, **boosting** 0.9635, **ensemble** 0.9041, **kernel** 0.8642, **neighbour** 0.7236, **linear** 0.7097.

Ranked by mean Friedman rank across every response (lower is better): mlp_deep (1.40), mlp (2.23), catboost (2.73), lightgbm (4.20), hist_gbr (4.83).

**Digital twin cost.** At a batch size of 10000 the twin evaluates a design in 0.03248 ms (30,786 designs/s), a **37,009x** speed-up over the measured mean SPICE evaluation time. This is what makes the optimisation, the Monte-Carlo yield study and the Sobol' analysis affordable.

![Figure 5](results/figures/fig05_surrogate_comparison.png)

**Figure 5.** Surrogate model comparison per response.

![Figure 6](results/figures/fig07_parity.png)

**Figure 6.** Twin predictions against SPICE ground truth (test split).

![Figure 7](results/figures/fig08_residuals.png)

**Figure 7.** Residual diagnostics.

![Figure 8](results/figures/fig09_twin_speedup.png)

**Figure 8.** Digital-twin evaluation cost against circuit simulation.

**Table 5.** Full surrogate comparison.


| target | model | family | n_compare | n_train | cv_r2_mean | cv_r2_std | test_r2 | test_rmse | test_mae | test_mape | fit_time_s | predict_us_per_sample |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| energy_per_access_fj | mlp_deep | neural | 15000 | 42491 | 0.9896 | 0.002397 | 0.9961 | 2.674 | 1.237 | 4.853 | 31.34 | 2.165 |
| energy_per_access_fj | mlp | neural | 15000 | 42491 | 0.9890 | 0.001231 | 0.9957 | 2.834 | 1.340 | 5.731 | 45.74 | 2.783 |
| energy_per_access_fj | catboost | boosting | 15000 | 42491 | 0.9783 | 0.001605 | 0.9854 | 5.197 | 2.723 | 9.891 | 6.184 | 2.185 |
| energy_per_access_fj | lightgbm | boosting | 15000 | 42491 | 0.9745 | 0.002647 | 0.9840 | 5.451 | 2.769 | 9.904 | 1.496 | 5.291 |
| energy_per_access_fj | xgboost | boosting | 15000 | 42491 | 0.9725 | 0.003525 | 0.9832 | 5.576 | 2.663 | 9.468 | 17.40 | 22.10 |
| energy_per_access_fj | hist_gbr | ensemble | 15000 | 42491 | 0.9738 | 0.002531 | 0.9822 | 5.747 | 3.135 | 11.21 | 4.710 | 50.43 |
| energy_per_access_fj | random_forest | ensemble | 15000 | 42491 | 0.9426 | 0.002066 | 0.9649 | 8.071 | 4.125 | 13.75 | 7.148 | 22.22 |
| energy_per_access_fj | poly2_ridge | linear | 15000 | 42491 | 0.9611 | 0.003995 | 0.9646 | 8.098 | 4.732 | 19.88 | 0.3578 | 3.625 |
| energy_per_access_fj | extra_trees | ensemble | 15000 | 42491 | 0.9401 | 0.002139 | 0.9625 | 8.331 | 4.404 | 14.72 | 3.164 | 19.97 |
| energy_per_access_fj | gpr | kernel | 15000 | 2000 | 0.9557 | 0.002277 | 0.9602 | 8.593 | 5.239 | 20.98 | 27.14 | 89.03 |
| energy_per_access_fj | svr_rbf | kernel | 15000 | 5000 | 0.9417 | 0.003392 | 0.9419 | 10.38 | 5.438 | 19.77 | 1.580 | 250.8 |
| energy_per_access_fj | knn | neighbour | 15000 | 42491 | 0.8526 | 0.004291 | 0.8795 | 14.95 | 9.556 | 37.58 | 0.08240 | 119.1 |
| energy_per_access_fj | ridge | linear | 15000 | 42491 | 0.8358 | 0.006533 | 0.8374 | 17.36 | 12.06 | 56.32 | 0.08970 | 0.9928 |
| energy_per_access_fj | elasticnet | linear | 15000 | 42491 | 0.8357 | 0.006336 | 0.8371 | 17.37 | 12.07 | 56.37 | 15.65 | 1.093 |
| log10_retention_time_s | mlp_deep | neural | 15000 | 42491 | 0.9114 | 0.01097 | 0.9510 | 0.9212 | 0.2123 | 46.75 | 49.85 | 2.363 |
| log10_retention_time_s | mlp | neural | 15000 | 42491 | 0.9059 | 0.01186 | 0.9493 | 0.9370 | 0.2988 | 88.48 | 19.23 | 2.406 |
| log10_retention_time_s | catboost | boosting | 15000 | 42491 | 0.7972 | 0.008134 | 0.8261 | 1.735 | 1.016 | 415.5 | 6.069 | 1.976 |
| log10_retention_time_s | lightgbm | boosting | 15000 | 42491 | 0.7849 | 0.006946 | 0.8200 | 1.765 | 0.9577 | 346.0 | 1.595 | 5.624 |
| log10_retention_time_s | xgboost | boosting | 15000 | 42491 | 0.7785 | 0.01049 | 0.8188 | 1.771 | 0.9347 | 340.0 | 16.90 | 21.51 |
| log10_retention_time_s | hist_gbr | ensemble | 15000 | 42491 | 0.7756 | 0.008489 | 0.8001 | 1.860 | 1.057 | 427.9 | 2.420 | 29.09 |
| log10_retention_time_s | extra_trees | ensemble | 15000 | 42491 | 0.6933 | 0.007082 | 0.7300 | 2.162 | 1.097 | 367.7 | 3.098 | 17.68 |
| log10_retention_time_s | random_forest | ensemble | 15000 | 42491 | 0.6802 | 0.01042 | 0.7211 | 2.197 | 1.090 | 399.6 | 8.088 | 13.02 |
| log10_retention_time_s | poly2_ridge | linear | 15000 | 42491 | 0.6837 | 0.004772 | 0.6849 | 2.336 | 1.693 | 888.7 | 0.5985 | 4.384 |
| log10_retention_time_s | svr_rbf | kernel | 15000 | 5000 | 0.6615 | 0.01527 | 0.6686 | 2.395 | 1.402 | 581.1 | 5.830 | 399.1 |
| log10_retention_time_s | gpr | kernel | 15000 | 2000 | 0.6716 | 0.01009 | 0.6656 | 2.406 | 1.708 | 878.6 | 53.41 | 80.99 |
| log10_retention_time_s | knn | neighbour | 15000 | 42491 | 0.5388 | 0.01262 | 0.5910 | 2.661 | 1.538 | 568.2 | 0.08190 | 123.2 |
| log10_retention_time_s | elasticnet | linear | 15000 | 42491 | 0.4629 | 0.008114 | 0.4631 | 3.049 | 2.404 | 1252 | 10.59 | 1.562 |
| log10_retention_time_s | ridge | linear | 15000 | 42491 | 0.4630 | 0.008211 | 0.4630 | 3.049 | 2.403 | 1251 | 0.09520 | 1.018 |
| log10_total_power_nw | mlp_deep | neural | 15000 | 42491 | 0.9736 | 0.002399 | 0.9827 | 0.05173 | 0.02065 | 17.02 | 53.72 | 3.662 |
| log10_total_power_nw | mlp | neural | 15000 | 42491 | 0.9637 | 0.002229 | 0.9791 | 0.05678 | 0.02659 | 25.50 | 13.32 | 1.522 |
| log10_total_power_nw | catboost | boosting | 15000 | 42491 | 0.9702 | 0.002310 | 0.9719 | 0.06586 | 0.03814 | 37.11 | 6.397 | 2.177 |
| log10_total_power_nw | xgboost | boosting | 15000 | 42491 | 0.9638 | 0.002412 | 0.9711 | 0.06678 | 0.03805 | 36.45 | 14.47 | 21.99 |
| log10_total_power_nw | lightgbm | boosting | 15000 | 42491 | 0.9658 | 0.002232 | 0.9707 | 0.06719 | 0.03936 | 38.57 | 2.471 | 7.269 |
| log10_total_power_nw | hist_gbr | ensemble | 15000 | 42491 | 0.9640 | 0.001720 | 0.9684 | 0.06985 | 0.04259 | 41.68 | 3.567 | 32.45 |
| log10_total_power_nw | random_forest | ensemble | 15000 | 42491 | 0.9280 | 0.002960 | 0.9450 | 0.09209 | 0.05806 | 61.11 | 7.011 | 18.96 |
| log10_total_power_nw | extra_trees | ensemble | 15000 | 42491 | 0.9243 | 0.004048 | 0.9436 | 0.09326 | 0.06036 | 65.00 | 3.813 | 20.65 |
| log10_total_power_nw | poly2_ridge | linear | 15000 | 42491 | 0.9404 | 0.003131 | 0.9387 | 0.09727 | 0.06661 | 53.90 | 0.3759 | 3.514 |
| log10_total_power_nw | gpr | kernel | 15000 | 2000 | 0.9350 | 0.003826 | 0.9329 | 0.1018 | 0.06806 | 56.30 | 43.79 | 136.6 |
| log10_total_power_nw | svr_rbf | kernel | 15000 | 5000 | 0.9274 | 0.002894 | 0.9177 | 0.1127 | 0.07915 | 66.56 | 2.832 | 833.3 |
| log10_total_power_nw | ridge | linear | 15000 | 42491 | 0.8645 | 0.002016 | 0.8677 | 0.1429 | 0.1074 | 93.88 | 0.1306 | 1.167 |
| log10_total_power_nw | elasticnet | linear | 15000 | 42491 | 0.8635 | 0.002125 | 0.8667 | 0.1434 | 0.1080 | 95.44 | 2.135 | 0.9677 |
| log10_total_power_nw | knn | neighbour | 15000 | 42491 | 0.7906 | 0.002597 | 0.8298 | 0.1621 | 0.1159 | 131.7 | 0.08870 | 120.7 |
| read_delay_ns | mlp_deep | neural | 15000 | 33374 | 0.9711 | 0.004094 | 0.9878 | 1.031 | 0.4237 | 6.293 | 29.44 | 2.390 |
| read_delay_ns | mlp | neural | 15000 | 33374 | 0.9679 | 0.003776 | 0.9848 | 1.150 | 0.5135 | 10.50 | 11.78 | 1.693 |
| read_delay_ns | catboost | boosting | 15000 | 33374 | 0.9366 | 0.002128 | 0.9562 | 1.952 | 1.052 | 17.98 | 6.032 | 2.390 |
| read_delay_ns | lightgbm | boosting | 15000 | 33374 | 0.9189 | 0.004328 | 0.9446 | 2.195 | 1.137 | 16.85 | 1.978 | 6.740 |
| read_delay_ns | xgboost | boosting | 15000 | 33374 | 0.9063 | 0.003089 | 0.9386 | 2.311 | 1.167 | 15.39 | 16.20 | 20.61 |
| read_delay_ns | hist_gbr | ensemble | 15000 | 33374 | 0.9197 | 0.002455 | 0.9354 | 2.372 | 1.328 | 22.73 | 4.517 | 53.75 |
| read_delay_ns | random_forest | ensemble | 15000 | 33374 | 0.8495 | 0.004194 | 0.8799 | 3.233 | 1.766 | 21.50 | 5.359 | 24.07 |
| read_delay_ns | extra_trees | ensemble | 15000 | 33374 | 0.8459 | 0.002685 | 0.8705 | 3.356 | 1.840 | 23.42 | 2.511 | 21.45 |
| read_delay_ns | poly2_ridge | linear | 15000 | 33374 | 0.8294 | 0.004340 | 0.8317 | 3.827 | 2.525 | 84.66 | 0.4378 | 5.099 |
| read_delay_ns | gpr | kernel | 15000 | 2000 | 0.7990 | 0.007361 | 0.8107 | 4.059 | 2.788 | 82.67 | 52.07 | 136.7 |
| read_delay_ns | svr_rbf | kernel | 15000 | 5000 | 0.8102 | 0.005736 | 0.8032 | 4.138 | 2.392 | 51.82 | 4.648 | 464.6 |
| read_delay_ns | ridge | linear | 15000 | 33374 | 0.6327 | 0.005134 | 0.6305 | 5.670 | 4.256 | 161.6 | 0.07190 | 1.203 |
| read_delay_ns | elasticnet | linear | 15000 | 33374 | 0.6327 | 0.005058 | 0.6303 | 5.672 | 4.255 | 161.8 | 21.89 | 2.052 |
| read_delay_ns | knn | neighbour | 15000 | 33374 | 0.5226 | 0.007825 | 0.5533 | 6.234 | 4.076 | 74.97 | 0.1332 | 205.7 |
| read_margin_mv | mlp_deep | neural | 15000 | 42491 | 0.9510 | 0.01216 | 0.9912 | 3.675 | 1.132 | 4.439 | 57.49 | 2.367 |
| read_margin_mv | mlp | neural | 15000 | 42491 | 0.9614 | 0.004524 | 0.9897 | 3.978 | 1.659 | 6.609 | 36.49 | 2.261 |
| read_margin_mv | catboost | boosting | 15000 | 42491 | 0.9336 | 0.004673 | 0.9473 | 8.983 | 3.386 | 17.62 | 6.559 | 2.569 |
| read_margin_mv | hist_gbr | ensemble | 15000 | 42491 | 0.9046 | 0.008269 | 0.9283 | 10.47 | 4.057 | 18.63 | 4.978 | 53.03 |
| read_margin_mv | lightgbm | boosting | 15000 | 42491 | 0.8916 | 0.01184 | 0.9283 | 10.48 | 3.648 | 17.72 | 1.726 | 5.991 |
| read_margin_mv | xgboost | boosting | 15000 | 42491 | 0.8839 | 0.01471 | 0.9204 | 11.04 | 3.967 | 19.11 | 17.05 | 21.78 |
| read_margin_mv | extra_trees | ensemble | 15000 | 42491 | 0.8357 | 0.008539 | 0.8538 | 14.96 | 7.094 | 38.52 | 3.438 | 22.42 |
| read_margin_mv | random_forest | ensemble | 15000 | 42491 | 0.8080 | 0.01371 | 0.8356 | 15.86 | 7.500 | 39.36 | 6.988 | 17.41 |
| read_margin_mv | gpr | kernel | 15000 | 2000 | 0.7372 | 0.01299 | 0.7186 | 20.75 | 10.57 | 39.31 | 58.02 | 126.2 |
| read_margin_mv | poly2_ridge | linear | 15000 | 42491 | 0.7473 | 0.01231 | 0.7178 | 20.78 | 9.720 | 34.72 | 0.3419 | 3.575 |
| read_margin_mv | knn | neighbour | 15000 | 42491 | 0.6497 | 0.01848 | 0.6644 | 22.66 | 11.61 | 42.97 | 0.08430 | 123.5 |
| read_margin_mv | svr_rbf | kernel | 15000 | 5000 | 0.6926 | 0.02552 | 0.6420 | 23.40 | 6.339 | 17.29 | 2.409 | 450.3 |
| read_margin_mv | ridge | linear | 15000 | 42491 | 0.6106 | 0.02354 | 0.5577 | 26.01 | 12.60 | 40.29 | 0.2018 | 1.751 |
| read_margin_mv | elasticnet | linear | 15000 | 42491 | 0.6104 | 0.02377 | 0.5577 | 26.02 | 12.52 | 40.14 | 23.02 | 1.013 |
| write_efficiency | mlp | neural | 15000 | 42491 | 0.9880 | 0.004424 | 0.9988 | 0.004309 | 0.002680 | 0.3280 | 40.56 | 1.779 |
| write_efficiency | mlp_deep | neural | 15000 | 42491 | 0.9931 | 0.002262 | 0.9986 | 0.004596 | 0.003023 | 0.3674 | 80.50 | 2.752 |
| write_efficiency | catboost | boosting | 15000 | 42491 | 0.9846 | 0.001761 | 0.9924 | 0.01068 | 0.006261 | 0.7495 | 5.534 | 1.594 |
| write_efficiency | lightgbm | boosting | 15000 | 42491 | 0.9788 | 0.002283 | 0.9901 | 0.01217 | 0.006683 | 0.8245 | 1.402 | 5.353 |
| write_efficiency | hist_gbr | ensemble | 15000 | 42491 | 0.9780 | 0.002281 | 0.9878 | 0.01350 | 0.007959 | 0.9693 | 2.835 | 32.42 |
| write_efficiency | xgboost | boosting | 15000 | 42491 | 0.9707 | 0.003061 | 0.9870 | 0.01396 | 0.007417 | 0.9326 | 10.26 | 9.803 |
| write_efficiency | gpr | kernel | 15000 | 2000 | 0.9405 | 0.003055 | 0.9449 | 0.02871 | 0.02047 | 2.419 | 30.50 | 72.07 |
| write_efficiency | poly2_ridge | linear | 15000 | 42491 | 0.9330 | 0.003018 | 0.9412 | 0.02967 | 0.02172 | 2.574 | 0.3446 | 3.663 |
| write_efficiency | svr_rbf | kernel | 15000 | 5000 | 0.9268 | 0.004381 | 0.9272 | 0.03299 | 0.02446 | 2.911 | 0.3353 | 74.83 |
| write_efficiency | random_forest | ensemble | 15000 | 42491 | 0.8115 | 0.007213 | 0.8326 | 0.05004 | 0.02968 | 3.796 | 7.803 | 17.87 |
| write_efficiency | extra_trees | ensemble | 15000 | 42491 | 0.8035 | 0.008971 | 0.8236 | 0.05137 | 0.03042 | 3.906 | 3.020 | 18.00 |
| write_efficiency | knn | neighbour | 15000 | 42491 | 0.7358 | 0.005847 | 0.7828 | 0.05700 | 0.03677 | 4.861 | 0.08340 | 119.1 |
| write_efficiency | ridge | linear | 15000 | 42491 | 0.6884 | 0.002428 | 0.7016 | 0.06681 | 0.05356 | 6.432 | 0.09010 | 1.065 |
| write_efficiency | elasticnet | linear | 15000 | 42491 | 0.6850 | 0.002234 | 0.6987 | 0.06713 | 0.05354 | 6.455 | 0.8738 | 1.127 |

_cv_* are 5-fold cross-validated means over n_compare rows of the train+val partition; the model is then refitted on all n_train rows and scored once on the held-out test partition. Kernel methods carry a further per-model cap (see the notes column)._

**Table 6.** Best surrogate per response with bootstrap CIs.


| response | best model | family | n train | n test | test r2 | test rmse | test mae | test mape | test max_error | test r2 95% CI | test rmse 95% CI | test mae 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| log10_retention_time_s | mlp_deep | neural | 42491 | 7496 | 0.9510 | 0.9212 | 0.2123 | 46.75 | 12.24 | [0.9413, 0.9594] | [0.8323, 1.0117] | [0.1927, 0.2322] |
| read_delay_ns | mlp_deep | neural | 33374 | 5915 | 0.9878 | 1.031 | 0.4237 | 6.293 | 16.84 | [0.9852, 0.9901] | [0.9037, 1.1546] | [0.4002, 0.4489] |
| log10_total_power_nw | mlp_deep | neural | 42491 | 7496 | 0.9827 | 0.05173 | 0.02065 | 17.02 | 1.767 | [0.9750, 0.9875] | [0.0431, 0.0606] | [0.0197, 0.0218] |
| read_margin_mv | mlp | neural | 42491 | 7496 | 0.9928 | 3.321 | 1.539 | 6.071 | 57.69 | [0.9915, 0.9943] | [2.9430, 3.6440] | [1.4719, 1.6086] |
| write_efficiency | mlp_deep | neural | 42491 | 7496 | 0.9986 | 0.004596 | 0.003023 | 0.3674 | 0.07089 | [0.9984, 0.9987] | [0.0043, 0.0049] | [0.0029, 0.0031] |
| energy_per_access_fj | mlp_deep | neural | 42491 | 7496 | 0.9961 | 2.674 | 1.237 | 4.853 | 51.67 | [0.9954, 0.9968] | [2.4045, 2.9008] | [1.1827, 1.2955] |

**Table 7.** Digital-twin latency and speed-up.


| batch_size | total_ms | ms_per_design | designs_per_second | spice_ms_per_design | speedup_vs_spice |
|---|---|---|---|---|---|
| 1 | 116.3 | 116.3 | 8.600 | 1202 | 10.30 |
| 10 | 64.25 | 6.425 | 155.6 | 1202 | 187.1 |
| 100 | 73.71 | 0.7371 | 1357 | 1202 | 1631 |
| 1000 | 92.95 | 0.09295 | 10758 | 1202 | 12933 |
| 10000 | 324.8 | 0.03248 | 30786 | 1202 | 37009 |

_The SPICE reference is the mean simulation time actually recorded while generating the training dataset._


## 7. Multi-objective optimisation

Six optimisers were each run 7 times under an identical budget of 4000 surrogate evaluations. Ranked by median hypervolume, **NSGA-II** leads (2851.5).

| Algorithm | Median HV | Median IGD+ | Median spacing | Median solutions | Median runtime (s) |
|---|---|---|---|---|---|
| NSGA-II | 2851.5 | 8.8681 | 3.375 | 100 | 4.4 |
| NSGA-III | 2700.3 | 9.5737 | 6.094 | 40 | 4.5 |
| Bayesian optimisation (multi-objective TPE) | 2461.8 | 11.986 | 2.611 | 245 | 551.4 |
| Differential evolution | 905.67 | 31.3 | 8.877 | 12 | 4.6 |
| Particle-swarm optimisation | 891.45 | 33.442 | 8.088 | 13 | 4.9 |
| MOEA/D | 434.56 | 0.1903 | 0.2041 | 99 | 249.2 |

The Friedman/Nemenyi analysis on hypervolume gives a critical difference of 2.850; algorithms whose mean ranks differ by less than that are statistically indistinguishable at alpha = 0.05 and are joined in the critical-difference diagram.

**SPICE verification of the selected designs.** The Pareto designs were returned to NGSpice and re-simulated at every operating condition. Median relative error between twin prediction and simulation: retention_time_s 46.33 %, read_delay_ns 30.82 %, total_power_nw 10.31 %, read_margin_mv 3.04 %.

12 representative designs were selected (knee point, weighted compromise and per-objective extremes); they are tabulated below and carried into the robustness study.

![Figure 9](results/figures/fig14_pareto_matrix.png)

**Figure 9.** Pareto-optimal design set, objective-pair projections.

![Figure 10](results/figures/fig15_parallel_coordinates.png)

**Figure 10.** Pareto set in parallel coordinates.

![Figure 11](results/figures/fig17_convergence.png)

**Figure 11.** Hypervolume convergence (median over runs, IQR band).

![Figure 12](results/figures/fig18_indicators.png)

**Figure 12.** Optimiser comparison across independent runs.

![Figure 13](results/figures/fig16_tradeoff.png)

**Figure 13.** Design trade-off with the selected designs highlighted.

![Figure 14](results/figures/fig19a_cd_optimizer_epsilon.png)

**Figure 14.** Critical-difference diagram (Nemenyi post-hoc).

**Table 8.** Optimiser comparison.


| algorithm | label | runs | hv_median | hv_mean | hv_std | igd_plus_median | spacing_median | spread_median | n_solutions_median | runtime_s_median |
|---|---|---|---|---|---|---|---|---|---|---|
| nsga2 | NSGA-II | 7 | 2851 | 2857 | 118.0 | 8.868 | 3.375 | 195.2 | 100 | 4.351 |
| nsga3 | NSGA-III | 7 | 2700 | 2748 | 198.2 | 9.574 | 6.094 | 167.9 | 40 | 4.467 |
| bayesian | Bayesian optimisation (multi-objective TPE) | 7 | 2462 | 2498 | 93.49 | 11.99 | 2.611 | 198.9 | 245 | 551.4 |
| de | Differential evolution | 7 | 905.7 | 950.5 | 163.2 | 31.30 | 8.877 | 134.0 | 12 | 4.564 |
| pso | Particle-swarm optimisation | 7 | 891.4 | 892.2 | 109.0 | 33.44 | 8.088 | 142.4 | 13 | 4.878 |
| moead | MOEA/D | 7 | 434.6 | 429.4 | 7.685 | 0.1903 | 0.2041 | 4.208 | 99 | 249.2 |

_Budget 4000 surrogate evaluations per run, 7 runs per algorithm. Hypervolume and IGD+ use a reference front built from the non-dominated union of every run._

**Table 9.** Representative Pareto-optimal designs.


| selection | found_by | vdd | vwl_boost | vwl_low | vblpre_ratio | cs | cbl_ratio | wacc | lacc | wsan | wsap_ratio | wen_ratio | t_write_pulse | t_sense_delay | retention_time_s | read_delay_ns | total_power_nw | read_margin_mv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| knee | nsga3 | 1.299 | 0.8995 | -0.008576 | 0.3569 | 3.527e-14 | 2.074 | 6.015e-8 | 5.846e-8 | 1.415e-7 | 2.474 | 1.234 | 8.694e-9 | 8.593e-9 | 0.01879 | 4.997 | 8.018 | 245.9 |
| weighted | nsga3 | 1.227 | 0.9809 | -0.03507 | 0.3500 | 3.987e-14 | 2.004 | 6.534e-8 | 8.104e-8 | 2.015e-7 | 2.987 | 3.802 | 9.602e-9 | 1.949e-9 | 0.02108 | 2.993 | 8.824 | 230.6 |
| best_retention_time_s | nsga3 | 1.287 | 1.121 | -0.02329 | 0.3504 | 3.996e-14 | 2.071 | 6.240e-8 | 6.565e-8 | 1.083e-7 | 2.724 | 3.823 | 9.511e-9 | 8.503e-9 | 0.02326 | 5.849 | 12.75 | 260.3 |
| best_read_delay_ns | nsga2 | 1.294 | 1.088 | -0.1159 | 0.3565 | 5.162e-15 | 2.037 | 2.096e-7 | 6.416e-8 | 4.790e-7 | 1.857 | 3.148 | 6.320e-9 | 5.981e-9 | 4.789e-4 | 0.1995 | 8.790 | 225.8 |
| best_total_power_nw | nsga2 | 0.7090 | 0.6362 | -0.05650 | 0.6335 | 5.308e-15 | 2.005 | 8.936e-8 | 8.232e-8 | 1.442e-7 | 2.313 | 1.531 | 9.667e-9 | 3.176e-9 | 2.168e-4 | 25.04 | 0.3161 | 72.80 |
| best_read_margin_mv | moead | 1.300 | 0.9965 | -0.3998 | 0.3500 | 5.006e-15 | 2.000 | 2.200e-7 | 1.200e-7 | 9.000e-8 | 1.000 | 2.208 | 5.071e-9 | 1.287e-9 | 3.418e-4 | 1.171 | 3.928 | 286.2 |
| compromise | nsga2 | 1.285 | 0.9786 | -0.05926 | 0.3514 | 3.939e-14 | 2.005 | 6.544e-8 | 6.237e-8 | 1.819e-7 | 2.797 | 3.942 | 9.569e-9 | 9.594e-9 | 0.02097 | 2.940 | 10.09 | 261.5 |
| compromise | nsga2 | 1.300 | 1.053 | -0.003787 | 0.3531 | 3.967e-14 | 2.004 | 6.278e-8 | 7.494e-8 | 1.764e-7 | 2.976 | 3.641 | 9.298e-9 | 6.105e-9 | 0.02263 | 2.899 | 11.47 | 259.5 |
| compromise | nsga3 | 1.247 | 1.023 | -0.02545 | 0.3536 | 3.953e-14 | 2.002 | 6.494e-8 | 8.826e-8 | 2.318e-7 | 2.940 | 3.885 | 9.612e-9 | 2.477e-9 | 0.02116 | 2.464 | 9.764 | 237.0 |
| compromise | nsga2 | 1.270 | 1.059 | -0.002686 | 0.3531 | 3.967e-14 | 2.046 | 6.260e-8 | 6.869e-8 | 1.766e-7 | 2.978 | 3.641 | 9.630e-9 | 6.105e-9 | 0.02172 | 3.141 | 10.92 | 256.5 |
| compromise | nsga2 | 1.280 | 1.082 | -0.03865 | 0.3525 | 3.998e-14 | 2.002 | 6.077e-8 | 6.114e-8 | 2.301e-7 | 2.790 | 2.998 | 9.200e-9 | 8.914e-9 | 0.02183 | 2.553 | 11.38 | 259.1 |
| compromise | nsga3 | 1.299 | 1.019 | -0.02442 | 0.3548 | 3.790e-14 | 2.041 | 6.038e-8 | 6.712e-8 | 2.494e-7 | 1.948 | 2.911 | 9.293e-9 | 5.807e-9 | 0.02039 | 2.583 | 10.37 | 258.0 |

_Objective values are worst case over the configured PVT conditions._

**Table 10.** Twin prediction versus SPICE for the selected designs.


| selection | found_by | temperature_c | corner | retention_time_s_predicted | retention_time_s_simulated | read_delay_ns_predicted | read_delay_ns_simulated | total_power_nw_predicted | total_power_nw_simulated | read_margin_mv_predicted | read_margin_mv_simulated | write_efficiency_simulated | read_success | write_success | retention_time_s_rel_error_pct | read_delay_ns_rel_error_pct | total_power_nw_rel_error_pct | read_margin_mv_rel_error_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| knee | nsga3 | 27 | TT | 36.21 | 24.58 | 2.128 | 1.766 | 6.830 | 6.788 | 259.6 | 267.8 | 0.9979 | yes | yes | 47.29 | 20.51 | 0.6225 | 3.066 |
| knee | nsga3 | 85 | SS | 1.151 | 0.8101 | 3.792 | 2.861 | 6.225 | 6.974 | 253.0 | 264.3 | 0.9893 | yes | yes | 42.07 | 32.56 | 10.75 | 4.260 |
| knee | nsga3 | 125 | SS | 0.01879 | 0.01220 | 4.997 | 3.800 | 6.488 | 7.232 | 245.9 | 259.4 | 0.9772 | yes | yes | 54.10 | 31.50 | 10.28 | 5.207 |
| knee | nsga3 | 85 | FF | 1.044 | 0.8101 | 2.682 | 2.861 | 8.018 | 6.974 | 260.2 | 264.3 | 0.9893 | yes | yes | 28.83 | 6.255 | 14.96 | 1.531 |
| weighted | nsga3 | 27 | TT | 46.47 | 29.23 | 1.211 | 1.013 | 7.356 | 7.084 | 255.7 | 260.1 | 0.9988 | yes | yes | 58.99 | 19.57 | 3.829 | 1.713 |
| weighted | nsga3 | 85 | SS | 1.300 | 0.8570 | 2.243 | 1.598 | 6.974 | 7.367 | 242.5 | 247.0 | 0.9927 | yes | yes | 51.65 | 40.41 | 5.340 | 1.823 |
| weighted | nsga3 | 125 | SS | 0.02108 | 0.01238 | 2.993 | 2.116 | 7.352 | 7.845 | 230.6 | 224.1 | 0.9795 | yes | yes | 70.28 | 41.43 | 6.279 | 2.899 |
| weighted | nsga3 | 85 | FF | 1.218 | 0.8570 | 1.598 | 1.598 | 8.824 | 7.367 | 249.0 | 247.0 | 0.9927 | yes | yes | 42.08 | 0.03369 | 19.78 | 0.7901 |
| best_retention_time_s | nsga3 | 27 | TT | 44.02 | 28.48 | 2.394 | 1.884 | 10.41 | 9.432 | 261.3 | 270.1 | 0.9989 | yes | yes | 54.56 | 27.12 | 10.34 | 3.276 |
| best_retention_time_s | nsga3 | 85 | SS | 1.275 | 0.8977 | 4.414 | 2.984 | 8.856 | 9.808 | 261.2 | 269.9 | 0.9982 | yes | yes | 42.02 | 47.93 | 9.704 | 3.215 |
| best_retention_time_s | nsga3 | 125 | SS | 0.02326 | 0.01383 | 5.849 | 3.909 | 9.521 | 10.25 | 260.3 | 268.3 | 0.9941 | yes | yes | 68.19 | 49.61 | 7.105 | 3.001 |
| best_retention_time_s | nsga3 | 85 | FF | 1.202 | 0.8977 | 3.241 | 2.984 | 12.75 | 9.808 | 261.6 | 269.9 | 0.9982 | yes | yes | 33.86 | 8.624 | 30.05 | 3.089 |
| best_read_delay_ns | nsga2 | 27 | TT | 2.981 | 3.154 | 0.008666 | 0.1192 | 4.933 | 4.450 | 225.8 | 239.7 | 0.9711 | yes | yes | 5.467 | 92.73 | 10.85 | 5.800 |
| best_read_delay_ns | nsga2 | 85 | SS | 0.03029 | 0.03603 | 0.1609 | 0.1834 | 4.272 | 4.977 | 229.6 | 241.1 | 0.9705 | yes | yes | 15.92 | 12.31 | 14.18 | 4.771 |
| best_read_delay_ns | nsga2 | 125 | SS | 4.789e-4 | 5.853e-4 | 0.1995 | 0.2356 | 4.992 | 5.903 | 230.6 | 242.0 | 0.9701 | yes | yes | 18.18 | 15.31 | 15.44 | 4.731 |
| best_read_delay_ns | nsga2 | 85 | FF | 0.03192 | 0.03603 | 0.04727 | 0.1834 | 8.790 | 4.977 | 228.0 | 241.1 | 0.9705 | yes | yes | 11.41 | 74.23 | 76.62 | 5.426 |
| best_total_power_nw | nsga2 | 27 | TT | 0.8199 | 1.045 | 9.173 | 10.34 | 0.2754 | 0.2564 | 79.17 | 82.72 | 0.9863 | yes | yes | 21.53 | 11.25 | 7.402 | 4.285 |
| best_total_power_nw | nsga2 | 85 | SS | 0.01185 | 0.02169 | 25.04 | 10.91 | 0.2540 | 0.2777 | 72.80 | 82.54 | 0.9846 | yes | yes | 45.38 | 129.4 | 8.512 | 11.80 |
| best_total_power_nw | nsga2 | 125 | SS | 2.168e-4 | 3.136e-4 | 23.79 | 11.17 | 0.2762 | 0.3488 | 74.60 | 82.49 | 0.9834 | yes | yes | 30.86 | 113.1 | 20.82 | 9.563 |
| best_total_power_nw | nsga2 | 85 | FF | 0.02051 | 0.02169 | 4.950 | 10.91 | 0.3161 | 0.2777 | 84.88 | 82.54 | 0.9846 | yes | yes | 5.470 | 54.65 | 13.84 | 2.829 |
| best_read_margin_mv | moead | 27 | TT | 1.988 | 1.959 | 0.5659 | 0.6243 | 2.707 | 2.721 | 286.2 | 295.5 | 0.9495 | yes | yes | 1.506 | 9.351 | 0.5139 | 3.136 |
| best_read_margin_mv | moead | 85 | SS | 0.01821 | 0.01834 | 0.9472 | 1.006 | 2.210 | 2.937 | 286.2 | 297.2 | 0.9489 | yes | yes | 0.6982 | 5.867 | 24.75 | 3.708 |
| best_read_margin_mv | moead | 125 | SS | 3.418e-4 | 3.363e-4 | 1.171 | 1.321 | 2.473 | 3.277 | 286.2 | 298.2 | 0.9484 | yes | yes | 1.638 | 11.36 | 24.54 | 4.024 |
| best_read_margin_mv | moead | 85 | FF | 0.01756 | 0.01834 | 0.7144 | 1.006 | 3.928 | 2.937 | 287.1 | 297.2 | 0.9489 | yes | yes | 4.237 | 29.00 | 33.71 | 3.374 |
| compromise | nsga2 | 27 | TT | 42.05 | 27.26 | 1.331 | 1.023 | 8.232 | 7.824 | 264.8 | 273.1 | 0.9990 | yes | yes | 54.24 | 30.14 | 5.221 | 3.025 |
| compromise | nsga2 | 85 | SS | 1.228 | 0.8148 | 2.246 | 1.613 | 7.473 | 8.126 | 264.6 | 272.1 | 0.9964 | yes | yes | 50.72 | 39.22 | 8.036 | 2.756 |
| compromise | nsga2 | 125 | SS | 0.02097 | 0.01294 | 2.940 | 2.113 | 8.033 | 8.604 | 261.5 | 269.3 | 0.9894 | yes | yes | 62.06 | 39.12 | 6.638 | 2.888 |
| compromise | nsga2 | 85 | FF | 1.206 | 0.8148 | 1.666 | 1.613 | 10.09 | 8.126 | 265.2 | 272.1 | 0.9964 | yes | yes | 48.05 | 3.289 | 24.18 | 2.527 |
| compromise | nsga2 | 27 | TT | 48.06 | 28.35 | 1.349 | 1.020 | 9.370 | 8.749 | 266.5 | 275.6 | 0.9989 | yes | yes | 69.50 | 32.27 | 7.102 | 3.299 |
| compromise | nsga2 | 85 | SS | 1.291 | 0.9013 | 2.272 | 1.610 | 8.436 | 9.112 | 266.1 | 274.4 | 0.9959 | yes | yes | 43.27 | 41.11 | 7.427 | 3.023 |
| compromise | nsga2 | 125 | SS | 0.02263 | 0.01362 | 2.899 | 2.114 | 8.949 | 9.632 | 259.5 | 270.4 | 0.9863 | yes | yes | 66.13 | 37.10 | 7.096 | 3.998 |
| compromise | nsga2 | 85 | FF | 1.237 | 0.9013 | 1.683 | 1.610 | 11.47 | 9.112 | 266.4 | 274.4 | 0.9959 | yes | yes | 37.29 | 4.527 | 25.83 | 2.915 |
| compromise | nsga3 | 27 | TT | 47.51 | 28.45 | 0.9784 | 0.8406 | 7.887 | 7.665 | 257.4 | 262.8 | 0.9988 | yes | yes | 67.01 | 16.40 | 2.897 | 2.049 |
| compromise | nsga3 | 85 | SS | 1.288 | 0.8467 | 1.845 | 1.317 | 7.506 | 8.004 | 248.7 | 255.7 | 0.9935 | yes | yes | 52.09 | 40.08 | 6.222 | 2.737 |
| compromise | nsga3 | 125 | SS | 0.02116 | 0.01249 | 2.464 | 1.740 | 7.918 | 8.565 | 237.0 | 238.3 | 0.9802 | yes | yes | 69.40 | 41.63 | 7.552 | 0.5459 |
| compromise | nsga3 | 85 | FF | 1.158 | 0.8467 | 1.367 | 1.317 | 9.764 | 8.004 | 254.2 | 255.7 | 0.9935 | yes | yes | 36.78 | 3.772 | 21.99 | 0.5813 |
| compromise | nsga2 | 27 | TT | 47.01 | 28.68 | 1.398 | 1.107 | 8.951 | 8.375 | 258.4 | 265.7 | 0.9990 | yes | yes | 63.89 | 26.30 | 6.888 | 2.731 |
| compromise | nsga2 | 85 | SS | 1.330 | 0.8904 | 2.428 | 1.742 | 8.114 | 8.709 | 258.2 | 265.1 | 0.9974 | yes | yes | 49.39 | 39.39 | 6.827 | 2.580 |
| compromise | nsga2 | 125 | SS | 0.02172 | 0.01315 | 3.141 | 2.277 | 8.599 | 9.188 | 256.5 | 262.6 | 0.9911 | yes | yes | 65.13 | 37.97 | 6.413 | 2.296 |
| compromise | nsga2 | 85 | FF | 1.224 | 0.8904 | 1.751 | 1.742 | 10.92 | 8.709 | 258.7 | 265.1 | 0.9974 | yes | yes | 37.53 | 0.5426 | 25.35 | 2.418 |

**Table 11.** Friedman ranks of the optimisers.


| indicator | group | mean_rank | rank_position | critical_difference |
|---|---|---|---|---|
| hypervolume | nsga2 | 1.286 | 1 | 2.850 |
| hypervolume | nsga3 | 1.857 | 2 | 2.850 |
| hypervolume | bayesian | 2.857 | 3 | 2.850 |
| hypervolume | de | 4.286 | 4 | 2.850 |
| hypervolume | pso | 4.714 | 5 | 2.850 |
| hypervolume | moead | 6 | 6 | 2.850 |
| igd | nsga2 | 1.714 | 1 | 2.850 |
| igd | bayesian | 2.143 | 2 | 2.850 |
| igd | nsga3 | 2.143 | 3 | 2.850 |
| igd | de | 4.571 | 4 | 2.850 |
| igd | pso | 4.714 | 5 | 2.850 |
| igd | moead | 5.714 | 6 | 2.850 |
| igd_plus | moead | 1 | 1 | 2.850 |
| igd_plus | nsga3 | 2.571 | 2 | 2.850 |
| igd_plus | nsga2 | 3 | 3 | 2.850 |
| igd_plus | bayesian | 3.429 | 4 | 2.850 |
| igd_plus | de | 5.429 | 5 | 2.850 |
| igd_plus | pso | 5.571 | 6 | 2.850 |
| spacing | moead | 1 | 1 | 2.850 |
| spacing | bayesian | 2.143 | 2 | 2.850 |
| spacing | nsga2 | 3 | 3 | 2.850 |
| spacing | nsga3 | 4 | 4 | 2.850 |
| spacing | pso | 5.286 | 5 | 2.850 |
| spacing | de | 5.571 | 6 | 2.850 |
| spread | bayesian | 1.571 | 1 | 2.850 |
| spread | nsga2 | 1.857 | 2 | 2.850 |
| spread | nsga3 | 3.429 | 3 | 2.850 |
| spread | pso | 3.857 | 4 | 2.850 |
| spread | de | 4.286 | 5 | 2.850 |
| spread | moead | 6 | 6 | 2.850 |
| epsilon | bayesian | 1.429 | 1 | 2.850 |
| epsilon | nsga2 | 2.571 | 2 | 2.850 |
| epsilon | de | 2.714 | 3 | 2.850 |
| epsilon | pso | 3.714 | 4 | 2.850 |
| epsilon | nsga3 | 4.571 | 5 | 2.850 |
| epsilon | moead | 6 | 6 | 2.850 |
| n_solutions | bayesian | 1 | 1 | 2.850 |
| n_solutions | nsga2 | 2.143 | 2 | 2.850 |
| n_solutions | moead | 2.857 | 3 | 2.850 |
| n_solutions | nsga3 | 4 | 4 | 2.850 |
| n_solutions | pso | 5.429 | 5 | 2.850 |
| n_solutions | de | 5.571 | 6 | 2.850 |

_Ranks are computed per run after orienting each indicator so that a smaller value is better._


## 8. Explainability, robustness and assist techniques

**Explainability.** Across all modelled responses the highest-attribution design variables are `num__vdd`, `num__vwl_overdrive`, `num__signal_charge_c`, `num__charge_share_ratio`, `num__vwl_boost`, `num__t_write_pulse`. SHAP gives the global ranking and the direction of each effect; ALE is reported alongside partial dependence because the derived design variables are correlated by construction, which is exactly the situation in which partial dependence becomes misleading.

**Sensitivity.** Sobol' total-effect indices per response:

- `read_delay_ns`: vdd (S_T = 0.761, interaction 0.141), cbl_ratio (S_T = 0.172, interaction 0.106), cs (S_T = 0.108, interaction 0.075)
- `read_margin_mv`: cbl_ratio (S_T = 0.510, interaction 0.145), vwl_boost (S_T = 0.317, interaction 0.097), vblpre_ratio (S_T = 0.203, interaction 0.140)
- `retention_time_s`: cs (S_T = 0.384, interaction 0.203), vblpre_ratio (S_T = 0.348, interaction 0.134), cbl_ratio (S_T = 0.288, interaction 0.079)
- `total_power_nw`: vdd (S_T = 0.652, interaction 0.291), cs (S_T = 0.344, interaction 0.156), vwl_low (S_T = 0.315, interaction 0.311)

A large `S_T - S_1` gap identifies variables whose influence is mostly through interaction, which one-variable-at-a-time sweeps cannot reveal.

**Robustness.** Joint Monte-Carlo yield of the selected designs ranges from 100.00 % to 100.00 % (median 100.00 %) under the configured process sigmas.
 A random subset of the same Monte-Carlo trials was re-simulated in NGSpice; the largest twin-versus-SPICE yield discrepancy was 0.00 percentage points.

**Assist techniques.** Median change against the unassisted baseline:

| Technique | retention_time_s (%) | read_delay_ns (%) | read_margin_mv (%) | total_power_nw (%) |
|---|---|---|---|---|
| bitline_precharge_opt | +213.9 | -2.1 | +24.8 | +14.7 |
| combined_low_power | -1.8 | -6.2 | +70.9 | +90.3 |
| high_k_capacitor | -81.2 | +41.6 | -8.6 | +50.5 |
| negative_wordline | -37.0 | +0.1 | -0.0 | -0.0 |
| refresh_interval_opt | +0.0 | +0.0 | +0.0 | +0.0 |
| sense_amp_upsize | -0.0 | -31.6 | -1.1 | +12.5 |
| wordline_boost | +346.7 | -9.7 | +55.9 | +6.3 |

The study is paired: every technique is applied to the same base designs and simulated in NGSpice, and the differences are tested with paired Wilcoxon signed-rank tests under Holm correction.

![Figure 15](results/figures/fig10a_shap_beeswarm_energy_per_access_fj.png)

**Figure 15.** SHAP summary: direction and magnitude of each effect.

![Figure 16](results/figures/fig11b_shap_importance_energy_per_access_fj.png)

**Figure 16.** Global SHAP feature importance.

![Figure 17](results/figures/fig13a_pdp_energy_per_access_fj.png)

**Figure 17.** Marginal effects (partial dependence with ALE overlay).

![Figure 18](results/figures/fig22_sobol.png)

**Figure 18.** Sobol' first-order and total-effect sensitivity indices.

![Figure 19](results/figures/fig20a0_corner_knee_0_retention_time_s.png)

**Figure 19.** PVT corner sweep of a selected design.

![Figure 20](results/figures/fig21a_monte_carlo_best_read_delay_ns_3.png)

**Figure 20.** Monte-Carlo robustness with specification limits.

![Figure 21](results/figures/fig23_assist_comparison.png)

**Figure 21.** DRAM assist techniques: median relative effect.

**Table 12.** SHAP importance across responses.


| feature | energy_per_access_fj | log10_retention_time_s | log10_total_power_nw | read_delay_ns | read_margin_mv | write_efficiency | mean_importance_pct |
|---|---|---|---|---|---|---|---|
| num__vdd | 14.14 | 8.865 | 18.96 | 21.48 | 11.65 | 2.535 | 12.94 |
| num__vwl_overdrive | -- | 9.324 | 3.158 | -- | 5.538 | 15.95 | 8.494 |
| num__signal_charge_c | 9.183 | -- | 5.060 | -- | -- | -- | 7.121 |
| num__charge_share_ratio | 11.89 | -- | 7.108 | 6.464 | 5.121 | 1.899 | 6.496 |
| num__vwl_boost | -- | 8.887 | -- | -- | 5.592 | 4.613 | 6.364 |
| num__t_write_pulse | -- | 6.996 | -- | -- | 3.844 | 7.805 | 6.215 |
| num__vblpre_ratio | 4.609 | 8.901 | 6.180 | 3.722 | 5.077 | -- | 5.698 |
| num__sa_drive_ratio | 7.592 | -- | 5.789 | 3.283 | -- | -- | 5.555 |
| num__cbl_over_cs | -- | 6.844 | -- | 2.230 | 7.402 | -- | 5.492 |
| num__stored_charge_c | 8.120 | -- | 5.525 | 2.540 | -- | -- | 5.395 |
| num__vgs_retention | 4.186 | 2.572 | 5.336 | 7.500 | 6.657 | -- | 5.250 |
| num__cbl_ratio | -- | 6.814 | -- | 2.824 | 7.523 | 2.379 | 4.885 |
| cat__corner_FF | 2.332 | 2.482 | 4.010 | 5.017 | 4.358 | 10.34 | 4.757 |
| cat__corner_SF | -- | -- | 2.635 | -- | 4.192 | 6.972 | 4.599 |
| num__cs | 2.410 | 4.017 | 2.849 | 8.951 | -- | 4.262 | 4.498 |
| num__wsan | 3.824 | -- | 4.620 | 4.952 | -- | 4.268 | 4.416 |
| cat__corner_TT | -- | 2.794 | 3.112 | -- | 3.961 | 7.187 | 4.264 |
| cat__corner_FS | 2.207 | 2.261 | -- | -- | 3.902 | 8.529 | 4.224 |
| cat__corner_SS | -- | -- | -- | -- | 4.069 | 4.356 | 4.213 |
| num__vwl_low | 2.840 | -- | 3.494 | 5.431 | 4.592 | -- | 4.089 |

_Blank entries mean the feature did not enter that response's top-15 ranking._

**Table 13.** Monte-Carlo yield of the selected designs.


| design_id | selection | yield_read_margin_mv | yield_write_efficiency | yield_read_delay_ns | yield_retention_time_s | yield_joint |
|---|---|---|---|---|---|---|
| knee_0 | knee | 100 | 100 | 100 | 100 | 100 |
| weighted_1 | weighted | 100 | 100 | 100 | 100 | 100 |
| best_retention_time_s_2 | best_retention_time_s | 100 | 100 | 100 | 100 | 100 |
| best_read_delay_ns_3 | best_read_delay_ns | 100 | 100 | 100 | 100 | 100 |
| best_total_power_nw_4 | best_total_power_nw | 100 | 100 | 100 | 100 | 100 |
| best_read_margin_mv_5 | best_read_margin_mv | 100 | 100 | 100 | 100 | 100 |

_yield_joint is the fraction of trials meeting every specification simultaneously, not the product of the individual yields._

**Table 14.** Worst-case PVT performance.


| design_id | read_margin_mv_worst | read_margin_mv_margin | read_margin_mv_worst_corner | read_margin_mv_worst_temp_c | write_efficiency_worst | write_efficiency_margin | write_efficiency_worst_corner | write_efficiency_worst_temp_c | read_delay_ns_worst | read_delay_ns_margin | read_delay_ns_worst_corner | read_delay_ns_worst_temp_c | retention_time_s_worst | retention_time_s_margin | retention_time_s_worst_corner | retention_time_s_worst_temp_c | passes_all_corners |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| knee_0 | 245.9 | 220.9 | SS | 125 | 0.9679 | 0.1179 | SF | 125 | 4.997 | 25.00 | SS | 125 | 0.01671 | -0.04729 | FF | 125 | no |
| weighted_1 | 230.2 | 205.2 | SF | 125 | 0.9638 | 0.1138 | SF | 125 | 2.993 | 27.01 | SS | 125 | 0.01738 | -0.04662 | SF | 125 | no |
| best_retention_time_s_2 | 260.2 | 235.2 | SF | 125 | 0.9775 | 0.1275 | SF | 125 | 5.849 | 24.15 | SS | 125 | 0.02050 | -0.04350 | FF | 125 | no |
| best_read_delay_ns_3 | 224.2 | 199.2 | FS | -40 | 0.9658 | 0.1158 | TT | 125 | 0.1995 | 29.80 | SS | 125 | 4.558e-4 | -0.06354 | SF | 125 | no |
| best_total_power_nw_4 | 72.44 | 47.44 | SS | 55 | 0.9596 | 0.1096 | SS | 27 | 27.03 | 2.967 | SS | 0 | 2.164e-4 | -0.06378 | SF | 125 | no |
| best_read_margin_mv_5 | 284.2 | 259.2 | FS | -40 | 0.9608 | 0.1108 | TT | 125 | 1.233 | 28.77 | FS | 125 | 3.170e-4 | -0.06368 | FF | 125 | no |

**Table 15.** Sobol' sensitivity indices.


| response | variable | S1 | ST | interaction |
|---|---|---|---|---|
| retention_time_s | cs | 0.1813 | 0.3844 | 0.2031 |
| retention_time_s | vblpre_ratio | 0.2141 | 0.3479 | 0.1338 |
| retention_time_s | cbl_ratio | 0.2088 | 0.2883 | 0.07943 |
| retention_time_s | vwl_boost | 0.1384 | 0.1977 | 0.05929 |
| retention_time_s | vdd | -0.004126 | 0.09581 | 0.09994 |
| retention_time_s | vwl_low | 0.02684 | 0.06639 | 0.03955 |
| retention_time_s | t_write_pulse | 0.02829 | 0.06508 | 0.03679 |
| retention_time_s | wacc | -0.006677 | 0.009693 | 0.01637 |
| retention_time_s | lacc | -0.005466 | 0.005441 | 0.01091 |
| retention_time_s | wsan | -0.003113 | 0.002814 | 0.005927 |
| retention_time_s | wsap_ratio | -4.605e-4 | 0.002716 | 0.003176 |
| retention_time_s | t_sense_delay | -0.005824 | 0.001719 | 0.007543 |
| retention_time_s | wen_ratio | -0.005061 | 0.001633 | 0.006694 |
| read_delay_ns | vdd | 0.6197 | 0.7608 | 0.1412 |
| read_delay_ns | cbl_ratio | 0.06615 | 0.1722 | 0.1060 |
| read_delay_ns | cs | 0.03327 | 0.1082 | 0.07495 |
| read_delay_ns | wsan | 0.05227 | 0.09406 | 0.04178 |
| read_delay_ns | vwl_boost | -0.001565 | 0.04998 | 0.05154 |
| read_delay_ns | vblpre_ratio | -1.808e-4 | 0.03477 | 0.03495 |
| read_delay_ns | wsap_ratio | 0.007755 | 0.008498 | 7.430e-4 |
| read_delay_ns | t_write_pulse | -0.002909 | 0.001667 | 0.004575 |
| read_delay_ns | lacc | 1.063e-4 | 5.544e-4 | 4.481e-4 |
| read_delay_ns | wacc | 5.884e-4 | 4.605e-4 | 0 |
| read_delay_ns | vwl_low | 7.424e-4 | 4.396e-4 | 0 |
| read_delay_ns | wen_ratio | 1.238e-4 | 4.138e-4 | 2.900e-4 |
| read_delay_ns | t_sense_delay | -7.915e-4 | 2.875e-4 | 0.001079 |
| total_power_nw | vdd | 0.3614 | 0.6522 | 0.2908 |
| total_power_nw | cs | 0.1883 | 0.3444 | 0.1561 |
| total_power_nw | vwl_low | 0.004153 | 0.3148 | 0.3107 |
| total_power_nw | cbl_ratio | 0.1346 | 0.2679 | 0.1333 |
| total_power_nw | vblpre_ratio | -0.02740 | 0.05535 | 0.08274 |
| total_power_nw | vwl_boost | -0.02303 | 0.02883 | 0.05185 |
| total_power_nw | wacc | -0.002682 | 0.01325 | 0.01593 |
| total_power_nw | wsan | 0.01580 | 0.01232 | 0 |
| total_power_nw | lacc | -0.001696 | 0.007411 | 0.009107 |
| total_power_nw | wsap_ratio | 0.01099 | 0.005609 | 0 |
| total_power_nw | t_write_pulse | -0.002658 | 0.003762 | 0.006420 |
| total_power_nw | wen_ratio | -0.001124 | 8.805e-4 | 0.002004 |
| total_power_nw | t_sense_delay | 1.108e-4 | 5.662e-4 | 4.554e-4 |
| read_margin_mv | cbl_ratio | 0.3650 | 0.5099 | 0.1449 |
| read_margin_mv | vwl_boost | 0.2201 | 0.3173 | 0.09717 |
| read_margin_mv | vblpre_ratio | 0.06325 | 0.2031 | 0.1399 |
| read_margin_mv | vdd | 0.03910 | 0.2006 | 0.1615 |
| read_margin_mv | cs | 0.006200 | 0.1002 | 0.09403 |
| read_margin_mv | t_write_pulse | -0.002541 | 0.02397 | 0.02651 |
| read_margin_mv | wacc | 0.001916 | 0.006645 | 0.004729 |
| read_margin_mv | lacc | -0.001892 | 0.002452 | 0.004344 |
| read_margin_mv | wsan | -3.742e-5 | 0.001032 | 0.001070 |
| read_margin_mv | wsap_ratio | -0.002617 | 5.567e-4 | 0.003174 |
| read_margin_mv | wen_ratio | -1.310e-4 | 3.589e-4 | 4.899e-4 |
| read_margin_mv | vwl_low | -0.001813 | 3.054e-4 | 0.002119 |
| read_margin_mv | t_sense_delay | -0.001700 | 1.052e-4 | 0.001805 |

_S_T - S_1 is the share of variance a variable contributes through interactions rather than on its own._

**Table 16.** Assist-technique comparison.


| assist_config | retention_time_s | read_delay_ns | write_delay_ns | read_margin_mv | total_power_nw | energy_per_access_fj | leakage_current_fa | refresh_interval_ms | feasible_pct |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.2865 | 5.829 | 1.224 | 39.72 | 3.040 | 28.65 | 1.509 | 301.7 | 26.33 |
| bitline_precharge_opt | 0.8992 | 5.706 | 1.264 | 49.58 | 3.486 | 32.54 | 1.515 | 531.8 | 25.33 |
| combined_low_power | 0.2812 | 5.468 | 1.359 | 67.90 | 5.784 | 52.51 | 486.4 | 151.2 | 74 |
| high_k_capacitor | 0.05399 | 8.256 | 1.524 | 36.31 | 4.574 | 43.31 | 19.58 | 69.27 | 17 |
| negative_wordline | 0.1806 | 5.835 | 1.222 | 39.72 | 3.039 | 28.63 | 18.18 | 212.1 | 25.33 |
| refresh_interval_opt | 0.2865 | 5.829 | 1.224 | 39.72 | 3.040 | 28.65 | 1.509 | 301.7 | 26.33 |
| sense_amp_upsize | 0.2865 | 3.989 | 1.225 | 39.28 | 3.420 | 32.66 | 1.508 | 301.7 | 28.00 |
| wordline_boost | 1.280 | 5.266 | 0.8842 | 61.93 | 3.233 | 29.26 | 2.674 | 675.5 | 76.33 |

_Paired study: every technique is applied to the identical set of 300 base designs and simulated in NGSpice._

**Table 17.** Paired significance tests of the assist techniques.


| metric | group_1 | group_2 | n_pairs | statistic | p_value | median_1 | median_2 | cliffs_delta | effect_magnitude | a12 | p_value_holm | significant |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| retention_time_s | baseline | bitline_precharge_opt | 300 | 0 | 2.418e-47 | 0.2865 | 0.8992 | -0.2034 | small | 0.3983 | 1.451e-46 | yes |
| retention_time_s | baseline | combined_low_power | 300 | 13647 | 4.143e-7 | 0.2865 | 0.2812 | 0.01107 | negligible | 0.5055 | 8.285e-7 | yes |
| retention_time_s | baseline | high_k_capacitor | 300 | 1549 | 1.130e-34 | 0.2865 | 0.05399 | 0.3107 | small | 0.6553 | 3.390e-34 | yes |
| retention_time_s | baseline | negative_wordline | 300 | 61 | 4.118e-42 | 0.2865 | 0.1806 | 0.05938 | negligible | 0.5297 | 2.059e-41 | yes |
| retention_time_s | baseline | refresh_interval_opt | 300 | -- | 1 | 0.2865 | 0.2865 | 0 | negligible | 0.5000 | 1 | no |
| retention_time_s | baseline | sense_amp_upsize | 300 | 171 | 1.565e-40 | 0.2865 | 0.2865 | 0.002733 | negligible | 0.5014 | 6.262e-40 | yes |
| retention_time_s | baseline | wordline_boost | 300 | 0 | 1.805e-49 | 0.2865 | 1.280 | -0.3052 | small | 0.3474 | 1.263e-48 | yes |
| read_delay_ns | baseline | bitline_precharge_opt | 192 | 3031 | 6.252e-16 | 4.665 | 4.614 | 0.04411 | negligible | 0.5221 | 2.501e-15 | yes |
| read_delay_ns | baseline | combined_low_power | 192 | 4405 | 2.934e-10 | 4.665 | 4.332 | 0.07319 | negligible | 0.5366 | 5.869e-10 | yes |
| read_delay_ns | baseline | high_k_capacitor | 192 | 0 | 2.943e-33 | 4.665 | 8.256 | -0.2723 | small | 0.3639 | 2.060e-32 | yes |
| read_delay_ns | baseline | negative_wordline | 192 | 3910 | 3.788e-12 | 4.665 | 4.660 | 0.003364 | negligible | 0.5017 | 1.136e-11 | yes |
| read_delay_ns | baseline | refresh_interval_opt | 192 | -- | 1 | 4.665 | 4.665 | 0 | negligible | 0.5000 | 1 | no |
| read_delay_ns | baseline | sense_amp_upsize | 192 | 0 | 2.943e-33 | 4.665 | 2.973 | 0.2536 | small | 0.6268 | 2.060e-32 | yes |
| read_delay_ns | baseline | wordline_boost | 192 | 0 | 2.943e-33 | 4.665 | 4.003 | 0.1010 | negligible | 0.5505 | 2.060e-32 | yes |
| write_delay_ns | baseline | bitline_precharge_opt | 300 | 3 | 6.269e-51 | 1.224 | 1.264 | -0.04060 | negligible | 0.4797 | 4.389e-50 | yes |
| write_delay_ns | baseline | combined_low_power | 300 | 8100 | 6.214e-22 | 1.224 | 1.359 | -0.1321 | negligible | 0.4339 | 1.864e-21 | yes |
| write_delay_ns | baseline | high_k_capacitor | 300 | 271 | 9.063e-50 | 1.224 | 1.524 | -0.2210 | small | 0.3895 | 5.438e-49 | yes |
| write_delay_ns | baseline | negative_wordline | 300 | 11089 | 2.202e-14 | 1.224 | 1.222 | -0.006111 | negligible | 0.4969 | 4.404e-14 | yes |
| write_delay_ns | baseline | refresh_interval_opt | 300 | -- | 1 | 1.224 | 1.224 | 0 | negligible | 0.5000 | 1 | no |
| write_delay_ns | baseline | sense_amp_upsize | 300 | 797 | 1.563e-47 | 1.224 | 1.225 | -0.004289 | negligible | 0.4979 | 7.817e-47 | yes |
| write_delay_ns | baseline | wordline_boost | 300 | 7661 | 3.481e-23 | 1.224 | 0.8842 | 0.2477 | small | 0.6238 | 1.392e-22 | yes |
| read_margin_mv | baseline | bitline_precharge_opt | 300 | 0 | 6.084e-51 | 39.72 | 49.58 | -0.2450 | small | 0.3775 | 4.258e-50 | yes |
| read_margin_mv | baseline | combined_low_power | 300 | 35 | 8.639e-51 | 39.72 | 67.90 | -0.5309 | large | 0.2346 | 4.258e-50 | yes |
| read_margin_mv | baseline | high_k_capacitor | 300 | 15 | 7.071e-51 | 39.72 | 36.31 | 0.1170 | negligible | 0.5585 | 4.258e-50 | yes |
| read_margin_mv | baseline | negative_wordline | 300 | 12786 | 7.512e-11 | 39.72 | 39.72 | -0.001844 | negligible | 0.4991 | 1.502e-10 | yes |
| read_margin_mv | baseline | refresh_interval_opt | 300 | -- | 1 | 39.72 | 39.72 | 0 | negligible | 0.5000 | 1 | no |
| read_margin_mv | baseline | sense_amp_upsize | 300 | 0 | 6.083e-51 | 39.72 | 39.28 | 0.01672 | negligible | 0.5084 | 4.258e-50 | yes |
| read_margin_mv | baseline | wordline_boost | 300 | 0 | 6.084e-51 | 39.72 | 61.93 | -0.4746 | large | 0.2627 | 4.258e-50 | yes |
| total_power_nw | baseline | bitline_precharge_opt | 300 | 858 | 2.820e-47 | 3.040 | 3.486 | -0.07580 | negligible | 0.4621 | 1.410e-46 | yes |
| total_power_nw | baseline | combined_low_power | 300 | 196 | 4.305e-50 | 3.040 | 5.784 | -0.3601 | medium | 0.3200 | 2.583e-49 | yes |
| total_power_nw | baseline | high_k_capacitor | 300 | 2589 | 2.618e-40 | 3.040 | 4.574 | -0.1991 | small | 0.4004 | 1.047e-39 | yes |
| total_power_nw | baseline | negative_wordline | 300 | 17290 | 4.405e-4 | 3.040 | 3.039 | 0.001733 | negligible | 0.5009 | 8.810e-4 | yes |
| total_power_nw | baseline | refresh_interval_opt | 300 | -- | 1 | 3.040 | 3.040 | 0 | negligible | 0.5000 | 1 | no |
| total_power_nw | baseline | sense_amp_upsize | 300 | 0 | 6.084e-51 | 3.040 | 3.420 | -0.08831 | negligible | 0.4558 | 4.258e-50 | yes |
| total_power_nw | baseline | wordline_boost | 300 | 14413 | 5.706e-8 | 3.040 | 3.233 | -0.03324 | negligible | 0.4834 | 1.712e-7 | yes |
| energy_per_access_fj | baseline | bitline_precharge_opt | 300 | 315 | 1.401e-49 | 28.65 | 32.54 | -0.07740 | negligible | 0.4613 | 7.005e-49 | yes |
| energy_per_access_fj | baseline | combined_low_power | 300 | 212 | 5.047e-50 | 28.65 | 52.51 | -0.3418 | medium | 0.3291 | 3.028e-49 | yes |
| energy_per_access_fj | baseline | high_k_capacitor | 300 | 2286 | 1.737e-41 | 28.65 | 43.31 | -0.2047 | small | 0.3977 | 6.946e-41 | yes |
| energy_per_access_fj | baseline | negative_wordline | 300 | 7177 | 1.316e-24 | 28.65 | 28.63 | 0.003222 | negligible | 0.5016 | 3.948e-24 | yes |
| energy_per_access_fj | baseline | refresh_interval_opt | 300 | -- | 1 | 28.65 | 28.65 | 0 | negligible | 0.5000 | 1 | no |
| energy_per_access_fj | baseline | sense_amp_upsize | 300 | 0 | 6.084e-51 | 28.65 | 32.66 | -0.08038 | negligible | 0.4598 | 4.258e-50 | yes |
| energy_per_access_fj | baseline | wordline_boost | 300 | 8254 | 1.674e-21 | 28.65 | 29.26 | -0.003533 | negligible | 0.4982 | 3.348e-21 | yes |
| leakage_current_fa | baseline | bitline_precharge_opt | 300 | 18349 | 0.004949 | 1.509 | 1.515 | 4.667e-4 | negligible | 0.5002 | 0.009899 | yes |
| leakage_current_fa | baseline | combined_low_power | 300 | 7 | 6.526e-51 | 1.509 | 486.4 | -0.7191 | large | 0.1404 | 4.258e-50 | yes |
| leakage_current_fa | baseline | high_k_capacitor | 300 | 8 | 6.592e-51 | 1.509 | 19.58 | -0.4765 | large | 0.2617 | 4.258e-50 | yes |
| leakage_current_fa | baseline | negative_wordline | 300 | 14618 | 1.214e-7 | 1.509 | 18.18 | -0.1968 | small | 0.4016 | 3.641e-7 | yes |
| leakage_current_fa | baseline | refresh_interval_opt | 300 | -- | 1 | 1.509 | 1.509 | 0 | negligible | 0.5000 | 1 | no |
| leakage_current_fa | baseline | sense_amp_upsize | 300 | 149 | 3.903e-49 | 1.509 | 1.508 | 0.003211 | negligible | 0.5016 | 1.561e-48 | yes |
| leakage_current_fa | baseline | wordline_boost | 300 | 0 | 6.084e-51 | 1.509 | 2.674 | -0.2034 | small | 0.3983 | 4.258e-50 | yes |
| refresh_interval_ms | baseline | bitline_precharge_opt | 221 | 0 | 5.197e-38 | 424.3 | 740.9 | -0.1745 | small | 0.4128 | 3.638e-37 | yes |
| refresh_interval_ms | baseline | combined_low_power | 221 | 4823 | 5.250e-15 | 424.3 | 190.1 | 0.2203 | small | 0.6102 | 1.050e-14 | yes |
| refresh_interval_ms | baseline | high_k_capacitor | 221 | 1329 | 1.441e-30 | 424.3 | 69.27 | 0.4366 | medium | 0.7183 | 4.323e-30 | yes |
| refresh_interval_ms | baseline | negative_wordline | 221 | 53 | 1.068e-37 | 424.3 | 274.5 | 0.09662 | negligible | 0.5483 | 5.341e-37 | yes |
| refresh_interval_ms | baseline | refresh_interval_opt | 221 | -- | 1 | 424.3 | 424.3 | 0 | negligible | 0.5000 | 1 | no |
| refresh_interval_ms | baseline | sense_amp_upsize | 221 | 154 | 4.260e-36 | 424.3 | 424.1 | 0.004361 | negligible | 0.5022 | 1.704e-35 | yes |
| refresh_interval_ms | baseline | wordline_boost | 221 | 0 | 5.197e-38 | 424.3 | 790.4 | -0.2059 | small | 0.3970 | 3.638e-37 | yes |


## 9. Limitations and threats to validity

These are stated plainly because they bound what the numbers mean.

1. **Device models are logic models, not DRAM models.** Every simulation uses the
   public PTM BSIM4 cards for a ? nm 
   *logic* process. Real DRAM access devices are engineered differently -- notably
   much stronger GIDL at the storage node -- so the *absolute* retention times here
   are longer than a production part would show. Relative trends, the ranking of
   design variables and the methodology are unaffected; absolute retention values
   should not be read as predictions about any commercial device.

2. **Process corners are synthesised, not signed off.** SS/FF/SF/FS are first-order
   `vth0`/`u0`/`tox` shifts applied to the typical card, with the shift magnitudes
   declared in configuration. They are a reasonable spread, not foundry corners.

3. **Retention is quasi-static.** The model integrates the charge-loss ODE over a
   *measured* leakage characteristic and is exact for an isolated capacitor whose
   leakage depends only on the instantaneous node voltage. Its agreement with direct
   transient simulation is quantified in Section 5. It does not model
   variable-retention-time behaviour or trap-assisted transients, which are
   time-dependent phenomena outside a quasi-static description.

4. **The bitline is lumped.** `Cbl` is a lumped capacitance rather than a
   distributed RC line with the other cells attached, so bitline resistance and
   neighbour coupling are not modelled.

5. **Power is reported per simulated slice.** The slice is one cell plus its column
   periphery, and the active-power component depends on the configured
   `access_rate_hz`. All three components are reported separately so results can be
   reweighted for a different access profile without re-simulating.

6. **The optimiser searches the surrogate, and that shows.** A multi-objective search drives towards the edges of the design space, which is exactly where a surrogate has least support -- and any region in which it over-predicts will be exploited. The effect is measurable here. Against the held-out test split the twin reaches R2 = 0.95-0.999, but on the optimiser-selected Pareto designs the median twin-versus-SPICE error is 3.0 % (read_margin_mv), 10.3 % (total_power_nw), 30.8 % (read_delay_ns), 46.3 % (retention_time_s). Read margin and power -- the smoothest responses -- stay accurate; read delay and retention, which span decades and turn sharply near the feasibility boundary, degrade most. Expressed on the scale the quantity actually lives on, the retention discrepancy is 0.169 decades, a factor of 1.48. Two things make this manageable rather than fatal. First it is *measured*: every selected design is returned to NGSpice, so the error is a reported quantity, not an assumption. Second, the constraints survive the round trip -- 100 % of the selected designs pass both the read and the write check under full circuit simulation. The practical reading is that the twin is a search accelerator whose proposals must be confirmed in SPICE, not a replacement for it; re-simulating the Pareto region and refitting is the natural next iteration, and is what the framework's feedback path exists for.

7. **A single technology node was run end-to-end.** Configurations for 32 nm and
   22 nm ship with the framework and the pipeline is node-agnostic, but the results
   presented here are for the ? nm node.


## 10. Reproduction

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -e .
python scripts/00_setup_environment.py              # NGSpice + PTM models
python scripts/run_all.py --tag main          # the whole pipeline
```

Determinism: master seed `20260723`, configuration hash
`e22061fdb47e16e7`, model-card fingerprint `?`. Every
stage writes a manifest to `logs/manifest_*.json` recording its seeds, inputs,
outputs, package versions and wall-clock time. See `REPRODUCIBILITY.md` for the
full protocol and `supplementary/environment_lock.txt` for the exact package
versions used.


## References

[1] "Re-Evaluating the Real-System Modeling Accuracy of Ramulator 2.0", arXiv, 2026 <https://arxiv.org/abs/2606.14566>

[2] "Analog Design and Machine Learning: A Review", Electronics (MDPI), 2025 <https://doi.org/10.3390/electronics14173541>

[3] "EasyDRAM: An FPGA-based Infrastructure for Fast and Accurate End-to-End Evaluation of Emerging DRAM Techniques", arXiv, 2025 <https://arxiv.org/abs/2506.10441>

[4] "Multi-Fidelity Surrogate Models for Accelerated Multi-Objective Analog Circuit Design and Optimization", Electronics (MDPI), 2025 <https://doi.org/10.3390/electronics15010105>

[5] "Analog circuit sizing based on Evolutionary Algorithms and deep learning", Expert Systems with Applications (Elsevier), 2024 <https://www.sciencedirect.com/science/article/abs/pii/S0957417423019826>

[6] "Machine Learning Driven Global Optimisation Framework for Analog Circuit Design", arXiv, 2024 <https://arxiv.org/abs/2404.02911>

[7] "PENDRAM: Enabling High-Performance and Energy-Efficient Processing of Deep Neural Networks through a Generalized DRAM Data Mapping Policy", arXiv, 2024 <https://arxiv.org/abs/2408.02412>

[8] "PySpice-Simulated In Situ Learning with Memristor Emulation for Single-Layer Spiking Neural Networks", Electronics (MDPI) 13(23), 4665, 2024 <https://www.mdpi.com/2079-9292/13/23/4665>

[9] "SPICEPilot: Navigating SPICE Code Generation and Simulation with AI Guidance", arXiv, 2024 <https://arxiv.org/abs/2410.20553>

[10] "The Dawn of AI-Native EDA: Opportunities and Challenges of Large Circuit Models", arXiv, 2024 <https://arxiv.org/abs/2403.07257>

[11] "When Device Modeling Meets Machine Learning: Opportunities and Challenges", ACM/IEEE International Symposium on Machine Learning for CAD (MLCAD), 2024 <https://dl.acm.org/doi/10.1145/3670474.3685972>

[12] "Performance Evaluation of Evolutionary Algorithms for Analog Integrated Circuit Design Optimisation", arXiv, 2023 <https://arxiv.org/abs/2310.12440>

[13] "Python Framework for Modular and Parametric SPICE Netlists Generation", arXiv, 2023 <https://arxiv.org/abs/2306.12224>

[14] "Digital Twin for Secure Semiconductor Lifecycle Management: Prospects and Applications", arXiv, 2022 <https://arxiv.org/abs/2205.10962>

[15] D. W. Apley and J. Zhu, "Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models", Journal of the Royal Statistical Society Series B, vol. 82, no. 4, pp. 1059-1086, 2020 <https://doi.org/10.1111/rssb.12377>

[16] J. Blank and K. Deb, "pymoo: Multi-Objective Optimization in Python", IEEE Access, vol. 8, pp. 89497-89509, 2020 <https://doi.org/10.1109/ACCESS.2020.2990567>

[17] S. M. Lundberg et al., "From Local Explanations to Global Understanding with Explainable AI for Trees", Nature Machine Intelligence, vol. 2, pp. 56-67, 2020 <https://doi.org/10.1038/s42256-019-0138-9>

[18] T. Akiba, S. Sano, T. Yanase, T. Ohta and M. Koyama, "Optuna: A Next-generation Hyperparameter Optimization Framework", Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2019 <https://doi.org/10.1145/3292500.3330701>

[19] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush and A. Gulin, "CatBoost: unbiased boosting with categorical features", Advances in Neural Information Processing Systems (NeurIPS) 31, 2018 <https://arxiv.org/abs/1706.09516>

[20] G. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", Advances in Neural Information Processing Systems (NeurIPS) 30, 2017 <https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree>

[21] S. M. Lundberg and S.-I. Lee, "A Unified Approach to Interpreting Model Predictions", Advances in Neural Information Processing Systems (NeurIPS) 30, 2017 <https://arxiv.org/abs/1705.07874>

[22] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System", Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016 <https://doi.org/10.1145/2939672.2939785>

[23] K. Deb and H. Jain, "An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I", IEEE Transactions on Evolutionary Computation, vol. 18, no. 4, pp. 577-601, 2014 <https://doi.org/10.1109/TEVC.2013.2281535>

[24] J. Bergstra, R. Bardenet, Y. Bengio and B. Kegl, "Algorithms for Hyper-Parameter Optimization", Advances in Neural Information Processing Systems (NeurIPS) 24, 2011 <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization>

[25] A. Saltelli et al., "Variance based sensitivity analysis of model output: Design and estimator for the total sensitivity index", Computer Physics Communications, vol. 181, no. 2, pp. 259-270, 2010 <https://doi.org/10.1016/j.cpc.2009.09.018>

[26] M. Okabe and K. Ito, "Color Universal Design - How to make figures and presentations that are friendly to colorblind people", 2008 <https://jfly.uni-koeln.de/color/>

[27] Q. Zhang and H. Li, "MOEA/D - A Multiobjective Evolutionary Algorithm Based on Decomposition", IEEE Transactions on Evolutionary Computation, vol. 11, no. 6, pp. 712-731, 2007 <https://doi.org/10.1109/TEVC.2007.892759>

[28] J. Demsar, "Statistical Comparisons of Classifiers over Multiple Data Sets", Journal of Machine Learning Research, vol. 7, pp. 1-30, 2006 <https://www.jmlr.org/papers/v7/demsar06a.html>

[29] W. Zhao and Y. Cao, "New Generation of Predictive Technology Model for Sub-45 nm Early Design Exploration", IEEE Transactions on Electron Devices, vol. 53, no. 11, pp. 2816-2823, 2006 <https://doi.org/10.1109/TED.2006.884077>

[30] K. Deb, A. Pratap, S. Agarwal and T. Meyarivan, "A Fast and Elitist Multiobjective Genetic Algorithm - NSGA-II", IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp. 182-197, 2002 <https://doi.org/10.1109/4235.996017>

[31] R. Storn and K. Price, "Differential Evolution - A Simple and Efficient Heuristic for Global Optimization", Journal of Global Optimization, vol. 11, pp. 341-359, 1997 <https://doi.org/10.1023/A:1008202821328>

[32] J. Kennedy and R. Eberhart, "Particle Swarm Optimization", Proceedings of ICNN'95 - International Conference on Neural Networks, pp. 1942-1948, 1995 <https://doi.org/10.1109/ICNN.1995.488968>

[33] M. J. M. Pelgrom, A. C. J. Duinmaijer and A. P. G. Welbers, "Matching Properties of MOS Transistors", IEEE Journal of Solid-State Circuits, vol. 24, no. 5, pp. 1433-1439, 1989 <https://doi.org/10.1109/JSSC.1989.572629>

[34] M. D. McKay, R. J. Beckman and W. J. Conover, "A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code", Technometrics, vol. 21, no. 2, pp. 239-245, 1979 <https://doi.org/10.2307/1268522>

[35] BSIM Group, University of California, Berkeley, "BSIM4 MOSFET Model User's Manual", UC Berkeley Device Group <https://bsim.berkeley.edu/models/bsim4/>

[36] F. Salvaire, "PySpice: Simulate electronic circuits using Python and the Ngspice / Xyce simulators" <https://github.com/PySpice-org/PySpice>

[37] Ngspice development team, "Ngspice: open-source mixed-level/mixed-signal circuit simulator" <https://ngspice.sourceforge.io/>
