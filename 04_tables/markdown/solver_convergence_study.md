**Sensitivity of the extracted metrics to the transient print step and integration method, relative to a 20 ps GEAR reference.**

| setting | ms_per_sample | speedup_vs_reference | read_margin_mv_median_err_pct | read_margin_mv_p95_err_pct | read_delay_ns_median_err_pct | read_delay_ns_p95_err_pct | write_delay_ns_median_err_pct | write_delay_ns_p95_err_pct | v_sn_written_v_median_err_pct | v_sn_written_v_p95_err_pct | retention_time_s_median_err_pct | retention_time_s_p95_err_pct | energy_per_access_fj_median_err_pct | energy_per_access_fj_p95_err_pct | total_power_nw_median_err_pct | total_power_nw_p95_err_pct | read_success_agreement_pct | write_success_agreement_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 ps / GEAR | 63 | 1.693 | 0.009100 | 0.1449 | 0.01530 | 0.2420 | 0.1520 | 0.7592 | 0.003800 | 0.04200 | 0.001800 | 0.4649 | 0.05650 | 0.3348 | 0.05490 | 0.3300 | 100 | 100 |
| 50 ps / TRAP | 60.90 | 1.752 | 0.003100 | 0.02840 | 0.01110 | 0.1547 | 0.03910 | 0.6964 | 7.000e-4 | 0.005800 | 3.000e-4 | 0.05170 | 0.02130 | 0.1653 | 0.02070 | 0.1591 | 100 | 100 |
| 100 ps / GEAR | 47.40 | 2.249 | 0.01990 | 0.5967 | 0.1581 | 0.8344 | 0.3680 | 1.367 | 0.01190 | 0.1842 | 0.003400 | 1.738 | 0.1905 | 0.5836 | 0.1788 | 0.5444 | 100 | 100 |

_Percentages are relative errors against the 20 ps GEAR setting over the same design points; agreement columns give the share of points whose pass/fail label is unchanged._
