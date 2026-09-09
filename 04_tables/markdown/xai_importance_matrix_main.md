**SHAP importance (% of total attribution) of each design variable across all modelled responses.**

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
