**Median relative effect of each assist technique against the unassisted baseline.**

| assist_config | retention_time_s | read_delay_ns | write_delay_ns | read_margin_mv | total_power_nw | energy_per_access_fj | leakage_current_fa | refresh_interval_ms | feasible_pct |
|---|---|---|---|---|---|---|---|---|---|
| bitline_precharge_opt | 213.9 | -2.107 | 3.261 | 24.82 | 14.66 | 13.59 | 0.4531 | 76.25 | 25.33 |
| combined_low_power | -1.824 | -6.188 | 11.06 | 70.93 | 90.28 | 83.29 | 32145 | -49.89 | 74 |
| high_k_capacitor | -81.15 | 41.64 | 24.47 | -8.589 | 50.47 | 51.16 | 1198 | -77.04 | 17 |
| negative_wordline | -36.97 | 0.1076 | -0.1344 | -0.007552 | -0.03839 | -0.06395 | 1105 | -29.69 | 25.33 |
| refresh_interval_opt | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 26.33 |
| sense_amp_upsize | -0.003166 | -31.56 | 0.1046 | -1.128 | 12.51 | 14.00 | -0.003248 | -0.008343 | 28.00 |
| wordline_boost | 346.7 | -9.659 | -27.77 | 55.89 | 6.347 | 2.138 | 77.29 | 123.9 | 76.33 |

_Percentage change of the median over the same 300 paired base designs; positive means the metric increased._
