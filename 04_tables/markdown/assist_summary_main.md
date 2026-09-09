**Median performance of each DRAM assist technique.**

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
