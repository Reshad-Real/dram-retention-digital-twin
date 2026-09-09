**Digital-twin evaluation latency and speed-up over SPICE.**

| batch_size | total_ms | ms_per_design | designs_per_second | spice_ms_per_design | speedup_vs_spice |
|---|---|---|---|---|---|
| 1 | 116.3 | 116.3 | 8.600 | 1202 | 10.30 |
| 10 | 64.25 | 6.425 | 155.6 | 1202 | 187.1 |
| 100 | 73.71 | 0.7371 | 1357 | 1202 | 1631 |
| 1000 | 92.95 | 0.09295 | 10758 | 1202 | 12933 |
| 10000 | 324.8 | 0.03248 | 30786 | 1202 | 37009 |

_The SPICE reference is the mean simulation time actually recorded while generating the training dataset._
