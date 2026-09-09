**Equivalence of the subprocess and PySpice shared-library execution backends.**

| sample | read_margin_mv_subprocess | read_margin_mv_shared | read_margin_mv_rel_diff_pct | read_delay_ns_subprocess | read_delay_ns_shared | read_delay_ns_rel_diff_pct | write_delay_ns_subprocess | write_delay_ns_shared | write_delay_ns_rel_diff_pct | v_sn_written_v_subprocess | v_sn_written_v_shared | v_sn_written_v_rel_diff_pct | read_energy_fj_subprocess | read_energy_fj_shared | read_energy_fj_rel_diff_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 45.07 | 45.07 | 0 | -- | -- | -- | 3.220 | 3.220 | 0 | 0.7813 | 0.7813 | 0 | 83.61 | 83.61 | 0 |
| 1 | 48.81 | 48.81 | 0 | -- | -- | -- | 1.431 | 1.431 | 0 | 0.5546 | 0.5546 | 0 | 9.483 | 9.483 | 0 |
| 2 | 48.88 | 48.88 | 0 | 3.744 | 3.744 | 0 | 0.4615 | 0.4615 | 0 | 1.098 | 1.098 | 0 | 113.2 | 113.2 | 3.610e-4 |
| 3 | 53.41 | 53.41 | 0 | 11.39 | 11.39 | 0 | 0.6892 | 0.6892 | 0 | 0.8018 | 0.8018 | 0 | 82.90 | 82.90 | 3.881e-4 |
| 4 | 51.70 | 51.70 | 0 | 29.59 | 29.59 | 0 | 0.8153 | 0.8153 | 0 | 0.9565 | 0.9565 | 0 | 63.40 | 63.40 | 5.859e-4 |
| 5 | 57.01 | 57.01 | 0 | -- | -- | -- | 0.2261 | 0.2261 | 0 | 0.8195 | 0.8195 | 0 | 21.32 | 21.32 | 0.001770 |
| 6 | 33.23 | 33.23 | 0 | -- | -- | -- | 0.5282 | 0.5282 | 0 | 0.8089 | 0.8089 | 0 | 28.99 | 28.99 | 0.001640 |
| 7 | 133.5 | 133.5 | 0 | 5.844 | 5.844 | 0 | 0.8109 | 0.8109 | 0 | 0.8160 | 0.8160 | 0 | 42.84 | 42.84 | 5.173e-4 |
