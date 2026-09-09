**Best surrogate per response with bootstrap confidence intervals.**

| response | best model | family | n train | n test | test r2 | test rmse | test mae | test mape | test max_error | test r2 95% CI | test rmse 95% CI | test mae 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| log10_retention_time_s | mlp_deep | neural | 42491 | 7496 | 0.9510 | 0.9212 | 0.2123 | 46.75 | 12.24 | [0.9413, 0.9594] | [0.8323, 1.0117] | [0.1927, 0.2322] |
| read_delay_ns | mlp_deep | neural | 33374 | 5915 | 0.9878 | 1.031 | 0.4237 | 6.293 | 16.84 | [0.9852, 0.9901] | [0.9037, 1.1546] | [0.4002, 0.4489] |
| log10_total_power_nw | mlp_deep | neural | 42491 | 7496 | 0.9827 | 0.05173 | 0.02065 | 17.02 | 1.767 | [0.9750, 0.9875] | [0.0431, 0.0606] | [0.0197, 0.0218] |
| read_margin_mv | mlp | neural | 42491 | 7496 | 0.9928 | 3.321 | 1.539 | 6.071 | 57.69 | [0.9915, 0.9943] | [2.9430, 3.6440] | [1.4719, 1.6086] |
| write_efficiency | mlp_deep | neural | 42491 | 7496 | 0.9986 | 0.004596 | 0.003023 | 0.3674 | 0.07089 | [0.9984, 0.9987] | [0.0043, 0.0049] | [0.0029, 0.0031] |
| energy_per_access_fj | mlp_deep | neural | 42491 | 7496 | 0.9961 | 2.674 | 1.237 | 4.853 | 51.67 | [0.9954, 0.9968] | [2.4045, 2.9008] | [1.1827, 1.2955] |
