**Multi-objective optimiser comparison over independent runs under an identical evaluation budget.**

| algorithm | label | runs | hv_median | hv_mean | hv_std | igd_plus_median | spacing_median | spread_median | n_solutions_median | runtime_s_median |
|---|---|---|---|---|---|---|---|---|---|---|
| nsga2 | NSGA-II | 7 | 2851 | 2857 | 118.0 | 8.868 | 3.375 | 195.2 | 100 | 4.351 |
| nsga3 | NSGA-III | 7 | 2700 | 2748 | 198.2 | 9.574 | 6.094 | 167.9 | 40 | 4.467 |
| bayesian | Bayesian optimisation (multi-objective TPE) | 7 | 2462 | 2498 | 93.49 | 11.99 | 2.611 | 198.9 | 245 | 551.4 |
| de | Differential evolution | 7 | 905.7 | 950.5 | 163.2 | 31.30 | 8.877 | 134.0 | 12 | 4.564 |
| pso | Particle-swarm optimisation | 7 | 891.4 | 892.2 | 109.0 | 33.44 | 8.088 | 142.4 | 13 | 4.878 |
| moead | MOEA/D | 7 | 434.6 | 429.4 | 7.685 | 0.1903 | 0.2041 | 4.208 | 99 | 249.2 |

_Budget 4000 surrogate evaluations per run, 7 runs per algorithm. Hypervolume and IGD+ use a reference front built from the non-dominated union of every run._
