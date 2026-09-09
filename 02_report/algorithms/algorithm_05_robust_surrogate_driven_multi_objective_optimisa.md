### Algorithm 5: Robust surrogate-driven multi-objective optimisation

```
Algorithm 5: Robust surrogate-driven multi-objective optimisation
Implemented by: dramdt.optimization.problem.DramDesignProblem + dramdt.optimization.runner.OptimizationRunner

Require: digital twin F, decision variables x
Require: objectives f_j with senses, constraints g_c
Require: operating conditions Q, budget B, repetitions R
Ensure:  Pareto set P and indicator distributions

  1: for each algorithm a, repetition r in 1..R
  2:   seed <- DeriveSeed(sigma, a, r); budget B // identical for every algorithm
  3:   while budget remains
  4:     for each candidate population X
  5:       for each condition q in Q
  6:         R_q <- F(Decode(X, q))
  7:       end for
  8:       f_j <- min_q if f_j maximised else max_q // worst case over PVT
  9:       g_c <- normalised violation, g_c <= 0 feasible
 10:     end for
 11:   end while
 12: end for
 13: P^ref <- NonDominated(_a,rP_a,r)
 14: compute HV, IGD^+, spacing, spread per run against P^ref
 15: return P^ref, indicator distributions

Note: Worst-case reduction over $\mathcal{Q}$ turns the search into a robust design problem; equal budgets and $R$ repetitions make the comparison statistically testable.
```
