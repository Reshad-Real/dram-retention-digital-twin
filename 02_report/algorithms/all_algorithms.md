### Algorithm 1: Configuration-driven DRAM design-space sampling

```
Algorithm 1: Configuration-driven DRAM design-space sampling
Implemented by: dramdt.doe.sampling.generate_design + dramdt.config.DesignSpace.decode_row

Require: design-space specification D=(v_i,lo_i,u_i,s_i,k_i)_i=1^d
Require: derived-quantity expressions E, constants K
Require: sample count N, master seed sigma
Ensure:  physical design points p^(n)_n=1^N

  1: sigma_doe <- DeriveSeed(sigma, `design')
  2: U <- LatinHypercube(N, d, sigma_doe) // one sample per stratum in every 1-D projection
  3: record Discrepancy(U) and min_a!= b|| u_a-u_b|| // design quality is measured, not assumed
  4: for n <- 1 to N
  5:   p^(n) <- K
  6:   for i <- 1 to d
  7:     if k_i is categorical
  8:       p^(n)_i <- c_floor( U_ni |C_i| )
  9:     else if s_i = log
 10:       p^(n)_i <- exp(ln lo_i + U_ni(ln u_i - ln lo_i))
 11:     else
 12:       p^(n)_i <- lo_i + U_ni(u_i-lo_i)
 13:     end if
 14:   end for
 15:   for e in E in declaration order
 16:     p^(n)[lhs(e)] <- Eval(e, p^(n))
 17:   end for
 18: end for
 19: return p^(n)

Note: Nothing about the DRAM architecture is hard-coded: retargeting the framework means supplying a different $\mathcal{D}$.
```


---

### Algorithm 2: SPICE characterisation of one DRAM design point

```
Algorithm 2: SPICE characterisation of one DRAM design point
Implemented by: dramdt.models.cell.DramCellBuilder + dramdt.simulation.runner.SpiceRunner

Require: design point p, technology T, timing schedule S
Ensure:  raw measurement set M, leakage characteristic I_leak(V)

  1: C <- T.CardFor(p.corner) // corner-specific BSIM4 card
  2: build 1T1C cell, bitline pair, boosted precharge, write driver, sense latch
  3: append leakage replica: an identical access device whose storage node is driven by probe source V_snp
  4: S <- Schedule(p) // write -> precharge -> read -> sense
  5: analysis 1: op => quiescent supply currents
  6: analysis 2: tran over S with loose ABSTOL // muA-scale switching
  7:   measure V_SN^hold, t_write, V_BL,V_mean BL at t_SA, t_read, int i_dd dt
  8: analysis 3: dc sweep V_snp: 0 V_DD with tight ABSTOL // fA-scale leakage
  9:   I_leak(V) <- - i(V_snp) // current leaving the storage node
 10: return M, I_leak(.)

Note: Tolerances are set per analysis: one global ABSTOL cannot resolve femto-ampere leakage and micro-ampere switching in the same deck.
```


---

### Algorithm 3: Quasi-static retention time from measured leakage

```
Algorithm 3: Quasi-static retention time from measured leakage
Implemented by: dramdt.simulation.retention.quasistatic_retention

Require: leakage characteristic I_leak(V) (measured, Algorithm 2)
Require: node capacitance C_node, written level V_init
Require: precharge level V_BLpre, capacitances C_s, C_BL, sense-amplifier offset Delta V_min
Ensure:  retention time t_ret and a censoring flag

  1: V_fail <- V_BLpre + Delta V_min (C_s+C_BL)/(C_s) // charge sharing no longer resolvable
  2: if V_init <= V_fail
  3:   return t_ret <- 0 // a real observation, not a missing value
  4: end if
  5: V <- uniform grid on [V_fail, V_init]
  6: I <- Interp(I_leak, V)
  7: if min I <= eps
  8:   return t_ret <- inf (censored) // node equilibrates above V_fail
  9: end if
 10: t_ret <- int_V_fail^V_init (C_node)/(I_leak)(V) dV // trapezoidal quadrature
 11: return t_ret

Note: Exact for an isolated capacitor under quasi-static leakage; validated against direct long-window transient simulation.
```


---

### Algorithm 4: Cross-validated surrogate selection

```
Algorithm 4: Cross-validated surrogate selection
Implemented by: dramdt.surrogate.train.SurrogateTrainer

Require: dataset D with a fixed train/val/test split
Require: model zoo Z, folds K, primary metric mu
Ensure:  per-target best surrogate f^*_y and its held-out score

  1: for each target y
  2:   D_y <- (x,y) in D : y observed // drop per target, never globally
  3:   for each model m in Z
  4:     for k <- 1 to K
  5:       fit m on D_y^dev\ fold_k; score on fold_k
  6:       retain the fold score // needed by the statistical tests
  7:     end for
  8:     refit m on all of D_y^dev; score once on D_y^test
  9:   end for
 10:   f^*_y <- _m mean mu_cv(m)
 11: end for
 12: return f^*_y

Note: The test split is touched exactly once per model and never participates in selection or tuning.
```


---

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


---

### Algorithm 6: Digital-twin robustness and yield estimation

```
Algorithm 6: Digital-twin robustness and yield estimation
Implemented by: dramdt.robustness.analysis.RobustnessAnalyzer

Require: design x^*, process sigmas sigma_p
Require: specification limits (y_s, tau_s, sense), trials M
Ensure:  joint yield Y, per-spec yields, worst-case corner, SPICE check

  1: for m <- 1 to M
  2:   x^(m) <- x^* * (1 + eps),\ eps_p N(0,sigma_p^2)
  3: end for
  4: Y <- F(x^(m)) // twin: M evaluations in milliseconds
  5: Y <- (1)/(M)|m : for all s,\ y_s^(m) meets tau_s| // joint, not the product of marginals
  6: G <- evaluate x^* over corner x temperature grid; record worst case per spec
  7: re-simulate a random subset of x^(m) in NGSpice; report |Y_twin - Y_SPICE|
  8: return Y, per-spec yields, G, agreement

Note: A robustness claim that has not been checked against the simulator is not reported as a result.
```
