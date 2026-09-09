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
