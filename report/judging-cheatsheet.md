# Judging cheat-sheet — things you must be able to do on a whiteboard without notes

Rule from the brief: "a team that cannot explain its own model will score as though it did not
have one." Rehearse each of these out loud once.

## 1. What the register minimises (they WILL ask this)

Start from Eq. (1): $H/\hbar = \sum_i \frac{\Omega}{2}\sigma^x_i - \delta \sum_i n_i + \sum_{i<j} \frac{C_6}{r_{ij}^6} n_i n_j$.
Set $\Omega = 0$ at the end of the sweep. The remaining terms are diagonal in the $n_i \in \{0,1\}$ basis, so
every bitstring $n$ has a classical energy
$E(n) = -\delta \sum_i n_i + \sum_{i<j} \frac{C_6}{r_{ij}^6} n_i n_j$.
Laser → one global $\delta$ (same linear coefficient on every variable). Positions → couplings, all
positive, $1/r^6$. Not controllable: signs, per-atom linear terms, anything outside ~5–10 µm.
With $\delta > 0$: "excite as many atoms as possible, but never two inside a blockade disc" = maximum
independent set on a unit-disk graph.

Numbers: $C_6/\hbar = 8.657\times10^5$ rad/µs·µm⁶. At 5 µm → 55.4 rad/µs; 6 µm → 18.6; 7.5 µm → 4.9;
9.5 µm → 1.2; 10 µm → 0.87. Doubling distance divides by 64. Blockade radius $(C_6/\Omega)^{1/6}$:
6.4 µm at $\Omega = 12.57$, 7.7 µm at $\Omega = 4$.

## 2. Why one-hot kills linear terms (the derivation they may ask you to redo)

If every customer is on exactly one vehicle, $\sum_k x_{ik} = 1$. Then for any coefficient that depends
only on the customer, $\sum_k c_i x_{ik} = c_i \sum_k x_{ik} = c_i$: a constant.
Quadratic load penalty for vehicle $k$: $(L_k - T)^2$ with $L_k = \sum_i d_i x_{ik}$.
Expand: $L_k^2 - 2 T L_k + T^2 = \sum_i d_i^2 x_{ik} + 2\sum_{i<j} d_i d_j x_{ik} x_{jk} - 2T \sum_i d_i x_{ik} + T^2$
(using $x^2 = x$). Sum over $k$: the terms $\sum_k x_{ik}$ collapse to constants, leaving
$2 \sum_k \sum_{i<j} d_i d_j x_{ik} x_{jk}$ + const. Conclusion: no slack, no linear term, only
same-vehicle couplings proportional to $d_i d_j$ — it enforces balance (Jensen: $\sum_k L_k^2$ with
$\sum_k L_k$ fixed is minimised when loads are equal), not an inequality at $Q$. And it is dense, which is
why it does not embed.

## 3. Why dense couplings cannot be embedded

Couplings live in a 5–10 µm band. Put atom $i$ at the centre; a neighbour that must couple strongly sits
at ≈6 µm; two such neighbours that must NOT couple to each other need ≥ 9.5 µm between them: on a circle
of radius 6 the chord $2\cdot6\sin(\theta/2) \ge 9.5$ needs $\theta \ge 105°$, so at most three such
neighbours. A set of mutually conflicting atoms must all lie inside one blockade disc: at 5 µm
separation that is a triangle (3) or a square with a 7.07 µm diagonal whose coupling is 1/8 of the side's.
So: degree ≤ ~3 per atom for "independent" neighbours, cliques ≤ 3–4. Anything denser is approximate;
measure it. I measured it by brute-forcing the realised cost (`classical_ground_states`,
`manifold_check`): dense one-hot on M lands 57–253 % off, dense bisection on N=10–15 is 25–55 % off.

## 4. The model, in one breath

"Each customer centre defines a candidate cluster by taking the nearest customers until 0.85 Q is reached,
so every candidate is capacity-feasible by construction. One atom per candidate. Candidates that share a
customer are placed inside the blockade radius, everything else outside. The ground state is a maximum
set of disjoint compact clusters. Uncovered customers are inserted greedily into the chosen cluster with
slack and that fraction is reported. Nothing is penalised, nothing is slack: the inequality never reaches
the device." Knobs: fill 0.85, Jaccard 0.6 (thin near-duplicates, because mutually overlapping candidates
are a clique). Atoms: 2–4 / 5–6 / 6–10. All exact emulation.

Why candidates from customer centres and not a grid: more candidates = denser overlap graph = cliques =
violable edges = raw feasibility collapses (tested: 23/51 edges violable, 0 % raw feasibility).

## 5. Embedding, delta, Omega (A3/A4 questions)

* Fit: log-distance least squares (conflict pairs → 6 µm, non-conflict → ≥ 9.5 µm), L-BFGS with
  analytic gradients, restarts from the candidate centres (map geometry) and MDS; then a hard projection
  onto ≥ 5 µm / ≤ 38 µm. pulser's `AnalogDevice` validates the result (it caught a 4.99 µm pair once).
* Fidelity: rel_err and log-RMS over wanted pairs; leakage ratio; and the graph-level one — violable
  edges (conflict pairs realised below $\delta_f$) / spurious edges (non-conflicts above). Say which is
  which. Violable edges predict raw feasibility (L-0003: 8/26 violable → 22 %; L-0001: 0 → 100 %).
* $\delta_f = \sqrt{18.6 \times 1.2} = 4.7$ rad/µs: geometric mean of conflict coupling and floor.
  Check: two excited conflicting atoms gain $2\delta$ = 9.4 and pay 18.6 → forbidden; two non-conflicting
  pay ≤ 1.2 → allowed.
* $\Omega_{\max} = 4$ rad/µs → blockade radius 7.7 µm, between 6 and 9.5. Larger $\Omega$ blurs the
  conflict/non-conflict distinction; smaller slows the sweep (gap ∝ Ω) within the 6 µs limit.
* Pulse: amplitude 0→Ω→0 (three-point interpolation), detuning $-1.5\delta_f \to +\delta_f$ (start in
  the empty register). Score by feasibility rate / P(max packing) / best gap — never mean energy.
  "slow-end" puts 70 % of the sweep on the positive side because that is where the gap closes for MIS.

## 6. References, gaps, completion

* CVRP reference: Held–Karp over every subset ($2^N \times N$ states) gives the exact tour of every
  possible cluster; set-partitioning IP over ALL capacity-feasible subsets (HiGHS) gives the proven
  optimum. Brute-force enumeration cross-checks S and M and counts distinct optima; Kevin's DP
  cross-checks L. So every gap is against a proven optimum, not a best-known.
* Completion (the classical step after the shot), two strengths, both reported: *greedy* (leftovers to
  the cheapest cluster with slack by pairwise distance) and *exact* (every assignment of ≤ 8 leftovers to
  the K vehicles, scored by exact tour cost; optimal given the device's candidates). The device's chosen
  candidates are never altered; if the device lit more than K, every K-subset is tried.
* Selection: best raw-feasible shot (independent set, ≤ K candidates) by exact routed cost; repaired
  shots only as fallback, flagged. `cvrp_gap_pct_any` = best over all shots, reported for honesty.
* Under the exact completion, model gap = CVRP gap (set partitioning with tour costs IS the CVRP).

## 7. The honest answers (updated with the 20-instance results)

* "Does the quantum part do anything?" Per shot, yes: 60–100 % of register ground states are valid
  maximum packings vs 3–6 % of random bitstrings on M/L (10–30×). On final route length, no: under
  the greedy completion the ideal register ties random (9.7 vs 9.9 % on L); under the exact completion
  random wins (1.7 vs 3.0 %) because low-coverage shots hand the exhaustive completion most of the
  problem. At ≤ 12 atoms the completion sets the gap. We show both; we don't hide it.
* "Then why bother?" The brief says no device beats a tuned classical solver and that is not what is
  measured. What we show: a formulation where the inequality never reaches the device, 2–12 atoms
  instead of 14–80, exact emulation on every tier, feasibility that tracks a measurable embedding
  metric, and a clean pulse-adiabaticity curve. And a measured ceiling: where the quantum step stops
  mattering.
* "Why not weight candidates (DMM)?" We computed the ideal maximum-weight packing: worse than
  uniform (L 8.7 vs 4.8 %). Fixing bigger candidates constrains the completion more. So we didn't
  build it. `AnalogDevice` has no DMM anyway; `DigitalAnalogDevice` does (emulation only).
* "Why not petals?" Tried: coverage up, gap not, embedding worse when mixed with balls.
* "Why is the oracle worse than the sweep?" The oracle returns only exact ground states; the sweep also
  returns low excited states, and best-of-500 picks a better one. Score pulses per shot.
* "Why not emu-mps?" Registers are ≤ 12 atoms; QuTiP is exact; no bond dimension to report.
* "Where did AI help?" Everywhere (brainstorm, derivation drafts, code, report draft); we re-ran and
  re-derived everything; the device checks that caught our errors are in the repo (`check_pulser.py`).
