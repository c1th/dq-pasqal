# Working notes for the 4-page report (Track A)

Everything below was produced by the code in this repo; numbers are from
`results/classical.csv` and the sandbox runs on 2026-10-24. Re-run before quoting.

## 1. The problem (Derive 2.1)

Sets: customers `i in {0..N-1}`, vehicles `k in {0..K-1}`, depot `0`.
Parameters: locations `P_i`, demands `d_i`, capacity `Q`, distances `D_uv` (Euclidean).
Decision variables (clustering): `x_ik = 1` iff customer `i` is served by vehicle `k`.
Objective (clustering model): minimise within-cluster pairwise distance
`sum_k sum_{i<j} D_ij x_ik x_jk` (the full CVRP objective is the total routed length).
Constraints: **equality** `sum_k x_ik = 1` for every `i`; **inequality** `sum_i d_i x_ik <= Q`
for every `k`.

Instance tiers S/M/L as in Table 1; depot fixed at (50, 50); every instance ships with a
bin-packing certificate (`instances/*.json`) and a load ratio / dmax ratio.

## 2. What the register minimises (Derive 3.1)

Omega -> 0 leaves the diagonal cost `E(n) = -delta * sum_a n_a + sum_{a<b} (C6 / r_ab^6) n_a n_b`.
* controlled through the laser: one global `delta` (same linear coefficient for every atom);
* controlled through positions: couplings, all positive, 1/r^6, never exactly zero;
* not controlled: sign of couplings, per-atom linear terms (global detuning), coupling
  magnitudes outside the 5-10 um band (min separation 5 um; at 10 um the coupling is 1/64 of
  its value at 5 um).
At `Omega_max = 12.57 rad/us` the blockade radius is `(C6/Omega)^(1/6) = 6.4 um`.
Native problem: maximum independent set on a unit-disk graph.

## 3. Variable accounting (Derive 4.1)

| encoding | count | S | M | L | fits 80? |
|---|---|---|---|---|---|
| assignment variables only | N K | 8 | 30 | 60 | yes |
| + binary slack, `ceil(log2(Q+1))` per vehicle | N K + K ceil(log2(Q+1)) | 14 | 45 | 80 | S, M yes; L on paper only (slack couples to every assignment atom) |
| one-hot by blockade, no slack | N K | 8 | 30 | 60 | yes, but unembeddable at M/L (Sec. 5) |
| 1-bit bisection / conflict split | N per run | 4 | 10 | 15 | yes |
| **ball packing (ours)** | #candidates <= N (+grid) | <= 4 | 5-9 | 8-13 | yes |

## 4. Key derivation: what survives one-hot

On the one-hot manifold `sum_k x_ik = 1`, any per-customer coefficient `c_i` gives
`sum_k c_i x_ik = c_i` = constant. Expanding a quadratic load penalty
`sum_k (L_k - T)^2`, `L_k = sum_i d_i x_ik`: all linear terms are constants on the manifold;
the only assignment-dependent term is `2 sum_k sum_{i<j} d_i d_j x_ik x_jk`.
So with one-hot enforced by blockade there is no slack and no linear term to realise; the
capacity penalty is a same-vehicle repulsion proportional to `d_i d_j`. It enforces balance,
not an inequality at Q. (This is unbalanced penalisation projected on one-hot.)
It is also why the model fails: the resulting coupling pattern is dense.

## 5. What did not work (with numbers)

* **Dense one-hot (8/30/60 atoms).** Tier S: the realised register's exact ground state
  decodes to the model optimum on seeds 0-3 (8 atoms, in-group coupling 55 rad/us,
  delta ~25). Tier M (30 atoms, K=3 triangles of side 5 um): effective (block-centred)
  coupling error 0.97-0.98 on seeds 0-2 for beta in {0, 1}; brute-forcing the realised
  energy over all 9842 clusterings, the realised minimiser is 57-253 % above the model
  optimum and the model optimum ranks 1349th-4770th. The K-gon per customer cannot be
  oriented to satisfy ~10 partners at once; a square (K=4) has a 7.07 um diagonal whose
  coupling is 1/8 of the side coupling (sqrt(2)^6 = 8), so one-hot at K=4 is marginal anyway.
* **Dense 1-bit bisection (N atoms).** Exact at N=4; at N=10/15 the least-squares embedding
  packs all atoms within ~8-10 um (every pair wants a coupling), log-RMS coupling error
  ~2-2.4, realised ground state 25-55 % above the K=2 model optimum.
* **Sparse far-conflict hierarchical bisection.** Embeds cleanly at <= 2 edges per atom
  (log-RMS 0.5-0.6, no violable edges) but even the *ideal* MIS of the conflict graph gives
  mean CVRP gaps of 10-16 % (M) and 21-24 % (L): "largest set with no far pair inside" is
  too coarse a split, and the error compounds over levels.
* **Lesson**: in 2D at 1/r^6 an atom has ~3 strongly coupled neighbours whose own mutual
  couplings are weak; any model needing more than that per variable is an approximation whose
  error you must measure. `embedding.classical_ground_states` / `manifold_check` are the
  tools: they tell you what a perfect sweep would return before you spend a single emulation.

## 6. Headline model: ball packing by MIS

Candidate clusters (one per customer centre, nearest customers until `fill*Q`, near-duplicates
thinned at Jaccard > 0.6) are capacity-feasible by construction. Atom per candidate,
blockade between overlapping candidates. Register = map geometry at scale (the embedder
starts from the candidate centres). Ground state = maximum packing of disjoint compact
clusters. Uncovered customers are inserted classically (stated, reported as coverage).

Classical-sampler results (5 instances per tier, 500 shots; `results/classical.csv`):

| tier | sampler | atoms | feasibility raw | CVRP gap mean / best |
|---|---|---|---|---|
| M | ideal-mis | 5-6 | 1.00 | 6.5 % / 0.0 % |
| M | oracle (realised register) | 5-6 | 0.8-1.0 | 13.8 % / 4.9 % |
| M | random bitstrings | 5-6 | ~0.1 | 3.6 % / 0.0 % (after post-processing; 0.67 of shots needed overlap resolution) |
| L | ideal-mis | 8-10 | 1.00 | 14.7 % / 5.6 % |
| L | oracle | 8-10 | 0.8-1.0 | 17.9 % / 5.6 % |
| L | random bitstrings | 8-10 | ~0.05 | 14.7 % / 5.6 % (0.87 of shots needed overlap resolution) |

Honest reading: at these sizes the stated post-processing (overlap resolution + greedy
insertion) is strong enough that best-of-500 random bitstrings match the ideal packing on the
final gap. The quantum signal is in the *raw* shots: 80-100 % of register ground states are
valid packings versus 5-10 % of random bitstrings. Report both, and report the covered
fraction. Decomposition gaps of the model itself (model optimum routed exactly vs CVRP
optimum) are 0-16 % on S, 3-13 % on M, 1-12 % on L.

## 7. Embedding and pulse (A3, A4)

Embedding fidelity formulas: `rel_err = sqrt(sum_wanted (U-T)^2 / sum_wanted T^2)`,
`log_rms = sqrt(mean_wanted log(U/T)^2)`, unwanted ratio `sum_unwanted U / sum_wanted T`, and for
MIS registers the graph-level count of violable edges (target edges realised below delta) and
spurious edges (non-edges above delta). Design rule: conflict pairs at 6 um (18.6 rad/us),
non-conflict pairs beyond 9.5 um (1.2 rad/us), `delta_final = sqrt(18.6 * 1.2) = 4.7 rad/us`,
`Omega_max = 4 rad/us` (blockade radius 7.7 um sits between the two distances).
Pulse: interpolated amplitude 0 -> Omega_max -> 0, linear detuning -1.5 delta_f -> delta_f,
4000 ns (device maximum 6000 ns). Score candidate pulses by feasibility rate and
P(best gap <= eps), not by mean energy.

## 8. First emulation results (pulser 1.9.1, AnalogDevice with automatic layout, QuTiP, 500 shots)

Smoke test: 3-atom path at 6 um spacing, baseline sweep (Omega_max 4 rad/us, delta -7 -> +4.7,
4000 ns): ground state `101` in 96 % of shots.

Full pipeline (`results/qutip.csv`, 15 instances x 3 runs):

| tier | atoms | raw feasibility | CVRP gap mean / best | P(gap <= 1 %) |
|---|---|---|---|---|
| S | 2-4 | 0.0-1.0 | 1.95 % / 0.0 % | 0.80 |
| M | 5-6 | 0.42-1.00 | 3.55 % / 0.0 % | 0.60 |
| L | 6-10 | 0.21-1.00 | 14.7 % / 5.6 % | 0.00 |

Two things worth saying in the report:
* raw feasibility tracks the embedding: L-0003 has 8 of 26 overlap edges realised below delta
  and 22 % raw feasibility; L-0001 has 0 violable edges and 100 %. The graph-level fidelity
  metric (violable / spurious edges) predicts the physics.
* on M the sweep *beats the exact ground state* (3.6 % vs 6.7 % mean gap): the sweep samples
  excited states too, and the pipeline keeps the best feasible shot. This is exactly the
  brief's remark that you only need one good shot, and it is why pulses must be scored by
  P(best gap <= eps) and feasibility rate, not by mean energy.

Selection rule now in force (state it in the report): the answer is the device's best
raw-feasible shot (an independent set of <= K candidates, most candidates kept, then lowest
objective after greedy completion); overlap resolution is only a fallback and is flagged
(`repaired`). `cvrp_gap_pct_any` keeps the best over all shots after full post-processing
for transparency.

## 9. Pulse sweep (240 QuTiP emulations, results/pulse_sweep.csv)

P(max packing / shot), mean over instances:

| tier | 1 us | 2 us | 4 us | 6 us (Omega 2, slow-end) |
|---|---|---|---|---|
| M | 0.27 | 0.53 | 0.69 | 0.73 (0.74 linear Omega 4) |
| L | 0.06 | 0.21 | 0.45 | 0.68 |

Best gap over 500 shots is identical for every pulse (every pulse hits a maximum packing at least once).
Omega 8 worst at every duration (blockade radius 6.9 um blurs the 6 / 9.5 um design). slow-end helps on
L at 6 us (+0.11 at Omega 2), neutral on M. Raw feasibility 0.57-0.83 regardless of pulse.

## 10. Open items

* Run `qutip` sampler on all instances (<= 13 atoms: exact), compare with `oracle` rows to
  isolate pulse error; sweep duration / Omega_max; try a slow-down near the end of the sweep.
* Candidate pool: `--grid 3` / two fill levels per centre to raise coverage; watch violable edges.
* Random-K-seeds classical baseline (stronger than random bitstrings) for the ablation.
* DMM variant (emulation only): per-candidate detuning proportional to candidate compactness,
  i.e. maximum-weight packing; label separately.
* Track B (if time): edge-based 2-factor TSP encoding, n <= 4 subtour-free lemma, k-NN pruning.

## 11. Second session (Sat 15:40-17:40): what moved the gap and what did not

Diagnosis: ideal-MIS gap == QuTiP gap on M and L, so the model, not the physics, set the gap.

**Adopted: exact-cost classical steps (v2).** Leftovers placed by exhaustive assignment with
exact tour costs; best shot by exact routed cost. On the first 5 instances/tier (classical
samplers): L ideal 14.7 -> 4.8 %, oracle 17.9 -> 8.1 %; M ideal/oracle 6.5 -> 2.7 %.
The same completion lifts random bitstrings to 1.3 % (L) and 0.0 % (M): with an exhaustive
completion of <= 8 leftovers the sampler barely matters at these sizes. Reported as such.

**Measured, not adopted.**
* Petal (angular-window) candidates: ideal gap 5.3 % (petals only) / 6.4 % (petals + balls)
  vs 4.8 % for balls on L; coverage up (0.91), gap not; petals + balls make the overlap graph
  dense (27-35 edges on 12 atoms, 5-15 violable) and the oracle raw feasibility drops to 0-0.2.
* Savings-weighted maximum-weight IS (what the DMM would implement; AnalogDevice has no DMM,
  DigitalAnalogDevice has dmm_0 with bottom detuning -125.7 rad/us): at the IDEAL level it is
  worse than uniform MIS, L 8.7 % vs 4.8 %, M 5.6 % vs 2.7 %. Higher coverage, worse gap: every
  candidate the device fixes is a constraint on the exact completion, and weighting fixes bigger
  ones. The pulser DMM sequence was therefore not built (keep-or-drop rule).
* Fill level: 0.7 gives 3.0 % ideal on L but 13 atoms and oracle raw feasibility 0.6; 0.85 stays.

**Honest conclusion for the report.** At <= 12 atoms the register is a reliable sampler of valid
maximum packings (per-shot advantage over random bitstrings of 10-50x), but the end-to-end gap is
set by the classical completion, and a strong completion makes the sampler irrelevant. The useful
quantum contribution would need registers where enumerating packings is not trivial, i.e. far
beyond 80 atoms for this encoding.
