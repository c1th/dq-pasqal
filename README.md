# CVRP on a neutral-atom analog device — DuQuantum 2026, Pasqal challenge (Track A)

Cluster-first / route-second CVRP where the clustering step runs as a **maximum
independent set on a Pasqal AnalogDevice register**, with capacity handled *by
construction* instead of by slack variables.

## One-command reproduction

```bash
pip install -r requirements.txt            # numpy/scipy/pandas only, for everything classical
python scripts/make_instances.py           # 15 certified instances (S/M/L x 5 seeds) + proven-optimal references
python scripts/run_pipeline.py --samplers ideal-mis oracle random --out results/classical.csv
# quantum emulation (needs `pip install pulser`; QuTiP backend is exact at <= ~12 atoms):
python scripts/check_pulser.py                                   # smoke test + device specs
python scripts/run_pipeline.py --samplers qutip --shots 500 --runs 3 --out results/results.csv --append
# figures for the report:
python scripts/plot_results.py --results results/results.csv --register instances/L-0000.json --sampler qutip
```

`--append` merges into one `results.csv` (rows with the same instance/method/sampler/seed are
replaced), so classical and quantum samplers end up in the single file the brief asks for.

`results/*.csv` have one row per (instance, method, sampler, run seed) with the metrics of
Section 9 of the brief (best gaps at model/CVRP level, feasibility rates raw and after the
stated post-processing, atom count, embedding fidelity, detuning, reference status).

## What the device minimises (Derive 3.1)

With the amplitude back at zero, Eq. (1) is the classical cost
`E(n) = -delta * sum_a n_a + sum_{a<b} C6/r_ab^6 n_a n_b`:
every atom has the **same** linear coefficient (global detuning) and every coupling is
**positive and ~1/r^6**. Within the hardware envelope (5 um minimum separation, 38 um
radius) the useful coupling band is ~5-10 um: each atom "sees" a handful of neighbours.
That cost is maximum independent set on a unit-disk graph. The whole design problem is
making CVRP clustering look like that.

## Models (qcvrp/models.py) and what happened to them

| model | atoms (S/M/L) | result |
|---|---|---|
| textbook one-hot + binary slack | 14 / 45 / 80 | not built: slack couples every atom to every assignment atom (dense, long range) |
| one-hot by blockade, dense same-vehicle repulsion `w = alpha*dist + beta*d_i d_j` | 8 / 30 / 60 | exact on S; **unembeddable on M**: effective coupling error 0.97, realised minimiser 57-250 % off, model optimum ranks ~2000th of 9842 (`embedding.manifold_check`) |
| 1-bit bisection (K=2), dense | 4 / 10 / 15 | exact on S; 25-55 % off at N = 10, 15 (dense target crams the register into an 8 um disc) |
| hierarchical bisection, sparse far-conflict MIS | <= 4 / 10 / 15 | embeds well at <= 2 edges/atom but the model is too weak (ideal MIS 10-25 % gaps) |
| **ball packing by MIS** (headline) | <= 4 / ~6-10 / ~8-13 | capacity feasible by construction, register = the map at scale, feasibility rate of raw shots ~0.8-1.0 vs 0.05 for random bitstrings; see results |

**Ball packing**: for every customer (optionally a grid) take the nearest customers until the
load would exceed `fill * Q`; that is a candidate cluster, feasible by construction. One atom
per candidate; candidates that share a customer sit inside the blockade radius, all others
outside. The ground state is a maximum set of disjoint compact clusters. Uncovered customers
are assigned to the chosen cluster with the cheapest marginal distance that has slack (stated
post-processing, identical for quantum and random shots; the covered fraction is reported).
No slack variables, no penalty weight to tune: the only knobs are `fill` and the
near-duplicate threshold `max_jaccard` (dense overlap cliques are unrealisable: a clique must
sit entirely inside one blockade disc).

## v2 (Phase 1 of the second session): exact-cost classical steps

`qcvrp/completion.py` + `qcvrp/pipeline_v2.py`: leftover customers are placed by exhaustive
assignment with exact tour costs (optimal given the chosen candidates when <= 8 customers are
uncovered), and the best shot is chosen by exact routed cost. `scripts/run_final.py` samples each
(instance, sampler, seed) once and post-processes the same shots both ways (`+greedy` = v1,
`+exact` = v2), so the effect of the classical completion is isolated in `results/final.csv`.
Measured and **not** adopted (code kept as evidence): petal (angular-window) candidates
(`qcvrp/models_v2.py`), savings-weighted maximum-weight IS (`ideal-mwis` / `oracle-weighted`
samplers; would need the DMM, which `AnalogDevice` lacks). See `report/notes.md` section 11.

```bash
python scripts/make_instances.py --seeds $(seq 0 19)                      # 20 instances per tier
python scripts/run_final.py --samplers ideal-mis oracle random --runs 3 --out results/final.csv
python scripts/run_final.py --samplers qutip --runs 3 --shots 500 --out results/final.csv --append   # Mac
```

## Samplers

* `ideal-mis` — all maximum independent sets of the target graph (model quality, no physics)
* `oracle` — exact ground states of the *realised* register cost (perfect sweep, real embedding)
* `random` — uniform random bitstrings through the same post-processing (the ablation)
* `qutip` — pulser-simulation of the adiabatic sweep; `emu-sv` for larger registers

Comparing the three classical samplers separates model error, embedding error and pulse error.

## Reference values (qcvrp/reference.py)

Held-Karp over every customer subset gives exact tours for every possible cluster; a
set-partitioning MILP over the *complete* pool of capacity-feasible subsets (scipy/HiGHS) gives
a **proven optimum** for the CVRP and for the clustering model on every tier (~2 s on L).
Brute-force partition enumeration cross-checks it on S and M and counts distinct optima
(vehicle relabellings identified).

## Layout

```
qcvrp/instances.py   tiers, seeded generator, bin-packing certificate, validators, JSON I/O
qcvrp/reference.py   Held-Karp tour table, set-partition MILP, enumeration cross-check
qcvrp/models.py      coupling targets, decoders, repair; OneHot / Bisection / ConflictSplit / Packing
qcvrp/embedding.py   least-squares register fit, fidelity metrics, exact ground states, manifold check
qcvrp/pulses.py      pulser sequences (adiabatic baseline) and emulator samplers
qcvrp/sampling.py    classical samplers (random / oracle / ideal-mis / boltzmann)
qcvrp/pipeline.py    ball-packing driver, hierarchical-split driver, stated post-processing
qcvrp/metrics.py     gaps, results.csv
scripts/             make_instances.py, run_pipeline.py, validate.py
instances/           generated instances with certificates, references.csv
results/             results.csv files
```
