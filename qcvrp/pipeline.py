"""End-to-end clustering pipeline: hierarchical bisection with a conflict-MIS register.

K vehicles are split as k = k1 + k2 (k1 = k // 2):  K=2 -> (1,1), K=3 -> (1,2), K=4 -> (2,2).
At every node the customer subset is embedded as a conflict-MIS register
(models.ConflictSplitModel), sampled, and the best feasible split is kept:

    feasible  : the excited side S and the rest R can each be packed into their vehicle
                count (k1 or k2) within capacity, exactly checked by bin packing; both
                orientations (S -> k1 or S -> k2) are tried;
    best      : lowest within-side pairwise distance (the clustering objective restricted
                to this node).

If no shot is feasible the node is repaired by the stated greedy step (move the customer
with the cheapest marginal distance across the split until both sides pack) and the
result is flagged.  Every level records atoms, edges, embedding fidelity, delta, shots,
feasibility rate and whether repair was used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .embedding import Embedding, embed, suggest_delta
from .instances import Instance, find_feasible_assignment
from .models import ConflictSplitModel


def split_plan(k: int) -> tuple[int, int]:
    k1 = k // 2
    return k1, k - k1


def within_cost(customers: Sequence[int], Dc: np.ndarray) -> float:
    cs = list(customers)
    if len(cs) < 2:
        return 0.0
    sub = Dc[np.ix_(cs, cs)]
    return float(np.triu(sub, 1).sum())


def packable(inst: Instance, customers: Sequence[int], k: int) -> bool:
    if len(customers) == 0:
        return True
    return find_feasible_assignment([inst.demands[c] for c in customers], k, inst.Q) is not None


def orientation(inst: Instance, S: Sequence[int], R: Sequence[int], k1: int, k2: int):
    """Vehicle counts for (S, R) that make both sides packable, or None."""
    if packable(inst, S, k1) and packable(inst, R, k2):
        return (k1, k2)
    if k1 != k2 and packable(inst, S, k2) and packable(inst, R, k1):
        return (k2, k1)
    return None


def repair_split(inst: Instance, S: list[int], R: list[int], k1: int, k2: int, Dc: np.ndarray, max_radius: int = 4):
    """Stated repair: the feasible split nearest to the shot in Hamming distance (fewest
    customers moved across the split), ties broken by within-side distance.  Searches
    radius 1, 2, ... up to max_radius; returns (S, R, (kS, kR)) or None."""
    from itertools import combinations
    customers = sorted(S) + sorted(R)
    side = {c: 0 for c in S} | {c: 1 for c in R}
    for radius in range(1, max_radius + 1):
        best = None
        for moved in combinations(customers, radius):
            s2 = dict(side)
            for c in moved:
                s2[c] ^= 1
            S2 = [c for c in customers if s2[c] == 0]
            R2 = [c for c in customers if s2[c] == 1]
            o = orientation(inst, S2, R2, k1, k2)
            if o is None:
                continue
            obj = within_cost(S2, Dc) + within_cost(R2, Dc)
            if best is None or obj < best[0]:
                best = (obj, S2, R2, o)
        if best is not None:
            return best[1], best[2], best[3]
    return None


@dataclass
class LevelRecord:
    customers: list[int]
    k: int
    k1: int
    k2: int
    n_atoms: int
    n_edges: int
    delta: float
    delta_bounds: tuple[float, float]
    fidelity: dict
    n_shots: int
    n_distinct: int
    feasibility_rate: float
    best_objective: float | None
    best_bits: str | None
    repaired: bool
    split: tuple[list[int], list[int]] | None
    vehicles: tuple[int, int] | None
    positions: np.ndarray = field(repr=False, default=None)


@dataclass
class PipelineResult:
    clusters: list[list[int]]
    levels: list[LevelRecord]
    feasible: bool
    repaired: bool

    @property
    def max_atoms(self) -> int:
        return max((l.n_atoms for l in self.levels), default=0)

    @property
    def min_feasibility_rate(self) -> float:
        return min((l.feasibility_rate for l in self.levels), default=float("nan"))

    @property
    def worst_fidelity(self) -> float:
        return max((l.fidelity.get("log_rms", 0.0) for l in self.levels), default=float("nan"))


# ---------------------------------------------------------------------------
# Ball packing (single run): choose disjoint candidate clusters, cover the rest classically
# ---------------------------------------------------------------------------
@dataclass
class PackingShot:
    bits: str
    count: int
    n_excited: int
    n_overlaps: int           # overlapping pairs among excited candidates (0 = independent set)
    n_kept: int               # candidates kept after resolving overlaps / capping at K
    coverage: float           # fraction of customers covered by kept candidates
    feasible_raw: bool        # independent set, <= K candidates: no overlap resolution needed
    feasible: bool            # final clustering feasible after assigning uncovered customers
    objective: float | None   # within-cluster pairwise distance of the final clustering
    clusters: list[list[int]] | None


def resolve_packing(inst: Instance, chosen: list[int], cands, Dc: np.ndarray) -> PackingShot | None:
    """Stated post-processing for one shot (the same for quantum and random shots):
    1. among the excited candidates, keep a disjoint sub-collection: scan by compactness
       (within_cost / size) and drop any candidate overlapping one already kept;
    2. keep at most K of them (most compact first);
    3. assign every uncovered customer to the kept cluster with the cheapest marginal
       distance that has capacity; if none fits and fewer than K clusters exist, open a
       new cluster for it; otherwise the shot is infeasible.
    """
    sets = [set(cands[a].members) for a in chosen]
    order = sorted(range(len(chosen)), key=lambda t: cands[chosen[t]].within_cost / max(len(sets[t]), 1))
    n_over = sum(1 for i in range(len(chosen)) for j in range(i + 1, len(chosen)) if sets[i] & sets[j])
    kept, used = [], set()
    for t in order:
        if sets[t] & used:
            continue
        if len(kept) >= inst.K:
            break
        kept.append(t)
        used |= sets[t]
    clusters = [sorted(sets[t]) for t in kept]
    loads = [sum(inst.demands[c] for c in cl) for cl in clusters]
    coverage = len(used) / inst.N
    uncovered = sorted(set(range(inst.N)) - used)
    feasible = True
    for c in sorted(uncovered, key=lambda c: -inst.demands[c]):
        best = None
        for k, cl in enumerate(clusters):
            if loads[k] + inst.demands[c] <= inst.Q:
                m = sum(Dc[c, j] for j in cl)
                if best is None or m < best[0]:
                    best = (m, k)
        if best is not None:
            clusters[best[1]].append(c)
            loads[best[1]] += inst.demands[c]
        elif len(clusters) < inst.K:
            clusters.append([c])
            loads.append(inst.demands[c])
        else:
            feasible = False
            break
    obj = sum(within_cost(cl, Dc) for cl in clusters) if feasible else None
    return PackingShot("", 0, len(chosen), n_over, len(kept), coverage,
                       n_over == 0 and len(chosen) <= inst.K, feasible, obj, clusters if feasible else None)


@dataclass
class PackingResult:
    clusters: list[list[int]]
    feasible: bool
    repaired: bool                      # best shot needed overlap resolution or capping
    shots: list[PackingShot]
    n_shots: int
    feasibility_rate_raw: float         # fraction of shots that are independent sets with <= K candidates
    feasibility_rate: float             # fraction of shots feasible after the stated post-processing
    coverage_best: float                # customers covered by the best shot's kept candidates
    coverage_mean: float
    levels: list                        # for results_row compatibility (one pseudo-level)
    clusters_any: list = field(default_factory=list)   # best over all shots after full post-processing
    mis_size: int = -1                  # maximum independent set size of the target graph
    p_mis_shot: float = float("nan")    # fraction of shots that ARE a maximum independent set
    candidates: list = field(default_factory=list, repr=False)
    target: object = field(default=None, repr=False)
    embedding: object = field(default=None, repr=False)

    @property
    def max_atoms(self) -> int:
        return self.levels[0].n_atoms if self.levels else 0

    @property
    def min_feasibility_rate(self) -> float:
        return self.feasibility_rate


def packing_cluster(inst: Instance, model, sampler: Callable, n_shots: int, seed: int = 0,
                    embed_kw: dict | None = None, sampler_kw: dict | None = None) -> PackingResult:
    rng = np.random.default_rng(seed)
    embed_kw = dict(J_max_phys=18.6, U_floor=1.2, n_restarts=8) | (embed_kw or {})
    sampler_kw = sampler_kw or {}
    Dc = inst.customer_dist()
    tgt, cands = model.target(inst)
    emb = embed(tgt, seed=int(rng.integers(1 << 31)), **embed_kw)
    sd = suggest_delta(emb, tgt)
    counts = sampler(tgt, emb, sd["delta"], n_shots, rng, **sampler_kw)
    shots = []
    for bits, c in counts.items():
        ps = resolve_packing(inst, model.decode(bits, cands), cands, Dc)
        ps.bits, ps.count = bits, c
        shots.append(ps)
    total = sum(s.count for s in shots)
    feas = [s for s in shots if s.feasible]
    # Selection rule (stated): the answer is the device's best RAW-feasible shot, i.e. an
    # independent set of <= K candidates with the most candidates kept (the device's job),
    # then the lowest objective after greedy completion of uncovered customers.  Only if no
    # shot is raw-feasible do we fall back to overlap resolution (repaired=True).  The best
    # objective over ALL shots after full post-processing is kept as `best_any` for
    # transparency (it is what a purely classical completion would claim).
    # size of a maximum independent set of the target graph (the device's reference answer)
    from .sampling import ideal_mis_sampler
    try:
        mis_size = max(sum(int(ch) for ch in b) for b in ideal_mis_sampler(tgt, emb, sd["delta"], 1, rng))
    except ValueError:
        mis_size = -1
    p_mis = sum(s.count for s in shots if s.feasible_raw and s.n_excited == mis_size) / max(total, 1)
    raw = [s for s in feas if s.feasible_raw and s.n_kept >= 1]
    if raw:
        best = min(raw, key=lambda s: (-s.n_kept, s.objective))
    else:
        best = min(feas, key=lambda s: s.objective) if feas else None
    best_any = min(feas, key=lambda s: s.objective) if feas else None
    level = LevelRecord(list(range(inst.N)), inst.K, 0, 0, tgt.n_atoms, int(tgt.J.sum() // 2), sd["delta"],
                        (sd["lower"], sd["upper"]), emb.fidelity, total, len(counts),
                        sum(s.count for s in shots if s.feasible_raw) / max(total, 1),
                        best.objective if best else None, best.bits if best else None,
                        bool(best and not best.feasible_raw), None, None, emb.positions)
    return PackingResult(
        clusters=best.clusters if best else [],
        feasible=best is not None,
        repaired=bool(best and not best.feasible_raw),
        shots=shots, n_shots=total,
        feasibility_rate_raw=level.feasibility_rate,
        feasibility_rate=sum(s.count for s in feas) / max(total, 1),
        coverage_best=best.coverage if best else 0.0,
        coverage_mean=sum(s.coverage * s.count for s in shots) / max(total, 1),
        levels=[level],
        clusters_any=best_any.clusters if best_any else [],
        mis_size=mis_size,
        p_mis_shot=p_mis,
        candidates=cands,
        target=tgt,
        embedding=emb,
    )


def hierarchical_cluster(inst: Instance, model: ConflictSplitModel, sampler: Callable, n_shots: int,
                         seed: int = 0, embed_kw: dict | None = None, sampler_kw: dict | None = None,
                         repair: bool = True) -> PipelineResult:
    rng = np.random.default_rng(seed)
    embed_kw = dict(J_max_phys=18.6, U_floor=1.2, n_restarts=8) | (embed_kw or {})
    sampler_kw = sampler_kw or {}
    Dc = inst.customer_dist()
    levels: list[LevelRecord] = []
    any_repair = False
    feasible = True

    def rec(customers: list[int], k: int) -> list[list[int]]:
        nonlocal any_repair, feasible
        if k == 1 or len(customers) <= 1:
            return [list(customers)]
        k1, k2 = split_plan(k)
        Q_small = min(k1, k2) * inst.Q
        tgt = model.target(inst, customers, Q_small=Q_small)
        emb = embed(tgt, seed=int(rng.integers(1 << 31)), **embed_kw)
        sd = suggest_delta(emb, tgt)
        counts = sampler(tgt, emb, sd["delta"], n_shots, rng, **sampler_kw)
        best = None
        n_feas = 0
        total = sum(counts.values())
        for bits, c in counts.items():
            S, R = model.decode(bits, customers)
            o = orientation(inst, S, R, k1, k2)
            if o is None:
                continue
            n_feas += c
            obj = within_cost(S, Dc) + within_cost(R, Dc)
            if best is None or obj < best[0] - 1e-9:
                best = (obj, S, R, o, bits)
        repaired = False
        if best is None:
            if not repair:
                feasible = False
                levels.append(LevelRecord(list(customers), k, k1, k2, tgt.n_atoms, int(tgt.J.sum() // 2), sd["delta"],
                                          (sd["lower"], sd["upper"]), emb.fidelity, total, len(counts), 0.0,
                                          None, None, False, None, None, emb.positions))
                return [list(customers)]
            # repair the lowest-objective shot (any shot)
            cand = min(counts, key=lambda b: within_cost(model.decode(b, customers)[0], Dc)
                       + within_cost(model.decode(b, customers)[1], Dc))
            S, R = model.decode(cand, customers)
            fixed = repair_split(inst, S, R, k1, k2, Dc)
            if fixed is None:
                feasible = False
                levels.append(LevelRecord(list(customers), k, k1, k2, tgt.n_atoms, int(tgt.J.sum() // 2), sd["delta"],
                                          (sd["lower"], sd["upper"]), emb.fidelity, total, len(counts), 0.0,
                                          None, cand, True, None, None, emb.positions))
                return [list(customers)]
            S, R, o = fixed
            best = (within_cost(S, Dc) + within_cost(R, Dc), S, R, o, cand)
            repaired = True
            any_repair = True
        obj, S, R, (kS, kR), bits = best
        levels.append(LevelRecord(list(customers), k, k1, k2, tgt.n_atoms, int(tgt.J.sum() // 2), sd["delta"],
                                  (sd["lower"], sd["upper"]), emb.fidelity, total, len(counts), n_feas / max(total, 1),
                                  obj, bits, repaired, (list(S), list(R)), (kS, kR), emb.positions))
        return rec(list(S), kS) + rec(list(R), kR)

    clusters = rec(list(range(inst.N)), inst.K)
    clusters = [c for c in clusters if c]
    # final feasibility: every cluster within capacity and at most K clusters
    if len(clusters) > inst.K or any(sum(inst.demands[c] for c in cl) > inst.Q for cl in clusters):
        feasible = False
    return PipelineResult(clusters=clusters, levels=levels, feasible=feasible, repaired=any_repair)
