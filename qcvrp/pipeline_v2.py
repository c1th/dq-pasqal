"""v2 packing pipeline: same register, exact-cost classical steps (Phase 1).

Differences from pipeline.packing_cluster (the v1 baseline, kept for comparison):
    * leftover customers are placed by completion.complete_exact (optimal given the chosen
      candidates, by exact routed cost), not by greedy pairwise distance;
    * the best shot is the raw-feasible shot (independent set, <= K candidates) with the
      lowest exact routed cost; repaired shots are a fallback and flagged;
    * the model objective IS the routed cost, so model gap == CVRP gap and the decomposition
      loss of the pairwise objective no longer enters.
Everything else (candidates, embedding, detuning, samplers) is unchanged, so a v1 -> v2
difference in results.csv is the effect of the classical steps alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .completion import complete_exact
from .embedding import embed, suggest_delta
from .instances import Instance
from .pipeline import LevelRecord
from .reference import TourTable


@dataclass
class ShotV2:
    bits: str
    count: int
    n_excited: int
    n_overlaps: int
    n_kept: int
    coverage: float
    feasible_raw: bool
    feasible: bool
    cost: float                      # exact routed cost after completion (inf if infeasible)
    exact_completion: bool
    clusters: list[list[int]] | None


def resolve_v2(inst: Instance, chosen: list[int], cands, tours: TourTable, exact_max: int = 8) -> ShotV2:
    """Overlap resolution (only needed for non-independent shots) + exact completion.

    Independent shots with more than K candidates: every K-subset of the chosen candidates
    is completed exactly and the cheapest feasible one is kept (at most C(12, 4) = 495
    subsets).  Overlapping shots: a disjoint sub-collection is kept greedily by routed cost
    per customer, then completed.  The candidates kept are always a subset of what the
    device chose; the completion never alters them."""
    from itertools import combinations
    sets = [set(cands[a].members) for a in chosen]
    n_over = sum(1 for i in range(len(chosen)) for j in range(i + 1, len(chosen)) if sets[i] & sets[j])
    raw_ok = n_over == 0 and len(chosen) <= inst.K
    dropped = False
    if n_over == 0:
        k_full = min(len(chosen), inst.K)
        best = None
        for k_keep in range(k_full, -1, -1):
            for sub in combinations(range(len(chosen)), k_keep):
                kept = [sorted(sets[t]) for t in sub]
                comp = complete_exact(inst, kept, tours, exact_max=exact_max)
                if comp.clusters is not None and (best is None or comp.cost < best[0].cost):
                    best = (comp, kept)
            if best is not None:
                dropped = k_keep < k_full   # had to drop device-chosen candidates to become feasible
                break
        if best is None:
            kept = [sorted(sets[t]) for t in range(k_full)]
            comp = complete_exact(inst, kept, tours, exact_max=exact_max)
        else:
            comp, kept = best
    else:
        order = sorted(range(len(chosen)), key=lambda t: tours.cost_of(sorted(sets[t])) / max(len(sets[t]), 1))
        kept, used = [], set()
        for t in order:
            if sets[t] & used:
                continue
            if len(kept) >= inst.K:
                break
            kept.append(sorted(sets[t]))
            used |= sets[t]
        comp = complete_exact(inst, kept, tours, exact_max=exact_max)
    covered = sum(len(cl) for cl in kept) / inst.N
    return ShotV2("", 0, len(chosen), n_over, len(kept), covered, raw_ok and not dropped, comp.clusters is not None,
                  comp.cost, comp.exact, comp.clusters)


@dataclass
class ResultV2:
    clusters: list[list[int]]
    feasible: bool
    repaired: bool
    shots: list[ShotV2]
    n_shots: int
    feasibility_rate_raw: float
    feasibility_rate: float
    coverage_best: float
    coverage_mean: float
    levels: list
    clusters_any: list = field(default_factory=list)
    mis_size: int = -1
    p_mis_shot: float = float("nan")
    best_cost: float = float("inf")
    best_exact_completion: bool = True
    candidates: list = field(default_factory=list, repr=False)
    target: object = field(default=None, repr=False)
    embedding: object = field(default=None, repr=False)
    objective_kind: str = "routed"

    @property
    def max_atoms(self) -> int:
        return self.levels[0].n_atoms if self.levels else 0

    @property
    def min_feasibility_rate(self) -> float:
        return self.feasibility_rate


def packing_cluster_v2(inst: Instance, model, sampler: Callable, n_shots: int, tours: TourTable, seed: int = 0,
                       embed_kw: dict | None = None, sampler_kw: dict | None = None, exact_max: int = 8,
                       delta_override: float | None = None) -> ResultV2:
    rng = np.random.default_rng(seed)
    embed_kw = dict(J_max_phys=18.6, U_floor=1.2, n_restarts=8) | (embed_kw or {})
    sampler_kw = sampler_kw or {}
    tgt, cands = model.target(inst)
    emb = embed(tgt, seed=int(rng.integers(1 << 31)), **embed_kw)
    sd = suggest_delta(emb, tgt)
    delta = sd["delta"] if delta_override is None else delta_override
    counts = sampler(tgt, emb, delta, n_shots, rng, **sampler_kw)
    shots = []
    for bits, c in counts.items():
        s = resolve_v2(inst, model.decode(bits, cands), cands, tours, exact_max=exact_max)
        s.bits, s.count = bits, c
        shots.append(s)
    total = sum(s.count for s in shots)
    feas = [s for s in shots if s.feasible]
    from .sampling import ideal_mis_sampler
    try:
        mis_size = max(sum(int(ch) for ch in b) for b in ideal_mis_sampler(tgt, emb, delta, 1, rng))
    except ValueError:
        mis_size = -1
    p_mis = sum(s.count for s in shots if s.feasible_raw and s.n_excited == mis_size) / max(total, 1)
    raw = [s for s in feas if s.feasible_raw and s.n_kept >= 1]
    best = min(raw, key=lambda s: s.cost) if raw else (min(feas, key=lambda s: s.cost) if feas else None)
    best_any = min(feas, key=lambda s: s.cost) if feas else None
    level = LevelRecord(list(range(inst.N)), inst.K, 0, 0, tgt.n_atoms, int(tgt.J.sum() // 2), delta,
                        (sd["lower"], sd["upper"]), emb.fidelity, total, len(counts),
                        sum(s.count for s in shots if s.feasible_raw) / max(total, 1),
                        best.cost if best else None, best.bits if best else None,
                        bool(best and not best.feasible_raw), None, None, emb.positions)
    return ResultV2(
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
        mis_size=mis_size, p_mis_shot=p_mis,
        best_cost=best.cost if best else float("inf"),
        best_exact_completion=best.exact_completion if best else True,
        candidates=cands, target=tgt, embedding=emb,
    )
