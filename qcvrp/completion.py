"""Exact-cost completion of a packing shot (Phase 1 of the v2 pipeline).

Given the candidate clusters the device chose (kept intact: the quantum decision stands),
place every uncovered customer so that the TOTAL EXACT ROUTED COST  sum_b tour(cluster_b)
is minimal.  Tour costs for every subset come from the Held-Karp table (reference.TourTable),
so the objective is the true CVRP objective, not a proxy.

    * |uncovered| <= `exact_max`  : exhaustive, vectorised over all K^|U| bin assignments
                                    (optimal completion given the chosen candidates)
    * |uncovered| >  `exact_max`  : greedy insertion by exact marginal routed cost, flagged
                                    (such shots have few candidates lit and never win selection
                                    in practice; the flag is recorded per shot)

Capacity is enforced on every bin; a completion with no feasible assignment marks the shot
infeasible.  The same function is applied to quantum, oracle and random shots.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .instances import Instance
from .reference import TourTable, mask_of


@dataclass
class Completion:
    clusters: list[list[int]] | None
    cost: float                 # exact routed cost (inf if infeasible)
    exact: bool                 # True if the completion sub-problem was solved exhaustively
    n_uncovered: int


def _bin_masks(kept: list[list[int]], K: int) -> list[int]:
    masks = [mask_of(cl) for cl in kept]
    return masks + [0] * (K - len(masks))


def complete_exact(inst: Instance, kept: list[list[int]], tours: TourTable, exact_max: int = 8) -> Completion:
    K, Q = inst.K, inst.Q
    d = inst.d
    if len(kept) > K:
        return Completion(None, float("inf"), True, 0)
    covered = set(c for cl in kept for c in cl)
    U = sorted(set(range(inst.N)) - covered)
    base = _bin_masks(kept, K)
    base_load = np.array([int(d[[c for c in range(inst.N) if m >> c & 1]].sum()) if m else 0 for m in base])
    if any(base_load > Q):
        return Completion(None, float("inf"), True, len(U))
    if not U:
        cost = float(sum(tours.cost[m] for m in base if m))
        return Completion([sorted(cl) for cl in kept], cost, True, 0)
    nU = len(U)
    if K ** nU <= K ** exact_max:
        # enumerate every assignment of the nU uncovered customers to the K bins
        n_assign = K ** nU
        idx = np.arange(n_assign)
        assign = np.zeros((n_assign, nU), dtype=np.int64)
        for j in range(nU):
            assign[:, j] = (idx // (K ** j)) % K
        bits = np.array([1 << c for c in U], dtype=np.int64)
        dem = d[U].astype(np.int64)
        masks = np.tile(np.array(base, dtype=np.int64), (n_assign, 1))
        loads = np.tile(base_load, (n_assign, 1))
        for j in range(nU):
            sel = assign[:, j]
            masks[np.arange(n_assign), sel] |= bits[j]
            loads[np.arange(n_assign), sel] += dem[j]
        feas = (loads <= Q).all(axis=1)
        if not feas.any():
            return Completion(None, float("inf"), True, nU)
        costs = tours.cost[masks].sum(axis=1)  # cost[0] = 0 for empty bins
        costs = np.where(feas, costs, np.inf)
        best = int(np.argmin(costs))
        clusters = [sorted(c for c in range(inst.N) if masks[best, b] >> c & 1) for b in range(K) if masks[best, b]]
        return Completion(clusters, float(costs[best]), True, nU)
    # greedy fallback by exact marginal routed cost (largest demand first)
    masks = list(base)
    loads = list(base_load)
    for u in sorted(U, key=lambda c: -d[c]):
        best = None
        for b in range(K):
            if loads[b] + d[u] > Q:
                continue
            delta = tours.cost[masks[b] | (1 << u)] - (tours.cost[masks[b]] if masks[b] else 0.0)
            if best is None or delta < best[0]:
                best = (delta, b)
        if best is None:
            return Completion(None, float("inf"), False, nU)
        masks[best[1]] |= 1 << u
        loads[best[1]] += int(d[u])
    clusters = [sorted(c for c in range(inst.N) if m >> c & 1) for m in masks if m]
    return Completion(clusters, float(sum(tours.cost[m] for m in masks if m)), False, nU)
