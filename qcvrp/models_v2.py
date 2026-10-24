"""Candidate menus v2 (Phase 2): petals + balls, ranked by exact routed cost.

PetalPackingModel extends PackingModel with
    * petal candidates: contiguous angular windows around the depot (the Sweep / Petal
      family of CVRP heuristics), one window per starting customer and direction, filled
      until `fill * Q`;
    * ranking and near-duplicate thinning by SAVINGS PER CUSTOMER,
          sav(S) = sum_{i in S} 2 d(0, i) - tour(S),   (>= 0 by the triangle inequality)
      i.e. how much driving the cluster saves relative to serving its customers one by one.
      Over a full partition, minimising total tour cost == maximising total savings, so
      this is the right per-candidate score for a packing;
    * a cap on the number of atoms (default 12: exact QuTiP emulation stays cheap).
The target graph, decoder and embedding are unchanged (overlap => blockade).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .instances import Instance
from .models import Candidate, CouplingTarget, PackingModel
from .reference import TourTable, mask_of


@dataclass
class PetalPackingModel(PackingModel):
    petals: bool = True
    balls: bool = True
    max_atoms: int = 12
    rank: str = "savings"          # "savings" (needs tours) | "compact" (pairwise, as v1)
    tours: TourTable | None = None
    name: str = "petal-packing"

    # -- scoring ---------------------------------------------------------------
    def savings(self, inst: Instance, members) -> float:
        D = inst.dist()
        members = list(members)
        return float(sum(2.0 * D[0, i + 1] for i in members) - self.tours.cost[mask_of(members)])

    def score(self, inst: Instance, c: Candidate) -> float:
        """Higher is better."""
        if self.rank == "savings":
            if self.tours is None:
                raise ValueError("rank='savings' needs tours=TourTable")
            return self.savings(inst, c.members) / len(c.members)
        return -c.within_cost / max(len(c.members), 1)

    # -- candidate generation --------------------------------------------------
    def petal_candidates(self, inst: Instance) -> list[Candidate]:
        P = inst.xy
        depot = np.asarray(inst.depot, float)
        ang = np.arctan2(P[:, 1] - depot[1], P[:, 0] - depot[0])
        order = list(np.argsort(ang))
        N = inst.N
        d = inst.d
        cap = self.fill * inst.Q
        Dc = inst.customer_dist()
        out = {}
        for direction in (1, -1):
            for start in range(N):
                members, load = [], 0
                for step in range(N):
                    i = int(order[(start + direction * step) % N])
                    if load + d[i] > cap:
                        break
                    members.append(i)
                    load += int(d[i])
                key = tuple(sorted(members))
                if not key or key in out:
                    continue
                sub = Dc[np.ix_(key, key)]
                centre = tuple(P[list(key)].mean(axis=0))
                out[key] = Candidate(center=centre, members=key, load=load,
                                     radius=float(np.linalg.norm(P[list(key)] - np.array(centre), axis=1).max()),
                                     within_cost=float(np.triu(sub, 1).sum()),
                                     source=f"petal from {order[start]} dir {direction}")
        return list(out.values())

    def candidates(self, inst: Instance) -> list[Candidate]:
        pool: dict[tuple, Candidate] = {}
        if self.balls:
            for c in PackingModel.candidates(self.__class__.__mro__[1](fill=self.fill, grid=self.grid,
                                                                        max_candidates=10_000, max_jaccard=1.0), inst):
                pool.setdefault(c.members, c)
        if self.petals:
            for c in self.petal_candidates(inst):
                pool.setdefault(c.members, c)
        cands = sorted(pool.values(), key=lambda c: -self.score(inst, c))
        kept: list[Candidate] = []
        for c in cands:
            A = set(c.members)
            if all(len(A & set(k.members)) / len(A | set(k.members)) <= self.max_jaccard for k in kept):
                kept.append(c)
            if len(kept) >= self.max_atoms:
                break
        return kept

    def target(self, inst: Instance, cands: list[Candidate] | None = None):
        """As PackingModel.target, plus per-atom weights = total savings of the candidate
        (used by the ideal-mwis sampler and by the DMM-weighted pulse)."""
        tgt, cands = PackingModel.target(self, inst, cands)
        if self.tours is not None:
            tgt.weights = np.array([self.savings(inst, c.members) for c in cands], float)
        return tgt, cands
