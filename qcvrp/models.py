"""Clustering models (Task A1): what the register is asked to minimise.

The device minimises  E(n) = -delta * sum_a n_a + sum_{a<b} U_ab n_a n_b  with a GLOBAL
delta and U_ab = C6 / r_ab^6 >= 0.  A model therefore has to be expressed as

    * a set of binary variables (atoms),
    * non-negative pairwise "wanted" couplings J_ab (the only thing the layout can encode),
    * one-hot groups: sets of atoms that must be mutually blockaded (at most one excited),
    * a decoder from bitstrings to customer->vehicle assignments.

Same-vehicle pair weight used by both models
--------------------------------------------
    w_ij = alpha * dist_ij / SQUARE  +  beta * d_i * d_j / Q^2

The distance part is the clustering objective (far customers should not share a
vehicle).  The demand-product part is what survives of a quadratic load penalty once
the one-hot constraint is exact: for sum_k (L_k - T)^2 with L_k = sum_i d_i x_ik, every
linear term is a constant on the one-hot manifold and the only assignment-dependent
term is 2 * sum_k sum_{i<j} d_i d_j x_ik x_jk ("heavy customers repel within a
vehicle").  No slack variables, all couplings positive.  It enforces *balance*, not an
inequality at Q, so capacity feasibility is measured (feasibility rate) and, when
needed, repaired by the stated greedy step below.

Models
------
OneHotModel      K atoms per customer, atom (i,k) excited <=> customer i on vehicle k.
                 One-hot is enforced by geometry (blockade) + positive final detuning.
                 J_(i,k),(j,k) = w_ij ; no wanted coupling across different vehicles.
                 N*K atoms: 8 / 30 / 60 on S / M / L.  No slack.

BisectionModel   K = 2 only. One atom per customer, excited <=> vehicle 1.
                 Same-vehicle cost sum_{i<j} w_ij [x_i == x_j] expands to
                 const - sum_i (sum_j w_ij) x_i + 2 sum_{i<j} w_ij x_i x_j,
                 so J_ij = 2 w_ij and the wanted linear term is -(row sum of w),
                 which a global detuning can only approximate by its mean.
                 N atoms: 4 / 10 / 15 (exact state-vector emulation on every tier).
                 For K > 2 it is applied recursively (not implemented here yet).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .instances import SQUARE, Instance, loads_of, validate_assignment


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class CouplingTarget:
    n_atoms: int
    J: np.ndarray                      # wanted couplings, symmetric, >= 0, zero diagonal
    h: np.ndarray                      # wanted linear terms (informational)
    groups: list[list[int]]            # one-hot groups
    labels: list[str]
    note: str = ""
    kind: str = "qubo"                 # "qubo" | "onehot" | "mis" (drives the detuning rule)
    init_positions: np.ndarray | None = None   # optional geometric start for the embedder (n x 2)

    def group_of(self) -> np.ndarray:
        g = -np.ones(self.n_atoms, dtype=int)
        for gi, atoms in enumerate(self.groups):
            g[atoms] = gi
        return g


@dataclass
class Decoded:
    assignment: list | None          # vehicle per customer (None if undecodable before repair)
    onehot_ok: bool
    capacity_ok: bool
    n_onehot_violations: int = 0
    repaired: bool = False
    repair_failed: bool = False

    @property
    def feasible(self) -> bool:
        return self.assignment is not None and self.onehot_ok and self.capacity_ok


def bits_to_array(bits) -> np.ndarray:
    if isinstance(bits, str):
        return np.array([int(c) for c in bits], dtype=int)
    return np.asarray(bits, dtype=int)


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------
def pair_weights(inst: Instance, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    """w_ij = alpha * dist_ij/SQUARE + beta * d_i d_j / Q^2 (zero diagonal)."""
    Dc = inst.customer_dist() / SQUARE
    d = inst.d.astype(float)
    W = alpha * Dc + beta * np.outer(d, d) / (inst.Q ** 2)
    np.fill_diagonal(W, 0.0)
    return W


def same_vehicle_cost(assignment: Sequence[int], W: np.ndarray) -> float:
    """sum_k sum_{i<j in k} W_ij for any symmetric pair-weight matrix."""
    a = np.asarray(assignment)
    same = (a[:, None] == a[None, :]).astype(float)
    return float(np.triu(W * same, 1).sum())


def clustering_objective(assignment: Sequence[int], inst: Instance) -> float:
    """The constrained clustering model's objective: within-cluster pairwise distance."""
    return same_vehicle_cost(assignment, inst.customer_dist())


def greedy_repair(assignment: list, inst: Instance, excited: list[list[int]] | None = None) -> tuple[list | None, bool]:
    """Stated classical post-processing (Pitfall 9.1).

    1. customers with no / several excited vehicle atoms are placed one by one on the
       vehicle minimising their marginal within-cluster distance, capacity permitting
       (several excited: choose among the excited vehicles first);
    2. while a vehicle is overloaded, move the customer whose move costs the least extra
       within-cluster distance to a vehicle with enough slack.
    Returns (assignment, success).  Never touches a shot that is already feasible.
    """
    Dc = inst.customer_dist()
    N, K, Q = inst.N, inst.K, inst.Q
    d = inst.demands
    a = [None if x is None else int(x) for x in assignment]

    def marginal(i, k):
        return sum(Dc[i, j] for j in range(N) if a[j] == k and j != i)

    def loads():
        L = [0] * K
        for j in range(N):
            if a[j] is not None:
                L[a[j]] += d[j]
        return L

    # 1. place unassigned customers (largest demand first)
    pending = [i for i in range(N) if a[i] is None]
    for i in sorted(pending, key=lambda i: -d[i]):
        L = loads()
        candidates = (excited[i] if excited and excited[i] else list(range(K)))
        ok = [k for k in candidates if L[k] + d[i] <= Q] or [k for k in range(K) if L[k] + d[i] <= Q]
        if not ok:
            ok = list(range(K))  # fix in step 2 if possible
        a[i] = min(ok, key=lambda k: marginal(i, k))
    # 2. capacity repair by cheapest moves
    for _ in range(4 * N):
        L = loads()
        over = [k for k in range(K) if L[k] > Q]
        if not over:
            return a, True
        k = max(over, key=lambda k: L[k])
        best = None
        for i in [j for j in range(N) if a[j] == k]:
            for k2 in range(K):
                if k2 == k or L[k2] + d[i] > Q:
                    continue
                delta = marginal(i, k2) - marginal(i, k)
                if best is None or delta < best[0]:
                    best = (delta, i, k2)
        if best is None:
            return a, False
        a[best[1]] = best[2]
    return a, False


# ---------------------------------------------------------------------------
# One-hot (K atoms per customer)
# ---------------------------------------------------------------------------
@dataclass
class OneHotModel:
    alpha: float = 1.0
    beta: float = 0.0
    name: str = "onehot"

    def atom(self, i: int, k: int, K: int) -> int:
        return i * K + k

    def W(self, inst: Instance) -> np.ndarray:
        return pair_weights(inst, self.alpha, self.beta)

    def target(self, inst: Instance) -> CouplingTarget:
        N, K = inst.N, inst.K
        n = N * K
        W = self.W(inst)
        J = np.zeros((n, n))
        for i in range(N):
            for j in range(i + 1, N):
                for k in range(K):
                    a, b = self.atom(i, k, K), self.atom(j, k, K)
                    J[a, b] = J[b, a] = W[i, j]
        groups = [[self.atom(i, k, K) for k in range(K)] for i in range(N)]
        labels = [f"c{i}v{k}" for i in range(N) for k in range(K)]
        return CouplingTarget(n, J, np.zeros(n), groups, labels,
                              note="one-hot by blockade; uniform linear term; J=w on same-vehicle pairs", kind="onehot")

    def decode(self, bits, inst: Instance, repair: bool = False) -> Decoded:
        x = bits_to_array(bits)
        N, K = inst.N, inst.K
        assert len(x) == N * K, f"expected {N*K} bits, got {len(x)}"
        excited = [[k for k in range(K) if x[self.atom(i, k, K)]] for i in range(N)]
        n_viol = sum(1 for e in excited if len(e) != 1)
        if n_viol == 0:
            assignment = [e[0] for e in excited]
            cap_ok = not validate_assignment(inst, assignment)
            dec = Decoded(assignment, True, cap_ok)
        else:
            dec = Decoded(None, False, False, n_viol)
        if repair and not dec.feasible:
            partial = [e[0] if len(e) == 1 else None for e in excited]
            fixed, ok = greedy_repair(partial, inst, excited)
            dec.repaired = True
            if ok:
                dec.assignment, dec.onehot_ok, dec.capacity_ok = fixed, True, True
            else:
                dec.repair_failed = True
        return dec


# ---------------------------------------------------------------------------
# Bisection (one atom per customer, K = 2)
# ---------------------------------------------------------------------------
@dataclass
class BisectionModel:
    alpha: float = 1.0
    beta: float = 0.0
    name: str = "bisection"

    def W(self, inst: Instance) -> np.ndarray:
        return pair_weights(inst, self.alpha, self.beta)

    def target(self, inst: Instance) -> CouplingTarget:
        if inst.K != 2:
            raise ValueError("BisectionModel needs K = 2 (use it recursively for K > 2)")
        W = self.W(inst)
        J = 2.0 * W
        h = -W.sum(axis=1)
        labels = [f"c{i}" for i in range(inst.N)]
        return CouplingTarget(inst.N, J, h, [], labels,
                              note="x_i = vehicle bit; J = 2w; wanted h_i = -sum_j w_ij (device realises its mean)")

    def decode(self, bits, inst: Instance, repair: bool = False) -> Decoded:
        x = bits_to_array(bits)
        assert len(x) == inst.N
        assignment = [int(v) for v in x]
        cap_ok = not validate_assignment(inst, assignment)
        dec = Decoded(assignment, True, cap_ok)
        if repair and not cap_ok:
            fixed, ok = greedy_repair(assignment, inst)
            dec.repaired = True
            if ok:
                dec.assignment, dec.capacity_ok = fixed, True
            else:
                dec.repair_failed = True
        return dec


# ---------------------------------------------------------------------------
# Conflict split (the embeddable one): MIS on a sparse conflict graph, one atom per customer
# ---------------------------------------------------------------------------
@dataclass
class ConflictSplitModel:
    """Split a customer subset into an excited side S and its complement.

    Decision variables: x_i = 1 iff customer i goes to side S (one atom per customer).
    Conflict graph G: (i, j) is an edge iff j is among the m_far farthest customers of i
    (or vice versa) or d_i + d_j exceeds the capacity of the smaller side (exact
    pairwise capacity conflict).  Conflict pairs are placed inside the blockade radius,
    all other pairs outside it, so the register's ground state is a maximum independent
    set of G: the largest set containing no far pair = a compact half.  The residual
    1/r^6 tail between non-conflicting atoms is a soft preference for mutually close
    customers.  Everything is positive and sparse (degree ~ m_far), which is what a 2D
    register can realise.  Used hierarchically for K > 2 (see pipeline.hierarchical_cluster).
    """
    m_far: int = 2
    heavy_conflicts: bool = True
    mutual: bool = False            # True: edge only if each is among the other's m_far farthest
    name: str = "conflict-split"

    def conflict_edges(self, inst: Instance, customers: Sequence[int], Q_small: int) -> list[tuple[int, int]]:
        cs = list(customers)
        n = len(cs)
        Dc = inst.customer_dist()[np.ix_(cs, cs)]
        far_sets = [set(int(b) for b in np.argsort(-Dc[a])[: min(self.m_far, n - 1)] if b != a) for a in range(n)]
        edges = set()
        for a in range(n):
            for b in far_sets[a]:
                if self.mutual and a not in far_sets[b]:
                    continue
                edges.add((min(a, b), max(a, b)))
        if self.heavy_conflicts:
            d = inst.d[cs]
            for a in range(n):
                for b in range(a + 1, n):
                    if d[a] + d[b] > Q_small:
                        edges.add((a, b))
        return sorted(edges)

    def target(self, inst: Instance, customers: Sequence[int] | None = None, Q_small: int | None = None) -> CouplingTarget:
        cs = list(range(inst.N)) if customers is None else list(customers)
        Q_small = inst.Q if Q_small is None else Q_small
        n = len(cs)
        J = np.zeros((n, n))
        for a, b in self.conflict_edges(inst, cs, Q_small):
            J[a, b] = J[b, a] = 1.0
        labels = [f"c{c}" for c in cs]
        return CouplingTarget(n, J, np.zeros(n), [], labels,
                              note=f"MIS on conflict graph (m_far={self.m_far}, heavy={self.heavy_conflicts}); "
                                   f"{int(J.sum() // 2)} edges on {n} atoms", kind="mis")

    def decode(self, bits, customers: Sequence[int]) -> tuple[list[int], list[int]]:
        x = bits_to_array(bits)
        cs = list(customers)
        assert len(x) == len(cs)
        S = [c for c, v in zip(cs, x) if v]
        R = [c for c, v in zip(cs, x) if not v]
        return S, R


# ---------------------------------------------------------------------------
# Capacitated ball packing by MIS (one atom per candidate cluster)
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    center: tuple[float, float]
    members: tuple[int, ...]
    load: int
    radius: float          # distance from centre to the farthest member
    within_cost: float     # sum of pairwise distances among members
    source: str            # "customer i" or "grid"


@dataclass
class PackingModel:
    """Change what the model asks for: choose disjoint, compact, capacity-feasible clusters.

    Candidates: for every centre (each customer, optionally a grid), take customers in
    order of distance until the load would exceed fill * Q.  Every candidate is
    capacity-feasible BY CONSTRUCTION, so the inequality never reaches the device.
    Decision variable: one atom per candidate, excited <=> candidate chosen.
    Constraint: chosen candidates are pairwise disjoint (share no customer) -> blockade.
    Objective (uniform detuning): as many disjoint candidates as possible, which for
    near-full candidates means covering as many customers as possible with compact
    clusters.  Two candidates overlap only when their centres are close on the map, so
    the conflict graph is (close to) a unit-disk graph of the map itself: the register
    is the map at scale, which is what a neutral-atom array embeds best.
    Customers left uncovered (and any shot with overlapping or too many candidates) are
    handled by the stated classical step in pipeline.packing_cluster, with the covered
    fraction reported for every shot.
    """
    fill: float = 0.85
    grid: int = 0                  # add an n x n grid of extra centres (0 = customers only)
    max_candidates: int = 80
    max_jaccard: float = 0.6       # thin near-duplicate candidates (Jaccard overlap above this)
    extra_fills: tuple = ()        # additional fill levels per centre, e.g. (0.55,) for size diversity
    name: str = "ball-packing"

    def candidates(self, inst: Instance) -> list[Candidate]:
        P = inst.xy
        d = inst.d
        centres = [(tuple(P[i]), f"customer {i}") for i in range(inst.N)]
        if self.grid > 1:
            g = np.linspace(0, SQUARE, self.grid + 2)[1:-1]
            centres += [((float(x), float(y)), "grid") for x in g for y in g]
        Dc = inst.customer_dist()
        seen = {}
        for fill in (self.fill, *self.extra_fills):
            cap = fill * inst.Q
            for c, src in centres:
                dist = np.linalg.norm(P - np.array(c), axis=1)
                order = np.argsort(dist)
                members, load, misses = [], 0, 0
                for i in order:
                    if load + d[i] <= cap:
                        members.append(int(i))
                        load += int(d[i])
                    else:
                        misses += 1
                        if misses >= 2:
                            break
                key = tuple(sorted(members))
                if not key or key in seen:
                    continue
                sub = Dc[np.ix_(key, key)]
                seen[key] = Candidate(center=c, members=key, load=load, radius=float(dist[list(key)].max()),
                                      within_cost=float(np.triu(sub, 1).sum()), source=f"{src} fill={fill}")
        cands = list(seen.values())
        # most compact first; drop near-duplicates (they would form unrealisable cliques
        # in the overlap graph) and cap the count
        cands.sort(key=lambda c: c.within_cost / max(len(c.members), 1))
        kept: list[Candidate] = []
        for c in cands:
            A = set(c.members)
            if all(len(A & set(k.members)) / len(A | set(k.members)) <= self.max_jaccard for k in kept):
                kept.append(c)
            if len(kept) >= self.max_candidates:
                break
        return kept

    def target(self, inst: Instance, cands: list[Candidate] | None = None) -> tuple[CouplingTarget, list[Candidate]]:
        cands = self.candidates(inst) if cands is None else cands
        n = len(cands)
        J = np.zeros((n, n))
        sets = [set(c.members) for c in cands]
        for a in range(n):
            for b in range(a + 1, n):
                if sets[a] & sets[b]:
                    J[a, b] = J[b, a] = 1.0
        labels = [f"ball{a}@{c.source}" for a, c in enumerate(cands)]
        tgt = CouplingTarget(n, J, np.zeros(n), [], labels,
                             note=f"MIS on candidate-overlap graph: {n} candidates, {int(J.sum() // 2)} overlaps, fill={self.fill}",
                             kind="mis")
        tgt.init_positions = np.array([c.center for c in cands])  # map geometry as the embedding start
        return tgt, cands

    def decode(self, bits, cands: list[Candidate]) -> list[int]:
        x = bits_to_array(bits)
        return [a for a in range(len(cands)) if x[a]]


# ---------------------------------------------------------------------------
# Penalty-weight calibration (how we set beta)
# ---------------------------------------------------------------------------
def all_assignments(N: int, K: int):
    """Iterate over every assignment with vehicle 0 used first (relabelling symmetry removed)."""
    a = [0] * N

    def rec(i, used):
        if i == N:
            yield list(a)
            return
        for k in range(min(used + 1, K)):
            a[i] = k
            yield from rec(i + 1, max(used, k + 1))

    yield from rec(0, 0)


def calibrate_beta(inst: Instance, alpha: float = 1.0, betas: Sequence[float] = (0, 0.25, 0.5, 1, 2, 4, 8)) -> dict:
    """For each beta, minimise the *penalised* same-vehicle cost over ALL assignments and
    record whether the unconstrained minimiser is capacity-feasible and how it compares
    to the constrained optimum of the distance objective. Brute force: tiers S and M.
    """
    if inst.K ** inst.N > 5_000_000:
        raise ValueError("calibrate_beta is brute force; use tiers S or M")
    Dc = inst.customer_dist()
    out = {}
    assignments = list(all_assignments(inst.N, inst.K))
    feas = [not validate_assignment(inst, a) for a in assignments]
    dist_cost = np.array([same_vehicle_cost(a, Dc) for a in assignments])
    constrained_opt = dist_cost[np.array(feas)].min()
    for beta in betas:
        W = pair_weights(inst, alpha, beta)
        pen = np.array([same_vehicle_cost(a, W) for a in assignments])
        best = int(np.argmin(pen))
        out[beta] = dict(
            minimiser_feasible=bool(feas[best]),
            minimiser_dist_cost=float(dist_cost[best]),
            gap_to_constrained_pct=100.0 * (dist_cost[best] - constrained_opt) / constrained_opt,
            n_minimisers=int(np.sum(np.abs(pen - pen[best]) < 1e-9)),
        )
    return out
