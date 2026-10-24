"""Exact reference values (Section 9.3) and exact sub-solvers used by the metrics.

Everything here is a *proven optimum* (brute force in the brief's hierarchy), not a
best-known solution:

* `TourTable`   Held-Karp dynamic programme over every subset of customers. Gives the
                exact depot-anchored TSP tour cost (and the number of distinct optimal
                tours) for every possible cluster at once. 2^N * N states: instant on
                S/M, a few seconds on L.
* `solve_partition`  Minimum-cost partition of the customers into <= K capacity-feasible
                blocks, with an arbitrary additive block cost. Solved as a
                set-partitioning integer programme over the COMPLETE pool of feasible
                subsets (scipy/HiGHS), so the optimum is proven. Used for
                  - the CVRP reference  (block cost = exact tour cost of the block),
                  - the clustering-model optimum (block cost = within-block pairwise
                    distance, i.e. our clustering objective),
                  - "routed exactly" costs of any given clustering.
* `enumerate_partitions`  Brute-force enumeration of all set partitions into <= K blocks,
                for S and M. Cross-checks the MILP (the brief asks for two methods to
                agree on tier S) and counts distinct optimal solutions, treating
                clusterings that differ only by vehicle relabelling as the same.

Tie tolerance for counting optima is `TOL` (absolute, in distance units).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .instances import Instance

TOL = 1e-7


# ---------------------------------------------------------------------------
# Subset helpers
# ---------------------------------------------------------------------------
def mask_of(customers: Sequence[int]) -> int:
    m = 0
    for c in customers:
        m |= 1 << int(c)
    return m


def customers_of(mask: int, N: int) -> list[int]:
    return [i for i in range(N) if mask >> i & 1]


def subset_loads(demands: np.ndarray) -> np.ndarray:
    """load[mask] for every mask over N customers, via lowest-set-bit recursion."""
    N = len(demands)
    load = np.zeros(1 << N, dtype=np.int64)
    for mask in range(1, 1 << N):
        low = mask & (-mask)
        load[mask] = load[mask ^ low] + demands[low.bit_length() - 1]
    return load


# ---------------------------------------------------------------------------
# Held-Karp over all subsets (depot-anchored tours)
# ---------------------------------------------------------------------------
@dataclass
class TourTable:
    N: int
    D: np.ndarray                 # (N+1)x(N+1), depot = 0
    cost: np.ndarray              # tour cost per mask (inf for mask 0 -> 0 by convention)
    count: np.ndarray             # number of distinct optimal undirected tours per mask
    _dp: np.ndarray = field(repr=False)
    _parent: np.ndarray = field(repr=False)

    def tour(self, mask: int) -> list[int]:
        """One optimal tour (customer order) for the cluster `mask`."""
        if mask == 0:
            return []
        closing = self._dp[mask] + self.D[1:, 0]
        j = int(np.argmin(np.where(np.isfinite(self._dp[mask]), closing, np.inf)))
        order = []
        m = mask
        while j >= 0:
            order.append(j)
            pj = int(self._parent[m, j])
            m ^= 1 << j
            j = pj
        return order[::-1]

    def tour_of(self, customers: Sequence[int]) -> list[int]:
        return self.tour(mask_of(customers))

    def cost_of(self, customers: Sequence[int]) -> float:
        return float(self.cost[mask_of(customers)])


def build_tour_table(inst: Instance) -> TourTable:
    D = inst.dist()
    N = inst.N
    full = 1 << N
    INF = np.inf
    dp = np.full((full, N), INF)
    cnt = np.zeros((full, N), dtype=np.int64)
    par = np.full((full, N), -1, dtype=np.int16)
    Dc = D[1:, 1:]  # customer-customer
    for j in range(N):
        dp[1 << j, j] = D[0, j + 1]
        cnt[1 << j, j] = 1
    for mask in range(1, full):
        if mask & (mask - 1) == 0:
            continue  # single customer: initialised above
        bits = [j for j in range(N) if mask >> j & 1]
        bits_arr = np.array(bits)
        for j in bits:
            prev = mask ^ (1 << j)
            ks = bits_arr[bits_arr != j]
            vals = dp[prev, ks] + Dc[ks, j]
            m = vals.min()
            dp[mask, j] = m
            sel = ks[np.abs(vals - m) <= TOL]
            cnt[mask, j] = cnt[prev, sel].sum()
            par[mask, j] = sel[0]
    cost = np.zeros(full)
    count = np.zeros(full, dtype=np.int64)
    count[0] = 1
    for mask in range(1, full):
        closing = dp[mask] + D[1:, 0]
        m = np.min(closing)
        cost[mask] = m
        sel = np.abs(closing - m) <= TOL
        directed = int(cnt[mask, sel].sum())
        # an undirected tour of >= 2 customers is counted once per direction
        count[mask] = directed if (mask & (mask - 1) == 0) else max(directed // 2, 1)
    return TourTable(N=N, D=D, cost=cost, count=count, _dp=dp, _parent=par)


# ---------------------------------------------------------------------------
# Block costs
# ---------------------------------------------------------------------------
def pairwise_block_costs(inst: Instance, W: np.ndarray | None = None) -> np.ndarray:
    """cost[mask] = sum_{i<j in mask} W[i,j]; W defaults to customer distances.

    This is our clustering objective ("total pairwise distance within a cluster").
    """
    N = inst.N
    W = inst.customer_dist() if W is None else np.asarray(W, float)
    cost = np.zeros(1 << N)
    for mask in range(1, 1 << N):
        low = mask & (-mask)
        i = low.bit_length() - 1
        rest = mask ^ low
        cost[mask] = cost[rest] + sum(W[i, j] for j in customers_of(rest, N))
    return cost


# ---------------------------------------------------------------------------
# Exact partition solver (set partitioning MILP over the complete column pool)
# ---------------------------------------------------------------------------
@dataclass
class PartitionResult:
    cost: float
    blocks: list[list[int]]       # customer lists (<= K of them)
    status: str                   # "optimal" (proven) or solver message
    n_columns: int


def feasible_masks(inst: Instance, load: np.ndarray | None = None) -> np.ndarray:
    load = subset_loads(inst.d) if load is None else load
    masks = np.arange(1, 1 << inst.N)
    return masks[load[masks] <= inst.Q]


def solve_partition(inst: Instance, block_cost: np.ndarray, time_limit: float | None = None) -> PartitionResult:
    """min sum_B block_cost[B] over partitions of all customers into <= K feasible blocks.

    Proven optimal: every capacity-feasible subset is a column. Blocks with infinite
    cost are excluded.
    """
    N, K = inst.N, inst.K
    masks = feasible_masks(inst)
    masks = masks[np.isfinite(block_cost[masks])]
    c = block_cost[masks].astype(float)
    ncol = len(masks)
    # coverage rows: customer i is covered by column b iff bit i of masks[b]
    rows, cols = [], []
    for b, m in enumerate(masks):
        for i in range(N):
            if m >> i & 1:
                rows.append(i)
                cols.append(b)
    A_cov = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, ncol)).tocsr()
    A_cnt = coo_matrix((np.ones(ncol), (np.zeros(ncol, int), np.arange(ncol))), shape=(1, ncol)).tocsr()
    constraints = [
        LinearConstraint(A_cov, lb=np.ones(N), ub=np.ones(N)),
        LinearConstraint(A_cnt, lb=0, ub=K),
    ]
    opts = {"disp": False}
    if time_limit is not None:
        opts["time_limit"] = float(time_limit)
    res = milp(c, constraints=constraints, integrality=np.ones(ncol), bounds=Bounds(0, 1), options=opts)
    if res.x is None:
        return PartitionResult(cost=float("inf"), blocks=[], status=str(res.message), n_columns=ncol)
    chosen = masks[res.x > 0.5]
    blocks = [customers_of(int(m), N) for m in chosen]
    status = "optimal" if res.status == 0 else str(res.message)
    return PartitionResult(cost=float(res.fun), blocks=blocks, status=status, n_columns=ncol)


# ---------------------------------------------------------------------------
# Brute-force enumeration of set partitions (S and M), with counting
# ---------------------------------------------------------------------------
@dataclass
class EnumResult:
    cost: float
    blocks: list[list[int]]
    n_partitions_checked: int
    n_optimal_clusterings: int    # clusterings (as sets of groups) attaining the optimum
    n_optimal_solutions: int      # clusterings x optimal tours per block (undirected)


def enumerate_partitions(inst: Instance, block_cost: np.ndarray, tours: TourTable | None = None,
                         max_partitions: int = 2_000_000) -> EnumResult:
    """Enumerate every partition of the customers into <= K capacity-feasible blocks.

    Uses restricted-growth strings (blocks are created in order of their smallest
    customer), so each clustering is visited exactly once regardless of vehicle labels.
    """
    N, K, Q = inst.N, inst.K, inst.Q
    d = inst.demands
    best = float("inf")
    best_blocks: list[list[int]] = []
    n_checked = 0
    n_opt_clusterings = 0
    n_opt_solutions = 0
    labels = [0] * N
    loads = [0] * K
    masks = [0] * K

    def rec(i: int, nblocks: int):
        nonlocal best, best_blocks, n_checked, n_opt_clusterings, n_opt_solutions
        if n_checked > max_partitions:
            raise RuntimeError("enumeration too large; use solve_partition instead")
        if i == N:
            n_checked += 1
            cost = sum(block_cost[masks[b]] for b in range(nblocks))
            if cost < best - TOL:
                best = cost
                best_blocks = [customers_of(masks[b], N) for b in range(nblocks)]
                n_opt_clusterings = 1
                n_opt_solutions = int(np.prod([tours.count[masks[b]] for b in range(nblocks)])) if tours else 1
            elif abs(cost - best) <= TOL:
                n_opt_clusterings += 1
                n_opt_solutions += int(np.prod([tours.count[masks[b]] for b in range(nblocks)])) if tours else 1
            return
        for b in range(min(nblocks + 1, K)):
            if loads[b] + d[i] > Q:
                continue
            loads[b] += d[i]
            masks[b] |= 1 << i
            labels[i] = b
            rec(i + 1, max(nblocks, b + 1))
            loads[b] -= d[i]
            masks[b] ^= 1 << i

    rec(0, 0)
    return EnumResult(best, best_blocks, n_checked, n_opt_clusterings, n_opt_solutions)


# ---------------------------------------------------------------------------
# Reference bundle for an instance
# ---------------------------------------------------------------------------
@dataclass
class Reference:
    inst_name: str
    tours: TourTable
    cvrp_cost: float                      # C* for the full CVRP (proven optimum)
    cvrp_blocks: list[list[int]]
    cvrp_routes: list[list[int]]
    cvrp_status: str
    n_optimal_cvrp_clusterings: int | None
    n_optimal_cvrp_solutions: int | None
    model_cost: float                     # optimum of the constrained clustering model
    model_blocks: list[list[int]]
    model_routed_exactly: float           # decomposition reference: model optimum, routed exactly
    pair_cost: np.ndarray = field(repr=False)   # clustering objective per block mask

    def routed_exactly(self, clusters: Sequence[Sequence[int]]) -> float:
        return float(sum(self.tours.cost_of(c) for c in clusters if len(c)))

    def clustering_objective(self, clusters: Sequence[Sequence[int]]) -> float:
        return float(sum(self.pair_cost[mask_of(c)] for c in clusters if len(c)))

    def summary(self) -> dict:
        return dict(
            instance=self.inst_name,
            cvrp_ref=self.cvrp_cost,
            cvrp_ref_status="proven optimum (complete set-partition MILP + Held-Karp)",
            n_optimal_cvrp_clusterings=self.n_optimal_cvrp_clusterings,
            n_optimal_cvrp_solutions=self.n_optimal_cvrp_solutions,
            model_opt=self.model_cost,
            model_opt_routed_exactly=self.model_routed_exactly,
            decomposition_gap_pct=100.0 * (self.model_routed_exactly - self.cvrp_cost) / self.cvrp_cost,
        )


def compute_reference(inst: Instance, pair_weights: np.ndarray | None = None,
                      cross_check: bool | None = None) -> Reference:
    """CVRP reference + clustering-model optimum for one instance.

    pair_weights: N x N matrix defining the clustering objective (default: distances).
    cross_check: also brute-force enumerate (default: tiers S and M), assert agreement,
                 and count distinct optima.
    """
    tours = build_tour_table(inst)
    pair_cost = pairwise_block_costs(inst, pair_weights)
    cvrp = solve_partition(inst, tours.cost)
    model = solve_partition(inst, pair_cost)
    if cross_check is None:
        cross_check = inst.N <= 10
    n_clusterings = n_solutions = None
    if cross_check:
        enum = enumerate_partitions(inst, tours.cost, tours)
        if abs(enum.cost - cvrp.cost) > 1e-6:
            raise AssertionError(f"brute force {enum.cost} != MILP {cvrp.cost} on {inst.name}")
        enum_model = enumerate_partitions(inst, pair_cost)
        if abs(enum_model.cost - model.cost) > 1e-6:
            raise AssertionError(f"model brute force {enum_model.cost} != MILP {model.cost} on {inst.name}")
        n_clusterings, n_solutions = enum.n_optimal_clusterings, enum.n_optimal_solutions
    routes = [tours.tour_of(b) for b in cvrp.blocks]
    return Reference(
        inst_name=inst.name,
        tours=tours,
        cvrp_cost=cvrp.cost,
        cvrp_blocks=cvrp.blocks,
        cvrp_routes=routes,
        cvrp_status=cvrp.status,
        n_optimal_cvrp_clusterings=n_clusterings,
        n_optimal_cvrp_solutions=n_solutions,
        model_cost=model.cost,
        model_blocks=model.blocks,
        model_routed_exactly=float(sum(tours.cost_of(b) for b in model.blocks)),
        pair_cost=pair_cost,
    )
