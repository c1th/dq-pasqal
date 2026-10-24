"""Deterministic classical CVRP clusterings, with no repair or exact completion.

Every returned customer is served once and each block respects capacity. A
heuristic can still need more than K blocks; callers must report that outcome
as infeasible rather than silently substituting the instance's certificate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .instances import Instance
from .reference import TourTable, build_tour_table


@dataclass
class BaselineResult:
    """Adapter for metrics.results_row; classical baselines use no atoms/shots."""

    clusters: list[list[int]]
    feasible: bool
    levels: list = field(default_factory=list)
    repaired: bool = False
    objective_kind: str = "routed"

    @property
    def max_atoms(self) -> int:
        return 0

    @property
    def min_feasibility_rate(self) -> float:
        return float(self.feasible)


def _check_demands(inst: Instance) -> None:
    if any(d < 0 or d > inst.Q for d in inst.demands):
        raise ValueError("each customer demand must lie in [0, Q]")


def sweep(inst: Instance, tours: TourTable | None = None) -> list[list[int]]:
    """Try every angular starting customer in both directions, route exactly.

    Select the least exact routed cost among feasible candidates. If no sweep
    respects K, return the candidate with fewest blocks, breaking ties by cost;
    its capacity-feasible blocks are retained for an honest infeasibility report.
    The runner supplies its shared tour table to avoid rebuilding it.
    """
    _check_demands(inst)
    if inst.N == 0:
        return []
    if tours is None:
        tours = build_tour_table(inst)
    order = sorted(range(inst.N), key=lambda i: (
        math.atan2(inst.coords[i][1] - inst.depot[1],
                   inst.coords[i][0] - inst.depot[0]), i))
    best = []
    best_key = (True, math.inf, math.inf)
    for direction in (1, -1):
        for start in range(inst.N):
            clusters = []
            block = []
            load = 0
            for offset in range(inst.N):
                customer = order[(start + direction * offset) % inst.N]
                demand = inst.demands[customer]
                if block and load + demand > inst.Q:
                    clusters.append(block)
                    block, load = [], 0
                block.append(customer)
                load += demand
            clusters.append(block)
            cost = sum(tours.cost_of(c) for c in clusters)
            infeasible = len(clusters) > inst.K
            key = (infeasible, len(clusters) if infeasible else 0, cost)
            if key < best_key:
                best, best_key = clusters, key
    return best


def clarke_wright(inst: Instance) -> list[list[int]]:
    """Parallel savings merges of capacity-feasible routes at their endpoints.

    Process all customer pairs in descending savings order (customer indices
    break ties deterministically). Stop once at most K routes remain, as in the
    issue specification. If no legal merge reaches K, return the remaining
    routes so the runner can explicitly report infeasibility; never repair them.
    """
    _check_demands(inst)
    distances = inst.dist()
    routes = {i: [i] for i in range(inst.N)}
    owner = list(range(inst.N))
    loads = list(inst.demands)
    savings = sorted(
        (-(distances[0, i + 1] + distances[0, j + 1]
           - distances[i + 1, j + 1]), i, j)
        for i in range(inst.N) for j in range(i + 1, inst.N))
    for _, i, j in savings:
        if len(routes) <= inst.K:
            break
        left_id, right_id = owner[i], owner[j]
        if left_id == right_id or loads[left_id] + loads[right_id] > inst.Q:
            continue
        left, right = routes[left_id], routes[right_id]
        if i not in (left[0], left[-1]) or j not in (right[0], right[-1]):
            continue
        # Orient the two routes so the proposed i--j edge joins their ends.
        merged = (left if left[-1] == i else left[::-1]) + (
            right if right[0] == j else right[::-1])
        routes[left_id] = merged
        loads[left_id] += loads[right_id]
        del routes[right_id]
        for customer in merged:
            owner[customer] = left_id
    return list(routes.values())
