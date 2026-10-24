"""Instance generation, feasibility certification, validation and I/O (Task A2).

Tier parameters (N, K, Q, demand range) are fixed by the challenge brief, Table 1.
Seeds, customer locations, the depot position and individual demands are ours.

Feasibility guarantee
---------------------
Every generated instance ships with a *certificate*: one customer->vehicle assignment
that respects every vehicle's capacity. We obtain it by an exact bin-packing search on
the drawn demands (largest-demand-first DFS with equal-load symmetry breaking). Draws
with no feasible packing are rejected and the RNG moves on, so generation is still a
deterministic function of (tier, seed, options).

Conventions
-----------
* Customers are indexed 0..N-1. In distance matrices from `Instance.dist()` the depot
  is row/column 0 and customer i is row/column i+1.
* An *assignment* is a list of length N with vehicle indices in 0..K-1.
* A *solution* is a list of routes; each route is an ordered list of customer indices.
  Every route implicitly starts and ends at the depot.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

SQUARE = 100.0  # side of the square containing depot and customers

TIERS = {
    "S": dict(N=4, K=2, Q=7, dmin=1, dmax=4),
    "M": dict(N=10, K=3, Q=20, dmin=1, dmax=8),
    "L": dict(N=15, K=4, Q=30, dmin=1, dmax=10),
}

FEASIBILITY_METHOD = (
    "exact bin-packing search (largest-first DFS, equal-load symmetry breaking); "
    "draws without a feasible packing are rejected; the packing found is the certificate"
)


@dataclass
class Instance:
    tier: str
    seed: int
    N: int
    K: int
    Q: int
    depot: list  # [x, y]
    coords: list  # N x 2
    demands: list  # N ints
    certificate: list  # vehicle index per customer
    feasibility_method: str = FEASIBILITY_METHOD

    # ---- derived -----------------------------------------------------------
    @property
    def name(self) -> str:
        return f"{self.tier}-{self.seed:04d}"

    @property
    def xy(self) -> np.ndarray:
        return np.asarray(self.coords, dtype=float)

    @property
    def d(self) -> np.ndarray:
        return np.asarray(self.demands, dtype=int)

    @property
    def load_ratio(self) -> float:
        """rho = sum_i d_i / (K * Q)."""
        return float(self.d.sum() / (self.K * self.Q))

    @property
    def dmax_ratio(self) -> float:
        """d_max / Q."""
        return float(self.d.max() / self.Q)

    def points(self) -> np.ndarray:
        """(N+1) x 2 array; row 0 is the depot, row i+1 is customer i."""
        return np.vstack([np.asarray(self.depot, dtype=float)[None, :], self.xy])

    def dist(self) -> np.ndarray:
        """(N+1) x (N+1) Euclidean distance matrix including the depot (index 0)."""
        P = self.points()
        return np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)

    def customer_dist(self) -> np.ndarray:
        """N x N Euclidean distances between customers only."""
        return self.dist()[1:, 1:]

    # ---- I/O ---------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["load_ratio"] = self.load_ratio
        d["dmax_ratio"] = self.dmax_ratio
        return d

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Instance":
        d = json.loads(Path(path).read_text())
        d.pop("load_ratio", None)
        d.pop("dmax_ratio", None)
        return cls(**d)


# ---------------------------------------------------------------------------
# Feasibility (bin packing)
# ---------------------------------------------------------------------------
def find_feasible_assignment(demands: Sequence[int], K: int, Q: int) -> list | None:
    """Exact search for a capacity-feasible assignment of customers to K identical bins.

    Returns a list `assign` with assign[i] in 0..K-1, or None if no packing exists.
    Largest-first ordering plus "never try two bins with the same current load"
    (identical vehicles) keeps the search tiny at N <= 15.
    """
    demands = list(int(x) for x in demands)
    order = sorted(range(len(demands)), key=lambda i: -demands[i])
    loads = [0] * K
    assign = [-1] * len(demands)

    def rec(t: int) -> bool:
        if t == len(order):
            return True
        i = order[t]
        di = demands[i]
        tried = set()
        for k in range(K):
            if loads[k] + di > Q or loads[k] in tried:
                continue
            tried.add(loads[k])
            loads[k] += di
            assign[i] = k
            if rec(t + 1):
                return True
            loads[k] -= di
            assign[i] = -1
        return False

    return assign if rec(0) else None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate(
    tier: str,
    seed: int,
    depot: Sequence[float] = (50.0, 50.0),
    rho_range: tuple[float, float] | None = None,
    max_draws: int = 100_000,
) -> Instance:
    """Draw a certified-feasible instance of `tier` from `seed`.

    rho_range: optional (lo, hi) band for the load ratio; draws outside it are
    rejected (still deterministic in the seed). Use e.g. (0.8, 0.95) for instances
    where capacity actually binds.
    """
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; choose from {list(TIERS)}")
    p = TIERS[tier]
    rng = np.random.default_rng(seed)
    for _ in range(max_draws):
        coords = rng.uniform(0.0, SQUARE, size=(p["N"], 2))
        demands = rng.integers(p["dmin"], p["dmax"] + 1, size=p["N"])
        rho = demands.sum() / (p["K"] * p["Q"])
        if rho_range is not None and not (rho_range[0] <= rho <= rho_range[1]):
            continue
        cert = find_feasible_assignment(demands.tolist(), p["K"], p["Q"])
        if cert is None:
            continue
        return Instance(
            tier=tier,
            seed=int(seed),
            N=p["N"],
            K=p["K"],
            Q=p["Q"],
            depot=[float(depot[0]), float(depot[1])],
            coords=np.round(coords, 4).tolist(),
            demands=demands.tolist(),
            certificate=cert,
        )
    raise RuntimeError(f"no feasible instance found for tier {tier}, seed {seed} in {max_draws} draws")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def loads_of(inst: Instance, assignment: Sequence[int]) -> np.ndarray:
    loads = np.zeros(inst.K, dtype=int)
    for i, k in enumerate(assignment):
        loads[k] += inst.demands[i]
    return loads


def validate_assignment(inst: Instance, assignment: Sequence[int]) -> list[str]:
    """Errors for a customer->vehicle assignment (empty list = valid)."""
    errs = []
    if len(assignment) != inst.N:
        return [f"assignment has length {len(assignment)}, expected N={inst.N}"]
    for i, k in enumerate(assignment):
        if not (0 <= int(k) < inst.K):
            errs.append(f"customer {i} assigned to vehicle {k} outside 0..{inst.K-1}")
    if errs:
        return errs
    loads = loads_of(inst, assignment)
    for k in range(inst.K):
        if loads[k] > inst.Q:
            errs.append(f"vehicle {k} load {loads[k]} exceeds Q={inst.Q}")
    return errs


def validate_instance(inst: Instance) -> list[str]:
    """Check tier parameters, geometry, demands and the feasibility certificate."""
    errs = []
    p = TIERS.get(inst.tier)
    if p is None:
        return [f"unknown tier {inst.tier!r}"]
    for key in ("N", "K", "Q"):
        if getattr(inst, key) != p[key]:
            errs.append(f"{key}={getattr(inst, key)} but tier {inst.tier} requires {p[key]}")
    if len(inst.coords) != inst.N or any(len(c) != 2 for c in inst.coords):
        errs.append("coords must be N x 2")
    if len(inst.demands) != inst.N:
        errs.append("demands must have length N")
    pts = inst.points() if not errs else None
    if pts is not None and (pts.min() < 0 or pts.max() > SQUARE):
        errs.append(f"all points must lie in [0, {SQUARE}]^2")
    for i, di in enumerate(inst.demands):
        if not (p["dmin"] <= int(di) <= p["dmax"]):
            errs.append(f"demand {di} of customer {i} outside [{p['dmin']}, {p['dmax']}]")
        if int(di) > inst.Q:
            errs.append(f"demand {di} of customer {i} exceeds Q={inst.Q} (assumption 5)")
    errs += [f"certificate: {e}" for e in validate_assignment(inst, inst.certificate)]
    return errs


def route_cost(inst: Instance, route: Sequence[int], D: np.ndarray | None = None) -> float:
    """Depot -> route[0] -> ... -> route[-1] -> depot. Empty route costs 0."""
    if len(route) == 0:
        return 0.0
    D = inst.dist() if D is None else D
    idx = [0] + [int(c) + 1 for c in route] + [0]
    return float(sum(D[a, b] for a, b in zip(idx[:-1], idx[1:])))


def validate_solution(inst: Instance, routes: Sequence[Sequence[int]]) -> tuple[float | None, list[str]]:
    """Validate a full CVRP solution. Returns (total cost or None, errors)."""
    errs = []
    if len(routes) > inst.K:
        errs.append(f"{len(routes)} routes but only K={inst.K} vehicles")
    seen = {}
    for r_idx, r in enumerate(routes):
        for c in r:
            c = int(c)
            if not (0 <= c < inst.N):
                errs.append(f"route {r_idx}: unknown customer {c}")
            elif c in seen:
                errs.append(f"customer {c} appears in routes {seen[c]} and {r_idx}")
            else:
                seen[c] = r_idx
        load = sum(inst.demands[int(c)] for c in r if 0 <= int(c) < inst.N)
        if load > inst.Q:
            errs.append(f"route {r_idx} load {load} exceeds Q={inst.Q}")
    missing = sorted(set(range(inst.N)) - set(seen))
    if missing:
        errs.append(f"customers never served: {missing}")
    if errs:
        return None, errs
    D = inst.dist()
    return float(sum(route_cost(inst, r, D) for r in routes)), []


def assignment_to_clusters(assignment: Sequence[int], K: int) -> list[list[int]]:
    clusters = [[] for _ in range(K)]
    for i, k in enumerate(assignment):
        clusters[int(k)].append(i)
    return clusters


def canonical_clustering(assignment: Sequence[int], K: int) -> tuple:
    """Vehicles are identical: compare clusterings as sets of customer groups (Pitfall 9.3)."""
    return tuple(sorted(tuple(c) for c in assignment_to_clusters(assignment, K) if c))


# ---------------------------------------------------------------------------
# Convenience: generate a batch and save
# ---------------------------------------------------------------------------
def generate_batch(
    tier: str,
    seeds: Iterable[int],
    out_dir: str | Path,
    rho_range: tuple[float, float] | None = None,
    depot: Sequence[float] = (50.0, 50.0),
) -> list[Instance]:
    out = []
    for s in seeds:
        inst = generate(tier, s, depot=depot, rho_range=rho_range)
        errs = validate_instance(inst)
        if errs:
            raise RuntimeError(f"generated instance failed validation: {errs}")
        inst.save(Path(out_dir) / f"{inst.name}.json")
        out.append(inst)
    return out
