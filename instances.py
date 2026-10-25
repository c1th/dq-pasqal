"""Instance generation and feasibility certification (Track A).

Certificate = an explicit assignment of customers to exactly K_min non-empty vehicles,
satisfying (C1) and (C2). Since K_min <= K, the same assignment (other vehicles empty)
is a valid plan for the full instance, so it certifies feasibility.
Minimality of K_min is certified by either the lower bound ceil(D/Q) or an independent
exhaustive check that K_min - 1 vehicles cannot pack the demands.
"""
import math
import numpy as np

TIERS = {"S": dict(N=4, K=2, Q=7, dmax=4),
         "M": dict(N=10, K=3, Q=20, dmax=8),
         "L": dict(N=15, K=4, Q=30, dmax=10)}
DEPOT = (50.0, 50.0)


def min_bins(d, Q):
    """Exact bin packing by the O(N 2^N) subset DP.
    dp[S] = lexicographic min of (bins used, load of current bin) to pack set S.
    Returns (K_min, assignment list: customer index -> bin index 0..K_min-1)."""
    N = len(d)
    INF = (math.inf, math.inf)
    dp = [INF] * (1 << N)
    parent = [-1] * (1 << N)
    dp[0] = (1, 0)                       # one open, empty bin
    for S in range(1 << N):
        if dp[S] == INF:
            continue
        b, load = dp[S]
        for i in range(N):
            if S >> i & 1:
                continue
            nxt = (b, load + d[i]) if load + d[i] <= Q else (b + 1, d[i])
            T = S | (1 << i)
            if nxt < dp[T]:
                dp[T], parent[T] = nxt, i
    # reconstruct insertion order, then replay it to get bins
    order, S = [], (1 << N) - 1
    while S:
        i = parent[S]
        order.append(i)
        S ^= 1 << i
    order.reverse()
    assign, b, load = [None] * N, 0, 0
    for i in order:
        if load + d[i] > Q:
            b, load = b + 1, 0
        assign[i] = b
        load += d[i]
    return dp[-1][0], assign


def can_pack(d, Q, bins):
    """Independent check (backtracking): can demands d be packed into `bins` bins of capacity Q?"""
    items = sorted(d, reverse=True)
    loads = [0] * bins

    def place(t):
        if t == len(items):
            return True
        seen = set()
        for b in range(bins):
            if loads[b] + items[t] <= Q and loads[b] not in seen:
                seen.add(loads[b])
                loads[b] += items[t]
                if place(t + 1):
                    return True
                loads[b] -= items[t]
        return False

    return place(0)


def generate(tier, seed, rho_band=(0.45, 0.95), max_tries=10_000):
    """Seeded rejection sampling. Accept iff 2 <= K_min <= K and rho in band."""
    p = TIERS[tier]
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        d = rng.integers(1, p["dmax"] + 1, size=p["N"]).tolist()
        rho = sum(d) / (p["K"] * p["Q"])
        if not (rho_band[0] <= rho <= rho_band[1]):
            continue
        K_min, cert = min_bins(d, p["Q"])
        if not (2 <= K_min <= p["K"]):   # infeasible (K_min > K) or trivial (K_min = 1)
            continue
        xy = rng.uniform(0, 100, size=(p["N"], 2)).tolist()
        return dict(tier=tier, seed=seed, N=p["N"], K=p["K"], Q=p["Q"], depot=DEPOT,
                    xy=xy, d=d, K_min=K_min, certificate=cert,
                    rho=rho, dmax_over_Q=max(d) / p["Q"])
    raise RuntimeError("no instance found; widen rho_band")


def validate(inst):
    """Independent validator. Raises AssertionError on any failure."""
    d, Q, K, Km, a = inst["d"], inst["Q"], inst["K"], inst["K_min"], inst["certificate"]
    N = len(d)
    # Section 2 assumptions
    assert all(1 <= di <= Q for di in d), "Assumption 5 violated"
    assert all(0 <= c <= 100 for xy in inst["xy"] for c in xy), "location outside square"
    # Certificate: (C1) each customer in exactly one bin in range; (C2) loads <= Q
    assert len(a) == N and all(isinstance(k, int) and 0 <= k < Km for k in a), "C1 violated"
    loads = [sum(d[i] for i in range(N) if a[i] == k) for k in range(Km)]
    assert all(L <= Q for L in loads), "C2 violated"
    assert all(L > 0 for L in loads), "certificate must use exactly K_min vehicles"
    assert 2 <= Km <= K, "K_min out of range"          # => feasible for the full fleet K
    # Minimality of K_min: lower bound, else independent exhaustive check
    D = sum(d)
    if Km > math.ceil(D / Q):
        assert not can_pack(d, Q, Km - 1), "K_min not minimal"
    return True


if __name__ == "__main__":
    # self-test 1: the brief's counterexample (3 customers of demand 2, capacity 3)
    assert min_bins([2, 2, 2], 3)[0] == 3 and not can_pack([2, 2, 2], 3, 2)
    # self-test 2: DP agrees with backtracking on random small cases
    rng = np.random.default_rng(0)
    for _ in range(2000):
        n, Q = int(rng.integers(1, 9)), int(rng.integers(3, 15))
        d = rng.integers(1, Q + 1, size=n).tolist()
        k, cert = min_bins(d, Q)
        assert can_pack(d, Q, k) and (k == 1 or not can_pack(d, Q, k - 1))
        assert all(sum(d[i] for i in range(n) if cert[i] == b) <= Q for b in range(k))
    # self-test 3: generate and validate instances on every tier
    gap = 0
    for tier in "SML":
        for seed in range(30):
            inst = generate(tier, seed)
            validate(inst)
            gap += inst["K_min"] > math.ceil(sum(inst["d"]) / inst["Q"])
    print("all checks passed; instances with K_min > ceil(D/Q):", gap)