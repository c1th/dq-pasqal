"""Step 4b: exact reference values on tier M (classical).

For each tier-M instance (seeds 0-59, as in Step 4a):
  * tour[S]  : optimal depot tour of every customer subset S (Held-Karp dynamic program),
  * CVRP     : optimum with the full fleet K, by an exact set-partition DP over capacity-feasible subsets
               (each partition counted once, so the number of distinct optimal solutions is unlabeled),
               cross-checked against enumeration of all K^N labelled assignments,
  * model    : optimum f_opt of the clustering model with K' = K_min, and the number of distinct optimal clusterings,
  * decomposition gap: an f-optimal clustering routed exactly, against the CVRP optimum, for K' = K_min and K' = K.
All values are proven optima (exhaustive), so every gap computed against them is exact.
Requires instances.py and step1_qubo_check.py.
"""
import csv, itertools, math
import numpy as np
from instances import generate, validate, DEPOT
from step1_qubo_check import distances

TIER, SEEDS = "M", range(60)


def held_karp(P):
    """tour[mask] = cost of the optimal tour depot -> customers in mask -> depot. P[0] is the depot."""
    N = len(P) - 1
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    full = 1 << N
    dp = np.full((full, N), np.inf)
    for j in range(N):
        dp[1 << j, j] = D[0, j + 1]
    for S in range(1, full):
        for j in range(N):
            if not (S >> j) & 1 or dp[S, j] == np.inf:
                continue
            base = dp[S, j]
            for k in range(N):
                if (S >> k) & 1:
                    continue
                T = S | (1 << k)
                v = base + D[j + 1, k + 1]
                if v < dp[T, k]:
                    dp[T, k] = v
    tour = np.zeros(full)
    for S in range(1, full):
        tour[S] = min(dp[S, j] + D[j + 1, 0] for j in range(N) if (S >> j) & 1)
    return tour


def partition_dp(cost, feasible, N, K):
    """Min total cost of covering all customers with at most K disjoint feasible subsets, and the number of
    distinct (unlabelled) optimal partitions. Each block is anchored on its lowest customer, so each partition
    is generated exactly once."""
    full = 1 << N
    best = [np.full(full, np.inf) for _ in range(K + 1)]
    cnt = [np.zeros(full, dtype=np.int64) for _ in range(K + 1)]
    for k in range(K + 1):
        best[k][0], cnt[k][0] = 0.0, 1
    for k in range(1, K + 1):
        for S in range(1, full):
            low = S & -S
            T = S
            b, c = np.inf, 0
            while T:
                if (T & low) and feasible[T] and best[k - 1][S ^ T] < np.inf:
                    v = cost[T] + best[k - 1][S ^ T]
                    if v < b - 1e-9:
                        b, c = v, cnt[k - 1][S ^ T]
                    elif abs(v - b) <= 1e-9:
                        c += cnt[k - 1][S ^ T]
                T = (T - 1) & S
            best[k][S], cnt[k][S] = b, c
    return best[K][full - 1], int(cnt[K][full - 1])


def masks_of(assign, Kp):
    """Customer bitmask of each vehicle, for an array of labelled assignments."""
    w = 1 << np.arange(assign.shape[1])
    return np.stack([((assign == k) * w).sum(1) for k in range(Kp)], axis=1)


def canonical(a):
    """Relabel vehicles by first appearance, so relabelled clusterings compare equal."""
    m, out = {}, []
    for v in a:
        m.setdefault(v, len(m)); out.append(m[v])
    return tuple(out)


def main():
    rows = []
    for seed in SEEDS:
        inst = generate(TIER, seed); validate(inst)
        d, Q, K, Kp = np.asarray(inst["d"]), inst["Q"], inst["K"], inst["K_min"]; N = len(d)
        c = distances(inst["xy"])
        P = np.vstack([DEPOT, inst["xy"]])
        tour = held_karp(P)
        full = 1 << N
        load = np.array([d[[i for i in range(N) if (S >> i) & 1]].sum() for S in range(full)])
        feasible = load <= Q
        cvrp_opt, n_cvrp_opt = partition_dp(tour, feasible, N, K)
        # cross-check by enumerating all K^N labelled assignments
        A = np.array(list(itertools.product(range(K), repeat=N)))
        MK = masks_of(A, K)
        okK = feasible[MK].all(1)
        routedK = tour[MK].sum(1)
        assert math.isclose(routedK[okK].min(), cvrp_opt, rel_tol=1e-9), "CVRP DP and enumeration disagree"

        iu = np.triu_indices(N, 1)

        def model(Kp_):
            Ap = np.array(list(itertools.product(range(Kp_), repeat=N)))
            Mp = masks_of(Ap, Kp_)
            ok = feasible[Mp].all(1)
            same = Ap[:, :, None] == Ap[:, None, :]
            f = (same[:, iu[0], iu[1]] * c[iu]).sum(1)
            fo = f[ok].min()
            opt = ok & np.isclose(f, fo, rtol=0, atol=1e-9)
            n_distinct = len({canonical(a) for a in Ap[opt]})
            routed = tour[Mp].sum(1)[opt]
            return fo, n_distinct, routed.min(), routed.max()

        f_opt, n_f_opt, r_min, r_max = model(Kp)
        fK, _, rK_min, rK_max = model(K)
        gap = lambda C: 100.0 * (C - cvrp_opt) / cvrp_opt
        rows.append(dict(seed=seed, K=K, K_min=Kp, rho=round(inst["rho"], 3), eff_load=round(d.sum() / (Kp * Q), 3),
                         f_opt=round(f_opt, 6), n_f_opt=n_f_opt, cvrp_opt=round(cvrp_opt, 6), n_cvrp_opt=n_cvrp_opt,
                         decomp_gap_Kmin=round(gap(r_min), 3), decomp_gap_Kmin_worst_tie=round(gap(r_max), 3),
                         decomp_gap_K=round(gap(rK_min), 3), decomp_gap_K_worst_tie=round(gap(rK_max), 3),
                         ref_status="proven optimum (exhaustive)"))
        print(f"seed {seed} (K'={Kp}) done", flush=True)

    with open("step4b_references.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nTier {TIER}: {len(rows)} instances; CVRP optimum (DP) matched enumeration of all K^N assignments on every instance.")
    for Kp in (2, 3):
        sub = [r for r in rows if r["K_min"] == Kp]
        if not sub:
            continue
        g_min = [r["decomp_gap_Kmin"] for r in sub]; g_K = [r["decomp_gap_K"] for r in sub]
        print(f"K_min = {Kp} ({len(sub)} instances): decomposition gap with K' = K_min: mean {np.mean(g_min):.2f}%, "
              f"median {np.median(g_min):.2f}%, max {np.max(g_min):.2f}%;  with K' = K: mean {np.mean(g_K):.2f}%, "
              f"median {np.median(g_K):.2f}%, max {np.max(g_K):.2f}%;  instances with a unique CVRP optimum: "
              f"{sum(r['n_cvrp_opt'] == 1 for r in sub)}/{len(sub)}")


if __name__ == "__main__":
    main()