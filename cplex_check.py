"""Tier-S cross-check: CPLEX vs brute force (brief, Sec. 9.3).

For each tier-S instance (seeds 0-19, as in results.csv), solve with CPLEX (docplex, community edition) and compare
with exhaustive enumeration:
  1. the constrained clustering model: min f(a) s.t. (C1), (C2), K' = K_min vehicles,
     with the products a_ik a_jk linearized by y_ijk >= a_ik + a_jk - 1, y_ijk >= 0;
  2. the full CVRP with the full fleet K: three-index arc formulation, (C1)-(C5) of the write-up, and
     Miller-Tucker-Zemlin load constraints instead of (C6) to eliminate subtours.
A mismatch would indicate a modelling or formulation error. Writes cplex_check.csv.
"""
import csv, itertools, math
import numpy as np
from docplex.mp.model import Model
from instances import generate, validate, DEPOT
from step1_qubo_check import distances, clustering_cost

TIER, SEEDS = "S", range(20)
TIME_LIMIT_S = 60      # cplex maxtime per solve (seconds)


def brute_cluster(d, Q, Kp, c):
    N = len(d)
    return min(clustering_cost(a, c) for a in itertools.product(range(Kp), repeat=N)
               if all(sum(d[i] for i in range(N) if a[i] == k) <= Q for k in range(Kp)))


def brute_cvrp(d, Q, K, P):
    N = len(d)

    def tour(cl):
        if not cl:
            return 0.0
        return min(sum(np.linalg.norm(P[r[t]] - P[r[t + 1]]) for t in range(len(r) - 1))
                   for r in ([0] + [i + 1 for i in perm] + [0] for perm in itertools.permutations(cl)))
    return min(sum(tour([i for i in range(N) if a[i] == k]) for k in range(K))
               for a in itertools.product(range(K), repeat=N)
               if all(sum(d[i] for i in range(N) if a[i] == k) <= Q for k in range(K)))


def cplex_cluster(d, Q, Kp, c):
    N = len(d)
    m = Model(name="clustering"); m.parameters.timelimit = TIME_LIMIT_S
    a = {(i, k): m.binary_var(name=f"a_{i}_{k}") for i in range(N) for k in range(Kp)}
    y = {(i, j, k): m.continuous_var(lb=0, name=f"y_{i}_{j}_{k}") for i in range(N) for j in range(i + 1, N) for k in range(Kp)}
    for i in range(N):
        m.add_constraint(m.sum(a[i, k] for k in range(Kp)) == 1)                       # (C1)
    for k in range(Kp):
        m.add_constraint(m.sum(d[i] * a[i, k] for i in range(N)) <= Q)                 # (C2)
    for (i, j, k), v in y.items():
        m.add_constraint(v >= a[i, k] + a[j, k] - 1)                                    # y = a_ik a_jk at optimum
    m.minimize(m.sum(c[i, j] * v for (i, j, k), v in y.items()))
    s = m.solve()
    return s.objective_value, m.solve_details.status


def cplex_cvrp(d, Q, K, P):
    N = len(d); V = range(N + 1); C = range(1, N + 1)
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    dem = {i + 1: d[i] for i in range(N)}
    m = Model(name="cvrp"); m.parameters.timelimit = TIME_LIMIT_S
    x = {(i, j, k): m.binary_var(name=f"x_{i}_{j}_{k}") for i in V for j in V if i != j for k in range(K)}
    a = {(i, k): m.binary_var(name=f"a_{i}_{k}") for i in C for k in range(K)}
    u = {i: m.continuous_var(lb=dem[i], ub=Q, name=f"u_{i}") for i in C}
    for i in C:
        m.add_constraint(m.sum(a[i, k] for k in range(K)) == 1)                                    # (C1)
    for k in range(K):
        m.add_constraint(m.sum(dem[i] * a[i, k] for i in C) <= Q)                                   # (C2)
        for i in C:
            m.add_constraint(m.sum(x[i, j, k] for j in V if j != i) == a[i, k])                    # (C3)
            m.add_constraint(m.sum(x[j, i, k] for j in V if j != i) == a[i, k])                    # (C4)
        m.add_constraint(m.sum(x[0, j, k] for j in C) <= 1)                                         # (C5)
        m.add_constraint(m.sum(x[0, j, k] for j in C) == m.sum(x[j, 0, k] for j in C))              # depot balance
    for i in C:                                                                                      # MTZ, replaces (C6)
        for j in C:
            if i != j:
                m.add_constraint(u[j] >= u[i] + dem[j] - Q * (1 - m.sum(x[i, j, k] for k in range(K))))
    m.minimize(m.sum(D[i, j] * v for (i, j, k), v in x.items()))
    s = m.solve()
    return s.objective_value, m.solve_details.status


def main():
    rows = []
    for seed in SEEDS:
        inst = generate(TIER, seed); validate(inst)
        d, Q, K, Kp = inst["d"], inst["Q"], inst["K"], inst["K_min"]
        c = distances(inst["xy"]); P = np.vstack([DEPOT, inst["xy"]])
        bf_cl, bf_cv = brute_cluster(d, Q, Kp, c), brute_cvrp(d, Q, K, P)
        cp_cl, st_cl = cplex_cluster(d, Q, Kp, c)
        cp_cv, st_cv = cplex_cvrp(d, Q, K, P)
        rows.append(dict(seed=seed, brute_cluster=round(bf_cl, 6), cplex_cluster=round(cp_cl, 6), cplex_cluster_status=st_cl,
                         agree_cluster=math.isclose(bf_cl, cp_cl, rel_tol=1e-6),
                         brute_cvrp=round(bf_cv, 6), cplex_cvrp=round(cp_cv, 6), cplex_cvrp_status=st_cv,
                         agree_cvrp=math.isclose(bf_cv, cp_cv, rel_tol=1e-6), cplex_maxtime_s=TIME_LIMIT_S))
    with open("cplex_check.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    n = len(rows)
    print(f"Tier {TIER}, {n} instances, CPLEX time limit {TIME_LIMIT_S} s per solve")
    print(f"clustering model: CPLEX agrees with brute force on {sum(r['agree_cluster'] for r in rows)}/{n}")
    print(f"CVRP (full fleet): CPLEX agrees with brute force on {sum(r['agree_cvrp'] for r in rows)}/{n}")
    print("statuses:", sorted({r['cplex_cluster_status'] for r in rows} | {r['cplex_cvrp_status'] for r in rows}))


if __name__ == "__main__":
    main()