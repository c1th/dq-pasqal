"""Write the tier-S instances (deliverable 3) to instances_tierS.json.

For every seed: locations, demands, depot, N, K, Q, load ratio, d_max/Q, K_min, the feasibility certificate
(an assignment of customers to K_min vehicles, checked by instances.validate), and the reference values:
the optimum of the clustering model and the CVRP optimum with the full fleet K, both by brute force (proven optima).
"""
import itertools, json
import numpy as np
from instances import generate, validate, DEPOT
from step1_qubo_check import distances, clustering_cost
from step3_emulate import routed_cost

TIER, SEEDS = "S", range(100)      # seeds 0-99 (Step 1); seeds 0-19 are the ones emulated in Step 3


def main():
    out = []
    for seed in SEEDS:
        inst = generate(TIER, seed)
        validate(inst)                                   # raises if the certificate or any assumption fails
        d, Q, K, Kp = inst["d"], inst["Q"], inst["K"], inst["K_min"]; N = len(d)
        c = distances(inst["xy"]); P = np.vstack([DEPOT, inst["xy"]])
        cap_ok = lambda a, k_: all(sum(d[i] for i in range(N) if a[i] == k) <= Q for k in range(k_))
        f_opt = min(clustering_cost(a, c) for a in itertools.product(range(Kp), repeat=N) if cap_ok(a, Kp))
        cvrp_opt = min(routed_cost(a, K, P) for a in itertools.product(range(K), repeat=N) if cap_ok(a, K))
        out.append(dict(seed=seed, tier=TIER, N=N, K=K, Q=Q, depot=list(DEPOT), xy=inst["xy"], d=d,
                        rho=inst["rho"], dmax_over_Q=inst["dmax_over_Q"], K_min=Kp,
                        certificate=inst["certificate"], validated=True,
                        clustering_optimum=f_opt, cvrp_optimum=cvrp_opt,
                        reference_status="proven optimum (brute force)"))
    with open("instances_tierS.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {len(out)} validated tier-{TIER} instances with certificates and reference values to instances_tierS.json")


if __name__ == "__main__":
    main()