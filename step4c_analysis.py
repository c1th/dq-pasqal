"""Step 4c analysis: how much objective information do the tier-M registers retain?

For every register from step4c_tierM_embed.py, rank all K'^N one-hot assignments by the REALIZED register energy and
compare the best feasible one with the model optimum (model gap), against the mean model gap of a uniformly random
feasible clustering. Also reports where the optimum ranks among one-hot states.
Requires step4c_results.csv, step4c_positions/, step4b_references.csv.
"""
import csv, itertools
import numpy as np
from instances import generate
from step1_qubo_check import distances
from step4c_tierM_embed import onehot_realized, C6


def main():
    rows = list(csv.DictReader(open("step4c_results.csv")))
    refs = {int(r["seed"]): r for r in csv.DictReader(open("step4b_references.csv"))}
    out = []
    for r in rows:
        seed, m = int(r["seed"]), r["method"]
        inst = generate("M", seed); d, Q, Kp = inst["d"], inst["Q"], inst["K_min"]; N = len(d)
        c = distances(inst["xy"])
        A = np.array(list(itertools.product(range(Kp), repeat=N)))
        L = np.stack([(A == k) @ np.asarray(d) for k in range(Kp)], 1); feas = (L <= Q).all(1)
        iu = np.triu_indices(N, 1); same = A[:, :, None] == A[:, None, :]
        f = (same[:, iu[0], iu[1]] * c[iu]).sum(1); fo = float(refs[seed]["f_opt"])
        pos = np.load(f"step4c_positions/seed{seed}_{m}.npy"); n = N * Kp; P = np.triu_indices(n, 1)
        U = C6 / np.sqrt(((pos[P[0]] - pos[P[1]]) ** 2).sum(1)) ** 6
        Uf = np.zeros((n, n)); Uf[P] = U; Uf = Uf + Uf.T
        order = np.argsort(onehot_realized(Uf, A, Kp))
        best_feas = next(i for i in order if feas[i])
        opt = np.flatnonzero(feas & np.isclose(f, fo, rtol=0, atol=1e-6))
        rank_pct = 100.0 * min(np.flatnonzero(order == o)[0] for o in opt) / len(A)
        out.append(dict(seed=seed, method=m, K_min=Kp,
                        gap_best_feasible_onehot=round(100 * (f[best_feas] - fo) / fo, 2),
                        gap_random_feasible_mean=round(100 * (f[feas].mean() - fo) / fo, 2),
                        optimum_rank_percentile=round(rank_pct, 2)))
    with open("step4c_analysis.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    for Kp, m in [(2, "ladder"), (2, "frobenius"), (3, "frobenius")]:
        sub = [o for o in out if o["K_min"] == Kp and o["method"] == m]
        if sub:
            med = lambda k: np.median([o[k] for o in sub])
            print(f"K'={Kp} {m:>9} ({len(sub)}): median model gap of best feasible one-hot state "
                  f"{med('gap_best_feasible_onehot'):.1f}% vs random feasible clustering {med('gap_random_feasible_mean'):.1f}%; "
                  f"optimum at median percentile {med('optimum_rank_percentile'):.1f}% of one-hot states")


if __name__ == "__main__":
    main()