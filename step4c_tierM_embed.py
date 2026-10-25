"""Step 4c: atom registers for tier M, with a static check on the one-hot subspace.

Embedders (Sec. 4 coverage plan): ladder on K' = 2 instances; coefficient matching ("frobenius") on all instances.
Weights from Step 4a: lambda_2 = 35 for K' = 2, lambda_2 = 1 for K' = 3, A = Gamma_max/2.
Full enumeration is impossible at 20-30 atoms, so the static check uses the K'^N one-hot assignments:
  * the lowest one-hot state of the REALIZED register, classified against the model optimum (from Step 4b),
  * the level at which the optimum appears among one-hot states,
  * whether that lowest one-hot state is a strict local minimum of the full realized register energy
    (a necessary condition for it to be the true ground state; for the ladder, (C1) is guaranteed by the claim in Sec. 4).
Positions and final detunings are saved for the Step 4d emulation.
Requires instances.py, step1_qubo_check.py, step2_embed_check.py, step2b_ladder_embed.py, step4b_references.csv.
"""
import csv, itertools, os
import numpy as np
from pulser import Register
from pulser.devices import AnalogDevice
from instances import generate, validate
from step1_qubo_check import distances, weights, build_qubo
import step2_embed_check as coef
import step2b_ladder_embed as lad

TIER, SEEDS = "M", range(60)
LAM2 = {2: 35.0, 3: 1.0}
A_MODE = "gamma_half"
C6, R_MIN, U_CEIL = coef.C6, coef.R_MIN, coef.U_CEIL
DELTA_MAX = AnalogDevice.channels["rydberg_global"].max_abs_detuning


def fit_ladder(w_mat, N, rng, restarts=40):
    """Ladder fit for tier M. Same objective as step2b_ladder_embed.fit_coords, but the register radius is measured
    from the mean position (as Pulser does) and penalized more strongly, because 10 customers on a line approach
    the 38 um radius. Returns centred rail coordinates and the scale t."""
    from scipy.optimize import minimize
    iu = np.triu_indices(N, 1); w = w_mat[iu]

    def loss(z):
        x = np.concatenate([[0.0], z[:N - 1]]); t = np.exp(z[-1])
        dx = np.maximum(np.abs(x[iu[0]] - x[iu[1]]), 1e-3)
        fit = ((lad.g(dx) - t * w) ** 2).sum() / (t * w.mean()) ** 2
        S = np.zeros((N, N)); S[iu] = C6 / dx ** 6; S = S + S.T
        pen = 1e3 * (np.maximum(0, R_MIN - dx) ** 2).sum()
        pen += 1e4 * (np.maximum(0, S.sum(1) / lad.U_OH - lad.WINDOW) ** 2).sum()
        xc = x - x.mean()
        pen += 1e5 * (np.maximum(0, np.abs(xc) - (lad.X_MAX - 0.5)) ** 2).sum()
        return fit + pen

    best = None
    for _ in range(restarts):
        z0 = np.concatenate([np.sort(rng.uniform(-30, 30, N - 1)), [np.log(0.01)]])
        with np.errstate(all="ignore"):
            r = minimize(loss, z0, method="Powell", options=dict(maxiter=20000, xtol=1e-6, ftol=1e-12))
        if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
            best = r
    x = np.concatenate([[0.0], best.x[:N - 1]])
    return x - x.mean(), np.exp(best.x[-1])


def onehot_realized(U_full, assign, Kp):
    """Realized interaction energy of each one-hot assignment (the -delta*N term is the same for all)."""
    N = assign.shape[1]
    idx = np.arange(N) * Kp + assign
    E = np.zeros(len(assign))
    for i in range(N):
        for j in range(i + 1, N):
            E += U_full[idx[:, i], idx[:, j]]
    return E


def strict_local_min(bits, U_full, delta):
    field = U_full @ bits                                  # U_full symmetric, zero diagonal
    dE = (1 - 2 * bits) * (-delta + field)
    return bool((dE > 1e-9).all())


def main():
    refs = {int(r["seed"]): r for r in csv.DictReader(open("step4b_references.csv"))}
    rng = np.random.default_rng(2024)
    rows = []
    os.makedirs("step4c_positions", exist_ok=True)
    for seed in SEEDS:
        inst = generate(TIER, seed); validate(inst)
        d, Q, Kp = inst["d"], inst["Q"], inst["K_min"]; N = len(d); n = N * Kp
        c = distances(inst["xy"]); lam2 = LAM2[Kp]
        f_opt = float(refs[seed]["f_opt"])
        assign = np.array(list(itertools.product(range(Kp), repeat=N)))
        L = np.stack([(assign == k) @ np.asarray(d) for k in range(Kp)], axis=1)
        feasible = (L <= Q).all(1)
        iuN = np.triu_indices(N, 1)
        same = assign[:, :, None] == assign[:, None, :]
        f = (same[:, iuN[0], iuN[1]] * c[iuN]).sum(1)
        is_opt = feasible & np.isclose(f, f_opt, rtol=0, atol=1e-6)
        methods = ["ladder", "frobenius"] if Kp == 2 else ["frobenius"]
        for method in methods:
            if method == "frobenius":
                A_i, delta, _ = weights(c, d, Q, lam2, A_MODE)
                M, _ = build_qubo(c, d, Q, Kp, lam2, A_i)
                iu = np.triu_indices(n, 1)
                T_q = M[iu]; sigma = U_CEIL / T_q.max(); T = sigma * T_q
                same_cust = (iu[0] // Kp) == (iu[1] // Kp)
                pos = coef.embed(T, n, "frobenius", rng, np.where(same_cust, 1.0, 1.0))
                s_dil = max(1.0, R_MIN / coef.pair_couplings(pos)[0].min() * (1 + 1e-6))
                pos = pos * s_dil; sig = sigma * s_dil ** -6
                r, U = coef.pair_couplings(pos)
                delta_dev = sig * delta
                fid_metric = "rel_frob_err"
                fid = float(np.linalg.norm(U / sig - T_q) / np.linalg.norm(T_q))
            else:
                w_mat = np.array([[c[i, j] + 2 * lam2 * d[i] * d[j] if i != j else 0.0 for j in range(N)] for i in range(N)])
                x, t = fit_ladder(w_mat, N, rng)
                pos = np.array([[x[i], (k - 0.5) * lad.RAIL] for i in range(N) for k in range(Kp)])
                iu = np.triu_indices(n, 1)
                r = np.sqrt(((pos[iu[0]] - pos[iu[1]]) ** 2).sum(1)); U = C6 / r ** 6
                dx = np.abs(x[:, None] - x[None, :]); np.fill_diagonal(dx, np.inf)
                B = (C6 / dx ** 6).sum(1)
                window_ok = B.max() < lad.U_OH
                delta_dev = (B.max() + lad.U_OH) / 2
                fid_metric = "rel_valid_energy_err"
                fid = float(np.linalg.norm(lad.g(dx[iuN]) / t - w_mat[iuN]) / np.linalg.norm(w_mat[iuN]))
            AnalogDevice.validate_register(Register.from_coordinates(pos, prefix="q"))
            assert delta_dev <= DELTA_MAX, "final detuning out of range"
            U_full = np.zeros((n, n)); U_full[iu] = U; U_full = U_full + U_full.T
            E = onehot_realized(U_full, assign, Kp)
            g = int(E.argmin())
            outcome = ("over_capacity" if not feasible[g] else "optimal" if is_opt[g] else "feasible_suboptimal")
            levels = np.unique(np.round(E, 9))
            opt_level = int(np.searchsorted(levels, round(E[is_opt].min(), 9)))
            bits = np.zeros(n); bits[np.arange(N) * Kp + assign[g]] = 1
            np.save(os.path.join("step4c_positions", f"seed{seed}_{method}.npy"), pos)
            rows.append(dict(seed=seed, method=method, K_min=Kp, atoms=n, lam2=lam2, rho=inst["rho"],
                             eff_load=round(sum(d) / (Kp * Q), 3), outcome=outcome, opt_level=opt_level,
                             local_min=strict_local_min(bits, U_full, delta_dev),
                             window_ok=(window_ok if method == "ladder" else ""),
                             emb_fidelity_metric=fid_metric, emb_fidelity=round(fid, 3),
                             delta_dev=round(float(delta_dev), 3), min_dist_um=round(float(r.min()), 2),
                             max_radius_um=round(float(np.sqrt(((pos - pos.mean(0)) ** 2).sum(1)).max()), 2)))
        print(f"seed {seed} (K'={Kp}) done", flush=True)

    with open("step4c_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nTier {TIER}: static check of realized registers on the one-hot subspace (all registers valid on AnalogDevice)")
    for Kp, method in [(2, "ladder"), (2, "frobenius"), (3, "frobenius")]:
        sub = [r for r in rows if r["K_min"] == Kp and r["method"] == method]
        if not sub:
            continue
        cnt = lambda k: sum(r["outcome"] == k for r in sub)
        print(f"K'={Kp} {method:>9} ({len(sub)}): lowest one-hot state optimal {cnt('optimal')}, feasible-suboptimal "
              f"{cnt('feasible_suboptimal')}, over-capacity {cnt('over_capacity')}; optimum in lowest 3 one-hot levels "
              f"{sum(r['opt_level'] <= 2 for r in sub)}; strict local min {sum(r['local_min'] for r in sub)}; median "
              f"{sub[0]['emb_fidelity_metric']} {np.median([r['emb_fidelity'] for r in sub]):.3f}; "
              f"detuning {min(r['delta_dev'] for r in sub):.1f}-{max(r['delta_dev'] for r in sub):.1f} rad/us")


if __name__ == "__main__":
    main()