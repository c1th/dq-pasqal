"""Step 2b: ladder embedding for K' = 2 (tier S), evaluated exhaustively.

Layout. Vehicle k's atoms sit on rail y = k*b (b = 5 um); customer i's two atoms share an x-coordinate x_i.
Then, exactly and for every customer pair:
    same-vehicle coupling   s_ij = C6/|x_i-x_j|^6                       (equal on both rails)
    cross-vehicle coupling  x_ij = C6/(|x_i-x_j|^2 + b^2)^3            (equal for both cross pairs)
    one-hot coupling        U_oh = C6/b^6                               (equal for every customer)
On any assignment satisfying (C1) the register energy is
    E = -delta_r N + sum_{i<j} x_ij + sum_{i<j, same vehicle} (s_ij - x_ij),
so the valid-state ordering of the target is reproduced when g(|x_i-x_j|) := s_ij - x_ij = t w_ij,
w_ij = c_ij + 2 lambda_2 d_i d_j. We fit the 1-D coordinates x_i (and the scale t) to this, using couplings only.

Validity (C1) is guaranteed without enumeration: with B_i = sum_j s_ij, any detuning B_max < delta_r < U_oh makes
dropping a customer (cost delta_r - sum >= delta_r - B_i > 0) and doubling a customer (cost >= U_oh - delta_r > 0)
unfavourable. We fit with B_max <= 0.8 U_oh and set delta_r to the midpoint of the window.
Enumeration below is used ONLY to evaluate the result.
"""
import csv, itertools, os
import numpy as np
from scipy.optimize import minimize
from pulser import Register
from pulser.devices import AnalogDevice
from instances import generate, validate
from step1_qubo_check import distances, classify, clustering_cost

TIER, SEEDS = "S", range(20)
LAM2 = 10.0
N_RESTARTS = 40
C6 = AnalogDevice.interaction_coeff
R_MIN, R_MAX = AnalogDevice.min_atom_distance, AnalogDevice.max_radial_distance
RAIL = R_MIN                                   # rail spacing b = 5 um -> strongest possible one-hot coupling
U_OH = C6 / RAIL ** 6
WINDOW = 0.8                                   # require B_max <= 0.8 U_oh
X_MAX = np.sqrt(R_MAX ** 2 - (RAIL / 2) ** 2)  # |x| bound so every atom is within the register radius


def g(dx):
    return C6 / dx ** 6 - C6 / (dx ** 2 + RAIL ** 2) ** 3


def fit_coords(w_mat, N, rng):
    iu = np.triu_indices(N, 1); w = w_mat[iu]

    def loss(z):
        x = np.concatenate([[0.0], z[:N - 1]]); t = np.exp(z[-1])
        dx = np.maximum(np.abs(x[iu[0]] - x[iu[1]]), 1e-3)
        fit = ((g(dx) - t * w) ** 2).sum() / (t * w.mean()) ** 2
        S = np.zeros((N, N)); S[iu] = C6 / dx ** 6; S = S + S.T
        pen = 1e3 * (np.maximum(0, R_MIN - dx) ** 2).sum()                          # spacing on a rail
        pen += 1e4 * (np.maximum(0, S.sum(1) / U_OH - WINDOW) ** 2).sum()            # validity window
        xc = x - (x.max() + x.min()) / 2
        pen += 1e3 * (np.maximum(0, np.abs(xc) - X_MAX) ** 2).sum()                   # register radius
        return fit + pen

    best = None
    for _ in range(N_RESTARTS):
        z0 = np.concatenate([rng.uniform(-30, 30, N - 1), [np.log(0.02)]])
        with np.errstate(all="ignore"):
            r = minimize(loss, z0, method="Nelder-Mead", options=dict(maxiter=6000, xatol=1e-7, fatol=1e-12))
        if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
            best = r
    x = np.concatenate([[0.0], best.x[:N - 1]])
    return x - (x.max() + x.min()) / 2, np.exp(best.x[-1])


def main():
    rng = np.random.default_rng(12345)
    rows = []
    for seed in SEEDS:
        inst = generate(TIER, seed); validate(inst)
        d, Q, Kp = inst["d"], inst["Q"], inst["K_min"]; N = len(d)
        assert Kp == 2, "ladder embedding is defined for K' = 2"
        c = distances(inst["xy"])
        f_opt = min(clustering_cost(a, c) for a in itertools.product(range(Kp), repeat=N)
                    if all(sum(d[i] for i in range(N) if a[i] == k) <= Q for k in range(Kp)))
        w_mat = np.array([[c[i, j] + 2 * LAM2 * d[i] * d[j] if i != j else 0.0 for j in range(N)] for i in range(N)])
        x, t = fit_coords(w_mat, N, rng)
        pos = np.array([[x[i], (k - 0.5) * RAIL] for i in range(N) for k in range(Kp)])   # atom index i*Kp + k
        AnalogDevice.validate_register(Register.from_coordinates(pos, prefix="q"))
        n = N * Kp
        iu = np.triu_indices(n, 1)
        r = np.sqrt(((pos[iu[0]] - pos[iu[1]]) ** 2).sum(1)); U = C6 / r ** 6
        dx = np.abs(x[:, None] - x[None, :]); np.fill_diagonal(dx, np.inf)
        B = (C6 / dx ** 6).sum(1)
        assert B.max() < U_OH, "validity window is empty"
        delta_r = (B.max() + U_OH) / 2
        M_real = np.zeros((n, n)); M_real[iu] = U; M_real[np.diag_indices(n)] = -delta_r
        bits = np.array(list(itertools.product([0, 1], repeat=n)), float)
        E = np.einsum("bi,ij,bj->b", bits, M_real, bits)
        gs = bits[np.isclose(E, E.min(), rtol=0, atol=1e-9 * max(1, abs(E.min())))]
        cats = {classify(b, d, Q, Kp, c, f_opt) for b in gs}
        outcome = cats.pop() if len(cats) == 1 else "tie_mixed"
        levels = np.unique(np.round(E, 9))
        opt_E = min(e for b, e in zip(bits, E) if classify(b, d, Q, Kp, c, f_opt) == "optimal")
        jj = np.triu_indices(N, 1)
        rel_err = np.linalg.norm(g(dx[jj]) / t - w_mat[jj]) / np.linalg.norm(w_mat[jj])
        rows.append(dict(seed=seed, rho=round(inst["rho"], 3), atoms=n, outcome=outcome,
                         opt_level=int(np.searchsorted(levels, round(opt_E, 9))),
                         rel_valid_energy_err=round(float(rel_err), 3), delta_dev=round(float(delta_r), 2),
                         B_max_over_Uoh=round(float(B.max() / U_OH), 3), min_dist_um=round(float(r.min()), 2),
                         max_radius_um=round(float(np.sqrt((pos ** 2).sum(1)).max()), 2)))
        os.makedirs("step2b_positions", exist_ok=True)
        np.save(os.path.join("step2b_positions", f"seed{seed}_ladder.npy"), pos)

    with open("step2b_results.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    n = len(rows); cnt = lambda k: sum(r["outcome"] == k for r in rows)
    print(f"{n} ladder embeddings on tier {TIER} (lam2={LAM2}); all registers valid on AnalogDevice.")
    print(f"realized GS optimal {cnt('optimal')}/{n}, feasible-suboptimal {cnt('feasible_suboptimal')}, "
          f"over-capacity {cnt('over_capacity')}, C1-violated {cnt('C1_violated')}, tie {cnt('tie_mixed')}")
    print(f"optimum within lowest 3 levels: {sum(r['opt_level'] <= 2 for r in rows)}/{n}; "
          f"median rel. valid-energy error {np.median([r['rel_valid_energy_err'] for r in rows]):.3f}; "
          f"max radius {max(r['max_radius_um'] for r in rows):.1f} um; detuning {rows[0]['delta_dev']:.1f} rad/us")


if __name__ == "__main__":
    main()