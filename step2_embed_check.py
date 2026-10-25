"""Step 2: first atom patterns + realized-QUBO check (tier S, classical, exhaustive).

For each instance:
  1. build the Step-1 QUBO (lambda_2 = 10, A = Gamma_max/2) and scale it to device units,
  2. find atom positions with two embedders (see EMBEDDERS below),
  3. build a Pulser Register and validate it against AnalogDevice (spacing, radius, atom count),
  4. compute the REALIZED couplings C6/r^6 for every pair (including pairs whose target is 0),
  5. brute-force the realized QUBO and classify its ground state against the constrained optimum.
Requires instances.py and step1_qubo_check.py in the same folder; pip install pulser scipy.
"""
import csv, itertools, math, os
import numpy as np
from scipy.optimize import minimize
from pulser import Register
from pulser.devices import AnalogDevice
from instances import generate, validate
from step1_qubo_check import distances, weights, build_qubo, classify, clustering_cost

TIER, SEEDS = "S", range(20)
LAM2, A_MODE = 10.0, "gamma_half"
N_RESTARTS = 20
C6 = AnalogDevice.interaction_coeff                  # rad/us * um^6
R_MIN = AnalogDevice.min_atom_distance               # um
R_MAX = AnalogDevice.max_radial_distance             # um
U_CEIL = C6 / R_MIN ** 6                             # strongest realizable coupling, rad/us
DELTA_MAX = AnalogDevice.channels["rydberg_global"].max_abs_detuning
EMBEDDERS = ["frobenius", "log_hinge"]
ONEHOT_WEIGHT = 10.0  # log_hinge: weight on same-customer (one-hot) couplings, which enforce (C1)
ZERO_FRACTION = 0.5   # log_hinge: unwanted couplings should stay below 50% of the weakest wanted one


def pair_couplings(pos):
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt((diff ** 2).sum(-1))
    iu = np.triu_indices(len(pos), 1)
    return r[iu], C6 / r[iu] ** 6


def loss_and_grad(flat, T, method, w=None):
    """Embedding loss and its exact gradient with respect to the flattened positions."""
    pos = flat.reshape(-1, 2)
    n = len(pos)
    iu = np.triu_indices(n, 1)
    diff = pos[iu[0]] - pos[iu[1]]                      # (pairs, 2)
    r = np.sqrt((diff ** 2).sum(1))
    U = C6 / r ** 6
    wanted = T > 0
    dL_dU = np.zeros_like(U)
    if method == "frobenius":                           # match every coupling, zeros included (naive baseline)
        S = (T ** 2).sum()
        fit = ((U - T) ** 2).sum() / S
        dL_dU = 2 * (U - T) / S
    else:                                               # log error on wanted couplings, hinge on unwanted ones
        w = np.ones(wanted.sum()) if w is None else w[wanted]
        lw = np.log(U[wanted]) - np.log(T[wanted])
        fit = (w * lw ** 2).sum() / w.sum()
        dL_dU[wanted] = 2 * w * lw / (w.sum() * U[wanted])
        ceiling = ZERO_FRACTION * T[wanted].min()
        h = np.maximum(0, np.log(U[~wanted]) - np.log(ceiling))
        fit += (h ** 2).mean()
        dL_dU[~wanted] = 2 * h / ((~wanted).sum() * U[~wanted])
    short = np.maximum(0, R_MIN - r)                    # spacing penalty
    pen = 1e3 * (short ** 2).sum()
    dL_dr = dL_dU * (-6 * U / r) - 2e3 * short
    g_pair = (dL_dr / r)[:, None] * diff
    grad = np.zeros_like(pos)
    np.add.at(grad, iu[0], g_pair)
    np.add.at(grad, iu[1], -g_pair)
    centred = pos - pos.mean(0)                         # radius penalty
    rad = np.sqrt((centred ** 2).sum(1))
    over = np.maximum(0, rad - R_MAX)
    pen += 1e3 * (over ** 2).sum()
    g_c = (2e3 * over / np.maximum(rad, 1e-12))[:, None] * centred
    grad += g_c - g_c.mean(0)
    return fit + pen, grad.ravel()


def embed(T, n_atoms, method, rng, w=None):
    best = None
    for _ in range(N_RESTARTS):
        x0 = rng.uniform(-20, 20, size=(n_atoms, 2)).ravel()
        res = minimize(loss_and_grad, x0, args=(T, method, w), jac=True, method="L-BFGS-B")
        if best is None or res.fun < best.fun:
            best = res
    pos = best.x.reshape(-1, 2)
    return pos - pos.mean(0)


def main():
    rng = np.random.default_rng(12345)
    rows = []
    for seed in SEEDS:
        inst = generate(TIER, seed); validate(inst)
        d, Q, Kp = inst["d"], inst["Q"], inst["K_min"]; N = len(d)
        c = distances(inst["xy"])
        f_opt = min(clustering_cost(a, c) for a in itertools.product(range(Kp), repeat=N)
                    if all(sum(d[i] for i in range(N) if a[i] == k) <= Q for k in range(Kp)))
        A_i, delta, _ = weights(c, d, Q, LAM2, A_MODE)
        M, const = build_qubo(c, d, Q, Kp, LAM2, A_i)
        n_atoms = N * Kp
        iu = np.triu_indices(n_atoms, 1)
        T_qubo = M[iu]                                   # target couplings, QUBO units
        sigma = U_CEIL / T_qubo.max()                    # QUBO units -> rad/us
        T = sigma * T_qubo
        assert sigma * delta <= DELTA_MAX, "final detuning out of range"
        bits = np.array(list(itertools.product([0, 1], repeat=n_atoms)), float)

        for method in EMBEDDERS:
            # pair weights: one-hot pairs (same customer, different vehicle) get ONEHOT_WEIGHT
            same_cust = (iu[0] // Kp) == (iu[1] // Kp)
            w = np.where(same_cust, ONEHOT_WEIGHT, 1.0)
            pos = embed(T, n_atoms, method, rng, w)
            # The optimizer enforces spacing by a penalty, so it can end marginally below 5 um.
            # Fix exactly by a uniform dilation s >= 1: all couplings scale by s^-6, which is the same
            # as rescaling the whole QUBO, so we rescale sigma (and hence the laser detuning) to match.
            s_dil = max(1.0, R_MIN / pair_couplings(pos)[0].min() * (1 + 1e-6))
            pos = pos * s_dil
            sig = sigma * s_dil ** -6
            reg = Register.from_coordinates(pos, prefix="q")
            AnalogDevice.validate_register(reg)          # raises if spacing/radius/count violated
            r, U = pair_couplings(pos)
            # realized QUBO in QUBO units: same uniform linear term, realized couplings
            M_real = np.zeros_like(M)
            M_real[iu] = U / sig
            M_real[np.diag_indices(n_atoms)] = -delta
            E = np.einsum("bi,ij,bj->b", bits, M_real, bits)
            gs = bits[np.isclose(E, E.min(), rtol=0, atol=1e-9 * max(1, abs(E.min())))]
            cats = {classify(b, d, Q, Kp, c, f_opt) for b in gs}
            outcome = cats.pop() if len(cats) == 1 else "tie_mixed"
            # energy level (0 = ground) at which the constrained optimum first appears
            levels = np.unique(np.round(E, 9))
            opt_E = min(e for b, e in zip(bits, E) if classify(b, d, Q, Kp, c, f_opt) == "optimal")
            opt_level = int(np.searchsorted(levels, round(opt_E, 9)))
            wanted = T > 0
            rows.append(dict(
                seed=seed, method=method, rho=round(inst["rho"], 3), atoms=n_atoms,
                outcome=outcome, opt_level=opt_level,
                rel_frob_err=round(np.linalg.norm(U / sig - T_qubo) / np.linalg.norm(T_qubo), 3),
                med_rel_err_wanted=round(float(np.median(np.abs((U / sig)[wanted] / T_qubo[wanted] - 1))), 3),
                max_unwanted_over_min_wanted=round(float((U / sig)[~wanted].max() / T_qubo[wanted].min()), 3),
                min_dist_um=round(float(r.min()), 2), max_radius_um=round(float(np.sqrt((pos ** 2).sum(1)).max()), 2),
                delta_dev=round(sig * delta, 2), dilation=round(s_dil, 4)))
            os.makedirs("step2_positions", exist_ok=True)
            np.save(os.path.join("step2_positions", f"seed{seed}_{method}.npy"), pos)

    with open("step2_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    print(f"{len(rows)} embeddings on {len(SEEDS)} tier-{TIER} instances (lam2={LAM2}, A={A_MODE}); "
          f"all registers valid on AnalogDevice.\n")
    for method in EMBEDDERS:
        sub = [r for r in rows if r["method"] == method]
        n = len(sub)
        cnt = lambda k: sum(r["outcome"] == k for r in sub)
        print(f"{method:>10}: realized GS optimal {cnt('optimal')}/{n}, feasible-suboptimal {cnt('feasible_suboptimal')}, "
              f"over-capacity {cnt('over_capacity')}, C1-violated {cnt('C1_violated')}, tie {cnt('tie_mixed')}")
        print(f"{'':>12}optimum within lowest 3 levels: {sum(r['opt_level'] <= 2 for r in sub)}/{n};  "
              f"median rel. Frobenius error {np.median([r['rel_frob_err'] for r in sub]):.3f};  "
              f"median wanted-coupling error {np.median([r['med_rel_err_wanted'] for r in sub]):.3f};  "
              f"median (max unwanted / min wanted) {np.median([r['max_unwanted_over_min_wanted'] for r in sub]):.2f}")


if __name__ == "__main__":
    main()