"""Step 4a: retune the penalty weights on tier M (classical).

Tier M has N = 10 customers and K' = K_min in {2, 3}, so the full QUBO has 2^20 or 2^30 states. We therefore
  1. enumerate all K'^N one-hot assignments (1,024 or 59,049), on which the assignment penalty is zero, and find the
     QUBO's lowest one-hot state for each lambda_2; classify it (optimal / feasible-suboptimal / over-capacity),
  2. check that this state is a strict local minimum of the FULL QUBO (every single-bit flip raises the energy),
     a necessary condition for it to be the true ground state,
  3. for the first FULL_CHECK_K2 instances with K' = 2, enumerate all 2^20 bitstrings and confirm the true ground
     state is one-hot, i.e. that A = Gamma_max/2 keeps (C1) there too.
Requires instances.py and step1_qubo_check.py in the same folder.
"""
import csv, itertools
import numpy as np
from instances import generate, validate
from step1_qubo_check import distances, weights, build_qubo

TIER, SEEDS = "M", range(60)
# K' = K_min instances are packed much tighter (effective load D/(K_min Q) = 0.75-1.0 for K' = 2), so each
# K' gets its own grid.
LAMBDA2S = {2: [5.0, 12.0, 20.0, 35.0, 50.0], 3: [0.5, 1.0, 2.0, 3.0, 5.0]}
A_MODES = ["gamma_half", "rigorous"]
FULL_CHECK_K2 = 6           # instances with K' = 2 that get the full 2^20 enumeration (at the chosen lambda_2)
FULL_CHECK_LAMBDA2 = 35.0   # lambda_2 used for the full check (chosen K' = 2 value)


def onehot_energies(assign, c, d, Q, Kp, lam2):
    """H on one-hot states, up to a constant: f(a) + lam2 * sum_k L_k^2 (the -2 lam2 Q sum_k L_k term is constant)."""
    same = assign[:, :, None] == assign[:, None, :]
    iu = np.triu_indices(assign.shape[1], 1)
    f = (same[:, iu[0], iu[1]] * c[iu]).sum(1)
    L = np.stack([(assign == k) @ np.asarray(d) for k in range(Kp)], axis=1)
    return f, L, f + lam2 * (L ** 2).sum(1)


def strict_local_min(bits, M):
    """True if every single-bit flip of `bits` strictly raises x^T M x (M upper triangular, diagonal = linear)."""
    S = M + M.T - np.diag(np.diag(M))                 # symmetric couplings, diagonal = linear terms
    field = S @ bits - np.diag(S) * bits              # sum over other excited atoms
    dE = (1 - 2 * bits) * (np.diag(S) + field)
    return bool((dE > 1e-9).all())


def full_ground_state(M, n, chunk=1 << 16):
    """Exact minimum of x^T M x over all 2^n bitstrings (n <= ~22)."""
    best_e, best_b = np.inf, None
    idx = np.arange(n)
    for start in range(0, 1 << n, chunk):
        ints = np.arange(start, min(start + chunk, 1 << n))
        B = ((ints[:, None] >> (n - 1 - idx)) & 1).astype(float)
        E = np.einsum("bi,ij,bj->b", B, M, B)
        k = int(E.argmin())
        if E[k] < best_e:
            best_e, best_b = E[k], B[k]
    return best_e, best_b


def main():
    rows, full_checks = [], []
    n_k2_checked = 0
    for seed in SEEDS:
        inst = generate(TIER, seed); validate(inst)
        d, Q, Kp = inst["d"], inst["Q"], inst["K_min"]; N = len(d)
        c = distances(inst["xy"])
        assign = np.array(list(itertools.product(range(Kp), repeat=N)))
        f, L, _ = onehot_energies(assign, c, d, Q, Kp, 0.0)
        feasible = (L <= Q).all(1)
        f_opt = f[feasible].min()
        for lam2 in LAMBDA2S[Kp]:
            _, _, E = onehot_energies(assign, c, d, Q, Kp, lam2)
            gs = np.flatnonzero(np.isclose(E, E.min(), rtol=0, atol=1e-9 * max(1, abs(E.min()))))
            cats = set()
            for g in gs:
                if not feasible[g]:
                    cats.add("over_capacity")
                elif np.isclose(f[g], f_opt, atol=1e-9):
                    cats.add("optimal")
                else:
                    cats.add("feasible_suboptimal")
            outcome = cats.pop() if len(cats) == 1 else "tie_mixed"
            order = np.argsort(E)
            levels = np.unique(np.round(E, 9))
            opt_E = E[feasible & np.isclose(f, f_opt, atol=1e-9)].min()
            first_feas_E = E[feasible].min()
            for mode in A_MODES:
                A_i, delta, bound = weights(c, d, Q, lam2, mode)
                M, _ = build_qubo(c, d, Q, Kp, lam2, A_i)
                g0 = gs[0]
                bits = np.zeros(N * Kp); bits[np.arange(N) * Kp + assign[g0]] = 1
                couplings = np.triu(M, 1)[np.triu(M, 1) > 1e-12]
                rows.append(dict(seed=seed, K_min=Kp, atoms=N * Kp, rho=round(inst["rho"], 3),
                                 eff_load=round(sum(d) / (Kp * Q), 3), lam2=lam2, mode=mode,
                                 outcome=outcome, opt_level=int(np.searchsorted(levels, round(opt_E, 9))),
                                 first_feasible_level=int(np.searchsorted(levels, round(first_feas_E, 9))),
                                 local_min=strict_local_min(bits, M), above_bound=bool((A_i > bound).all()),
                                 dyn_range=round(couplings.max() / couplings.min(), 1),
                                 delta_dev=round(55.4 * delta / couplings.max(), 2)))
        if Kp == 2 and n_k2_checked < FULL_CHECK_K2:
            A_i, delta, bound = weights(c, d, Q, FULL_CHECK_LAMBDA2, "gamma_half")
            M, _ = build_qubo(c, d, Q, Kp, FULL_CHECK_LAMBDA2, A_i)
            _, b = full_ground_state(M, N * Kp)
            onehot = bool((b.reshape(N, Kp).sum(1) == 1).all())
            full_checks.append((seed, onehot)); n_k2_checked += 1
        print(f"seed {seed} (K'={Kp}) done", flush=True)

    with open("step4a_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    cats = ["optimal", "feasible_suboptimal", "over_capacity", "tie_mixed"]
    for Kp in (2, 3):
        sub_all = [r for r in rows if r["K_min"] == Kp and r["mode"] == "gamma_half"]
        n_inst = len({r["seed"] for r in sub_all})
        if not n_inst:
            continue
        effs = sorted({r["seed"]: r["eff_load"] for r in sub_all}.values())
        print(f"\nTier {TIER}, K' = {Kp} ({n_inst} instances, {N_atoms(Kp)} atoms), lowest one-hot state, A = Gamma_max/2;"
              f" effective load D/(K' Q) {effs[0]:.2f}-{effs[-1]:.2f}")
        print(f"{'lam2':>5} | " + " ".join(f"{x[:12]:>12}" for x in cats) +
              " | opt in lowest 3 | local min | med dyn range | max delta_dev")
        for lam2 in LAMBDA2S[Kp]:
            sub = [r for r in sub_all if r["lam2"] == lam2]
            print(f"{lam2:>5} | " + " ".join(f"{sum(r['outcome'] == x for r in sub):>12}" for x in cats) +
                  f" | {sum(r['opt_level'] <= 2 for r in sub):>15} | {sum(r['local_min'] for r in sub):>9}"
                  f" | {np.median([r['dyn_range'] for r in sub]):>13.1f} | {max(r['delta_dev'] for r in sub):>13.1f}")
    print(f"\nFull 2^20 check at lambda_2 = {FULL_CHECK_LAMBDA2}, A = Gamma_max/2 (K' = 2 instances): "
          + ", ".join(f"seed {s}: {'one-hot' if ok else 'NOT one-hot'}" for s, ok in full_checks))


def N_atoms(Kp):
    return 10 * Kp


if __name__ == "__main__":
    main()