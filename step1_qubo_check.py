"""Step 1: tier-S QUBO validation (classical, exhaustive).

For each instance and each (A-mode, lambda_2) setting:
  1. build H(a) from its three terms (model write-up, eq. qubo),
  2. check that every linear coefficient equals -delta (uniform detuning),
  3. check the QUBO energy against a direct evaluation of H(a) on all 2^(N K') bitstrings,
  4. brute-force the QUBO ground state(s) and classify them against the constrained optimum.
Requires instances.py in the same folder.
"""
import csv, itertools, math, sys
import numpy as np
from instances import generate, validate

TIER, N_SEEDS = "S", 100
LAMBDA2S = [6.0, 8.0, 10.0, 12.0, 15.0]
A_MODES = ["rigorous", "tight", "gamma_half", "gamma_quarter"]
EPS = 1e-6


def distances(xy):
    xy = np.asarray(xy)
    return np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)


def weights(c, d, Q, lam2, mode):
    """Return (A_i array, delta). g_i = d_i(d_i - 2Q) is the capacity linear term / lam2."""
    N = len(d)
    d = np.asarray(d, float)
    g = d * (d - 2 * Q)
    Gamma = np.array([sum(c[i, j] + 2 * lam2 * d[i] * d[j] for j in range(N) if j != i) for i in range(N)])
    bound = np.maximum(Gamma, lam2 * d * (2 * Q - d))           # A_i must exceed this (rigorous)
    if mode == "rigorous":       # one baseline A above every bound; A_i = A + lam2 (g_i - min g)
        A = bound.max() * (1 + EPS)
        delta = A - lam2 * g.min()
    elif mode == "tight":        # smallest delta with every A_i above its own bound
        delta = (bound - lam2 * g).max() * (1 + EPS)
    elif mode == "gamma_half":
        A = Gamma.max() / 2
        delta = A - lam2 * g.min()
    elif mode == "gamma_quarter":
        A = Gamma.max() / 4
        delta = A - lam2 * g.min()
    A_i = delta + lam2 * g                                       # makes every linear coeff = -delta
    return A_i, delta, bound


def build_qubo(c, d, Q, Kp, lam2, A_i):
    """Upper-triangular matrix M (diagonal = linear terms) and constant, built term by term."""
    N = len(d)
    v = lambda i, k: i * Kp + k
    M = np.zeros((N * Kp, N * Kp))
    const = 0.0
    for k in range(Kp):                                          # f(a)
        for i in range(N):
            for j in range(i + 1, N):
                M[v(i, k), v(j, k)] += c[i, j]
    for i in range(N):                                           # A_i (1 - sum_k a_ik)^2
        const += A_i[i]
        for k in range(Kp):
            M[v(i, k), v(i, k)] -= A_i[i]
            for l in range(k + 1, Kp):
                M[v(i, k), v(i, l)] += 2 * A_i[i]
    for k in range(Kp):                                          # lam2 (L_k^2 - 2 Q L_k)
        for i in range(N):
            M[v(i, k), v(i, k)] += lam2 * (d[i] ** 2 - 2 * Q * d[i])
            for j in range(i + 1, N):
                M[v(i, k), v(j, k)] += 2 * lam2 * d[i] * d[j]
    return M, const


def H_direct(bits, c, d, Q, Kp, lam2, A_i):
    """Independent evaluation of H(a) straight from its definition."""
    N = len(d)
    a = np.asarray(bits).reshape(N, Kp)
    f = sum(c[i, j] * a[i, k] * a[j, k] for k in range(Kp) for i in range(N) for j in range(i + 1, N))
    pen = sum(A_i[i] * (1 - a[i].sum()) ** 2 for i in range(N))
    L = np.asarray(d) @ a
    cap = lam2 * (L ** 2 - 2 * Q * L).sum()
    return f + pen + cap


def clustering_cost(assign, c):
    N = len(assign)
    return sum(c[i, j] for i in range(N) for j in range(i + 1, N) if assign[i] == assign[j])


def classify(bits, d, Q, Kp, c, f_opt):
    a = np.asarray(bits).reshape(len(d), Kp)
    if not (a.sum(axis=1) == 1).all():
        return "C1_violated"
    assign = a.argmax(axis=1)
    if any(sum(d[i] for i in range(len(d)) if assign[i] == k) > Q for k in range(Kp)):
        return "over_capacity"
    return "optimal" if math.isclose(clustering_cost(assign, c), f_opt, abs_tol=1e-9) else "feasible_suboptimal"


def main():
    rows = []
    for seed in range(N_SEEDS):
        inst = generate(TIER, seed)
        validate(inst)
        d, Q, Kp = inst["d"], inst["Q"], inst["K_min"]
        N = len(d)
        c = distances(inst["xy"])
        # constrained optimum over all K'^N assignments satisfying C2 (C1 holds by construction)
        f_opt = min(clustering_cost(a, c) for a in itertools.product(range(Kp), repeat=N)
                    if all(sum(d[i] for i in range(N) if a[i] == k) <= Q for k in range(Kp)))
        all_bits = np.array(list(itertools.product([0, 1], repeat=N * Kp)), dtype=float)
        for lam2 in LAMBDA2S:
            for mode in A_MODES:
                A_i, delta, bound = weights(c, d, Q, lam2, mode)
                M, const = build_qubo(c, d, Q, Kp, lam2, A_i)
                # check 1: uniform linear coefficients
                assert np.allclose(np.diag(M), -delta), "linear coefficients not uniform"
                # check 2: all couplings non-negative
                assert (np.triu(M, 1) >= -1e-12).all(), "negative coupling"
                # check 3: matrix energy == direct H(a) on every bitstring
                E = np.einsum("bi,ij,bj->b", all_bits, M, all_bits) + const
                for b, e in zip(all_bits, E):
                    assert math.isclose(e, H_direct(b, c, d, Q, Kp, lam2, A_i), rel_tol=1e-9, abs_tol=1e-9)
                # ground state(s)
                gs = all_bits[np.isclose(E, E.min(), rtol=0, atol=1e-9 * max(1, abs(E.min())))]
                cats = {classify(b, d, Q, Kp, c, f_opt) for b in gs}
                outcome = cats.pop() if len(cats) == 1 else "tie_mixed"
                rigorous_ok = bool((A_i > bound).all())
                if rigorous_ok:
                    assert outcome != "C1_violated", "Claim violated: check the derivation!"
                couplings = np.triu(M, 1)[np.triu(M, 1) > 1e-12]
                U_max = couplings.max()
                rows.append(dict(seed=seed, lam2=lam2, mode=mode, K_min=Kp, D=sum(d),
                                 rho=round(inst["rho"], 3), outcome=outcome, n_ground=len(gs),
                                 A_min=round(A_i.min(), 3), A_max=round(A_i.max(), 3),
                                 delta=round(delta, 3), above_bound=rigorous_ok,
                                 U_max=round(U_max, 3), dyn_range=round(U_max / couplings.min(), 1),
                                 delta_dev=round(55.4 * delta / U_max, 2)))
    with open("step1_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print(f"{len(rows)} runs on {N_SEEDS} tier-{TIER} instances; all internal checks passed.\n")
    cats = ["optimal", "feasible_suboptimal", "over_capacity", "C1_violated", "tie_mixed"]
    print(f"{'lam2':>5} {'A mode':>14} | " + " ".join(f"{x[:12]:>12}" for x in cats) +
          " | med dyn range | max delta_dev")
    for lam2 in LAMBDA2S:
        for mode in A_MODES:
            sub = [r for r in rows if r["lam2"] == lam2 and r["mode"] == mode]
            print(f"{lam2:>5} {mode:>14} | " + " ".join(f"{sum(r['outcome'] == x for r in sub):>12}" for x in cats) +
                  f" | {np.median([r['dyn_range'] for r in sub]):>13.1f} | {max(r['delta_dev'] for r in sub):>13.1f}")

    # one instance printed in full for the hand check
    inst = generate(TIER, 0); d, Q, Kp = inst["d"], inst["Q"], inst["K_min"]
    c = distances(inst["xy"]); A_i, delta, bound = weights(c, d, Q, 0.5, "rigorous")
    print("\nHAND CHECK (seed 0, lam2 = 0.5, rigorous):")
    print("  d =", d, " Q =", Q, " K_min =", Kp)
    print("  c =\n", np.round(c, 2))
    print("  A_i =", np.round(A_i, 3), " delta =", round(delta, 3), " bound_i =", np.round(bound, 3))


if __name__ == "__main__":
    main()