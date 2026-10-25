"""Step 3: first end-to-end runs on tier S (emulator, baseline adiabatic pulse).

For each tier-S instance (seeds 0-19) and each embedder (ladder, frobenius), plus a random-sampler baseline:
  1. load the Step 2 / 2b atom positions and final detuning,
  2. build a Pulser Sequence on AnalogDevice with the baseline pulse (3-point amplitude ramp, 3-point detuning sweep),
  3. emulate exactly (pulser-simulation, QutipBackendV2) and sample SHOTS bitstrings from the final state,
  4. decode every shot, evaluate it raw and after a light repair, and compute the Sec. 9 metrics,
  5. write one row per (instance, method, seed) to results.csv.
Needs instances.py, step1_qubo_check.py, step2_results.csv, step2b_results.csv, step2_positions/, step2b_positions/.
"""
import csv, itertools, math
import numpy as np
from pulser import Register, Sequence, Pulse
from pulser.waveforms import InterpolatedWaveform
from pulser.devices import AnalogDevice
from pulser_simulation import QutipBackendV2
from instances import generate, validate, DEPOT
from step1_qubo_check import distances, clustering_cost

TIER, SEEDS = "S", range(20)
SHOTS = 1000
T_NS = 4000                      # pulse duration (ns), multiple of the 4 ns clock, <= 6000
OMEGA = 2 * np.pi * 1.5          # peak Rabi frequency (rad/us), <= 12.57
EPS_GAP = 0.0                    # "reached the reference": model gap <= EPS_GAP percent
METHODS = ["ladder", "frobenius", "random"]


# ---------------- decoding, repair, evaluation ----------------
def decode(bitstring, N, Kp):
    return np.array([int(b) for b in bitstring]).reshape(N, Kp)       # atom i*Kp + k <-> a_ik


def loads(a, d):
    return np.asarray(d) @ a


def is_feasible(a, d, Q):
    return bool((a.sum(1) == 1).all() and (loads(a, d) <= Q).all())


def f_value(a, c):
    assign = a.argmax(1)
    return clustering_cost(assign, c)


def repair(a, d, Q, c):
    """Light, fixed repair: (1) keep the cheapest vehicle of a doubly assigned customer, (2) place unassigned
    customers on the cheapest vehicle with room, (3) move single customers out of overloaded vehicles.
    Returns a feasible assignment or None. At most O(N^2 K') work per shot."""
    a = a.copy(); N, Kp = a.shape; d = np.asarray(d)
    added = lambda i, k: sum(c[i, j] for j in range(N) if j != i and a[j, k])
    for i in range(N):
        on = np.flatnonzero(a[i])
        if len(on) > 1:
            keep = min(on, key=lambda k: added(i, k)); a[i] = 0; a[i, keep] = 1
    for i in sorted(range(N), key=lambda i: -d[i]):
        if a[i].sum() == 0:
            room = [k for k in range(Kp) if loads(a, d)[k] + d[i] <= Q]
            a[i, min(room or range(Kp), key=lambda k: added(i, k))] = 1
    for _ in range(N):
        L = loads(a, d)
        over = [k for k in range(Kp) if L[k] > Q]
        if not over:
            return a
        k = over[0]; best = None
        for i in np.flatnonzero(a[:, k]):
            for l in range(Kp):
                if l != k and L[l] + d[i] <= Q:
                    a[i, k] = 0; delta = added(i, l) - added(i, k); a[i, k] = 1
                    if best is None or delta < best[0]:
                        best = (delta, i, l)
        if best is None:
            return None
        _, i, l = best; a[i, k] = 0; a[i, l] = 1
    return a if is_feasible(a, d, Q) else None


def tour_cost(cluster, P):
    """Exact TSP from the depot (node 0) through the customers in `cluster` (brute force; tier S)."""
    if not cluster:
        return 0.0
    best = math.inf
    for perm in itertools.permutations(cluster):
        route = [0] + [i + 1 for i in perm] + [0]
        best = min(best, sum(np.linalg.norm(P[route[t]] - P[route[t + 1]]) for t in range(len(route) - 1)))
    return best


def routed_cost(assign, Kp, P):
    return sum(tour_cost([i for i in range(len(assign)) if assign[i] == k], P) for k in range(Kp))


def gap(C, Cref):
    return 100.0 * (C - Cref) / Cref


# ---------------- emulation ----------------
def final_probabilities(pos, delta_f):
    reg = Register.from_coordinates(pos, prefix="q")
    seq = Sequence(reg, AnalogDevice)
    seq.declare_channel("rg", "rydberg_global")
    amp = InterpolatedWaveform(T_NS, [1e-9, OMEGA, 1e-9])
    det = InterpolatedWaveform(T_NS, [-delta_f, 0.0, delta_f])
    seq.add(Pulse(amp, det, 0), "rg")
    seq.measure("ground-rydberg")
    res = QutipBackendV2(seq).run()
    assert list(res.atom_order) == [f"q{m}" for m in range(len(pos))], "unexpected atom order"
    probs = res.final_state.bitstring_probabilities()
    assert all(set(b) <= {"0", "1"} for b in probs), "unexpected bitstring format"
    return probs


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def main():
    step2 = {(int(r["seed"]), r["method"]): r for r in read_csv("step2_results.csv")}
    step2b = {int(r["seed"]): r for r in read_csv("step2b_results.csv")}
    rows = []
    for seed in SEEDS:
        inst = generate(TIER, seed); validate(inst)
        d, Q, K, Kp = inst["d"], inst["Q"], inst["K"], inst["K_min"]; N = len(d)
        c = distances(inst["xy"])
        P = np.vstack([DEPOT, inst["xy"]])
        feas_assign = [a for a in itertools.product(range(Kp), repeat=N)
                       if all(sum(d[i] for i in range(N) if a[i] == k) <= Q for k in range(Kp))]
        f_opt = min(clustering_cost(a, c) for a in feas_assign)
        # CVRP reference: full fleet K, exact routing of every capacity-feasible assignment (proven optimum)
        cvrp_opt = min(routed_cost(a, K, P) for a in itertools.product(range(K), repeat=N)
                       if all(sum(d[i] for i in range(N) if a[i] == k) <= Q for k in range(K)))
        rng = np.random.default_rng(1000 + seed)
        for method in METHODS:
            n = N * Kp
            if method == "random":
                shots = ["".join(map(str, rng.integers(0, 2, n))) for _ in range(SHOTS)]
                p_opt_exact, fid_metric, fid, delta_f = math.nan, "", math.nan, math.nan
            else:
                if method == "ladder":
                    pos = np.load(f"step2b_positions/seed{seed}_ladder.npy")
                    meta = step2b[seed]; fid_metric, fid = "rel_valid_energy_err", float(meta["rel_valid_energy_err"])
                else:
                    pos = np.load(f"step2_positions/seed{seed}_frobenius.npy")
                    meta = step2[(seed, "frobenius")]; fid_metric, fid = "rel_frob_err", float(meta["rel_frob_err"])
                delta_f = float(meta["delta_dev"])
                probs = final_probabilities(pos, delta_f)
                keys = list(probs); pv = np.array([probs[k] for k in keys]); pv = pv / pv.sum()
                shots = [keys[i] for i in rng.choice(len(keys), size=SHOTS, p=pv)]
                p_opt_exact = sum(p for b, p in probs.items()
                                  if is_feasible(decode(b, N, Kp), d, Q)
                                  and math.isclose(f_value(decode(b, N, Kp), c), f_opt, abs_tol=1e-9))
            raw_f, rep_f, raw_assign, rep_assign = [], [], {}, {}
            for b in shots:
                a = decode(b, N, Kp)
                if is_feasible(a, d, Q):
                    fv = f_value(a, c); raw_f.append(fv); raw_assign[fv] = tuple(a.argmax(1))
                ar = repair(a, d, Q, c)
                if ar is not None:
                    fv = f_value(ar, c); rep_f.append(fv); rep_assign[fv] = tuple(ar.argmax(1))
                else:
                    rep_f.append(None)
            reached = lambda fs: sum(1 for fv in fs if fv is not None and gap(fv, f_opt) <= EPS_GAP + 1e-9) / SHOTS
            best_raw = min(raw_f) if raw_f else None
            rep_ok = [fv for fv in rep_f if fv is not None]
            best_rep = min(rep_ok) if rep_ok else None
            rows.append(dict(
                tier=TIER, seed=seed, method=method, atoms=n, K_min=Kp, rho=round(inst["rho"], 3), shots=SHOTS,
                emulator="pulser-simulation QutipBackendV2 (exact)" if method != "random" else "uniform random",
                T_ns=T_NS, omega=round(OMEGA, 3), delta_f=round(delta_f, 2) if not math.isnan(delta_f) else "",
                emb_fidelity_metric=fid_metric, emb_fidelity="" if math.isnan(fid) else fid,
                feas_rate_raw=round(len(raw_f) / SHOTS, 4),
                best_model_gap_raw="undefined" if best_raw is None else round(gap(best_raw, f_opt), 3),
                p_reach_raw=round(reached(raw_f), 4),
                p_opt_exact="" if math.isnan(p_opt_exact) else round(p_opt_exact, 4),
                feas_rate_rep=round(len(rep_ok) / SHOTS, 4),
                best_model_gap_rep="undefined" if best_rep is None else round(gap(best_rep, f_opt), 3),
                p_reach_rep=round(reached(rep_f), 4),
                cvrp_gap_raw="undefined" if best_raw is None else round(gap(routed_cost(raw_assign[best_raw], Kp, P), cvrp_opt), 3),
                cvrp_gap_rep="undefined" if best_rep is None else round(gap(routed_cost(rep_assign[best_rep], Kp, P), cvrp_opt), 3),
                eps_gap_pct=EPS_GAP, ref_status="proven optimum (brute force)"))
        print(f"seed {seed} done", flush=True)

    with open("results.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)

    print(f"\nTier {TIER}, {len(SEEDS)} instances, {SHOTS} shots, T = {T_NS} ns, Omega = {OMEGA:.2f} rad/us, eps = {EPS_GAP}%")
    hdr = ["feas raw", "gap0 raw", "med gap raw", "P(reach) raw", "P(opt) exact", "feas rep", "gap0 rep", "med CVRP gap rep"]
    print(f"{'method':>10} | " + " | ".join(f"{h:>14}" for h in hdr))
    for m in METHODS:
        sub = [r for r in rows if r["method"] == m]
        num = lambda key: [r[key] for r in sub if r[key] not in ("undefined", "")]
        zero = lambda key: sum(1 for r in sub if r[key] not in ("undefined", "") and r[key] <= 1e-9)
        vals = [np.mean(num("feas_rate_raw")), f"{zero('best_model_gap_raw')}/{len(sub)}",
                np.median(num("best_model_gap_raw")) if num("best_model_gap_raw") else float("nan"),
                np.mean(num("p_reach_raw")),
                np.mean(num("p_opt_exact")) if num("p_opt_exact") else float("nan"),
                np.mean(num("feas_rate_rep")), f"{zero('best_model_gap_rep')}/{len(sub)}",
                np.median(num("cvrp_gap_rep")) if num("cvrp_gap_rep") else float("nan")]
        print(f"{m:>10} | " + " | ".join(f"{v:>14}" if isinstance(v, str) else f"{v:>14.3f}" for v in vals))


if __name__ == "__main__":
    main()