#!/usr/bin/env python3
"""Final results run: every (instance, sampler, seed) is SAMPLED ONCE and the same shots are
post-processed two ways, so the effect of the classical completion is isolated:

    +greedy  (v1)  overlap resolution by compactness, leftovers by greedy pairwise distance,
                   shot selection by within-cluster pairwise distance
    +exact   (v2)  overlap resolution by routed cost, leftovers by exhaustive assignment with
                   exact tour costs (<= 8 leftovers), shot selection by exact routed cost

    python scripts/run_final.py --samplers ideal-mis oracle random --runs 3 --out results/final.csv
    python scripts/run_final.py --samplers qutip --runs 3 --shots 500 --out results/final.csv --append
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from qcvrp.instances import Instance  # noqa: E402
from qcvrp.metrics import results_row, summarize, write_results  # noqa: E402
from qcvrp.models import PackingModel  # noqa: E402
from qcvrp.pipeline import packing_cluster  # noqa: E402
from qcvrp.pipeline_v2 import packing_cluster_v2  # noqa: E402
from qcvrp.reference import compute_reference  # noqa: E402
from qcvrp.sampling import SAMPLERS  # noqa: E402


def get_sampler(name):
    if name in SAMPLERS:
        return SAMPLERS[name]
    from qcvrp.pulses import QUANTUM_SAMPLERS
    return QUANTUM_SAMPLERS[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="instances")
    ap.add_argument("--tiers", nargs="+", default=["S", "M", "L"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="instance seeds (default: all files)")
    ap.add_argument("--samplers", nargs="+", default=["ideal-mis", "oracle", "random"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--shots", type=int, default=500)
    ap.add_argument("--fills", nargs="+", type=float, default=[0.85])
    ap.add_argument("--jaccard", type=float, default=0.6)
    ap.add_argument("--omega", type=float, default=2.0)
    ap.add_argument("--duration", type=int, default=6000)
    ap.add_argument("--det-shape", default="slow-end")
    ap.add_argument("--out", default="results/final.csv")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    paths = sorted(p for p in Path(args.instances).glob("*.json") if p.stem.split("-")[0] in args.tiers)
    if args.seeds is not None:
        paths = [p for p in paths if int(p.stem.split("-")[1]) in args.seeds]
    rows = []
    t_start = time.time()
    for p in paths:
        inst = Instance.load(p)
        ref = compute_reference(inst)
        for fill in args.fills:
            model = PackingModel(fill=fill, max_jaccard=args.jaccard)
            for sname in args.samplers:
                sampler = get_sampler(sname)
                quantum = sname in ("qutip", "emu-sv")
                sampler_kw = dict(omega_max=args.omega, duration_ns=args.duration, det_shape=args.det_shape) if quantum else {}
                for run in range(args.runs if sname != "oracle" else 1):
                    t0 = time.time()
                    # sample once ...
                    captured = {}

                    def once(tgt, emb, delta, n, rng, **kw):
                        if "counts" not in captured:
                            captured["counts"] = sampler(tgt, emb, delta, n, rng, **sampler_kw)
                        return captured["counts"]

                    r2 = packing_cluster_v2(inst, model, once, args.shots, ref.tours, seed=run)
                    r1 = packing_cluster(inst, model, once, args.shots, seed=run)
                    assert r1.n_shots == r2.n_shots
                    emulator = {"qutip": "pulser-simulation (QuTiP, exact)", "emu-sv": "emu-sv"}.get(sname, "classical")
                    pulse = (f"interp amp 0->{args.omega}->0 rad/us, {args.det_shape} det -1.5*delta_f->delta_f, "
                             f"{args.duration} ns") if quantum else ""
                    base = f"pack(fill={fill},jaccard={args.jaccard})"
                    rows.append(results_row(inst, ref, base + "+greedy", sname, run, r1, emulator=emulator, pulse=pulse))
                    rows.append(results_row(inst, ref, base + "+exact", sname, run, r2, emulator=emulator, pulse=pulse))
                    print(f"{inst.name} fill={fill} {sname:9s} run{run} | atoms={r2.max_atoms} raw_feas={r2.feasibility_rate_raw:.2f} "
                          f"P(MIS)={r2.p_mis_shot:.2f} cov={r2.coverage_best:.2f} | gap greedy={rows[-2]['cvrp_gap_pct']:.1f}% "
                          f"exact={rows[-1]['cvrp_gap_pct']:.1f}% [{time.time()-t0:.1f}s]")
    out = write_results(rows, args.out, append=args.append)
    print(f"\nwrote {len(rows)} rows to {out} [{time.time()-t_start:.0f}s total]\n")
    print(summarize(rows))


if __name__ == "__main__":
    main()
