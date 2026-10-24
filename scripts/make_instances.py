#!/usr/bin/env python3
"""Generate certified instances and their proven-optimal reference values.

    python scripts/make_instances.py --tiers S M L --seeds 0 1 2 3 4 --out instances

Writes instances/<tier>-<seed>.json (instance + certificate) and instances/references.csv
(CVRP optimum, clustering-model optimum, decomposition gap, counts of optimal solutions).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qcvrp.instances import generate, validate_instance, validate_solution  # noqa: E402
from qcvrp.reference import compute_reference  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+", default=["S", "M", "L"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--out", default="instances")
    ap.add_argument("--rho", nargs=2, type=float, default=None, help="load-ratio band, e.g. 0.8 0.95")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for tier in args.tiers:
        for seed in args.seeds:
            t0 = time.time()
            inst = generate(tier, seed, rho_range=tuple(args.rho) if args.rho else None)
            errs = validate_instance(inst)
            assert not errs, errs
            inst.save(out / f"{inst.name}.json")
            ref = compute_reference(inst)
            cost, errs = validate_solution(inst, ref.cvrp_routes)
            assert not errs and abs(cost - ref.cvrp_cost) < 1e-6
            s = ref.summary()
            s.update(tier=tier, seed=seed, N=inst.N, K=inst.K, Q=inst.Q, load_ratio=round(inst.load_ratio, 4),
                     dmax_ratio=round(inst.dmax_ratio, 4), feasibility_method=inst.feasibility_method,
                     cvrp_routes=str(ref.cvrp_routes), model_blocks=str(ref.model_blocks),
                     seconds=round(time.time() - t0, 2))
            rows.append(s)
            print(f"{inst.name}: rho={inst.load_ratio:.2f} dmax/Q={inst.dmax_ratio:.2f} C*={ref.cvrp_cost:.2f} "
                  f"({ref.cvrp_status}; {s['n_optimal_cvrp_clusterings']} optimal clusterings) "
                  f"model_opt={ref.model_cost:.2f} decomposition gap={s['decomposition_gap_pct']:.1f}%  [{s['seconds']}s]")
    with (out / "references.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} instances + references to {out}/")


if __name__ == "__main__":
    main()
