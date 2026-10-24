#!/usr/bin/env python3
"""Run classical baselines, route each clustering exactly, and append results.

    python scripts/run_baselines.py
    python scripts/run_baselines.py --tiers M --seeds 0 --out /tmp/baselines.csv

Shared reference/tour-table computation is evaluation overhead. Sweep uses the
same exact tour table to select among its 2N candidates; savings uses only the
distance matrix. Neither heuristic uses optimal partitions or certificates.
Infeasible outcomes are retained as failed rows, with no repair or finite gap.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qcvrp.baselines import BaselineResult, clarke_wright, sweep  # noqa: E402
from qcvrp.instances import Instance, validate_solution  # noqa: E402
from qcvrp.metrics import results_row, summarize, write_results  # noqa: E402
from qcvrp.reference import compute_reference  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances", type=Path, default=ROOT / "instances")
    ap.add_argument("--tiers", nargs="+", choices=["S", "M", "L"], default=["S", "M", "L"])
    ap.add_argument("--seeds", nargs="+", type=int, help="instance seeds (default: all files)")
    ap.add_argument("--methods", nargs="+", choices=["sweep", "savings"], default=["sweep", "savings"])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "results.csv")
    args = ap.parse_args()
    paths = sorted(p for tier in args.tiers for p in args.instances.glob(f"{tier}-*.json"))
    if args.seeds is not None:
        paths = [p for p in paths if int(p.stem.split("-")[1]) in args.seeds]
    if not paths:
        ap.error("no instance files match the requested tiers and seeds")

    rows = []
    for path in paths:
        inst = Instance.load(path)
        print(f"Computing reference for {inst.name}...", flush=True)
        ref = compute_reference(inst)
        for method in args.methods:
            clusters = sweep(inst, ref.tours) if method == "sweep" else clarke_wright(inst)
            routes = [ref.tours.tour_of(c) for c in clusters]
            _, errors = validate_solution(inst, routes)
            feasible = not errors
            result = BaselineResult(clusters=clusters, feasible=feasible)
            notes = ("all 2N angular sweeps; selected by exact routed cost" if method == "sweep"
                     else "descending Clarke-Wright savings; endpoint/capacity merges; stop at K")
            if errors:
                notes += "; INFEASIBLE: " + "; ".join(errors)
            row = results_row(inst, ref, method="classical", sampler=f"classical-{method}",
                              seed_run=0, result=result, emulator="classical", notes=notes)
            rows.append(row)
            if feasible:
                # results_row independently re-costs and validates these exact tours.
                cost = ref.routed_exactly(clusters)
                print(f"{inst.name} classical-{method}: vehicles={len(clusters)}/{inst.K} "
                      f"cost={cost:.10f} gap={row['cvrp_gap_pct']:.4f}%", flush=True)
                print(f"  routes: {json.dumps(routes)}", flush=True)
                print(f"  validate: {sys.executable} scripts/validate.py {path} "
                      f"--routes '{json.dumps(routes)}'", flush=True)
            else:
                print(f"{inst.name} classical-{method}: INFEASIBLE: {'; '.join(errors)}", flush=True)

    out = write_results(rows, args.out, append=True)
    print(f"\nwrote {len(rows)} rows to {out} (merged; same keys replaced)\n")
    print(summarize(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
