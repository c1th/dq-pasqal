#!/usr/bin/env python3
"""Run the ball-packing pipeline on saved instances and write results.csv.

Classical samplers (no pulser needed):
    python scripts/run_pipeline.py --instances instances --samplers ideal-mis oracle random --out results/classical.csv
Quantum emulation (needs pulser; QuTiP backend is exact up to ~12 atoms):
    python scripts/run_pipeline.py --instances instances --samplers qutip --shots 500 --out results/qutip.csv

Every row is one (instance, method, sampler, run seed) with the metrics of Section 9.
Use --model split for the hierarchical conflict-split model instead of ball packing.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qcvrp.instances import Instance  # noqa: E402
from qcvrp.metrics import results_row, summarize, write_results  # noqa: E402
from qcvrp.models import ConflictSplitModel, PackingModel  # noqa: E402
from qcvrp.pipeline import hierarchical_cluster, packing_cluster  # noqa: E402
from qcvrp.reference import compute_reference  # noqa: E402
from qcvrp.sampling import SAMPLERS  # noqa: E402


def get_sampler(name: str):
    if name in SAMPLERS:
        return SAMPLERS[name]
    from qcvrp.pulses import QUANTUM_SAMPLERS
    return QUANTUM_SAMPLERS[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="instances")
    ap.add_argument("--tiers", nargs="+", default=["S", "M", "L"])
    ap.add_argument("--samplers", nargs="+", default=["ideal-mis", "oracle", "random"])
    ap.add_argument("--runs", type=int, default=3, help="independent runs per (instance, sampler)")
    ap.add_argument("--shots", type=int, default=500)
    ap.add_argument("--model", choices=["pack", "split"], default="pack")
    ap.add_argument("--fill", type=float, default=0.85)
    ap.add_argument("--jaccard", type=float, default=0.6)
    ap.add_argument("--grid", type=int, default=0)
    ap.add_argument("--extra-fills", nargs="*", type=float, default=[], help="extra fill levels per centre, e.g. 0.55")
    ap.add_argument("--m-far", type=int, default=2)
    ap.add_argument("--omega", type=float, default=4.0)
    ap.add_argument("--duration", type=int, default=4000)
    ap.add_argument("--det-shape", default="linear", choices=["linear", "slow-end", "slow-start"])
    ap.add_argument("--out", default="results/results.csv")
    ap.add_argument("--append", action="store_true", help="merge into an existing CSV (same keys are replaced)")
    args = ap.parse_args()

    paths = sorted(p for p in Path(args.instances).glob("*.json") if p.stem.split("-")[0] in args.tiers)
    rows = []
    for p in paths:
        inst = Instance.load(p)
        ref = compute_reference(inst)
        for sname in args.samplers:
            sampler = get_sampler(sname)
            sampler_kw = (dict(omega_max=args.omega, duration_ns=args.duration, det_shape=args.det_shape)
                          if sname in ("qutip", "emu-sv") else {})
            for run in range(args.runs if sname != "oracle" else 1):
                t0 = time.time()
                if args.model == "pack":
                    model = PackingModel(fill=args.fill, max_jaccard=args.jaccard, grid=args.grid,
                                         extra_fills=tuple(args.extra_fills))
                    res = packing_cluster(inst, model, sampler, args.shots, seed=run, sampler_kw=sampler_kw)
                    method = (f"ball-packing(fill={args.fill}{'+' + ','.join(map(str, args.extra_fills)) if args.extra_fills else ''},"
                              f"jaccard={args.jaccard},grid={args.grid})")
                else:
                    model = ConflictSplitModel(m_far=args.m_far)
                    res = hierarchical_cluster(inst, model, sampler, args.shots, seed=run, sampler_kw=sampler_kw)
                    method = f"conflict-split(m_far={args.m_far})"
                emulator = {"qutip": "pulser-simulation (QuTiP, exact)", "emu-sv": "emu-sv"}.get(sname, "classical")
                pulse = (f"interp amp 0->{args.omega}->0 rad/us, {args.det_shape} det -1.5*delta_f->delta_f, {args.duration} ns"
                         if sname in ("qutip", "emu-sv") else "")
                row = results_row(inst, ref, method, sname, run, res, emulator=emulator, pulse=pulse)
                rows.append(row)
                lv = res.levels[0]
                f = lv.fidelity
                print(f"{inst.name} {sname:9s} run{run} | atoms={lv.n_atoms} edges={f.get('n_edges', lv.n_edges)} "
                      f"violable={f.get('violable_edges', '-')} spurious={f.get('spurious_edges', '-')} delta={lv.delta:.2f} | "
                      f"feas_raw={getattr(res, 'feasibility_rate_raw', float('nan')):.2f} P(MIS)={getattr(res, 'p_mis_shot', float('nan')):.2f} feas={res.min_feasibility_rate:.2f} "
                      f"cov={getattr(res, 'coverage_best', float('nan')):.2f} | model gap={row['model_gap_pct']:.1f}% "
                      f"CVRP gap={row['cvrp_gap_pct']:.1f}% (decomp {row['decomposition_gap_pct']:.1f}%) [{time.time()-t0:.1f}s]")
    out = write_results(rows, args.out, append=args.append)
    print(f"\nwrote {len(rows)} rows to {out}" + (" (merged)" if args.append else "") + "\n")
    print(summarize(rows))


if __name__ == "__main__":
    main()
