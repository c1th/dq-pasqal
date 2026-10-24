"""Gaps and results rows (Section 9).

gap = (C - C*) / C* * 100 on feasible solutions only.  Levels reported:
    model gap   : clustering objective (within-cluster pairwise distance) of the best
                  feasible clustering vs the optimum of the constrained clustering model
    CVRP gap    : best feasible clustering routed exactly vs the proven CVRP optimum
    decomposition gap (instance property): model optimum routed exactly vs CVRP optimum
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from .instances import Instance
from .reference import Reference


def gap_pct(C: float, C_star: float) -> float:
    return 100.0 * (C - C_star) / C_star


def evaluate_clustering(inst: Instance, ref: Reference, clusters: Sequence[Sequence[int]]) -> dict:
    clusters = [list(c) for c in clusters if len(c)]
    feasible = (len(clusters) <= inst.K and sorted(sum(clusters, [])) == list(range(inst.N))
                and all(sum(inst.demands[c] for c in cl) <= inst.Q for cl in clusters))
    if not feasible:
        return dict(feasible=False, model_obj=np.nan, model_gap_pct=np.nan, routed_exactly=np.nan, cvrp_gap_pct=np.nan)
    obj = ref.clustering_objective(clusters)
    routed = ref.routed_exactly(clusters)
    return dict(
        feasible=True,
        model_obj=obj,
        model_gap_pct=gap_pct(obj, ref.model_cost),
        routed_exactly=routed,
        cvrp_gap_pct=gap_pct(routed, ref.cvrp_cost),
        clusters=clusters,
        routes=[ref.tours.tour_of(c) for c in clusters],
    )


RESULT_COLUMNS = [
    "instance", "tier", "seed_instance", "method", "sampler", "seed_run",
    "N", "K", "Q", "load_ratio", "dmax_ratio",
    "feasible", "repaired", "model_obj", "model_gap_pct", "routed_exactly", "cvrp_gap_pct",
    "feasibility_rate_min", "feasibility_rate_levels", "n_shots_per_level",
    "atoms_max", "atoms_levels", "edges_levels", "fidelity_log_rms_levels", "unwanted_ratio_levels",
    "violable_edges_levels", "spurious_edges_levels", "coverage_best", "feasibility_rate_raw",
    "model_gap_pct_any", "cvrp_gap_pct_any", "mis_size", "p_mis_shot",
    "delta_levels", "cvrp_ref", "cvrp_ref_status", "model_opt", "decomposition_gap_pct",
    "emulator", "emulator_params", "pulse", "notes",
]


def results_row(inst: Instance, ref: Reference, method: str, sampler: str, seed_run: int, result, emulator: str = "",
                emulator_params: str = "", pulse: str = "", notes: str = "") -> dict:
    ev = evaluate_clustering(inst, ref, result.clusters)
    ev_any = evaluate_clustering(inst, ref, getattr(result, "clusters_any", result.clusters) or result.clusters)
    levels = result.levels
    row = dict(
        instance=inst.name, tier=inst.tier, seed_instance=inst.seed, method=method, sampler=sampler, seed_run=seed_run,
        N=inst.N, K=inst.K, Q=inst.Q, load_ratio=round(inst.load_ratio, 4), dmax_ratio=round(inst.dmax_ratio, 4),
        feasible=ev["feasible"] and result.feasible, repaired=result.repaired,
        model_obj=ev["model_obj"], model_gap_pct=ev["model_gap_pct"],
        routed_exactly=ev["routed_exactly"], cvrp_gap_pct=ev["cvrp_gap_pct"],
        feasibility_rate_min=result.min_feasibility_rate,
        feasibility_rate_levels=";".join(f"{l.feasibility_rate:.3f}" for l in levels),
        n_shots_per_level=";".join(str(l.n_shots) for l in levels),
        atoms_max=result.max_atoms,
        atoms_levels=";".join(str(l.n_atoms) for l in levels),
        edges_levels=";".join(str(l.n_edges) for l in levels),
        fidelity_log_rms_levels=";".join(f"{l.fidelity.get('log_rms', float('nan')):.3f}" for l in levels),
        unwanted_ratio_levels=";".join(f"{l.fidelity.get('unwanted_ratio', float('nan')):.3f}" for l in levels),
        delta_levels=";".join(f"{l.delta:.2f}" for l in levels),
        violable_edges_levels=";".join(str(l.fidelity.get("violable_edges", "")) for l in levels),
        spurious_edges_levels=";".join(str(l.fidelity.get("spurious_edges", "")) for l in levels),
        coverage_best=getattr(result, "coverage_best", float("nan")),
        feasibility_rate_raw=getattr(result, "feasibility_rate_raw", float("nan")),
        model_gap_pct_any=ev_any["model_gap_pct"],
        cvrp_gap_pct_any=ev_any["cvrp_gap_pct"],
        mis_size=getattr(result, "mis_size", ""),
        p_mis_shot=getattr(result, "p_mis_shot", float("nan")),
        cvrp_ref=ref.cvrp_cost, cvrp_ref_status="proven optimum", model_opt=ref.model_cost,
        decomposition_gap_pct=gap_pct(ref.model_routed_exactly, ref.cvrp_cost),
        emulator=emulator, emulator_params=emulator_params, pulse=pulse, notes=notes,
    )
    return row


def write_results(rows: list[dict], path: str | Path, append: bool = False) -> Path:
    """Write rows to CSV. With append=True, existing rows are kept and rows with the same
    (instance, method, sampler, seed_run) key are replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = lambda r: (str(r["instance"]), str(r["method"]), str(r["sampler"]), str(r["seed_run"]))
    merged: dict = {}
    if append and path.exists():
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                merged[key(r)] = r
    for r in rows:
        merged[key(r)] = r
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in merged.values():
            w.writerow(r)
    return path


def summarize(rows: list[dict]) -> str:
    """Per (tier, method) summary: mean/best gaps, feasibility, P(reach 1%)."""
    import pandas as pd
    df = pd.DataFrame(rows)
    if df.empty:
        return "(no rows)"
    for col in ("cvrp_gap_pct", "model_gap_pct", "cvrp_gap_pct_any", "atoms_max", "feasibility_rate_raw", "coverage_best", "p_mis_shot"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("feasible", "repaired"):
        df[col] = df[col].astype(str).str.lower().isin(["true", "1"])
    out = []
    for (tier, method, sampler), g in df.groupby(["tier", "method", "sampler"]):
        feas = g["feasible"].astype(bool)
        gf = g[feas]
        out.append(
            f"{tier} {method[:28]:28s} {sampler:9s} runs={len(g):3d} feasible={feas.mean():.2f} "
            f"repaired={g['repaired'].mean():.2f} raw_feas={g['feasibility_rate_raw'].mean():.2f} "
            f"P(MIS/shot)={g['p_mis_shot'].mean():.2f} cov={g['coverage_best'].mean():.2f} | "
            f"CVRP gap mean={gf['cvrp_gap_pct'].mean():6.2f}% best={gf['cvrp_gap_pct'].min():6.2f}% "
            f"P(gap<=1%)={(gf['cvrp_gap_pct'] <= 1.0).mean() if len(gf) else 0:.2f} | "
            f"any-shot gap mean={gf['cvrp_gap_pct_any'].mean():6.2f}% | atoms_max={g['atoms_max'].max()}"
        )
    return "\n".join(out)
