#!/usr/bin/env python3
"""Figures for the report.

    python scripts/plot_results.py --results results/results.csv --out report/figs
    python scripts/plot_results.py --register instances/L-0000.json --sampler qutip --out report/figs

--results  : per-tier bar charts of CVRP gap (mean / best) and P(maximum packing per shot)
             for every sampler in the CSV (the quantum-vs-random ablation in one picture).
--register : the embedded register for one instance (atoms = candidate clusters, blockade
             discs at the final detuning, target overlap edges green if realised above delta,
             red if violable) next to the map with the chosen clusters and exact routes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qcvrp.device import separation  # noqa: E402
from qcvrp.instances import Instance  # noqa: E402
from qcvrp.metrics import evaluate_clustering  # noqa: E402
from qcvrp.models import PackingModel  # noqa: E402
from qcvrp.pipeline import packing_cluster  # noqa: E402
from qcvrp.reference import compute_reference  # noqa: E402
from qcvrp.sampling import SAMPLERS  # noqa: E402

COLORS = {"ideal-mis": "#4C72B0", "oracle": "#55A868", "qutip": "#C44E52", "emu-sv": "#8172B2", "random": "#937860"}


def plot_results(csv_path: Path, out: Path):
    df = pd.read_csv(csv_path)
    for col in ("cvrp_gap_pct", "p_mis_shot", "feasibility_rate_raw", "coverage_best"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["feasible"] = df["feasible"].astype(str).str.lower().isin(["true", "1"])
    tiers = [t for t in ["S", "M", "L"] if t in set(df["tier"])]
    samplers = [s for s in COLORS if s in set(df["sampler"])] + sorted(set(df["sampler"]) - set(COLORS))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    width = 0.8 / max(len(samplers), 1)
    for si, s in enumerate(samplers):
        g = df[(df["sampler"] == s) & df["feasible"]]
        means = [g[g.tier == t]["cvrp_gap_pct"].mean() for t in tiers]
        bests = [g[g.tier == t]["cvrp_gap_pct"].min() for t in tiers]
        pmis = [df[(df["sampler"] == s) & (df.tier == t)]["p_mis_shot"].mean() for t in tiers]
        x = np.arange(len(tiers)) + (si - (len(samplers) - 1) / 2) * width
        c = COLORS.get(s, None)
        axes[0].bar(x, means, width, label=s, color=c)
        axes[1].bar(x, bests, width, label=s, color=c)
        axes[2].bar(x, pmis, width, label=s, color=c)
    for ax, title, ylabel in zip(axes, ["CVRP gap, mean over instances", "CVRP gap, best instance",
                                        "P(shot is a maximum packing)"],
                                 ["gap to proven optimum (%)", "gap to proven optimum (%)", "fraction of shots"]):
        ax.set_xticks(np.arange(len(tiers)))
        ax.set_xticklabels([f"tier {t}" for t in tiers])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[2].set_ylim(0, 1.0)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    p = out / "results_summary.png"
    fig.savefig(p, dpi=160)
    print("wrote", p)


def plot_register(inst_path: Path, sampler: str, out: Path, shots: int = 500, seed: int = 0, fill: float = 0.85,
                  jaccard: float = 0.6):
    inst = Instance.load(inst_path)
    ref = compute_reference(inst)
    if sampler in SAMPLERS:
        samp = SAMPLERS[sampler]
    else:
        from qcvrp.pulses import QUANTUM_SAMPLERS
        samp = QUANTUM_SAMPLERS[sampler]
    res = packing_cluster(inst, PackingModel(fill=fill, max_jaccard=jaccard), samp, shots, seed=seed)
    emb, tgt, cands = res.embedding, res.target, res.candidates
    delta = res.levels[0].delta
    r_block = separation(delta)  # pairs closer than this couple above delta
    pos = emb.positions
    n = len(pos)
    ev = evaluate_clustering(inst, ref, res.clusters)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 6))
    # --- register
    circle = plt.Circle((0, 0), 38, fill=False, ls="--", color="grey", lw=0.8)
    ax1.add_patch(circle)
    for a in range(n):
        ax1.add_patch(plt.Circle(pos[a], r_block / 2, color="#4C72B0", alpha=0.10))
    for a in range(n):
        for b in range(a + 1, n):
            if tgt.J[a, b] > 0:
                ok = emb.U[a, b] > delta
                ax1.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]], color="#55A868" if ok else "#C44E52",
                         lw=1.6 if ok else 1.2, ls="-" if ok else "--", zorder=1)
    chosen = set()
    if res.levels[0].best_bits:
        chosen = {a for a, ch in enumerate(res.levels[0].best_bits) if ch == "1"}
    ax1.scatter(pos[:, 0], pos[:, 1], s=[160 if a in chosen else 70 for a in range(n)],
                c=["#C44E52" if a in chosen else "black" for a in range(n)], zorder=3)
    for a in range(n):
        ax1.annotate(str(a), pos[a] + np.array([0.6, 0.6]), fontsize=8)
    f = emb.fidelity
    ax1.set_title(f"register: {n} atoms, {f['n_edges']} overlap edges, {f['violable_edges']} violable, "
                  f"{f['spurious_edges']} spurious\nblockade at delta_f={delta:.2f} rad/us -> r={r_block:.1f} um "
                  f"(discs = r/2); red atoms = best shot", fontsize=9)
    ax1.set_xlim(-40, 40)
    ax1.set_ylim(-40, 40)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x (um)")
    ax1.set_ylabel("y (um)")
    # --- map with clusters and routes
    P = inst.points()
    ax2.scatter(P[0, 0], P[0, 1], marker="s", s=120, c="black", zorder=4, label="depot")
    cmap = plt.get_cmap("tab10")
    for k, cl in enumerate(res.clusters):
        tour = ref.tours.tour_of(cl)
        pts = np.vstack([P[0], P[[c + 1 for c in tour]], P[0]])
        ax2.plot(pts[:, 0], pts[:, 1], color=cmap(k), lw=1.2, zorder=2)
        ax2.scatter(inst.xy[cl, 0], inst.xy[cl, 1], s=60 + 18 * inst.d[cl], color=cmap(k), zorder=3,
                    label=f"vehicle {k}: load {sum(inst.demands[c] for c in cl)}/{inst.Q}")
    for i in range(inst.N):
        ax2.annotate(str(i), inst.xy[i] + 1.2, fontsize=8)
    for a in chosen:
        c = cands[a]
        ax2.add_patch(plt.Circle(c.center, c.radius, fill=False, ls=":", color="grey", lw=0.9))
    ax2.set_title(f"{inst.name}: {sampler} best shot -> clusters routed exactly\n"
                  f"CVRP gap {ev.get('cvrp_gap_pct', float('nan')):.1f}%  (model gap {ev.get('model_gap_pct', float('nan')):.1f}%, "
                  f"coverage {res.coverage_best:.2f}, raw feasibility {res.feasibility_rate_raw:.2f}, P(MIS) {res.p_mis_shot:.2f})",
                  fontsize=9)
    ax2.set_xlim(-3, 103)
    ax2.set_ylim(-3, 103)
    ax2.set_aspect("equal")
    ax2.legend(fontsize=7, loc="upper right", frameon=False)
    fig.tight_layout()
    p = out / f"register_{inst.name}_{sampler}.png"
    fig.savefig(p, dpi=160)
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    ap.add_argument("--register", default=None, help="instance JSON to draw")
    ap.add_argument("--sampler", default="oracle")
    ap.add_argument("--shots", type=int, default=500)
    ap.add_argument("--out", default="report/figs")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.results:
        plot_results(Path(args.results), out)
    if args.register:
        plot_register(Path(args.register), args.sampler, out, shots=args.shots)


if __name__ == "__main__":
    main()
