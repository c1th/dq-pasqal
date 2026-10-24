#!/usr/bin/env python3
"""Independently cross-check every tier-L CVRP reference with exact subset DP.

    python scripts/crosscheck_L.py

Prints full-precision comparisons and validates recovered routes using validate.py.
Exits nonzero on a disagreement or an invalid solution; never rewrites references.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qcvrp.instances import Instance, validate_solution  # noqa: E402
from qcvrp.reference import build_tour_table, subset_loads  # noqa: E402

TOL = 1e-6


def solve_subset_dp(inst: Instance, block_cost) -> tuple[float, list[int]]:
    """Return the minimum cost and block masks for at most K feasible blocks.

    f[k][mask] serves exactly mask with at most k vehicles. Every partition has
    one block containing mask's lowest bit, so considering only those blocks
    removes vehicle-label symmetry without omitting a partition. Empty masks
    cost zero at every layer, permitting fewer than K vehicles.

    Time O(K * 3**N), space O(K * 2**N) for reconstruction choices, with only
    two cost layers retained. No MILP results or reference routes enter the DP.
    """
    size = 1 << inst.N
    load = subset_loads(inst.d).tolist()
    cost = list(map(float, block_cost))
    feasible = [value <= inst.Q for value in load]
    previous = [math.inf] * size
    previous[0] = 0.0
    choices = []
    for k in range(1, inst.K + 1):
        current = [math.inf] * size
        current[0] = 0.0
        parent = [0] * size
        for mask in range(1, size):
            if load[mask] > k * inst.Q:
                continue  # total capacity is a necessary feasibility condition
            low = mask & -mask
            rest = mask ^ low
            sub = rest
            best = math.inf
            chosen = 0
            while True:
                block = sub | low
                if feasible[block]:
                    candidate = previous[mask ^ block] + cost[block]
                    if candidate < best:
                        best, chosen = candidate, block
                if sub == 0:
                    break
                sub = (sub - 1) & rest
            current[mask], parent[mask] = best, chosen
        choices.append(parent)
        previous = current

    optimum = previous[-1]
    if not math.isfinite(optimum):
        return optimum, []
    blocks = []
    mask = size - 1
    k = inst.K
    while mask:
        block = choices[k - 1][mask]
        if block == 0:
            raise RuntimeError("finite DP optimum has no reconstruction choice")
        blocks.append(block)
        mask ^= block
        k -= 1
    return optimum, blocks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances", type=Path, default=ROOT / "instances",
                    help="directory of L-*.json files and references.csv")
    args = ap.parse_args()
    with (args.instances / "references.csv").open(newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["tier"] == "L"]
    references = {row["instance"]: float(row["cvrp_ref"]) for row in rows}
    paths = sorted(args.instances.glob("L-*.json"))
    if not paths or len(references) != len(rows) or set(references) != {p.stem for p in paths}:
        ap.error("tier-L instance files and unique reference rows must match exactly")

    results = []
    success = True
    for path in paths:
        inst = Instance.load(path)
        if inst.tier != "L" or inst.name != path.stem:
            ap.error(f"instance metadata does not match tier-L filename: {path}")
        print(f"Checking {inst.name} (N={inst.N}, K={inst.K})...", flush=True)
        tours = build_tour_table(inst)
        optimum, blocks = solve_subset_dp(inst, tours.cost)
        reference = references[inst.name]
        agree = math.isfinite(optimum) and math.isclose(
            optimum, reference, rel_tol=0.0, abs_tol=TOL)
        results.append((inst.name, reference, optimum, agree))
        success = success and agree
        if not math.isfinite(optimum):
            print(f"{inst.name}: no feasible partition recovered", flush=True)
            continue
        routes = [tours.tour(mask) for mask in blocks]
        print("routes:", json.dumps(routes), flush=True)
        route_cost, errors = validate_solution(inst, routes)
        valid = not errors and math.isclose(route_cost, optimum, rel_tol=0.0, abs_tol=TOL)
        if not valid:
            print(f"{inst.name}: route cost={route_cost:.17g}, DP={optimum:.17g}, errors={errors}")
        # Invoke the existing CLI with this interpreter, including outside the repo cwd.
        checked = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "validate.py"),
             str(path.resolve()), "--routes", json.dumps(routes)],
            check=False,
        )
        success = success and valid and checked.returncode == 0

    print(f"\nAbsolute comparison tolerance: {TOL:g}")
    print(f"{'instance':<10} {'MILP optimum':>22} {'DP optimum':>22} {'agree?':>7}")
    for name, reference, optimum, agree in results:
        print(f"{name:<10} {reference:22.17g} {optimum:22.17g} {'YES' if agree else 'NO':>7}")
    if not success:
        print("FAILED: preserve and report any disagreement; references were not changed.")
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
