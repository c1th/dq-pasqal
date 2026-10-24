#!/usr/bin/env python3
"""Validator CLI (deliverable 3): check an instance file and its certificate, and
optionally a solution given as JSON routes.

    python scripts/validate.py instances/M-0000.json
    python scripts/validate.py instances/M-0000.json --routes '[[3,4,2,8,7,0],[9,1,5,6]]'
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qcvrp.instances import Instance, validate_instance, validate_solution  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--routes", default=None, help="JSON list of routes (customer indices)")
    args = ap.parse_args()
    inst = Instance.load(args.instance)
    errs = validate_instance(inst)
    print(f"{inst.name}: N={inst.N} K={inst.K} Q={inst.Q} rho={inst.load_ratio:.3f} dmax/Q={inst.dmax_ratio:.3f}")
    print("certificate:", inst.certificate, "->", "VALID" if not errs else errs)
    if args.routes:
        cost, errs = validate_solution(inst, json.loads(args.routes))
        print("solution:", "VALID, cost = %.4f" % cost if not errs else errs)
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
