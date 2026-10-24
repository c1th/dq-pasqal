#!/usr/bin/env python3
"""Smoke test for the pulser layer: prints the installed version and AnalogDevice specs,
builds a 3-atom sequence with our baseline pulse, runs it on the QuTiP backend and prints
the sampled bitstrings.  Run this before run_pipeline.py --samplers qutip.

    python scripts/check_pulser.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qcvrp.device import C6, MAX_RADIUS, MIN_SEP, OMEGA_MAX  # noqa: E402


def main():
    import pulser
    from pulser.devices import AnalogDevice

    print("pulser", pulser.__version__)
    dev = AnalogDevice
    print("device:", dev.name)
    for attr in ("max_atom_num", "min_atom_distance", "max_radial_distance", "max_sequence_duration",
                 "requires_layout", "interaction_coeff", "max_layout_filling"):
        print(f"  {attr} = {getattr(dev, attr, 'n/a')}")
    for ch_name, ch in dev.channels.items():
        print(f"  channel {ch_name}: max_amp={getattr(ch, 'max_amp', None)} max_abs_detuning={getattr(ch, 'max_abs_detuning', None)} "
              f"clock_period={getattr(ch, 'clock_period', None)} min_duration={getattr(ch, 'min_duration', None)} "
              f"min_avg_amp={getattr(ch, 'min_avg_amp', None)}")
    assert abs(float(getattr(dev, "interaction_coeff", C6)) - C6) / C6 < 1e-3, "C6 mismatch with qcvrp.device"
    assert float(dev.min_atom_distance) == MIN_SEP and float(dev.max_radial_distance) == MAX_RADIUS

    from qcvrp.pulses import adiabatic_sequence, qutip_sampler
    from qcvrp.embedding import realised_couplings, Embedding
    from qcvrp.models import CouplingTarget

    # a 3-atom path: q0-q1 blockaded (6 um), q1-q2 blockaded (6 um), q0-q2 at 12 um (free)
    pos = np.array([[-6.0, 0.0], [0.0, 0.0], [6.0, 0.0]])
    J = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], float)
    tgt = CouplingTarget(3, J, np.zeros(3), [], ["q0", "q1", "q2"], kind="mis")
    emb = Embedding(positions=pos, lam=18.6, U=realised_couplings(pos), T=18.6 * J, wanted=None, unwanted=None,
                    ingroup=None, fidelity={})
    seq = adiabatic_sequence(pos, delta_final=4.7, omega_max=4.0, duration_ns=4000)
    print("sequence duration (ns):", seq.get_duration())
    counts = qutip_sampler(tgt, emb, 4.7, 200, np.random.default_rng(0), omega_max=4.0, duration_ns=4000)
    print("counts:", dict(sorted(counts.items(), key=lambda kv: -kv[1])))
    print("expected ground state: '101' (independent set of the path), with '010' / '100' / '001' as excitations")


if __name__ == "__main__":
    main()
