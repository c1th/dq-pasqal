"""Samplers: anything that turns (target, embedding, delta_final) into bitstring counts.

All samplers share the signature
    sampler(target, embedding, delta_final, n_shots, rng, **kw) -> dict[str, int]
with bitstrings in atom order and '1' = Rydberg / excited.

Classical samplers live here (no pulser needed):
    random_sampler   uniform random bitstrings           -> the ablation the brief asks for
    oracle_sampler   exact ground state(s) of the REALISED register cost
                     (-delta |x| + sum C6/r^6 x x)       -> what a perfect adiabatic sweep
                     would return; isolates embedding error from pulse error
The emulator samplers (pulser-simulation / emu-sv / emu-mps) are in pulses.py.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .embedding import Embedding, classical_ground_states
from .models import CouplingTarget


def random_sampler(target: CouplingTarget, emb: Embedding, delta: float, n_shots: int,
                   rng: np.random.Generator, **kw) -> dict[str, int]:
    n = target.n_atoms
    bits = rng.integers(0, 2, size=(n_shots, n))
    return dict(Counter("".join(map(str, row)) for row in bits))


def oracle_sampler(target: CouplingTarget, emb: Embedding, delta: float, n_shots: int,
                   rng: np.random.Generator, **kw) -> dict[str, int]:
    gs = classical_ground_states(emb.U, delta)
    counts: dict[str, int] = {}
    for i, b in enumerate(gs):
        counts[b] = n_shots // len(gs) + (1 if i < n_shots % len(gs) else 0)
    return counts


def boltzmann_sampler(target: CouplingTarget, emb: Embedding, delta: float, n_shots: int,
                      rng: np.random.Generator, temperature: float = 1.0, **kw) -> dict[str, int]:
    """Thermal samples of the realised cost (a classical stand-in for an imperfect sweep)."""
    from .embedding import diagonal_energies
    E, bits = diagonal_energies(emb.U, delta)
    w = np.exp(-(E - E.min()) / temperature)
    p = w / w.sum()
    idx = rng.choice(len(E), size=n_shots, p=p)
    return dict(Counter("".join(str(int(v)) for v in bits[i]) for i in idx))


def ideal_mis_sampler(target: CouplingTarget, emb: Embedding, delta: float, n_shots: int,
                      rng: np.random.Generator, **kw) -> dict[str, int]:
    """All maximum independent sets of the TARGET conflict graph (ignores the embedding).

    Separates the two error sources: is conflict-MIS a good clustering model at all
    (ideal) vs does the register realise it (oracle) vs does the sweep find it (emulator).
    """
    n = target.n_atoms
    if n > 22:
        raise ValueError("too many atoms for exact MIS enumeration")
    ints = np.arange(1 << n, dtype=np.int64)
    bits = ((ints[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(np.int8)
    iu = np.triu_indices(n, 1)
    edge = target.J[iu] > 0
    conflict = (bits[:, iu[0][edge]] & bits[:, iu[1][edge]]).any(axis=1) if edge.any() else np.zeros(len(ints), bool)
    size = bits.sum(axis=1)
    size = np.where(conflict, -1, size)
    best = size.max()
    idx = np.flatnonzero(size == best)
    counts: dict[str, int] = {}
    for i, j in enumerate(idx):
        counts["".join(str(int(v)) for v in bits[j])] = n_shots // len(idx) + (1 if i < n_shots % len(idx) else 0)
    return counts


SAMPLERS = {
    "random": random_sampler,
    "oracle": oracle_sampler,
    "boltzmann": boltzmann_sampler,
    "ideal-mis": ideal_mis_sampler,
}
