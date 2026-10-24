"""Pulse shaping and emulation (Task A4).  Requires pulser (+ pulser-simulation / QuTiP).

Baseline (Section 8.2 of the brief): three-point interpolated amplitude ramp
0 -> Omega_max -> 0 and a linear detuning sweep from -delta_initial to +delta_final,
over `duration_ns` (<= 6000 ns on AnalogDevice).  delta_final comes from the embedding's
design rule (embedding.suggest_delta); Omega_max is chosen so that the blockade radius
at full power, (C6/Omega)^(1/6), sits between the conflict distance (~6 um) and the
non-conflict floor (~9.5 um): Omega_max = 4 rad/us gives R_b ~ 7.7 um.

Everything here is validated by pulser's own AnalogDevice object when the Sequence is
built, so a register or pulse outside the hardware envelope raises before any emulation.

NOTE: this module was written against pulser's documented API (Register, Sequence,
Pulse, InterpolatedWaveform, QutipEmulator.from_sequence / sample_final_state).  It is
exercised on the team's machines, not in the sandbox where the rest of the package was
tested, so the first run may need a one-line fix if the installed version differs.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .device import OMEGA_MAX, T_MAX_NS
from .embedding import Embedding
from .models import CouplingTarget


def _require_pulser():
    try:
        import pulser  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install pulser  (pulser-core + pulser-simulation)") from e


def build_register(positions: np.ndarray, device=None):
    """Register from positions (um). If the device requires a trap layout (newer pulser
    AnalogDevice versions do), one is generated automatically around these positions."""
    _require_pulser()
    from pulser import Register
    reg = Register({f"q{i}": (float(x), float(y)) for i, (x, y) in enumerate(positions)})
    if device is not None and getattr(device, "requires_layout", False) and getattr(reg, "layout", None) is None:
        if hasattr(reg, "with_automatic_layout"):
            reg = reg.with_automatic_layout(device)
        else:  # pragma: no cover - very old pulser
            raise RuntimeError("this device requires a register layout; upgrade pulser (>= 0.20)")
    return reg


DET_SHAPES = {
    # (normalised time, detuning as a fraction of delta_final; negative = fraction of delta_initial)
    "linear": [(0.0, -1.0), (0.5, 0.0), (1.0, 1.0)],
    # spend most of the sweep on the positive side, where the gap closes for MIS registers
    "slow-end": [(0.0, -1.0), (0.3, 0.0), (0.65, 0.6), (1.0, 1.0)],
    # the opposite, as a control
    "slow-start": [(0.0, -1.0), (0.6, 0.0), (1.0, 1.0)],
}


def adiabatic_sequence(positions: np.ndarray, delta_final: float, omega_max: float = 4.0,
                       delta_initial: float | None = None, duration_ns: int = 4000, device=None,
                       amp_shape: str = "interpolated", det_shape: str = "linear"):
    """Return a pulser Sequence implementing the adiabatic sweep.

    amp_shape : "interpolated" (smooth 0 -> omega_max -> 0 through three points)
    det_shape : key of DET_SHAPES; the detuning is an interpolated waveform through those
                points, scaled by delta_initial (negative side) and delta_final (positive side).
    """
    _require_pulser()
    from pulser import Pulse, Sequence
    from pulser.devices import AnalogDevice
    from pulser.waveforms import InterpolatedWaveform

    device = AnalogDevice if device is None else device
    if omega_max > OMEGA_MAX + 1e-9:
        raise ValueError(f"omega_max {omega_max} exceeds device maximum {OMEGA_MAX}")
    duration_ns = int(min(duration_ns, T_MAX_NS))
    duration_ns -= duration_ns % 4  # AnalogDevice clock period is 4 ns
    delta_initial = abs(delta_final) * 1.5 if delta_initial is None else abs(delta_initial)
    reg = build_register(positions, device)
    seq = Sequence(reg, device)
    seq.declare_channel("ryd", "rydberg_global")
    if amp_shape == "interpolated":
        amp = InterpolatedWaveform(duration_ns, [1e-9, omega_max, 1e-9])
    else:
        raise ValueError(amp_shape)
    pts = DET_SHAPES[det_shape]
    times = [t for t, _ in pts]
    values = [(-delta_initial * -v if v < 0 else delta_final * v) for _, v in pts]
    if det_shape == "linear":
        det = InterpolatedWaveform(duration_ns, values)
    else:
        det = InterpolatedWaveform(duration_ns, values, times=times)
    seq.add(Pulse(amp, det, 0.0), "ryd")
    seq.measure("ground-rydberg")
    return seq


def sequence_summary(seq) -> dict:
    """Parameters worth putting in results.csv."""
    try:
        total = seq.get_duration()
    except Exception:  # pragma: no cover
        total = None
    return dict(duration_ns=total, device=getattr(seq.device, "name", str(seq.device)))


def qutip_sampler(target: CouplingTarget, emb: Embedding, delta: float, n_shots: int,
                  rng: np.random.Generator, omega_max: float = 4.0, delta_initial: float | None = None,
                  duration_ns: int = 4000, det_shape: str = "linear", sampling_rate: float = 0.1,
                  **kw) -> dict[str, int]:
    """Exact state-vector emulation with pulser-simulation (QuTiP). Fine up to ~12 atoms."""
    _require_pulser()
    from pulser_simulation import QutipEmulator

    seq = adiabatic_sequence(emb.positions, delta, omega_max=omega_max, delta_initial=delta_initial,
                             duration_ns=duration_ns, det_shape=det_shape)
    sim = QutipEmulator.from_sequence(seq, sampling_rate=sampling_rate)
    results = sim.run()
    counts = results.sample_final_state(N_samples=int(n_shots))
    # pulser returns bitstrings in register (qubit) order with '1' = Rydberg state
    return dict(Counter({str(k): int(v) for k, v in counts.items()}))


def emu_sv_sampler(target: CouplingTarget, emb: Embedding, delta: float, n_shots: int,
                   rng: np.random.Generator, omega_max: float = 4.0, delta_initial: float | None = None,
                   duration_ns: int = 4000, det_shape: str = "linear", **kw) -> dict[str, int]:
    """State-vector emulation with emu-sv (up to ~20 atoms).  API sketch; verify against the
    installed emu-sv version (pasqal-io/emulators) before relying on it."""
    _require_pulser()
    try:
        from emu_sv import SVBackend, SVConfig
        from pulser.backend import BitStrings
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install emu-sv") from e
    seq = adiabatic_sequence(emb.positions, delta, omega_max=omega_max, delta_initial=delta_initial,
                             duration_ns=duration_ns, det_shape=det_shape)
    obs = BitStrings(evaluation_times=[1.0], num_shots=int(n_shots))
    backend = SVBackend(seq, config=SVConfig(observables=[obs]))
    results = backend.run()
    counts = results.get_result(obs, 1.0)
    return dict(Counter({str(k): int(v) for k, v in dict(counts).items()}))


QUANTUM_SAMPLERS = {
    "qutip": qutip_sampler,
    "emu-sv": emu_sv_sampler,
}
