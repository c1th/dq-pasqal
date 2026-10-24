#!/usr/bin/env python3
"""Does the installed pulser expose a detuning map modulator (DMM), on which device, and with
which API?  Paste the full output back; it decides whether weighted (MWIS) packing can be emulated.

    python scripts/check_dmm.py
"""
import inspect

import pulser
from pulser import Register, Sequence

print("pulser", pulser.__version__)
from pulser.devices import AnalogDevice, DigitalAnalogDevice, MockDevice

for dev in (AnalogDevice, DigitalAnalogDevice, MockDevice):
    print(f"\n== {dev.name}")
    dmm = getattr(dev, "dmm_channels", None)
    print("  dmm_channels:", dmm)
    if dmm:
        for name, ch in dmm.items():
            print(f"   {name}: bottom_detuning={getattr(ch, 'bottom_detuning', None)} "
                  f"total_bottom_detuning={getattr(ch, 'total_bottom_detuning', None)} "
                  f"clock_period={getattr(ch, 'clock_period', None)} min_duration={getattr(ch, 'min_duration', None)}")
    print("  requires_layout:", getattr(dev, "requires_layout", None),
          " max_atom_num:", getattr(dev, "max_atom_num", None),
          " rydberg_global max_amp:", dev.channels["rydberg_global"].max_amp if "rydberg_global" in dev.channels else None)

print("\n== API")
print("  Register.define_detuning_map:", inspect.signature(Register.define_detuning_map) if hasattr(Register, "define_detuning_map") else "MISSING")
for m in ("config_detuning_map", "add_dmm_detuning", "config_slm_mask"):
    print(f"  Sequence.{m}:", inspect.signature(getattr(Sequence, m)) if hasattr(Sequence, m) else "MISSING")
try:
    from pulser.register.weight_maps import DetuningMap
    print("  DetuningMap:", inspect.signature(DetuningMap))
    print("  DetuningMap doc:", (DetuningMap.__doc__ or "").strip().splitlines()[0])
except Exception as e:
    print("  DetuningMap import failed:", e)

# a 3-atom trial on whichever device has a DMM
dev = next((d for d in (AnalogDevice, DigitalAnalogDevice, MockDevice) if getattr(d, "dmm_channels", None)), None)
if dev is None:
    print("\nNo device exposes a DMM in this pulser version.")
else:
    print(f"\n== trial sequence on {dev.name}")
    from pulser import Pulse
    from pulser.waveforms import InterpolatedWaveform, RampWaveform
    reg = Register({"q0": (-6.0, 0.0), "q1": (0.0, 0.0), "q2": (6.0, 0.0)})
    if getattr(dev, "requires_layout", False):
        reg = reg.with_automatic_layout(dev)
    try:
        dmap = reg.define_detuning_map({"q0": 1.0, "q1": 0.5, "q2": 0.0})
        seq = Sequence(reg, dev)
        seq.declare_channel("ryd", "rydberg_global")
        seq.config_detuning_map(dmap, "dmm_0")
        T = 2000
        seq.add(Pulse(InterpolatedWaveform(T, [1e-9, 4.0, 1e-9]), InterpolatedWaveform(T, [-7.0, 0.0, 4.7]), 0.0), "ryd")
        seq.add_dmm_detuning(RampWaveform(T, 0.0, -2.0), "dmm_0")
        print("  built OK; duration", seq.get_duration())
        try:
            from pulser_simulation import QutipEmulator
            res = QutipEmulator.from_sequence(seq, sampling_rate=0.1).run()
            print("  QuTiP counts:", dict(res.sample_final_state(200)))
        except Exception as e:
            print("  emulation failed:", repr(e))
    except Exception as e:
        print("  building failed:", repr(e))
