"""Hardware envelope used by the classical code (Section 3.1 of the brief).

Numbers mirror pulser's AnalogDevice.  The quantum layer (pulses.py) validates
against pulser's own device object; these constants are for the embedder, the
calibration helpers and the reports, which must not depend on pulser being installed.
"""
C6 = 8.657e5            # rad/us * um^6   (C6 / hbar)
MIN_SEP = 5.0           # um, minimum atom separation
MAX_RADIUS = 38.0       # um, maximum distance of any atom from the register centre
MAX_ATOMS = 80
OMEGA_MAX = 12.57       # rad/us  (2 pi x 2 MHz)
DELTA_MAX = 125.66      # rad/us  (2 pi x 20 MHz), |delta| <= DELTA_MAX
T_MAX_NS = 6000         # ns, maximum sequence duration


def coupling(r):
    """Interaction energy C6/r^6 (rad/us) at separation r (um)."""
    return C6 / r ** 6


def separation(U):
    """Separation r (um) at which the coupling equals U (rad/us)."""
    return (C6 / U) ** (1.0 / 6.0)


def blockade_radius(omega):
    """R_b = (C6 / Omega)^(1/6): ~6.4 um at OMEGA_MAX."""
    return separation(omega)
