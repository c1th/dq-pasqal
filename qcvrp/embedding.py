"""Register embedding (Task A3): atom positions whose C6/r^6 couplings approximate the
wanted couplings lambda * J, within the device envelope.

What is being fitted
--------------------
* wanted pairs (J_ab > 0, not in the same one-hot group): couple at  T_ab = lambda * J_ab,
  i.e. sit at distance  r*_ab = (C6 / T_ab)^(1/6).  Fitted in LOG distance, which turns
  the 1/r^6 stiffness into a plain least-squares (this is weighted MDS on log distances).
* unwanted pairs (J_ab = 0, different groups): should couple weakly.  Hinge penalty if
  they come closer than `r_floor` (coupling above `U_floor`).
* one-hot groups: placed as rigid regular K-gons of side `side` (5 um by default, the
  minimum separation), so every in-group pair is strongly blockaded.  K = 4 is a square
  whose diagonal is 7.07 um: the diagonal coupling is 1/8 of the side coupling, the
  known weak point of one-hot-by-blockade at K = 4.
* envelope: min separation and max register radius as quadratic barriers, verified
  afterwards (hard check).

Parameters: group centres (x, y) and rotation angles, or raw atom positions when the
model has no groups.  Optimiser: L-BFGS-B with analytic gradients, several restarts
(classical MDS of the target distances + random).

Fidelity report (Section 9.4, "state your formula")
--------------------------------------------------
    rel_err      = sqrt( sum_wanted (U - T)^2 / sum_wanted T^2 )
    log_rms      = sqrt( mean_wanted  log(U / T)^2 )
    unwanted     = sum_{unwanted pairs} U / sum_wanted T      (leakage ratio)
    ingroup_min  = min over in-group pairs of U               (must be >> delta_final)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .device import C6, MAX_RADIUS, MIN_SEP, coupling, separation
from .models import CouplingTarget


# ---------------------------------------------------------------------------
# K-gon templates for one-hot groups
# ---------------------------------------------------------------------------
def kgon(K: int, side: float) -> np.ndarray:
    """Vertices of a regular K-gon with the given side length, centred at the origin."""
    if K == 1:
        return np.zeros((1, 2))
    if K == 2:
        return np.array([[-side / 2, 0.0], [side / 2, 0.0]])
    R = side / (2 * np.sin(np.pi / K))
    ang = 2 * np.pi * np.arange(K) / K
    return np.stack([R * np.cos(ang), R * np.sin(ang)], axis=1)


def rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def drot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[-s, -c], [c, -s]])


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class Embedding:
    positions: np.ndarray          # n x 2 (um), centred
    lam: float                     # physical coupling per unit of J (rad/us)
    U: np.ndarray                  # realised couplings C6/r^6 (n x n, zero diagonal)
    T: np.ndarray                  # target couplings lambda*J (n x n)
    wanted: np.ndarray             # boolean mask of wanted pairs (upper triangle)
    unwanted: np.ndarray           # boolean mask of unwanted cross-group pairs
    ingroup: np.ndarray            # boolean mask of in-group pairs
    fidelity: dict = field(default_factory=dict)
    loss: float = float("nan")

    def summary(self) -> str:
        f = self.fidelity
        s = (f"{len(self.positions)} atoms, radius {f['radius']:.1f} um, min sep {f['min_sep']:.2f} um | "
             f"rel_err {f['rel_err']:.3f}, log_rms {f['log_rms']:.3f}, unwanted/wanted {f['unwanted_ratio']:.3f}")
        if "eff_rel_err" in f:
            s += f", effective(block) rel_err {f['eff_rel_err']:.3f}, in-group U >= {f['ingroup_min_U']:.1f} rad/us"
        if "n_edges" in f:
            s += f" | graph@delta={f['delta_used']:.2f}: {f['n_edges']} edges, {f['violable_edges']} violable, {f['spurious_edges']} spurious"
        return s


def pair_masks(target: CouplingTarget):
    n = target.n_atoms
    g = target.group_of()
    iu = np.triu_indices(n, 1)
    same_group = (g[iu[0]] == g[iu[1]]) & (g[iu[0]] >= 0)
    wanted = (target.J[iu] > 0) & ~same_group
    unwanted = (target.J[iu] <= 0) & ~same_group
    return iu, wanted, unwanted, same_group


def enforce_envelope(pos: np.ndarray, margin: float = 2e-3, max_iter: int = 200) -> np.ndarray:
    """Hard projection into the device envelope after the soft-barrier optimisation:
    pairs closer than MIN_SEP (+margin) are pushed apart symmetrically along their
    connecting line; atoms beyond MAX_RADIUS are shrunk towards the centre.  pulser's
    AnalogDevice validates both to 1e-6 um, so the margin keeps us clear of its check.
    Coordinates are rounded to 1e-4 um for clean JSON / register definitions."""
    pos = np.array(pos, dtype=float)
    n = len(pos)
    target = MIN_SEP + margin
    for _ in range(max_iter):
        pos = np.round(pos - pos.mean(axis=0), 4)
        moved = False
        rmax = np.linalg.norm(pos, axis=1).max() if n else 0.0
        if rmax > MAX_RADIUS - margin:
            pos *= (MAX_RADIUS - margin) / rmax
            moved = True
        for a in range(n):
            for b in range(a + 1, n):
                d = pos[a] - pos[b]
                r = np.linalg.norm(d)
                if r < target:
                    if r < 1e-9:
                        d = np.array([1.0, 0.0])
                        r = 1.0
                    push = 0.5 * (target - r + 1e-3) * d / r
                    pos[a] += push
                    pos[b] -= push
                    moved = True
        if not moved:
            break
    return pos


def realised_couplings(positions: np.ndarray) -> np.ndarray:
    diff = positions[:, None, :] - positions[None, :, :]
    r = np.linalg.norm(diff, axis=-1)
    with np.errstate(divide="ignore"):
        U = np.where(r > 0, C6 / np.maximum(r, 1e-9) ** 6, 0.0)
    np.fill_diagonal(U, 0.0)
    return U


def fidelity_report(positions: np.ndarray, T: np.ndarray, target: CouplingTarget) -> dict:
    n = len(positions)
    iu, wanted, unwanted, ingroup = pair_masks(target)
    U = realised_coupling_matrix = realised_couplings(positions)
    diff = positions[:, None, :] - positions[None, :, :]
    r = np.linalg.norm(diff, axis=-1)[iu]
    Uw, Tw = U[iu][wanted], T[iu][wanted]
    rep = dict(
        n_atoms=n,
        radius=float(np.linalg.norm(positions, axis=1).max()) if n else 0.0,
        min_sep=float(r.min()) if len(r) else float("inf"),
        rel_err=float(np.sqrt(((Uw - Tw) ** 2).sum() / (Tw ** 2).sum())) if wanted.any() else 0.0,
        log_rms=float(np.sqrt(np.mean(np.log(Uw / Tw) ** 2))) if wanted.any() else 0.0,
        unwanted_ratio=float(U[iu][unwanted].sum() / Tw.sum()) if wanted.any() else float("nan"),
        unwanted_max_U=float(U[iu][unwanted].max()) if unwanted.any() else 0.0,
        ingroup_min_U=float(U[iu][ingroup].min()) if ingroup.any() else float("nan"),
        ingroup_max_dist=float(r[ingroup].max()) if ingroup.any() else float("nan"),
        wanted_max_T=float(Tw.max()) if wanted.any() else 0.0,
        wanted_min_T=float(Tw.min()) if wanted.any() else 0.0,
        envelope_ok=bool(r.min() >= MIN_SEP - 1e-6 and np.linalg.norm(positions, axis=1).max() <= MAX_RADIUS + 1e-6),
    )
    if target.groups:
        # effective (manifold) fidelity: centred K x K blocks, which is what an assignment sees
        num = den = 0.0
        for ga, gb in _block_structure(target):
            B, Tb = U[np.ix_(ga, gb)], T[np.ix_(ga, gb)]
            num += np.sum(((B - B.mean()) - (Tb - Tb.mean())) ** 2)
            den += np.sum((Tb - Tb.mean()) ** 2)
        rep["eff_rel_err"] = float(np.sqrt(num / den)) if den > 0 else float("nan")
    return rep


# ---------------------------------------------------------------------------
# The optimiser
# ---------------------------------------------------------------------------
def _positions_from_params(params, target: CouplingTarget, side: float):
    """Return atom positions (n x 2) and, for grouped targets, the K-gon vertex offsets."""
    n = target.n_atoms
    if not target.groups:
        return params.reshape(n, 2), None
    G = len(target.groups)
    centres = params[: 2 * G].reshape(G, 2)
    thetas = params[2 * G:]
    pos = np.zeros((n, 2))
    offs = np.zeros((n, 2))
    for gi, atoms in enumerate(target.groups):
        v = kgon(len(atoms), side)
        R = rot(thetas[gi])
        for a, va in zip(atoms, v):
            offs[a] = va
            pos[a] = centres[gi] + R @ va
    return pos, offs


def _block_structure(target: CouplingTarget):
    """For grouped targets: list of (atoms_g, atoms_g') for every pair of groups."""
    G = len(target.groups)
    return [(np.array(target.groups[g]), np.array(target.groups[h])) for g in range(G) for h in range(g + 1, G)]


def _loss_and_grad(params, target: CouplingTarget, r_star: np.ndarray, r_floor: float, side: float,
                   w_unwanted: float, w_barrier: float, T: np.ndarray | None = None, scale: float = 1.0):
    n = target.n_atoms
    pos, offs = _positions_from_params(params, target, side)
    iu, wanted, unwanted, ingroup = pair_masks(target)
    a_idx, b_idx = iu
    diff = pos[a_idx] - pos[b_idx]
    r = np.linalg.norm(diff, axis=1) + 1e-12
    if target.groups:
        # Block loss on the one-hot manifold: for customers (g, g') exactly one entry of the
        # K x K coupling block is "on", so the block mean is irrelevant; fit the centred
        # block to the centred target block, in raw coupling units (scaled by `scale`).
        U = np.zeros((n, n))
        Uij = C6 / r ** 6
        U[a_idx, b_idx] = Uij
        U[b_idx, a_idx] = Uij
        loss = 0.0
        gU = np.zeros((n, n))
        for ga, gb in _block_structure(target):
            B = U[np.ix_(ga, gb)]
            Tb = T[np.ix_(ga, gb)]
            res = (B - B.mean()) - (Tb - Tb.mean())
            loss += np.sum(res ** 2) / scale ** 2
            gU[np.ix_(ga, gb)] += 2.0 * res / scale ** 2
        # chain: dU/dr = -6 U / r
        dL_dr = gU[a_idx, b_idx] * (-6.0 * Uij / r)
    else:
        logr = np.log(r)
        # wanted: log-distance least squares, scaled so a factor-2 coupling error ~ 0.67^2
        resid = np.where(wanted, logr - np.log(r_star), 0.0)
        loss = 36.0 * np.sum(resid ** 2)
        dL_dr = 36.0 * 2.0 * resid / r
        # unwanted: hinge if closer than r_floor
        viol = np.where(unwanted, np.maximum(0.0, r_floor - r), 0.0)
        loss += w_unwanted * np.sum(viol ** 2)
        dL_dr += -2.0 * w_unwanted * viol
    # min separation barrier (all pairs, incl. in-group which sit at `side` >= MIN_SEP)
    sep = np.maximum(0.0, MIN_SEP - r)
    loss += w_barrier * np.sum(sep ** 2)
    dL_dr += -2.0 * w_barrier * sep
    # radius barrier
    rho = np.linalg.norm(pos, axis=1) + 1e-12
    rv = np.maximum(0.0, rho - MAX_RADIUS)
    loss += w_barrier * np.sum(rv ** 2)
    # gradients wrt positions
    gpos = np.zeros((n, 2))
    gpair = (dL_dr / r)[:, None] * diff
    np.add.at(gpos, a_idx, gpair)
    np.add.at(gpos, b_idx, -gpair)
    gpos += (2.0 * w_barrier * rv / rho)[:, None] * pos
    if not target.groups:
        return loss, gpos.ravel()
    G = len(target.groups)
    grad = np.zeros_like(params)
    thetas = params[2 * G:]
    for gi, atoms in enumerate(target.groups):
        grad[2 * gi: 2 * gi + 2] = gpos[atoms].sum(axis=0)
        dR = drot(thetas[gi])
        grad[2 * G + gi] = sum(gpos[a] @ (dR @ offs[a]) for a in atoms)
    return loss, grad


def effective_center_distance(contrast: float, side: float) -> float:
    """Centre distance D of two parallel dimers (side `side`) whose same-vehicle minus
    cross-vehicle coupling,  U(D) - U(sqrt(D^2 + side^2)),  equals `contrast`.
    Monotone in D, solved by bisection on [side, 60]. Returns 60 if unreachable."""
    f = lambda D: coupling(D) - coupling(np.sqrt(D * D + side * side)) - contrast
    lo, hi = side, 60.0
    if f(lo) < 0:
        return lo
    if f(hi) > 0:
        return hi
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _mds_init(target: CouplingTarget, r_star_full: np.ndarray, rng: np.random.Generator,
              T: np.ndarray | None = None, side: float = MIN_SEP) -> np.ndarray:
    """Classical MDS on the target distance matrix (group level when grouped)."""
    if target.groups:
        G = len(target.groups)
        D = np.zeros((G, G))
        for gi in range(G):
            for gj in range(gi + 1, G):
                block = T[np.ix_(target.groups[gi], target.groups[gj])]
                contrast = float(block.max() - block.mean()) if block.size else 0.0
                D[gi, gj] = D[gj, gi] = effective_center_distance(contrast, side) if contrast > 0 else np.nan
        far = np.nanmax(D) * 1.5 if np.isfinite(D).any() else 15.0
        D = np.where(np.isnan(D), far, D)
        n = G
    else:
        D = r_star_full.copy()
        far = np.nanmax(D[np.isfinite(D)]) * 1.5 if np.isfinite(D).any() else 15.0
        D = np.where(np.isfinite(D), D, far)
        np.fill_diagonal(D, 0.0)
        n = len(D)
    if n == 1:
        X = np.zeros((1, 2))
    else:
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ (D ** 2) @ J
        w, v = np.linalg.eigh(B)
        idx = np.argsort(w)[::-1][:2]
        X = v[:, idx] * np.sqrt(np.maximum(w[idx], 1e-9))
        if X.shape[1] < 2:
            X = np.hstack([X, np.zeros((n, 2 - X.shape[1]))])
    X = X + rng.normal(scale=0.5, size=X.shape)
    if target.groups:
        return np.concatenate([X.ravel(), rng.uniform(0, 2 * np.pi, size=len(target.groups))])
    return X.ravel()


def embed(target: CouplingTarget, J_max_phys: float = 4.0, side: float = MIN_SEP,
          U_floor: float | None = None, n_restarts: int = 8, seed: int = 0,
          w_unwanted: float = 10.0, w_barrier: float = 1e3, maxiter: int = 2000) -> Embedding:
    """Fit atom positions to the wanted couplings.

    J_max_phys : physical coupling (rad/us) given to the strongest wanted pair; sets lambda.
    U_floor    : unwanted pairs are pushed below this coupling (default: 0.1 * J_max_phys).
    """
    rng = np.random.default_rng(seed)
    n = target.n_atoms
    Jmax = target.J.max()
    lam = J_max_phys / Jmax if Jmax > 0 else 1.0
    T = lam * target.J
    U_floor = 0.1 * J_max_phys if U_floor is None else U_floor
    r_floor = separation(U_floor)
    iu, wanted, unwanted, ingroup = pair_masks(target)
    with np.errstate(divide="ignore"):
        r_star_full = np.where(T > 0, separation(np.maximum(T, 1e-30)), np.inf)
    r_star = np.where(wanted, r_star_full[iu], 1.0)
    best = None
    scale = max(J_max_phys, 1e-9)
    for rs in range(n_restarts):
        if rs == 0 and target.init_positions is not None and not target.groups:
            # geometric start: the model's own map, scaled so conflicting pairs sit at r*
            X = np.asarray(target.init_positions, float)
            X = X - X.mean(axis=0)
            dpair = np.linalg.norm(X[iu[0]] - X[iu[1]], axis=1)
            if wanted.any() and np.median(dpair[wanted]) > 0:
                s_ = np.median(r_star[wanted]) / np.median(dpair[wanted])
            else:
                s_ = 1.0
            X = X * s_
            rmax = np.linalg.norm(X, axis=1).max()
            if rmax > MAX_RADIUS:
                X *= MAX_RADIUS / rmax
            x0 = X.ravel() + rng.normal(scale=0.1, size=X.size)
        elif rs <= 1:
            x0 = _mds_init(target, r_star_full, rng, T=T, side=side)
        else:
            if target.groups:
                G = len(target.groups)
                x0 = np.concatenate([rng.uniform(-MAX_RADIUS * 0.6, MAX_RADIUS * 0.6, size=2 * G),
                                     rng.uniform(0, 2 * np.pi, size=G)])
            else:
                x0 = rng.uniform(-MAX_RADIUS * 0.6, MAX_RADIUS * 0.6, size=2 * n)
        res = minimize(_loss_and_grad, x0, args=(target, r_star, r_floor, side, w_unwanted, w_barrier, T, scale),
                       jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter))
        if best is None or res.fun < best.fun:
            best = res
    pos, _ = _positions_from_params(best.x, target, side)
    pos = enforce_envelope(pos - pos.mean(axis=0))
    U = realised_couplings(pos)
    fid = fidelity_report(pos, T, target)
    fid["U_floor"] = float(U_floor)
    fid["r_floor"] = float(r_floor)
    fid["J_design"] = float(J_max_phys)
    emb = Embedding(positions=pos, lam=lam, U=U, T=T, wanted=wanted, unwanted=unwanted,
                    ingroup=ingroup, fidelity=fid, loss=float(best.fun))
    if target.kind == "mis":
        emb.fidelity.update(graph_fidelity(emb, target, suggest_delta(emb, target)["delta"]))
    return emb


def graph_fidelity(emb: Embedding, target: CouplingTarget, delta: float) -> dict:
    """Graph-level fidelity of an MIS register at final detuning `delta`: a pair is
    effectively blockaded iff its realised coupling exceeds delta.  Reports target edges
    realised below delta (violable) and non-edges realised above delta (spurious)."""
    n = target.n_atoms
    iu = np.triu_indices(n, 1)
    edge = target.J[iu] > 0
    Uf = emb.U[iu]
    return dict(
        n_edges=int(edge.sum()),
        violable_edges=int((edge & (Uf <= delta)).sum()),
        spurious_edges=int((~edge & (Uf > delta)).sum()),
        delta_used=float(delta),
    )


# ---------------------------------------------------------------------------
# Final detuning
# ---------------------------------------------------------------------------
def suggest_delta(emb: Embedding, target: CouplingTarget) -> dict:
    """Final detuning for the realised register.

    One-hot groups:  exciting one atom per group must beat zero (delta > total coupling
    the atom feels from other groups) and two per group must lose (delta < in-group
    coupling).  We take the geometric mean of the two bounds and report both.
    No groups (bisection):  delta = (1/2) * mean_i sum_j U_ij, the uniform detuning that
    best matches the wanted linear term -sum_j w_ij (see models.BisectionModel).
    """
    U = emb.U
    n = len(U)
    if target.kind == "mis":
        # Design rule for MIS: delta sits at the geometric mean of the design edge coupling
        # (lambda, the coupling given to a conflict pair at its target distance) and the
        # unwanted-coupling floor; the realised graph is then reported at that delta via
        # graph_fidelity (edges below delta are violable, non-edges above it are spurious).
        iu = np.triu_indices(n, 1)
        edge = target.J[iu] > 0
        J_edge = float(emb.lam * target.J.max()) if target.J.max() > 0 else float(emb.fidelity.get("J_design", 18.6))
        U_floor = float(emb.fidelity.get("U_floor", 0.1 * J_edge))
        delta = float(np.sqrt(J_edge * U_floor))
        wanted_min = float(U[iu][edge].min()) if edge.any() else float("inf")
        unwanted_max = float(U[iu][~edge].max()) if (~edge).any() else 0.0
        return dict(delta=delta, lower=U_floor, upper=J_edge, onehot_enforceable=wanted_min > delta > unwanted_max,
                    wanted_min=wanted_min, unwanted_max=unwanted_max)
    if target.groups:
        g = target.group_of()
        cross = U * (g[:, None] != g[None, :])
        lower = float(cross.sum(axis=1).max())
        upper = float(emb.fidelity["ingroup_min_U"])
        delta = float(np.sqrt(max(lower, 1e-6) * upper))
        ok = lower < upper
        return dict(delta=delta, lower=lower, upper=upper, onehot_enforceable=ok)
    rows = U.sum(axis=1)
    return dict(delta=float(0.5 * rows.mean()), lower=float(0.5 * rows.min()), upper=float(0.5 * rows.max()),
                onehot_enforceable=True)


# ---------------------------------------------------------------------------
# Exact classical diagonal energies on small registers
# ---------------------------------------------------------------------------
def diagonal_energies(U: np.ndarray, delta: float, max_atoms: int = 22) -> tuple[np.ndarray, np.ndarray]:
    """E(x) = -delta * |x| + sum_{a<b} U_ab x_a x_b for every bitstring (as integers).

    Returns (energies, bit matrix) with bit order: atom a is bit (n-1-a) so that the
    row index written in binary reads like the register bitstring.
    """
    n = len(U)
    if n > max_atoms:
        raise ValueError(f"{n} atoms is too many for exact enumeration")
    ints = np.arange(1 << n, dtype=np.int64)
    bits = ((ints[:, None] >> (n - 1 - np.arange(n))[None, :]) & 1).astype(np.int8)
    E = -delta * bits.sum(axis=1).astype(float)
    iu = np.triu_indices(n, 1)
    # chunked to bound memory on n ~ 20
    for start in range(0, len(ints), 1 << 16):
        b = bits[start: start + (1 << 16)].astype(float)
        E[start: start + (1 << 16)] += ((b[:, iu[0]] * b[:, iu[1]]) * U[iu]).sum(axis=1)
    return E, bits


def classical_ground_states(U: np.ndarray, delta: float, tol: float = 1e-9) -> list[str]:
    """All bitstrings minimising the REALISED cost (-delta|x| + sum U x x)."""
    E, bits = diagonal_energies(U, delta)
    m = E.min()
    idx = np.flatnonzero(E <= m + tol)
    return ["".join(str(int(v)) for v in bits[i]) for i in idx]


def manifold_energies(U: np.ndarray, assignments: np.ndarray, K: int) -> np.ndarray:
    """Realised pair energy of each assignment on the one-hot manifold (atom (i,k) = i*K+k).

    assignments: M x N integer array. Returns length-M array of sum_{i<j} U[(i,a_i),(j,a_j)];
    the -delta*N term is constant on the manifold and omitted.
    """
    A = np.asarray(assignments)
    M, N = A.shape
    E = np.zeros(M)
    for i in range(N):
        for j in range(i + 1, N):
            E += U[i * K + A[:, i], j * K + A[:, j]]
    return E


def manifold_check(emb: Embedding, inst, model, ref, max_assignments: int = 2_000_000) -> dict:
    """Brute-force the realised one-hot-manifold energy over all assignments (S, M tiers).

    Reports the realised minimiser's feasibility and its gap to the model optimum, which
    separates embedding error from everything the pulse will add later.
    """
    from .models import all_assignments, clustering_objective
    from .instances import validate_assignment
    N, K = inst.N, inst.K
    assigns = np.array(list(all_assignments(N, K)))
    if len(assigns) > max_assignments:
        raise ValueError("too many assignments for a brute-force manifold check")
    E = manifold_energies(emb.U, assigns, K)
    order = np.argsort(E)
    best = assigns[order[0]]
    feas = not validate_assignment(inst, best.tolist())
    obj = clustering_objective(best.tolist(), inst)
    # rank of the model optimum among realised energies
    opt_assign = np.zeros(N, dtype=int)
    for k, block in enumerate(ref.model_blocks):
        for c in block:
            opt_assign[c] = k
    from .instances import canonical_clustering
    opt_can = canonical_clustering(opt_assign.tolist(), K)
    rank = next((r for r, idx in enumerate(order) if canonical_clustering(assigns[idx].tolist(), K) == opt_can), None)
    return dict(
        realised_minimiser=best.tolist(),
        realised_minimiser_feasible=feas,
        realised_minimiser_model_gap_pct=100.0 * (obj - ref.model_cost) / ref.model_cost,
        model_optimum_rank_in_realised=rank,
        n_assignments=len(assigns),
        energy_gap_to_first_excited=float(E[order[1]] - E[order[0]]) if len(E) > 1 else float("nan"),
    )
