"""Invert ODMR line centres into the bias magnetic-field vector (and zero-field splitting D).

The eight electron-spin lines of an ensemble depend on (B, D) through the four NV
orientations. We fit B in the crystal frame and D by weighted least squares, assigning
measured to predicted lines by sorting (the optimal 1-D assignment) when all eight are
resolved, or by nearest-neighbour matching in both directions when lines overlap.
Multi-start over the fundamental domain of the cubic group avoids local minima, and the
result is reported in canonical form because the ensemble cannot distinguish the 48
symmetry-equivalent fields (see ``physics.canonical_field``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import least_squares

from .physics import D_GS, GAMMA_E, NV_AXES, axial_transverse, electron_lines

_B_BOUND = 280.0  # keeps |B| below the model's 500 G validity limit during optimisation


@dataclass
class FieldFit:
    b_canonical: list[float]  # G, |Bx| >= |By| >= |Bz| >= 0 (crystal frame)
    b_err: list[float]
    b_magnitude: float
    b_magnitude_err: float
    D: float
    D_err: float | None
    angle_to_nearest_axis_deg: float
    axial_transverse_g: list[list[float]]  # per orientation [|B_par|, B_perp], strongest axial first
    centers: list[float]
    errors: list[float]
    predicted_lines: list[float]
    residuals_mhz: list[float]  # measured - nearest predicted
    norm_residuals: list[float]
    chisqr: float
    redchi: float | None
    dof: int
    consistent: bool
    message: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FieldFit":
        return cls(**d)


def _start_directions(divisions: int = 6) -> np.ndarray:
    """Directions covering the spherical triangle [100]-[110]-[111] (the fundamental domain)."""
    v1, v2, v3 = np.array([1.0, 0, 0]), np.array([1.0, 1, 0]) / np.sqrt(2), np.ones(3) / np.sqrt(3)
    dirs = []
    for i in range(divisions + 1):
        for j in range(divisions + 1 - i):
            k = divisions - i - j
            v = i * v1 + j * v2 + k * v3
            dirs.append(v / np.linalg.norm(v))
    return np.array(dirs)


def _signed_permutation(b: np.ndarray) -> np.ndarray:
    """Matrix P with P @ b = canonical(b)."""
    order = np.argsort(-np.abs(b), kind="stable")
    p = np.zeros((3, 3))
    for row, col in enumerate(order):
        p[row, col] = 1.0 if b[col] >= 0 else -1.0
    return p


def fit_field(
    centers,
    errors=None,
    fit_D: bool = True,
    D0: float = D_GS,
    window: tuple[float, float] | None = None,
    consistency_sigma: float = 4.0,
) -> FieldFit:
    """Fit (B, D) to measured line centres (MHz) with 1-sigma ``errors`` (MHz).

    ``window`` is the measured frequency range: predicted lines outside it are not required
    to have a measured partner (only relevant when fewer than eight lines are given).
    """
    c = np.asarray(centers, dtype=float)
    e = np.full(len(c), 0.05) if errors is None else np.asarray(errors, dtype=float)
    if len(c) > 8:
        raise ValueError(f"{len(c)} lines given: one NV ensemble in one field has at most 8 electron-spin lines")
    if len(c) < 3:
        raise ValueError("need at least 3 line centres to constrain a field vector")
    order = np.argsort(c)
    c, e = c[order], np.maximum(e[order], 1e-6)
    m, n_par = len(c), 4 if fit_D else 3

    def predicted(x):
        return np.sort(electron_lines(x[:3], x[3] if fit_D else D0).ravel())

    def measured_residuals(x):
        p = predicted(x)
        if m == 8:
            return (c - p) / e
        nearest = np.argmin(np.abs(c[:, None] - p[None, :]), axis=1)
        return (c - p[nearest]) / e

    def residuals(x):
        r = measured_residuals(x)
        if m == 8:
            return r
        p = predicted(x)
        gap = p[:, None] - c[None, :]
        partner = np.argmin(np.abs(gap), axis=1)
        r_pred = gap[np.arange(8), partner] / e[partner]
        if window is not None:
            r_pred = np.where((p > window[0]) & (p < window[1]), r_pred, 0.0)
        return np.concatenate([r, r_pred])

    span = c[-1] - c[0]
    magnitudes = np.array([1.0, 1.35, 1.7]) * max(span, 1.0) / (2 * GAMMA_E)
    lower = [-_B_BOUND] * 3 + ([D0 - 30.0] if fit_D else [])
    upper = [_B_BOUND] * 3 + ([D0 + 30.0] if fit_D else [])
    best = None
    for direction in _start_directions():
        for mag in magnitudes:
            x0 = np.concatenate([direction * min(mag, 0.9 * _B_BOUND), [D0] if fit_D else []])
            sol = least_squares(residuals, x0, bounds=(lower, upper), x_scale=[10, 10, 10, 1][:n_par])
            if best is None or sol.cost < best.cost:
                best = sol
    x = best.x

    # Covariance from the measured-line residuals only (the reverse matches double count).
    jac = np.empty((m, n_par))
    for i in range(n_par):
        h = 1e-4 * max(1.0, abs(x[i]))
        dx = np.zeros(n_par)
        dx[i] = h
        jac[:, i] = (measured_residuals(x + dx) - measured_residuals(x - dx)) / (2 * h)
    jac = -jac  # d(model - data)/dx
    r = measured_residuals(x)
    chisqr = float(r @ r)
    dof = m - n_par
    redchi = chisqr / dof if dof > 0 else None
    cov = np.linalg.pinv(jac.T @ jac) * (max(1.0, redchi) if redchi is not None else 1.0)

    b = x[:3]
    perm = _signed_permutation(b)
    b_canonical = perm @ b
    b_cov = perm @ cov[:3, :3] @ perm.T
    magnitude = float(np.linalg.norm(b))
    grad = b / max(magnitude, 1e-12)
    cos_nearest = float(np.max(np.abs(NV_AXES @ b)) / max(magnitude, 1e-12))
    at = axial_transverse(b)
    at = at[np.argsort(-at[:, 0])]
    p = predicted(x)
    nearest = np.argmin(np.abs(c[:, None] - p[None, :]), axis=1)
    resid_mhz = c - p[nearest]
    norm = resid_mhz / e
    consistent = bool(np.max(np.abs(norm)) <= consistency_sigma)
    worst = int(np.argmax(np.abs(norm)))
    message = (
        f"all {m} lines explained by one field vector (max |residual| = {abs(norm[worst]):.1f} sigma)"
        if consistent
        else f"line at {c[worst]:.3f} MHz is {norm[worst]:+.1f} sigma from the best single-field prediction"
    )
    return FieldFit(
        b_canonical=[float(v) for v in b_canonical],
        b_err=[float(v) for v in np.sqrt(np.clip(np.diag(b_cov), 0.0, None))],
        b_magnitude=magnitude,
        b_magnitude_err=float(np.sqrt(max(grad @ cov[:3, :3] @ grad, 0.0))),
        D=float(x[3]) if fit_D else float(D0),
        D_err=float(np.sqrt(max(cov[3, 3], 0.0))) if fit_D else None,
        angle_to_nearest_axis_deg=float(np.degrees(np.arccos(min(cos_nearest, 1.0)))),
        axial_transverse_g=[[float(a), float(t)] for a, t in at],
        centers=[float(v) for v in c],
        errors=[float(v) for v in e],
        predicted_lines=[float(v) for v in p],
        residuals_mhz=[float(v) for v in resid_mhz],
        norm_residuals=[float(v) for v in norm],
        chisqr=chisqr,
        redchi=redchi,
        dof=dof,
        consistent=consistent,
        message=message,
    )
