"""Invert ODMR line centres into the bias magnetic-field vector (and zero-field splitting D).

The eight electron-spin lines of an ensemble depend on (B, D) through the four NV
orientations. We fit B in the crystal frame and D by weighted least squares. Measured lines
are matched to predicted lines in frequency order; when fewer than eight lines are given,
the fit chooses which predicted lines stay unmatched (all order-preserving choices are
enumerated). An unmatched line costs nothing if it lies outside the measured coverage or
coincides with a matched line (degenerate orientations), and a fixed chi-square penalty
otherwise, because a line inside measured data should have been seen. Multi-start over the
fundamental domain of the cubic group avoids local minima, and the result is reported in
canonical form because the ensemble cannot distinguish the 48 symmetry-equivalent fields
(see ``physics.canonical_field``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy.optimize import least_squares

from .physics import D_GS, GAMMA_E, NV_AXES, axial_transverse, electron_lines

_B_BOUND = 280.0  # keeps |B| below the model's 500 G validity limit during optimisation
MISSING_LINE_PENALTY = 25.0  # chi^2 cost of a covered, unmatched predicted line (a ~5 sigma miss)
COINCIDENT_MHZ = 1.0  # predicted lines closer than this cannot be resolved as separate lines


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
    residuals_mhz: list[float]  # measured - matched prediction
    norm_residuals: list[float]
    unmatched_predicted: list[float]  # predicted lines with no measured partner (not coincident with one)
    unmatched_in_coverage: list[float]  # ... of which inside the measured coverage
    coverage: list[list[float]]
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


def coverage_intervals(freqs, max_gap: float = 2.5, pad: float = 0.5) -> list[tuple[float, float]]:
    """Frequency intervals actually covered by data: gaps wider than ``max_gap`` MHz split them."""
    f = np.sort(np.asarray(freqs, dtype=float))
    breaks = np.nonzero(np.diff(f) > max_gap)[0]
    starts, ends = np.concatenate([[0], breaks + 1]), np.concatenate([breaks, [len(f) - 1]])
    return [(float(f[s] - pad), float(f[e] + pad)) for s, e in zip(starts, ends)]


@lru_cache(maxsize=None)
def _subsets(n: int, m: int) -> np.ndarray:
    return np.array(list(combinations(range(n), m)), dtype=int)


def _assign(p: np.ndarray, c: np.ndarray, e: np.ndarray, covered: np.ndarray):
    """Best order-preserving match of the m measured lines to m of the n predicted lines.

    Returns (indices of the matched predicted lines, normalised residuals, mask of
    penalised unmatched predictions, mask of all unmatched non-coincident predictions).
    """
    n, m = len(p), len(c)
    subsets = _subsets(n, m)
    resid = (c[None, :] - p[subsets]) / e[None, :]
    cost = np.sum(resid**2, axis=1)
    matched = np.zeros((len(subsets), n), dtype=bool)
    np.put_along_axis(matched, subsets, True, axis=1)
    gap = np.abs(p[:, None] - p[None, :])
    nearest_matched = np.where(matched[:, :, None], gap[None, :, :], np.inf).min(axis=1)
    lonely = ~matched & (nearest_matched > COINCIDENT_MHZ)
    penalised = lonely & covered[None, :]
    cost = cost + MISSING_LINE_PENALTY * penalised.sum(axis=1)
    k = int(np.argmin(cost))
    return subsets[k], resid[k], penalised[k], lonely[k]


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
    coverage=None,
    consistency_sigma: float = 4.0,
) -> FieldFit:
    """Fit (B, D) to measured line centres (MHz) with 1-sigma ``errors`` (MHz).

    ``coverage``: intervals (MHz) where data were taken, e.g. ``coverage_intervals(freqs)``.
    By default the span of the given lines plus 10 MHz on either side counts as covered.
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
    spans = np.array(coverage if coverage is not None else [(c[0] - 10.0, c[-1] + 10.0)], dtype=float)

    def predicted(x):
        return np.sort(electron_lines(x[:3], x[3] if fit_D else D0).ravel())

    def covered(p):
        return ((p[:, None] >= spans[None, :, 0]) & (p[:, None] <= spans[None, :, 1])).any(axis=1)

    def residuals(x):
        p = predicted(x)
        _, r, penalised, _ = _assign(p, c, e, covered(p))
        return np.concatenate([r, np.sqrt(MISSING_LINE_PENALTY) * penalised])

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
    p = predicted(x)
    subset, r, penalised, lonely = _assign(p, c, e, covered(p))

    # Covariance from the measured-line residuals with the assignment held fixed.
    def fixed_residuals(xx):
        return (c - predicted(xx)[subset]) / e

    jac = np.empty((m, n_par))
    for i in range(n_par):
        h = 1e-4 * max(1.0, abs(x[i]))
        dx = np.zeros(n_par)
        dx[i] = h
        jac[:, i] = -(fixed_residuals(x + dx) - fixed_residuals(x - dx)) / (2 * h)
    chisqr = float(r @ r)
    dof = m - n_par
    redchi = chisqr / dof if dof > 0 else None
    cov = np.linalg.pinv(jac.T @ jac) * (max(1.0, redchi) if redchi is not None else 1.0)

    b = x[:3]
    perm = _signed_permutation(b)
    b_cov = perm @ cov[:3, :3] @ perm.T
    magnitude = float(np.linalg.norm(b))
    grad = b / max(magnitude, 1e-12)
    cos_nearest = float(np.max(np.abs(NV_AXES @ b)) / max(magnitude, 1e-12))
    at = axial_transverse(b)
    at = at[np.argsort(-at[:, 0])]
    resid_mhz = c - p[subset]
    consistent = bool(np.max(np.abs(r)) <= consistency_sigma)
    worst = int(np.argmax(np.abs(r)))
    message = (
        f"all {m} lines explained by one field vector (max |residual| = {abs(r[worst]):.1f} sigma)"
        if consistent
        else f"line at {c[worst]:.3f} MHz is {r[worst]:+.1f} sigma from the best single-field prediction"
    )
    if lonely.any():
        message += "; no fitted line at predicted " + ", ".join(f"{v:.1f}" for v in p[lonely]) + " MHz"
    return FieldFit(
        b_canonical=[float(v) for v in perm @ b],
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
        norm_residuals=[float(v) for v in r],
        unmatched_predicted=[float(v) for v in p[lonely]],
        unmatched_in_coverage=[float(v) for v in p[penalised]],
        coverage=[[float(a), float(b_)] for a, b_ in spans],
        chisqr=chisqr,
        redchi=redchi,
        dof=dof,
        consistent=consistent,
        message=message,
    )
