"""Adaptive ODMR acquisition: a Gaussian-process surrogate chooses the next MW frequencies.

Narrow (~1 MHz) lines cannot be *found* without covering the band at ~1 MHz resolution,
whatever the sampler. The physics offers a way out: at high MW power the lines
power-broaden to several MHz, so a coarse survey locates them cheaply. A low-power GP
refinement then spends points only where the lines are, using an uncertainty-based
acquisition function weighted by the predicted dip depth.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from .fitting import _sorted_rates, estimate_baseline, find_lines, robust_baseline

MeasureFn = Callable[[np.ndarray, float], np.ndarray]  # (frequencies, MW power in dBm) -> counts


class OdmrSurrogate:
    """GP over normalised fluorescence y = rate / baseline - 1 (0 off resonance, < 0 in dips).

    Shot noise enters as known per-point variance (``alpha``), so the kernel only has to
    learn the dip amplitude and the line-shape length scale. A stationary kernel learns its
    amplitude from all points, so in mostly flat data the posterior under-states dip depth;
    the acquisition only relies on where dips are and how uncertain the GP is.
    """

    def __init__(self, length_scale: float = 1.0, length_scale_bounds=(0.3, 3.0), n_restarts: int = 2):
        # Low-power ODMR features are ~0.5-3 MHz wide. Leaving the length scale free lets
        # maximum likelihood over-smooth sparse, noisy dips (it chose 7-20 MHz in tests).
        self.kernel = ConstantKernel(1e-4, (1e-8, 1e-1)) * Matern(length_scale, length_scale_bounds, nu=2.5)
        self.n_restarts = n_restarts
        self.gp: GaussianProcessRegressor | None = None

    def fit(self, freqs, counts, dwell_ms, optimize: bool = True, baseline=None) -> "OdmrSurrogate":
        """``baseline``: optional callable f -> baseline rate (counts/ms), e.g. from a survey."""
        f, rate, sigma = _sorted_rates(freqs, counts, dwell_ms)
        base = baseline(f) if baseline is not None else robust_baseline(f, rate, sigma)[0]
        reuse = not optimize and self.gp is not None
        gp = GaussianProcessRegressor(
            kernel=self.gp.kernel_ if reuse else self.kernel,
            alpha=(sigma / base) ** 2,
            normalize_y=False,
            optimizer=None if reuse else "fmin_l_bfgs_b",
            n_restarts_optimizer=0 if reuse else self.n_restarts,
            random_state=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(f[:, None], rate / base - 1.0)
        self.gp = gp
        return self

    def predict(self, freqs) -> tuple[np.ndarray, np.ndarray]:
        mu, sd = self.gp.predict(np.asarray(freqs, dtype=float)[:, None], return_std=True)
        return mu, sd

    @property
    def length_scale(self) -> float:
        return float(self.gp.kernel_.k2.length_scale)

    def acquisition(self, freqs, beta: float = 1.0) -> np.ndarray:
        """sigma * (predicted dip depth + beta * sigma): uncertainty, weighted towards signal."""
        mu, sd = self.predict(freqs)
        return sd * (np.maximum(-mu, 0.0) + beta * sd)


def suggest_frequencies(
    surrogate: OdmrSurrogate, candidates, n: int, min_spacing: float | None = None, beta: float = 1.0
) -> np.ndarray:
    """Greedy batch of ``n`` acquisition maxima, at least ``min_spacing`` MHz apart."""
    cand = np.asarray(candidates, dtype=float)
    score = surrogate.acquisition(cand, beta).astype(float)
    spacing = max(0.1, 0.3 * surrogate.length_scale) if min_spacing is None else min_spacing
    picks = []
    for _ in range(n):
        k = int(np.argmax(score))
        if not np.isfinite(score[k]):
            break
        picks.append(cand[k])
        score[np.abs(cand - cand[k]) < spacing] = -np.inf
    return np.array(picks)


def windows_around(centers, half_width: float) -> list[tuple[float, float]]:
    """Merged frequency intervals [c - half_width, c + half_width]."""
    merged: list[list[float]] = []
    for c in sorted(centers):
        lo, hi = c - half_width, c + half_width
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(float(a), float(b)) for a, b in merged]


@dataclass
class RefineResult:
    freqs: np.ndarray
    counts: np.ndarray
    windows: list[tuple[float, float]]
    length_scales: list[float] = field(default_factory=list)


def refine_odmr(
    measure: MeasureFn,
    centers,
    n_points: int,
    mw_power_dbm: float,
    dwell_ms: float,
    half_width: float = 8.0,
    batch_size: int = 8,
    seeds_per_window: int = 8,
    beta: float = 4.0,
    refit_every: int = 3,
    step: float = 0.05,
    baseline=None,
    surrogate: OdmrSurrogate | None = None,
    allocation: str = "per_window",
) -> RefineResult:
    """Spend ``n_points`` measurements around ``centers`` using GP-guided batches.

    allocation="per_window" (default) takes one acquisition maximum inside every window per
    batch, so weak lines are not starved by strong ones; "global" takes the ``batch_size``
    best points anywhere. ``baseline`` (callable f -> counts/ms) should come from data with
    off-resonant points, e.g. the survey; without it the baseline is re-estimated jointly
    with the lines each time the GP hyperparameters are re-optimised.
    """
    if allocation not in ("per_window", "global"):
        raise ValueError("allocation must be 'per_window' or 'global'")
    windows = windows_around(centers, half_width)
    seeds = np.concatenate([np.linspace(a, b, seeds_per_window + 2)[1:-1] for a, b in windows])[:n_points]
    freqs, counts = seeds, measure(seeds, mw_power_dbm)
    grids = [np.arange(a, b + step / 2, step) for a, b in windows]
    grid = np.concatenate(grids)
    surrogate = surrogate if surrogate is not None else OdmrSurrogate()
    scales, batch, base = [], 0, baseline
    while len(freqs) < n_points:
        reoptimise = batch % refit_every == 0
        if baseline is None and reoptimise:
            base = estimate_baseline(freqs, counts, dwell_ms)
        surrogate.fit(freqs, counts, dwell_ms, optimize=reoptimise, baseline=base)
        scales.append(surrogate.length_scale)
        remaining = n_points - len(freqs)
        if allocation == "per_window":
            ranked = sorted(grids, key=lambda g: -float(surrogate.acquisition(g, beta).max()))
            picks = np.concatenate([suggest_frequencies(surrogate, g, 1, beta=beta) for g in ranked[:remaining]])
        else:
            picks = suggest_frequencies(surrogate, grid, min(batch_size, remaining), beta=beta)
        freqs = np.concatenate([freqs, picks])
        counts = np.concatenate([counts, measure(picks, mw_power_dbm)])
        batch += 1
    return RefineResult(freqs, counts, windows, scales)


@dataclass
class AdaptiveConfig:
    """Defaults tuned on development seeds 1000-1019 (see benchmarks/)."""

    survey_step: float = 3.0  # MHz
    survey_power_dbm: float = 10.0  # power-broadened lines are easy to find on a coarse grid
    refine_power_dbm: float = 0.0  # saturation s ~ 2 maximises depth / width for centre estimation
    refine_points: int = 384
    batch_size: int = 8
    window_half_width: float = 8.0
    seeds_per_window: int = 8
    beta: float = 4.0
    laser_mw: float = 2.0
    dwell_ms: float = 20.0


@dataclass
class AdaptiveRun:
    survey_freqs: np.ndarray
    survey_counts: np.ndarray
    survey_centers: list[float]
    refine: RefineResult
    baseline: Callable[[np.ndarray], np.ndarray]  # survey baseline, counts/ms vs MHz

    @property
    def refine_freqs(self) -> np.ndarray:
        return self.refine.freqs

    @property
    def refine_counts(self) -> np.ndarray:
        return self.refine.counts

    @property
    def n_points(self) -> int:
        return len(self.survey_freqs) + len(self.refine.freqs)


def survey_and_refine(measure: MeasureFn, f_start: float, f_stop: float, config: AdaptiveConfig) -> AdaptiveRun:
    """Coarse high-power survey, then low-power GP refinement around every detected line."""
    survey_f = np.arange(f_start, f_stop + config.survey_step / 2, config.survey_step)
    survey_c = measure(survey_f, config.survey_power_dbm)
    # Keep every survey candidate: a spurious window costs a few points, a missed line
    # costs the measurement. Overlapping windows merge.
    cands = find_lines(survey_f, survey_c, config.dwell_ms, merge_width_factor=0.25, max_lines=32)
    centers = sorted(c.center for c in cands)
    baseline = estimate_baseline(survey_f, survey_c, config.dwell_ms)
    refine = refine_odmr(
        measure,
        centers,
        config.refine_points,
        config.refine_power_dbm,
        config.dwell_ms,
        half_width=config.window_half_width,
        batch_size=config.batch_size,
        seeds_per_window=config.seeds_per_window,
        beta=config.beta,
        baseline=baseline,
    )
    return AdaptiveRun(survey_f, survey_c, centers, refine, baseline)
