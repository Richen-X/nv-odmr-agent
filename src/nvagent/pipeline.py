"""One-call ODMR analysis (detection -> lmfit -> field vector), shared by tools and benchmarks."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fieldfit import FieldFit, coverage_intervals, fit_field
from .fitting import LineCandidate, OdmrFit, find_lines, fit_odmr

STRONG_Z = 6.0  # blind-detection threshold over a wide band
GUIDED_Z = 4.0  # threshold inside windows where a survey already placed a line


@dataclass
class OdmrAnalysis:
    candidates: list[LineCandidate]
    fit: OdmrFit
    field: FieldFit | None
    notes: list[str] = field(default_factory=list)


def survey_candidates(freqs, counts, dwell_ms, z_min: float = STRONG_Z) -> list[LineCandidate]:
    """Line candidates in a coarse, power-broadened survey.

    Keeps every candidate (a spurious refinement window costs a few points, a missed line
    costs the measurement); broad candidates closer than about 1.2 hf + w/4 merge.
    """
    return find_lines(freqs, counts, dwell_ms, z_min=z_min, merge_width_factor=0.25, max_lines=32)


def select_candidates(
    cands: list[LineCandidate],
    prior_centers=None,
    max_lines: int = 12,
    match_radius: float = 4.5,
    min_separation: float = 1.0,
) -> list[LineCandidate]:
    """Choose at most ``max_lines`` line candidates, guided by rough prior positions.

    Every prior keeps the detections assigned to it (several if they are resolved, e.g.
    a pair that merged in a power-broadened survey) or, if none, itself as a placeholder
    (z = 0, unknown width/depth). Strong detections far from every prior are kept too.
    """
    if prior_centers is None:
        return sorted(sorted(cands, key=lambda c: -c.z)[:max_lines], key=lambda c: c.center)
    priors = np.sort(np.asarray(prior_centers, dtype=float))
    groups: dict[int, list[LineCandidate]] = {i: [] for i in range(len(priors))}
    chosen: list[LineCandidate] = []
    for c in cands:
        dist = np.abs(priors - c.center)
        i = int(np.argmin(dist))
        if dist[i] <= match_radius:
            groups[i].append(c)
        elif c.z >= STRONG_Z:
            chosen.append(c)
    for i, prior in enumerate(priors):
        kept: list[LineCandidate] = []
        for c in sorted(groups[i], key=lambda c: -c.z):
            if all(abs(c.center - k.center) >= min_separation for k in kept):
                kept.append(c)
        chosen.extend(kept or [LineCandidate(float(prior), float("nan"), float("nan"), 0.0)])
    chosen = sorted(chosen, key=lambda c: -c.z)[:max_lines]
    return sorted(chosen, key=lambda c: c.center)


def drop_least_needed_line(result: OdmrFit, fit, notes: list[str], split_radius: float = 3.5) -> OdmrFit:
    """Backward elimination: remove the line whose removal raises chi^2 the least.

    Individual depth/error is a poor guide when lines are degenerate: a triplet split into
    two overlapping ones gives two "significant" lines, yet either can go at almost no
    cost, whereas an isolated real line cannot. Candidates are lines with a neighbour within
    ``split_radius`` MHz plus the three individually least significant lines.
    """
    lines = result.lines
    centres = np.array([ln.center for ln in lines])
    significance = np.array([ln.depth / ln.depth_err if ln.depth_err else 0.0 for ln in lines])
    crowded = {k for k in range(len(lines)) if np.sum(np.abs(centres - centres[k]) < split_radius) > 1}
    candidates = sorted(crowded | set(np.argsort(significance)[:3].tolist()))
    best = None
    for k in candidates:
        rest = [ln for j, ln in enumerate(lines) if j != k]
        trial = fit([ln.center for ln in rest], [ln.fwhm for ln in rest], [ln.depth for ln in rest])
        if best is None or trial.chisqr < best[0].chisqr:
            best = (trial, k)
    trial, k = best
    notes.append(f"dropped line at {lines[k].center:.2f} MHz (chi2 +{trial.chisqr - result.chisqr:.1f} without it)")
    return trial


def hyperfine_alias_check(result: OdmrFit, fit, notes: list[str], min_gain: float = 4.0, rounds: int = 2) -> OdmrFit:
    """Test moving lines by one hyperfine spacing; keep moves that lower chi^2 by > min_gain.

    Two triplets about 3 hf apart form a near-regular six-line comb, and a triplet placed
    one hf off (sharing three of the six components) is a local minimum of the fit. Only
    lines with a neighbour within 4 hf can be trapped this way, so only those are tested.
    """
    for _ in range(rounds):
        moved = False
        for k in range(len(result.lines)):
            centres = [ln.center for ln in result.lines]
            if not any(abs(c - centres[k]) < 4 * result.hf_split for j, c in enumerate(centres) if j != k):
                continue
            for shift in (-result.hf_split, result.hf_split):
                trial_centres = list(centres)
                trial_centres[k] += shift
                trial = fit(trial_centres, [ln.fwhm for ln in result.lines], [ln.depth for ln in result.lines])
                if trial.chisqr < result.chisqr - min_gain:
                    notes.append(
                        f"hyperfine-alias check moved {centres[k]:.2f} -> {trial_centres[k]:.2f} MHz "
                        f"(chi2 {result.chisqr:.1f} -> {trial.chisqr:.1f})"
                    )
                    result, moved = trial, True
                    break
        if not moved:
            break
    return result


def analyze_odmr(
    freqs,
    counts,
    dwell_ms,
    prior_centers=None,
    z_min: float | None = None,
    max_lines: int = 8,
    max_candidates: int = 12,
    hyperfine: str = "shared",
    width_mode: str = "per_line",
    with_field: bool = True,
    baseline=None,
) -> OdmrAnalysis:
    """Find lines, fit them with lmfit and invert the centres into a field vector.

    ``baseline`` (callable f -> counts/ms, e.g. ``estimate_baseline`` of a survey at the same
    laser power) is needed for data taken only around the lines.
    """
    z = z_min if z_min is not None else (GUIDED_Z if prior_centers is not None else STRONG_Z)
    cands = find_lines(freqs, counts, dwell_ms, z_min=z, max_lines=32, baseline=baseline, prior_centers=prior_centers)
    chosen = select_candidates(cands, prior_centers, max_candidates)
    if not chosen:
        raise ValueError(f"no line found above z = {z}")
    notes = []
    placeholders = [c.center for c in chosen if c.z == 0.0]
    if placeholders:
        notes.append(f"no detection near prior(s) {np.round(placeholders, 2).tolist()}: fitted from the prior position")

    def fit(centers, widths, depths):
        return fit_odmr(freqs, counts, dwell_ms, centers, widths, depths, hyperfine=hyperfine, width_mode=width_mode)

    result = fit([c.center for c in chosen], [c.width for c in chosen], [c.depth for c in chosen])
    while len(result.lines) > max_lines:
        result = drop_least_needed_line(result, fit, notes)
    result = hyperfine_alias_check(result, fit, notes)
    field_fit = None
    if with_field and len(result.lines) >= 3:
        field_fit = fit_field(
            [ln.center for ln in result.lines],
            [ln.center_err if ln.center_err else 0.5 for ln in result.lines],
            coverage=coverage_intervals(freqs),
        )
    return OdmrAnalysis(chosen, result, field_fit, notes)
