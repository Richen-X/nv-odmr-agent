"""Line finding and lmfit models for ODMR spectra and pulsed traces.

All fits work in count-rate space (counts per ms of dwell) with Poisson weights, so a
correct model gives a reduced chi-square close to 1 regardless of how the data were
sampled (uniform sweeps, adaptive point sets, mixed dwell times).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import lmfit
import numpy as np
from scipy.optimize import nnls
from scipy.signal import fftconvolve

HF_SPLIT = 2.15  # MHz, initial guess for the 14N triplet spacing
TEMPLATE_WIDTHS = tuple(0.5 * 1.25**k for k in range(13))  # 0.5 ... 7.3 MHz


# --------------------------------------------------------------------------- utilities


def _sorted_rates(freqs, counts, dwell_ms):
    x = np.asarray(freqs, dtype=float)
    c = np.asarray(counts, dtype=float)
    d = np.broadcast_to(np.asarray(dwell_ms, dtype=float), c.shape)
    order = np.argsort(x, kind="stable")
    x, c, d = x[order], c[order], d[order]
    return x, c / d, np.sqrt(np.maximum(c, 1.0)) / d


def _lorentz(f, center, width):
    return 1.0 / (1.0 + ((f - center) / (0.5 * width)) ** 2)


def _line_shape(f, center, width, hf, hyperfine: bool):
    if not hyperfine:
        return _lorentz(f, center, width)
    return _lorentz(f, center - hf, width) + _lorentz(f, center, width) + _lorentz(f, center + hf, width)


def robust_baseline(f, rate, sigma, clip: float = 2.5, n_iter: int = 8):
    """Weighted linear baseline that iteratively ignores points sitting in dips.

    Starts from the upper half of the data so that it still lands near the baseline when
    most points are inside dips. Returns (baseline at f, (offset, slope), reference freq).
    """
    f_ref = 0.5 * (f[0] + f[-1])
    mask = rate >= np.median(rate)
    for _ in range(n_iter):
        sw = 1.0 / sigma[mask]
        design = np.vstack([np.ones(mask.sum()), f[mask] - f_ref]).T
        coef = np.linalg.lstsq(design * sw[:, None], rate[mask] * sw, rcond=None)[0]
        base = coef[0] + coef[1] * (f - f_ref)
        new_mask = (rate - base) / sigma > -clip
        if new_mask.sum() < 3 or np.array_equal(new_mask, mask):
            break
        mask = new_mask
    return base, coef, f_ref


# ------------------------------------------------------------------------ line finding


@dataclass
class LineCandidate:
    center: float  # MHz
    depth: float  # fractional depth of one hyperfine component
    width: float  # MHz, best-matching template FWHM
    z: float  # matched-filter significance


def _matching_pursuit(f, resid, weight, hf_split, widths, z_min, step, max_iter) -> list[LineCandidate]:
    """Greedy matched-filter detection of triplets in ``resid`` (positive inside dips).

    For every trial centre g and width w the best-fit depth of a triplet template T is
    a = sum(w_i y_i T_i) / sum(w_i T_i^2) with significance z = a / sigma_a. Working on a
    binned grid turns both sums into FFT convolutions, so irregular (adaptive) sampling
    is handled exactly like uniform sweeps. After each detection its template is
    subtracted, which removes the side lobes a triplet leaves at +-hf and +-2hf.
    """
    idx = np.rint((f - f[0]) / step).astype(int)
    n = int(idx[-1]) + 1
    grid = f[0] + step * np.arange(n)
    occupancy = np.bincount(idx, minlength=n).astype(float)
    binned_weight = np.bincount(idx, weights=weight, minlength=n)

    templates, denominators, supported = {}, {}, {}
    for w in widths:
        half = int((8 * w + 2 * hf_split) / step)
        t = _line_shape(step * np.arange(-half, half + 1), 0.0, w, hf_split, True)
        templates[w] = t
        denominators[w] = np.maximum(fftconvolve(binned_weight, t**2, mode="same"), 1e-300)
        # A line may only be claimed where its core was measured: >= 3 points within
        # +-(hf + w/2). Otherwise a broad template centred in a data gap "detects" the
        # wings of real lines on either side.
        core = np.ones(2 * int((hf_split + 0.5 * w) / step) + 1)
        supported[w] = fftconvolve(occupancy, core, mode="same") > 2.5

    found: list[LineCandidate] = []
    y = np.array(resid, dtype=float)
    for _ in range(max_iter):
        binned = np.bincount(idx, weights=weight * y, minlength=n)
        best = None
        for w in widths:
            num = fftconvolve(binned, templates[w], mode="same")
            z = np.where(supported[w], num / np.sqrt(denominators[w]), -np.inf)
            k = int(np.argmax(z))
            if best is None or z[k] > best[0]:
                best = (z[k], k, w, z, num[k] / denominators[w][k])
        z_best, k, w, z, depth = best
        if not np.isfinite(z_best) or z_best < z_min:
            break
        centre = grid[k]
        if 0 < k < n - 1 and np.all(np.isfinite(z[k - 1 : k + 2])):
            curvature = z[k - 1] - 2 * z[k] + z[k + 1]
            if curvature < 0:
                centre += 0.5 * step * (z[k - 1] - z[k + 1]) / curvature
        found.append(LineCandidate(float(centre), float(depth), float(w), float(z_best)))
        y = y - depth * _line_shape(f, centre, w, hf_split, True)
    return found


def _detect_one_width_class(f, y, weight, hf_split, widths, z_min, step, max_iter) -> list[LineCandidate]:
    """Matching pursuit restricted to the width class that best explains the spectrum.

    All lines in one dataset share MW and laser power and so have similar widths. The class
    is the width whose single-width pursuit explains the most chi-square (sum of z^2 minus
    an AIC-like cost of 6 per line); by Cauchy-Schwarz a broad template cannot out-score
    correctly matched narrow ones on a resolved pair. The final search then allows widths
    within a factor 1.6 of that class to accommodate lines driven with different strength.
    """
    scores = {
        w: sum(c.z**2 - 6.0 for c in _matching_pursuit(f, y, weight, hf_split, [w], z_min, step, max_iter))
        for w in widths
    }
    best = max(scores, key=scores.get)
    if scores[best] <= 0:
        return []
    allowed = [w for w in widths if best / 1.6 <= w <= best * 1.6]
    return _matching_pursuit(f, y, weight, hf_split, allowed, z_min, step, max_iter)


def _baseline_by_joint_detection(f, rate, sigma, hf_split, widths, step, max_iter, rounds: int = 2):
    """Baseline (offset, slope, f_ref) estimated jointly with permissively detected lines.

    When nearly every point lies in a dip or its Lorentzian tails no point measures the
    baseline directly, and any estimate that ignores the dips is biased low. We detect at
    z >= 3, refit the baseline by linear least squares together with templates of those
    detections, and repeat.
    """
    _, coef, f_ref = robust_baseline(f, rate, sigma)
    for _ in range(rounds):
        base = coef[0] + coef[1] * (f - f_ref)
        found = _detect_one_width_class(f, 1.0 - rate / base, (base / sigma) ** 2, hf_split, widths, 3.0, step, max_iter)
        if not found:
            break
        cols = [np.ones_like(f), f - f_ref] + [-_line_shape(f, c.center, c.width, hf_split, True) for c in found]
        coef = np.linalg.lstsq(np.vstack(cols).T / sigma[:, None], rate / sigma, rcond=None)[0][:2]
    return coef, f_ref


def _nnls_lines(f, rate, sigma, base, width, hf_split, regions, z_min, step: float = 0.1) -> list[LineCandidate]:
    """Sparse non-negative deconvolution of the dips into triplet templates.

    Solves min || (base - rate)/sigma - sum_j a_j T_j base/sigma ||^2 with a_j >= 0 over
    trial centres on a 0.1 MHz grid. Unlike greedy matching pursuit this explains all
    lines jointly, so two triplets about 3 hf apart (whose six components form a near
    regular comb) are not mistaken for one triplet centred between them. Contiguous runs
    of active atoms are one line: centre = amplitude-weighted mean, depth = summed amplitude.
    The reported z (depth / single-template error) is a ranking heuristic, not a calibrated
    significance: non-negativity selects atoms and adjacent atoms are summed.
    """
    grid = np.arange(f[0] - 1.0, f[-1] + 1.0, step)
    reach = hf_split + 0.5 * width
    core = np.searchsorted(f, grid + reach, side="right") - np.searchsorted(f, grid - reach)
    keep = core >= 3
    if len(regions):
        keep &= np.min(np.abs(grid[:, None] - np.asarray(regions, dtype=float)[None, :]), axis=1) < 3 * hf_split + 4 * width
    grid = grid[keep]
    if len(grid) == 0:
        return []
    scale = base / sigma
    design = _line_shape(f[:, None], grid[None, :], width, hf_split, True) * scale[:, None]
    amps, _ = nnls(design, (base - rate) / sigma, maxiter=20 * len(grid))
    active = np.nonzero(amps > 0)[0]
    if len(active) == 0:
        return []
    runs = np.split(active, np.nonzero(np.diff(grid[active]) > 0.6 * width)[0] + 1)
    lines = [(float(np.sum(amps[r] * grid[r]) / amps[r].sum()), float(amps[r].sum())) for r in runs]
    merged: list[list[float]] = []  # runs split by noise belong to the same line
    for centre, depth in lines:
        if merged and centre - merged[-1][0] < max(1.0, 0.8 * width):
            c0, d0 = merged[-1]
            merged[-1] = [(c0 * d0 + centre * depth) / (d0 + depth), d0 + depth]
        else:
            merged.append([centre, depth])
    out = []
    for centre, depth in merged:
        z = depth * float(np.sqrt(np.sum((_line_shape(f, centre, width, hf_split, True) * scale) ** 2)))
        if z >= z_min:
            out.append(LineCandidate(centre, depth, float(width), z))
    return out


def _detect(f, rate, sigma, hf_split, widths, z_min, step, max_iter, baseline=None, prior_centers=None):
    """Find lines: baseline -> width class -> joint non-negative deconvolution.

    ``baseline`` (callable f -> counts/ms) should be supplied for data without
    off-resonant points (e.g. from a survey at the same laser power); otherwise it is
    estimated jointly with the lines. Returns (candidates, (offset, slope), f_ref).
    """
    if baseline is None:
        coef, f_ref = _baseline_by_joint_detection(f, rate, sigma, hf_split, widths, step, max_iter)
        base = coef[0] + coef[1] * (f - f_ref)
    else:
        f_ref = 0.5 * (f[0] + f[-1])
        base = np.asarray(baseline(f), dtype=float)
        coef = np.polyfit(f - f_ref, base, 1)[::-1]
    y, weight = 1.0 - rate / base, (base / sigma) ** 2
    rough = _matching_pursuit(f, y, weight, hf_split, widths, 2.5, step, 40)
    cls = _detect_one_width_class(f, y, weight, hf_split, widths, min(z_min, 4.0), step, max_iter) or rough
    width = float(np.median([c.width for c in cls])) if cls else 1.0
    regions = [c.center for c in rough] + ([] if prior_centers is None else list(prior_centers))
    return _nnls_lines(f, rate, sigma, base, width, hf_split, regions, z_min), coef, f_ref


def estimate_baseline(freqs, counts, dwell_ms, hf_split: float = HF_SPLIT):
    """Baseline count rate (counts per ms) as a function of frequency, robust to dip-dominated data."""
    f, rate, sigma = _sorted_rates(freqs, counts, dwell_ms)
    coef, f_ref = _baseline_by_joint_detection(f, rate, sigma, hf_split, TEMPLATE_WIDTHS, 0.05, 24)
    return lambda x: coef[0] + coef[1] * (np.asarray(x, dtype=float) - f_ref)


def find_lines(
    freqs,
    counts,
    dwell_ms,
    hf_split: float = HF_SPLIT,
    widths=TEMPLATE_WIDTHS,
    z_min: float = 6.0,
    min_separation: float = 1.5,
    merge_width_factor: float = 0.0,
    max_lines: int = 12,
    step: float = 0.05,
    max_iter: int = 24,
    baseline=None,
    prior_centers=None,
) -> list[LineCandidate]:
    """Detect NV lines (14N triplets) in ODMR data with any sampling pattern.

    Candidates closer than max(min_separation, 1.2 hf + merge_width_factor * width) are
    merged, keeping the more significant one. The joint deconvolution has no side lobes,
    so by default only near-coincident candidates merge; weak extra structure (e.g.
    nuclear-spin-flip lines) is left for the fit to prune. ``baseline``: callable
    f -> counts/ms for data without off-resonant points. Returns candidates by frequency.
    """
    f, rate, sigma = _sorted_rates(freqs, counts, dwell_ms)
    found, _, _ = _detect(f, rate, sigma, hf_split, widths, z_min, step, max_iter, baseline, prior_centers)

    kept: list[LineCandidate] = []
    for cand in sorted(found, key=lambda c: -c.z):
        radius = lambda other: max(min_separation, 1.2 * hf_split + merge_width_factor * max(cand.width, other.width))
        if all(abs(cand.center - other.center) >= radius(other) for other in kept):
            kept.append(cand)
    return sorted(kept[:max_lines], key=lambda c: c.center)


# ----------------------------------------------------------------------------- ODMR fit


@dataclass
class LineFit:
    center: float
    center_err: float | None
    depth: float
    depth_err: float | None
    fwhm: float
    fwhm_err: float | None


@dataclass
class OdmrFit:
    lines: list[LineFit]
    hyperfine: str
    hf_split: float
    hf_split_err: float | None
    base_rate: float  # counts per ms at f_ref
    slope: float  # fractional baseline change per MHz
    f_ref: float
    redchi: float
    chisqr: float
    ndata: int
    nvarys: int
    success: bool
    message: str
    diagnostics: dict = field(default_factory=dict)

    def model_rate(self, f) -> np.ndarray:
        f = np.asarray(f, dtype=float)
        shape = np.zeros_like(f)
        for ln in self.lines:
            shape += ln.depth * _line_shape(f, ln.center, ln.fwhm, self.hf_split, self.hyperfine != "none")
        return self.base_rate * (1 + self.slope * (f - self.f_ref)) * (1 - shape)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OdmrFit":
        d = dict(d)
        d["lines"] = [LineFit(**ln) for ln in d["lines"]]
        return cls(**d)


def _err(param) -> float | None:
    return None if param.stderr is None or not np.isfinite(param.stderr) else float(param.stderr)


def fit_odmr(
    freqs,
    counts,
    dwell_ms,
    centers,
    widths=None,
    depths=None,
    hyperfine: str = "shared",
    width_mode: str = "per_line",
    hf_split: float = HF_SPLIT,
) -> OdmrFit:
    """Fit a sum of Lorentzian lines (or 14N triplets) on a sloped baseline.

    hyperfine: "shared" (triplets, common free spacing), "fixed" (spacing held at
    ``hf_split``) or "none" (single Lorentzian per line, e.g. for power-broadened data).
    width_mode: "per_line" or "shared".
    """
    if hyperfine not in ("shared", "fixed", "none"):
        raise ValueError("hyperfine must be 'shared', 'fixed' or 'none'")
    if width_mode not in ("per_line", "shared"):
        raise ValueError("width_mode must be 'per_line' or 'shared'")
    f, rate, sigma = _sorted_rates(freqs, counts, dwell_ms)
    order = np.argsort(np.asarray(centers, dtype=float))
    centers = np.asarray(centers, dtype=float)[order]
    if len(centers) == 0:
        raise ValueError("need at least one line centre")
    triplet = hyperfine != "none"
    base, coef, f_ref = robust_baseline(f, rate, sigma)
    dip = 1.0 - rate / base

    # Initial widths/depths: given values, or estimated from the data where missing (NaN).
    w_init = np.asarray(widths, dtype=float)[order] if widths is not None else np.full(len(centers), np.nan)
    a_init = np.asarray(depths, dtype=float)[order] if depths is not None else np.full(len(centers), np.nan)
    params = lmfit.Parameters()
    params.add("base", value=coef[0], min=0.0)
    params.add("slope", value=coef[1] / coef[0], min=-1e-3, max=1e-3)
    params.add("hf", value=hf_split, min=1.8, max=2.5, vary=hyperfine == "shared")
    for k, c in enumerate(centers):
        w0 = float(w_init[k]) if np.isfinite(w_init[k]) else 1.0
        if np.isfinite(a_init[k]):
            a0 = float(a_init[k])
        else:
            near = np.abs(f - c) < max(0.5, w0 / 2)
            peak = dip[near].max() if near.any() else 0.005
            a0 = peak / (1 + 2 * _lorentz(hf_split, 0.0, w0)) if triplet else peak
        reach = max(1.5, w0)
        params.add(f"c{k}", value=c, min=c - reach, max=c + reach)
        params.add(f"a{k}", value=float(np.clip(a0, 1e-4, 0.4)), min=0.0, max=0.5)
        if width_mode == "per_line" or k == 0:
            params.add(f"w{k}", value=w0, min=0.05, max=30.0)
        else:
            params.add(f"w{k}", expr="w0")

    def residual(p):
        v = p.valuesdict()
        shape = np.zeros_like(f)
        for k in range(len(centers)):
            shape += v[f"a{k}"] * _line_shape(f, v[f"c{k}"], v[f"w{k}"], v["hf"], triplet)
        return (rate - v["base"] * (1 + v["slope"] * (f - f_ref)) * (1 - shape)) / sigma

    out = lmfit.minimize(residual, params, method="least_squares")
    if not out.errorbars:  # retry from the optimum with Levenberg-Marquardt for a covariance
        out = lmfit.minimize(residual, out.params, method="leastsq")
    p = out.params
    lines = [
        LineFit(
            center=float(p[f"c{k}"].value),
            center_err=_err(p[f"c{k}"]),
            depth=float(p[f"a{k}"].value),
            depth_err=_err(p[f"a{k}"]),
            fwhm=float(p[f"w{k}"].value),
            fwhm_err=_err(p[f"w{k}"]),
        )
        for k in range(len(centers))
    ]
    lines.sort(key=lambda ln: ln.center)
    hf = float(p["hf"].value)
    diagnostics = residual_diagnostics(
        f, residual(p), [ln.center for ln in lines], [ln.fwhm for ln in lines], hf if triplet else 0.0
    )
    return OdmrFit(
        lines=lines,
        hyperfine=hyperfine,
        hf_split=hf,
        hf_split_err=_err(p["hf"]) if hyperfine == "shared" else None,
        base_rate=float(p["base"].value),
        slope=float(p["slope"].value),
        f_ref=float(f_ref),
        redchi=float(out.redchi),
        chisqr=float(out.chisqr),
        ndata=int(out.ndata),
        nvarys=int(out.nvarys),
        success=bool(out.success),
        message=str(out.message),
        diagnostics=diagnostics,
    )


def _clusters(f, score, mask, sign):
    out, i, n = [], 0, len(f)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        seg = slice(i, j + 1)
        k = i + int(np.argmax(sign * score[seg]))
        out.append({"freq_mhz": round(float(f[k]), 3), "significance": round(float(score[k]), 1), "n_points": j - i + 1})
        i = j + 1
    return out


def residual_diagnostics(f, r, centers, widths, hf, window: float = 1.0, threshold: float = 4.0) -> dict:
    """Structure in normalised residuals r = (data - model) / sigma, f sorted ascending.

    The local score S_i = sum(r_j) / sqrt(n) over |f_j - f_i| < window flags unexplained dips
    (S < -threshold, e.g. a missed line) and places where the model is too deep (S > threshold).
    """
    n = len(r)
    positive = r > 0
    n1, n2 = int(positive.sum()), int(n - positive.sum())
    runs = 1 + int(np.count_nonzero(positive[1:] != positive[:-1]))
    mu = 2 * n1 * n2 / n + 1
    var = (mu - 1) * (mu - 2) / (n - 1) if n > 1 else 0.0
    runs_z = (runs - mu) / np.sqrt(var) if var > 0 else 0.0
    lo = np.searchsorted(f, f - window, side="left")
    hi = np.searchsorted(f, f + window, side="right")
    csum = np.concatenate([[0.0], np.cumsum(r)])
    score = (csum[hi] - csum[lo]) / np.sqrt(hi - lo)
    local = []
    for c, w in zip(centers, widths):
        near = np.abs(f - c) < hf + 2 * w
        local.append(round(float(np.mean(r[near] ** 2)), 2) if near.sum() > 2 else None)
    k = int(np.argmax(np.abs(r)))
    return {
        "runs_z": round(float(runs_z), 2),
        "max_abs_resid": round(float(abs(r[k])), 2),
        "max_abs_resid_freq_mhz": round(float(f[k]), 3),
        "unexplained_dips": _clusters(f, score, score < -threshold, -1),
        "model_too_deep": _clusters(f, score, score > threshold, +1),
        "line_local_redchi": local,
    }


# ----------------------------------------------------------------------- pulsed traces


@dataclass
class TraceFit:
    kind: str  # "rabi" or "decay"
    params: dict  # name -> [value, stderr]
    redchi: float
    ndata: int
    success: bool
    derived: dict

    def model_rate(self, x) -> np.ndarray:
        v = {k: val for k, (val, _) in self.params.items()}
        x = np.asarray(x, dtype=float)
        if self.kind == "rabi":
            t = x / 1000.0
            return v["c"] + v["A"] * np.exp(-t / v["T"]) * np.cos(2 * np.pi * v["f"] * t + v["phi"])
        return v["c"] + v["A"] * np.exp(-((x / v["T"]) ** v["beta"]))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TraceFit":
        return cls(**d)


def _pack(params) -> dict:
    return {k: [float(p.value), _err(p)] for k, p in params.items()}


def fit_rabi(taus_ns, counts, dwell_ms) -> TraceFit:
    """Damped cosine c + A exp(-t/T) cos(2 pi f t + phi); pi time = 1 / (2 f)."""
    tau, rate, sigma = _sorted_rates(taus_ns, counts, dwell_ms)
    t = tau / 1000.0  # us, so f is in MHz
    detrended = rate - rate.mean()
    span = t[-1] - t[0]
    trial = np.linspace(0.5 / span, 0.5 / max(np.median(np.diff(t)), 1e-6), 2000)
    power = np.abs(np.exp(-2j * np.pi * trial[:, None] * t[None, :]) @ detrended)
    f0 = float(trial[np.argmax(power)])

    params = lmfit.Parameters()
    params.add("c", value=float(rate.mean()))
    params.add("A", value=float(max(rate[0] - rate.mean(), 1e-6)), min=0.0)
    params.add("f", value=f0, min=0.0, max=float(trial[-1]))
    params.add("T", value=float(span), min=1e-3, max=1e4)
    params.add("phi", value=0.0, min=-np.pi, max=np.pi)

    def residual(p):
        v = p.valuesdict()
        return (rate - (v["c"] + v["A"] * np.exp(-t / v["T"]) * np.cos(2 * np.pi * v["f"] * t + v["phi"]))) / sigma

    out = lmfit.minimize(residual, params, method="least_squares")
    f_fit, f_err = out.params["f"].value, _err(out.params["f"])
    return TraceFit(
        kind="rabi",
        params=_pack(out.params),
        redchi=float(out.redchi),
        ndata=int(out.ndata),
        success=bool(out.success),
        derived={
            "rabi_freq_mhz": float(f_fit),
            "pi_time_ns": float(1000.0 / (2 * f_fit)),
            "pi_time_err_ns": None if f_err is None else float(1000.0 * f_err / (2 * f_fit**2)),
            "contrast": float(2 * out.params["A"].value / (out.params["c"].value + out.params["A"].value)),
        },
    )


def fit_decay(taus_us, counts, dwell_ms, stretched: bool = True) -> TraceFit:
    """(Stretched) exponential c + A exp(-(t/T)^beta), e.g. for T1 or Hahn-echo T2."""
    t, rate, sigma = _sorted_rates(taus_us, counts, dwell_ms)
    tail = max(3, len(t) // 10)
    c0 = float(rate[-tail:].mean())
    a0 = float(rate[:tail].mean() - c0)
    below = np.nonzero(np.abs(rate - c0) < abs(a0) / np.e)[0]
    t0 = float(t[below[0]]) if len(below) else float(np.median(t))

    params = lmfit.Parameters()
    params.add("c", value=c0)
    params.add("A", value=a0)
    params.add("T", value=max(t0, 1e-6), min=1e-9)
    params.add("beta", value=0.8 if stretched else 1.0, min=0.2, max=3.0, vary=stretched)

    def residual(p):
        v = p.valuesdict()
        return (rate - (v["c"] + v["A"] * np.exp(-((t / v["T"]) ** v["beta"])))) / sigma

    out = lmfit.minimize(residual, params, method="least_squares")
    p = out.params
    return TraceFit(
        kind="decay",
        params=_pack(p),
        redchi=float(out.redchi),
        ndata=int(out.ndata),
        success=bool(out.success),
        derived={"T_us": float(p["T"].value), "T_err_us": _err(p["T"]), "beta": float(p["beta"].value), "beta_err": _err(p["beta"])},
    )
