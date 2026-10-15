"""Static figures of fits (PNG), for the lab record and for the agents to inspect."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .fitting import OdmrFit, TraceFit, _sorted_rates  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
DATA, MODEL = "#2a78d6", "#eb6834"  # validated categorical slots 1 and 2


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_odmr_fit(path, freqs, counts, dwell_ms, fit: OdmrFit, title: str) -> None:
    """Overview of the measured band plus one zoom per fitted line (data vs. model)."""
    f, rate, _ = _sorted_rates(freqs, counts, dwell_ms)
    baseline = lambda x: fit.base_rate * (1 + fit.slope * (x - fit.f_ref))
    dense = np.linspace(f[0], f[-1], 4000)
    n = len(fit.lines)
    cols = min(4, max(n, 1))
    rows = 1 + int(np.ceil(n / cols))
    fig = plt.figure(figsize=(11, 2.3 * rows), facecolor=SURFACE)
    grid = fig.add_gridspec(rows, cols, hspace=0.6, wspace=0.3)
    ax = fig.add_subplot(grid[0, :])
    _style(ax)
    ax.plot(f, rate / baseline(f), "o", ms=2.5, color=DATA, label="data", zorder=2)
    ax.plot(dense, fit.model_rate(dense) / baseline(dense), "-", lw=1.4, color=MODEL, label="fit", zorder=3)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel("MW frequency (MHz)", fontsize=8, color=INK_2)
    ax.set_ylabel("fluorescence (norm.)", fontsize=8, color=INK_2)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for k, line in enumerate(fit.lines):
        ax_k = fig.add_subplot(grid[1 + k // cols, k % cols])
        _style(ax_k)
        near = np.abs(f - line.center) < 6.0
        zoom = np.linspace(line.center - 6.0, line.center + 6.0, 600)
        ax_k.plot(f[near] - line.center, rate[near] / baseline(f[near]), "o", ms=3, color=DATA)
        ax_k.plot(zoom - line.center, fit.model_rate(zoom) / baseline(zoom), "-", lw=1.4, color=MODEL)
        err = f" ± {line.center_err * 1e3:.0f} kHz" if line.center_err else ""
        ax_k.set_title(f"{line.center:.3f} MHz{err}", fontsize=8, color=INK)
        ax_k.set_xlim(-6.0, 6.0)
        ax_k.set_xlabel("detuning (MHz)", fontsize=7, color=INK_2)
    fig.savefig(path, dpi=90, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_trace_fit(path, x, counts, dwell_ms, fit: TraceFit, xlabel: str, title: str, logx: bool = False) -> None:
    """Pulsed trace (Rabi or relaxation) with its model and normalised residuals."""
    xs, rate, sigma = _sorted_rates(x, counts, dwell_ms)
    dense = np.geomspace(max(xs[0], 1e-3), xs[-1], 800) if logx else np.linspace(xs[0], xs[-1], 800)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True, height_ratios=[3, 1], facecolor=SURFACE)
    for ax in (top, bottom):
        _style(ax)
        if logx:
            ax.set_xscale("log")
    top.plot(xs, rate, "o", ms=3, color=DATA, label="data")
    top.plot(dense, fit.model_rate(dense), "-", lw=1.4, color=MODEL, label="fit")
    top.set_ylabel("counts / ms", fontsize=8, color=INK_2)
    top.set_title(title, fontsize=10, color=INK, loc="left")
    top.legend(frameon=False, fontsize=8)
    bottom.plot(xs, (rate - fit.model_rate(xs)) / sigma, "o", ms=2.5, color=DATA)
    bottom.axhline(0.0, color=INK_2, lw=0.8)
    bottom.set_ylabel("resid. (σ)", fontsize=8, color=INK_2)
    bottom.set_xlabel(xlabel, fontsize=8, color=INK_2)
    fig.savefig(path, dpi=90, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
