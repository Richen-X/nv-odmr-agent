"""Figures and a markdown table from benchmarks/results/adaptive_vs_uniform.json.

    python benchmarks/plot_results.py

Writes docs/img/benchmark_light.png, docs/img/benchmark_dark.png and benchmarks/results/summary.md.
The uniform sweep is shown at its best fixed MW power for every step size (lowest median
field error), which is the most favourable comparison for the baseline.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import NullFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results" / "adaptive_vs_uniform.json"
TARGET_MG = 40.0  # fixed before the held-out run: median field error <= 40 mG with >= 90 % success

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e4e3de",
              "series": {"gp": "#2a78d6", "uniform": "#eb6834", "uniform_windows": "#1baf7a"}},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "grid": "#33332f",
             "series": {"gp": "#3987e5", "uniform": "#d95926", "uniform_windows": "#199e70"}},
}
LABELS = {"gp": "GP two-stage (this work)", "uniform": "Uniform sweep, best power", "uniform_windows": "Two-stage, no GP (ablation)"}
MARKERS = {"gp": "o", "uniform": "s", "uniform_windows": "^"}


def curves(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["method"], r.get("step"), r.get("refine"), r["power"])
        groups.setdefault(key, []).append(r)
    points = []
    for (method, step, refine, power), rs in groups.items():
        field = np.array([r["field_err"] if r["field_err"] is not None else np.inf for r in rs]) * 1e3
        line = np.array([r["max_line_err"] if r["max_line_err"] is not None else np.inf for r in rs]) * 1e3
        points.append({
            "method": method, "step": step, "refine": refine, "power": power,
            "n_points": int(np.median([r["n_points"] for r in rs])),
            "success": float(np.mean(line < 500.0)),
            "field_med": float(np.median(field)), "field_q1": float(np.percentile(field, 25)), "field_q3": float(np.percentile(field, 75)),
            "field_p90": float(np.percentile(field, 90)), "line_med": float(np.median(line)), "n": len(rs),
        })
    out: dict[str, list[dict]] = {}
    for method in ("gp", "uniform_windows"):
        out[method] = sorted([p for p in points if p["method"] == method], key=lambda p: p["n_points"])
    best = {}
    for p in (p for p in points if p["method"] == "uniform"):
        cur = best.get(p["step"])
        if cur is None or (p["field_med"], -p["success"]) < (cur["field_med"], -cur["success"]):
            best[p["step"]] = p
    out["uniform"] = sorted(best.values(), key=lambda p: p["n_points"])
    return out


def cheapest_meeting_target(curve: list[dict]) -> dict | None:
    ok = [p for p in curve if p["success"] >= 0.9 and p["field_med"] <= TARGET_MG]
    return min(ok, key=lambda p: p["n_points"]) if ok else None


def figure(data: dict, theme: str, path: Path) -> None:
    t = THEMES[theme]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=t["surface"])
    for ax in axes:
        ax.set_facecolor(t["surface"])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(t["ink2"])
        ax.tick_params(colors=t["ink2"], labelsize=9)
        ax.grid(color=t["grid"], linewidth=0.7)
        ax.set_axisbelow(True)
        ax.set_xscale("log")
        ax.set_xlabel("measured points (20 ms each)", color=t["ink2"], fontsize=9)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks([250, 400, 600, 1000, 1800])
        ax.set_xticklabels(["250", "400", "600", "1000", "1800"])
    left, right = axes
    for method in ("uniform", "uniform_windows", "gp"):
        c = data[method]
        n = [p["n_points"] for p in c]
        color = t["series"][method]
        finite = [np.isfinite(p["field_med"]) for p in c]
        nf = [x for x, ok in zip(n, finite) if ok]
        left.fill_between(nf, [p["field_q1"] for p, ok in zip(c, finite) if ok], [p["field_q3"] for p, ok in zip(c, finite) if ok],
                          color=color, alpha=0.12, linewidth=0)
        left.plot(nf, [p["field_med"] for p, ok in zip(c, finite) if ok], marker=MARKERS[method], ms=6, lw=2, color=color,
                  label=LABELS[method], markeredgecolor=t["surface"], markeredgewidth=1.2)
        right.plot(n, [100 * p["success"] for p in c], marker=MARKERS[method], ms=6, lw=2, color=color, label=LABELS[method],
                   markeredgecolor=t["surface"], markeredgewidth=1.2)
    left.axhline(TARGET_MG, color=t["ink2"], lw=1, ls=(0, (4, 3)))
    left.text(1830, TARGET_MG * 1.04, "target 40 mG", color=t["ink2"], fontsize=8, ha="right", va="bottom")
    gp_hit, uni_hit = cheapest_meeting_target(data["gp"]), cheapest_meeting_target(data["uniform"])
    if gp_hit and uni_hit:
        left.annotate(f"{gp_hit['n_points']} vs {uni_hit['n_points']} points\n({100 * (1 - gp_hit['n_points'] / uni_hit['n_points']):.0f}% fewer)",
                      xy=(gp_hit["n_points"], gp_hit["field_med"]), xytext=(gp_hit["n_points"] * 1.05, 95), color=t["ink"], fontsize=8.5,
                      arrowprops={"arrowstyle": "-", "color": t["ink2"], "lw": 0.8})
    left.set_ylim(0, 140)
    left.set_ylabel("median field error |ΔB| (mG)", color=t["ink2"], fontsize=9)
    left.set_title("Field-vector error (band: interquartile range)", color=t["ink"], fontsize=10, loc="left")
    right.set_ylim(0, 105)
    right.set_ylabel("runs with all 8 lines within 0.5 MHz (%)", color=t["ink2"], fontsize=9)
    right.set_title("Reliability", color=t["ink"], fontsize=10, loc="left")
    handles, labels = left.get_legend_handles_labels()
    order = [labels.index(LABELS[m]) for m in ("gp", "uniform", "uniform_windows")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center", ncol=3, frameon=False,
               fontsize=9, labelcolor=t["ink"], bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=t["surface"])
    plt.close(fig)


def table(data: dict) -> str:
    out = ["| method | setting | points | success | median worst-line error | median / p90 field error |", "|---|---|---|---|---|---|"]
    for method in ("gp", "uniform_windows", "uniform"):
        for p in data[method]:
            setting = f"{p['refine']} refine pts @ 0 dBm" if p["refine"] else f"{p['step']} MHz step @ {p['power']:+.0f} dBm"
            line = f"{p['line_med']:.0f} kHz" if np.isfinite(p["line_med"]) else "n/a"
            field = f"{p['field_med']:.0f} / {p['field_p90']:.0f} mG" if np.isfinite(p["field_p90"]) else (
                f"{p['field_med']:.0f} mG / n/a" if np.isfinite(p["field_med"]) else "n/a")
            out.append(f"| {LABELS[method]} | {setting} | {p['n_points']} | {100 * p['success']:.0f}% | {line} | {field} |")
    return "\n".join(out)


def per_config(rows: list[dict]) -> dict[tuple, dict]:
    """Per configuration: median point count, field error (mG, inf on failure) and success, by seed."""
    seeds = sorted({r["seed"] for r in rows})
    index = {s: i for i, s in enumerate(seeds)}
    out: dict[tuple, dict] = {}
    for r in rows:
        key = (r["method"], r.get("step"), r.get("refine"), r["power"])
        d = out.setdefault(key, {"points": [], "field": np.full(len(seeds), np.inf), "ok": np.zeros(len(seeds), bool)})
        i = index[r["seed"]]
        d["points"].append(r["n_points"])
        if r["field_err"] is not None:
            d["field"][i] = 1e3 * r["field_err"]
        d["ok"][i] = r["max_line_err"] is not None and r["max_line_err"] < 0.5
    for d in out.values():
        d["points"] = int(np.median(d["points"]))
    return out


def bootstrap_target(cfgs: dict, n_boot: int = 4000, seed: int = 0) -> dict:
    """Resample the samples (seeds) and find, per replicate, the cheapest configuration of each
    method that meets the target; uniform sweeps may pick any step and power in each replicate."""
    rng = np.random.default_rng(seed)
    n = len(next(iter(cfgs.values()))["field"])
    idx = rng.integers(0, n, size=(n_boot, n))
    cheapest = {}
    for method in ("gp", "uniform"):
        best = np.full(n_boot, np.inf)
        for key, d in cfgs.items():
            if key[0] == method:
                hit = (np.median(d["field"][idx], axis=1) <= TARGET_MG) & (d["ok"][idx].mean(axis=1) >= 0.9)
                best = np.where(hit, np.minimum(best, d["points"]), best)
        cheapest[method] = best
    both = np.isfinite(cheapest["gp"]) & np.isfinite(cheapest["uniform"])
    saving = 1 - cheapest["gp"][both] / cheapest["uniform"][both]
    return {
        "saving_median": float(np.median(saving)) if both.any() else None,
        "saving_ci95": [float(v) for v in np.percentile(saving, [2.5, 97.5])] if both.any() else None,
    }


def median_ci(values: np.ndarray, n_boot: int = 4000, seed: int = 1) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    meds = np.median(values[rng.integers(0, len(values), size=(n_boot, len(values)))], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def paired_gp_vs_ablation(cfgs: dict) -> list[dict]:
    """Same samples, same refinement budget: does the GP beat uniform windows? (Wilcoxon signed-rank)"""
    from scipy.stats import wilcoxon

    out = []
    for key, d in cfgs.items():
        if key[0] != "gp":
            continue
        ablation = cfgs[("uniform_windows", None, key[2], key[3])]
        gp, ab = np.minimum(d["field"], 1e4), np.minimum(ablation["field"], 1e4)  # failures rank last
        out.append({
            "points": d["points"], "gp_median_mg": float(np.median(d["field"])), "ablation_median_mg": float(np.median(ablation["field"])),
            "gp_better_fraction": float(np.mean(gp < ab)), "p_value": float(wilcoxon(gp, ab).pvalue),
        })
    return sorted(out, key=lambda r: r["points"])


def main() -> None:
    raw = json.loads(RESULTS.read_text())
    data = curves(raw["rows"])
    for theme in THEMES:
        figure(data, theme, ROOT / "docs" / "img" / f"benchmark_{theme}.png")
    cfgs = per_config(raw["rows"])
    gp_hit, uni_hit = cheapest_meeting_target(data["gp"]), cheapest_meeting_target(data["uniform"])
    boot = bootstrap_target(cfgs)
    summary = [f"Seeds {raw['seeds']} ({data['gp'][0]['n']} samples per configuration).", ""]
    if gp_hit and uni_hit:
        ci = boot["saving_ci95"]
        summary.append(
            f"Target (median |dB| <= {TARGET_MG:.0f} mG with >= 90% of runs finding all 8 lines within 0.5 MHz): "
            f"GP two-stage needs {gp_hit['n_points']} points, the best uniform sweep {uni_hit['n_points']} "
            f"({100 * (1 - gp_hit['n_points'] / uni_hit['n_points']):.0f}% fewer). Bootstrap over samples: saving "
            f"{100 * boot['saving_median']:.0f}% (95% CI {100 * ci[0]:.0f}-{100 * ci[1]:.0f}%)."
        )
    summary += ["", "Median field error with bootstrap 95% CI of the median:", ""]
    summary.append("| method | setting | points | median field error (95% CI) |")
    summary.append("|---|---|---|---|")
    for method in ("gp", "uniform_windows", "uniform"):
        for p in data[method]:
            key = (method, p["step"], p["refine"], p["power"])
            lo, hi = median_ci(cfgs[key]["field"])
            setting = f"{p['refine']} refine pts @ 0 dBm" if p["refine"] else f"{p['step']} MHz step @ {p['power']:+.0f} dBm"
            summary.append(f"| {LABELS[method]} | {setting} | {p['n_points']} | {p['field_med']:.0f} mG ({lo:.0f}-{hi:.0f}) |")
    summary += ["", "GP vs. the same budget spread uniformly over the same windows (paired by sample, Wilcoxon signed-rank):", ""]
    summary.append("| points | GP median | no-GP median | GP better in | p |")
    summary.append("|---|---|---|---|---|")
    for r in paired_gp_vs_ablation(cfgs):
        summary.append(f"| {r['points']} | {r['gp_median_mg']:.0f} mG | {r['ablation_median_mg']:.0f} mG | {100 * r['gp_better_fraction']:.0f}% | {r['p_value']:.3f} |")
    summary += ["", "All configurations:", "", table(data)]
    (ROOT / "benchmarks" / "results" / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
