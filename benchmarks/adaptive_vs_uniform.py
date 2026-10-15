"""Benchmark: GP-guided two-stage acquisition vs uniform ODMR sweeps on synthetic samples.

Every method measures the same random NV-ensemble samples (random field direction,
|B| in 25-70 G, all eight lines resolvable) with the same 20 ms dwell per point, and every
dataset goes through the same analysis (``pipeline.analyze_odmr``). We report how close the
recovered line centres and field vector come to the hidden ground truth as a function of
the number of measured points.

    python benchmarks/adaptive_vs_uniform.py --seeds 0-29 --out benchmarks/results/test.json

Tuning used seeds 1000+; the reported numbers use seeds 0-29, which were not used for tuning.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

F_START, F_STOP = 2640.0, 3100.0
DWELL_MS, LASER_MW = 20.0, 2.0
SURVEY_STEP, SURVEY_POWER = 3.0, 10.0


def methods() -> list[dict]:
    out = []
    for power in (-4.0, 0.0, 4.0):  # uniform sweeps get their best fixed power per step
        for step in (0.25, 0.35, 0.5, 0.7, 1.0, 1.4):
            out.append({"method": "uniform", "step": step, "power": power})
    for n in (96, 160, 256, 384, 576, 768):
        out.append({"method": "gp", "refine": n, "power": 0.0})
        out.append({"method": "uniform_windows", "refine": n, "power": 0.0})  # ablation: no GP
    return out


def run_one(task: tuple[int, dict]) -> dict:
    warnings.simplefilter("ignore")
    from nvagent.adaptive import refine_odmr, windows_around
    from nvagent.fitting import estimate_baseline, find_lines
    from nvagent.physics import canonical_field
    from nvagent.pipeline import analyze_odmr
    from nvagent.simulator import Sample, VirtualNVSetup

    seed, cfg = task
    sample = Sample.random(seed)
    setup = VirtualNVSetup(sample)
    rng = np.random.default_rng([seed, zlib.crc32(json.dumps(cfg, sort_keys=True).encode())])
    measure = lambda f, p: setup.measure_odmr(np.asarray(f), p, LASER_MW, DWELL_MS, rng=rng)
    t0 = time.time()
    priors = baseline = None
    if cfg["method"] == "uniform":
        f = np.arange(F_START, F_STOP + 1e-9, cfg["step"])
        c = measure(f, cfg["power"])
        n_points = len(f)
    else:
        sf = np.arange(F_START, F_STOP + 1e-9, SURVEY_STEP)
        sc = measure(sf, SURVEY_POWER)
        priors = [x.center for x in find_lines(sf, sc, DWELL_MS, merge_width_factor=0.25, max_lines=32)]
        baseline = estimate_baseline(sf, sc, DWELL_MS)
        if not priors:
            return {"seed": seed, **cfg, "n_points": len(sf), "n_lines": 0, "max_line_err": None, "field_err": None}
        if cfg["method"] == "uniform_windows":
            wins = windows_around(priors, 8.0)
            total = sum(b - a for a, b in wins)
            f = np.concatenate([np.linspace(a, b, max(2, int(round(cfg["refine"] * (b - a) / total)))) for a, b in wins])
            c = measure(f, cfg["power"])
        else:
            res = refine_odmr(measure, priors, cfg["refine"], cfg["power"], DWELL_MS, baseline=baseline)
            f, c = res.freqs, res.counts
        n_points = len(sf) + len(f)
    out = {"seed": seed, **cfg, "n_points": int(n_points), "n_lines": 0, "max_line_err": None, "field_err": None}
    try:
        res = analyze_odmr(f, c, DWELL_MS, prior_centers=priors, baseline=baseline)
    except ValueError:
        return out
    centres = np.array([ln.center for ln in res.fit.lines])
    out["n_lines"] = int(len(centres))
    if len(centres) == 8 and res.field is not None:
        truth_b = canonical_field(sample.b_field)
        out["max_line_err"] = float(np.max(np.abs(centres - sample.true_lines())))
        out["field_err"] = float(np.linalg.norm(np.array(res.field.b_canonical) - truth_b))
        out["field_err_sigma"] = float(np.max(np.abs(np.array(res.field.b_canonical) - truth_b) / np.array(res.field.b_err)))
    out["seconds"] = round(time.time() - t0, 2)
    return out


def summarise(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        key = json.dumps({k: r[k] for k in ("method", "step", "refine", "power") if k in r}, sort_keys=True)
        groups.setdefault(key, []).append(r)
    table = []
    for key, rs in groups.items():
        line = np.array([r["max_line_err"] if r["max_line_err"] is not None else np.inf for r in rs])
        field = np.array([r["field_err"] if r["field_err"] is not None else np.inf for r in rs])
        table.append(
            {
                **json.loads(key),
                "n_points": int(np.median([r["n_points"] for r in rs])),
                "success": float(np.mean(line < 0.5)),
                "median_max_line_err_khz": float(np.median(line) * 1e3),
                "p90_max_line_err_khz": float(np.percentile(line, 90) * 1e3),
                "median_field_err_mg": float(np.median(field) * 1e3),
                "p90_field_err_mg": float(np.percentile(field, 90) * 1e3),
                "n_samples": len(rs),
            }
        )
    return sorted(table, key=lambda t: (t["method"], t.get("power", 0), t["n_points"]))


def parse_seeds(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-29")
    ap.add_argument("--out", default="benchmarks/results/adaptive_vs_uniform.json")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", default=None, help="comma-separated methods to run")
    args = ap.parse_args()
    cfgs = [c for c in methods() if args.only is None or c["method"] in args.only.split(",")]
    tasks = [(s, c) for s in parse_seeds(args.seeds) for c in cfgs]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_one, tasks, chunksize=2))
    table = summarise(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"seeds": args.seeds, "rows": rows, "summary": table}, indent=1))
    print(f"{len(tasks)} runs in {time.time() - t0:.0f} s -> {out}")
    print(f"{'method':16s} {'step/ref':>8s} {'dBm':>4s} {'points':>6s} {'ok':>5s} {'med|df|':>8s} {'p90|df|':>8s} {'med|dB|':>8s} {'p90|dB|':>8s}")
    for t in table:
        print(
            f"{t['method']:16s} {t.get('step', t.get('refine')):>8} {t['power']:>4.0f} {t['n_points']:>6d} {t['success']:>5.2f} "
            f"{t['median_max_line_err_khz']:>7.0f}k {t['p90_max_line_err_khz']:>7.0f}k {t['median_field_err_mg']:>7.0f}m {t['p90_field_err_mg']:>7.0f}m"
        )


if __name__ == "__main__":
    main()
