"""Instrument, analysis and review tools behind the MCP server, as plain testable functions.

Each function is one MCP tool: its signature becomes the JSON schema the model sees and its
docstring the description. State lives in the run directory (``runstate.RunState``); the
hidden sample is only ever touched to simulate measurements.
"""
from __future__ import annotations

import os

import numpy as np

from .adaptive import refine_odmr
from .fieldfit import fit_field
from .fitting import OdmrFit, TraceFit, estimate_baseline, find_lines
from .fitting import fit_decay as _fit_decay
from .fitting import fit_odmr as _fit_odmr
from .fitting import fit_rabi as _fit_rabi
from .pipeline import analyze_odmr
from .plotting import plot_odmr_fit, plot_trace_fit
from .qualitygate import gate_issues
from .runstate import RunState
from .safety import TOOL_DEFAULTS, check_request, planned_points
from .simulator import VirtualNVSetup

try:  # mcp >= 2
    from mcp.server.mcpserver import Image
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import Image
    from mcp.server.fastmcp.exceptions import ToolError

STATE: RunState | None = None
WIDE_SWEEP_MHZ = 100.0  # a sweep this wide has enough off-resonant points to set the baseline

PHYSICS_NOTES = (
    "NV ground state: zero-field splitting D ~ 2870 MHz (shifts with temperature and strain); "
    "electron gyromagnetic ratio 2.8025 MHz/G. Each of the four NV orientations gives a lower "
    "and an upper line, so up to 8 lines (fewer if orientations are degenerate). Every line is a "
    "14N hyperfine triplet split by ~2.16 MHz. CW lines saturate and power-broaden with MW power: "
    "at low power they are narrow (~1-1.5 MHz) but shallow, at high power deep but several MHz "
    "wide, which makes them easy to find on a coarse grid. More laser power gives more counts "
    "(saturating near 2 mW) and slightly broader lines. Counts are Poissonian, so noise falls as "
    "1/sqrt(dwell). The field vector is only defined up to the cubic symmetry of the crystal."
)


def init(run_dir) -> RunState:
    global STATE
    STATE = RunState(run_dir)
    return STATE


def _state() -> RunState:
    if STATE is None:
        raise ToolError("the nvlab server was started without a run directory")
    return STATE


def _r(value, digits: int = 3):
    return None if value is None else round(float(value), digits)


# --------------------------------------------------------------------------- helpers


def _guard(tool: str, args: dict) -> None:
    """Server-side copy of the interlock: holds even if the hook is not configured."""
    st = _state()
    problems = check_request(tool, args, st.limits(), st.points_used())
    if problems:
        st.audit("server_refused", tool=tool, args=args, reasons=problems)
        raise ToolError("Refused by the instrument interlock: " + "; ".join(problems))


def _setup() -> VirtualNVSetup:
    return VirtualNVSetup(_state().truth())


def _record(kind, mode, params, x, counts, dwell_ms, unit, extra=None) -> dict:
    st = _state()
    meta = st.save_dataset(kind, mode, params, x, counts, dwell_ms, unit)
    st.audit("measurement", dataset=meta["id"], kind=kind, mode=mode, n_points=meta["n_points"], params=params)
    used, budget = st.points_used(), st.config["budget_points"]
    out = {"dataset_id": meta["id"], "kind": kind, "n_points": meta["n_points"], "points_used_total": used, "points_remaining": budget - used}
    out.update(extra or {})
    return out


def _quicklook(f, counts, dwell_ms) -> dict:
    rate = counts / dwell_ms
    dip = 1 - rate / np.percentile(rate, 90)
    deepest = np.argsort(dip)[::-1][:5]
    return {
        "shot_noise_per_point_pct": _r(100 / np.sqrt(np.median(counts)), 3),
        "deepest_points": [{"freq_mhz": _r(f[i], 2), "dip_pct": _r(100 * dip[i], 2)} for i in deepest],
    }


def _odmr_datasets(dataset_ids):
    st = _state()
    odmr = [d for d in st.datasets() if d["kind"] == "odmr"]
    if not odmr:
        raise ToolError("no ODMR data yet")
    if dataset_ids:
        chosen = [d for d in odmr if d["id"] in dataset_ids]
        missing = sorted(set(dataset_ids) - {d["id"] for d in chosen})
        if missing:
            raise ToolError(f"unknown ODMR dataset(s): {', '.join(missing)}")
    else:
        last = odmr[-1]["params"]
        chosen = [
            d for d in odmr
            if d["params"]["mw_power_dbm"] == last["mw_power_dbm"] and d["params"]["laser_power_mw"] == last["laser_power_mw"]
        ]
    settings = {(d["params"]["mw_power_dbm"], d["params"]["laser_power_mw"]) for d in chosen}
    if len(settings) > 1:
        raise ToolError(
            "the selected datasets mix MW/laser settings "
            f"{sorted(settings)}; line widths and depths differ between them, so fit one setting at a time"
        )
    arrays = [st.dataset(d["id"]) for d in chosen]
    f = np.concatenate([a["x"] for a in arrays])
    counts = np.concatenate([a["counts"] for a in arrays])
    dwell = np.concatenate([np.full(len(a["x"]), a["dwell_ms"]) for a in arrays])
    return chosen, f, counts, dwell


def _survey_baseline(laser_power_mw: float):
    """Baseline (counts/ms vs MHz) from the latest wide sweep at this laser power, if any."""
    st = _state()
    for d in reversed(st.datasets()):
        p = d["params"]
        if d["kind"] == "odmr" and d["mode"] == "sweep" and p["laser_power_mw"] == laser_power_mw and d["x_range"][1] - d["x_range"][0] >= WIDE_SWEEP_MHZ:
            data = st.dataset(d["id"])
            return estimate_baseline(data["x"], data["counts"], data["dwell_ms"]), d["id"]
    return None, None


def _baseline_for(chosen):
    if any(d["mode"] == "sweep" and d["x_range"][1] - d["x_range"][0] >= WIDE_SWEEP_MHZ for d in chosen):
        return None, "joint estimate from the selected data"
    baseline, source = _survey_baseline(chosen[0]["params"]["laser_power_mw"])
    return baseline, source or "joint estimate (no wide sweep at this laser power)"


def _priors_for(chosen):
    centres = [c for d in chosen if d["mode"] == "adaptive" for c in d["params"]["centers_mhz"]]
    return sorted(centres) or None


def _odmr_summary(fit: OdmrFit) -> dict:
    diag = fit.diagnostics
    return {
        "lines": [
            {"center_mhz": _r(ln.center), "center_err_mhz": _r(ln.center_err, 4), "depth_pct": _r(100 * ln.depth, 3), "fwhm_mhz": _r(ln.fwhm, 3)}
            for ln in fit.lines
        ],
        "hyperfine": fit.hyperfine,
        "hf_split_mhz": _r(fit.hf_split, 4),
        "hf_split_err_mhz": _r(fit.hf_split_err, 4),
        "reduced_chi2": _r(fit.redchi, 3),
        "n_points": fit.ndata,
        "residuals": {
            "unexplained_dips": diag["unexplained_dips"],
            "model_too_deep": diag["model_too_deep"],
            "max_abs_resid_sigma": diag["max_abs_resid"],
            "runs_test_z": diag["runs_z"],
        },
        "converged": fit.success,
    }


def _field_summary(field: dict) -> dict:
    return {
        "b_canonical_g": [_r(v, 4) for v in field["b_canonical"]],
        "b_err_g": [_r(v, 4) for v in field["b_err"]],
        "b_magnitude_g": _r(field["b_magnitude"], 4),
        "b_magnitude_err_g": _r(field["b_magnitude_err"], 4),
        "angle_to_nearest_nv_axis_deg": _r(field["angle_to_nearest_axis_deg"], 2),
        "D_mhz": _r(field["D"], 4),
        "D_err_mhz": _r(field["D_err"], 4),
        "axial_transverse_g": [[_r(a, 2), _r(t, 2)] for a, t in field["axial_transverse_g"]],
        "line_residuals": [
            {"center_mhz": _r(c), "residual_khz": _r(1e3 * r, 1), "sigma": _r(s, 2)}
            for c, r, s in zip(field["centers"], field["residuals_mhz"], field["norm_residuals"])
        ],
        "reduced_chi2": _r(field["redchi"], 3),
        "consistent_with_one_field": field["consistent"],
        "message": field["message"],
        "predicted_lines_mhz": [_r(p) for p in field["predicted_lines"]],
        "note": "B is in the crystal frame, canonical up to the 48 cubic symmetry operations (|Bx| >= |By| >= |Bz| >= 0).",
    }


# ------------------------------------------------------------------ shared tools


def get_run_summary() -> dict:
    """Goal, point budget, every dataset and fit so far, quality-gate status and recent notebook entries."""
    st = _state()
    cfg = st.config
    used = st.points_used()
    datasets = [
        {
            "id": d["id"], "kind": d["kind"], "mode": d["mode"], "n_points": d["n_points"],
            "x_range": [_r(v, 2) for v in d["x_range"]], "x_unit": d["x_unit"],
            "mw_power_dbm": d["params"].get("mw_power_dbm"), "laser_power_mw": d["params"].get("laser_power_mw"), "dwell_ms": d["dwell_ms"],
        }
        for d in st.datasets()
    ]
    fits = []
    for f in st.fits():
        item = {"id": f["id"], "kind": f["kind"]}
        if f["kind"] == "odmr":
            item.update(datasets=f["dataset_ids"], n_lines=len(f["fit"]["lines"]), reduced_chi2=_r(f["fit"]["redchi"], 3))
        elif f["kind"] == "field":
            item.update(source_fit=f["source_fit"], b_magnitude_g=_r(f["field"]["b_magnitude"], 4), consistent=f["field"]["consistent"])
        else:
            item.update(dataset=f["dataset_id"], derived={k: _r(v, 4) for k, v in f["fit"]["derived"].items()})
        fits.append(item)
    return {
        "goal": cfg["goal"], "frequency_window_mhz": cfg["window_mhz"], "point_budget": cfg["budget_points"],
        "points_used": used, "points_remaining": cfg["budget_points"] - used,
        "datasets": datasets, "fits": fits, "quality_gate": st.gate_status(), "notebook_tail": st.notebook_tail(5),
    }


def add_notebook_entry(text: str) -> dict:
    """Append a note to the shared lab notebook (decisions, reasoning, open questions)."""
    _state().note(os.environ.get("NVAGENT_ROLE", "agent"), text)
    return {"ok": True}


# ------------------------------------------------------------------- instruments


def get_instrument_info() -> dict:
    """Instrument limits and defaults, the run's point budget and the physics needed to plan NV ODMR measurements."""
    st = _state()
    cfg = st.config
    return {
        "goal": cfg["goal"],
        "frequency_window_mhz": cfg["window_mhz"],
        "point_budget": cfg["budget_points"],
        "points_used": st.points_used(),
        "limits": st.limits(),
        "tool_defaults": TOOL_DEFAULTS,
        "setup": "NV ensemble in a (001) diamond chip on a coplanar waveguide. The MW field is roughly along [001], "
        "so all four NV orientations are driven about equally. Green laser excitation, photon counting.",
        "physics": PHYSICS_NOTES,
    }


def measure_odmr_sweep(
    start_mhz: float, stop_mhz: float, step_mhz: float, mw_power_dbm: float, laser_power_mw: float = 2.0, dwell_ms: float = 20.0
) -> dict:
    """CW ODMR on a uniform grid from start_mhz in steps of step_mhz up to stop_mhz. A coarse step at high MW
    power locates power-broadened lines cheaply; a fine step at low power resolves the hyperfine triplets.
    Every point costs one unit of the run's point budget."""
    args = {"start_mhz": start_mhz, "stop_mhz": stop_mhz, "step_mhz": step_mhz, "mw_power_dbm": mw_power_dbm, "laser_power_mw": laser_power_mw, "dwell_ms": dwell_ms}
    _guard("measure_odmr_sweep", args)
    f = start_mhz + step_mhz * np.arange(planned_points("measure_odmr_sweep", args))
    counts = _setup().measure_odmr(f, mw_power_dbm, laser_power_mw, dwell_ms, rng=_state().measurement_rng())
    return _record("odmr", "sweep", args, f, counts, dwell_ms, "MHz", _quicklook(f, counts, dwell_ms))


def measure_odmr_points(frequencies_mhz: list[float], mw_power_dbm: float, laser_power_mw: float = 2.0, dwell_ms: float = 20.0) -> dict:
    """CW ODMR at an explicit list of MW frequencies (MHz). Each point costs one unit of the point budget."""
    args = {"frequencies_mhz": list(frequencies_mhz), "mw_power_dbm": mw_power_dbm, "laser_power_mw": laser_power_mw, "dwell_ms": dwell_ms}
    _guard("measure_odmr_points", args)
    f = np.asarray(frequencies_mhz, dtype=float)
    counts = _setup().measure_odmr(f, mw_power_dbm, laser_power_mw, dwell_ms, rng=_state().measurement_rng())
    return _record("odmr", "points", args, f, counts, dwell_ms, "MHz", _quicklook(f, counts, dwell_ms))


def run_adaptive_odmr(
    centers_mhz: list[float],
    n_points: int,
    mw_power_dbm: float = 0.0,
    laser_power_mw: float = 2.0,
    dwell_ms: float = 20.0,
    window_mhz: float = 8.0,
) -> dict:
    """GP-guided refinement: spend n_points measurements in windows (centre +- window_mhz) around rough line
    positions, e.g. from a survey. In every batch a Gaussian-process surrogate of the spectrum picks, in each
    window, the frequency where its uncertainty weighted by the predicted dip depth is largest. Around 0 dBm
    the lines are narrow and well resolved; ~40-50 points per line give ~0.1 MHz centre precision."""
    args = {"centers_mhz": list(centers_mhz), "n_points": n_points, "mw_power_dbm": mw_power_dbm, "laser_power_mw": laser_power_mw, "dwell_ms": dwell_ms, "window_mhz": window_mhz}
    _guard("run_adaptive_odmr", args)
    st, setup = _state(), _setup()
    rng = st.measurement_rng()
    baseline, source = _survey_baseline(laser_power_mw)
    res = refine_odmr(
        lambda f, p: setup.measure_odmr(f, p, laser_power_mw, dwell_ms, rng=rng),
        centers_mhz, int(n_points), mw_power_dbm, dwell_ms, half_width=window_mhz, baseline=baseline,
    )
    windows = [
        {"from_mhz": _r(a, 2), "to_mhz": _r(b, 2), "n_points": int(np.sum((res.freqs >= a) & (res.freqs <= b)))}
        for a, b in res.windows
    ]
    extra = {"windows": windows, "gp_length_scale_mhz": _r(res.length_scales[-1]) if res.length_scales else None, "baseline_from": source}
    return _record("odmr", "adaptive", {**args, "baseline_from": source}, res.freqs, res.counts, dwell_ms, "MHz", extra)


def measure_rabi(
    frequency_mhz: float, mw_power_dbm: float, tau_max_ns: float = 2000.0, n_points: int = 101, laser_power_mw: float = 2.0, dwell_ms: float = 50.0
) -> dict:
    """Pulsed Rabi: fluorescence after a MW pulse of length 0..tau_max_ns (n_points, linear) at frequency_mhz.
    Drive on a fitted line centre; the oscillation gives the pi-pulse length."""
    args = {"frequency_mhz": frequency_mhz, "mw_power_dbm": mw_power_dbm, "tau_max_ns": tau_max_ns, "n_points": n_points, "laser_power_mw": laser_power_mw, "dwell_ms": dwell_ms}
    _guard("measure_rabi", args)
    taus = np.linspace(0.0, tau_max_ns, int(n_points))
    counts = _setup().measure_rabi(taus, frequency_mhz, mw_power_dbm, laser_power_mw, dwell_ms, rng=_state().measurement_rng())
    return _record("rabi", "rabi", args, taus, counts, dwell_ms, "ns")


def measure_t1(tau_max_us: float = 20000.0, n_points: int = 40, laser_power_mw: float = 2.0, dwell_ms: float = 100.0) -> dict:
    """All-optical T1: fluorescence after a dark time tau, log-spaced from 1 us to tau_max_us (n_points)."""
    args = {"tau_max_us": tau_max_us, "n_points": n_points, "laser_power_mw": laser_power_mw, "dwell_ms": dwell_ms}
    _guard("measure_t1", args)
    taus = np.geomspace(1.0, tau_max_us, int(n_points))
    counts = _setup().measure_t1(taus, laser_power_mw, dwell_ms, rng=_state().measurement_rng())
    return _record("t1", "t1", args, taus, counts, dwell_ms, "us")


# ---------------------------------------------------------------------- analysis


def find_odmr_lines(dataset_ids: list[str] | None = None, z_min: float = 6.0) -> dict:
    """Detect NV lines (14N hyperfine triplets) by sparse deconvolution of the dips. By default uses all ODMR
    data taken with the MW/laser settings of the latest ODMR dataset. Use z_min ~6 for a blind search over a
    wide band and ~4 inside windows that a survey already located."""
    chosen, f, counts, dwell = _odmr_datasets(dataset_ids)
    baseline, source = _baseline_for(chosen)
    cands = find_lines(f, counts, dwell, z_min=z_min, max_lines=16, baseline=baseline, prior_centers=_priors_for(chosen))
    return {
        "dataset_ids": [d["id"] for d in chosen],
        "n_points": int(len(f)),
        "baseline_from": source,
        "candidates": [{"center_mhz": _r(c.center), "z": _r(c.z, 1), "fwhm_mhz": _r(c.width, 2), "depth_pct": _r(100 * c.depth, 2)} for c in cands],
    }


def fit_odmr(
    dataset_ids: list[str] | None = None, centers_mhz: list[float] | None = None, hyperfine: str = "shared", width_mode: str = "per_line"
) -> dict:
    """lmfit model of ODMR data: 14N triplets of Lorentzians (hyperfine "shared" or "fixed") or single Lorentzians
    ("none"), on a sloped baseline, Poisson-weighted. Without centers_mhz the lines are found automatically
    (deconvolution, backward elimination to at most 8 lines, hyperfine-alias check). Returns centres with
    1-sigma errors, reduced chi2 and residual diagnostics; view the plot with view_fit_plot."""
    chosen, f, counts, dwell = _odmr_datasets(dataset_ids)
    baseline, source = _baseline_for(chosen)
    notes: list[str] = []
    try:
        if centers_mhz:
            fit = _fit_odmr(f, counts, dwell, centers_mhz, hyperfine=hyperfine, width_mode=width_mode)
        else:
            analysis = analyze_odmr(
                f, counts, dwell, prior_centers=_priors_for(chosen), baseline=baseline, hyperfine=hyperfine, width_mode=width_mode, with_field=False
            )
            fit, notes = analysis.fit, analysis.notes
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    st = _state()
    params = chosen[0]["params"]
    fit_id = st.save_fit(
        "odmr",
        {
            "dataset_ids": [d["id"] for d in chosen],
            "selection": {"mw_power_dbm": params["mw_power_dbm"], "laser_power_mw": params["laser_power_mw"]},
            "n_points": int(len(f)),
            "window_mhz": [float(f.min()), float(f.max())],
            "baseline_from": source,
            "fit": fit.to_dict(),
            "notes": notes,
        },
    )
    plot_odmr_fit(st.plot_path(fit_id), f, counts, dwell, fit, f"{fit_id}: {len(fit.lines)} lines, reduced chi2 = {fit.redchi:.2f}")
    st.audit("fit", fit_id=fit_id, kind="odmr", n_lines=len(fit.lines), redchi=fit.redchi)
    return {"fit_id": fit_id, "dataset_ids": [d["id"] for d in chosen], **_odmr_summary(fit), "notes": notes}


def fit_field_vector(odmr_fit_id: str | None = None) -> dict:
    """Invert the line centres of an ODMR fit (default: the latest) into the bias field B and the zero-field
    splitting D, with 1-sigma errors, per-line residuals and whether one field explains every line."""
    st = _state()
    source = st.fit(odmr_fit_id) if odmr_fit_id else st.latest_fit("odmr")
    if source is None or source["kind"] != "odmr":
        raise ToolError("no ODMR fit to invert: run fit_odmr first")
    lines = source["fit"]["lines"]
    if len(lines) < 3:
        raise ToolError(f"{source['id']} has only {len(lines)} line(s); at least 3 are needed for a field vector")
    try:
        field = fit_field(
            [ln["center"] for ln in lines], [ln["center_err"] or 0.5 for ln in lines], window=tuple(source["window_mhz"])
        ).to_dict()
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    fit_id = st.save_fit("field", {"source_fit": source["id"], "field": field})
    st.audit("fit", fit_id=fit_id, kind="field", b_magnitude=field["b_magnitude"], consistent=field["consistent"])
    return {"fit_id": fit_id, "source_fit": source["id"], **_field_summary(field)}


def _trace(dataset_id: str, kind: str) -> dict:
    try:
        data = _state().dataset(dataset_id)
    except KeyError as exc:
        raise ToolError(str(exc)) from exc
    if data["kind"] != kind:
        raise ToolError(f"{dataset_id} is a {data['kind']} dataset, not {kind}")
    return data


def fit_rabi(dataset_id: str) -> dict:
    """Damped-cosine fit of a Rabi dataset: Rabi frequency, pi-pulse length and contrast."""
    data = _trace(dataset_id, "rabi")
    fit = _fit_rabi(data["x"], data["counts"], data["dwell_ms"])
    st = _state()
    fit_id = st.save_fit("rabi", {"dataset_id": dataset_id, "fit": fit.to_dict()})
    plot_trace_fit(st.plot_path(fit_id), data["x"], data["counts"], data["dwell_ms"], fit, "MW pulse length (ns)", f"{fit_id}: Rabi")
    return {"fit_id": fit_id, "dataset_id": dataset_id, "reduced_chi2": _r(fit.redchi), **{k: _r(v, 4) for k, v in fit.derived.items()}}


def fit_decay(dataset_id: str, stretched: bool = True) -> dict:
    """(Stretched-)exponential fit c + A exp(-(t/T)^beta) of a T1 dataset."""
    data = _trace(dataset_id, "t1")
    fit = _fit_decay(data["x"], data["counts"], data["dwell_ms"], stretched=stretched)
    st = _state()
    fit_id = st.save_fit("decay", {"dataset_id": dataset_id, "fit": fit.to_dict()})
    plot_trace_fit(st.plot_path(fit_id), data["x"], data["counts"], data["dwell_ms"], fit, "dark time (us)", f"{fit_id}: T1", logx=True)
    return {"fit_id": fit_id, "dataset_id": dataset_id, "reduced_chi2": _r(fit.redchi), **{k: _r(v, 4) for k, v in fit.derived.items()}}


def view_fit_plot(fit_id: str) -> Image:
    """Image of a fit: data, model and (for ODMR) a zoom on every line. Use it to judge fit quality by eye."""
    path = _state().plot_path(fit_id)
    if not path.exists():
        raise ToolError(f"no plot for {fit_id}")
    return Image(path=str(path))


# ------------------------------------------------------------------------ review


def get_fit_report(fit_id: str) -> dict:
    """Full record of one fit: parameters, errors, residual diagnostics, notes and the data it used."""
    st = _state()
    try:
        record = st.fit(fit_id)
    except KeyError as exc:
        raise ToolError(str(exc)) from exc
    if record["kind"] == "odmr":
        fit = OdmrFit.from_dict(record["fit"])
        return {
            "fit_id": fit_id, "kind": "odmr", "dataset_ids": record["dataset_ids"], "selection": record["selection"],
            "baseline_from": record["baseline_from"], **_odmr_summary(fit),
            "line_local_reduced_chi2": fit.diagnostics["line_local_redchi"], "notes": record["notes"],
        }
    if record["kind"] == "field":
        return {"fit_id": fit_id, "kind": "field", "source_fit": record["source_fit"], **_field_summary(record["field"])}
    fit = TraceFit.from_dict(record["fit"])
    return {"fit_id": fit_id, "kind": record["kind"], "dataset_id": record["dataset_id"], "params": fit.params, "reduced_chi2": fit.redchi, "derived": fit.derived}


def check_field_consistency(field_fit_id: str | None = None) -> dict:
    """Skeptical checks of a field fit (default: the latest): lines that miss the single-field prediction,
    predicted lines inside the measured window with no fitted line, plausibility of D and of the hyperfine
    spacing, and the analyst's quality-gate status."""
    st = _state()
    record = st.fit(field_fit_id) if field_fit_id else st.latest_fit("field")
    if record is None or record["kind"] != "field":
        raise ToolError("no field fit to check: the analyst must run fit_field_vector first")
    field, source = record["field"], st.fit(record["source_fit"])
    fitted = np.array(field["centers"])
    lo, hi = source["window_mhz"]
    unmatched = [p for p in field["predicted_lines"] if lo <= p <= hi and np.min(np.abs(fitted - p)) > 1.5]
    hf = source["fit"]["hf_split"]
    return {
        "field_fit_id": record["id"],
        "source_fit": source["id"],
        "n_lines": int(len(fitted)),
        "lines_off_prediction": [
            {"center_mhz": _r(c), "sigma": _r(s, 2)} for c, s in zip(field["centers"], field["norm_residuals"]) if abs(s) > 3
        ],
        "predicted_but_not_fitted_mhz": [_r(p) for p in unmatched],
        "D_mhz": _r(field["D"], 4),
        "D_plausible": 2860.0 <= field["D"] <= 2885.0,
        "hf_split_mhz": _r(hf, 4),
        "hf_split_plausible": source["fit"]["hyperfine"] == "none" or 2.0 <= hf <= 2.3,
        "reduced_chi2_odmr": _r(source["fit"]["redchi"], 3),
        "reduced_chi2_field": _r(field["redchi"], 3),
        "quality_gate_issues": gate_issues(st),
        "quality_gate_status": st.gate_status(),
    }
