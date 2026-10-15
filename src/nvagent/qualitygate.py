"""Quality gate enforced by the analyst's Stop hook and reported to the reviewer."""
from __future__ import annotations

from .runstate import RunState

DEFAULT_GATE = {
    "redchi_max": 2.0,  # Poisson-weighted fits of a correct model give ~1
    "center_err_max_mhz": 0.25,
    "unexplained_dip_sigma": 5.0,  # residual dips at least this significant mean a missed line
    "field_max_norm_resid": 4.0,  # every line within 4 sigma of the single-field prediction
    "max_blocks": 3,  # after this many refusals the stop is allowed and the failure recorded
}


def gate_settings(state: RunState) -> dict:
    return {**DEFAULT_GATE, **state.config.get("quality_gate", {})}


def _same_settings(dataset: dict, selection: dict) -> bool:
    p = dataset["params"]
    return p.get("mw_power_dbm") == selection["mw_power_dbm"] and p.get("laser_power_mw") == selection["laser_power_mw"]


def gate_issues(state: RunState) -> list[str]:
    """Everything that stops the latest ODMR + field fits from being acceptable."""
    g = gate_settings(state)
    record = state.latest_fit("odmr")
    if record is None:
        return ["No ODMR fit yet: run fit_odmr on the refinement data, then fit_field_vector."]
    issues = []
    current = {d["id"] for d in state.datasets() if d["kind"] == "odmr" and _same_settings(d, record["selection"])}
    new = sorted(current - set(record["dataset_ids"]))
    if new:
        issues.append(f"{record['id']} predates new data taken with the same settings ({', '.join(new)}): refit.")
    fit = record["fit"]
    if fit["redchi"] > g["redchi_max"]:
        issues.append(f"Reduced chi2 of {record['id']} is {fit['redchi']:.2f} > {g['redchi_max']}: model and data disagree.")
    loose = [ln for ln in fit["lines"] if ln["center_err"] is None or ln["center_err"] > g["center_err_max_mhz"]]
    if loose:
        issues.append(
            f"{len(loose)} line centre(s) lack an uncertainty or exceed {g['center_err_max_mhz']} MHz: "
            + ", ".join(f"{ln['center']:.2f}" for ln in loose)
            + " MHz."
        )
    dips = [d for d in fit["diagnostics"].get("unexplained_dips", []) if abs(d["significance"]) >= g["unexplained_dip_sigma"]]
    if dips:
        issues.append(
            "Unexplained dip(s) in the residuals at "
            + ", ".join(f"{d['freq_mhz']:.1f} MHz ({d['significance']:+.0f} sigma)" for d in dips)
            + ": a line may be missing."
        )
    if len(fit["lines"]) > 8:
        issues.append(f"{len(fit['lines'])} lines fitted, but an NV ensemble has at most 8.")
    field = state.latest_fit("field")
    if field is None or field.get("source_fit") != record["id"]:
        issues.append(f"Run fit_field_vector on {record['id']}.")
    elif max(abs(r) for r in field["field"]["norm_residuals"]) > g["field_max_norm_resid"]:
        issues.append("The lines are not explained by one field vector: " + field["field"]["message"] + ".")
    elif field["field"].get("unmatched_in_coverage"):
        issues.append(
            "The field fit predicts line(s) inside the measured data with no fitted line: "
            + ", ".join(f"{v:.1f}" for v in field["field"]["unmatched_in_coverage"])
            + " MHz. Fit them (centers_mhz) or explain why they are absent."
        )
    return issues
