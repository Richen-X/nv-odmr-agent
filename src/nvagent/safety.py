"""Deterministic instrument interlocks, shared by the PreToolUse hook and the MCP server.

The language model may propose any instrument call; these pure functions decide whether it
runs. They depend only on the request, the configured limits and the points already used.
"""
from __future__ import annotations

import math

DEFAULT_LIMITS: dict = {
    "mw_frequency_mhz": [2000.0, 3800.0],
    "mw_power_dbm": [-40.0, 10.0],
    "laser_power_mw": [0.05, 5.0],
    "dwell_ms": [0.1, 200.0],
    "rabi_tau_ns": [0.0, 10000.0],
    "t1_tau_us": [0.0, 100000.0],
    "max_points_per_call": 2000,
    "max_total_points": 3000,
}

# Defaults of the instrument tools; the hook sees only the arguments the model passed,
# so omitted arguments must be checked at the values the server will actually use.
TOOL_DEFAULTS: dict[str, dict] = {
    "measure_odmr_sweep": {"laser_power_mw": 2.0, "dwell_ms": 20.0},
    "measure_odmr_points": {"laser_power_mw": 2.0, "dwell_ms": 20.0},
    "run_adaptive_odmr": {"mw_power_dbm": 0.0, "laser_power_mw": 2.0, "dwell_ms": 20.0, "window_mhz": 8.0},
    "measure_rabi": {"tau_max_ns": 2000.0, "n_points": 101, "laser_power_mw": 2.0, "dwell_ms": 50.0},
    "measure_t1": {"tau_max_us": 20000.0, "n_points": 40, "laser_power_mw": 2.0, "dwell_ms": 100.0},
}
INSTRUMENT_TOOLS = tuple(TOOL_DEFAULTS)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def planned_points(tool: str, args: dict) -> int:
    """Number of points a (valid) instrument request will measure."""
    a = {**TOOL_DEFAULTS.get(tool, {}), **args}
    if tool == "measure_odmr_sweep":
        return int(math.floor((a["stop_mhz"] - a["start_mhz"]) / a["step_mhz"] + 1e-9)) + 1
    if tool == "measure_odmr_points":
        return len(a["frequencies_mhz"])
    return int(a["n_points"])


def check_request(tool: str, args: dict, limits: dict | None = None, points_used: int = 0) -> list[str]:
    """Reasons to refuse an instrument request; an empty list means it may run."""
    if tool not in TOOL_DEFAULTS:
        return [f"{tool} is not an instrument tool"]
    lim = {**DEFAULT_LIMITS, **(limits or {})}
    a = {**TOOL_DEFAULTS[tool], **(args or {})}
    problems: list[str] = []

    def number(name):
        value = a.get(name)
        if not _is_number(value):
            problems.append(f"{name} must be a finite number (got {value!r})")
            return None
        return float(value)

    def within(label, value, key, unit):
        lo, hi = lim[key]
        if value is not None and not lo <= value <= hi:
            problems.append(f"{label}={value:g} {unit} is outside the allowed range [{lo:g}, {hi:g}] {unit}")

    def number_list(name):
        value = a.get(name)
        if not isinstance(value, list) or not value or not all(_is_number(x) for x in value):
            problems.append(f"{name} must be a non-empty list of numbers")
            return []
        return [float(x) for x in value]

    def count(name, minimum):
        value = number(name)
        if value is not None and (value < minimum or value != int(value)):
            problems.append(f"{name} must be an integer >= {minimum}")

    freqs: list[float] = []
    if tool == "measure_odmr_sweep":
        start, stop, step = number("start_mhz"), number("stop_mhz"), number("step_mhz")
        if None not in (start, stop, step):
            if step <= 0 or stop < start:
                problems.append("need step_mhz > 0 and stop_mhz >= start_mhz")
            else:
                freqs = [start, stop]
    elif tool == "measure_odmr_points":
        freqs = number_list("frequencies_mhz")
    elif tool == "run_adaptive_odmr":
        centres, window = number_list("centers_mhz"), number("window_mhz")
        count("n_points", 1)
        if centres and window is not None:
            if window <= 0:
                problems.append("window_mhz must be positive")
            freqs = [min(centres) - window, max(centres) + window]
    elif tool == "measure_rabi":
        f0 = number("frequency_mhz")
        freqs = [] if f0 is None else [f0]
        within("tau_max_ns", number("tau_max_ns"), "rabi_tau_ns", "ns")
        count("n_points", 2)
    elif tool == "measure_t1":
        within("tau_max_us", number("tau_max_us"), "t1_tau_us", "us")
        count("n_points", 2)

    for f in freqs:
        within("MW frequency", f, "mw_frequency_mhz", "MHz")
    if tool != "measure_t1":
        within("mw_power_dbm", number("mw_power_dbm"), "mw_power_dbm", "dBm")
    within("laser_power_mw", number("laser_power_mw"), "laser_power_mw", "mW")
    within("dwell_ms", number("dwell_ms"), "dwell_ms", "ms")
    if not problems:
        n = planned_points(tool, a)
        if n > lim["max_points_per_call"]:
            problems.append(f"{n} points in one call exceeds max_points_per_call = {lim['max_points_per_call']}")
        if points_used + n > lim["max_total_points"]:
            problems.append(
                f"{n} more points would exceed the run budget ({points_used} of {lim['max_total_points']} already used)"
            )
    return problems
