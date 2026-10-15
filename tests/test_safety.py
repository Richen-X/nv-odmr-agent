"""Instrument interlocks: pure checks shared by the PreToolUse hook and the server."""
from nvagent.safety import check_request, planned_points

SWEEP = {"start_mhz": 2640.0, "stop_mhz": 3100.0, "step_mhz": 3.0, "mw_power_dbm": 10.0}


def test_in_range_sweep_is_allowed():
    assert check_request("measure_odmr_sweep", SWEEP) == []


def test_mw_power_above_limit_is_blocked_with_a_reason():
    problems = check_request("measure_odmr_sweep", {**SWEEP, "mw_power_dbm": 15.0})
    assert len(problems) == 1
    assert "mw_power_dbm" in problems[0]


def test_frequency_outside_the_band_is_blocked():
    problems = check_request("measure_odmr_points", {"frequencies_mhz": [1500.0, 2870.0], "mw_power_dbm": 0.0})
    assert any("MHz" in p for p in problems)


def test_omitted_arguments_are_checked_at_their_defaults():
    # measure_rabi defaults to 2 mW of laser; a 1 mW cap must catch the omitted argument.
    args = {"frequency_mhz": 2870.0, "mw_power_dbm": 0.0}
    assert check_request("measure_rabi", args) == []
    assert check_request("measure_rabi", args, limits={"laser_power_mw": [0.05, 1.0]}) != []


def test_request_exceeding_the_run_budget_is_blocked():
    problems = check_request("measure_odmr_sweep", SWEEP, limits={"max_total_points": 3000}, points_used=2900)
    assert any("budget" in p for p in problems)


def test_malformed_arguments_are_rejected():
    assert check_request("measure_odmr_sweep", {**SWEEP, "mw_power_dbm": "high"}) != []
    assert check_request("measure_odmr_points", {"frequencies_mhz": [], "mw_power_dbm": 0.0}) != []


def test_sweep_point_count_includes_both_ends():
    # 2640, 2643, ..., 3099 is 154 points; 3100 itself is not on the 3 MHz grid.
    assert planned_points("measure_odmr_sweep", SWEEP) == 154
    assert planned_points("measure_odmr_sweep", {**SWEEP, "stop_mhz": 2652.0}) == 5
