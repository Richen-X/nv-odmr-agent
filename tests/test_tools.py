"""Tool layer behind the MCP server: bookkeeping, interlocks, analysis, role separation."""
import asyncio
import json

import numpy as np
import pytest

from nvagent import tools
from nvagent.mcp_server import build_server
from nvagent.physics import canonical_field


def test_sweep_records_a_dataset_and_charges_the_budget(run_state):
    out = tools.measure_odmr_sweep(2640.0, 3100.0, 3.0, 10.0)
    assert out["n_points"] == 154
    assert out["points_used_total"] == 154
    assert out["points_remaining"] == 3000 - 154
    assert run_state.dataset(out["dataset_id"])["counts"].shape == (154,)


def test_server_refuses_out_of_range_requests_even_without_the_hook(run_state):
    with pytest.raises(tools.ToolError, match="mw_power_dbm"):
        tools.measure_odmr_sweep(2640.0, 3100.0, 3.0, 15.0)
    assert run_state.points_used() == 0


def test_repeated_measurements_get_fresh_noise(run_state):
    a = run_state.dataset(tools.measure_odmr_points([2650.0] * 50, -10.0)["dataset_id"])["counts"]
    b = run_state.dataset(tools.measure_odmr_points([2650.0] * 50, -10.0)["dataset_id"])["counts"]
    assert not np.array_equal(a, b)


def test_survey_refine_and_fit_recover_the_hidden_field(run_state):
    tools.measure_odmr_sweep(2640.0, 3100.0, 3.0, 10.0)
    cands = tools.find_odmr_lines(z_min=6.0)["candidates"]
    tools.run_adaptive_odmr([c["center_mhz"] for c in cands], 384, mw_power_dbm=0.0)
    fit = tools.fit_odmr()
    assert len(fit["lines"]) == 8
    field = tools.fit_field_vector(fit["fit_id"])
    truth = canonical_field(run_state.truth().b_field)
    assert np.linalg.norm(np.array(field["b_canonical_g"]) - truth) < 0.1


def test_agent_facing_outputs_never_reveal_ground_truth(run_state):
    tools.measure_odmr_sweep(2640.0, 3100.0, 3.0, 10.0)
    text = json.dumps(tools.get_run_summary()) + json.dumps(tools.get_instrument_info())
    truth = run_state.truth()
    assert "b_field" not in text
    assert f"{truth.b_field[0]:.3f}" not in text and f"{truth.D:.3f}" not in text


def tool_names(role):
    return {t.name for t in asyncio.run(build_server(role).list_tools())}


def test_each_role_sees_only_its_own_tools(run_state):
    planner, analyst, reviewer = tool_names("planner"), tool_names("analyst"), tool_names("reviewer")
    assert {"measure_odmr_sweep", "run_adaptive_odmr"} <= planner and "fit_odmr" not in planner
    assert {"fit_odmr", "fit_field_vector"} <= analyst and "measure_odmr_sweep" not in analyst
    assert reviewer.isdisjoint({"measure_odmr_sweep", "run_adaptive_odmr", "fit_odmr"})
    assert "check_field_consistency" in reviewer


def test_tool_call_through_mcp_returns_json_text(run_state):
    res = asyncio.run(build_server("planner").call_tool("get_instrument_info", {}))
    assert not res.is_error
    assert "mw_power_dbm" in json.loads(res.content[0].text)["limits"]
