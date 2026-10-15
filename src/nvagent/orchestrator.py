"""Closed-loop experiment: planner -> analyst -> reviewer, each a headless Claude Code agent.

    nvagent run --seed 3 --model sonnet --max-rounds 3

Every role runs as its own ``claude -p`` process with its own system prompt, a role-filtered
MCP server (so tool permissions are enforced, not just requested), the instrument-interlock
PreToolUse hook, and a JSON schema for its final answer; the analyst also gets the Stop-hook
quality gate. The orchestrator passes the structured outputs between roles, logs everything in
the run directory and finally scores the result against the hidden ground truth.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

from .mcp_server import ROLE_TOOLS
from .physics import canonical_field
from .runstate import RunState
from .simulator import Sample

ROLES = ("planner", "analyst", "reviewer")
DEFAULT_GOAL = (
    "Find every ODMR resonance of the NV ensemble between {lo:.0f} and {hi:.0f} MHz and determine the bias "
    "magnetic-field vector (crystal frame) and the zero-field splitting D with uncertainties, using as few "
    "measured points as reasonably possible."
)
# Variables that tie a process to an enclosing Claude Code session; a child agent must not inherit them.
_SESSION_ENV = (
    "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_HOST_SESSION_ID", "CLAUDE_CODE_SDK_HAS_HOST_AUTH_REFRESH",
    "CLAUDE_CODE_SESSION_ATTENDED", "CLAUDE_PID", "CLAUDE_AGENT_SDK_VERSION", "CLAUDE_EFFORT",
)


def _role_text(name: str) -> str:
    return resources.files("nvagent.roles").joinpath(name).read_text(encoding="utf-8")


def _command_path(path: str) -> str:
    """Quoted, forward-slash path that works in the shells Claude Code uses for hooks."""
    return '"' + str(Path(path)).replace("\\", "/") + '"'


@dataclass
class AgentConfig:
    claude_cmd: list[str]
    model: str = "sonnet"
    effort: str = "medium"
    max_budget_usd: float = 2.0
    timeout_s: float = 1800.0


def write_role_config(run_dir: Path, role: str) -> tuple[Path, Path]:
    """MCP config (role-filtered nvlab server) and settings (hooks) for one role."""
    config_dir = run_dir / "config"
    config_dir.mkdir(exist_ok=True)
    env = {"NVAGENT_RUN_DIR": str(run_dir), "NVAGENT_ROLE": role}
    mcp = {
        "mcpServers": {
            "nvlab": {"command": sys.executable, "args": ["-m", "nvagent.mcp_server", "--role", role, "--run-dir", str(run_dir)], "env": env}
        }
    }
    python = _command_path(sys.executable)
    hooks = {
        "PreToolUse": [
            {"matcher": "mcp__nvlab__.*", "hooks": [{"type": "command", "command": f"{python} -m nvagent.hooks.pre_tool_guard", "timeout": 60}]}
        ]
    }
    if role == "analyst":
        hooks["Stop"] = [{"hooks": [{"type": "command", "command": f"{python} -m nvagent.hooks.stop_quality_gate", "timeout": 120}]}]
    mcp_path, settings_path = config_dir / f"mcp.{role}.json", config_dir / f"settings.{role}.json"
    mcp_path.write_text(json.dumps(mcp, indent=1), encoding="utf-8")
    settings_path.write_text(json.dumps({"hooks": hooks}, indent=1), encoding="utf-8")
    return mcp_path, settings_path


def run_role(role: str, prompt: str, run_dir: Path, cfg: AgentConfig, tag: str) -> dict:
    """One headless agent invocation; returns the parsed `claude -p --output-format json` result."""
    mcp_path, settings_path = write_role_config(run_dir, role)
    schema = json.loads(_role_text("schemas.json"))[role]
    allowed = ",".join(f"mcp__nvlab__{fn.__name__}" for fn in ROLE_TOOLS[role])
    cmd = cfg.claude_cmd + [
        "-p",
        "--output-format", "json",
        "--model", cfg.model,
        "--effort", cfg.effort,
        "--system-prompt", _role_text(f"{role}.md"),
        "--json-schema", json.dumps(schema),
        "--mcp-config", str(mcp_path),
        "--strict-mcp-config",
        "--settings", str(settings_path),
        "--setting-sources", "project",
        "--tools", "",
        "--allowedTools", allowed,
        "--permission-mode", "dontAsk",
        "--max-budget-usd", str(cfg.max_budget_usd),
        "--no-session-persistence",
    ]
    env = {k: v for k, v in os.environ.items() if k not in _SESSION_ENV}
    env.update(NVAGENT_RUN_DIR=str(run_dir), NVAGENT_ROLE=role)
    state = RunState(run_dir)
    state.audit("agent_start", role=role, tag=tag, model=cfg.model)
    t0 = time.time()
    proc = subprocess.run(
        cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", env=env, cwd=run_dir, timeout=cfg.timeout_s
    )
    agents_dir = run_dir / "agents"
    agents_dir.mkdir(exist_ok=True)
    (agents_dir / f"{tag}.prompt.txt").write_text(prompt, encoding="utf-8")
    (agents_dir / f"{tag}.stdout.json").write_text(proc.stdout, encoding="utf-8")
    if proc.stderr.strip():
        (agents_dir / f"{tag}.stderr.txt").write_text(proc.stderr, encoding="utf-8")
    try:
        result = json.loads(proc.stdout)
    except json.JSONDecodeError:
        result = {"is_error": True, "result": proc.stdout[-2000:] or proc.stderr[-2000:]}
    output = result.get("structured_output")
    if output is None and isinstance(result.get("result"), str):
        output = _json_in_text(result["result"])
    summary = {
        "role": role,
        "tag": tag,
        "ok": proc.returncode == 0 and not result.get("is_error") and output is not None,
        "exit_code": proc.returncode,
        "output": output,
        "cost_usd": result.get("total_cost_usd"),
        "turns": result.get("num_turns"),
        "seconds": round(time.time() - t0, 1),
        "error": None if output is not None else (result.get("result") or proc.stderr[-2000:]),
    }
    state.audit("agent_end", **{k: v for k, v in summary.items() if k != "output"})
    return summary


def _json_in_text(text: str):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _planner_prompt(goal, state, rnd, max_rounds, review):
    cfg = state.config
    lines = [
        f"GOAL: {goal}",
        f"Frequency window: {cfg['window_mhz'][0]:.0f}-{cfg['window_mhz'][1]:.0f} MHz. Point budget: {cfg['budget_points']} points "
        f"in total; {state.points_used()} used so far.",
        f"Round {rnd} of at most {max_rounds}.",
    ]
    if review:
        lines.append("The reviewer did not accept the previous result: " + review.get("summary", ""))
        lines += [f"- {r}" for r in review.get("requests_for_planner", [])] or ["(no measurement requests for you)"]
    lines.append("Take the measurements needed now, then reply with your JSON summary.")
    return "\n".join(lines)


def _analyst_prompt(goal, rnd, planner_output, review):
    lines = [f"GOAL: {goal}", f"Round {rnd}. The planner reports:", json.dumps(planner_output, indent=1)]
    if review:
        lines.append("The reviewer's requests for you:")
        lines += [f"- {r}" for r in review.get("requests_for_analyst", [])] or ["(none)"]
    lines.append("Fit the data, determine the field vector, then reply with your JSON summary.")
    return "\n".join(lines)


def _reviewer_prompt(goal, rnd, max_rounds, analyst_output):
    last = "This is the last round: accept only a sound result, otherwise list what is wrong." if rnd == max_rounds else ""
    return "\n".join(
        [f"GOAL: {goal}", f"Round {rnd} of {max_rounds}. The analyst reports:", json.dumps(analyst_output, indent=1),
         "Review the result critically, then reply with your JSON verdict. " + last]
    )


def score(state: RunState, field_fit_id: str | None) -> dict:
    """Compare the agent's final field fit and lines with the hidden ground truth."""
    truth = state.truth()
    record = None
    if field_fit_id:
        try:
            record = state.fit(field_fit_id)
        except KeyError:
            record = None
    record = record or state.latest_fit("field")
    if record is None:
        return {"scored": False, "reason": "no field fit"}
    field = record["field"]
    b_true = canonical_field(truth.b_field)
    b_fit, b_err = np.array(field["b_canonical"]), np.array(field["b_err"])
    lines = np.sort(field["centers"])
    true_lines = truth.true_lines()
    out = {
        "scored": True,
        "field_fit_id": record["id"],
        "b_true_canonical_g": b_true.round(4).tolist(),
        "b_fit_canonical_g": b_fit.round(4).tolist(),
        "b_err_g": b_err.round(4).tolist(),
        "field_error_g": round(float(np.linalg.norm(b_fit - b_true)), 4),
        "field_error_sigma_max": round(float(np.max(np.abs(b_fit - b_true) / b_err)), 2),
        "D_error_mhz": round(float(field["D"] - truth.D), 4),
        "n_lines": int(len(lines)),
        "points_used": state.points_used(),
    }
    if len(lines) == 8:
        out["max_line_error_mhz"] = round(float(np.max(np.abs(lines - true_lines))), 4)
        out["rms_line_error_mhz"] = round(float(np.sqrt(np.mean((lines - true_lines) ** 2))), 4)
    return out


def run_experiment(
    seed: int,
    cfg: AgentConfig,
    runs_dir: Path = Path("runs"),
    max_rounds: int = 3,
    budget_points: int = 1200,
    window_mhz: tuple[float, float] = (2640.0, 3100.0),
    goal: str | None = None,
) -> dict:
    sample = Sample.random(seed, window=window_mhz)
    run_dir = (runs_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-seed{seed}").resolve()
    goal = goal or DEFAULT_GOAL.format(lo=window_mhz[0], hi=window_mhz[1])
    state = RunState.create(run_dir, sample, goal=goal, budget_points=budget_points, window_mhz=window_mhz)
    history, review, analyst_output = [], None, None
    for rnd in range(1, max_rounds + 1):
        planner = run_role("planner", _planner_prompt(goal, state, rnd, max_rounds, review), run_dir, cfg, f"r{rnd}-planner")
        history.append(planner)
        if not planner["ok"]:
            break
        analyst = run_role("analyst", _analyst_prompt(goal, rnd, planner["output"], review), run_dir, cfg, f"r{rnd}-analyst")
        history.append(analyst)
        if not analyst["ok"]:
            break
        analyst_output = analyst["output"]
        reviewer = run_role("reviewer", _reviewer_prompt(goal, rnd, max_rounds, analyst_output), run_dir, cfg, f"r{rnd}-reviewer")
        history.append(reviewer)
        if not reviewer["ok"]:
            break
        review = reviewer["output"]
        if review.get("verdict") == "accept":
            break
    report = {
        "seed": seed,
        "run_dir": str(run_dir),
        "model": cfg.model,
        "rounds": max(int(h["tag"][1:].split("-")[0]) for h in history) if history else 0,
        "accepted": bool(review and review.get("verdict") == "accept"),
        "final_answer": analyst_output,
        "final_review": review,
        "score": score(state, (analyst_output or {}).get("field_fit_id")),
        "agents": [{k: v for k, v in h.items() if k != "output"} for h in history],
        "total_cost_usd": round(sum(h["cost_usd"] or 0.0 for h in history), 4),
        "total_seconds": round(sum(h["seconds"] for h in history), 1),
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (run_dir / "report.md").write_text(render_report(report), encoding="utf-8")
    return report


def render_report(report: dict) -> str:
    s = report["score"]
    lines = [f"# Run seed {report['seed']} ({report['model']})", ""]
    lines.append(f"- accepted by reviewer: {report['accepted']} after {report['rounds']} round(s)")
    lines.append(f"- agent invocations: {len(report['agents'])}, cost ${report['total_cost_usd']:.2f}, wall time {report['total_seconds']:.0f} s")
    if s.get("scored"):
        lines += [
            f"- points measured: {s['points_used']}",
            f"- field: fit {s['b_fit_canonical_g']} +- {s['b_err_g']} G vs truth {s['b_true_canonical_g']} G "
            f"(|dB| = {1e3 * s['field_error_g']:.1f} mG, worst component {s['field_error_sigma_max']:.1f} sigma)",
            f"- D error: {1e3 * s['D_error_mhz']:.1f} kHz; lines found: {s['n_lines']}",
        ]
        if "max_line_error_mhz" in s:
            lines.append(f"- line centres: max error {1e3 * s['max_line_error_mhz']:.0f} kHz, rms {1e3 * s['rms_line_error_mhz']:.0f} kHz")
    else:
        lines.append(f"- not scored: {s.get('reason')}")
    if report.get("final_review"):
        lines += ["", "## Final review", "", report["final_review"].get("summary", "")]
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="nvagent", description="Run the closed-loop NV experiment with Claude Code agents.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run planner -> analyst -> reviewer on a random synthetic sample")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--model", default="sonnet")
    run.add_argument("--effort", default="medium")
    run.add_argument("--max-rounds", type=int, default=3)
    run.add_argument("--budget", type=int, default=1200, help="total measured points allowed")
    run.add_argument("--max-budget-usd", type=float, default=2.0, help="spend cap per agent invocation")
    run.add_argument("--runs-dir", type=Path, default=Path("runs"))
    run.add_argument("--claude", default=shutil.which("claude") or "claude", help="claude executable (or a test double)")
    args = parser.parse_args(argv)
    cfg = AgentConfig(claude_cmd=args.claude.split("|"), model=args.model, effort=args.effort, max_budget_usd=args.max_budget_usd)
    report = run_experiment(args.seed, cfg, args.runs_dir, args.max_rounds, args.budget)
    print(render_report(report))
    print(f"run directory: {report['run_dir']}")


if __name__ == "__main__":
    main()
