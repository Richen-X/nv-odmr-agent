# nv-odmr-agent

[![tests](https://github.com/Richen-X/nv-odmr-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Richen-X/nv-odmr-agent/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

An LLM lab agent that **plans, runs and analyses NV-centre magnetic-resonance measurements in a
closed loop**. Three [Claude Code](https://docs.claude.com/en/docs/claude-code) agents run headless,
each with its own role, tools and output schema: a **planner** that operates the instruments, an
**analyst** that fits the data and a **skeptical reviewer**. They work through an
[MCP](https://modelcontextprotocol.io) server that wraps a physics simulator, lmfit models and a
Gaussian-process sampler. Two deterministic hooks keep them honest: an instrument interlock the model
cannot override, and a quality gate the analyst cannot finish without passing.

The loop is developed against a simulator of NV-ensemble ODMR with realistic shot noise. That way
every component can be tested end to end, and scored against a known ground truth, before the
same tools are pointed at real hardware.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/benchmark_dark.png">
  <img alt="Field-vector error and reliability versus number of measured points for GP-guided two-stage acquisition, uniform sweeps and the no-GP ablation" src="docs/img/benchmark_light.png">
</picture>

**Result (100 held-out synthetic samples).** The target is a median field-vector error of 40 mG,
with at least 90 % of runs finding all eight lines.
- **Points to target:** the two-stage strategy with GP refinement reaches it with **538 measured points;
  the best uniform sweep needs 1315**. Uniform misses the target at 921 points even at its best MW
  power. That is **59 % fewer points**, with a bootstrap 95 % CI of 30–69 %.
- **Where the saving comes from:** mostly from the two-stage design, a coarse high-power survey followed
  by refinement windows. The GP surrogate lowers the median field error by a further 20–25 % at mid
  budgets, which is significant at 3 of 6 budgets.

Details: [Benchmark](#benchmark).

## How it works

```mermaid
flowchart LR
    O["orchestrator.py"] -->|"prompt + JSON schema"| P["planner<br/>claude -p"]
    O --> A["analyst<br/>claude -p"]
    O --> R["reviewer<br/>claude -p"]
    P -->|"instrument tools"| S[("nvlab MCP server<br/>(role-filtered)")]
    A -->|"fit tools"| S
    R -->|"review tools"| S
    G{{"PreToolUse hook:<br/>instrument interlock"}} -. "blocks unsafe calls" .-> P
    Q{{"Stop hook:<br/>quality gate"}} -. "refuses to stop" .-> A
    S --> SIM["physics simulator<br/>NV + 14N Hamiltonian"]
    S --> FIT["lmfit models +<br/>field inversion"]
    S --> GP["GP surrogate,<br/>adaptive sampling"]
    S --> RUN[("run directory:<br/>datasets, fits, plots,<br/>audit log, notebook")]
```

**Roles.** Each role is a separate `claude -p` process. It gets its own system prompt
([`roles/`](src/nvagent/roles)), a JSON schema for its final answer (`--json-schema`), and an MCP server
that exposes *only that role's tools*. The separation of duties is therefore enforced, not just
requested: the analyst cannot measure, and the reviewer can neither measure nor fit. Built-in tools are
disabled (`--tools ""`), permissions are pre-approved per tool (`--permission-mode dontAsk`), and every
invocation has a spending cap (`--max-budget-usd`). The orchestrator chains the roles through their
structured outputs and repeats planner → analyst → reviewer until the reviewer accepts or the round
limit is reached. It then scores the final answer against the hidden ground truth.

| role | tools | guardrail |
|---|---|---|
| planner | `get_instrument_info`, `measure_odmr_sweep`, `measure_odmr_points`, `find_odmr_lines`, `run_adaptive_odmr`, `measure_rabi`, `measure_t1` | PreToolUse interlock |
| analyst | `find_odmr_lines`, `fit_odmr`, `fit_field_vector`, `fit_rabi`, `fit_decay`, `view_fit_plot` | Stop-hook quality gate |
| reviewer | `get_fit_report`, `check_field_consistency`, `view_fit_plot` | read-only |

All roles also have `get_run_summary` and `add_notebook_entry`. `view_fit_plot` returns the fit as an
image, so the analyst and reviewer judge fits by eye as well as by numbers.

**Guardrails.**
- [`pre_tool_guard`](src/nvagent/hooks/pre_tool_guard.py) runs before every instrument call. It checks
  MW frequency and power, laser power, dwell, points per call and the run's total point budget against
  [`safety.py`](src/nvagent/safety.py). It blocks with exit code 2 and tells the model why. The MCP server
  applies the same checks again, so the interlock holds even without the hook.
- [`stop_quality_gate`](src/nvagent/hooks/stop_quality_gate.py) runs when the analyst tries to finish. It
  answers `{"decision": "block"}` until all of the following hold:
  - reduced χ² ≤ 2 and every line centre has an error ≤ 0.25 MHz;
  - no residual dip of 5σ or more remains unexplained (a missed line);
  - the fit includes all current data taken with the same MW and laser settings;
  - a single field vector explains every line, and no line it predicts inside the measured data is
    left unfitted.

  This turns "fit this data" into a verify-and-retry loop. After three refusals the stop is allowed and
  the failure recorded, so a run that cannot pass still ends and the reviewer sees why.
- Every interlock decision (allowed or refused), measurement, fit, notebook entry and quality-gate
  outcome is appended to `audit.jsonl` in the run directory. Each agent's prompt and raw final output
  are kept in `agents/`. Read-only tool calls are not logged.

**Physics simulator** ([`physics.py`](src/nvagent/physics.py), [`simulator.py`](src/nvagent/simulator.py)).
- **Hamiltonian:** for each of the four ⟨111⟩ orientations, the 9×9 electron + ¹⁴N Hamiltonian (D, E,
  Zeeman, A∥, A⊥, quadrupole, nuclear Zeeman) is diagonalised in an arbitrary bias field.
- **Transition strengths:** taken from the MW selection rules and the bright-state character of the
  eigenstates. The weak nuclear-spin-flip lines that off-axis fields allow come out on their own.
- **CW lines:** they saturate and power-broaden (depth ∝ s/(1+s), width ∝ √(1+s)). Saturation
  equalises the two populations, so the CW depth approaches half the pulsed π contrast. Optical pumping
  shifts saturation and broadens the intrinsic line.
- **Pulsed signals:** Rabi traces include off-resonant driving of every transition; T1 is a stretched
  exponential.
- **Noise:** counts are Poissonian, with optional technical noise and a baseline slope.
- **Validation:** checked against analytical limits in [`tests/test_physics.py`](tests/test_physics.py),
  including D ± γB for an axial field, the transverse two-level solution, the hyperfine triplet, the
  selection-rule sum rule and cubic symmetry.

**Analysis** ([`fitting.py`](src/nvagent/fitting.py), [`fieldfit.py`](src/nvagent/fieldfit.py), [`pipeline.py`](src/nvagent/pipeline.py)).
- **Line detection:** a matched filter picks the line-width class, and a sparse non-negative
  deconvolution into ¹⁴N-triplet templates finds the lines. Both work on any sampling pattern. The
  deconvolution explains all dips jointly, so two triplets about 3 hyperfine spacings apart are not
  mistaken for one line between them.
- **Line fits:** lmfit Lorentzian-triplet models with Poisson weights, so χ²_red ≈ 1 when the model is
  right. They come with residual diagnostics (runs test, local residual scans that flag missed lines).
- **Model selection:** backward elimination by Δχ² down to at most 8 lines, followed by a
  hyperfine-alias check.
- **Field inversion:** a multi-start least-squares fit of B and D to the line centres, with
  uncertainties from the Jacobian. B is reported in canonical form because an ensemble cannot
  distinguish the 48 cubic-symmetry-equivalent fields. If fewer than eight lines are given, the fit
  chooses which predicted lines stay unmatched, enumerating every order-preserving choice. Unmatched
  lines are free outside the measured coverage and penalised inside it. The fit then reports where the
  missing lines should be, so the planner can measure there.

**Adaptive sampling** ([`adaptive.py`](src/nvagent/adaptive.py)). No sampler can *find* a 1 MHz line
without covering the band at roughly 1 MHz resolution. The physics offers a shortcut, in two stages:
1. **Survey.** At high MW power the lines power-broaden to several MHz, so a 3 MHz survey (154 points)
   locates them all.
2. **Refinement.** At about 0 dBm, near the saturation (s ≈ 2) that maximises depth/width for locating
   line centres, a scikit-learn GP surrogate (Matérn-5/2, shot noise as known variance, length scale
   bounded by the physics to 0.3–3 MHz) chooses the next frequencies. In every batch it takes, in each
   window, the point that maximises σ·(predicted dip depth + βσ). That is an uncertainty-based
   acquisition weighted towards signal, and the per-window allocation keeps weak lines from being
   starved.

The LLM handles strategy (which powers, how many points, what to re-measure); the GP handles the
numerical search.

## Benchmark

[`benchmarks/adaptive_vs_uniform.py`](benchmarks/adaptive_vs_uniform.py) measures the same random
samples with every strategy. A sample has a random field direction, |B| = 25–70 G and eight resolvable
lines in 2640–3100 MHz. Every strategy uses the same 20 ms dwell per point and is scored against the
hidden truth.

- **Fair baseline:** uniform sweeps get their best fixed MW power, from 0 dBm up to the interlock's
  +10 dBm, at every step size.
- **No tuning on test data:** all tuning used seeds 1000–1019; the numbers below use seeds 0–99.
- **Target fixed in advance:** median field error ≤ 40 mG with ≥ 90 % of runs finding all eight
  lines within 0.5 MHz.

| method | points | runs with all 8 lines | median worst-line error | median field error (95 % CI) | p90 field error |
|---|---|---|---|---|---|
| GP two-stage | 410 | 85 % | 207 kHz | 37 mG (34–45) | 91 mG |
| **GP two-stage** | **538** | **94 %** | 194 kHz | **31 mG (25–35)** | 75 mG |
| GP two-stage | 922 | 94 % | 119 kHz | 22 mG (19–24) | 54 mG |
| two-stage, no GP (ablation) | 538 | 90 % | 203 kHz | 40 mG (35–45) | 78 mG |
| uniform, 0.5 MHz step, best power | 921 | 84 % | 266 kHz | 41 mG (38–48) | 179 mG |
| **uniform, 0.35 MHz step, best power** | **1315** | **95 %** | 195 kHz | **33 mG (29–39)** | 70 mG |
| uniform, 0.25 MHz step, best power | 1841 | 97 % | 154 kHz | 27 mG (23–32) | 72 mG |

**Reading the result.**
- **Points to target:** the bootstrap 95 % interval for the saving (538 vs 1315 points) is 30–69 %. The
  step sizes are discrete, but uniform sweeps miss the target at 921 points at every power, so the
  saving is at least 42 %.
- **The GP versus the same budget spread uniformly over the same windows** (paired by sample, Wilcoxon):
  the GP gives 37 vs 50 mG at 410 points, 31 vs 40 mG at 538 and 22 vs 28 mG at 922 (p = 0.04, 0.04,
  0.005). The differences at 250, 314 and 730 points are not significant. Most of the saving comes from
  the two-stage design itself.
- **The GP's weak spot:** at large budgets it has more gross failures than the densest uniform sweep,
  about 6 % against 3 %. These are mostly a line placed one hyperfine spacing off. In the agent loop,
  the reviewer and the quality gate exist to catch such failures.
- **The analysis is not identical across arms:** uniform sweeps are analysed blind (z ≥ 6). The
  two-stage arms reuse the survey's line positions and baseline (z ≥ 4), which is information the survey
  provides by design.

Full tables with intervals and paired tests are in [`benchmarks/results/summary.md`](benchmarks/results/summary.md),
and per-run results are in [`benchmarks/results/adaptive_vs_uniform.json`](benchmarks/results/adaptive_vs_uniform.json).

## Quickstart

```bash
git clone https://github.com/Richen-X/nv-odmr-agent && cd nv-odmr-agent
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                                                 # ~1 min; no API calls
```

Run the agent loop on a random synthetic sample (needs the [Claude Code CLI](https://docs.claude.com/en/docs/claude-code), logged in):

```bash
nvagent run --seed 3 --model sonnet --max-rounds 3 --budget 1200
```

Each run writes `runs/<timestamp>-seed<N>/`: datasets, fits and their plots, every agent's prompt and
raw output, `audit.jsonl`, the shared `notebook.md`, and `report.md` with the score against the
ground truth. To reproduce the benchmark (about 45 minutes on 18 cores):

```bash
python benchmarks/adaptive_vs_uniform.py --seeds 0-99
python benchmarks/plot_results.py
```

The MCP server also works on its own, for example from an interactive Claude Code session:
`python -m nvagent.mcp_server --role all --run-dir <run dir>`.

## Limitations

- **The instruments are simulated.** The tool layer is shaped for real drivers: replace the
  simulator calls in [`tools.py`](src/nvagent/tools.py) and keep the interlock, the gate and the
  analysis unchanged.
- **The simulator is simplified.** Lines are Lorentzian, E = 0, and there are no ¹³C neighbours and no
  slow drifts. Real spectra will add structure that the residual diagnostics and the reviewer should
  catch, and the thresholds will need retuning.
- **The benchmark compares acquisition strategies, not instrument overheads.** The LLM agents add
  decision latency and API cost on top of the measurement time.
- **The GP is a heuristic sampler.** A stationary GP learns its amplitude from all points, so it
  under-states dip depth in mostly flat data. The acquisition relies only on where dips are and how
  uncertain the GP is. A parametric Bayesian design over the line-shape model is the natural next step.

## Repository layout

```
src/nvagent/
  physics.py, simulator.py      NV Hamiltonian, ensemble transitions, virtual setup
  fitting.py, fieldfit.py       line finding, lmfit models, field-vector inversion
  pipeline.py, adaptive.py      guided analysis, GP-guided two-stage acquisition
  safety.py, qualitygate.py     interlock rules and quality gate (pure functions)
  tools.py, mcp_server.py       MCP tools and the role-filtered server
  hooks/                        PreToolUse interlock, Stop-hook quality gate
  roles/                        system prompts and JSON schemas of the three agents
  orchestrator.py, runstate.py  closed loop, run directory, scoring
benchmarks/                     acquisition benchmark, figures, results
tests/                          63 tests, including hooks run as subprocesses and a fake `claude`
docs/design.md                  design decisions and what the debugging taught
```

Developed with Claude Code, which also serves as the agent harness here.
