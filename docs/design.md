# Design notes

## Goal

A first, testable piece of an AI lab agent for optically addressable spins: the agent receives a
measurement goal ("find all ODMR resonances of this NV ensemble and determine the bias field"),
decides what to measure, operates the instruments, analyses the data and checks its own result,
under guardrails that do not depend on the model behaving well. Success means: the loop runs end
to end, its answers are scored against ground truth, and the adaptive strategy needs measurably
fewer points than the standard uniform sweep for the same accuracy.

## Architecture decisions

**Separate headless processes per role, not one agent with sub-agents.** Each role is its own
`claude -p` invocation with its own system prompt, JSON schema and a role-filtered MCP server.
Tool permissions are therefore structural: the analyst's server simply has no measurement tools.
A Python orchestrator passes the structured outputs along and keeps the loop logic (rounds,
acceptance, scoring) deterministic and testable with a stand-in `claude`.

**All state on disk.** Every agent invocation starts a fresh MCP server, so datasets, fits, plots,
counters and the audit log live in the run directory. Measurement noise is seeded by (sample seed,
measurement index), which makes runs reproducible without ever reusing noise. The ground truth
sits in `.truth/` and no agent-facing tool returns it (a test checks the summaries for it).

**Interlock twice.** The PreToolUse hook is the visible guardrail (it explains the refusal to the
model and audits the decision); the server applies the same pure function again, so a missing or
misconfigured hook cannot open a path around the limits. The hook fails closed on unreadable input.

**Quality gate as a Stop hook.** The analyst cannot end its turn until reduced chi-square, centre
uncertainties, residual structure and single-field consistency pass. Global reduced chi-square alone
is too weak: one missed triplet in a 1840-point sweep moves it from 1.0 to only about 1.08, so the
gate also checks a local residual scan for unexplained dips. A cap of three refusals keeps a run
that cannot pass from looping forever; the failure is recorded for the reviewer.

## Numerical decisions, and what the debugging taught

**Two-stage acquisition.** Narrow lines cannot be found without dense coverage, whatever the
sampler. Power broadening makes a coarse high-power survey sufficient for detection; the precision
work then happens at low power where lines are narrow.

**Maximum-likelihood GP hyperparameters over-smooth.** On sparse, noisy ODMR data the fitted
length scale ran to 7-20 MHz, far wider than the ~1.5 MHz lines, and the acquisition then wasted
points at window edges. The length scale is now bounded by the physics (0.3-3 MHz).

**Greedy global acquisition starves weak lines.** Taking the best points anywhere concentrated the
budget on the deepest dips. Allocating one acquisition maximum per window per batch fixed the
catastrophic misses; a larger exploration weight (beta = 4) and eight evenly spaced seed points per
window then reduced the tail of partial failures. These settings were tuned on development seeds
only.

**Baseline when every point is in a dip.** After refinement nearly every point lies in a dip or its
Lorentzian tail, so no point measures the baseline and any estimate that ignores dips is biased
low by ~0.3-0.5 %, enough to hide lines. Detection therefore fits the baseline jointly with the
detected lines, and refinement data borrow the baseline of the wide survey at the same laser power
(fluorescence off resonance does not depend on MW power).

**Greedy matching pursuit aliases hyperfine combs.** Two triplets about 3 hf apart form a nearly
regular six-line comb; a triplet centred between them matches three real components and wins the
greedy search. A sparse non-negative deconvolution into triplet templates explains all dips jointly
and recovers the true pair. Detection candidates are then carried into the joint lmfit fit, pruned
by backward elimination on delta chi-square (individual depth/error misleads when two candidates
split one line) and checked for one-spacing aliases.

**Know when the data cannot decide.** At low saturation and SNR the true and the aliased
decompositions of a close pair can differ by only delta chi-square ~ 3. Lowering the alias-check
threshold then moved correct lines onto aliases, so the threshold stays at a statistically
meaningful delta chi-square > 4 and refinement runs at s ~ 2, where the lines are deeper.

## Evaluation protocol

Development seeds 1000-1019 for every tuning decision; reported results on seeds 0-29. Uniform
sweeps get their best fixed MW power per step size. The target accuracy (median field error
<= 40 mG with >= 90 % of runs finding all lines within 0.5 MHz) was fixed before the held-out run.
The GP's contribution is isolated by an ablation that spends the same refinement budget uniformly
over the same windows.
