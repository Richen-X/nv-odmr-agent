"""End-to-end ODMR analysis: detection -> lmfit -> field vector."""
import numpy as np

from nvagent.adaptive import windows_around
from nvagent.fitting import estimate_baseline
from nvagent.physics import canonical_field
from nvagent.pipeline import analyze_odmr
from nvagent.simulator import Sample, VirtualNVSetup


def test_uniform_sweep_analysis_recovers_lines_and_field():
    sample = Sample.random(seed=4)
    setup = VirtualNVSetup(sample)
    f = np.arange(2640.0, 3100.0, 0.25)
    counts = setup.measure_odmr(f, -4.0, 2.0, 20.0, rng=np.random.default_rng(1))
    res = analyze_odmr(f, counts, 20.0)
    centres = np.array([ln.center for ln in res.fit.lines])
    assert len(centres) == 8
    assert np.max(np.abs(centres - sample.true_lines())) < 0.2
    assert np.linalg.norm(np.array(res.field.b_canonical) - canonical_field(sample.b_field)) < 0.1


def survey_baseline(setup, seed):
    """Baseline from a coarse full-band sweep, as the adaptive workflow measures first."""
    f = np.arange(2640.0, 3100.0, 3.0)
    return estimate_baseline(f, setup.measure_odmr(f, 10.0, 2.0, 20.0, rng=np.random.default_rng(seed)), 20.0)


def test_sparse_window_data_use_prior_centres_from_a_survey():
    # 32 points per window is too little for a blind z >= 6 search over the whole band,
    # but enough to refine lines whose rough positions a survey already gave.
    sample = Sample.random(seed=1000)
    setup = VirtualNVSetup(sample)
    rough = sample.true_lines() + np.random.default_rng(0).uniform(-1.0, 1.0, 8)
    f = np.concatenate([np.linspace(a, b, 32) for a, b in windows_around(rough, 7.0)])
    counts = setup.measure_odmr(f, -4.0, 2.0, 20.0, rng=np.random.default_rng(1))
    res = analyze_odmr(f, counts, 20.0, prior_centers=rough, baseline=survey_baseline(setup, 9))
    centres = np.array([ln.center for ln in res.fit.lines])
    assert len(centres) == 8
    assert np.max(np.abs(centres - sample.true_lines())) < 0.5


def test_merged_survey_candidate_is_split_by_refinement_data():
    # Two lines 6-8 MHz apart look like one dip in a power-broadened survey; refinement
    # data at the slope-optimal saturation (s ~ 2, about 0 dBm here) must resolve both,
    # including a pair ~3 hf apart whose hyperfine combs interleave.
    sample = Sample.random(seed=7)
    setup = VirtualNVSetup(sample)
    truth = sample.true_lines()
    merged = 0.5 * (truth[0::2] + truth[1::2])  # one prior per pair
    f = np.concatenate([np.arange(a, b, 0.2) for a, b in windows_around(merged, 8.0)])
    counts = setup.measure_odmr(f, 0.0, 2.0, 20.0, rng=np.random.default_rng(2))
    res = analyze_odmr(f, counts, 20.0, prior_centers=merged, baseline=survey_baseline(setup, 9))
    centres = np.array([ln.center for ln in res.fit.lines])
    assert len(centres) == 8
    assert np.max(np.abs(centres - truth)) < 0.3
