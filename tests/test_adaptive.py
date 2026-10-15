"""Gaussian-process surrogate and adaptive frequency selection."""
import numpy as np
import pytest

from nvagent.adaptive import AdaptiveConfig, OdmrSurrogate, survey_and_refine, suggest_frequencies
from nvagent.physics import canonical_field
from nvagent.pipeline import analyze_odmr
from nvagent.simulator import Sample, VirtualNVSetup


def lorentz_counts(f, centre=20.0, depth=0.02, fwhm=1.0, level=1e6):
    return level * (1 - depth / (1 + ((f - centre) / (fwhm / 2)) ** 2))


def test_surrogate_locates_a_known_dip_and_is_uncertain_away_from_data():
    # Realistic shot noise: 2e5 counts per point (0.22 %) and a 1 % dip, as in the simulator.
    # What the acquisition needs is where the dip is and where the GP is unsure; the depth
    # itself is shrunk because the kernel amplitude is learned from mostly flat data.
    f = np.arange(0.0, 40.0, 0.5)
    counts = np.random.default_rng(0).poisson(lorentz_counts(f, depth=0.01, fwhm=1.5, level=2e5))
    gp = OdmrSurrogate().fit(f, counts, dwell_ms=1.0)
    grid = np.arange(0.0, 40.0, 0.05)
    mu_grid, _ = gp.predict(grid)
    assert grid[np.argmin(mu_grid)] == pytest.approx(20.0, abs=0.5)
    mu, sd = gp.predict(np.array([20.0, 10.0, 70.0]))  # normalised: 0 = baseline
    assert -mu[0] > 3 * sd[0]
    assert mu[1] == pytest.approx(0.0, abs=0.0015)
    # 30 MHz beyond the data the posterior reverts exactly to the prior (kernel amplitude).
    assert sd[2] == pytest.approx(np.sqrt(gp.gp.kernel_.k1.constant_value), rel=1e-3)
    assert sd[2] > sd[0]


def test_acquisition_concentrates_points_where_a_dip_was_seen():
    f = np.arange(0.0, 40.01, 2.0)
    gp = OdmrSurrogate().fit(f, lorentz_counts(f, fwhm=1.5, level=2e5), dwell_ms=1.0)
    grid = np.arange(0.0, 40.0, 0.05)
    picks = suggest_frequencies(gp, grid, n=8, min_spacing=0.3)
    assert len(picks) == 8
    assert np.all((picks >= 0.0) & (picks <= 40.0))
    assert np.min(np.diff(np.sort(picks))) >= 0.3 - 1e-9
    assert np.sum(np.abs(picks - 20.0) < 3.0) >= 6


@pytest.fixture(scope="module")
def adaptive_run():
    sample = Sample.random(seed=7)
    setup = VirtualNVSetup(sample)
    rng = np.random.default_rng(0)
    cfg = AdaptiveConfig()

    def measure(freqs, mw_power_dbm):
        return setup.measure_odmr(freqs, mw_power_dbm, cfg.laser_mw, cfg.dwell_ms, rng=rng)

    return sample, cfg, survey_and_refine(measure, 2640.0, 3100.0, cfg)


def test_survey_and_refine_spends_exactly_its_point_budget(adaptive_run):
    _, cfg, run = adaptive_run
    n_survey = len(np.arange(2640.0, 3100.0 + cfg.survey_step / 2, cfg.survey_step))
    assert len(run.survey_freqs) == n_survey
    assert len(run.refine_freqs) == cfg.refine_points
    assert run.n_points == n_survey + cfg.refine_points


def test_refined_data_recover_lines_and_field(adaptive_run):
    sample, cfg, run = adaptive_run
    res = analyze_odmr(
        run.refine_freqs, run.refine_counts, cfg.dwell_ms, prior_centers=run.survey_centers, baseline=run.baseline
    )
    centres = np.array([ln.center for ln in res.fit.lines])
    assert len(centres) == 8
    assert np.max(np.abs(centres - sample.true_lines())) < 0.3
    assert np.linalg.norm(np.array(res.field.b_canonical) - canonical_field(sample.b_field)) < 0.1
