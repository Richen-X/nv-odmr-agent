"""Line finding and lmfit models recover simulator ground truth."""
import math

import numpy as np
import pytest

from nvagent.fitting import find_lines, fit_decay, fit_odmr, fit_rabi
from nvagent.simulator import Sample, VirtualNVSetup


@pytest.fixture(scope="module")
def sweep():
    sample = Sample.random(seed=4)
    setup = VirtualNVSetup(sample)
    f = np.arange(2640.0, 3100.0, 0.25)
    counts = setup.measure_odmr(f, mw_power_dbm=-4.0, laser_mw=2.0, dwell_ms=20.0, rng=np.random.default_rng(1))
    return sample, f, counts


def nearest_errors(found, truth):
    found = np.asarray(found)
    return np.array([np.min(np.abs(found - t)) for t in truth])


def test_find_lines_locates_every_hyperfine_triplet(sweep):
    sample, f, counts = sweep
    cands = find_lines(f, counts, dwell_ms=20.0)
    assert len(cands) == 8
    assert np.all(nearest_errors([c.center for c in cands], sample.true_lines()) < 0.5)


def test_find_lines_works_when_every_point_sits_inside_a_dip():
    # Adaptive refinement measures only near lines, so no point is off-resonance and the
    # baseline must be estimated jointly with the dips.
    sample = Sample.random(seed=4)
    setup = VirtualNVSetup(sample)
    f = np.concatenate([np.arange(c - 5.0, c + 5.0, 0.2) for c in sample.true_lines()])
    counts = setup.measure_odmr(f, mw_power_dbm=-4.0, laser_mw=2.0, dwell_ms=20.0, rng=np.random.default_rng(6))
    cands = find_lines(f, counts, dwell_ms=20.0)
    assert len(cands) == 8
    assert np.all(nearest_errors([c.center for c in cands], sample.true_lines()) < 0.5)


def test_fit_odmr_recovers_centres_within_quoted_uncertainty(sweep):
    sample, f, counts = sweep
    truth = sample.true_lines()
    fit = fit_odmr(f, counts, dwell_ms=20.0, centers=truth + 0.4)
    centres = np.array([ln.center for ln in fit.lines])
    errs = np.array([ln.center_err for ln in fit.lines])
    assert np.all(errs < 0.2)
    assert np.all(np.abs(centres - truth) < 4 * errs + 0.01)
    assert 0.8 < fit.redchi < 1.3
    assert fit.hf_split == pytest.approx(2.14, abs=0.05)


def test_fit_with_a_missing_line_reports_an_unexplained_dip(sweep):
    sample, f, counts = sweep
    truth = sample.true_lines()
    fit = fit_odmr(f, counts, dwell_ms=20.0, centers=np.delete(truth, 5))
    # One missing triplet is ~25 of 1840 points, so the global reduced chi2 barely moves
    # (~1.08); the local residual scan is what has to catch it.
    dips = [d["freq_mhz"] for d in fit.diagnostics["unexplained_dips"]]
    assert any(abs(d - truth[5]) < 3.0 for d in dips)


def test_fit_rabi_recovers_pi_time():
    b = tuple(50.0 * np.array([1, 1, 1]) / math.sqrt(3))
    setup = VirtualNVSetup(Sample(b_field=b, hyperfine=False, rabi_decay_us=2.0))
    taus = np.arange(0.0, 3000.0, 20.0)
    counts = setup.measure_rabi(taus, 2870.0 - 2.8025 * 50.0, 0.0, 2.0, 50.0, rng=np.random.default_rng(2))
    fit = fit_rabi(taus, counts, dwell_ms=50.0)
    # 0 dBm drives the [111] line at 0.8165 MHz -> pi time 612.4 ns.
    assert fit.derived["pi_time_ns"] == pytest.approx(612.4, rel=0.02)


def test_fit_decay_recovers_stretched_t1():
    setup = VirtualNVSetup(Sample(b_field=(10.0, 20.0, 30.0), t1_us=2000.0, t1_stretch=0.7))
    taus = np.geomspace(1.0, 20000.0, 40)
    counts = setup.measure_t1(taus, laser_mw=2.0, dwell_ms=100.0, rng=np.random.default_rng(3))
    fit = fit_decay(taus, counts, dwell_ms=100.0, stretched=True)
    assert fit.derived["T_us"] == pytest.approx(2000.0, rel=0.05)
    assert fit.derived["beta"] == pytest.approx(0.7, abs=0.05)
