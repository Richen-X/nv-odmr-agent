"""Virtual NV setup: line shapes, noise statistics and pulsed signals."""
import math

import numpy as np
import pytest

from nvagent.simulator import Sample, VirtualNVSetup

B111 = tuple(50.0 * np.array([1, 1, 1]) / math.sqrt(3))
F_LOW_111 = 2870.0 - 2.8025 * 50.0  # lower line of the [111]-aligned orientation


def single_line_sample(**kw):
    # No hyperfine, no slope, no technical noise: one Lorentzian per transition.
    return Sample(b_field=B111, hyperfine=False, linewidth=0.9, contrast=0.15, excess_noise=0.0, **kw)


def test_cw_line_depth_and_width_follow_saturation_model():
    setup = VirtualNVSetup(single_line_sample())
    # 0 dBm -> Rabi 1.0 MHz * sqrt(b_perp^2 = 2/3) = 0.8165 MHz; s = (0.8165/0.4)^2 = 4.1667.
    depth = 0.25 * 0.15 * (4.1667 / 5.1667)  # quarter of NVs per orientation, saturated fraction
    fwhm = 0.9 * math.sqrt(5.1667)  # power broadening
    sig = setup.odmr_signal(np.array([F_LOW_111, F_LOW_111 + fwhm / 2, 2650.0]), mw_power_dbm=0.0, laser_mw=0.0)
    assert sig[0] == pytest.approx(1 - depth, abs=2e-4)
    assert sig[1] == pytest.approx(1 - depth / 2, abs=2e-4)
    assert sig[2] == pytest.approx(1.0, abs=1e-4)


def test_counts_are_poisson_with_mean_set_by_laser_saturation():
    setup = VirtualNVSetup(single_line_sample())
    rng = np.random.default_rng(0)
    counts = setup.measure_odmr(np.full(4000, 2650.0), mw_power_dbm=-10, laser_mw=2.0, dwell_ms=1.0, rng=rng)
    # laser at saturation power -> rate = 2e7 * 1/2 = 1e7 counts/s -> 1e4 counts per 1 ms point.
    assert counts.mean() == pytest.approx(1e4, rel=0.005)
    assert counts.var() == pytest.approx(1e4, rel=0.1)


def test_same_rng_state_reproduces_measurement():
    setup = VirtualNVSetup(single_line_sample())
    f = np.linspace(2700, 2760, 50)
    a = setup.measure_odmr(f, 0.0, 1.0, 5.0, rng=np.random.default_rng(5))
    b = setup.measure_odmr(f, 0.0, 1.0, 5.0, rng=np.random.default_rng(5))
    assert np.array_equal(a, b)


def test_resonant_pi_pulse_fully_inverts_the_driven_orientation():
    setup = VirtualNVSetup(single_line_sample(rabi_decay_us=1e6))
    # Rabi frequency 0.8165 MHz -> pi time = 1 / (2 * 0.8165 MHz) = 612.4 ns, 2*pi at 1224.7 ns.
    sig = setup.rabi_signal(np.array([612.37, 1224.74]), freq_mhz=F_LOW_111, mw_power_dbm=0.0)
    # Full inversion of one orientation class (a quarter of the NVs) at pi; back to bright at 2*pi.
    # Off-resonant driving of the other orientations (~90 MHz away) contributes < 1e-4.
    assert sig == pytest.approx([1 - 0.25 * 0.15, 1.0], abs=1e-4)


def test_t1_signal_at_t1_has_decayed_by_one_over_e():
    setup = VirtualNVSetup(single_line_sample(t1_us=2000.0, t1_stretch=0.7))
    sig = setup.t1_signal(np.array([0.0, 2000.0]))
    # Polarised -> thermal loses 2/3 of the contrast: 1 - 0.15 * 2/3 * (1 - 1/e).
    assert sig == pytest.approx([1.0, 0.936788], abs=1e-6)


def test_random_samples_are_reproducible_and_resolvable():
    a, b = Sample.random(seed=11), Sample.random(seed=11)
    assert a == b
    lines = np.sort(a.true_lines())
    assert len(lines) == 8
    assert np.min(np.diff(lines)) >= 6.0
    assert lines.min() > 2640 and lines.max() < 3100


def test_sample_round_trips_through_dict():
    s = Sample.random(seed=3)
    assert Sample.from_dict(s.to_dict()) == s
