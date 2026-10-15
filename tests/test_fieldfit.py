"""Field-vector inversion from ODMR line centres."""
import numpy as np
import pytest

from nvagent.fieldfit import fit_field
from nvagent.physics import electron_lines


def test_exact_lines_recover_canonical_field_and_D():
    lines = electron_lines((20.0, -35.0, 12.0), D=2871.3).ravel()
    fit = fit_field(lines, errors=np.full(8, 0.01))
    assert fit.b_canonical == pytest.approx([35.0, 20.0, 12.0], abs=1e-3)
    assert fit.D == pytest.approx(2871.3, abs=1e-3)
    assert fit.consistent


def test_noisy_lines_give_calibrated_uncertainties():
    rng = np.random.default_rng(0)
    truth = np.array([41.0, 18.0, 7.0])
    misses = []
    for _ in range(20):
        lines = electron_lines(truth, D=2870.0).ravel() + rng.normal(0, 0.05, 8)
        fit = fit_field(lines, errors=np.full(8, 0.05))
        misses.append(np.abs(np.asarray(fit.b_canonical) - truth) / np.asarray(fit.b_err))
    # With correct errors, ~99.7% of components should land within 3 sigma.
    assert np.mean(np.array(misses) < 3) > 0.9
    assert np.all(np.asarray(fit.b_err) < 0.1)


def test_degenerate_orientations_still_determine_field():
    # B in a {110} mirror plane: two orientations see the same field, leaving 6 distinct lines.
    lines = np.unique(np.round(electron_lines((30.0, 30.0, 10.0)).ravel(), 6))
    assert len(lines) == 6
    fit = fit_field(lines, errors=np.full(6, 0.02))
    assert fit.b_canonical == pytest.approx([30.0, 30.0, 10.0], abs=0.01)


def test_line_inconsistent_with_any_single_field_is_flagged():
    lines = electron_lines((20.0, -35.0, 12.0)).ravel()
    lines[3] += 3.0
    fit = fit_field(lines, errors=np.full(8, 0.05))
    assert not fit.consistent


def test_more_than_eight_lines_is_rejected():
    with pytest.raises(ValueError):
        fit_field(np.linspace(2700, 3000, 9))
