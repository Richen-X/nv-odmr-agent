"""Physics checks against hand-derived limits of the NV ground-state Hamiltonian."""
import math

import numpy as np
import pytest

from nvagent.physics import canonical_field, electron_lines, transitions

D = 2870.0
GAMMA = 2.8025  # MHz/G


def test_zero_field_lines_sit_at_zero_field_splitting():
    lines = electron_lines((0.0, 0.0, 0.0), D=D)
    assert lines.shape == (4, 2)
    assert np.allclose(lines, D, atol=1e-9)


def test_field_along_111_splits_aligned_orientation_by_twice_gamma_b():
    b = 50.0 * np.array([1, 1, 1]) / math.sqrt(3)
    lines = electron_lines(b, D=D)
    # Orientation 0 is the [111] axis: purely axial field, exact result D -/+ gamma*B.
    assert lines[0] == pytest.approx([D - GAMMA * 50.0, D + GAMMA * 50.0], abs=1e-6)
    # The other three axes are related by the C3 rotation about [111], so they are degenerate.
    assert np.allclose(lines[1], lines[2], atol=1e-9)
    assert np.allclose(lines[1], lines[3], atol=1e-9)


def test_purely_transverse_field_matches_two_level_block_solution():
    # 100 G along [1,-1,0] is perpendicular to the [111] axis (orientation 0).
    b = 100.0 * np.array([1, -1, 0]) / math.sqrt(2)
    lines = electron_lines(b, D=D)
    # H = D Sz^2 + g B Sx: (|+1>-|-1>)/sqrt2 stays at D; (|+1>+|-1>)/sqrt2 couples to |0> with g*B.
    r = math.sqrt(D**2 / 4 + (GAMMA * 100.0) ** 2)
    assert lines[0] == pytest.approx([D / 2 + r, 2 * r], abs=1e-6)


def test_14n_hyperfine_splits_each_line_into_triplet():
    b = 30.0 * np.array([1, 1, 1]) / math.sqrt(3)
    trs = [t for t in transitions(b, D=D, hyperfine=True) if t.orientation == 0]
    freqs = np.sort([t.freq for t in trs])
    assert len(freqs) == 6
    low, high = freqs[:3], freqs[3:]
    for triplet, centre in ((low, D - GAMMA * 30.0), (high, D + GAMMA * 30.0)):
        assert np.diff(triplet) == pytest.approx([2.14, 2.14], abs=0.02)
        assert triplet[1] == pytest.approx(centre, abs=0.02)
    # Unpolarised nuclear spin: each component carries a third of the optical contrast.
    assert [t.contrast for t in trs] == pytest.approx([1 / 3] * 6, abs=0.01)
    # MW along [001] makes angle arccos(1/sqrt3) with every NV axis: b_perp^2 = 2/3.
    assert [t.strength for t in trs] == pytest.approx([2 / 3] * 6, abs=0.01)


def test_mw_field_along_an_nv_axis_cannot_drive_that_orientation():
    b = (10.0, 4.0, 1.0)
    trs = transitions(b, D=D, hyperfine=False, mw_dir=(1, 1, 1))
    assert {t.orientation for t in trs} == {1, 2, 3}
    # Transverse fields mix m_s=+-1 and redistribute strength between the two lines, but the
    # sum rule holds: total strength per orientation = 2 * b_perp^2, with b_perp^2 = 1 - 1/9.
    for o in (1, 2, 3):
        assert sum(t.strength for t in trs if t.orientation == o) == pytest.approx(16 / 9, abs=0.01)


def test_generic_field_gives_eight_electron_transitions():
    trs = transitions((20.0, -35.0, 12.0), D=D, hyperfine=False)
    assert len(trs) == 8


def test_spectrum_is_invariant_under_cubic_symmetry_operations():
    rng = np.random.default_rng(1)
    b = rng.normal(size=3) * 40
    # A signed permutation of the cubic axes maps the set of <111> axes onto itself.
    b_sym = np.array([-b[2], b[0], -b[1]])
    assert np.allclose(np.sort(electron_lines(b).ravel()), np.sort(electron_lines(b_sym).ravel()), atol=1e-9)


def test_canonical_field_sorts_absolute_components():
    assert canonical_field((3.0, -5.0, 1.0)) == pytest.approx([5.0, 3.0, 1.0])


def test_electron_lines_rejects_fields_beyond_low_field_model():
    with pytest.raises(ValueError):
        electron_lines((2000.0, 0.0, 0.0))
