"""NV-centre ground-state spin physics for ensembles in single-crystal diamond.

Conventions
-----------
* Frequencies are in MHz, magnetic fields in gauss.
* Vectors are given in the cubic crystal frame (x, y, z along [100], [010], [001]).
* The four NV axes are the <111> body diagonals. NV frame ``i`` has z along its axis and
  x along a <1-10> direction perpendicular to it, which fixes the meaning of the strain
  term E in that frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

D_GS = 2870.0  # ground-state zero-field splitting at room temperature
GAMMA_E = 2.8025  # NV electron gyromagnetic ratio, MHz/G
A_PAR = -2.14  # 14N hyperfine coupling along the NV axis
A_PERP = -2.70  # 14N hyperfine coupling perpendicular to the NV axis
Q_N14 = -4.945  # 14N nuclear quadrupole splitting
GAMMA_N14 = 3.077e-4  # 14N gyromagnetic ratio, MHz/G
MAX_FIELD_G = 500.0  # the bright state is the lowest level well below the 1024 G anticrossing

NV_AXES = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float) / np.sqrt(3)

_r = 1 / np.sqrt(2)
SX = _r * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex)
SY = _r * np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex)
SZ = np.diag([1.0, 0.0, -1.0]).astype(complex)  # basis order |+1>, |0>, |-1>
ID3 = np.eye(3, dtype=complex)
P0 = np.diag([0.0, 1.0, 0.0]).astype(complex)  # projector onto m_s = 0


def _nv_frames() -> np.ndarray:
    frames = np.empty((4, 3, 3))
    for i, n in enumerate(NV_AXES):
        a, b, _ = np.sign(n)
        x = np.array([a, -b, 0.0]) / np.sqrt(2)
        frames[i] = np.vstack([x, np.cross(n, x), n])
    return frames


NV_FRAMES = _nv_frames()  # NV_FRAMES[i] @ v expresses crystal-frame vector v in NV frame i


@dataclass(frozen=True)
class Transition:
    """One magnetic-dipole ODMR transition of one NV orientation class."""

    freq: float  # MHz
    strength: float  # 2|<b|b_mw.S|a>|^2: 1 for an ideal m_s=0 <-> +-1 line driven perpendicular to the axis
    contrast: float  # (bright-character difference)^2 / nuclear multiplicity
    orientation: int  # index into NV_AXES


def electron_hamiltonian(b_nv, D: float = D_GS, E: float = 0.0) -> np.ndarray:
    """Electron-spin Hamiltonian (MHz) for a field ``b_nv`` (G) expressed in the NV frame."""
    bx, by, bz = GAMMA_E * np.asarray(b_nv, dtype=float)
    return D * (SZ @ SZ - (2 / 3) * ID3) + E * (SX @ SX - SY @ SY) + bx * SX + by * SY + bz * SZ


def full_hamiltonian(b_nv, D: float = D_GS, E: float = 0.0) -> np.ndarray:
    """Electron + 14N Hamiltonian (9x9, MHz) in the product basis |m_s> (x) |m_I>."""
    h = np.kron(electron_hamiltonian(b_nv, D, E), ID3)
    h += A_PAR * np.kron(SZ, SZ) + A_PERP * (np.kron(SX, SX) + np.kron(SY, SY))
    h += Q_N14 * np.kron(ID3, SZ @ SZ - (2 / 3) * ID3)
    bnx, bny, bnz = GAMMA_N14 * np.asarray(b_nv, dtype=float)
    h -= np.kron(ID3, bnx * SX + bny * SY + bnz * SZ)
    return h


def transitions(
    b,
    D: float = D_GS,
    E: float = 0.0,
    mw_dir=(0.0, 0.0, 1.0),
    hyperfine: bool = True,
    min_weight: float = 1e-3,
) -> list[Transition]:
    """All optically detectable transitions of an NV ensemble in crystal-frame field ``b``.

    The ODMR weight of a transition between eigenstates a and c is modelled as the MW
    matrix element times the squared difference of their m_s = 0 (bright) character:
    optical pumping populates the bright state and the bright state fluoresces more.
    """
    b = np.asarray(b, dtype=float)
    m = np.asarray(mw_dir, dtype=float)
    m = m / np.linalg.norm(m)
    nuc = 3 if hyperfine else 1
    p0 = np.kron(P0, ID3) if hyperfine else P0
    out: list[Transition] = []
    for i, frame in enumerate(NV_FRAMES):
        b_nv, m_nv = frame @ b, frame @ m
        drive = m_nv[0] * SX + m_nv[1] * SY + m_nv[2] * SZ
        if hyperfine:
            h, drive = full_hamiltonian(b_nv, D, E), np.kron(drive, ID3)
        else:
            h = electron_hamiltonian(b_nv, D, E)
        evals, evecs = np.linalg.eigh(h)
        bright = np.real(np.einsum("ik,ij,jk->k", evecs.conj(), p0, evecs))
        elements = evecs.conj().T @ drive @ evecs
        for a in range(len(evals)):
            for c in range(a + 1, len(evals)):
                strength = 2 * abs(elements[a, c]) ** 2
                contrast = (bright[a] - bright[c]) ** 2 / nuc
                if strength * contrast * nuc > min_weight:
                    out.append(Transition(float(evals[c] - evals[a]), float(strength), float(contrast), i))
    return out


def electron_lines(b, D: float = D_GS, E: float = 0.0) -> np.ndarray:
    """Electron-only transition frequencies, shape (4, 2): (lower, upper) line per orientation.

    Vectorised over the four orientations; used by the field-vector inversion.
    """
    b = np.asarray(b, dtype=float)
    if np.linalg.norm(b) > MAX_FIELD_G:
        raise ValueError(f"|B| = {np.linalg.norm(b):.0f} G is outside the low-field model (< {MAX_FIELD_G:.0f} G)")
    bn = GAMMA_E * (NV_FRAMES @ b)
    static = D * (SZ @ SZ - (2 / 3) * ID3) + E * (SX @ SX - SY @ SY)
    h = static[None] + bn[:, 0, None, None] * SX + bn[:, 1, None, None] * SY + bn[:, 2, None, None] * SZ
    ev = np.linalg.eigvalsh(h)
    return np.stack([ev[:, 1] - ev[:, 0], ev[:, 2] - ev[:, 0]], axis=1)


def axial_transverse(b) -> np.ndarray:
    """|B_parallel| and B_perp (G) seen by each orientation, shape (4, 2)."""
    b = np.asarray(b, dtype=float)
    par = NV_AXES @ b
    perp = np.sqrt(np.maximum(b @ b - par**2, 0.0))
    return np.stack([np.abs(par), perp], axis=1)


def canonical_field(b) -> np.ndarray:
    """Representative of ``b`` under the cubic point group O_h.

    An ensemble spectrum depends only on the set of unoriented <111> axes, which every
    signed permutation of (x, y, z) preserves. The field is therefore only measurable up
    to these 48 operations; we report |Bx| >= |By| >= |Bz| >= 0.
    """
    return np.sort(np.abs(np.asarray(b, dtype=float)))[::-1]
