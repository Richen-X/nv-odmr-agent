"""Virtual NV-ensemble setup: a sample with hidden ground truth and simulated instruments.

The model is deliberately physical rather than cosmetic:

* line positions, strengths and optical contrasts come from diagonalising the NV + 14N
  Hamiltonian for all four orientations (``physics.transitions``);
* CW ODMR lines saturate and power-broaden with MW Rabi frequency
  (depth ~ s/(1+s), FWHM ~ sqrt(1+s), s = (Omega/Omega_sat)^2) towards half the pulsed
  contrast, with optical pumping raising Omega_sat and broadening the intrinsic line;
* detected counts are Poissonian, with optional multiplicative technical noise and a
  slow baseline slope as seen in real sweeps.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from functools import cached_property

import numpy as np

from .physics import D_GS, electron_lines, transitions


@dataclass
class Sample:
    """Ground truth for one NV-ensemble chip and its optical/MW setup."""

    b_field: tuple[float, float, float]  # G, crystal frame
    D: float = D_GS
    E: float = 0.0
    linewidth: float = 0.9  # MHz, Lorentzian FWHM at low MW and laser power
    contrast: float = 0.15  # pulsed contrast of one orientation class after a pi pulse
    mw_dir: tuple[float, float, float] = (0.0, 0.0, 1.0)  # direction of the MW magnetic field
    hyperfine: bool = True
    rabi_ref: float = 1.0  # MHz, Rabi frequency of an ideal line at 0 dBm
    rabi_sat: float = 0.4  # MHz, Rabi frequency that half-saturates CW ODMR without laser
    count_rate_sat: float = 8.0e7  # counts/s at optical saturation
    laser_sat_mw: float = 2.0
    rabi_decay_us: float = 2.0
    t1_us: float = 2000.0
    t1_stretch: float = 0.8
    baseline_slope: float = 0.0  # fractional fluorescence change per MHz away from 2870 MHz
    excess_noise: float = 0.0  # fractional technical noise per point
    seed: int = 0

    def __post_init__(self):
        self.b_field = tuple(float(x) for x in self.b_field)
        self.mw_dir = tuple(float(x) for x in self.mw_dir)
        for f in fields(self):
            if f.type == "float":
                setattr(self, f.name, float(getattr(self, f.name)))

    @classmethod
    def random(
        cls,
        seed: int,
        field_range: tuple[float, float] = (25.0, 70.0),
        min_separation: float = 6.0,
        window: tuple[float, float] = (2640.0, 3100.0),
        margin: float = 8.0,
    ) -> "Sample":
        """A random sample whose eight lines are resolvable and lie inside ``window``."""
        rng = np.random.default_rng(seed)
        D = D_GS + rng.normal(0.0, 0.5)
        while True:
            direction = rng.normal(size=3)
            b = direction / np.linalg.norm(direction) * rng.uniform(*field_range)
            lines = np.sort(electron_lines(b, D).ravel())
            if (
                lines[0] > window[0] + margin
                and lines[-1] < window[1] - margin
                and np.diff(lines).min() >= min_separation
            ):
                break
        return cls(
            b_field=tuple(b),
            D=D,
            linewidth=rng.uniform(0.6, 1.2),
            contrast=rng.uniform(0.10, 0.20),
            baseline_slope=rng.uniform(-2e-5, 2e-5),
            excess_noise=3e-4,
            rabi_decay_us=rng.uniform(1.5, 3.0),
            t1_us=rng.uniform(1000.0, 5000.0),
            t1_stretch=rng.uniform(0.6, 1.0),
            seed=seed,
        )

    def to_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Sample":
        return cls(**d)

    def true_lines(self) -> np.ndarray:
        """Sorted electron-spin line centres (MHz): the hyperfine-triplet centres."""
        return np.sort(electron_lines(self.b_field, self.D, self.E).ravel())

    @cached_property
    def line_table(self) -> dict[str, np.ndarray]:
        trs = transitions(self.b_field, self.D, self.E, self.mw_dir, hyperfine=self.hyperfine)
        return {
            "freq": np.array([t.freq for t in trs]),
            "strength": np.array([t.strength for t in trs]),
            "contrast": np.array([t.contrast for t in trs]),
        }


class VirtualNVSetup:
    """Simulated laser + MW source + photodetector around one :class:`Sample`."""

    def __init__(self, sample: Sample):
        self.sample = sample

    def count_rate(self, laser_mw: float) -> float:
        s = laser_mw / self.sample.laser_sat_mw
        return self.sample.count_rate_sat * s / (1 + s)

    def _rabi_frequencies(self, mw_power_dbm: float) -> np.ndarray:
        return self.sample.rabi_ref * 10 ** (mw_power_dbm / 20) * np.sqrt(self.sample.line_table["strength"])

    def odmr_signal(self, freqs, mw_power_dbm: float, laser_mw: float) -> np.ndarray:
        """Noise-free CW ODMR fluorescence, normalised to 1 far from resonance."""
        s, t = self.sample, self.sample.line_table
        f = np.asarray(freqs, dtype=float)
        optical = laser_mw / s.laser_sat_mw
        gamma0 = s.linewidth * (1 + 0.5 * optical / (1 + optical))
        sat = (self._rabi_frequencies(mw_power_dbm) / (s.rabi_sat * np.sqrt(1 + optical))) ** 2
        # CW driving can at most equalise the two populations: half the pi-pulse contrast.
        depth = 0.25 * s.contrast * t["contrast"] * 0.5 * sat / (1 + sat)
        half_width = 0.5 * gamma0 * np.sqrt(1 + sat)
        x = (f[None, :] - t["freq"][:, None]) / half_width[:, None]
        dips = (depth[:, None] / (1 + x**2)).sum(axis=0)
        return (1 + s.baseline_slope * (f - D_GS)) * (1 - dips)

    def rabi_signal(self, taus_ns, freq_mhz: float, mw_power_dbm: float) -> np.ndarray:
        """Pulsed Rabi fluorescence after a MW pulse of length tau, normalised to 1 at tau = 0."""
        s, t = self.sample, self.sample.line_table
        tau = np.asarray(taus_ns, dtype=float) / 1000.0
        omega = self._rabi_frequencies(mw_power_dbm)
        generalised = np.sqrt(omega**2 + (freq_mhz - t["freq"]) ** 2)
        amplitude = omega**2 / np.maximum(generalised, 1e-12) ** 2
        flip = 0.5 * amplitude[:, None] * (
            1 - np.cos(2 * np.pi * generalised[:, None] * tau[None, :]) * np.exp(-tau / s.rabi_decay_us)[None, :]
        )
        return 1 - (0.25 * s.contrast * t["contrast"][:, None] * flip).sum(axis=0)

    def t1_signal(self, taus_us) -> np.ndarray:
        """All-optical T1: polarised -> thermal relaxation loses 2/3 of the contrast."""
        s = self.sample
        decay = np.exp(-((np.asarray(taus_us, dtype=float) / s.t1_us) ** s.t1_stretch))
        return 1 - s.contrast * (2 / 3) * (1 - decay)

    def acquire(self, signal: np.ndarray, laser_mw: float, dwell_ms: float, rng: np.random.Generator) -> np.ndarray:
        mean = self.count_rate(laser_mw) * dwell_ms * 1e-3 * np.asarray(signal, dtype=float)
        if self.sample.excess_noise:
            mean = mean * (1 + self.sample.excess_noise * rng.standard_normal(mean.shape))
        return rng.poisson(np.clip(mean, 0.0, None))

    def measure_odmr(self, freqs, mw_power_dbm, laser_mw, dwell_ms, rng=None) -> np.ndarray:
        rng = rng if rng is not None else np.random.default_rng()
        return self.acquire(self.odmr_signal(freqs, mw_power_dbm, laser_mw), laser_mw, dwell_ms, rng)

    def measure_rabi(self, taus_ns, freq_mhz, mw_power_dbm, laser_mw, dwell_ms, rng=None) -> np.ndarray:
        rng = rng if rng is not None else np.random.default_rng()
        return self.acquire(self.rabi_signal(taus_ns, freq_mhz, mw_power_dbm), laser_mw, dwell_ms, rng)

    def measure_t1(self, taus_us, laser_mw, dwell_ms, rng=None) -> np.ndarray:
        rng = rng if rng is not None else np.random.default_rng()
        return self.acquire(self.t1_signal(taus_us), laser_mw, dwell_ms, rng)
