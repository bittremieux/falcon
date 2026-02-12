"""Tests for falcon.cluster.spectrum — P1: Spectrum processing pipeline."""

import math

import numpy as np
import pytest
import spectrum_utils.spectrum as sus

from falcon.cluster import spectrum


# ---------------------------------------------------------------------------
# _check_spectrum_valid
# ---------------------------------------------------------------------------

class TestCheckSpectrumValid:
    def test_valid(self):
        mz = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        assert spectrum._check_spectrum_valid(mz, min_peaks=5, min_mz_range=250.0)

    def test_exact_min_peaks(self):
        mz = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        assert spectrum._check_spectrum_valid(mz, min_peaks=5, min_mz_range=100.0)

    def test_too_few_peaks(self):
        mz = np.array([100.0, 200.0, 300.0])
        assert not spectrum._check_spectrum_valid(mz, min_peaks=5, min_mz_range=100.0)

    def test_exact_min_mz_range(self):
        mz = np.array([100.0, 150.0, 200.0, 250.0, 350.0])
        assert spectrum._check_spectrum_valid(mz, min_peaks=5, min_mz_range=250.0)

    def test_too_narrow_range(self):
        mz = np.array([100.0, 100.5, 101.0, 101.5, 102.0])
        assert not spectrum._check_spectrum_valid(mz, min_peaks=5, min_mz_range=250.0)


# ---------------------------------------------------------------------------
# _norm_intensity
# ---------------------------------------------------------------------------

class TestNormIntensity:
    def test_unit_norm(self):
        intensity = np.array([3.0, 4.0], dtype=np.float32)
        normed = spectrum._norm_intensity(intensity)
        assert abs(np.linalg.norm(normed) - 1.0) < 1e-6

    def test_preserves_ratio(self):
        intensity = np.array([6.0, 8.0], dtype=np.float32)
        normed = spectrum._norm_intensity(intensity)
        assert abs(normed[0] / normed[1] - 6.0 / 8.0) < 1e-6


# ---------------------------------------------------------------------------
# get_dim
# ---------------------------------------------------------------------------

class TestGetDim:
    def test_basic(self):
        n_bins, start, end = spectrum.get_dim(
            np.float32(100.0), np.float32(200.0), np.float32(1.0)
        )
        assert n_bins == 101
        assert start <= 100.0
        assert end >= 200.0

    def test_non_aligned_boundaries(self):
        n_bins, start, end = spectrum.get_dim(
            np.float32(101.0), np.float32(1500.0), np.float32(0.05)
        )
        assert start <= 101.0
        assert end >= 1500.0
        assert n_bins == math.ceil((end - start) / 0.05)


# ---------------------------------------------------------------------------
# process_spectrum
# ---------------------------------------------------------------------------

class TestProcessSpectrum:
    def test_valid_spectrum(self, simple_spectrum):
        result = spectrum.process_spectrum(
            simple_spectrum,
            min_peaks=5,
            min_mz_range=250.0,
            mz_min=100.0,
            mz_max=1500.0,
        )
        assert result is not None
        assert "identifier" in result
        assert "precursor_mz" in result
        assert "precursor_charge" in result
        assert "mz" in result
        assert "intensity" in result
        assert "retention_time" in result

    def test_too_few_peaks(self, few_peaks_spectrum):
        result = spectrum.process_spectrum(
            few_peaks_spectrum,
            min_peaks=5,
            min_mz_range=10.0,
            mz_min=100.0,
            mz_max=1500.0,
        )
        assert result is None

    def test_too_narrow_mz_range(self, narrow_spectrum):
        result = spectrum.process_spectrum(
            narrow_spectrum,
            min_peaks=3,
            min_mz_range=250.0,
            mz_min=100.0,
            mz_max=1500.0,
        )
        assert result is None

    def test_precursor_removal(self, simple_spectrum):
        """Peaks near precursor m/z (500.0) should be removed."""
        result = spectrum.process_spectrum(
            simple_spectrum,
            min_peaks=3,
            min_mz_range=50.0,
            mz_min=100.0,
            mz_max=1500.0,
            remove_precursor_tolerance=1.5,
        )
        assert result is not None
        # Peaks at 490.0 should be within 1.5 Da tolerance from charge-1
        # precursor at 500.0 => removed; peaks at 480.0 may survive.
        for mz_val in result["mz"]:
            assert abs(mz_val - 500.0) > 1.5 or abs(mz_val - 500.0) == 0

    def test_intensity_filter(self):
        """Peaks below min_intensity fraction of base peak should be removed."""
        spec = sus.MsmsSpectrum(
            "test:scan:5",
            precursor_mz=500.0,
            precursor_charge=2,
            mz=np.array(
                [110.0, 200.0, 300.0, 400.0, 450.0, 480.0], dtype=np.float32
            ),
            intensity=np.array(
                [1000.0, 5.0, 800.0, 900.0, 700.0, 600.0], dtype=np.float32
            ),
            retention_time=100.0,
        )
        result = spectrum.process_spectrum(
            spec,
            min_peaks=3,
            min_mz_range=50.0,
            mz_min=100.0,
            mz_max=1500.0,
            min_intensity=0.01,
        )
        assert result is not None
        # The 5.0 intensity peak (0.5% of 1000) should be removed at 1% threshold
        assert len(result["mz"]) < 6

    def test_max_peaks_used(self, simple_spectrum):
        """Only the top N most intense peaks should be retained."""
        result = spectrum.process_spectrum(
            simple_spectrum,
            min_peaks=3,
            min_mz_range=50.0,
            mz_min=100.0,
            mz_max=1500.0,
            max_peaks_used=3,
        )
        assert result is not None
        assert len(result["mz"]) <= 3

    def test_scaling_root(self, simple_spectrum):
        """Square root scaling should be applied."""
        result = spectrum.process_spectrum(
            simple_spectrum,
            min_peaks=3,
            min_mz_range=50.0,
            mz_min=100.0,
            mz_max=1500.0,
            scaling="root",
        )
        assert result is not None
        # Intensities should be normalized (unit norm) after scaling
        assert abs(np.linalg.norm(result["intensity"]) - 1.0) < 1e-5

    def test_scaling_log(self, simple_spectrum):
        """Log scaling should be applied."""
        result = spectrum.process_spectrum(
            simple_spectrum,
            min_peaks=3,
            min_mz_range=50.0,
            mz_min=100.0,
            mz_max=1500.0,
            scaling="log",
        )
        assert result is not None
        assert abs(np.linalg.norm(result["intensity"]) - 1.0) < 1e-5

    def test_no_charge(self, no_charge_spectrum):
        """Spectra with charge=None should be handled correctly."""
        result = spectrum.process_spectrum(
            no_charge_spectrum,
            min_peaks=5,
            min_mz_range=250.0,
            mz_min=100.0,
            mz_max=1500.0,
            remove_precursor_tolerance=1.5,
        )
        assert result is not None
        assert result["precursor_charge"] is None

    def test_output_normalized(self, simple_spectrum):
        """Output intensities should be L2-normalized."""
        result = spectrum.process_spectrum(
            simple_spectrum,
            min_peaks=5,
            min_mz_range=250.0,
            mz_min=100.0,
            mz_max=1500.0,
        )
        assert result is not None
        assert abs(np.linalg.norm(result["intensity"]) - 1.0) < 1e-5


