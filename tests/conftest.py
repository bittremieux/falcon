"""Shared test fixtures for falcon tests."""

import numpy as np
import spectrum_utils.spectrum as sus

import pytest

from falcon.cluster import similarity


@pytest.fixture
def simple_spectrum():
    """A valid MsmsSpectrum for testing preprocessing."""
    return sus.MsmsSpectrum(
        "test:scan:1",
        precursor_mz=500.0,
        precursor_charge=2,
        mz=np.array(
            [110.0, 200.0, 300.0, 400.0, 450.0, 480.0, 490.0], dtype=np.float32
        ),
        intensity=np.array(
            [100.0, 200.0, 50.0, 300.0, 150.0, 80.0, 10.0], dtype=np.float32
        ),
        retention_time=120.0,
    )


@pytest.fixture
def narrow_spectrum():
    """A spectrum with a narrow m/z range (should be rejected by min_mz_range)."""
    return sus.MsmsSpectrum(
        "test:scan:2",
        precursor_mz=500.0,
        precursor_charge=2,
        mz=np.array(
            [200.0, 200.5, 201.0, 201.5, 202.0], dtype=np.float32
        ),
        intensity=np.array(
            [100.0, 200.0, 50.0, 300.0, 150.0], dtype=np.float32
        ),
        retention_time=130.0,
    )


@pytest.fixture
def few_peaks_spectrum():
    """A spectrum with too few peaks (should be rejected by min_peaks)."""
    return sus.MsmsSpectrum(
        "test:scan:3",
        precursor_mz=500.0,
        precursor_charge=2,
        mz=np.array([200.0, 400.0], dtype=np.float32),
        intensity=np.array([100.0, 200.0], dtype=np.float32),
        retention_time=140.0,
    )


@pytest.fixture
def no_charge_spectrum():
    """A spectrum without a charge state."""
    return sus.MsmsSpectrum(
        "test:scan:4",
        precursor_mz=500.0,
        precursor_charge=None,
        mz=np.array(
            [110.0, 200.0, 300.0, 400.0, 450.0, 480.0, 490.0], dtype=np.float32
        ),
        intensity=np.array(
            [100.0, 200.0, 50.0, 300.0, 150.0, 80.0, 10.0], dtype=np.float32
        ),
        retention_time=150.0,
    )


@pytest.fixture
def spectrum_tuple_pair():
    """A pair of SpectrumTuples for similarity/distance tests."""
    int1 = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    int1 = int1 / np.linalg.norm(int1)
    int2 = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    int2 = int2 / np.linalg.norm(int2)
    spec1 = similarity.SpectrumTuple(
        precursor_mz=100.0,
        precursor_charge=2,
        mz=np.array([100.0, 101.0, 102.0], dtype=np.float32),
        intensity=int1,
    )
    spec2 = similarity.SpectrumTuple(
        precursor_mz=100.0,
        precursor_charge=2,
        mz=np.array([100.0, 101.0, 102.0], dtype=np.float32),
        intensity=int2,
    )
    return spec1, spec2


@pytest.fixture
def orthogonal_spectrum_tuple_pair():
    """A pair of SpectrumTuples with non-overlapping peaks (distance ~1)."""
    int1 = np.array([1.0, 0.0], dtype=np.float32)
    int2 = np.array([1.0, 0.0], dtype=np.float32)
    spec1 = similarity.SpectrumTuple(
        precursor_mz=100.0,
        precursor_charge=2,
        mz=np.array([100.0, 101.0], dtype=np.float32),
        intensity=int1,
    )
    spec2 = similarity.SpectrumTuple(
        precursor_mz=100.0,
        precursor_charge=2,
        mz=np.array([200.0, 201.0], dtype=np.float32),
        intensity=int2,
    )
    return spec1, spec2


@pytest.fixture
def mock_spectra():
    """Factory fixture to create a list of SpectrumTuples with similar peaks."""
    def _make(n, mz_base=100.0, mz_step=1.0, n_peaks=3):
        spectra = []
        for i in range(n):
            mz = np.array(
                [mz_base + j * mz_step for j in range(n_peaks)], dtype=np.float32
            )
            intensity = np.array(
                [0.5 + 0.1 * j for j in range(n_peaks)], dtype=np.float32
            )
            intensity = intensity / np.linalg.norm(intensity)
            spectra.append(
                similarity.SpectrumTuple(
                    precursor_mz=mz_base,
                    precursor_charge=2,
                    mz=mz,
                    intensity=intensity,
                )
            )
        return spectra
    return _make
