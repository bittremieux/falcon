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


def make_spectrum_row(
    identifier,
    precursor_mz,
    mz,
    intensity,
    precursor_charge=2,
    retention_time=0.0,
):
    """Build a single Lance row dict for a spectrum (L2-normalized intensity)."""
    mz = np.asarray(mz, dtype=np.float32)
    intensity = np.asarray(intensity, dtype=np.float32)
    intensity = intensity / np.linalg.norm(intensity)
    return {
        "identifier": identifier,
        "precursor_mz": np.float32(precursor_mz),
        "precursor_charge": precursor_charge,
        "mz": mz,
        "intensity": intensity,
        "retention_time": np.float32(retention_time),
    }


@pytest.fixture
def spectrum_row():
    """Expose ``make_spectrum_row`` as a fixture."""
    return make_spectrum_row


@pytest.fixture
def lance_dataset(tmp_path):
    """Factory fixture that writes rows to a Lance dataset and returns it.

    The dataset path mirrors the production layout
    (``spectra_charge_<charge>.lance``) so ``generate_clusters`` can derive the
    charge from the URI.
    """
    import pyarrow as pa
    import lance

    schema = pa.schema(
        [
            pa.field("identifier", pa.string()),
            pa.field("precursor_mz", pa.float32()),
            pa.field("precursor_charge", pa.int8()),
            pa.field("mz", pa.list_(pa.float32())),
            pa.field("intensity", pa.list_(pa.float32())),
            pa.field("retention_time", pa.float32()),
        ]
    )

    def _make(rows, charge=2):
        path = str(tmp_path / f"spectra_charge_{charge}.lance")
        return lance.write_dataset(
            pa.Table.from_pylist(rows, schema),
            path,
            mode="overwrite",
            data_storage_version="stable",
        )

    return _make
