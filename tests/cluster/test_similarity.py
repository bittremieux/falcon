import math

import numpy as np
import pandas as pd
import pytest

from falcon.cluster import similarity


class TestSimilarity:
    def test_cosine_fast_equal(self, mock_spectra):
        """Test the cosine similarity between two equal spectra."""
        spectra = mock_spectra(2, n_peaks=3)
        # mock_spectra generates identical spectra
        spec1, spec2 = spectra[0], spectra[1]

        score, matched_peaks = similarity.cosine_fast(spec1, spec2, 0.5)
        assert score == 1.0
        assert matched_peaks == 3

    def test_cosine_fast_different(self):
        """Test the cosine similarity between two different spectra."""
        int1 = np.array([0.5, 0.6, 0.7], dtype=np.float32)
        int1 = int1 / np.linalg.norm(int1)
        spec1 = similarity.SpectrumTuple(
            precursor_mz=100.0,
            precursor_charge=2,
            mz=np.array([100.0, 101.0, 102.0], dtype=np.float32),
            intensity=int1,
        )

        int2 = np.array([0.7, 0.6, 0.5], dtype=np.float32)
        int2 = int2 / np.linalg.norm(int2)
        spec2 = similarity.SpectrumTuple(
            precursor_mz=100.0,
            precursor_charge=2,
            mz=np.array([100.0, 101.0, 102.0], dtype=np.float32),
            intensity=int2,
        )
        score, matched_peaks = similarity.cosine_fast(spec1, spec2, 0.1)
        assert round(score, 3) == 0.964
        assert matched_peaks == 3

    def test_cosine_fast_partial_overlap(self):
        """Only the shared peaks contribute to the match count and score."""
        inten = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        inten = inten / np.linalg.norm(inten)
        spec1 = similarity.SpectrumTuple(
            precursor_mz=100.0,
            precursor_charge=2,
            mz=np.array([100.0, 200.0, 300.0], dtype=np.float32),
            intensity=inten,
        )
        spec2 = similarity.SpectrumTuple(
            precursor_mz=100.0,
            precursor_charge=2,
            mz=np.array([100.0, 200.0, 999.0], dtype=np.float32),
            intensity=inten,
        )
        score, matched = similarity.cosine_fast(spec1, spec2, 0.1)
        assert matched == 2
        # Two of three normalized peaks match => score 2/3.
        assert score == pytest.approx(2.0 / 3.0, rel=1e-3)

    def test_cosine_fast_tolerance_boundary(self):
        """A peak just outside the tolerance must not match."""
        inten = np.array([1.0], dtype=np.float32)
        spec1 = similarity.SpectrumTuple(
            precursor_mz=100.0,
            precursor_charge=2,
            mz=np.array([100.0], dtype=np.float32),
            intensity=inten,
        )
        spec2 = similarity.SpectrumTuple(
            precursor_mz=100.0,
            precursor_charge=2,
            mz=np.array([100.2], dtype=np.float32),
            intensity=inten,
        )
        # Difference 0.2 > tolerance 0.1 => no match.
        score, matched = similarity.cosine_fast(spec1, spec2, 0.1)
        assert matched == 0
        assert score == 0.0
        # Within tolerance => match.
        score, matched = similarity.cosine_fast(spec1, spec2, 0.25)
        assert matched == 1
        assert score == pytest.approx(1.0)

    def test_cosine_fast_empty_other(self):
        """Comparing against an empty spectrum yields no matches."""
        spec1 = similarity.SpectrumTuple(
            precursor_mz=100.0,
            precursor_charge=2,
            mz=np.array([100.0, 200.0], dtype=np.float32),
            intensity=np.array([0.7, 0.7], dtype=np.float32),
        )
        spec2 = similarity.SpectrumTuple(
            precursor_mz=100.0,
            precursor_charge=2,
            mz=np.array([], dtype=np.float32),
            intensity=np.array([], dtype=np.float32),
        )
        score, matched = similarity.cosine_fast(spec1, spec2, 0.1)
        assert matched == 0
        assert score == 0.0

    def test_df_row_to_spectrum_tuple(self):
        """Test the conversion of a DataFrame row to a SpectrumTuple."""
        int1 = np.array([0.1, 0.2, 0.3])
        int1 = int1 / np.linalg.norm(int1)
        int2 = np.array([0.9, 0.5, 0.1])
        int2 = int2 / np.linalg.norm(int2)
        int3 = np.array([0.6, 0.4, 0.6])
        int3 = int3 / np.linalg.norm(int3)
        df = pd.DataFrame(
            {
                "precursor_mz": [100.0, 101.0, 102.0],
                "precursor_charge": [1, np.nan, 2],
                "mz": [
                    np.array([100.0, 101.0, 102.0]),
                    np.array([200.0, 201.0, 202.0]),
                    np.array([300.0, 301.0, 302.0]),
                ],
                "intensity": [int1, int2, int3],
            }
        )
        spectra = df.apply(similarity.df_row_to_spectrum_tuple, axis=1)

        assert spectra[0].precursor_mz == 100.0
        assert spectra[0].precursor_charge == 1
        assert np.array_equal(spectra[0].mz, np.array([100.0, 101.0, 102.0]))
        assert np.array_equal(spectra[0].intensity, int1)

        assert spectra[1].precursor_mz == 101.0
        assert math.isnan(spectra[1].precursor_charge)
        assert np.array_equal(spectra[1].mz, np.array([200.0, 201.0, 202.0]))
        assert np.array_equal(spectra[1].intensity, int2)

        assert spectra[2].precursor_mz == 102.0
        assert spectra[2].precursor_charge == 2
        assert np.array_equal(spectra[2].mz, np.array([300.0, 301.0, 302.0]))
        assert np.array_equal(spectra[2].intensity, int3)
