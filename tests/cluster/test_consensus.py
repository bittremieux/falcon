"""Tests for consensus spectrum methods — P2: Averaging, medoids, binning, outlier rejection."""

import numpy as np
import numba as nb
import pytest

from falcon.cluster import cluster, similarity, consensus

# ---------------------------------------------------------------------------
# _spectrum_binning
# ---------------------------------------------------------------------------


class TestSpectrumBinning:
    def test_basic(self, mock_spectra):
        """Two identical spectra should produce non-empty bins."""
        spectra = mock_spectra(2)
        bins_idx, bins_peaks, bins_mz = consensus._spectrum_binning(
            spectra, min_mz=50.0, max_mz=200.0, bin_size=1.0
        )
        assert len(bins_idx) > 0
        assert len(bins_peaks) == len(bins_idx)
        assert len(bins_mz) == len(bins_idx)

    def test_frequency_filter(self):
        """Peaks in <70% of spectra should be filtered out."""
        # 10 spectra where only 6 share a peak at mz=150 (60% < 70%)
        spectra = []
        for i in range(10):
            if i < 6:
                mz = np.array([100.0, 101.0, 150.0], dtype=np.float32)
                intensity = np.array([0.5, 0.5, 0.3], dtype=np.float32)
            else:
                mz = np.array([100.0, 101.0], dtype=np.float32)
                intensity = np.array([0.5, 0.5], dtype=np.float32)
            intensity = intensity / np.linalg.norm(intensity)
            spectra.append(
                similarity.SpectrumTuple(
                    precursor_mz=100.0,
                    precursor_charge=2,
                    mz=mz,
                    intensity=intensity,
                )
            )
        bins_idx, bins_peaks, bins_mz = consensus._spectrum_binning(
            spectra, min_mz=50.0, max_mz=200.0, bin_size=1.0
        )
        # The bin containing mz=150 should be filtered out (only 60% presence)
        bin_mz_values = [float(m) for mz_bin in bins_mz for m in mz_bin]
        assert not any(
            149.5 <= m <= 150.5 for m in bin_mz_values
        ), "Peak at mz=150 (60% presence) should be filtered out"

    def test_empty_bins_excluded(self, mock_spectra):
        """Bins with no peaks should not appear in the output."""
        spectra = mock_spectra(3, mz_base=100.0, mz_step=1.0, n_peaks=2)
        bins_idx, bins_peaks, bins_mz = consensus._spectrum_binning(
            spectra, min_mz=50.0, max_mz=200.0, bin_size=1.0
        )
        # All returned bins should have actual data
        for peak_arr in bins_peaks:
            assert len(peak_arr) > 0


# ---------------------------------------------------------------------------
# _outlier_rejection
# ---------------------------------------------------------------------------


class TestOutlierRejection:
    def test_no_outliers(self):
        """When all values are similar, nothing should be removed."""
        bins_indices = np.array([0, 1], dtype=np.int32)
        p1 = np.array([0.5, 0.51, 0.49, 0.50], dtype=np.float32)
        p2 = np.array([0.3, 0.31, 0.29, 0.30], dtype=np.float32)
        bins_peaks = nb.typed.List([p1, p2])
        m1 = np.array([100.0, 100.1, 99.9, 100.0], dtype=np.float32)
        m2 = np.array([101.0, 101.1, 100.9, 101.0], dtype=np.float32)
        bins_mz = nb.typed.List([m1, m2])
        out_idx, out_peaks, out_mz = consensus._outlier_rejection(
            bins_indices, bins_peaks, bins_mz, 1.5, 1.5
        )
        assert len(out_idx) == 2

    def test_removes_outlier(self):
        """An extreme outlier intensity should be removed."""
        bins_indices = np.array([0], dtype=np.int32)
        peaks = np.array([0.5, 0.51, 0.49, 10.0], dtype=np.float32)
        mzs = np.array([100.0, 100.1, 99.9, 100.0], dtype=np.float32)
        bins_peaks = nb.typed.List([peaks])
        bins_mz = nb.typed.List([mzs])
        out_idx, out_peaks, out_mz = consensus._outlier_rejection(
            bins_indices, bins_peaks, bins_mz, 1.5, 1.5
        )
        # The average should be closer to 0.5 (outlier 10.0 removed)
        assert len(out_idx) == 1
        assert out_peaks[0] < 1.0


# ---------------------------------------------------------------------------
# _sigma_clipping
# ---------------------------------------------------------------------------


class TestSigmaClipping:
    def test_convergence(self):
        """Sigma clipping should converge and not loop infinitely."""
        intensities = np.array([1.0, 1.1, 0.9, 1.0, 100.0], dtype=np.float32)
        mzs = np.array([100.0, 100.1, 99.9, 100.0, 100.0], dtype=np.float32)
        result_i, result_m = consensus._sigma_clipping(
            intensities, mzs, 1.5, 1.5
        )
        # The 100.0 outlier should be removed
        assert len(result_i) < len(intensities)
        assert np.max(result_i) < 2.0

    def test_all_identical(self):
        """When std=0, the loop should break without removing values."""
        intensities = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        mzs = np.array([100.0, 100.0, 100.0], dtype=np.float32)
        result_i, result_m = consensus._sigma_clipping(
            intensities, mzs, 1.5, 1.5
        )
        assert len(result_i) == 3


# ---------------------------------------------------------------------------
# _sigma_clip
# ---------------------------------------------------------------------------


class TestSigmaClip:
    def test_mask_correct(self):
        """Lower and upper bound masks should be correct."""
        values = np.array([1.0, 5.0, 10.0, 15.0, 20.0], dtype=np.float32)
        median = 10.0
        std = 5.0
        mask = consensus._sigma_clip(values, median, std, 1.0, 1.0)
        # bounds: [10 - 5, 10 + 5] = [5, 15]
        expected = np.array([False, True, True, True, False])
        np.testing.assert_array_equal(mask, expected)


# ---------------------------------------------------------------------------
# _construct_average_spectrum
# ---------------------------------------------------------------------------


class TestConstructAverageSpectrum:
    def test_output_types(self):
        bins_indices = np.array([5, 10], dtype=np.int32)
        bins_peaks = np.array([0.5, 0.7], dtype=np.float32)
        bins_mz = np.array([100.0, 101.0], dtype=np.float32)
        result = consensus._construct_average_spectrum(
            bins_indices,
            bins_peaks,
            bins_mz,
            avg_precursor_mz=200.0,
            charge=2,
            avg_rt=60.0,
            cluster=5,
        )
        precursor_mz, charge, mz, intensity, rt, cluster_id = result
        assert precursor_mz == 200.0
        assert charge == 2
        assert len(mz) == 2
        assert len(intensity) == 2
        assert rt == 60.0
        assert cluster_id == 5


# ---------------------------------------------------------------------------
# _get_cluster_medoids
# ---------------------------------------------------------------------------


class TestGetClusterMedoids:
    def test_small_cluster(self, mock_spectra):
        """Clusters with ≤2 spectra should use the first spectrum."""
        spectra = mock_spectra(2)
        labels = np.array([0, 0], dtype=np.int32)
        rts = np.array([10.0, 20.0], dtype=np.float32)
        order_map = np.array([0, 1], dtype=np.int64)
        pdist = np.array([0.5], dtype=np.float32)  # 1 pair
        result = consensus._get_cluster_medoids(
            spectra, labels, rts, order_map, pdist
        )
        precursor_mzs, _, _, _, retention_times, _ = result
        assert len(precursor_mzs) == 1
        assert precursor_mzs[0] == spectra[0].precursor_mz
        assert retention_times[0] == 10.0

    def test_single_cluster_three_spectra(self, mock_spectra):
        """Medoid should be the spectrum with minimum total distance."""
        spectra = mock_spectra(3)
        labels = np.array([0, 0, 0], dtype=np.int32)
        rts = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        order_map = np.array([0, 1, 2], dtype=np.int64)
        # distances: (0,1)=0.1, (0,2)=0.5, (1,2)=0.2 => medoid is 1 (sum=0.3)
        pdist = np.array([0.1, 0.5, 0.2], dtype=np.float32)
        result = consensus._get_cluster_medoids(
            spectra, labels, rts, order_map, pdist
        )
        (
            precursor_mzs,
            charges,
            mzs,
            intensities,
            retention_times,
            cluster_ids,
        ) = result
        assert len(precursor_mzs) == 1
        # Spectrum 1 should be the medoid (lowest row sum: 0.1+0.2=0.3)
        assert retention_times[0] == 20.0


# ---------------------------------------------------------------------------
# _get_representative_spectra
# ---------------------------------------------------------------------------


class TestGetRepresentativeSpectra:
    def test_medoid_dispatch(self, mock_spectra):
        spectra = mock_spectra(3)
        labels = np.array([0, 0, 0], dtype=np.int32)
        rts = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        order_map = np.array([0, 1, 2], dtype=np.int64)
        pdist = np.array([0.1, 0.5, 0.2], dtype=np.float32)
        result = consensus._get_representative_spectra(
            spectra, labels, rts, order_map, "medoid", {"pdist": pdist}
        )
        assert len(result) == 1
        assert hasattr(result[0], "precursor_mz")

    def test_invalid_method(self, mock_spectra):
        spectra = mock_spectra(2)
        labels = np.array([0, 0], dtype=np.int32)
        rts = np.array([10.0, 20.0], dtype=np.float32)
        order_map = np.array([0, 1], dtype=np.int64)
        with pytest.raises(
            ValueError, match="Unknown consensus spectrum method"
        ):
            consensus._get_representative_spectra(
                spectra, labels, rts, order_map, "nonexistent", {}
            )

    def test_average_dispatch(self, mock_spectra):
        """The 'average' method should dispatch to the averaging path."""
        spectra = mock_spectra(3, mz_base=100.0, mz_step=1.0, n_peaks=3)
        labels = np.array([0, 0, 0], dtype=np.int32)
        rts = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        order_map = np.array([0, 1, 2], dtype=np.int64)
        params = {
            "min_mz": 50.0,
            "max_mz": 200.0,
            "bin_size": 1.0,
            "outlier_cutoff_lower": 1.5,
            "outlier_cutoff_upper": 1.5,
        }
        result = consensus._get_representative_spectra(
            spectra, labels, rts, order_map, "average", params
        )
        assert len(result) == 1
        assert hasattr(result[0], "precursor_mz")
        assert len(result[0].mz) == len(result[0].intensity)


# ---------------------------------------------------------------------------
# _get_cluster_average
# ---------------------------------------------------------------------------


class TestGetClusterAverage:
    def test_single_cluster_averaged(self, mock_spectra):
        """A cluster of 3 identical spectra averages into one spectrum."""
        spectra = mock_spectra(3, mz_base=100.0, mz_step=1.0, n_peaks=3)
        labels = np.array([0, 0, 0], dtype=np.int32)
        rts = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        order_map = np.array([0, 1, 2], dtype=np.int64)
        result = consensus._get_cluster_average(
            spectra,
            labels,
            rts,
            order_map,
            min_mz=50.0,
            max_mz=200.0,
            bin_size=1.0,
            outlier_cutoff_lower=1.5,
            outlier_cutoff_upper=1.5,
        )
        (
            precursor_mzs,
            charges,
            mzs,
            intensities,
            retention_times,
            cluster_ids,
        ) = result
        assert len(precursor_mzs) == 1
        # Averaged RT of identical-content spectra is the mean of the inputs.
        assert retention_times[0] == pytest.approx(20.0)
        assert cluster_ids[0] == 0
        assert len(mzs[0]) > 0

    def test_singleton_passthrough(self, mock_spectra):
        """A singleton cluster returns the original spectrum unchanged."""
        spectra = mock_spectra(1, n_peaks=3)
        labels = np.array([0], dtype=np.int32)
        rts = np.array([42.0], dtype=np.float32)
        order_map = np.array([0], dtype=np.int64)
        result = consensus._get_cluster_average(
            spectra,
            labels,
            rts,
            order_map,
            min_mz=50.0,
            max_mz=200.0,
            bin_size=1.0,
            outlier_cutoff_lower=1.5,
            outlier_cutoff_upper=1.5,
        )
        (
            precursor_mzs,
            charges,
            mzs,
            intensities,
            retention_times,
            cluster_ids,
        ) = result
        assert len(precursor_mzs) == 1
        assert retention_times[0] == 42.0
        np.testing.assert_array_equal(mzs[0], spectra[0].mz)


# ---------------------------------------------------------------------------
# typed_list_to_numpy
# ---------------------------------------------------------------------------


class TestTypedListToNumpy:
    def test_roundtrip(self):
        lst = nb.typed.List.empty_list(nb.types.float32)
        for v in (1.0, 2.5, 3.5):
            lst.append(np.float32(v))
        arr = consensus.typed_list_to_numpy(lst)
        assert arr.dtype == np.float32
        np.testing.assert_array_equal(
            arr, np.array([1.0, 2.5, 3.5], np.float32)
        )
