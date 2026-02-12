import pytest
import numba as nb
import numpy as np
from falcon.cluster import cluster, distance_matrix


class TestGetPrecursorMzSplits:
    def test_get_precursor_mz_splits_single_split(self):
        """Test the case where there is only one split."""
        precursor_mzs = np.array([100.0, 100.5, 101.0, 101.5])
        splits = cluster._get_precursor_mz_splits(
            precursor_mzs, 0.5, "Da", batch_size=100
        )
        assert splits == nb.typed.List([0, 4])

    def test_get_precursor_mz_splits_multiple_splits(self):
        """Test the case where there are multiple splits."""
        precursor_mzs = np.array([100.0, 100.5, 101.5, 102.0])
        splits = cluster._get_precursor_mz_splits(
            precursor_mzs, 0.5, "Da", batch_size=100
        )
        assert splits == nb.typed.List([0, 2, 4])

    def test_get_precursor_mz_splits_single_split_ppm(self):
        """Test the case where there is only one split with ppm tolerance."""
        precursor_mzs = np.array([100.0, 100.002, 100.003, 100.005])
        splits = cluster._get_precursor_mz_splits(
            precursor_mzs, 50, "ppm", batch_size=100
        )
        assert splits == nb.typed.List([0, 4])

    def test_get_precursor_mz_splits_multiple_splits_ppm(self):
        """Test the case where there are multiple splits with ppm tolerance."""
        precursor_mzs = np.array([100.0, 100.005, 101.5, 101.005])
        splits = cluster._get_precursor_mz_splits(
            precursor_mzs, 50, "ppm", batch_size=100
        )
        assert splits == nb.typed.List([0, 2, 4])

    def test_get_precursor_mz_splits_batch_size(self):
        """Test the case where the batch size is smaller than the number of precursor m/z values."""
        precursor_mzs = np.array([100.0, 100.2, 100.4, 100.5])
        splits = cluster._get_precursor_mz_splits(
            precursor_mzs, 0.5, "Da", batch_size=2
        )
        assert splits == nb.typed.List([0, 2, 4])

        precursor_mzs = np.array([100, 101, 102, 103, 106, 107])
        splits = cluster._get_precursor_mz_splits(
            precursor_mzs, 1, "Da", batch_size=3
        )
        assert splits == nb.typed.List([0, 3, 4, 6])

    def test_get_precursor_mz_splits_empty(self):
        """Test the case where the precursor m/z values are empty."""
        precursor_mzs = np.array([])
        splits = cluster._get_precursor_mz_splits(
            precursor_mzs, 0.5, "Da", batch_size=100
        )
        assert splits == nb.typed.List([0])

    def test_get_precursor_mz_splits_single_value(self):
        """Test the case where there is only one precursor m/z value."""
        precursor_mzs = np.array([100.0])
        splits = cluster._get_precursor_mz_splits(
            precursor_mzs, 0.5, "Da", batch_size=100
        )
        assert splits == nb.typed.List([0, 1])


class TestLinkage:
    def test_linkage_da(self):
        """Test the linkage function with Da tolerance."""
        values = np.array([100, 101, 105, 107, 110])
        linkage_matrix = cluster._linkage(values, "Da")
        assert linkage_matrix.shape == (4, 4)
        print(linkage_matrix)
        assert (
            linkage_matrix
            == np.array([[0, 1, 1, 2], [2, 3, 2, 2], [6, 4, 5, 3], [5, 7, 10, 5]])
        ).all()




# ---------------------------------------------------------------------------
# P3: _get_cluster_group_idx
# ---------------------------------------------------------------------------


class TestGetClusterGroupIdx:
    def test_with_noise(self):
        """Noise points (-1) should be yielded as individual singletons."""
        clusters = np.array([-1, -1, 0, 0, 1, 1, 1], dtype=np.int32)
        groups = list(cluster._get_cluster_group_idx(clusters))
        # Two noise singletons, then two clusters
        assert groups[0] == (0, 1)
        assert groups[1] == (1, 2)
        assert groups[2] == (2, 4)  # cluster 0
        assert groups[3] == (4, 7)  # cluster 1

    def test_contiguous(self):
        """Contiguous blocks produce correct (start, stop) ranges."""
        clusters = np.array([0, 0, 0, 1, 1, 2], dtype=np.int32)
        groups = list(cluster._get_cluster_group_idx(clusters))
        assert groups == [(0, 3), (3, 5), (5, 6)]

    def test_empty(self):
        """Empty array should yield nothing."""
        clusters = np.array([], dtype=np.int32)
        groups = list(cluster._get_cluster_group_idx(clusters))
        assert groups == []


# ---------------------------------------------------------------------------
# P3: _postprocess_cluster
# ---------------------------------------------------------------------------


class TestPostprocessCluster:
    def test_homogeneous_cluster(self):
        """A cluster with close precursor m/z should keep one label."""
        labels = np.array([99, 99, 99, 99], dtype=np.int32)
        mzs = np.array([100.0, 100.1, 100.2, 100.3], dtype=np.float64)
        rts = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
        n = cluster._postprocess_cluster(
            labels, mzs, rts,
            precursor_tol_mass=0.5, precursor_tol_mode="Da",
            rt_tol=None, min_samples=2, start_label=0
        )
        # All spectra should be in one cluster
        assert np.all(labels == 0)
        assert n == 1

    def test_splits_distant_mz(self):
        """Spectra with distant precursor m/z should be split."""
        labels = np.array([0, 0, 0, 0], dtype=np.int32)
        mzs = np.array([100.0, 100.1, 200.0, 200.1], dtype=np.float64)
        rts = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
        n = cluster._postprocess_cluster(
            labels, mzs, rts,
            precursor_tol_mass=0.5, precursor_tol_mode="Da",
            rt_tol=None, min_samples=2, start_label=0
        )
        # Should be split into at least 2 clusters
        assert n >= 2
        assert labels[0] == labels[1]  # Close mzs grouped
        assert labels[2] == labels[3]  # Close mzs grouped
        assert labels[0] != labels[2]  # Distant mzs separated

    def test_all_singletons(self):
        """When all spectra are too far apart, all become singletons."""
        labels = np.array([0, 0, 0], dtype=np.int32)
        mzs = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        rts = np.array([10.0, 20.0, 30.0], dtype=np.float64)
        n = cluster._postprocess_cluster(
            labels, mzs, rts,
            precursor_tol_mass=0.5, precursor_tol_mode="Da",
            rt_tol=None, min_samples=2, start_label=0
        )
        # All singletons => each has a unique label
        assert len(np.unique(labels)) == 3

    def test_too_few_items(self):
        """Fewer than min_samples items should get individual labels."""
        labels = np.array([0], dtype=np.int32)
        mzs = np.array([100.0], dtype=np.float64)
        rts = np.array([10.0], dtype=np.float64)
        n = cluster._postprocess_cluster(
            labels, mzs, rts,
            precursor_tol_mass=0.5, precursor_tol_mode="Da",
            rt_tol=None, min_samples=2, start_label=5
        )
        assert n == 1
        assert labels[0] == 5


# ---------------------------------------------------------------------------
# P3: cost_based_chunking
# ---------------------------------------------------------------------------


class TestCostBasedChunking:
    def test_balanced(self):
        """Chunks should be roughly balanced."""
        splits = nb.typed.List([0, 100, 300, 500, 600])
        chunks = cluster.cost_based_chunking(splits, 2)
        assert len(chunks) == 2
        # All tasks should be assigned
        all_tasks = [t for chunk in chunks for t in chunk]
        assert len(all_tasks) == 4

    def test_single_worker(self):
        """All tasks should go to one chunk with 1 worker."""
        splits = nb.typed.List([0, 10, 20, 30])
        chunks = cluster.cost_based_chunking(splits, 1)
        assert len(chunks) == 1
        assert len(chunks[0]) == 3



