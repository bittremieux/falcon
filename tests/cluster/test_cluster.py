import pytest
import numba as nb
import numpy as np
import pandas as pd
from falcon.cluster import cluster, distance_matrix
from falcon.cluster.consensus import ConsensusTuple


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
        assert (
            linkage_matrix
            == np.array(
                [[0, 1, 1, 2], [2, 3, 2, 2], [6, 4, 5, 3], [5, 7, 10, 5]]
            )
        ).all()

    def test_linkage_two_elements(self):
        """Two values produce a single merge at their absolute distance."""
        values = np.array([100.0, 103.0])
        linkage_matrix = cluster._linkage(values, "Da")
        assert linkage_matrix.shape == (1, 4)
        # Merge the two singleton clusters (indices 0 and 1) at distance 3.
        assert sorted(linkage_matrix[0, :2]) == [0.0, 1.0]
        assert linkage_matrix[0, 2] == pytest.approx(3.0)
        assert linkage_matrix[0, 3] == 2

    def test_linkage_ppm(self):
        """ppm mode scales distances relative to the lower m/z of each pair."""
        values = np.array([100.0, 100.001, 200.0])
        linkage_matrix = cluster._linkage(values, "ppm")
        # First merge joins the closest pair (100.0, 100.001): 10 ppm.
        assert linkage_matrix[0, 2] == pytest.approx(10.0, rel=1e-3)
        # Distinct from Da, where that same gap would be 0.001.
        linkage_da = cluster._linkage(values, "Da")
        assert linkage_da[0, 2] == pytest.approx(0.001, rel=1e-3)

    def test_linkage_rt_none_mode(self):
        """RT linkage (tol_mode=None) uses absolute distances."""
        values = np.array([10.0, 11.0, 50.0])
        linkage_matrix = cluster._linkage(values, None)
        # Closest pair (10, 11) merges first at absolute distance 1.
        assert linkage_matrix[0, 2] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _get_cluster_group_idx
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
# _postprocess_cluster
# ---------------------------------------------------------------------------


class TestPostprocessCluster:
    def test_homogeneous_cluster(self):
        """A cluster with close precursor m/z should keep one label."""
        labels = np.array([99, 99, 99, 99], dtype=np.int32)
        mzs = np.array([100.0, 100.1, 100.2, 100.3], dtype=np.float64)
        rts = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float64)
        n = cluster._postprocess_cluster(
            labels,
            mzs,
            rts,
            precursor_tol_mass=0.5,
            precursor_tol_mode="Da",
            rt_tol=None,
            min_samples=2,
            start_label=0,
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
            labels,
            mzs,
            rts,
            precursor_tol_mass=0.5,
            precursor_tol_mode="Da",
            rt_tol=None,
            min_samples=2,
            start_label=0,
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
            labels,
            mzs,
            rts,
            precursor_tol_mass=0.5,
            precursor_tol_mode="Da",
            rt_tol=None,
            min_samples=2,
            start_label=0,
        )
        # All singletons => each has a unique label
        assert len(np.unique(labels)) == 3

    def test_too_few_items(self):
        """Fewer than min_samples items should get individual labels."""
        labels = np.array([0], dtype=np.int32)
        mzs = np.array([100.0], dtype=np.float64)
        rts = np.array([10.0], dtype=np.float64)
        n = cluster._postprocess_cluster(
            labels,
            mzs,
            rts,
            precursor_tol_mass=0.5,
            precursor_tol_mode="Da",
            rt_tol=None,
            min_samples=2,
            start_label=5,
        )
        assert n == 1
        assert labels[0] == 5

    def test_rt_merge_no_spurious_collision(self):
        """Combining the m/z and RT partitions must keep distinct groups apart.

        Regression test for the combined-label encoding. With a 4 (m/z) x 3
        (RT) grid of well-separated groups, the correct result is 12 clusters
        of 2 spectra each. A non-injective encoding (e.g. ``2*a + 3*b``) maps
        the m/z/RT pairs (a=3, b=0) and (a=0, b=2) to the same value (6),
        wrongly merging two groups into one and yielding only 11 clusters.
        """
        mz_islands = [100.0, 200.0, 300.0, 400.0]  # gaps >> precursor_tol
        rt_islands = [10.0, 50.0, 90.0]  # gaps >> rt_tol
        mzs, rts = [], []
        for base_mz in mz_islands:
            for base_rt in rt_islands:
                # Two spectra per (m/z, RT) combination so every combined
                # group survives the min_samples=2 filter.
                mzs.extend([base_mz, base_mz + 0.05])
                rts.extend([base_rt, base_rt + 0.1])
        mzs = np.array(mzs, dtype=np.float64)
        rts = np.array(rts, dtype=np.float64)
        labels = np.zeros(len(mzs), dtype=np.int32)
        n = cluster._postprocess_cluster(
            labels,
            mzs,
            rts,
            precursor_tol_mass=0.5,
            precursor_tol_mode="Da",
            rt_tol=1.0,
            min_samples=2,
            start_label=0,
        )
        assert n == 12
        assert len(np.unique(labels)) == 12
        # No final cluster may span more than one m/z or RT island.
        for label in np.unique(labels):
            mask = labels == label
            assert mzs[mask].max() - mzs[mask].min() <= 0.5
            assert rts[mask].max() - rts[mask].min() <= 1.0

    def test_rt_nan_spectra_not_split_by_rt(self):
        """Spectra with NaN RT must not be split by the RT filter.

        Two m/z groups each contain one real-RT pair and one NaN-RT pair.
        The NaN-RT spectra should stay together within their m/z group
        (not be split away) and must not merge with the other m/z group.
        Expected result: 4 clusters (2 m/z groups x 2 RT sub-groups each:
        real-RT pair + NaN-RT pair).
        """
        mzs = np.array(
            [100.0, 100.05, 100.1, 100.15, 200.0, 200.05, 200.1, 200.15],
            dtype=np.float64,
        )
        rts = np.array(
            [10.0, 10.1, np.nan, np.nan, 10.0, 10.1, np.nan, np.nan],
            dtype=np.float64,
        )
        labels = np.zeros(len(mzs), dtype=np.int32)
        n = cluster._postprocess_cluster(
            labels,
            mzs,
            rts,
            precursor_tol_mass=0.5,
            precursor_tol_mode="Da",
            rt_tol=1.0,
            min_samples=2,
            start_label=0,
        )
        assert n == 4
        # NaN-RT spectra within the same m/z group share a label.
        assert labels[2] == labels[3]  # NaN pair in m/z group 1
        assert labels[6] == labels[7]  # NaN pair in m/z group 2
        # The two m/z groups are separated.
        assert labels[0] != labels[4]
        # NaN-RT spectra do not merge with real-RT spectra.
        assert labels[2] != labels[0]
        assert labels[2] != labels[4]


# ---------------------------------------------------------------------------
# cost_based_chunking
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

    def test_no_tasks(self):
        """A single split point defines no intervals, so no chunks."""
        splits = nb.typed.List([0])
        chunks = cluster.cost_based_chunking(splits, 4)
        assert chunks == []

    def test_more_workers_than_tasks(self):
        """Empty chunks are dropped when there are more workers than tasks."""
        splits = nb.typed.List([0, 10, 20])
        chunks = cluster.cost_based_chunking(splits, 5)
        # Only 2 tasks => at most 2 non-empty chunks.
        assert len(chunks) == 2
        all_tasks = [t for chunk in chunks for t in chunk]
        assert len(all_tasks) == 2

    def test_largest_task_isolated(self):
        """The most expensive task should not share a chunk with others."""
        # One dominant interval (cost 100^2) and three small ones.
        splits = nb.typed.List([0, 100, 101, 102, 103])
        chunks = cluster.cost_based_chunking(splits, 2)
        # The big task (index 0, interval 0..100) lands alone; the three
        # cheap tasks pile into the other chunk.
        sizes = sorted(len(chunk) for chunk in chunks)
        assert sizes == [1, 3]


# ---------------------------------------------------------------------------
# _offset_cluster_labels / _assign_global_cluster_labels
# ---------------------------------------------------------------------------


def _make_rep(mz_split, cluster_id):
    """Minimal ConsensusTuple for relabeling tests."""
    return ConsensusTuple(
        precursor_mz=np.float32(100.0),
        precursor_charge=np.int32(2),
        mz=np.array([100.0], dtype=np.float32),
        intensity=np.array([1.0], dtype=np.float32),
        retention_time=np.float32(0.0),
        cluster_id=np.int32(cluster_id),
        mz_split=np.int32(mz_split),
    )


class TestOffsetClusterLabels:
    def test_offsets_make_labels_globally_unique(self):
        """Per-split local labels are shifted so labels never collide."""
        # Split 0 has local labels {0, 1}; split 1 has local labels {0, 1}.
        labels = np.array([0, 0, 1, 0, 1, 1], dtype=np.int32)
        splits = nb.typed.List([0, 3, 6])
        offsets = cluster._offset_cluster_labels(labels, splits)
        assert list(offsets) == [0, 2]
        # Split 1's labels are offset by 2 => globally unique.
        assert list(labels) == [0, 0, 1, 2, 3, 3]

    def test_single_split(self):
        """One split leaves labels unchanged with a zero offset."""
        labels = np.array([0, 1, 2], dtype=np.int32)
        splits = nb.typed.List([0, 3])
        offsets = cluster._offset_cluster_labels(labels, splits)
        assert list(offsets) == [0]
        assert list(labels) == [0, 1, 2]


class TestAssignGlobalClusterLabels:
    def test_reps_relabeled_per_split(self):
        """Representative cluster_ids are offset to match the global labels."""
        labels = np.array([0, 0, 1, 0, 1, 1], dtype=np.int32)
        splits = nb.typed.List([0, 3, 6])
        reps = [
            _make_rep(mz_split=0, cluster_id=0),
            _make_rep(mz_split=0, cluster_id=1),
            _make_rep(mz_split=1, cluster_id=0),
            _make_rep(mz_split=1, cluster_id=1),
        ]
        out = cluster._assign_global_cluster_labels(labels, reps, splits)
        # Split 0 reps keep ids {0, 1}; split 1 reps shift by offset 2 => {2, 3}.
        assert sorted(int(s.cluster_id) for s in out) == [0, 1, 2, 3]

    def test_singleton_only_splits(self):
        """Splits of all singletons still relabel without collisions."""
        labels = np.array([0, 1, 0], dtype=np.int32)
        splits = nb.typed.List([0, 2, 3])
        reps = [
            _make_rep(mz_split=0, cluster_id=0),
            _make_rep(mz_split=0, cluster_id=1),
            _make_rep(mz_split=1, cluster_id=0),
        ]
        out = cluster._assign_global_cluster_labels(labels, reps, splits)
        assert sorted(int(s.cluster_id) for s in out) == [0, 1, 2]


# ---------------------------------------------------------------------------
# generate_clusters (end-to-end over a Lance dataset)
# ---------------------------------------------------------------------------


def _attach_labels(dataset, labels):
    """Reproduce the caller's label re-attachment (see falcon.main).

    Returns a mapping from spectrum identifier to its cluster label.
    """
    meta = dataset.to_table(
        columns=["identifier", "precursor_mz", "retention_time"]
    ).to_pandas()
    meta = meta.sort_values(
        ["precursor_mz", "retention_time", "identifier"]
    ).reset_index(drop=True)
    meta["cluster"] = np.asarray(labels)
    return dict(zip(meta["identifier"], meta["cluster"]))


class TestGenerateClusters:
    def test_label_alignment_under_ties(self, lance_dataset, spectrum_row):
        """Labels bind to the correct spectra when sort keys fully tie.
        Identifier serves as a final tie-breaker, so the order of spectra
        with identical precursor m/z and RT is deterministic. This removes
        the dependency on the underlying sorting algorithm's stability.
        """
        peaks_a = [100.0, 200.0, 300.0, 400.0, 500.0]
        peaks_b = [600.0, 700.0, 800.0, 900.0, 1000.0]
        inten = [1.0, 0.8, 0.6, 0.4, 0.2]
        rows = [
            spectrum_row(
                "f:scan:B1", 100.0, peaks_b, inten, retention_time=10.0
            ),
            spectrum_row(
                "f:scan:A2", 100.0, peaks_a, inten, retention_time=10.0
            ),
            spectrum_row(
                "f:scan:B2", 100.0, peaks_b, inten, retention_time=10.0
            ),
            spectrum_row(
                "f:scan:A1", 100.0, peaks_a, inten, retention_time=10.0
            ),
        ]
        dataset = lance_dataset(rows)
        labels, _ = cluster.generate_clusters(
            dataset,
            "complete",
            0.1,
            0,
            0.5,
            "Da",
            None,
            0.05,
            2**15,
            "medoid",
            {},
        )
        by_id = _attach_labels(dataset, labels)
        assert by_id["f:scan:A1"] == by_id["f:scan:A2"]
        assert by_id["f:scan:B1"] == by_id["f:scan:B2"]
        assert by_id["f:scan:A1"] != by_id["f:scan:B1"]
        assert len(np.unique(np.asarray(labels))) == 2

    def test_dissimilar_spectra_become_singletons(
        self, lance_dataset, spectrum_row
    ):
        """Spectra with no shared peaks stay as distinct singleton clusters."""
        rows = [
            spectrum_row(
                "f:scan:1",
                100.0,
                [100.0, 200.0, 300.0, 400.0, 500.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                retention_time=10.0,
            ),
            spectrum_row(
                "f:scan:2",
                100.1,
                [600.0, 700.0, 800.0, 900.0, 1000.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                retention_time=20.0,
            ),
            spectrum_row(
                "f:scan:3",
                100.2,
                [1100.0, 1200.0, 1300.0, 1400.0, 1500.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
                retention_time=30.0,
            ),
        ]
        dataset = lance_dataset(rows)
        labels, reps = cluster.generate_clusters(
            dataset,
            "complete",
            0.1,
            0,
            0.5,
            "Da",
            None,
            0.05,
            2**15,
            "medoid",
            {},
        )
        # All distinct => three singleton clusters, one representative each.
        assert len(np.unique(np.asarray(labels))) == 3
        assert len(reps) == 3

    def test_isolated_precursor_mz_single_spectrum_split(
        self, lance_dataset, spectrum_row
    ):
        """An m/z-isolated spectrum forms its own single-spectrum split."""
        peaks = [100.0, 200.0, 300.0, 400.0, 500.0]
        inten = [1.0, 0.8, 0.6, 0.4, 0.2]
        rows = [
            spectrum_row("f:scan:1", 100.0, peaks, inten, retention_time=10.0),
            spectrum_row("f:scan:2", 100.1, peaks, inten, retention_time=11.0),
            # Far outside the 0.5 Da precursor tolerance => its own split.
            spectrum_row("f:scan:3", 900.0, peaks, inten, retention_time=12.0),
        ]
        dataset = lance_dataset(rows)
        labels, _ = cluster.generate_clusters(
            dataset,
            "complete",
            0.1,
            0,
            0.5,
            "Da",
            None,
            0.05,
            2**15,
            "medoid",
            {},
        )
        labels = np.asarray(labels)
        assert len(labels) == 3
        # Every spectrum is assigned a non-negative cluster label.
        assert (labels >= 0).all()
        by_id = _attach_labels(dataset, labels)
        # The isolated spectrum cannot share a cluster with the close pair.
        assert by_id["f:scan:3"] != by_id["f:scan:1"]

    def test_average_consensus_end_to_end(self, lance_dataset, spectrum_row):
        """The 'average' consensus path produces representative spectra."""
        peaks = [100.0, 200.0, 300.0, 400.0, 500.0]
        inten = [1.0, 0.8, 0.6, 0.4, 0.2]
        rows = [
            spectrum_row("f:scan:1", 100.0, peaks, inten, retention_time=10.0),
            spectrum_row(
                "f:scan:2", 100.05, peaks, inten, retention_time=11.0
            ),
            spectrum_row("f:scan:3", 100.1, peaks, inten, retention_time=12.0),
        ]
        dataset = lance_dataset(rows)
        consensus_params = {
            "min_mz": 100.0,
            "max_mz": 600.0,
            "bin_size": 0.1,
            "outlier_cutoff_lower": 1.5,
            "outlier_cutoff_upper": 1.5,
        }
        labels, reps = cluster.generate_clusters(
            dataset,
            "complete",
            0.1,
            0,
            0.5,
            "Da",
            None,
            0.05,
            2**15,
            "average",
            consensus_params,
        )
        # Three identical-shaped spectra collapse to one averaged cluster.
        assert len(np.unique(np.asarray(labels))) == 1
        assert len(reps) == 1
        assert len(reps[0].mz) > 0
        assert len(reps[0].mz) == len(reps[0].intensity)
