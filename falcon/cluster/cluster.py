import gc
import logging
import math
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, List, Optional, Tuple

import fastcluster
import joblib
import lance
import numba as nb
import numpy as np
import pyarrow as pa
import scipy.cluster.hierarchy as sch
import spectrum_utils.utils as suu
from scipy.cluster.hierarchy import fcluster
from tqdm import tqdm

from . import similarity

logger = logging.getLogger("falcon")


def generate_clusters(
    dataset: lance.LanceDataset,
    linkage: str,
    distance_threshold: float,
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    rt_tol: float,
    fragment_tol: float,
) -> np.ndarray:
    """
    Hierarchical clustering of the given pairwise distance matrix.

    Parameters
    ----------
    dataset : lance.LanceDataset
        The dataset containing the spectra to be clustered.
    linkage: str
        The linkage method to use for hierarchical clustering.
    distance_threshold : float
        The linkage distance threshold at or above which clusters will not be merged.
    precursor_tol_mass : float
        Maximum precursor mass tolerance for points to be clustered together.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    rt_tol : float
        The retention time tolerance for points to be clustered together. If
        `None`, do not restrict the retention time.
    fragment_tol: float
        The fragment m/z tolerance.

    Returns
    -------
    np.ndarray
        Cluster labels. Noisy samples are given the label -1.
    """
    # Hierarchical clustering using the precomputed pairwise distance matrix.
    min_samples = 2
    logger.debug(
        "Hierarchical clustering (distance_threshold=%.4f, min_samples=%d)",
        distance_threshold,
        min_samples,
    )
    # Sort the metadata by increasing precursor m/z for easy subsetting.
    data = (
        dataset.to_table(
            columns=[
                "precursor_mz",
                "precursor_charge",
                "retention_time",
                "mz",
                "intensity",
            ]
        )
        .add_column(0, "index", pa.array(range(dataset.count_rows())))
        .sort_by("precursor_mz")
        .to_pandas()
    )
    # Cluster per contiguous block of precursor m/z's (relative to the
    # precursor m/z threshold).
    logging.info(
        "Cluster %d spectra using %s linkage and distance threshold %.3f",
        len(data),
        linkage,
        distance_threshold,
    )
    with tempfile.NamedTemporaryFile(suffix=".npy") as cluster_file:
        cluster_filename = cluster_file.name
    cluster_labels = np.lib.format.open_memmap(
        cluster_filename, mode="w+", dtype=np.int32, shape=(data.shape[0],)
    )
    cluster_labels.fill(-1)
    max_label, medoids = 0, []
    with tqdm(
        total=len(data), desc="Clustering", unit="spectra", smoothing=0
    ) as pbar:
        idx = data["index"].values
        mz = data["precursor_mz"].values
        rt = data["retention_time"].values
        splits = _get_precursor_mz_splits(
            mz, precursor_tol_mass, precursor_tol_mode, 2**15
        )
        spec_tuples = data.apply(
            similarity.df_row_to_spectrum_tuple, axis=1
        ).tolist()
        del data
        # Per-split cluster labels.
        for interval_medoids in joblib.Parallel(
            n_jobs=-1, backend="threading"
        )(
            joblib.delayed(_cluster_interval)(
                spec_tuples,
                idx,
                mz,
                rt,
                cluster_labels,
                splits[i],
                splits[i + 1],
                linkage,
                distance_threshold,
                precursor_tol_mass,
                precursor_tol_mode,
                rt_tol,
                fragment_tol,
                pbar,
            )
            for i in range(len(splits) - 1)
        ):
            if interval_medoids is not None:
                medoids.append(interval_medoids)
        max_label = _assign_global_cluster_labels(
            cluster_labels, idx, splits, max_label
        )
    cluster_labels.flush()
    medoids = np.hstack(medoids)
    noise_mask = cluster_labels == -1
    n_clusters, n_noise = np.amax(cluster_labels) + 1, noise_mask.sum()
    logger.info(
        "%d spectra grouped in %d clusters, %d spectra remain as singletons",
        (cluster_labels != -1).sum(),
        n_clusters,
        n_noise,
    )
    # Reassign noise points to singleton clusters.
    cluster_labels[noise_mask] = np.arange(n_clusters, n_clusters + n_noise)
    return cluster_labels, medoids


@nb.njit
def _get_precursor_mz_splits(
    precursor_mzs: np.ndarray,
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    batch_size: int,
) -> nb.typed.List:
    """
    Find contiguous blocks of precursor m/z's, relative to the precursor m/z
    tolerance.

    Parameters
    ----------
    precursor_mzs : np.ndarray
        The sorted precursor m/z's.
    precursor_tol_mass : float
        The value of the precursor m/z tolerance.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    batch_size : int
        Maximum interval size.

    Returns
    -------
    nb.typed.List[int]
        A list of start and end indices of blocks of precursor m/z's that do
        not exceed the precursor m/z tolerance and are separated by at least
        the precursor m/z tolerance.
    """
    splits, i = nb.typed.List([0]), 1
    for i in range(1, len(precursor_mzs)):
        if (
            suu.mass_diff(
                precursor_mzs[i],
                precursor_mzs[i - 1],
                precursor_tol_mode == "Da",
            )
            > precursor_tol_mass
        ):
            block_size = i - splits[-1]
            if block_size < batch_size:
                splits.append(i)
            else:
                n_chunks = math.ceil(block_size / batch_size)
                chunk_size = block_size // n_chunks
                for _ in range(block_size % n_chunks):
                    splits.append(splits[-1] + chunk_size + 1)
                for _ in range(n_chunks - (block_size % n_chunks)):
                    splits.append(splits[-1] + chunk_size)
    splits.append(len(precursor_mzs))
    return splits


def _cluster_interval(
    spectra: List[similarity.SpectrumTuple],
    idx: np.ndarray,
    mzs: np.ndarray,
    rts: np.ndarray,
    cluster_labels: np.ndarray,
    interval_start: int,
    interval_stop: int,
    linkage: str,
    distance_threshold: float,
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    rt_tol: float,
    fragment_mz_tol: float,
    pbar: tqdm,
) -> np.ndarray:
    """
    Cluster the vectors in the given interval.

    Parameters
    ----------
    spectra : List[similarity.SpectrumTuple]
        The spectra to cluster.
    idx : np.ndarray
        The indexes of the vectors in the current interval.
    mzs : np.ndarray
        The precursor m/z's corresponding to the current interval indexes.
    rts : np.ndarray
        The retention times corresponding to the current interval indexes.
    cluster_labels : np.ndarray
        Array in which to fill the cluster label assignments.
    interval_start : int
        The current interval start index.
    interval_stop : int
        The current interval stop index.
    linkage : str
        Linkage method to calculate the cluster distances. See
        `scipy.cluster.hierarchy.linkage` for possible options.
    distance_threshold : float
        The maximum linkage distance threshold during clustering.
    precursor_tol_mass : float
        The value of the precursor m/z tolerance.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    rt_tol : float
        The retention time tolerance for points to be clustered together. If
        `None`, do not restrict the retention time.
    fragment_mz_tol : float
        The fragment m/z tolerance.
    pbar : tqdm.tqdm
        Tqdm progress bar.

    Returns
    -------
    np.ndarray
        List with indexes of the medoids for each cluster.
    """
    n_vectors = interval_stop - interval_start
    if n_vectors > 1:
        idx_interval = idx[interval_start:interval_stop]
        mzs_interval = mzs[interval_start:interval_stop]
        rts_interval = rts[interval_start:interval_stop]
        # Hierarchical clustering of the vectors.
        # Subtract 1 because fcluster starts with cluster label 1 instead of 0
        # (like Scikit-Learn does).
        pdist = compute_condensed_distance_matrix(
            spectra[interval_start:interval_stop], fragment_mz_tol
        )
        labels = (
            sch.fcluster(
                fastcluster.linkage(pdist, linkage),
                distance_threshold,
                "distance",
            )
            - 1
        )
        # Refine initial clusters to make sure spectra within a cluster don't
        # have an excessive precursor m/z difference.
        order = np.argsort(labels)
        idx_interval, mzs_interval, rts_interval = (
            idx_interval[order],
            mzs_interval[order],
            rts_interval[order],
        )
        labels, current_label = labels[order], 0
        for start_i, stop_i in _get_cluster_group_idx(labels):
            n_clusters = _postprocess_cluster(
                labels[start_i:stop_i],
                mzs_interval[start_i:stop_i],
                rts_interval[start_i:stop_i],
                precursor_tol_mass,
                precursor_tol_mode,
                rt_tol,
                2,
                current_label,
            )
            current_label += n_clusters
        # Assign cluster labels.
        cluster_labels[idx_interval] = labels
        if current_label > 0:
            # Compute cluster medoids.
            order_ = np.argsort(labels)
            idx_interval, labels = idx_interval[order_], labels[order_]
            order_map = order[order_]
            medoids = _get_cluster_medoids(
                idx_interval, labels, pdist, order_map
            )
        else:
            medoids = range(interval_start, interval_stop)
        # Force memory clearing.
        del pdist
        if n_vectors > 2**11:
            gc.collect()
    else:
        medoids = [interval_start]
    pbar.update(n_vectors)
    return medoids


@nb.njit
def _get_cluster_group_idx(clusters: np.ndarray) -> Iterator[Tuple[int, int]]:
    """
    Get start and stop indexes for unique cluster labels.

    Parameters
    ----------
    clusters : np.ndarray
        The ordered cluster labels (noise points are -1).

    Returns
    -------
    Iterator[Tuple[int, int]]
        Tuples with the start index (inclusive) and end index (exclusive) of
        the unique cluster labels.
    """
    start_i = 0
    while clusters[start_i] == -1 and start_i < clusters.shape[0]:
        yield start_i, start_i + 1
        start_i += 1
    stop_i = start_i
    while stop_i < clusters.shape[0]:
        start_i, label = stop_i, clusters[stop_i]
        while stop_i < clusters.shape[0] and clusters[stop_i] == label:
            stop_i += 1
        yield start_i, stop_i


@nb.njit(boundscheck=False)
def _postprocess_cluster(
    cluster_labels: np.ndarray,
    cluster_mzs: np.ndarray,
    cluster_rts: np.ndarray,
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    rt_tol: float,
    min_samples: int,
    start_label: int,
) -> int:
    """
    Partitioning based on the precursor m/z's within each initial cluster to
    avoid that spectra within a cluster have an excessive precursor m/z
    difference.

    Parameters
    ----------
    cluster_labels : np.ndarray
        Array in which to write the cluster labels.
    cluster_mzs : np.ndarray
        Precursor m/z's of the samples in a single initial cluster.
    cluster_rts: np.ndarray
        Retention times of the samples in a single intial cluster.
    precursor_tol_mass : float
        Maximum precursor mass tolerance for points to be clustered together.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    rt_tol: float
        Maximum retention time tolerance for points to be clustered together.
    min_samples : int
        The minimum number of samples in a cluster.
    start_label : int
        The first cluster label.

    Returns
    -------
    int
        The number of clusters after splitting on precursor m/z.
    """
    # No splitting needed if there are too few items in cluster.
    if cluster_labels.shape[0] < min_samples:
        cluster_labels.fill(-1)
        return 0
    else:
        # Group items within the cluster based on their precursor m/z.
        # Precursor m/z's within a single group can't exceed the specified
        # precursor m/z tolerance (`distance_threshold`).
        # Subtract 1 because fcluster starts with cluster label 1 instead of 0
        # (like Scikit-Learn does).
        linkage = _linkage(cluster_mzs, precursor_tol_mode)
        with nb.objmode(cluster_assignments="int32[:]"):
            cluster_assignments = (
                sch.fcluster(linkage, precursor_tol_mass, "distance") - 1
            )
        # Optionally restrict clusters by their retention time as well.
        if rt_tol is not None:
            with nb.objmode(cluster_assignments="int32[:]"):
                cluster_assignments_rt = (
                    fcluster(_linkage(cluster_rts), rt_tol, "distance") - 1
                )
                # Merge cluster assignments based on precursor m/z and RT.
                # First prime factorization is used to get unique combined cluster
                # labels, after which consecutive labels are obtained.
                cluster_assignments = np.unique(
                    cluster_assignments * 2 + cluster_assignments_rt * 3,
                    return_inverse=True,
                )[1]

        n_clusters = cluster_assignments.max() + 1
        # Update cluster assignments.
        if n_clusters == 1:
            # Single homogeneous cluster.
            cluster_labels.fill(start_label)
        elif n_clusters == cluster_mzs.shape[0]:
            # Only singletons.
            cluster_labels.fill(-1)
            n_clusters = 0
        else:
            labels = nb.typed.Dict.empty(
                key_type=nb.int64, value_type=nb.int64
            )
            for i, label in enumerate(cluster_assignments):
                labels[label] = labels.get(label, 0) + 1
            n_clusters = 0
            for label, count in labels.items():
                if count < min_samples:
                    labels[label] = -1
                else:
                    labels[label] = start_label + n_clusters
                    n_clusters += 1
            for i, label in enumerate(cluster_assignments):
                cluster_labels[i] = labels[label]
        return n_clusters


@nb.njit(cache=True, fastmath=True)
def _linkage(values: np.ndarray, tol_mode: str = None) -> np.ndarray:
    """
    Perform hierarchical clustering of a one-dimensional m/z or RT array.

    Because the data is one-dimensional, no pairwise distance matrix needs to
    be computed, but rather sorting can be used.

    For information on the linkage output format, see:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

    Parameters
    ----------
    values : np.ndarray
        The precursor m/z's or RTs for which pairwise distances are computed.
    tol_mode : str
        The unit of the tolerance ('Da' or 'ppm' for precursor m/z, 'rt' for
        retention time).

    Returns
    -------
    np.ndarray
        The hierarchical clustering encoded as a linkage matrix.
    """
    linkage = np.zeros((values.shape[0] - 1, 4), np.double)
    # min, max, cluster index, number of cluster elements
    # noinspection PyUnresolvedReferences
    clusters = [(values[i], values[i], i, 1) for i in np.argsort(values)]
    for it in range(values.shape[0] - 1):
        min_dist, min_i = np.inf, -1
        for i in range(len(clusters) - 1):
            dist = clusters[i + 1][1] - clusters[i][0]  # Always positive.
            if tol_mode == "ppm":
                dist = dist / clusters[i][0] * 10**6
            if dist < min_dist:
                min_dist, min_i = dist, i
        n_points = clusters[min_i][3] + clusters[min_i + 1][3]
        linkage[it, :] = [
            clusters[min_i][2],
            clusters[min_i + 1][2],
            min_dist,
            n_points,
        ]
        clusters[min_i] = (
            clusters[min_i][0],
            clusters[min_i + 1][1],
            values.shape[0] + it,
            n_points,
        )
        del clusters[min_i + 1]

    return linkage


@nb.njit(fastmath=True, boundscheck=False)
def _get_cluster_medoids(
    idx_interval: np.ndarray,
    labels: np.ndarray,
    pdist: np.ndarray,
    order_map: np.ndarray,
) -> np.ndarray:
    """
    Get the indexes of the cluster medoids.

    Parameters
    ----------
    idx_interval : np.ndarray
        vector indexes.
    labels : np.ndarray
        Cluster labels.
    pdist : np.ndarray
        Condensed pairwise distance matrix.
    order_map : np.ndarray
        Map to convert label indexes to pairwise distance matrix indexes.

    Returns
    -------
    np.ndarray
        Array with indexes of the medoids for each cluster.
    """
    medoids, m = [], len(idx_interval)
    for start_i, stop_i in _get_cluster_group_idx(labels):
        if stop_i - start_i > 1:
            row_sum = np.zeros(stop_i - start_i, np.float32)
            for row in range(stop_i - start_i):
                for col in range(row + 1, stop_i - start_i):
                    i, j = order_map[start_i + row], order_map[start_i + col]
                    if i > j:
                        i, j = j, i
                    pdist_ij = pdist[m * i + j - ((i + 2) * (i + 1)) // 2]
                    row_sum[row] += pdist_ij
                    row_sum[col] += pdist_ij
            medoids.append(idx_interval[start_i + np.argmin(row_sum)])
        else:
            medoids.append(idx_interval[start_i])
    return np.asarray(medoids, dtype=np.int32)


@nb.njit(boundscheck=False)
def _assign_global_cluster_labels(
    cluster_labels: np.ndarray,
    idx: np.ndarray,
    splits: nb.typed.List,
    current_label: int,
) -> int:
    """
    Convert cluster labels per split to globally unique labels.

    Parameters
    ----------
    cluster_labels : np.ndarray
        The cluster labels.
    idx : np.ndarray
        The label indexes.
    splits : nb.typed.List
        A list of start and end indices of cluster chunks.
    current_label : int
        First cluster label.

    Returns
    -------
    int
        Last cluster label.
    """
    max_label = current_label
    for i in range(len(splits) - 1):
        for j in idx[splits[i] : splits[i + 1]]:
            if cluster_labels[j] != -1:
                cluster_labels[j] += current_label
                if cluster_labels[j] > max_label:
                    max_label = cluster_labels[j]
        current_label = max_label + 1
    return max_label


@nb.njit(cache=True, parallel=True)
def get_cluster_representatives(
    clusters: np.ndarray,
    pairwise_indptr: np.ndarray,
    pairwise_indices: np.ndarray,
    pairwise_data: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Get indexes of the cluster representative spectra (medoids).

    Parameters
    ----------
    clusters : np.ndarray
        Cluster label assignments.
    pairwise_indptr : np.ndarray
        The index pointers for the nearest neighbor distances. See
        `scipy.sparse.csr_matrix`.
    pairwise_indices : np.ndarray
        The column indices for the nearest neighbor distances. See
        `scipy.sparse.csr_matrix`.
    pairwise_data : np.ndarray
        The nearest neighbor distances. See `scipy.sparse.csr_matrix` (`data`).

    Returns
    -------
    Optional[np.ndarray]
        The indexes of the medoid elements for all clusters.
    """
    # Find the indexes of the representatives for each unique cluster.
    # noinspection PyUnresolvedReferences
    order, min_i = np.argsort(clusters), 0
    cluster_idx, max_i = [], min_i
    while max_i < order.shape[0]:
        while (
            max_i < order.shape[0]
            and clusters[order[min_i]] == clusters[order[max_i]]
        ):
            max_i += 1
        cluster_idx.append((min_i, max_i))
        min_i = max_i
    representatives = np.empty(len(cluster_idx), np.uint)
    for i in nb.prange(len(cluster_idx)):
        representatives[i] = _get_cluster_medoid_index(
            order[cluster_idx[i][0] : cluster_idx[i][1]],
            pairwise_indptr,
            pairwise_indices,
            pairwise_data,
        )
    return representatives


@nb.njit(cache=True, fastmath=True)
def _get_cluster_medoid_index(
    cluster_mask: np.ndarray,
    pairwise_indptr: np.ndarray,
    pairwise_indices: np.ndarray,
    pairwise_data: np.ndarray,
) -> int:
    """
    Get the index of the cluster medoid element.

    Parameters
    ----------
    cluster_mask : np.ndarray
        Indexes of the items belonging to the current cluster.
    pairwise_indptr : np.ndarray
        The index pointers for the nearest neighbor distances. See
        `scipy.sparse.csr_matrix`.
    pairwise_indices : np.ndarray
        The column indices for the nearest neighbor distances. See
        `scipy.sparse.csr_matrix`.
    pairwise_data : np.ndarray
        The nearest neighbor distances. See `scipy.sparse.csr_matrix` (`data`).

    Returns
    -------
    int
        The index of the cluster's medoid element.
    """
    if len(cluster_mask) <= 2:
        # Pairwise distances will be identical.
        return cluster_mask[0]
    min_i, min_avg = 0, np.inf
    for row_i in range(cluster_mask.shape[0]):
        indices = pairwise_indices[
            pairwise_indptr[cluster_mask[row_i]] : pairwise_indptr[
                cluster_mask[row_i] + 1
            ]
        ]
        data = pairwise_data[
            pairwise_indptr[cluster_mask[row_i]] : pairwise_indptr[
                cluster_mask[row_i] + 1
            ]
        ]
        col_i = np.asarray(
            [
                i
                for cm in cluster_mask
                for i, ind in enumerate(indices)
                if cm == ind
            ]
        )
        # noinspection PyUnresolvedReferences
        row_avg = np.mean(data[col_i]) if len(col_i) > 0 else np.inf
        if row_avg < min_avg:
            min_i, min_avg = row_i, row_avg
    return cluster_mask[min_i]


def compute_condensed_distance_matrix(
    spec_tuples: List[similarity.SpectrumTuple],
    fragment_mz_tol: float,
) -> np.ndarray:
    """
    Compute the condensed pairwise distance matrix for the given spectra.

    Parameters
    ----------
    spec_tuples : List[similarity.SpectrumTuple]
        The spectra to compute the pairwise distance matrix for.
    fragment_mz_tolerance : float
        The fragment m/z tolerance.

    Returns
    -------
    np.ndarray
        The condensed pairwise distance matrix.
    """
    n = len(spec_tuples)
    condensed_dist_matrix = np.zeros(n * (n - 1) // 2)

    def worker(i, j):
        spec_tup1 = spec_tuples[i]
        spec_tup2 = spec_tuples[j]
        sim, _ = similarity.cosine_fast(spec_tup1, spec_tup2, fragment_mz_tol)
        distance = 1.0 - sim
        idx = condensed_index(i, j, n)
        condensed_dist_matrix[idx] = distance

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(worker, i, j)
            for i in range(n - 1)
            for j in range(i + 1, n)
        ]
        for future in futures:
            future.result()

    return condensed_dist_matrix


@nb.njit
def condensed_index(i: int, j: int, n: int) -> int:
    """
    Get the index of the condensed distance matrix.

    Parameters
    ----------
    i : int
        The row index.
    j : int
        The column index.
    n : int
        The number of spectra.

    Returns
    -------
    int
        The index of the condensed distance matrix.
    """
    if i == j:
        raise ValueError("No diagonal elements in condensed matrix")
    if i > j:
        i, j = j, i
    return int(n * i + j - ((i + 2) * (i + 1)) // 2)
