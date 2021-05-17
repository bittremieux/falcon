import logging
import math
import pickle
from typing import Callable, List, Optional, Tuple

import faiss
import fastcluster
import joblib
import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse as ss
import tqdm
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform
# noinspection PyProtectedMember
from sklearn.cluster._dbscan_inner import dbscan_inner
from sklearn.metrics import pairwise_distances


logger = logging.getLogger('falcon')


def compute_pairwise_distances(
        n_spectra: int, bucket_filenames: List[str],
        process_spectrum: Callable, vectorize: Callable,
        precursor_tol_mass: float, precursor_tol_mode: str, rt_tol: float,
        n_neighbors: int, n_neighbors_ann: int, batch_size: int, n_probe: int)\
        -> Tuple[ss.csr_matrix, pd.DataFrame]:
    """
    Compute a pairwise distance matrix for the persisted spectra with the given
    precursor charge.

    Parameters
    ----------
    n_spectra: int
        The total number of spectra to be processed.
    bucket_filenames : List[str]
        List of bucket file names.
    process_spectrum : Callable
        Function to preprocess the spectra.
    vectorize : Callable
        Function to convert the spectra to vectors.
    precursor_tol_mass : float
        The precursor tolerance mass for vectors to be considered as neighbors.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    rt_tol : float
        The retention time tolerance for vectors to be considered as neighbors.
        If `None`, do not filter neighbors on retention time.
    n_neighbors : int
        The final (maximum) number of neighbors to retrieve for each vector.
    n_neighbors_ann : int
        The number of neighbors to retrieve using the ANN index. This can
        exceed the final number of neighbors (`n_neighbors`) to maximize the
        number of neighbors within the precursor m/z tolerance.
    batch_size : int
        The number of vectors to be simultaneously processed.
    n_probe : int
        The number of cells to visit during ANN querying.

    Returns
    -------
    Tuple[ss.csr_matrix, pd.DataFrame]
        A tuple with the sparse pairwise distance matrix containing the cosine
        distances between similar neighbors for the given vectors, and the
        corresponding spectrum metadata (identifier, precursor charge,
        precursor m/z).
    """
    logger.debug('Compute nearest neighbor pairwise distances (%d spectra, %d'
                 ' neighbors)', n_spectra, n_neighbors)
    max_num_embeddings = n_spectra * n_neighbors
    dtype = (np.int32 if max_num_embeddings < np.iinfo(np.int32).max
             else np.int64)
    distances = np.zeros(max_num_embeddings, np.float32)
    indices = np.zeros(max_num_embeddings, dtype)
    indptr = np.zeros(n_spectra + 1, dtype)
    # Create the ANN indexes (if this hasn't been done yet) and calculate
    # pairwise distances.
    metadata = _build_query_ann_index(
        bucket_filenames, process_spectrum, vectorize, n_probe, batch_size,
        n_neighbors, n_neighbors_ann, precursor_tol_mass, precursor_tol_mode,
        rt_tol, distances, indices, indptr)
    # Update the number of spectra because of skipped low-quality spectra.
    n_spectra = len(metadata)
    indptr = indptr[:n_spectra + 1]
    distances, indices = distances[:indptr[-1]], indices[:indptr[-1]]
    # Convert to a sparse pairwise distance matrix. This matrix might not be
    # entirely symmetrical, but that shouldn't matter too much.
    logger.debug('Construct %d-by-%d sparse pairwise distance matrix with %d '
                 'non-zero values', n_spectra, n_spectra, len(distances))
    pairwise_dist_matrix = ss.csr_matrix(
        (distances, indices, indptr), (n_spectra, n_spectra), np.float32,
        False)
    return pairwise_dist_matrix, metadata


def _build_query_ann_index(
        bucket_filenames: List[str], process_spectrum: Callable,
        vectorize: Callable, n_probe: int, batch_size: int, n_neighbors: int,
        n_neighbors_ann: int, precursor_tol_mass: float,
        precursor_tol_mode: str, rt_tol: float, distances: np.ndarray,
        indices: np.ndarray, indptr: np.ndarray) \
        -> pd.DataFrame:
    """
    Create ANN index(es) for spectra with the given charge per precursor m/z
    split.

    Parameters
    ----------
    bucket_filenames : List[str]
        List of bucket file names.
    process_spectrum : Callable
        Function to preprocess the spectra.
    vectorize : Callable
        Function to convert the spectra to vectors.
    n_probe : int
        The number of cells to consider during NN index querying.
    batch_size : int
        The number of vectors to be simultaneously added to the index.
    n_neighbors : int
        The final (maximum) number of neighbors to retrieve for each vector.
    n_neighbors_ann : int
        The number of neighbors to retrieve using the ANN index. This can
        exceed the final number of neighbors (`n_neighbors`) to maximize the
        number of neighbors within the precursor m/z tolerance.
    precursor_tol_mass : float
        The precursor tolerance mass for vectors to be considered as neighbors.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    rt_tol : float
        The retention time tolerance for vectors to be considered as neighbors.
        If `None`, do not filter neighbors on retention time.
    distances : np.ndarray
        The nearest neighbor distances. See `scipy.sparse.csr_matrix` (`data`).
    indices : np.ndarray
        The column indices for the nearest neighbor distances. See
        `scipy.sparse.csr_matrix`.
    indptr : np.ndarray
        The index pointers for the nearest neighbor distances. See
        `scipy.sparse.csr_matrix`.

    Returns
    -------
    pd.DataFrame
        Metadata (identifier, precursor charge, precursor m/z, retention time)
        of the spectra for which indexes were built.
    """
    identifiers, precursor_mzs, rts = [], [], []
    indptr_i = 0
    # Find neighbors per specified precursor m/z interval.
    for pkl_filename in tqdm.tqdm(bucket_filenames, desc='Buckets queried',
                                  unit='bucket'):
        # Read the spectra for the m/z split.
        spectra_split, precursor_mzs_split, rts_split = [], [], []
        with open(pkl_filename, 'rb') as f_in:
            for spec_raw in pickle.load(f_in):
                spec_processed = process_spectrum(spec_raw)
                # Discard low-quality spectra.
                if spec_processed is not None:
                    spectra_split.append(spec_processed)
                    identifiers.append(spec_processed.identifier)
                    precursor_mzs_split.append(spec_processed.precursor_mz)
                    rts_split.append(spec_processed.retention_time)
        if len(spectra_split) == 0:
            continue
        precursor_mzs.append(np.asarray(precursor_mzs_split))
        rts.append(np.asarray(rts_split))
        # Convert the spectra to vectors.
        vectors_split = vectorize(spectra_split)
        n_split, dim = vectors_split.shape
        # Figure out a decent value for the n_list hyperparameter based on
        # the number of vectors.
        # Rules of thumb from the Faiss wiki:
        # https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index#how-big-is-the-dataset
        if n_split == 0:
            continue
        if n_split < 10**2:
            # Use a brute-force index instead of an ANN index when there
            # are only a few items.
            n_list = -1
        elif n_split < 10**6:
            n_list = 2**math.floor(math.log2(n_split / 39))
        elif n_split < 10**7:
            n_list = 2**16
        elif n_split < 10**8:
            n_list = 2**18
        else:
            n_list = 2**20
            if n_split > 10**9:
                logger.warning('More than 1B vectors to be indexed, consider '
                               'decreasing the ANN size')
        # Create an ANN index using the inner product (proxy for cosine
        # distance) for fast NN queries.
        if n_list <= 0:
            index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        else:
            index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, n_list,
                                       faiss.METRIC_INNER_PRODUCT)
            index.nprobe = min(math.ceil(index.nlist / 8), n_probe)
        # Compute cluster centroids.
        # noinspection PyArgumentList
        index.train(vectors_split)
        # Add the vectors to the index in batches.
        batch_size = min(n_split, batch_size)
        for batch_start in range(0, n_split, batch_size):
            batch_stop = min(batch_start + batch_size, n_split)
            # noinspection PyArgumentList
            index.add_with_ids(vectors_split[batch_start:batch_stop],
                               np.arange(batch_start, batch_stop))
        # Query the index to calculate NN distances.
        _dist_mz_interval(
            index, vectors_split, precursor_mzs[-1], rts[-1], batch_size,
            n_neighbors, n_neighbors_ann, precursor_tol_mass,
            precursor_tol_mode, rt_tol, distances, indices, indptr, indptr_i)
        index.reset()
        indptr_i += n_split
    if len(identifiers) == 0:
        return pd.DataFrame(columns=['identifier', 'precursor_mz',
                                     'retention_time'])
    else:
        return pd.DataFrame({'identifier': identifiers,
                             'precursor_mz': np.hstack(precursor_mzs),
                             'retention_time': np.hstack(rts)})


def _dist_mz_interval(
        index: faiss.Index, vectors: np.ndarray, precursor_mzs: np.ndarray,
        rts: np.ndarray, batch_size: int, n_neighbors: int,
        n_neighbors_ann: int, precursor_tol_mass: float,
        precursor_tol_mode: str, rt_tol: float, distances: np.ndarray,
        indices: np.ndarray, indptr: np.ndarray, indptr_i: int) -> None:
    """
    Compute distances to the nearest neighbors for the given precursor m/z
    interval.

    Parameters
    ----------
    index : faiss.Index
        The NN index used to efficiently find distances to similar spectra.
    vectors : np.ndarray
        The spectrum vectors to be queried against the NN index.
    precursor_mzs : np.ndarray
        Precorsor m/z's of the spectra corresponding to the given vectors.
    rts : np.ndarray
        Retention times corresponding to the vectors.
    batch_size : int
        The number of vectors to be simultaneously queried.
    n_neighbors : int
        The final (maximum) number of neighbors to retrieve for each vector.
    n_neighbors_ann : int
        The number of neighbors to retrieve using the ANN index. This can
        exceed the final number of neighbors (`n_neighbors`) to maximize the
        number of neighbors within the precursor m/z tolerance.
    precursor_tol_mass : float
        The precursor tolerance mass for vectors to be considered as neighbors.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    rt_tol : float
        The retention time tolerance for vectors to be considered as neighbors.
        If `None`, do not filter neighbors on retention time.
    distances : np.ndarray
        The nearest neighbor distances. See `scipy.sparse.csr_matrix` (`data`).
    indices : np.ndarray
        The column indices for the nearest neighbor distances. See
        `scipy.sparse.csr_matrix`.
    indptr : np.ndarray
        The index pointers for the nearest neighbor distances. See
        `scipy.sparse.csr_matrix`.
    indptr_i : int
        The current start index in `indptr`.
    """
    for batch_start in range(0, vectors.shape[0], batch_size):
        batch_stop = min(batch_start + batch_size, vectors.shape[0])
        # Find nearest neighbors using ANN index searching.
        # noinspection PyArgumentList
        nn_dists, nn_idx_ann = index.search(vectors[batch_start:batch_stop],
                                            n_neighbors_ann)
        # Filter the neighbors based on the precursor m/z tolerance and assign
        # distances.
        _filter_neighbors_mz(
            precursor_mzs, rts, batch_start, batch_stop, precursor_tol_mass,
            precursor_tol_mode, rt_tol, nn_dists, nn_idx_ann, n_neighbors,
            distances, indices, indptr, indptr_i + batch_start)


@nb.njit
def _filter_neighbors_mz(
        precursor_mzs: np.ndarray, rts: np.ndarray, batch_start: int,
        batch_stop: int, precursor_tol_mass: float, precursor_tol_mode: str,
        rt_tol: float, nn_dists: np.ndarray, nn_idx_ann: np.ndarray,
        n_neighbors: int, distances: np.ndarray, indices: np.ndarray,
        indptr: np.ndarray, indptr_i: int) -> None:
    """
    Filter ANN neighbor indexes by precursor m/z tolerances and assign
    pairwise distances.

    Parameters
    ----------
    precursor_mzs : np.ndarray
        Precursor m/z's corresponding to the vectors.
    rts : np.ndarray
        Retention times corresponding to the vectors.
    batch_start, batch_stop : int
        The indexes in the precursor m/z's of the current batch.
    precursor_tol_mass : float
        The precursor tolerance mass for vectors to be considered as neighbors.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    rt_tol : float
        The retention time tolerance for vectors to be considered as neighbors.
        If `None`, do not filter neighbors on retention time.
    nn_dists : np.ndarray
        Distances of the nearest neighbors.
    nn_idx_ann : np.ndarray
        Indexes of the nearest neighbors.
    n_neighbors : int
        The (maximum) number of neighbors to set for each vector.
    distances : np.ndarray
        The nearest neighbor distances. See `scipy.sparse.csr_matrix` (`data`).
    indices : np.ndarray
        The column indices for the nearest neighbor distances. See
        `scipy.sparse.csr_matrix`.
    indptr : np.ndarray
        The index pointers for the nearest neighbor distances. See
        `scipy.sparse.csr_matrix`.
    indptr_i : int
        The current start index in `indptr`.
    """
    nn_idx_mz = _get_neighbors_idx(
        precursor_mzs, batch_start, batch_stop,
        precursor_tol_mass, precursor_tol_mode)
    if rt_tol is not None:
        nn_idx_rt = _get_neighbors_idx(rts, batch_start, batch_stop, rt_tol,
                                       'rt')
        nn_idx_mz = [_intersect_idx_ann_mz(idx_mz, idx_rt, None, True)
                     for idx_mz, idx_rt in zip(nn_idx_mz, nn_idx_rt)]
    indptr_i_start = indptr_i - batch_start
    for i, (idx_ann, idx_mz, dists) in enumerate(
            zip(nn_idx_ann, nn_idx_mz, nn_dists), indptr_i):
        mask = _intersect_idx_ann_mz(idx_ann, idx_mz, n_neighbors)
        indptr[i + 1] = indptr[i] + len(mask)
        # Convert cosine similarity to cosine distance.
        distances[indptr[i]:indptr[i + 1]] = np.maximum(1 - dists[mask], 0)
        indices[indptr[i]:indptr[i + 1]] = indptr_i_start + idx_ann[mask]


@nb.njit
def _get_neighbors_idx(values: np.ndarray, start_i: int, stop_i: int,
                       tol: float, tol_mode: str) \
        -> List[np.ndarray]:
    """
    Filter nearest neighbor candidates on precursor m/z or retention time.

    Parameters
    ----------
    values : np.ndarray
        The precursor m/z or retention time values of the candidates.
    start_i, stop_i : int
        Indexes used to slice the values to be considered in the batch
        (inclusive start_i, exclusive stop_i).
    tol : float
        The tolerance for vectors to be considered as neighbors.
    tol_mode : str
        The unit of the tolerance ('Da' or 'ppm' for precursor m/z, 'rt' for
        retention time).

    Returns
    -------
    List[np.ndarray]
        A list of sorted NumPy arrays with the indexes of the nearest neighbor
        candidates for each item.
    """
    batch_values = values[start_i:stop_i]
    if tol_mode == 'Da':
        min_value = batch_values[0] - tol
        max_value = batch_values[-1] + tol
    elif tol_mode == 'ppm':
        min_value = batch_values[0] - batch_values[0] * tol / 10 ** 6
        max_value = batch_values[-1] + batch_values[-1] * tol / 10 ** 6
    elif tol_mode == 'rt':
        # RT values are not sorted.
        min_value = batch_values.min() - tol
        max_value = batch_values.max() + tol
    else:
        raise ValueError('Unknown tolerance filter')
    min_value, max_value = max(0, min_value), max(0, max_value)
    match_i = np.searchsorted(values, [min_value, max_value])
    match_values = (values[match_i[0]:match_i[1]]
                    .reshape((1, match_i[1] - match_i[0])))
    match_values_i = np.arange(match_i[0], match_i[1])
    batch_values = batch_values.reshape((stop_i - start_i, 1))
    if tol_mode in ('Da', 'rt'):
        masks = np.abs(batch_values - match_values) < tol
    elif tol_mode == 'ppm':
        masks = (np.abs(batch_values - match_values) / match_values * 10 ** 6
                 < tol)
    # noinspection PyUnboundLocalVariable
    return [np.sort(match_values_i[mask]) for mask in masks]


@nb.njit
def _intersect_idx_ann_mz(idx_ann: np.ndarray, idx_mz: np.ndarray,
                          max_neighbors: int = None, is_sorted: bool = False) \
        -> np.ndarray:
    """
    Find the intersection between identifiers from ANN filtering and precursor
    m/z filtering.

    Parameters
    ----------
    idx_ann : np.ndarray
        Identifiers from ANN filtering.
    idx_mz : np.ndarray
        SORTED identifiers from precursor m/z filtering.
    max_neighbors : int
        The maximum number of best matching neighbors to retain.
    is_sorted : bool
        Flag indicating whether the first array is sorted or not.

    Returns
    -------
    np.ndarray
        A mask to select the joint identifiers in the `idx_ann` array.
    """
    i_mz = 0
    idx_ann_order = (np.argsort(idx_ann) if not is_sorted
                     else np.arange(len(idx_ann)))
    idx = []
    for i_order, i_ann in enumerate(idx_ann_order):
        if idx_ann[i_ann] != -1:
            while i_mz < len(idx_mz) and idx_mz[i_mz] < idx_ann[i_ann]:
                i_mz += 1
            if i_mz == len(idx_mz):
                break
            if idx_ann[i_ann] == idx_mz[i_mz]:
                idx.append(idx_ann_order[i_order])
                i_mz += 1
    idx = np.asarray(idx)
    if max_neighbors is None or max_neighbors >= len(idx):
        return idx
    else:
        return np.partition(idx, max_neighbors)[:max_neighbors]


def generate_clusters(pairwise_dist_matrix: ss.csr_matrix, eps: float,
                      min_samples: int, precursor_mzs: np.ndarray,
                      precursor_tol_mass: float, precursor_tol_mode: str) \
        -> np.ndarray:
    """
    DBSCAN clustering of the given pairwise distance matrix.

    Parameters
    ----------
    pairwise_dist_matrix : ss.csr_matrix
        A sparse pairwise distance matrix used for clustering.
    eps : float
        The maximum distance between two samples for one to be considered as in
        the neighborhood of the other.
    min_samples : int
        The number of samples in a neighborhood for a point to be considered as
        a core point. This includes the point itself.
    precursor_mzs : np.ndarray
        Precursor m/z's matching the pairwise distance matrix.
    precursor_tol_mass : float
        Maximum precursor mass tolerance for points to be clustered together.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').

    Returns
    -------
    np.ndarray
        Cluster labels. Noisy samples are given the label -1.
    """
    # DBSCAN clustering using the precomputed pairwise distance matrix.
    logger.debug('DBSCAN clustering (eps=%.4f, min_samples=%d) of precomputed '
                 'pairwise distance matrix', eps, min_samples)
    # Reimplement DBSCAN preprocessing to avoid unnecessary memory consumption.
    # Find the eps-neighborhoods for all points.
    mask = pairwise_dist_matrix.data <= eps
    indices = pairwise_dist_matrix.indices[mask].astype(np.intp)
    indptr = np.zeros(len(mask) + 1, dtype=np.int64)
    np.cumsum(mask, out=indptr[1:])
    indptr = indptr[pairwise_dist_matrix.indptr]
    neighborhoods = np.split(indices, indptr[1:-1])
    # Initially, all samples are noise.
    clusters = np.full(pairwise_dist_matrix.shape[0], -1, dtype=np.intp)
    # A list of all core samples found.
    n_neighbors = np.fromiter(map(len, neighborhoods), np.uint32)
    core_samples = n_neighbors >= min_samples
    # Run Scikit-Learn DBSCAN.
    neighborhoods_arr = np.empty(len(neighborhoods), dtype=np.object)
    neighborhoods_arr[:] = neighborhoods
    dbscan_inner(core_samples, neighborhoods_arr, clusters)
    # Refine initial clusters to make sure spectra within a cluster don't have
    # an excessive precursor m/z difference.
    order = np.argsort(clusters)
    reverse_order = np.argsort(order)
    clusters, precursor_mzs = clusters[order], precursor_mzs[order]
    logger.debug('Finetune %d initial unique clusters to not exceed %.2f %s '
                 'precursor m/z tolerance', clusters[-1] + 1,
                 precursor_tol_mass, precursor_tol_mode)
    group_idx = _get_cluster_group_idx(clusters)
    if len(group_idx) == 0:     # Only noise samples.
        return -np.ones_like(precursor_mzs, dtype=np.int64)
    cluster_reassignments = nb.typed.List(joblib.Parallel(n_jobs=-1)(
        joblib.delayed(_postprocess_cluster)
        (precursor_mzs[start_i:stop_i], precursor_tol_mass,
         precursor_tol_mode, min_samples) for start_i, stop_i in group_idx))
    clusters = _assign_unique_cluster_labels(
        group_idx, cluster_reassignments, min_samples)[reverse_order]
    logger.debug('%d unique clusters after precursor m/z finetuning',
                 np.amax(clusters) + 1)
    return clusters


@nb.njit
def _get_cluster_group_idx(clusters: np.ndarray) -> nb.typed.List:
    """
    Get start and stop indexes for unique cluster labels.

    Parameters
    ----------
    clusters : np.ndarray
        The ordered cluster labels (noise points are -1).

    Returns
    -------
    nb.typed.List[Tuple[int, int]]
        Tuples with the start index (inclusive) and end index (exclusive) of
        the unique cluster labels.
    """
    start_i = 0
    while clusters[start_i] == -1 and start_i < clusters.shape[0]:
        start_i += 1
    group_idx, stop_i = nb.typed.List(), start_i
    while stop_i < clusters.shape[0]:
        start_i, label = stop_i, clusters[stop_i]
        while stop_i < clusters.shape[0] and clusters[stop_i] == label:
            stop_i += 1
        group_idx.append((start_i, stop_i))
    return group_idx


def _postprocess_cluster(cluster_mzs: np.ndarray, precursor_tol_mass: float,
                         precursor_tol_mode: str, min_samples: int) \
        -> Tuple[np.ndarray, int]:
    """
    Agglomerative clustering of the precursor m/z's within each initial
    cluster to avoid that spectra within a cluster have an excessive precursor
    m/z difference.

    Parameters
    ----------
    cluster_mzs : np.ndarray
        Precursor m/z's of the samples in a single initial cluster.
    precursor_tol_mass : float
        Maximum precursor mass tolerance for points to be clustered together.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    min_samples : int
        The minimum number of samples in a cluster.

    Returns
    -------
    Tuple[np.ndarray, int]
        A tuple with cluster assignments starting at 0 and the number of
        clusters.
    """
    cluster_labels = -np.ones_like(cluster_mzs, np.int64)
    # No splitting needed if there are too few items in cluster.
    # This seems to happen sometimes despite that DBSCAN requires a higher
    # `min_samples`.
    if cluster_labels.shape[0] < min_samples:
        n_clusters = 0
    else:
        cluster_mzs = cluster_mzs.reshape(-1, 1)
        # Pairwise differences in Dalton.
        pairwise_mz_diff = pairwise_distances(cluster_mzs)
        if precursor_tol_mode == 'ppm':
            pairwise_mz_diff = pairwise_mz_diff / cluster_mzs * 10**6
        # Group items within the cluster based on their precursor m/z.
        # Precursor m/z's within a single group can't exceed the specified
        # precursor m/z tolerance (`distance_threshold`).
        # Subtract 1 because fcluster starts with cluster label 1 instead of 0
        # (like scikit-learn does).
        cluster_assignments = fcluster(
            fastcluster.linkage(
                squareform(pairwise_mz_diff, checks=False), 'complete'),
            precursor_tol_mass, 'distance') - 1
        n_clusters = cluster_assignments.max() + 1
        # Update cluster assignments.
        if n_clusters == 1:
            # Single homogeneous cluster.
            cluster_labels[:] = 0
        elif n_clusters == cluster_mzs.shape[0]:
            # Only singletons.
            n_clusters = 0
        else:
            cluster_assignments = cluster_assignments.reshape(1, -1)
            label, labels = 0, np.arange(n_clusters).reshape(-1, 1)
            # noinspection PyTypeChecker
            for mask in cluster_assignments == labels:
                if mask.sum() >= min_samples:
                    cluster_labels[mask] = label
                    label += 1
            n_clusters = label
    return cluster_labels, n_clusters


@nb.njit
def _assign_unique_cluster_labels(group_idx: nb.typed.List,
                                  cluster_reassignments: nb.typed.List,
                                  min_samples: int) -> np.ndarray:
    """
    Make sure all cluster labels are unique after potential splitting of
    clusters to avoid excessive precursor m/z differences.

    Parameters
    ----------
    group_idx : nb.typed.List[Tuple[int, int]]
        Tuples with the start index (inclusive) and end index (exclusive) of
        the unique cluster labels.
    cluster_reassignments : nb.typed.List[Tuple[np.ndarray, int]]
        Tuples with cluster assignments starting at 0 and the number of
        clusters.
    min_samples : int
        The minimum number of samples in a cluster.

    Returns
    -------
    np.ndarray
        An array with globally unique cluster labels.
    """
    clusters, current_label = -np.ones(group_idx[-1][1], np.int64), 0
    for (start_i, stop_i), (cluster_reassignment, n_clusters) in zip(
            group_idx, cluster_reassignments):
        if n_clusters > 0 and stop_i - start_i >= min_samples:
            clusters[start_i:stop_i] = cluster_reassignment + current_label
            current_label += n_clusters
    return clusters


@nb.njit(parallel=True)
def get_cluster_representatives(clusters: np.ndarray,
                                pairwise_indptr: np.ndarray,
                                pairwise_indices: np.ndarray,
                                pairwise_data: np.ndarray) \
        -> Optional[np.ndarray]:
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
        The indexes of the medoid elements for all non-noise clusters, or None
        if only noise clusters are present.
    """
    order, min_i = np.argsort(clusters), 0
    while min_i < clusters.shape[0] and clusters[order[min_i]] == -1:
        min_i += 1
    # Only noise clusters.
    if min_i == clusters.shape[0]:
        return None
    # Find the indexes of the representatives for each unique cluster.
    cluster_idx, max_i = [], min_i
    while max_i < order.shape[0]:
        while (max_i < order.shape[0] and
               clusters[order[min_i]] == clusters[order[max_i]]):
            max_i += 1
        cluster_idx.append((min_i, max_i))
        min_i = max_i
    representatives = np.empty(len(cluster_idx), np.uint)
    for i in nb.prange(len(cluster_idx)):
        representatives[i] = _get_cluster_medoid_index(
            order[cluster_idx[i][0]:cluster_idx[i][1]], pairwise_indptr,
            pairwise_indices, pairwise_data)
    return representatives


@nb.njit(fastmath=True)
def _get_cluster_medoid_index(cluster_mask: np.ndarray,
                              pairwise_indptr: np.ndarray,
                              pairwise_indices: np.ndarray,
                              pairwise_data: np.ndarray) -> int:
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
        indices = pairwise_indices[pairwise_indptr[cluster_mask[row_i]]:
                                   pairwise_indptr[cluster_mask[row_i] + 1]]
        data = pairwise_data[pairwise_indptr[cluster_mask[row_i]]:
                             pairwise_indptr[cluster_mask[row_i] + 1]]
        col_i = np.asarray([i for cm in cluster_mask
                            for i, ind in enumerate(indices) if cm == ind])
        row_avg = np.mean(data[col_i]) if len(col_i) > 0 else np.inf
        if row_avg < min_avg:
            min_i, min_avg = row_i, row_avg
    return cluster_mask[min_i]
