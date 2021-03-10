import logging
import math
import os
import pickle
from typing import Callable, Iterable, List, Tuple

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


logger = logging.getLogger('spectrum_clustering')


def compute_pairwise_distances(
        charge: int, n_spectra: int, mz_splits: np.ndarray,
        process_spectrum: Callable, vectorize: Callable,
        precursor_tol_mass: float, precursor_tol_mode: str, n_neighbors: int,
        n_neighbors_ann: int, batch_size: int, n_probe: int, work_dir: str) \
        -> Tuple[ss.csr_matrix, pd.DataFrame]:
    """
    Compute a pairwise distance matrix for the persisted spectra with the given
    precursor charge.

    Parameters
    ----------
    charge : int
        Precursor charge of the spectra to be processed.
    n_spectra: int
        The total number of spectra to be processed.
    mz_splits : np.ndarray
        M/z intervals used to split the spectra on their precursor m/z.
    process_spectrum : Callable
        Function to preprocess the spectra.
    vectorize : Callable
        Function to convert the spectra to vectors.
    precursor_tol_mass : float
        The precursor tolerance mass for vectors to be considered as neighbors.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
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
    work_dir : str
        Directory to read and store (intermediate) results.

    Returns
    -------
    Tuple[ss.csr_matrix, pd.DataFrame]
        A tuple with the sparse pairwise distance matrix containing the cosine
        distances between similar neighbors for the given vectors, and the
        corresponding spectrum metadata (identifier, precursor charge,
        precursor m/z).
    """
    n_probe, n_neighbors_ann = _check_ann_config(n_probe, n_neighbors_ann)
    logger.info('Compute pairwise distances between similar spectra '
                '(%d spectra, %d neighbors)', n_spectra, n_neighbors)
    max_num_embeddings = n_spectra * n_neighbors
    dtype = (np.int32 if max_num_embeddings < np.iinfo(np.int32).max
             else np.int64)
    distances = np.zeros(max_num_embeddings, np.float32)
    indices = np.zeros(max_num_embeddings, dtype)
    indptr = np.zeros(n_spectra + 1, dtype)
    # Create the ANN indexes (if this hasn't been done yet) and calculate
    # pairwise distances.
    metadata = _build_query_ann_index(
        charge, mz_splits, process_spectrum, vectorize, n_probe, batch_size,
        n_neighbors, n_neighbors_ann, precursor_tol_mass, precursor_tol_mode,
        distances, indices, indptr, work_dir)
    # Update the number of spectra because of skipped low-quality spectra.
    n_spectra = len(metadata)
    indptr = indptr[:n_spectra + 1]
    distances, indices = distances[:indptr[-1]], indices[:indptr[-1]]
    # Convert to a sparse pairwise distance matrix. This matrix might not be
    # entirely symmetrical, but that shouldn't matter too much.
    logger.debug('Construct pairwise distance matrix')
    pairwise_dist_matrix = ss.csr_matrix(
        (distances, indices, indptr), (n_spectra, n_spectra), np.float32,
        False)
    return pairwise_dist_matrix, metadata


def _check_ann_config(n_probe: int, n_neighbors: int) -> Tuple[int, int]:
    """
    Make sure that the configuration values adhere to the limitations imposed
    by running Faiss on a GPU.

    GPU indexes can only handle maximum 2048 probes and neighbors.
    https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU#limitations

    Parameters
    ----------
    """
    if n_probe > 2048:
        logger.warning('Using num_probe=2048 (maximum supported value for '
                       'GPU-enabled ANN indexing), %d was supplied', n_probe)
        n_probe = 2048
    if n_neighbors > 2048:
        logger.warning('Using num_neighbours=2048 (maximum supported value '
                       'for GPU-enabled ANN indexing), %d was supplied',
                       n_neighbors)
        n_neighbors = 2048
    return n_probe, n_neighbors


def _load_ann_index(index_filename: str, n_probe: int) -> faiss.Index:
    """
    Load the ANN index from the given file.

    Parameters
    ----------
    index_filename : str
        The ANN index filename.
    n_probe : int
        The number of cells to visit during ANN querying.

    Returns
    -------
    faiss.Index
        The Faiss `Index`.
    """
    index = faiss.read_index(index_filename)
    # IndexIVF has a `nprobe` hyperparameter, flat indexes don't.
    if hasattr(index, 'nprobe'):
        index.nprobe = min(math.ceil(index.nlist / 2), n_probe)
    return index


def _build_query_ann_index(
        charge: int, mz_splits: np.ndarray, process_spectrum: Callable,
        vectorize: Callable, n_probe: int, batch_size: int, n_neighbors: int,
        n_neighbors_ann: int, precursor_tol_mass: float,
        precursor_tol_mode: str, distances: np.ndarray, indices: np.ndarray,
        indptr: np.ndarray, work_dir: str) -> pd.DataFrame:
    """
    Create ANN index(es) for spectra with the given charge per precursor m/z
    split.

    Parameters
    ----------
    charge : int
        Precursor charge of the spectra to be processed.
    mz_splits : np.ndarray
        M/z splits used to create separate ANN indexes.
    process_spectrum : Callable
        Function to preprocess the spectra.
    vectorize : Callable
        Function to convert the spectra to vectors.
    batch_size : int
        The number of vectors to be simultaneously added to the index.
    work_dir : str
        Directory to read and store (intermediate) results.

    Returns
    -------
    pd.DataFrame
        Metadata (identifier, precursor charge, precursor m/z) of the spectra
        for which indexes were built.
    """
    identifiers, precursor_mzs = [], []
    indptr_i = 0
    # Find neighbors per specified precursor m/z interval.
    for mz in tqdm.tqdm(mz_splits, desc='Intervals queried', unit='index'):
        pkl_filename = os.path.join(work_dir, 'spectra', f'{charge}_{mz}.pkl')
        if not os.path.isfile(pkl_filename):
            continue
        # Read the spectra for the m/z split.
        spectra_split, precursor_mzs_split = [], []
        with open(pkl_filename, 'rb') as f_in:
            for spec in pickle.load(f_in):
                spec = process_spectrum(spec)
                # Discard low-quality spectra.
                if spec is not None:
                    spectra_split.append(spec)
                    identifiers.append(spec.identifier)
                    precursor_mzs_split.append(spec.precursor_mz)
        if len(spectra_split) == 0:
            continue
        precursor_mzs.append(np.asarray(precursor_mzs_split))
        # Convert the spectra to vectors.
        vectors_split = vectorize(spectra_split)
        n_split, dim = vectors_split.shape
        # Figure out a decent value for the n_list hyperparameter based on
        # the number of vectors.
        # Rules of thumb from the Faiss wiki:
        # https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index#how-big-is-the-dataset
        if n_split == 0:
            continue
        if n_split < 10e2:
            # Use a brute-force index instead of an ANN index when there
            # are only a few items.
            n_list = -1
        elif n_split < 10e5:
            n_list = 2**math.floor(math.log2(n_split / 39))
        elif n_split < 10e6:
            n_list = 2**16
        elif n_split < 10e7:
            n_list = 2**18
        else:
            n_list = 2**20
            if n_split > 10e8:
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
            index, vectors_split, precursor_mzs[-1], batch_size, n_neighbors,
            n_neighbors_ann, precursor_tol_mass, precursor_tol_mode,
            distances, indices, indptr, indptr_i)
        indptr_i += n_split
    return pd.DataFrame({'identifier': identifiers, 'precursor_charge': charge,
                         'precursor_mz': np.hstack(precursor_mzs)})


def _dist_mz_interval(
        index: faiss.Index, vectors: np.ndarray, precursor_mzs: np.ndarray,
        batch_size: int, n_neighbors: int, n_neighbors_ann: int,
        precursor_tol_mass: float, precursor_tol_mode: str,
        distances: np.ndarray, indices: np.ndarray, indptr: np.ndarray,
        indptr_i: int) -> None:
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
        batch_stop = min(batch_start + batch_size, index.ntotal)
        # Find nearest neighbors using ANN index searching.
        # noinspection PyArgumentList
        nn_dists, nn_idx_ann = index.search(vectors[batch_start:batch_stop],
                                            n_neighbors_ann)
        # Filter the neighbors based on the precursor m/z tolerance and assign
        # distances.
        _filter_neighbors_mz(
            precursor_mzs, batch_start, batch_stop, precursor_tol_mass,
            precursor_tol_mode, nn_dists, nn_idx_ann, n_neighbors, distances,
            indices, indptr, indptr_i + batch_start)


@nb.njit(parallel=True)
def _filter_neighbors_mz(
        precursor_mzs: np.ndarray, batch_start: int, batch_stop: int,
        precursor_tol_mass: float, precursor_tol_mode: str,
        nn_dists: np.ndarray, nn_idx_ann: np.ndarray, n_neighbors: int,
        distances: np.ndarray, indices: np.ndarray, indptr: np.ndarray,
        indptr_i: int) -> None:
    """
    Filter ANN neighbor indexes by precursor m/z tolerances and assign
    pairwise distances.

    Parameters
    ----------
    precursor_mzs : np.ndarray
        Precursor m/z's corresponding to the vectors.
    batch_start, batch_stop : int
        The indexes in the precursor m/z's of the current batch.
    precursor_tol_mass : float
        The precursor tolerance mass for vectors to be considered as neighbors.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
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
    indptr_i_start = indptr_i - batch_start
    for idx_ann, idx_mz, dists in zip(nn_idx_ann, nn_idx_mz, nn_dists):
        mask = _intersect_idx_ann_mz(idx_ann, idx_mz, n_neighbors)
        indptr[indptr_i + 1] = indptr[indptr_i] + len(mask)
        # Convert cosine similarity to cosine distance.
        distances[indptr[indptr_i]:indptr[indptr_i + 1]] = \
            np.maximum(1 - dists[mask], 0)
        indices[indptr[indptr_i]:indptr[indptr_i + 1]] = (indptr_i_start
                                                          + idx_ann[mask])
        indptr_i += 1


@nb.njit(parallel=True)
def _get_neighbors_idx(mzs: np.ndarray, start_i: int, stop_i: int,
                       precursor_tol_mass: float, precursor_tol_mode: str) \
        -> List[np.ndarray]:
    """
    Filter nearest neighbor candidates on precursor m/z.

    Parameters
    ----------
    mzs : np.ndarray
        The precursor m/z's of the nearest neighbor candidates.
    start_i, stop_i : int
        Indexes used to slice the m/z's to be considered in the batch
        (inclusive start_i, exclusive stop_i).
    precursor_tol_mass : float
        The precursor tolerance mass for vectors to be considered as neighbors.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').

    Returns
    -------
    List[np.ndarray]
        A list of NumPy arrays with the indexes of the nearest neighbor
        candidates for each item.
    """
    batch_mzs = mzs[start_i:stop_i].reshape((stop_i - start_i, 1))
    if precursor_tol_mode == 'Da':
        min_mz = batch_mzs[0] - precursor_tol_mass
        max_mz = batch_mzs[-1] + precursor_tol_mass
    elif precursor_tol_mode == 'ppm':
        min_mz = batch_mzs[0] - batch_mzs[0] * precursor_tol_mass / 10**6
        max_mz = batch_mzs[-1] + batch_mzs[-1] * precursor_tol_mass / 10**6
    else:
        raise ValueError('Unknown precursor tolerance filter')
    match_i = np.searchsorted(mzs, [min_mz[0], max_mz[0]])
    match_mzs = (mzs[match_i[0]:match_i[1]]
                 .reshape((1, match_i[1] - match_i[0])))
    match_mzs_i = np.arange(match_i[0], match_i[1])
    if precursor_tol_mode == 'Da':
        masks = np.abs(batch_mzs - match_mzs) < precursor_tol_mass
    elif precursor_tol_mode == 'ppm':
        masks = (np.abs(batch_mzs - match_mzs) / match_mzs * 10**6
                 < precursor_tol_mass)
    # noinspection PyUnboundLocalVariable
    return [match_mzs_i[mask] for mask in masks]


@nb.njit
def _intersect_idx_ann_mz(idx_ann: np.ndarray, idx_mz: np.ndarray,
                          max_neighbors: int) -> np.ndarray:
    """
    Find the intersection between identifiers from ANN filtering and precursor
    m/z filtering.

    Parameters
    ----------
    idx_ann : np.ndarray
        Identifiers from ANN filtering.
    idx_mz : np.ndarray
        Identifiers from precursor m/z filtering.
    max_neighbors : int
        The maximum number of best matching neighbors to retain.

    Returns
    -------
    np.ndarray
        A mask to select the joint identifiers in the `idx_ann` array.
    """
    idx_mz, i_mz = np.sort(idx_mz), 0
    idx_ann_order = np.argsort(idx_ann)
    idx_ann_intersect = []
    for i_order, i_ann in enumerate(idx_ann_order):
        if idx_ann[i_ann] != -1:
            while i_mz < len(idx_mz) and idx_mz[i_mz] < idx_ann[i_ann]:
                i_mz += 1
            if i_mz == len(idx_mz):
                break
            if idx_ann[i_ann] == idx_mz[i_mz]:
                idx_ann_intersect.append(i_order)
                i_mz += 1
    # FIXME: Sorting could be avoided here using np.argpartition, but this is
    #        currently not supported by Numba.
    #        https://github.com/numba/numba/issues/2445
    return (np.sort(idx_ann_order[np.asarray(idx_ann_intersect)])
            [:max_neighbors])


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
    logger.info('DBSCAN clustering (eps=%.4f, min_samples=%d) of precomputed '
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
         precursor_tol_mode) for start_i, stop_i in group_idx))
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
    while clusters[start_i] == -1:
        start_i += 1
    group_idx, stop_i = nb.typed.List(), start_i
    while stop_i < clusters.shape[0]:
        start_i, label = stop_i, clusters[stop_i]
        while stop_i < clusters.shape[0] and clusters[stop_i] == label:
            stop_i += 1
        group_idx.append((start_i, stop_i))
    return group_idx


def _postprocess_cluster(cluster_mzs: np.ndarray, precursor_tol_mass: float,
                         precursor_tol_mode: str) -> Tuple[np.ndarray, int]:
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

    Returns
    -------
    Tuple[np.ndarray, int]
        A tuple with cluster assignments starting at 0 and the number of
        clusters.
    """
    cluster_labels = -np.ones_like(cluster_mzs, np.int64)
    # No splitting possible if only 1 item in cluster.
    # This seems to happen sometimes despite that DBSCAN requires a higher
    # `min_samples`.
    if cluster_labels.shape[0] == 1:
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
                if mask.sum() > 1:
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


def get_cluster_representatives(clusters: np.ndarray,
                                pairwise_dist_matrix: ss.csr_matrix) \
        -> Iterable[Tuple[int, int]]:
    """
    Get indexes of the cluster representative spectra (medoids).

    Parameters
    ----------
    clusters : np.ndarray
        Cluster label assignments, excluding noise clusters.
    pairwise_dist_matrix : ss.csr_matrix
        Pairwise distance matrix.

    Returns
    -------
    List[int]
        The indexes of the medoid elements for all clusters.
    """
    labels = np.arange(np.amin(clusters[clusters != -1]),
                       np.amax(clusters) + 1)
    # noinspection PyTypeChecker
    yield from joblib.Parallel(n_jobs=-1, prefer='threads')(
        joblib.delayed(_get_cluster_medoid_index)(
            mask, pairwise_dist_matrix.indptr, pairwise_dist_matrix.indices,
            pairwise_dist_matrix.data)
        for mask in clusters.reshape(1, -1) == labels.reshape(-1, 1))


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
        A boolean mask array.
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
    cluster_mask = np.where(cluster_mask)[0]
    if len(cluster_mask) <= 2:
        # Pairwise distances will be identical.
        return cluster_mask[0]
    min_i, min_avg = 0, np.inf
    for row_i in range(cluster_mask.shape[0]):
        indices = pairwise_indices[pairwise_indptr[cluster_mask[row_i]]:
                                   pairwise_indptr[cluster_mask[row_i] + 1]]
        data = pairwise_data[pairwise_indptr[cluster_mask[row_i]]:
                             pairwise_indptr[cluster_mask[row_i] + 1]]
        row_sum, row_count = 0, 0
        for col_i in range(cluster_mask.shape[0]):
            for pairwise_i, ind in enumerate(indices):
                if ind == cluster_mask[col_i]:
                    break
            else:
                continue
            # noinspection PyUnboundLocalVariable
            row_sum += data[pairwise_i]
            row_count += 1
        # The pairwise distance matrix is not fully symmetrical so some values
        # might be missing. In the extreme case a point doesn't have any of the
        # other cluster members as its row neighbors (but connected as neighbor
        # of another point; i.e. column neighbor).
        if row_count > 0:
            row_avg = row_sum / row_count
            if row_avg < min_avg:
                min_i, min_avg = row_i, row_avg
    return cluster_mask[min_i]
