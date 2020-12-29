import logging
import math
import os
from typing import Any, List, Optional, Sequence, Tuple

import faiss
import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse as ss
import tqdm


logger = logging.getLogger('spectrum_clustering')


def compute_pairwise_distances(
        vectors: np.ndarray, precursor_mzs: np.ndarray,
        precursor_tol_mass: float, precursor_tol_mode: str, mz_interval: float,
        n_neighbors: int, n_neighbors_ann: int, mz_margin: float,
        mz_margin_mode: str, batch_size: int, n_probe: int,
        work_dir: str = '/tmp') -> ss.csr_matrix:
    """
    Compute a pairwise distance matrix for the given cluster vectors.

    The given vectors and precursor m/z's MUST be sorted by ascending precursor
    m/z.

    Parameters
    ----------
    vectors : np.ndarray
        The vectors for which to compute pairwise distances.
    precursor_mzs : np.ndarray
        Precursor m/z's corresponding to the vectors.
    mz_interval : float
        The width of the m/z interval.
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
    mz_margin : float
        The m/z margin to create slightly overlapping intervals to avoid
        missing edge neighbors.
    mz_margin_mode : str
        The unit of the m/z margin ('Da' or 'ppm'). If not 'Da' or 'ppm' no
        margin around the m/z intervals will be used.
    batch_size : int
        The number of vectors to be simultaneously processed.
    n_probe : int
        The number of cells to visit during ANN querying.
    work_dir : str
        Directory to store temporary results such as the ANN indexes.

    Returns
    -------
    ss.csr_matrix
        A sparse pairwise distance matrix containing the cosine distance
        between similar neighbors in the given vectors.
    """
    if not _is_sorted(precursor_mzs):
        raise ValueError("The precursor m/z's (and the corresponding vectors) "
                         "must be supplied in sorted order")
    n_probe, n_neighbors_ann = _check_ann_config(n_probe, n_neighbors_ann)
    ann_dir = os.path.join(work_dir, 'ann')
    os.makedirs(ann_dir, exist_ok=True)
    index_filename = os.path.join(ann_dir, 'ann_{}.faiss')
    sparse_filename = os.path.join(ann_dir, '{}.npy')
    mz_splits = np.arange(
        math.floor(np.amin(precursor_mzs) / mz_interval) * mz_interval,
        math.ceil(np.max(precursor_mzs) / mz_interval) * mz_interval,
        mz_interval)
    # Normalize the vectors for inner product search.
    faiss.normalize_L2(vectors)
    # Create the ANN indexes (if this hasn't been done yet).
    _build_ann_index(vectors, precursor_mzs, index_filename, mz_splits,
                     mz_interval, mz_margin, mz_margin_mode, batch_size)
    # Calculate pairwise distances.
    logger.info('Compute pairwise distances between similar vectors '
                '(%d vectors, %d neighbors)', vectors.shape[0], n_neighbors)
    if vectors.shape[0] > np.iinfo(np.int64).max:
        raise OverflowError('Too many vectors to fit into int64')
    if (not os.path.isfile(sparse_filename.format('data')) or
            not os.path.isfile(sparse_filename.format('indices')) or
            not os.path.isfile(sparse_filename.format('indptr'))):
        max_num_embeddings = vectors.shape[0] * n_neighbors
        distances = np.zeros(max_num_embeddings, np.float32)
        indices = np.zeros(max_num_embeddings, np.int64)
        indptr = np.zeros(vectors.shape[0] + 1, np.int64)
        for mz in tqdm.tqdm(mz_splits, desc='Distances calculated',
                            unit='index'):
            _dist_mz_interval(
                index_filename.format(mz), n_probe, vectors, precursor_mzs,
                mz, mz_interval, batch_size, n_neighbors, n_neighbors_ann,
                precursor_tol_mass, precursor_tol_mode, distances, indices,
                indptr)
        distances, indices = distances[:indptr[-1]], indices[:indptr[-1]]
        np.save(sparse_filename.format('data'), distances)
        np.save(sparse_filename.format('indices'), indices)
        np.save(sparse_filename.format('indptr'), indptr)
    else:
        distances = np.load(sparse_filename.format('data'))
        indices = np.load(sparse_filename.format('indices'))
        indptr = np.load(sparse_filename.format('indptr'))
    # Convert to a sparse pairwise distance matrix. This matrix might not be
    # entirely symmetrical, but that shouldn't matter too much.
    logger.debug('Construct pairwise distance matrix')
    pairwise_dist_matrix = ss.csr_matrix(
        (distances, indices, indptr), (vectors.shape[0], vectors.shape[0]),
        np.float32, False)
    ss.save_npz(os.path.join(ann_dir, 'dist.npz'), pairwise_dist_matrix)
    os.remove(sparse_filename.format('data'))
    os.remove(sparse_filename.format('indices'))
    os.remove(sparse_filename.format('indptr'))
    return pairwise_dist_matrix


@ nb.njit
def _is_sorted(values: Sequence[Any]):
    """
    Checks whether the given sequence is sorted in ascending order.

    Parameters
    ----------
    values : Sequence[Any]
        The values which order is checked.

    Returns
    -------
    True if the values are sorted, False otherwise.
    """
    for i in range(1, len(values)):
        if values[i - 1] > values[i]:
            return False
    return True


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


def _build_ann_index(vectors: np.ndarray, precursor_mzs: np.ndarray,
                     index_filename: str, mz_splits: np.ndarray,
                     mz_interval: float, mz_margin: float, mz_margin_mode: str,
                     batch_size: int) -> None:
    """
    Create ANN index(es) for the given vectors.

    Vectors will be split over multiple ANN indexes based on the given m/z
    interval.

    Parameters
    ----------
    vectors : np.ndarray
        The vectors to build the ANN index.
    precursor_mzs : np.ndarray
        Precursor m/z's corresponding to the vectors, used to split them over
        multiple ANN indexes per m/z interval.
    index_filename : str
        Base file name of the ANN index. Separate indexes for the given m/z
        splits will be created.
    mz_splits : np.ndarray
        M/z splits used to create separate ANN indexes.
    mz_interval : float
        The width of the m/z interval.
    mz_margin : float
        The m/z margin to create slightly overlapping intervals to avoid
        missing edge neighbors.
    mz_margin_mode : str
        The unit of the m/z margin ('Da' or 'ppm'). If not 'Da' or 'ppm' no
        margin around the m/z intervals will be used.
    batch_size : int
        The number of vectors to be simultaneously added to the index.
    """
    logger.info('Use %d GPUs for ANN index construction', faiss.get_num_gpus())
    # Create separate indexes per specified precursor m/z interval.
    for mz in tqdm.tqdm(mz_splits, desc='Indexes built', unit='index'):
        if os.path.isfile(index_filename.format(mz)):
            continue
        # Create an ANN index using the inner product (proxy for cosine
        # distance) for fast NN queries.
        start_i, stop_i = _get_precursor_mz_interval_i(
            precursor_mzs, mz, mz_interval, mz_margin, mz_margin_mode)
        n_vectors_split = stop_i - start_i
        # Figure out a decent value for the n_list hyperparameter based on
        # the number of vectors.
        # Rules of thumb from the Faiss wiki:
        # https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index#how-big-is-the-dataset
        if n_vectors_split == 0:
            continue
        if n_vectors_split < 10e2:
            # Use a brute-force index instead of an ANN index when there
            # are only a few items.
            n_list = -1
        elif n_vectors_split < 10e5:
            n_list = 2**math.floor(math.log2(n_vectors_split / 39))
        elif n_vectors_split < 10e6:
            n_list = 2**16
        elif n_vectors_split < 10e7:
            n_list = 2**18
        else:
            n_list = 2**20
            if n_vectors_split > 10e8:
                logger.warning('More than 1B vectors to be indexed, consider '
                               'decreasing the ANN size')
        logger.debug('Build the ANN index for precursor m/z %d–%d '
                     '(%d vector, %d lists)', int(mz), int(mz + mz_interval),
                     n_vectors_split, n_list)
        # Create a suitable index and compute cluster centroids.
        if n_list <= 0:
            index = faiss.IndexIDMap(faiss.IndexFlatIP(vectors.shape[1]))
        else:
            index = faiss.IndexIVFFlat(faiss.IndexFlatIP(vectors.shape[1]),
                                       vectors.shape[1], n_list,
                                       faiss.METRIC_INNER_PRODUCT)
        # noinspection PyArgumentList
        index.train(vectors[start_i:stop_i])
        # Add the vectors to the index in batches.
        logger.debug('Add %d vectors to the ANN index', n_vectors_split)
        batch_size = min(n_vectors_split, batch_size)
        for batch_start in range(start_i, stop_i, batch_size):
            batch_stop = min(batch_start + batch_size, stop_i)
            # noinspection PyArgumentList
            index.add_with_ids(vectors[batch_start:batch_stop],
                               np.arange(batch_start, batch_stop))
        # Save the index to disk.
        logger.debug('Save the ANN index to file %s',
                     index_filename.format(mz))
        faiss.write_index(index, index_filename.format(mz))
        index.reset()


@nb.njit
def _get_precursor_mz_interval_i(precursor_mzs: np.ndarray, start_mz: float,
                                 mz_interval: float, mz_margin: float,
                                 mz_margin_mode: Optional[str]) \
        -> Tuple[int, int]:
    """
    Get the indexes of the vectors falling within the specified precursor m/z
    interval (taking a small margin for overlapping intervals into account).

    Parameters
    ----------
    precursor_mzs : np.ndarray
        Array of sorted precursor m/z's.
    start_mz : float
        The lower end of the m/z interval.
    mz_interval : float
        The width of the m/z interval.
    mz_margin : float
        The value of the precursor m/z tolerance.
    mz_margin_mode : Optional[str]
        The unit of the precursor m/z tolerance ('Da' or 'ppm'). If not 'Da' or
        'ppm' no margin around the m/z intervals will be used.

    Returns
    -------
    Tuple[int, int]
        The start and stop index of the vector indexes falling within
        the specified precursor m/z interval.
    """
    if mz_margin_mode == 'Da':
        pass
    elif mz_margin_mode == 'ppm':
        mz_margin = mz_margin * start_mz / 10 ** 6
    else:
        mz_margin = 0
    mz_margin = max(mz_margin, mz_interval / 100) if mz_margin > 0 else 0
    idx = np.searchsorted(precursor_mzs, [start_mz - mz_margin,
                                          start_mz + mz_interval + mz_margin])
    return idx[0], idx[1]


def _dist_mz_interval(index_filename: str, n_probe: int, vectors: np.ndarray,
                      precursor_mzs: np.ndarray, mz: int, mz_interval: float,
                      batch_size: int, n_neighbors: int, n_neighbors_ann: int,
                      precursor_tol_mass: float, precursor_tol_mode: str,
                      distances: np.ndarray, indices: np.ndarray,
                      indptr: np.ndarray) -> None:
    """
    Compute distances to the nearest neighbors for the given precursor m/z
    interval.

    Parameters
    ----------
    index_filename : str
        File name of the ANN index.
    n_probe : int
        The number of cells to visit during ANN querying.
    vectors : np.ndarray
        The vectors to be queried.
    precursor_mzs : np.ndarray
        Precursor m/z's corresponding to the embedding vectors.
    mz : int
        The active precursor m/z split.
    mz_interval : float
        The width of the m/z interval.
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
    """
    if not os.path.isfile(index_filename):
        return
    index = _load_ann_index(index_filename, n_probe)
    start_i, stop_i = _get_precursor_mz_interval_i(
        precursor_mzs, mz, mz_interval, 0, None)
    for batch_start in range(start_i, stop_i, batch_size):
        batch_stop = min(batch_start + batch_size, stop_i)
        # Find nearest neighbors using ANN index searching.
        # noinspection PyArgumentList
        nn_dists, nn_idx_ann = index.search(vectors[batch_start:batch_stop],
                                            n_neighbors_ann)
        # Filter the neighbors based on the precursor m/z tolerance.
        nn_idx_mz = _get_neighbors_idx(
            precursor_mzs, batch_start, batch_stop, precursor_tol_mass,
            precursor_tol_mode)
        for i, idx_ann, idx_mz, dists in zip(
                np.arange(batch_start, batch_stop), nn_idx_ann, nn_idx_mz,
                nn_dists):
            mask = _intersect_idx_ann_mz(idx_ann, idx_mz, n_neighbors)
            indptr[i + 1] = indptr[i] + len(mask)
            # Convert cosine similarity to cosine distance.
            distances[indptr[i]:indptr[i + 1]] = np.clip(1 - dists[mask], 0, 1)
            indices[indptr[i]:indptr[i + 1]] = idx_ann[mask]
    index.reset()


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
    if precursor_tol_mode == 'Da':
        min_mz = mzs[start_i] - precursor_tol_mass
        max_mz = mzs[stop_i - 1] + precursor_tol_mass
        mz_filter = 'abs(batch_mzs - match_mzs) < precursor_tol_mass'
    elif precursor_tol_mode == 'ppm':
        min_mz = mzs[start_i] - mzs[start_i] * precursor_tol_mass / 10**6
        max_mz = mzs[stop_i - 1] + mzs[stop_i - 1] * precursor_tol_mass / 10**6
        mz_filter = ('abs(batch_mzs - match_mzs)'
                     '/ match_mzs * 10**6 < precursor_tol_mass')
    else:
        raise ValueError('Unknown precursor tolerance filter')
    batch_mzs = mzs[start_i:stop_i].reshape((-1, 1))
    match_i = np.searchsorted(mzs, [min_mz, max_mz])
    match_mzs = mzs[match_i[0]:match_i[1]].reshape((1, -1))
    match_mzs_i = np.arange(match_i[0], match_i[1])
    # FIXME: try this with Numba.
    import numexpr as ne
    return [match_mzs_i[mask] for mask in ne.evaluate(mz_filter)]


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
