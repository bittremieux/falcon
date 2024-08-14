import gc
import logging
import math
import tempfile
from typing import Callable, Iterator, List, Optional, Tuple

import faiss
import joblib
import lance
import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse as ss
from scipy.cluster.hierarchy import fcluster
from spectrum_utils.spectrum import MsmsSpectrum

# noinspection PyProtectedMember
from sklearn.cluster._dbscan_inner import dbscan_inner

from . import spectrum


logger = logging.getLogger("falcon")


def compute_pairwise_distances(
    dataset: lance.LanceDataset,
    charge: int,
    bucket_id: int,
    vectorize: Callable,
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    rt_tol: float,
    n_neighbors: int,
    batch_size: int,
    n_probe: int,
) -> Tuple[ss.csr_matrix, pd.DataFrame]:
    """
    Compute a pairwise distance matrix for the persisted spectra with the given
    precursor charge.

    Parameters
    ----------
    dataset: lance.LanceDataset
        The dataset containing the spectra to be indexed.
    charge : int
        The precursor charge for which to compute the pairwise distances.
    bucket_id : int
        The charge bucket identifier.
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
    n_spectra = dataset.count_rows(filter=f"precursor_charge == {charge}")
    logger.debug(
        "Compute nearest neighbor pairwise distances (%d spectra, %d"
        " neighbors)",
        n_spectra,
        n_neighbors,
    )
    max_num_embeddings = n_spectra * n_neighbors
    dtype = (
        np.int32 if max_num_embeddings < np.iinfo(np.int32).max else np.int64
    )
    distances = np.zeros(max_num_embeddings, np.float32)
    indices = np.zeros(max_num_embeddings, dtype)
    indptr = np.zeros(n_spectra + 1, dtype)
    # Create the ANN indexes (if this hasn't been done yet) and calculate
    # pairwise distances.
    metadata = _build_query_ann_index(
        dataset,
        charge,
        bucket_id,
        vectorize,
        n_probe,
        batch_size,
        n_neighbors,
        precursor_tol_mass,
        precursor_tol_mode,
        rt_tol,
        distances,
        indices,
        indptr,
    )
    distances, indices = distances[: indptr[-1]], indices[: indptr[-1]]
    # Convert to a sparse pairwise distance matrix. This matrix might not be
    # entirely symmetrical, but that shouldn't matter too much.
    logger.debug(
        "Construct %d-by-%d sparse pairwise distance matrix with %d "
        "non-zero values",
        n_spectra,
        n_spectra,
        len(distances),
    )
    pairwise_dist_matrix = ss.csr_matrix(
        (distances, indices, indptr), (n_spectra, n_spectra), np.float32, False
    )
    return pairwise_dist_matrix, metadata


def _build_query_ann_index(
    dataset: lance.LanceDataset,
    charge: int,
    bucket_id: int,
    vectorize: Callable,
    n_probe: int,
    batch_size: int,
    n_neighbors: int,
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    rt_tol: float,
    distances: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
) -> pd.DataFrame:
    """
    Create ANN index(es) for spectra with the given charge per precursor m/z
    split.

    Parameters
    ----------
    dataset : lance.LanceDataset
        The dataset containing the spectra to be indexed.
    charge : int
        The precursor charge for which to compute the pairwise distances.
    bucket_id : int
        The charge bucket identifier.
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
        Metadata (filename, identifier, precursor charge, precursor m/z,
        retention time) of the spectra for which indexes were built.
    """
    query = f"precursor_charge == {charge}"
    n_spectra = dataset.count_rows(filter=query)

    # Read the spectra for the m/z bucket.
    filenames, identifiers, bucket_ids, precursor_mzs, rts = [], [], [], [], []
    indptr_i = 0

    # Figure out a decent value for the n_list hyperparameter based on
    # the number of vectors.
    # Rules of thumb from the Faiss wiki:
    # https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index#how-big-is-the-dataset
    # if n_vectors == 0:
    #     continue
    if n_spectra < 10**2:
        # Use a brute-force index instead of an ANN index when there
        # are only a few items.
        n_list = -1
    elif n_spectra < 10**6:
        n_list = 2 ** math.floor(math.log2(n_spectra / 39))
    elif n_spectra < 10**7:
        n_list = 2**16
    elif n_spectra < 10**8:
        n_list = 2**18
    else:
        n_list = 2**20
        if n_spectra > 10**9:
            logger.warning(
                "More than 1B vectors to be indexed, consider "
                "decreasing the ANN size"
            )
    train_spectra = dataset.to_table(filter=query).to_pandas()
    train_spectra = train_spectra.apply(
        spectrum.df_row_to_spec, axis=1
    ).tolist()
    train_vectors = vectorize(spectra=train_spectra)
    dim = train_vectors.shape[1]
    # Create an ANN index using the inner product (proxy for cosine
    # distance) for fast NN queries.
    if n_list <= 0:
        index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    else:
        index = faiss.IndexIVFFlat(
            faiss.IndexFlatIP(dim), dim, n_list, faiss.METRIC_INNER_PRODUCT
        )
        index.nprobe = min(math.ceil(index.nlist / 8), n_probe)
    # Compute cluster centroids.
    # noinspection PyArgumentList
    index.train(train_vectors)

    # Add the vectors to the index in batches.
    batch_start = 0
    for i, batch in enumerate(
        dataset.to_batches(batch_size=batch_size, filter=query)
    ):
        logger.debug(
            "Add batch %d/%d to ANN index (%d spectra)",
            i + 1,
            n_spectra // batch_size + 1,
            len(batch),
        )
        batch_spectra = (
            batch.to_pandas().apply(spectrum.df_row_to_spec, axis=1).tolist()
        )
        proc_batch = []
        for spec in batch_spectra:
            proc_batch.append(spec)
            filenames.append(spec.filename)
            identifiers.append(spec.identifier)
            bucket_ids.append(bucket_id)
            precursor_mzs.append(spec.precursor_mz)
            rts.append(spec.retention_time)
        batch_stop = batch_start + len(proc_batch)

        batch_vectors = vectorize(spectra=proc_batch)
        index.add_with_ids(
            batch_vectors,
            np.arange(batch_start, batch_stop, dtype=np.int64),
        )
        batch_start = batch_stop
    precursor_mzs = np.asarray(precursor_mzs)
    rts = np.asarray(rts)
    # Query the index to calculate NN distances.
    _dist_mz_interval(
        index,
        dataset,
        query,
        vectorize,
        precursor_mzs,
        rts,
        batch_size,
        n_neighbors,
        precursor_tol_mass,
        precursor_tol_mode,
        rt_tol,
        distances,
        indices,
        indptr,
    )
    index.reset()
    indptr_i += n_spectra

    return pd.DataFrame(
        {
            "filename": filenames,
            "spectrum_id": identifiers,
            "bucket_id": bucket_ids,
            "precursor_mz": np.hstack(precursor_mzs),
            "retention_time": np.hstack(rts),
        }
    )


def _dist_mz_interval(
    index: faiss.Index,
    dataset: lance.LanceDataset,
    query: str,
    vectorize: Callable,
    precursor_mzs: np.ndarray,
    rts: np.ndarray,
    batch_size: int,
    n_neighbors: int,
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    rt_tol: float,
    distances: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
) -> None:
    """
    Compute distances to the nearest neighbors for the given precursor m/z
    interval.

    Parameters
    ----------
    index : faiss.Index
        The NN index used to efficiently find distances to similar spectra.
    dataset : lance.LanceDataset
        The dataset containing the spectra to be indexed.
    query : str
        The query to filter the spectra.
    vectorize : Callable
        Function to convert the spectra to vectors.
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
    batch_start = 0
    nn_mz = _get_mz_neighbors(
        precursor_mzs, precursor_tol_mass, precursor_tol_mode
    )
    if rt_tol is not None:  # TODO: test this
        nn_rt = _get_mz_neighbors(rts, rt_tol, "rt")
        nn_mz = [np.intersect1d(mz, rt) for mz, rt in zip(nn_mz, nn_rt)]
    for batch in dataset.to_batches(batch_size=batch_size, filter=query):
        batch_spectra = (
            batch.to_pandas().apply(spectrum.df_row_to_spec, axis=1).tolist()
        )
        batch_spectra = nb.typed.List(batch_spectra)
        vectors = vectorize(spectra=batch_spectra)
        batch_stop = batch_start + vectors.shape[0]
        # Find nearest neighbors using ANN index searching.
        for i, vec in enumerate(vectors, batch_start):
            sel = faiss.IDSelectorBatch(nn_mz[i % batch_size])
            params = faiss.SearchParametersIVF(sel=sel, nprobe=1)
            # noinspection PyArgumentList
            vec = vec.reshape(1, -1)
            d, idx = index.search(vec, n_neighbors, params=params)
            # filter out indices and distances < 0
            mask = idx >= 0
            d, idx = d[mask], idx[mask]
            # Build csr matrix
            indptr[i + 1] = indptr[i] + len(d)  # d.shape[1]  # len(d)
            # Convert cosine similarity to cosine distance.
            distances[indptr[i] : indptr[i + 1]] = np.maximum(1 - d, 0)
            indices[indptr[i] : indptr[i + 1]] = idx
        batch_start = batch_stop


def _get_mz_neighbors(
    values: np.ndarray,
    tol: float,
    tol_mode: str,
) -> List[np.ndarray]:
    """
    Get the mz neighbors for the given precursor m/z's.

    Parameters
    ----------
    values : np.ndarray
        The precursor m/z or retention time values of the spectra.
    tol: float
        The tolerance for vectors to be considered as neighbors.
    tol_mode : str
        The unit of the tolerance ('Da' or 'ppm' for precursor m/z,
        'rt' for rentention time).

    Returns
    -------
    List[np.ndarray]
        The indices of the NN candidates.
    """
    nn = []
    if tol_mode in ("Da", "ppm"):
        order = np.argsort(values)
        for mz in values:
            if tol_mode == "ppm":
                mz_min = mz - mz * tol / 1**6
                mz_max = mz + mz * tol / 1**6
            elif tol_mode == "Da":
                mz_min = mz - tol
                mz_max = mz + tol
            mz_min, mz_max = max(0, mz_min), max(0, mz_max)
            match_i = np.searchsorted(values[order], [mz_min, mz_max])
            idx = np.arange(match_i[0], match_i[1])
            nn.append(order[idx])
    elif tol_mode == "rt":  # TODO: test this
        for rt in values:
            rt_min = max(0, rt - tol)
            rt_max = max(0, rt + tol)
            match_values_i = np.where(
                np.logical_and(values >= rt_min, values <= rt_max)
            )
            nn.append(match_values_i)
    else:
        raise ValueError("Unknown tolerance filter")
    return nn


def generate_clusters(
    pairwise_dist_matrix: ss.csr_matrix,
    eps: float,
    precursor_mzs: np.ndarray,
    rts: np.ndarray,
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    rt_tol: float,
) -> np.ndarray:
    """
    DBSCAN clustering of the given pairwise distance matrix.

    Parameters
    ----------
    pairwise_dist_matrix : ss.csr_matrix
        A sparse pairwise distance matrix used for clustering.
    eps : float
        The maximum distance between two samples for one to be considered as in
        the neighborhood of the other.
    precursor_mzs : np.ndarray
        Precursor m/z's matching the pairwise distance matrix.
    rts : np.ndarray
        Retention times matching the pairwise distance matrix.
    precursor_tol_mass : float
        Maximum precursor mass tolerance for points to be clustered together.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    rt_tol : float
        The retention time tolerance for points to be clustered together. If
        `None`, do not restrict the retention time.

    Returns
    -------
    np.ndarray
        Cluster labels. Noisy samples are given the label -1.
    """
    # DBSCAN clustering using the precomputed pairwise distance matrix.
    min_samples = 2
    logger.debug(
        "DBSCAN clustering (eps=%.4f, min_samples=%d) of precomputed "
        "pairwise distance matrix",
        eps,
        min_samples,
    )
    with tempfile.NamedTemporaryFile() as mmap_file:
        # Reimplement DBSCAN preprocessing to avoid unnecessary memory
        # consumption.
        dist_data = pairwise_dist_matrix.data
        dist_indices = pairwise_dist_matrix.indices
        dist_indptr = pairwise_dist_matrix.indptr
        n_spectra = pairwise_dist_matrix.shape[0]
        # Find the eps-neighborhoods for all points.
        mask = dist_data <= eps
        indptr = _cumsum(mask)[dist_indptr]
        indices = dist_indices[mask].astype(np.intp, copy=False)
        neighborhoods = np.split(indices, indptr[1:-1])
        # Initially, all samples are noise.
        # (Memmap for shared memory multiprocessing.)
        clusters = np.memmap(mmap_file, np.intp, "w+", shape=(n_spectra,))
        clusters.fill(-1)
        # A list of all core samples found.
        n_neighbors = np.fromiter(map(len, neighborhoods), np.uint32)
        core_samples = n_neighbors >= min_samples
        # Run Scikit-Learn DBSCAN.
        # noinspection PyUnresolvedReferences
        neighborhoods_arr = np.empty(len(neighborhoods), dtype=object)
        neighborhoods_arr[:] = neighborhoods
        dbscan_inner(core_samples, neighborhoods_arr, clusters)

        # Free up memory by deleting DBSCAN-related data structures.
        del pairwise_dist_matrix, mask, indptr, indices
        del neighborhoods, n_neighbors, core_samples, neighborhoods_arr
        gc.collect()

        # Refine initial clusters to make sure spectra within a cluster don't
        # have an excessive precursor m/z difference.
        # noinspection PyUnresolvedReferences
        order = np.argsort(clusters)
        # noinspection PyUnresolvedReferences
        reverse_order = np.argsort(order)
        clusters[:] = clusters[order]
        precursor_mzs, rts = precursor_mzs[order], rts[order]
        logger.debug(
            "Finetune %d initial unique non-singleton clusters to not"
            " exceed %.2f %s precursor m/z tolerance%s",
            clusters[-1] + 1,
            precursor_tol_mass,
            precursor_tol_mode,
            (
                f" and {rt_tol} retention time tolerance"
                if rt_tol is not None
                else ""
            ),
        )
        if clusters[-1] == -1:  # Only noise samples.
            clusters.fill(-1)
            noise_mask = np.ones_like(clusters, dtype=np.bool_)
            n_clusters, n_noise = 0, len(noise_mask)
        else:
            group_idx = nb.typed.List(_get_cluster_group_idx(clusters))
            n_clusters = nb.typed.List(
                joblib.Parallel(n_jobs=-1, prefer="threads")(
                    joblib.delayed(_postprocess_cluster)(
                        clusters[start_i:stop_i],
                        precursor_mzs[start_i:stop_i],
                        rts[start_i:stop_i],
                        precursor_tol_mass,
                        precursor_tol_mode,
                        rt_tol,
                        min_samples,
                    )
                    for start_i, stop_i in group_idx
                )
            )
            _assign_unique_cluster_labels(
                clusters, group_idx, n_clusters, min_samples
            )
            clusters[:] = clusters[reverse_order]
            noise_mask = clusters == -1
            # noinspection PyUnresolvedReferences
            n_clusters, n_noise = np.amax(clusters) + 1, noise_mask.sum()
        logger.debug(
            "%d unique non-singleton clusters after precursor m/z "
            "finetuning, %d total clusters",
            n_clusters,
            n_clusters + n_noise,
        )
        # Reassign noise points to singleton clusters.
        clusters[noise_mask] = np.arange(n_clusters, n_clusters + n_noise)
        return np.asarray(clusters)


@nb.njit
def _cumsum(a: np.ndarray) -> np.ndarray:
    """
    Cumulative sum of the elements.

    Try to avoid inadvertent copies in `np.cumsum`.

    Parameters
    ----------
    a : np.ndarray
        Input array

    Returns
    -------
    np.ndarray
        The cumulative sum in an array of size len(a) + 1 (first element is 0).
    """
    out = np.zeros(len(a) + 1, dtype=np.int64)
    for i in range(len(out) - 1):
        out[i + 1] = out[i] + a[i]
    return out


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
        start_i += 1
    stop_i = start_i
    while stop_i < clusters.shape[0]:
        start_i, label = stop_i, clusters[stop_i]
        while stop_i < clusters.shape[0] and clusters[stop_i] == label:
            stop_i += 1
        yield start_i, stop_i


def _postprocess_cluster(
    cluster_labels: np.ndarray,
    cluster_mzs: np.ndarray,
    cluster_rts: np.ndarray,
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    rt_tol: float,
    min_samples: int,
) -> int:
    """
    Agglomerative clustering of the precursor m/z's within each initial
    cluster to avoid that spectra within a cluster have an excessive precursor
    m/z difference.

    Parameters
    ----------
    cluster_labels : np.ndarray
        Array in which to write the cluster labels.
    cluster_mzs : np.ndarray
        Precursor m/z's of the samples in a single initial cluster.
    cluster_rts : np.ndarray
        Retention times of the samples in a single initial cluster.
    precursor_tol_mass : float
        Maximum precursor mass tolerance for points to be clustered together.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    rt_tol : float
        The retention time tolerance for points to be clustered together. If
        `None`, do not restrict the retention time.
    min_samples : int
        The minimum number of samples in a cluster.

    Returns
    -------
    int
        The number of clusters after splitting on precursor m/z.
    """
    cluster_labels.fill(-1)
    # No splitting needed if there are too few items in cluster.
    # This seems to happen sometimes despite that DBSCAN requires a higher
    # `min_samples`.
    if cluster_labels.shape[0] < min_samples:
        n_clusters = 0
    else:
        # Group items within the cluster based on their precursor m/z.
        # Precursor m/z's within a single group can't exceed the specified
        # precursor m/z tolerance (`distance_threshold`).
        # Subtract 1 because fcluster starts with cluster label 1 instead of 0
        # (like Scikit-Learn does).
        cluster_assignments = (
            fcluster(
                _linkage(cluster_mzs, precursor_tol_mode),
                precursor_tol_mass,
                "distance",
            )
            - 1
        )
        # Optionally restrict clusters by their retention time as well.
        if rt_tol is not None:
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
            cluster_labels.fill(0)
        elif n_clusters == cluster_mzs.shape[0]:
            # Only singletons.
            n_clusters = 0
        else:
            unique, inverse, counts = np.unique(
                cluster_assignments, return_inverse=True, return_counts=True
            )
            non_noise_clusters = np.where(counts >= min_samples)[0]
            labels = -np.ones_like(unique)
            labels[non_noise_clusters] = np.unique(
                unique[non_noise_clusters], return_inverse=True
            )[1]
            cluster_labels[:] = labels[inverse]
            n_clusters = len(non_noise_clusters)
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


@nb.njit(cache=True)
def _assign_unique_cluster_labels(
    cluster_labels: np.ndarray,
    group_idx: nb.typed.List,
    n_clusters: nb.typed.List,
    min_samples: int,
) -> None:
    """
    Make sure all cluster labels are unique after potential splitting of
    clusters to avoid excessive precursor m/z differences.

    Parameters
    ----------
    cluster_labels : np.ndarray
        Cluster labels per cluster grouping.
    group_idx : nb.typed.List[Tuple[int, int]]
        Tuples with the start index (inclusive) and end index (exclusive) of
        the cluster groupings.
    n_clusters: nb.typed.List[int]
        The number of clusters per cluster grouping.
    min_samples : int
        The minimum number of samples in a cluster.
    """
    current_label = 0
    for (start_i, stop_i), n_cluster in zip(group_idx, n_clusters):
        if n_cluster > 0 and stop_i - start_i >= min_samples:
            current_labels = cluster_labels[start_i:stop_i]
            current_labels[current_labels != -1] += current_label
            current_label += n_cluster
        else:
            cluster_labels[start_i:stop_i].fill(-1)


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
