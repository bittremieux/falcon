import gc
import logging
import math
import multiprocessing
import tempfile
from functools import partial
from typing import List, Tuple

import fastcluster
import joblib
import lance
import numba as nb
import numpy as np
import scipy.cluster.hierarchy as sch
import spectrum_utils.utils as suu
from scipy.cluster.hierarchy import fcluster
from tqdm import tqdm

from . import similarity
from .consensus import (
    ConsensusTuple,
    _get_cluster_group_idx,
    _get_representative_spectra,
)
from .distance_matrix import compute_condensed_distance_matrix

logger = logging.getLogger("falcon")


def generate_clusters(
    dataset: lance.LanceDataset,
    linkage: str,
    distance_threshold: float,
    min_matches: int,
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    rt_tol: float,
    fragment_tol: float,
    batch_size: int,
    consensus_method: str,
    consensus_params: dict,
) -> Tuple[np.ndarray, List[ConsensusTuple]]:
    """
    Cluster the spectra in the given dataset using hierarchical clustering.

    Parameters
    ----------
    dataset : lance.LanceDataset
        The dataset containing the spectra to be clustered.
    linkage : str
        The linkage method to use for hierarchical clustering
        ('single', 'complete', or 'average').
    distance_threshold : float
        The linkage distance threshold at or above which clusters will not be
        merged.
    min_matches : int
        The minimum number of matched peaks to consider two spectra similar.
    precursor_tol_mass : float
        Maximum precursor mass tolerance for points to be clustered together.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    rt_tol : float
        The retention time tolerance for points to be clustered together.
        Spectra with NaN retention time are not split by RT but remain
        eligible for m/z-based splitting. If `None`, RT is not used.
    fragment_tol : float
        The fragment m/z tolerance.
    batch_size : int
        Maximum number of spectra per precursor m/z split.
    consensus_method : str
        The method to use for consensus spectrum computation
        ('medoid' or 'average').
    consensus_params : dict
        Additional parameters for the consensus spectrum computation.

    Returns
    -------
    Tuple[np.ndarray, List[ConsensusTuple]]
        The cluster labels and the representative spectra for each cluster.
    """
    # Hierarchical clustering using the precomputed pairwise distance matrix.
    min_samples = 2
    logger.debug(
        "Hierarchical clustering (distance_threshold=%.4f, min_samples=%d)",
        distance_threshold,
        min_samples,
    )
    # Load only the columns needed for sorting and splitting; full spectra data
    # (mz, intensity) is fetched on demand per interval inside _cluster_mz_interval.
    data = dataset.to_table(
        columns=["identifier", "precursor_mz", "retention_time"]
    ).to_pandas()
    # Sort by precursor m/z and retention time, with the unique identifier as
    # a tiebreaker so the ordering is a deterministic total order. The returned
    # cluster labels are aligned to this order positionally, and the caller
    # re-attaches them after an independent sort of the same dataset; without a
    # unique tiebreaker, ties could be ordered differently between the two
    # (non-stable) sorts and bind labels to the wrong spectra.
    data = data.reset_index().sort_values(
        ["precursor_mz", "retention_time", "identifier"],
    )
    # Cluster per contiguous block of precursor m/z's (relative to the
    # precursor m/z threshold).
    logger.info(
        "Cluster %d spectra with charge %s",
        len(data),
        dataset.uri.split("spectra_charge_")[-1]
        .split(".")[0]
        .replace("_", ", "),
    )
    with tempfile.NamedTemporaryFile(suffix=".npy") as cluster_file:
        cluster_filename = cluster_file.name
        cluster_labels = np.lib.format.open_memmap(
            cluster_filename, mode="w+", dtype=np.int32, shape=(data.shape[0],)
        )
        cluster_labels.fill(-1)
        representative_spectra = []
        with tqdm(
            total=len(data), desc="Clustering", unit="spectra", smoothing=0
        ) as pbar:
            idx = data["index"].values
            mzs = data["precursor_mz"].values
            splits = _get_precursor_mz_splits(
                mzs, precursor_tol_mass, precursor_tol_mode, batch_size
            )
            # Per m/z split clustering.
            # TODO: check if still needed with joblib
            chunks = cost_based_chunking(splits, multiprocessing.cpu_count())
            # Cluster m/z splits
            if len(chunks) > 0:
                # Process chunks
                process_chunk = partial(
                    cluster_chunk,
                    dataset=dataset,
                    linkage=linkage,
                    distance_threshold=distance_threshold,
                    min_matches=min_matches,
                    precursor_tol_mass=precursor_tol_mass,
                    precursor_tol_mode=precursor_tol_mode,
                    rt_tol=rt_tol,
                    fragment_mz_tol=fragment_tol,
                    consensus_method=consensus_method,
                    consensus_params=consensus_params,
                )
                # TODO: move to chunking method
                data_chunks = []
                for chunk in chunks:
                    data_chunk = []
                    for task_id, (interval_start, interval_stop) in chunk:
                        row_ids = idx[interval_start:interval_stop]
                        idx_interval = idx[interval_start:interval_stop]
                        mz_interval = mzs[interval_start:interval_stop]
                        data_chunk.append(
                            (
                                task_id,
                                row_ids.tolist(),
                                idx_interval,
                                mz_interval,
                            )
                        )
                    data_chunks.append(data_chunk)

                results = joblib.Parallel(
                    n_jobs=-1, backend="loky", verbose=0
                )(
                    joblib.delayed(process_chunk)(data_chunk)
                    for data_chunk in data_chunks
                )
                flattened_results = [
                    split_result
                    for chunk_results in results
                    for split_result in chunk_results
                ]
                flattened_results.sort(key=lambda x: x[0])
                n_total = len(data)
                n_processed = 0
                next_heartbeat = n_total // 10
                for task_id, (
                    interval_rep_spectra,
                    labels,
                ) in flattened_results:
                    if interval_rep_spectra is not None:
                        # add task id (mz_split) to rep_spectra
                        interval_rep_spectra = [
                            s._replace(mz_split=np.int32(task_id))
                            for s in interval_rep_spectra
                        ]
                        representative_spectra.extend(interval_rep_spectra)
                        cluster_labels[
                            splits[task_id] : splits[task_id + 1]
                        ] = labels
                        n_processed += len(labels)
                        pbar.update(len(labels))
                        if (
                            next_heartbeat > 0
                            and n_processed >= next_heartbeat
                        ):
                            logger.info(
                                "Clustering progress: %d/%d spectra (%.0f%%)",
                                n_processed,
                                n_total,
                                100 * n_processed / n_total,
                            )
                            next_heartbeat += n_total // 10
            representative_spectra = _assign_global_cluster_labels(
                cluster_labels, representative_spectra, splits
            )
        _, counts = np.unique(cluster_labels, return_counts=True)
        n_clusters = np.count_nonzero(counts > 1)
        n_noise = np.count_nonzero(counts == 1)
        logger.info(
            "%d spectra grouped in %d clusters, %d spectra remain as singletons",
            counts[counts > 1].sum(),
            n_clusters,
            n_noise,
        )
        cluster_labels.flush()
        return cluster_labels, representative_spectra


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
    splits = nb.typed.List([0])
    for i in range(1, len(precursor_mzs)):
        block_size = i - splits[-1]
        if (
            suu.mass_diff(
                precursor_mzs[i],
                precursor_mzs[i - 1],
                precursor_tol_mode == "Da",
            )
            > precursor_tol_mass
        ):
            if block_size < batch_size:
                splits.append(i)
            else:
                # split into evenly sized chunks of at most batch_size
                n_chunks = math.ceil(block_size / batch_size)
                chunk_size = block_size // n_chunks
                for _ in range(block_size % n_chunks):
                    splits.append(splits[-1] + chunk_size + 1)
                for _ in range(n_chunks - (block_size % n_chunks)):
                    splits.append(splits[-1] + chunk_size)
        elif block_size >= batch_size:
            splits.append(i)
    if splits[-1] != len(precursor_mzs):
        splits.append(len(precursor_mzs))
    return splits


def cost_based_chunking(
    tasks: List[int], num_chunks: int
) -> List[List[Tuple[int, Tuple[int, int]]]]:
    """
    Distribute tasks across chunks to balance estimated computational cost.

    Cost is proportional to the square of the interval size. Tasks are
    assigned greedily to the chunk with the lowest cumulative cost.

    Parameters
    ----------
    tasks : List[int]
        Sorted split-point indices defining the task intervals.
    num_chunks : int
        Number of chunks (typically equal to the number of workers).

    Returns
    -------
    List[List[Tuple[int, Tuple[int, int]]]]
        A list of chunks. Each chunk is a list of (task_index,
        (interval_start, interval_stop)) tuples.
    """
    split_tuples = [(tasks[i], tasks[i + 1]) for i in range(len(tasks) - 1)]
    indexed_tasks = list(enumerate(split_tuples))
    indexed_tasks.sort(key=lambda x: (x[1][1] - x[1][0]) ** 2, reverse=True)

    # Initialize chunks and their cumulative costs
    chunks = [[] for _ in range(num_chunks)]
    costs = [0] * num_chunks

    # Assign tasks to the chunk with the least cumulative cost
    for index, task in indexed_tasks:
        idx = costs.index(min(costs))  # Find the chunk with the least cost
        chunks[idx].append((index, task))
        costs[idx] += (task[1] - task[0]) ** 2

    # Remove empty chunks
    chunks = [chunk for chunk in chunks if chunk]

    return chunks


def cluster_chunk(
    chunk: List[Tuple[int, Tuple[int, int]]],
    dataset: lance.LanceDataset,
    linkage: str,
    distance_threshold: float,
    min_matches: int,
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    rt_tol: float,
    fragment_mz_tol: float,
    consensus_method: str,
    consensus_params: dict,
) -> List[Tuple[int, Tuple[List[ConsensusTuple], np.ndarray]]]:
    """
    Cluster all m/z intervals in the given chunk.

    Parameters
    ----------
    chunk : List[Tuple[int, Tuple[int, int]]]
        The cluster tasks: each entry is (task_id, (interval_start,
        interval_stop)) where the interval bounds index into the sorted
        precursor m/z array.
    dataset : lance.LanceDataset
        The dataset containing the spectra to be clustered.
    linkage : str
        Linkage method for hierarchical clustering
        ('single', 'complete', or 'average').
    distance_threshold : float
        The maximum linkage distance threshold during clustering.
    min_matches : int
        The minimum number of matched peaks to consider two spectra similar.
    precursor_tol_mass : float
        The value of the precursor m/z tolerance.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    rt_tol : float
        The retention time tolerance. Spectra with NaN RT are not split by RT.
        If `None`, RT is not used.
    fragment_mz_tol : float
        The fragment m/z tolerance.
    consensus_method : str
        The method to use for consensus spectrum computation
        ('medoid' or 'average').
    consensus_params : dict
        Additional parameters for the consensus spectrum computation.

    Returns
    -------
    List[Tuple[int, Tuple[List[ConsensusTuple], np.ndarray]]]
        For each task: the task index and the result of `_cluster_mz_interval`.
    """

    return [
        (
            i,
            _cluster_mz_interval(
                dataset,
                row_ids,
                idx,
                mzs,
                linkage,
                distance_threshold,
                min_matches,
                precursor_tol_mass,
                precursor_tol_mode,
                rt_tol,
                fragment_mz_tol,
                consensus_method,
                consensus_params,
            ),
        )
        for i, row_ids, idx, mzs in chunk
    ]


def _cluster_mz_interval(
    dataset: lance.LanceDataset,
    row_ids: List[int],
    idx: np.ndarray,
    mzs: np.ndarray,
    linkage: str,
    distance_threshold: float,
    min_matches: int,
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    rt_tol: float,
    fragment_mz_tol: float,
    consensus_method: str,
    consensus_params: dict,
) -> Tuple[List[ConsensusTuple], np.ndarray]:
    """
    Cluster the spectra in a single precursor m/z interval.

    Parameters
    ----------
    dataset : lance.LanceDataset
        The dataset from which spectra are fetched on demand.
    row_ids : List[int]
        Lance row indices of the spectra in this interval.
    idx : np.ndarray
        Sorted positional indices of the spectra within the global ordering.
    mzs : np.ndarray
        Precursor m/z values corresponding to `idx`.
    linkage : str
        Linkage method for hierarchical clustering. See
        `scipy.cluster.hierarchy.linkage` for options.
    distance_threshold : float
        The maximum linkage distance threshold during clustering.
    min_matches : int
        The minimum number of matched peaks to consider two spectra similar.
    precursor_tol_mass : float
        The value of the precursor m/z tolerance.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    rt_tol : float
        The retention time tolerance. Spectra with NaN RT are not split by RT.
        If `None`, RT is not used.
    fragment_mz_tol : float
        The fragment m/z tolerance.
    consensus_method : str
        The method to use for consensus spectrum computation
        ('medoid' or 'average').
    consensus_params : dict
        Additional parameters for the consensus spectrum computation.

    Returns
    -------
    Tuple[List[ConsensusTuple], np.ndarray]
        The representative spectrum for each cluster and the cluster label
        array aligned to the input interval order.
    """
    spectra = dataset.take(
        indices=row_ids,
        columns=[
            "identifier",
            "precursor_mz",
            "precursor_charge",
            "retention_time",
            "mz",
            "intensity",
        ],
    ).to_pandas()
    # Sort with the same total-order key as the outer `data` sort in
    # `generate_clusters` so this interval's spectra line up positionally with
    # the `idx`/`mzs` arrays (which come from that outer order). `idx`/`mzs`
    # (outer order) and `rts`/`labels` (this order) are used together below, so
    # any mismatch on (precursor_mz, retention_time) ties would corrupt the
    # clustering; `identifier` is the unique tiebreaker that keeps them aligned.
    spectra = spectra.sort_values(
        ["precursor_mz", "retention_time", "identifier"]
    )
    rts = spectra["retention_time"].values
    spectra = spectra.apply(
        similarity.df_row_to_spectrum_tuple, axis=1
    ).tolist()
    n_spectra = len(spectra)
    cluster_labels = -np.ones(n_spectra, np.int32)
    if n_spectra > 1:
        # Hierarchical clustering of the vectors.
        # Subtract 1 because fcluster starts with cluster label 1 instead of 0
        # (like Scikit-Learn does).
        pdist = compute_condensed_distance_matrix(
            spectra,
            fragment_mz_tol,
            min_matches,
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
        rev_order = np.argsort(order)
        idx, mzs, rts = (
            idx[order],
            mzs[order],
            rts[order],
        )
        labels, current_label = labels[order], 0
        for start_i, stop_i in _get_cluster_group_idx(labels):
            n_clusters = _postprocess_cluster(
                labels[start_i:stop_i],
                mzs[start_i:stop_i],
                rts[start_i:stop_i],
                precursor_tol_mass,
                precursor_tol_mode,
                rt_tol,
                2,
                current_label,
            )
            current_label += n_clusters
        # Get representative spectra for clusters.
        if current_label < n_spectra:
            order_ = np.argsort(labels)
            rev_order_ = np.argsort(order_)
            idx = idx[order_]
            labels = labels[order_]
            rts = rts[order_]
            order_map = np.arange(len(labels))[order][order_]
            if consensus_method == "medoid":
                consensus_params["pdist"] = pdist
            rep_spectra = _get_representative_spectra(  # representative spectra are sorted by label
                spectra,
                labels,
                rts,
                order_map,
                consensus_method,
                consensus_params,
            )
            cluster_labels = labels[rev_order_[rev_order]]
        else:  # only singletons
            rep_spectra = spectra
            rep_spectra = [
                ConsensusTuple(
                    precursor_mz=np.float32(spec.precursor_mz),
                    precursor_charge=(
                        np.int32(spec.precursor_charge)
                        if not np.isnan(spec.precursor_charge)
                        else np.nan
                    ),
                    mz=spec.mz.astype(np.float32),
                    intensity=spec.intensity.astype(np.float32),
                    retention_time=np.float32(rts[rev_order][i]),
                    cluster_id=np.int32(labels[rev_order][i]),
                    mz_split=None,
                )
                for i, spec in enumerate(rep_spectra)
            ]
            cluster_labels = labels[rev_order]
        # Force memory clearing.
        del pdist
        if n_spectra > 2**11:
            gc.collect()
    else:  # mz split contains only 1 spectrum
        spec = spectra[0]
        rep_spectra = [
            ConsensusTuple(
                precursor_mz=np.float32(spec.precursor_mz),
                precursor_charge=(
                    np.int32(spec.precursor_charge)
                    if not np.isnan(spec.precursor_charge)
                    else np.nan
                ),
                mz=spec.mz.astype(np.float32),
                intensity=spec.intensity.astype(np.float32),
                retention_time=np.float32(rts[0]),
                cluster_id=np.int32(0),
                mz_split=None,
            )
        ]
        cluster_labels[0] = 0
    return rep_spectra, cluster_labels


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
    Split an initial cluster on precursor m/z (and optionally RT) to prevent
    spectra with excessive precursor m/z differences from sharing a cluster.

    Parameters
    ----------
    cluster_labels : np.ndarray
        Array in which to write the output cluster labels (mutated in place).
    cluster_mzs : np.ndarray
        Precursor m/z values of the spectra in this initial cluster.
    cluster_rts : np.ndarray
        Retention times of the spectra. NaN entries are excluded from the RT
        linkage but remain in the cluster (not split by RT).
    precursor_tol_mass : float
        Maximum precursor m/z difference allowed within a cluster.
    precursor_tol_mode : str
        The unit of the precursor m/z tolerance ('Da' or 'ppm').
    rt_tol : float
        Maximum retention time difference allowed within a cluster. If `None`,
        retention time is not used for splitting.
    min_samples : int
        Minimum number of spectra required to form a cluster (smaller groups
        become singletons).
    start_label : int
        The first cluster label to assign.

    Returns
    -------
    int
        The total number of output clusters (including singletons).
    """
    # No splitting needed if there are too few items in cluster.
    if cluster_labels.shape[0] < min_samples:
        # fill with increasing labels
        cluster_labels[:] = np.arange(
            start_label, start_label + cluster_labels.shape[0]
        )
        return cluster_labels.shape[0]
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
        # Only spectra with a known RT participate in the RT linkage; NaN-RT
        # spectra are not split by RT but receive a distinct RT label so they
        # cannot accidentally merge with real-RT spectra in the combined encoding.
        if rt_tol is not None:
            with nb.objmode(cluster_assignments="int32[:]"):
                nan_mask = np.isnan(cluster_rts)
                if not nan_mask.all():
                    valid_rts = cluster_rts[~nan_mask]
                    rt_labels = np.zeros(len(cluster_rts), np.int32)
                    if len(valid_rts) >= 2:
                        rt_labels[~nan_mask] = (
                            fcluster(_linkage(valid_rts), rt_tol, "distance")
                            - 1
                        )
                    # Label for NaN-RT spectra: one past the highest real RT label,
                    # so it is distinct from every real-RT label but the same for
                    # all NaN spectra in the cluster (they stay together by m/z).
                    n_rt = int(rt_labels[~nan_mask].max()) + 1
                    rt_labels[nan_mask] = n_rt
                    # Injective mixed-radix encoding of the (m/z, RT) pair.
                    # n_rt + 1 is the RT radix: covers labels 0..n_rt (inclusive).
                    cluster_assignments = np.unique(
                        cluster_assignments * (n_rt + 1) + rt_labels,
                        return_inverse=True,
                    )[1].astype(np.int32)
                else:
                    # All RTs unknown: skip RT splitting, keep m/z assignments.
                    cluster_assignments = cluster_assignments.copy()

        n_clusters = cluster_assignments.max() + 1
        # Update cluster assignments.
        if n_clusters == 1:
            # Single homogeneous cluster.
            cluster_labels.fill(start_label)
        elif n_clusters == cluster_labels.shape[0]:
            # Only singletons.
            cluster_labels.fill(-1)
            n_clusters = 0
        else:
            labels = nb.typed.Dict.empty(
                key_type=nb.int64, value_type=nb.int64
            )
            # Count cluster sizes
            for label in cluster_assignments:
                labels[label] = labels.get(label, 0) + 1
            n_clusters = 0
            # Assign unique cluster labels
            for label, count in labels.items():
                if count < min_samples:
                    labels[label] = -1
                else:
                    labels[label] = start_label + n_clusters
                    n_clusters += 1
            for i, label in enumerate(cluster_assignments):
                cluster_labels[i] = labels[label]
        # fill all  -1 labels with increasing labels
        mask = cluster_labels == -1
        n_singletons = np.count_nonzero(mask)
        cluster_labels[mask] = np.arange(
            start_label + n_clusters,
            start_label + n_clusters + n_singletons,
        )
        n_clusters += n_singletons
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
        The unit of the tolerance ('Da' or 'ppm' for precursor m/z;
        `None` for retention time, where distances are absolute).

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


@nb.njit(boundscheck=False)
def _offset_cluster_labels(
    cluster_labels: np.ndarray,
    splits: nb.typed.List,
) -> np.ndarray:
    """
    Renumber cluster_labels per split in-place so labels are globally unique,
    and return the label offset applied to each split.

    Parameters
    ----------
    cluster_labels : np.ndarray
        The cluster labels (mutated in place).
    splits : nb.typed.List
        A list of start and end indices of cluster chunks.

    Returns
    -------
    np.ndarray
        Per-split offsets (int64), one entry per split.
    """
    n_splits = len(splits) - 1
    offsets = np.empty(n_splits, np.int64)
    current_label = 0
    for i in range(n_splits):
        start, end = splits[i], splits[i + 1]
        offsets[i] = current_label
        cluster_labels[start:end] += current_label
        current_label = np.max(cluster_labels[start:end]) + 1
    return offsets


def _assign_global_cluster_labels(
    cluster_labels: np.ndarray,
    rep_spectra: List[ConsensusTuple],
    splits: nb.typed.List,
) -> List[ConsensusTuple]:
    """
    Convert cluster labels per split to unique labels (within charge).

    Parameters
    ----------
    cluster_labels : np.ndarray
        The cluster labels (mutated in place).
    rep_spectra : List[ConsensusTuple]
        The representative spectra.
    splits : nb.typed.List
        A list of start and end indices of cluster chunks.

    Returns
    -------
    List[ConsensusTuple]
        The representative spectra with updated cluster IDs.
    """
    offsets = _offset_cluster_labels(cluster_labels, splits)
    # Group reps by mz_split in one pass —- O(R) instead of O(S x R).
    reps_by_split: dict = {}
    for s in rep_spectra:
        reps_by_split.setdefault(int(s.mz_split), []).append(s)
    relabeled: List[ConsensusTuple] = []
    for i, offset in enumerate(offsets):
        for s in reps_by_split.get(i, ()):
            relabeled.append(
                s._replace(cluster_id=np.int32(s.cluster_id + int(offset)))
            )
    return relabeled
