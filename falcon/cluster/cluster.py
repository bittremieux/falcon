import gc
import logging
import math
import multiprocessing
import os
import tempfile
import time
from collections import defaultdict
from functools import partial
from typing import List, Tuple

import fastcluster
import joblib
import lance
import numba as nb
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import spectrum_utils.utils as suu
from scipy.cluster.hierarchy import fcluster

from . import similarity
from .consensus import (
    ConsensusTuple,
    _get_cluster_group_idx,
    _get_representative_spectra,
)
from .distance_matrix import (
    _condensed_rows,
    compute_condensed_distance_matrix,
    open_shared_condensed,
)

logger = logging.getLogger("falcon")

# Number of cluster tasks (cost-balanced chunks of m/z splits) to create per CPU
# worker. Making the global task count several times the worker count lets joblib
# load-balance dynamically — a worker that finishes early steals the next task —
# instead of stranding cores on the tail of a fixed one-chunk-per-worker split.
_CHUNKS_PER_WORKER = 8

# Columns fetched per interval from the lance dataset (the spectra needed to
# build the distance matrix and representatives).
_INTERVAL_COLUMNS = [
    "identifier",
    "precursor_mz",
    "precursor_charge",
    "retention_time",
    "mz",
    "intensity",
]

# Minimum pairs per distance tile. A giant interval is split into at most
# `cpu_count` tiles, but never so finely that a tile computes fewer than this
# many pairs — otherwise each tiny tile would re-read the whole interval for too
# little compute. ~few million pairs keeps the per-tile lance read well
# amortized over the cosine work.
_MIN_PAIRS_PER_TILE = 4_000_000

# Minimum spectra a cluster needs to survive post-processing splitting; passed
# as ``min_samples`` to `_postprocess_cluster` (singletons are dropped).
_MIN_CLUSTER_SAMPLES = 2

# Trigger an explicit `gc.collect()` only after finishing an interval with more
# than this many spectra; below it the per-interval allocations are small enough
# that forcing collection is pure overhead.
_GC_COLLECT_MIN_SPECTRA = 2**11


def _tile_threshold(batch_size: int) -> int:
    """
    Minimum interval size (spectra) that triggers a tiled distance build.

    Intervals at least this large would each grind in a single worker and strand
    cores at the end of the run, so their cosine build is tiled across the pool.
    Tracking ``batch_size // 2`` ties the threshold to the cap on interval size
    (`_get_precursor_mz_splits`), so it self-scales when ``batch_size`` changes
    and catches exactly the forced-split giants in ``[batch_size/2, batch_size]``.
    """
    return max(2, batch_size // 2)


def _format_hms(seconds: float) -> str:
    """Format a duration in seconds as ``HH:MM:SS`` (hours not zero-padded)."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _charge_label(dataset: "lance.LanceDataset") -> str:
    """
    Human-readable charge of a per-charge bucket, parsed from its dataset URI
    (e.g. ``spectra_charge_2.lance`` -> ``2``, ``spectra_charge_1_2`` -> ``1, 2``,
    ``spectra_charge_unknown`` -> ``unknown``).
    """
    return (
        dataset.uri.split("spectra_charge_")[-1].split(".")[0].replace("_", ", ")
    )


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
    # Single-bucket convenience wrapper around the shared-pool implementation.
    return generate_clusters_multi(
        [dataset],
        linkage,
        distance_threshold,
        min_matches,
        precursor_tol_mass,
        precursor_tol_mode,
        rt_tol,
        fragment_tol,
        batch_size,
        consensus_method,
        consensus_params,
    )[0]


def generate_clusters_multi(
    bucket_datasets: List["lance.LanceDataset"],
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
) -> List[Tuple[np.ndarray, List[ConsensusTuple]]]:
    """
    Cluster several per-charge datasets using a single shared worker pool.

    All buckets' m/z-split tasks are dispatched into one ``joblib`` pool, so a
    worker that finishes one bucket's work immediately picks up another bucket's
    work instead of each bucket running its own pool with an end-of-bucket
    barrier (which left cores idle between/within small buckets). Clustering of
    each m/z split is independent of every other split and bucket, so the
    resulting partitions are identical to clustering the buckets one at a time.

    Parameters
    ----------
    bucket_datasets : List[lance.LanceDataset]
        The per-charge datasets to cluster.
    linkage, distance_threshold, min_matches, precursor_tol_mass,
    precursor_tol_mode, rt_tol, fragment_tol, batch_size, consensus_method,
    consensus_params
        See `generate_clusters`; applied identically to every bucket.

    Returns
    -------
    List[Tuple[np.ndarray, List[ConsensusTuple]]]
        For each input dataset (in the same order): the cluster labels and the
        representative spectra for that bucket.
    """
    logger.debug(
        "Hierarchical clustering (distance_threshold=%.4f, min_samples=%d)",
        distance_threshold,
        _MIN_CLUSTER_SAMPLES,
    )
    # Build the m/z-split tasks for every bucket up front. `prepared` keeps only
    # the lightweight (n, splits) needed to reassemble each bucket; the loaded
    # metadata DataFrames are released inside `_prepare_bucket_chunks`.
    #
    # Create several chunks per worker (not one) so the single shared pool has
    # many more tasks than workers and can load-balance dynamically.
    n_workers = multiprocessing.cpu_count()
    num_chunks = _CHUNKS_PER_WORKER * n_workers
    # Intervals with at least half of `batch_size` spectra would each grind in a
    # single worker and strand cores at the end of the run; their distance build
    # is tiled across the pool instead. The threshold tracks `batch_size` (which
    # caps interval size), so it self-scales if `batch_size` changes.
    tile_threshold = _tile_threshold(batch_size)
    prepared = []  # per bucket: (n_spectra, splits)
    chunk_tasks = []  # (cost, bucket_id, n_chunk, dataset, data_chunk)
    giants = []  # per giant interval: routing + size, augmented below
    for bucket_id, dataset in enumerate(bucket_datasets):
        n, splits, data_chunks, giant_intervals = _prepare_bucket_chunks(
            dataset,
            precursor_tol_mass,
            precursor_tol_mode,
            batch_size,
            num_chunks,
            tile_threshold,
        )
        prepared.append((n, splits))
        for data_chunk in data_chunks:
            # Cost is ~quadratic in interval size (the pairwise distance matrix
            # dominates), so use it to schedule the heaviest chunks first and to
            # drive the progress estimate (cost ~ compute time, so % of cost
            # done grows ~linearly with wall-clock when all cores are busy).
            cost = sum(len(idx_iv) ** 2 for _, _, idx_iv, _ in data_chunk)
            n_chunk = sum(len(idx_iv) for _, _, idx_iv, _ in data_chunk)
            chunk_tasks.append(
                (cost, bucket_id, n_chunk, dataset, data_chunk)
            )
        for task_id, row_ids, idx_iv, mz_iv, n_iv in giant_intervals:
            giants.append(
                {
                    "bucket_id": bucket_id,
                    "dataset": dataset,
                    "task_id": task_id,
                    "row_ids": row_ids,
                    "idx": idx_iv,
                    "mz": mz_iv,
                    "n": n_iv,
                    "cost": n_iv * n_iv,  # progress weight (chunk convention)
                }
            )

    # Dispatch the heaviest chunks first so the dominant bucket's big intervals
    # start immediately and the many small-bucket chunks backfill around them,
    # instead of the dominant bucket (often the last one, e.g. unknown charge)
    # running alone at the end with cores tapering off. Sort on cost only —
    # datasets are not orderable.
    chunk_tasks.sort(key=lambda t: t[0], reverse=True)

    results_by_bucket = defaultdict(list)
    memmap_paths = []
    try:
        # Allocate one shared, on-disk condensed matrix per giant interval and
        # build its work-balanced distance tiles. Several processes fill disjoint
        # row-bands of the same memmap concurrently; the file is unlinked in the
        # `finally` once every tile and the finalize step are done.
        tile_tasks = []  # (giant_idx, row_ids, r0, r1, path, tile_cost)
        for gi, g in enumerate(giants):
            n = g["n"]
            path, mm = open_shared_condensed(n)
            mm.flush()
            del mm  # workers reopen the file by path; drop the writer view here
            memmap_paths.append(path)
            g["path"] = path
            total_pairs = n * (n - 1) // 2
            n_tiles = min(
                n_workers, max(1, total_pairs // _MIN_PAIRS_PER_TILE)
            )
            for r0, r1 in _interval_row_bounds(n, n_tiles):
                tile_pairs = (2 * n - 1 - r0 - r1) * (r1 - r0) / 2
                tile_cost = g["cost"] * tile_pairs / total_pairs
                tile_tasks.append((gi, g["row_ids"], r0, r1, path, tile_cost))

        total_cost = sum(t[0] for t in chunk_tasks) + sum(
            g["cost"] for g in giants
        )
        total_spectra = sum(n for n, _ in prepared)
        n_pass1 = len(chunk_tasks) + len(tile_tasks)
        done_cost = done_spectra = done_tasks = 0
        next_log_pct, start = 5, time.time()
        if n_pass1:
            logger.info(
                "Clustering %d spectra across %d charge bucket(s) in %d "
                "task(s) (%d giant interval(s) tiled) on %d worker(s)...",
                total_spectra,
                len(bucket_datasets),
                n_pass1,
                len(giants),
                n_workers,
            )
            process_chunk = partial(
                _cluster_chunk_tagged,
                linkage=linkage,
                distance_threshold=distance_threshold,
                min_matches=min_matches,
                precursor_tol_mass=precursor_tol_mass,
                precursor_tol_mode=precursor_tol_mode,
                rt_tol=rt_tol,
                fragment_mz_tol=fragment_tol,
                consensus_method=consensus_method,
                consensus_params=consensus_params,
                batch_size=batch_size,
            )

            # One shared pool. Giant tiles are submitted first (they are the
            # critical path) and the normal chunks backfill; `generator_unordered`
            # yields each result as it finishes so progress reflects actual
            # completions and each result carries the tag needed to route it.
            def _pass1_jobs():
                for gi, row_ids, r0, r1, path, tile_cost in tile_tasks:
                    yield joblib.delayed(_tile_distance)(
                        giants[gi]["dataset"],
                        row_ids,
                        r0,
                        r1,
                        path,
                        fragment_tol,
                        min_matches,
                        gi,
                        tile_cost,
                    )
                for cost, bucket_id, n_chunk, dataset, data_chunk in chunk_tasks:
                    yield joblib.delayed(process_chunk)(
                        data_chunk,
                        dataset=dataset,
                        bucket_id=bucket_id,
                        cost=cost,
                        n_chunk=n_chunk,
                    )

            results = joblib.Parallel(
                n_jobs=-1,
                backend="loky",
                verbose=0,
                return_as="generator_unordered",
            )(_pass1_jobs())
            for result in results:
                if result[0] == "chunk":
                    _, bucket_id, cost, n_chunk, chunk_results = result
                    results_by_bucket[bucket_id].extend(chunk_results)
                    done_cost += cost
                    done_spectra += n_chunk
                else:  # "tile" — a giant interval's row-band finished
                    done_cost += result[2]
                done_tasks += 1
                pct = 100 * done_cost / total_cost if total_cost else 100
                if pct >= next_log_pct or done_tasks == n_pass1:
                    elapsed = time.time() - start
                    eta = (
                        elapsed * (total_cost - done_cost) / done_cost
                        if done_cost
                        else 0
                    )
                    logger.info(
                        "Clustering progress: %3.0f%% | %d/%d tasks | "
                        "%d/%d spectra processed | elapsed %s | ETA ~%s",
                        pct,
                        done_tasks,
                        n_pass1,
                        done_spectra,
                        total_spectra,
                        _format_hms(elapsed),
                        _format_hms(eta),
                    )
                    # Advance to the next 5% milestone past current progress.
                    next_log_pct = (int(pct) // 5 + 1) * 5

        # Pass 2: cluster each giant interval from its now-filled memmap. The
        # cosine build is done; only the cheap linkage/refinement remains, run on
        # the pool so the few giants' finalizes overlap.
        if giants:
            logger.info("Finalizing %d giant interval(s)...", len(giants))
            finalize_results = joblib.Parallel(
                n_jobs=-1,
                backend="loky",
                verbose=0,
                return_as="generator_unordered",
            )(
                joblib.delayed(_finalize_giant)(
                    g["dataset"],
                    g["row_ids"],
                    g["idx"],
                    g["mz"],
                    g["path"],
                    g["bucket_id"],
                    g["task_id"],
                    g["n"],
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
                for g in giants
            )
            for _, bucket_id, task_id, n_chunk, result in finalize_results:
                results_by_bucket[bucket_id].append((task_id, result))
                done_spectra += n_chunk
    finally:
        for path in memmap_paths:
            try:
                os.unlink(path)
            except OSError:
                pass

    # Reassemble each bucket independently from its split results.
    return [
        _finalize_bucket(
            n,
            splits,
            results_by_bucket[bucket_id],
            _charge_label(bucket_datasets[bucket_id]),
        )
        for bucket_id, (n, splits) in enumerate(prepared)
    ]


def _cluster_chunk_tagged(
    data_chunk: List[Tuple], dataset: "lance.LanceDataset", **cluster_kwargs
) -> Tuple[str, int, int, int, List[Tuple]]:
    """
    Run `cluster_chunk` and tag the result for routing and progress accounting.

    Returns ``("chunk", bucket_id, cost, n_chunk, chunk_results)``. The leading
    ``"chunk"`` kind tag lets the consumer distinguish full-clustering results
    from distance-tile results (`_tile_distance`) when both stream out of one
    pool in submission-independent order; the rest routes each chunk back to its
    bucket and counts its cost/size toward progress. `bucket_id`, `cost` and
    `n_chunk` are popped from `cluster_kwargs` (the rest go to `cluster_chunk`).
    """
    bucket_id = cluster_kwargs.pop("bucket_id")
    cost = cluster_kwargs.pop("cost")
    n_chunk = cluster_kwargs.pop("n_chunk")
    return (
        "chunk",
        bucket_id,
        cost,
        n_chunk,
        cluster_chunk(data_chunk, dataset=dataset, **cluster_kwargs),
    )


def _interval_row_bounds(n: int, n_tiles: int) -> List[Tuple[int, int]]:
    """
    Split rows ``[0, n)`` of an interval's upper triangle into ``n_tiles``
    work-balanced contiguous ranges.

    Only the upper triangle is computed, so row ``i`` contributes ``n - 1 - i``
    pairs; equal row counts would make early tiles do far more work. Splitting on
    the cumulative pair count gives each tile ~equal pairs (early tiles get a few
    heavy rows, later tiles many light rows). Empty ranges are dropped.
    """
    cum = np.cumsum(n - 1 - np.arange(n))  # cum[-1] == n * (n - 1) // 2
    edges = [
        int(np.searchsorted(cum, cum[-1] * p / n_tiles))
        for p in range(1, n_tiles)
    ]
    bounds = [0] + edges + [n]
    bounds = sorted(set(min(b, n) for b in bounds))
    return [
        (bounds[i], bounds[i + 1])
        for i in range(len(bounds) - 1)
        if bounds[i] < bounds[i + 1]
    ]


def _tile_distance(
    dataset: "lance.LanceDataset",
    row_ids: List[int],
    r0: int,
    r1: int,
    memmap_path: str,
    fragment_mz_tol: float,
    min_matches: int,
    giant_idx: int,
    tile_cost: float,
) -> Tuple[str, int, float]:
    """
    Compute rows ``[r0, r1)`` of one giant interval's condensed distance matrix.

    Reads the interval's spectra from `dataset`, orders them with the shared
    `_interval_spectrum_tuples` (so the matrix is indexed identically to the
    finalize step), and writes its row-band into the interval's shared memmap.
    Runs single-threaded — parallelism comes from dispatching one of these per
    tile into the pool. Returns ``("tile", giant_idx, tile_cost)`` for routing
    and progress accounting.
    """
    spectra = dataset.take(indices=row_ids, columns=_INTERVAL_COLUMNS)
    spectra, _ = _interval_spectrum_tuples(spectra.to_pandas())
    condensed = np.lib.format.open_memmap(memmap_path, mode="r+")
    _condensed_rows(
        condensed, spectra, r0, r1, fragment_mz_tol, min_matches
    )
    condensed.flush()
    return ("tile", giant_idx, tile_cost)


def _finalize_giant(
    dataset: "lance.LanceDataset",
    row_ids: List[int],
    idx_interval: np.ndarray,
    mz_interval: np.ndarray,
    memmap_path: str,
    bucket_id: int,
    task_id: int,
    n_chunk: int,
    **cluster_kwargs,
) -> Tuple[str, int, int, int, Tuple]:
    """
    Cluster a giant interval from its already-filled condensed matrix.

    Opens the shared memmap (filled by this interval's `_tile_distance` tasks)
    and runs the back half of `_cluster_mz_interval` (linkage, flat clustering,
    m/z/RT refinement, representatives) with the cosine build skipped. Returns
    ``("giant", bucket_id, task_id, n_chunk, (rep_spectra, labels))`` so the
    consumer can route it exactly like a normal chunk's per-interval result.
    """
    spectra = dataset.take(indices=row_ids, columns=_INTERVAL_COLUMNS)
    pdist = np.lib.format.open_memmap(memmap_path, mode="r")
    result = _cluster_mz_interval(
        spectra.to_pandas(),
        idx_interval,
        mz_interval,
        pdist=pdist,
        **cluster_kwargs,
    )
    return ("giant", bucket_id, task_id, n_chunk, result)


def _prepare_bucket_chunks(
    dataset: "lance.LanceDataset",
    precursor_tol_mass: float,
    precursor_tol_mode: str,
    batch_size: int,
    num_chunks: int,
    tile_threshold: int,
) -> Tuple[int, "nb.typed.List", List[List[Tuple]], List[Tuple]]:
    """
    Load and sort one bucket's metadata and build its m/z-split cluster tasks.

    Parameters
    ----------
    num_chunks : int
        Target number of cost-balanced chunks to bundle this bucket's m/z splits
        into. Empty chunks are dropped, so a bucket with fewer splits than
        `num_chunks` simply yields one chunk per split.
    tile_threshold : int
        m/z splits with at least this many spectra are "giant" intervals whose
        single-worker cosine build would strand cores at the tail. They are kept
        out of the cost-balanced chunks and returned separately so the caller can
        tile their distance build across the pool.

    Returns
    -------
    Tuple[int, nb.typed.List, List[List[Tuple]], List[Tuple]]
        The number of spectra, the m/z split boundaries, the cost-balanced chunks
        of normal intervals (each a list of ``(task_id, row_ids, idx_interval,
        mz_interval)``) ready for `cluster_chunk`, and the giant intervals (each
        ``(task_id, row_ids, idx_interval, mz_interval, n_interval)``). The loaded
        DataFrame is released before returning; only the lightweight arrays
        referenced by the chunks/giants survive.
    """
    # Load only the columns needed for sorting and splitting; full spectra data
    # (mz, intensity) is fetched on demand per interval inside
    # _cluster_mz_interval.
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
    n = data.shape[0]
    logger.info(
        "Cluster %d spectra with charge %s", n, _charge_label(dataset)
    )
    idx = data["index"].values
    mzs = data["precursor_mz"].values
    splits = _get_precursor_mz_splits(
        mzs, precursor_tol_mass, precursor_tol_mode, batch_size
    )
    # Per m/z split clustering, bundled into cost-balanced chunks to amortize
    # per-task dispatch overhead while keeping enough chunks for load balancing.
    chunks = cost_based_chunking(splits, num_chunks)
    data_chunks = []
    giant_intervals = []
    for chunk in chunks:
        data_chunk = []
        for task_id, (interval_start, interval_stop) in chunk:
            row_ids = idx[interval_start:interval_stop]
            idx_interval = idx[interval_start:interval_stop]
            mz_interval = mzs[interval_start:interval_stop]
            n_interval = interval_stop - interval_start
            if n_interval >= tile_threshold:
                # Giant interval: handled via a tiled distance build, not bundled
                # into a single-worker chunk.
                giant_intervals.append(
                    (
                        task_id,
                        row_ids.tolist(),
                        idx_interval,
                        mz_interval,
                        n_interval,
                    )
                )
            else:
                data_chunk.append(
                    (task_id, row_ids.tolist(), idx_interval, mz_interval)
                )
        if data_chunk:
            data_chunks.append(data_chunk)
    return n, splits, data_chunks, giant_intervals


def _finalize_bucket(
    n: int,
    splits: "nb.typed.List",
    flattened_results: List[
        Tuple[int, Tuple[List[ConsensusTuple], np.ndarray]]
    ],
    charge: str,
) -> Tuple[np.ndarray, List[ConsensusTuple]]:
    """
    Assemble one bucket's per-split results into cluster labels and reps.

    Parameters
    ----------
    n : int
        The number of spectra in the bucket.
    splits : nb.typed.List
        The m/z split boundaries from `_prepare_bucket_chunks`.
    flattened_results : List[Tuple[int, Tuple[List[ConsensusTuple], np.ndarray]]]
        The ``(task_id, (rep_spectra, labels))`` results for this bucket's splits
        (in any order; re-sorted by task_id here).
    charge : str
        Human-readable charge label of this bucket (for logging).

    Returns
    -------
    Tuple[np.ndarray, List[ConsensusTuple]]
        The cluster labels (aligned to the bucket's sorted order) and the
        representative spectra with bucket-unique cluster IDs.
    """
    with tempfile.NamedTemporaryFile(suffix=".npy") as cluster_file:
        cluster_filename = cluster_file.name
        cluster_labels = np.lib.format.open_memmap(
            cluster_filename, mode="w+", dtype=np.int32, shape=(n,)
        )
        cluster_labels.fill(-1)
        representative_spectra = []
        # Restore split order so positional scatter into cluster_labels is
        # correct regardless of worker completion order.
        for task_id, (interval_rep_spectra, labels) in sorted(
            flattened_results, key=lambda x: x[0]
        ):
            if interval_rep_spectra is not None:
                # add task id (mz_split) to rep_spectra
                interval_rep_spectra = [
                    s._replace(mz_split=np.int32(task_id))
                    for s in interval_rep_spectra
                ]
                representative_spectra.extend(interval_rep_spectra)
                cluster_labels[splits[task_id] : splits[task_id + 1]] = labels
        representative_spectra = _assign_global_cluster_labels(
            cluster_labels, representative_spectra, splits
        )
        _, counts = np.unique(cluster_labels, return_counts=True)
        n_clusters = np.count_nonzero(counts > 1)
        n_noise = np.count_nonzero(counts == 1)
        logger.info(
            "Charge %s: %d spectra grouped in %d clusters, %d spectra remain "
            "as singletons",
            charge,
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
    batch_size: int = None,
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
    batch_size : int, optional
        Maximum number of spectra to hold in memory per batched Lance read.
        Intervals are read in groups whose cumulative size stays within this
        cap, bounding a worker's resident memory to roughly one interval's worth
        regardless of the chunk's total size (important at hundreds of millions
        / billions of spectra). If `None`, the whole chunk is read at once.

    Returns
    -------
    List[Tuple[int, Tuple[List[ConsensusTuple], np.ndarray]]]
        For each task: the task index and the result of `_cluster_mz_interval`.
    """
    if not chunk:
        return []
    columns = _INTERVAL_COLUMNS

    def _cluster_group(group):
        # One batched Lance read for the group amortizes the per-interval read
        # and pandas-conversion overhead; `take` preserves index order, so each
        # interval gets a contiguous slice in its original `row_ids` order.
        group_row_ids = [rid for _, row_ids, _, _ in group for rid in row_ids]
        group_spectra = dataset.take(
            indices=group_row_ids, columns=columns
        ).to_pandas()
        out, offset = [], 0
        for i, row_ids, idx, mzs in group:
            interval_spectra = group_spectra.iloc[
                offset : offset + len(row_ids)
            ]
            offset += len(row_ids)
            out.append(
                (
                    i,
                    _cluster_mz_interval(
                        interval_spectra,
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
            )
        return out

    # Read intervals in groups capped at `batch_size` spectra so a single read
    # never holds more than ~one interval's worth in memory, while still
    # amortizing the read over many small intervals. A single interval (already
    # <= batch_size) is never split across reads.
    results = []
    group, group_size = [], 0
    for task in chunk:
        m = len(task[1])
        if group and batch_size is not None and group_size + m > batch_size:
            results.extend(_cluster_group(group))
            group, group_size = [], 0
        group.append(task)
        group_size += m
    if group:
        results.extend(_cluster_group(group))
    return results


def _interval_spectrum_tuples(
    spectra: pd.DataFrame,
) -> Tuple[List["similarity.SpectrumTuple"], np.ndarray]:
    """
    Sort an interval's spectra into the canonical order and build SpectrumTuples.

    Single source of truth for an interval's spectrum ordering: the spectra are
    sorted with the same total-order key as the outer sort in
    `_prepare_bucket_chunks` (``precursor_mz``, ``retention_time``,
    ``identifier`` as a unique tiebreaker) so positions line up with the
    ``idx``/``mzs`` arrays. Both `_cluster_mz_interval` and the tiled distance
    build (`_tile_distance`) call this, guaranteeing the condensed matrix and the
    finalize step index the same spectra in the same order.

    Returns
    -------
    Tuple[List[similarity.SpectrumTuple], np.ndarray]
        The SpectrumTuples in sorted order and the matching retention times.
    """
    spectra = spectra.sort_values(
        ["precursor_mz", "retention_time", "identifier"]
    )
    rts = spectra["retention_time"].values
    # Build SpectrumTuples straight from the columns instead of a row-wise
    # pandas `apply` (which constructs a Series per spectrum and dominates the
    # single-threaded per-interval cost on large intervals).
    precursor_mz = spectra["precursor_mz"].to_numpy()
    precursor_charge = spectra["precursor_charge"].to_numpy()
    mz = spectra["mz"].to_numpy()
    intensity = spectra["intensity"].to_numpy()
    spectra = [
        similarity.SpectrumTuple(
            precursor_mz[i], precursor_charge[i], mz[i], intensity[i]
        )
        for i in range(len(precursor_mz))
    ]
    return spectra, rts


def _cluster_mz_interval(
    spectra: pd.DataFrame,
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
    pdist: np.ndarray = None,
) -> Tuple[List[ConsensusTuple], np.ndarray]:
    """
    Cluster the spectra in a single precursor m/z interval.

    Parameters
    ----------
    spectra : pd.DataFrame
        This interval's spectra (identifier, precursor_mz, precursor_charge,
        retention_time, mz, intensity), pre-fetched in batch by `cluster_chunk`.
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
    pdist : np.ndarray, optional
        A precomputed condensed distance matrix for this interval (in the order
        produced by `_interval_spectrum_tuples`). When provided (the tiled giant
        path), the expensive `compute_condensed_distance_matrix` build is
        skipped and this matrix is used directly. Must be indexed identically to
        the spectra returned by `_interval_spectrum_tuples`.

    Returns
    -------
    Tuple[List[ConsensusTuple], np.ndarray]
        The representative spectrum for each cluster and the cluster label
        array aligned to the input interval order.
    """
    spectra, rts = _interval_spectrum_tuples(spectra)
    n_spectra = len(spectra)
    cluster_labels = -np.ones(n_spectra, np.int32)
    if n_spectra > 1:
        # Hierarchical clustering of the vectors.
        # Subtract 1 because fcluster starts with cluster label 1 instead of 0
        # (like Scikit-Learn does).
        if pdist is None:
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
                _MIN_CLUSTER_SAMPLES,
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
        if n_spectra > _GC_COLLECT_MIN_SPECTRA:
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
            ).astype(np.int32)
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
                        ).astype(np.int32)
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
