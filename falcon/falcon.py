import collections
import functools
import glob
import logging
import multiprocessing
import os
import queue
import shutil
import sys
import tempfile
import threading
from typing import Callable, Dict, List, Set, Tuple, Union

import joblib
import lance
import natsort
import numpy as np
import pandas as pd
import pyarrow as pa

from . import __version__, utils
from .cluster import cluster, spectrum
from .config import config
from .ms_io import ms_io

logger = logging.getLogger("falcon")

utils.set_seeds()

# Flush a charge bucket's buffered spectra to its lance dataset once this many
# have accumulated, bounding the per-bucket in-memory batch.
_LANCE_WRITE_BATCH_SIZE = 10_000
# Cap on spectra buffered in the reader -> writer queue, bounding peak memory
# during spectrum preprocessing.
_MAX_SPECTRA_IN_MEMORY = 1_000_000


def main(args: Union[str, List[str], None] = None) -> int:
    # Configure logging.
    logger = utils.configure_logger()

    # Load the configuration.
    config.parse(args)
    logger.info("falcon version %s", str(__version__))
    logger.debug("work_dir = %s", config.work_dir)
    logger.debug("overwrite = %s", config.overwrite)
    logger.debug("export_representatives = %s", config.export_representatives)
    logger.debug("precursor_tol = %.2f %s", *config.precursor_tol)
    logger.debug("rt_tol = %s", config.rt_tol)
    logger.debug("fragment_tol = %.2f", config.fragment_tol)
    logger.debug("linkage = %s", config.linkage)
    logger.debug("distance_threshold = %.3f", config.distance_threshold)
    logger.debug("min_matched_peaks = %d", config.min_matched_peaks)
    logger.debug("consensus_method = %s", config.consensus_method)
    logger.debug("outlier_cutoff_lower = %.2f", config.outlier_cutoff_lower)
    logger.debug("outlier_cutoff_upper = %.2f", config.outlier_cutoff_upper)
    logger.debug("batch_size = %d", config.batch_size)
    logger.debug(
        "precursor_charge_buckets = %s", config.precursor_charge_buckets
    )
    logger.debug("min_peaks = %d", config.min_peaks)
    logger.debug("min_mz_range = %.2f", config.min_mz_range)
    logger.debug("min_mz = %.2f", config.min_mz)
    logger.debug("max_mz = %.2f", config.max_mz)
    logger.debug("remove_precursor_tol = %.2f", config.remove_precursor_tol)
    logger.debug("min_intensity = %.2f", config.min_intensity)
    logger.debug("max_peaks_used = %d", config.max_peaks_used)
    logger.debug("scaling = %s", config.scaling)

    rm_work_dir = False
    if config.work_dir is None:
        config.work_dir = tempfile.mkdtemp()
        rm_work_dir = True
    elif os.path.isdir(config.work_dir):
        logger.warning(
            "Working directory %s already exists, previous "
            "results might get overwritten",
            config.work_dir,
        )
    os.makedirs(config.work_dir, exist_ok=True)
    os.makedirs(os.path.join(config.work_dir, "spectra"), exist_ok=True)

    # Clean all intermediate and final results if "overwrite" is specified,
    # otherwise abort if the output files already exist.
    exit_exists = False
    if os.path.isfile(f"{config.output_filename}.csv"):
        if config.overwrite:
            logger.warning(
                "Output file %s (cluster assignments) already "
                "exists and will be overwritten",
                f"{config.output_filename}.csv",
            )
            os.remove(f"{config.output_filename}.csv")
        else:
            logger.error(
                "Output file %s (cluster assignments) already "
                "exists, aborting...",
                f"{config.output_filename}.csv",
            )
            exit_exists = True
    if os.path.isfile(f"{config.output_filename}.mgf"):
        if config.overwrite:
            logger.warning(
                "Output file %s (cluster representatives) already "
                "exists and will be overwritten",
                f"{config.output_filename}.mgf",
            )
            os.remove(f"{config.output_filename}.mgf")
        else:
            logger.error(
                "Output file %s (cluster representatives) already "
                "exists, aborting...",
                f"{config.output_filename}.mgf",
            )
            exit_exists = True
    if exit_exists:
        logging.shutdown()
        return 1

    # Check if the spectral averaging configuration is valid.
    if (
        config.consensus_method == "average"
        and config.outlier_cutoff_lower < 1
        and config.outlier_cutoff_upper < 1
    ):
        logger.warning(
            "Setting both outlier_cutoff_lower and outlier_cutoff_upper "
            "to values less than 1 can lead to unexpected results. It "
            "is advised to set either outlier_cutoff_lower or "
            "outlier_cutoff_upper to a value >= 1."
        )

    _, min_mz, max_mz = spectrum.get_dim(
        config.min_mz, config.max_mz, config.fragment_tol
    )
    process_spectrum = functools.partial(
        spectrum.process_spectrum,
        min_peaks=config.min_peaks,
        min_mz_range=config.min_mz_range,
        mz_min=min_mz,
        mz_max=max_mz,
        remove_precursor_tolerance=config.remove_precursor_tol,
        min_intensity=config.min_intensity,
        max_peaks_used=config.max_peaks_used,
        scaling=None if config.scaling == "off" else config.scaling,
    )

    if config.overwrite:
        for filename in os.listdir(os.path.join(config.work_dir, "spectra")):
            path = os.path.join(config.work_dir, "spectra", filename)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    charge_path = os.path.join(config.work_dir, "spectra", "charges.joblib")
    if os.path.isfile(charge_path) and not config.overwrite:
        cached = joblib.load(charge_path)
        # Validate against the *configuration* that produced the cached
        # partitioning, not just the resulting buckets. This catches switching
        # between explicit buckets and auto mode (None), which would otherwise
        # silently reuse a partitioning grouped differently than expected.
        if (
            not isinstance(cached, dict)
            or cached["config"] != config.precursor_charge_buckets
        ):
            raise ValueError(
                "Cached charge buckets do not match the current "
                "--precursor_charge_buckets configuration, rerun "
                "with --overwrite or a different --work_dir."
            )
        charge_buckets = cached["buckets"]
    else:
        # Recalculate the charge buckets and recreate dataset.
        charge_buckets = _prepare_spectra(
            process_spectrum, config.precursor_charge_buckets
        )
        joblib.dump(
            {
                "config": config.precursor_charge_buckets,
                "buckets": charge_buckets,
            },
            charge_path,
        )

    # Cluster the spectra per charge. All non-empty charge buckets are clustered
    # in a single shared worker pool (`cluster.generate_clusters_multi`) so cores
    # stay busy across bucket boundaries instead of idling at each bucket's
    # end-of-pool barrier; the resulting partitions are identical to clustering
    # each bucket on its own.
    consensus_params = {}
    if config.consensus_method == "average":
        consensus_params["min_mz"] = config.min_mz
        consensus_params["max_mz"] = config.max_mz
        consensus_params["bin_size"] = 2 * config.fragment_tol
        consensus_params["outlier_cutoff_lower"] = config.outlier_cutoff_lower
        consensus_params["outlier_cutoff_upper"] = config.outlier_cutoff_upper

    bucket_datasets = []
    for bucket in charge_buckets:
        dataset_path = os.path.join(
            config.work_dir,
            "spectra",
            f"spectra_charge{bucket_key_to_str(bucket)}.lance",
        )
        dataset = lance.dataset(dataset_path)
        # No valid spectra found with the current charge.
        if dataset.count_rows() == 0:
            continue
        bucket_datasets.append(dataset)

    cluster_results = cluster.generate_clusters_multi(
        bucket_datasets,
        config.linkage,
        config.distance_threshold,
        config.min_matched_peaks,
        config.precursor_tol[0],
        config.precursor_tol[1],
        config.rt_tol,
        config.fragment_tol,
        config.batch_size,
        config.consensus_method,
        consensus_params,
    )

    clusters_all, current_label, representatives = [], 0, []
    for dataset, (clusters, rep_spectra) in zip(
        bucket_datasets, cluster_results
    ):
        # Make sure that different charges have non-overlapping cluster labels.
        clusters += current_label
        rep_spectra = [
            s._replace(cluster_id=s.cluster_id + current_label)
            for s in rep_spectra
        ]
        # noinspection PyUnresolvedReferences
        current_label = np.amax(clusters) + 1
        # Save cluster assignments.
        metadata = (
            dataset.to_table(
                columns=[
                    "identifier",
                    "precursor_charge",
                    "precursor_mz",
                    "retention_time",
                ]
            ).to_pandas()
            # Must match the ordering used in `_prepare_bucket_chunks` exactly so
            # the positionally-aligned cluster labels bind to the correct
            # spectra; `identifier` is the unique tiebreaker that makes both
            # sorts agree.
            .sort_values(["precursor_mz", "retention_time", "identifier"])
        )
        metadata["cluster"] = clusters
        clusters_all.append(metadata)
        # Extract identifiers for cluster representatives (medoids).
        if config.export_representatives:
            representatives.extend(rep_spectra)

    # Export cluster memberships and representative spectra.
    clusters_all = pd.concat(clusters_all, ignore_index=True).sort_values(
        ["identifier"], key=natsort.natsort_keygen()
    )
    logger.info(
        "Export cluster assignments of %d spectra to %d unique "
        "clusters to output file %s",
        len(clusters_all),
        clusters_all["cluster"].nunique(),
        f"{config.output_filename}.csv",
    )
    # Perform IO in a separate worker process.
    write_csv_worker = threading.Thread(
        target=_write_cluster_info, args=(clusters_all,), daemon=True
    )
    write_csv_worker.start()
    if config.export_representatives:
        logger.info(
            "Export %d cluster representative spectra to output file %s",
            len(representatives),
            f"{config.output_filename}.mgf",
        )
        # Perform IO in a separate worker process.
        write_mgf_worker = threading.Thread(
            target=ms_io.write_spectra,
            args=(f"{config.output_filename}.mgf", representatives),
            daemon=True,
        )
        write_mgf_worker.start()
        write_mgf_worker.join()
    write_csv_worker.join()

    if rm_work_dir:
        shutil.rmtree(config.work_dir)

    logging.shutdown()
    return 0


def _prepare_spectra(
    process_spectrum: Callable,
    charge_buckets: List[Union[Set[Union[int, str]], str]],
) -> List[Union[Set[Union[int, str]], str]]:
    """
    Read the spectra from the input peak files and partition to intermediate
    files split and sorted by precursor m/z.

    Parameters
    ----------
    process_spectrum : Callable
        The function to process the spectra.

    Returns
    -------
    List[Set[Union[int, str]]]
        The valid precursor charge buckets.
    """
    input_filenames = [
        fn for pattern in config.input_filenames for fn in glob.glob(pattern)
    ]
    logger.info(
        "Reading spectra from %d peak file(s)...", len(input_filenames)
    )
    # Use multiple worker processes to read the peak files.
    max_file_workers = min(len(input_filenames), multiprocessing.cpu_count())
    # Restrict the number of spectra simultaneously in memory to avoid
    # excessive memory requirements.
    spectra_queue = queue.Queue(maxsize=_MAX_SPECTRA_IN_MEMORY)
    # Per-charge locks so writers serialize only within a charge's dataset,
    # not across unrelated charges.
    lance_locks = _PerChargeLockRegistry()
    schema = pa.schema(
        [
            pa.field("identifier", pa.string()),
            pa.field("precursor_mz", pa.float32()),
            pa.field("precursor_charge", pa.int8()),
            pa.field("mz", pa.list_(pa.float32())),
            pa.field("intensity", pa.list_(pa.float32())),
            pa.field("retention_time", pa.float32()),
        ]
    )
    # When no buckets are specified, every distinct charge (including missing
    # charges) is caught and clustered separately.
    auto_bucket = charge_buckets is None
    # create a mapping from charge to bucket
    charge_to_bucket = {}
    catch_other_charges = False
    if not auto_bucket:
        for bucket in charge_buckets:
            if bucket == "other":
                catch_other_charges = True
                continue  # handled later
            for charge in bucket:
                if charge in charge_to_bucket:
                    raise ValueError(
                        f"Charge {charge} appears in more than one bucket"
                    )
                bucket_tuple = tuple(
                    sorted(bucket, key=lambda x: (isinstance(x, str), x))
                )
                charge_to_bucket[charge] = bucket_tuple

    lance_writers = multiprocessing.pool.ThreadPool(
        max_file_workers,
        _write_spectra_lance,
        (
            spectra_queue,
            lance_locks,
            schema,
            charge_to_bucket,
            catch_other_charges,
            auto_bucket,
        ),
    )
    # Read the peak files and put their spectra in the queue for consumption
    # by the lance writers. Using return_as="generator_unordered" so results
    # are yielded as each worker finishes rather than after all workers are
    # done, allowing the queue's maxsize to actually bound memory usage.
    low_quality_counter = 0
    for file_spectra, lqc in joblib.Parallel(
        n_jobs=max_file_workers, return_as="generator_unordered"
    )(
        joblib.delayed(_read_spectra)(file, process_spectrum)
        for file in input_filenames
    ):
        low_quality_counter += lqc
        for spec in file_spectra:
            spectra_queue.put(spec)
    # Add sentinels to indicate stopping.
    for _ in range(max_file_workers):
        spectra_queue.put(None)
    lance_writers.close()
    lance_writers.join()

    # Count the total number of spectra in the datasets.
    lance_dir = os.path.join(config.work_dir, "spectra")
    # In auto mode the buckets are not known in advance, so discover the
    # per-charge datasets that the writers created.
    candidate_buckets = (
        _discover_auto_buckets(lance_dir) if auto_bucket else charge_buckets
    )
    n_spectra = 0
    valid_buckets = []
    for bucket in candidate_buckets:
        dataset_path = os.path.join(
            lance_dir, f"spectra_charge{bucket_key_to_str(bucket)}.lance"
        )
        try:
            dataset = lance.dataset(dataset_path)
            n_spectra += dataset.count_rows()
            valid_buckets.append(bucket)
        except (ValueError, FileNotFoundError):
            logger.debug("No spectra for bucket %s, skipping", bucket)
    logger.info(
        "Read %d spectra from %d peak file(s)",
        n_spectra,
        len(input_filenames),
    )
    logger.info("Skipped %d low-quality spectra", low_quality_counter)
    return valid_buckets


def _discover_auto_buckets(
    lance_dir: str,
) -> List[Tuple[Union[int, str], ...]]:
    """
    Discover the per-charge buckets created in auto mode.

    In auto mode every distinct charge is written to its own singleton bucket,
    so the buckets are not known until the spectra have been read. Reconstruct
    them from the ``spectra_charge<key>.lance`` datasets on disk.

    Parameters
    ----------
    lance_dir : str
        The directory containing the per-charge lance datasets.

    Returns
    -------
    List[Tuple[Union[int, str], ...]]
        The discovered singleton charge buckets, sorted for deterministic
        cluster labeling (numeric charges first, then "unknown").
    """
    prefix, suffix = "spectra_charge_", ".lance"
    buckets = []
    for name in os.listdir(lance_dir):
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        key = name[len(prefix) : -len(suffix)]
        bucket = (key,) if key == "unknown" else (int(key),)
        buckets.append(bucket)
    return sorted(buckets, key=lambda b: (isinstance(b[0], str), b[0]))


def _create_lance_dataset(
    charge_bucket: Union[int, Tuple[int, ...]], schema: pa.Schema
) -> lance.LanceDataset:
    """
    Create a lance dataset.

    Parameters
    ----------
    charge_bucket : Union[int, Tuple[int, ...]]
        The key of the charge bucket for which the dataset is created.
    schema : pa.Schema
        The schema of the dataset.

    Returns
    -------
    lance.LanceDataset
        The lance dataset.
    """
    lance_path = os.path.join(
        config.work_dir,
        "spectra",
        f"spectra_charge{bucket_key_to_str(charge_bucket)}.lance",
    )
    # Use mode="create" (not "overwrite"): this is only called when the dataset
    # does not yet exist (guarded by the caller), and "overwrite"/"append" on a
    # missing path makes lance emit a redundant WARN ("No existing dataset ...,
    # it will be created") that duplicates the debug message below.
    dataset = lance.write_dataset(
        pa.Table.from_pylist([], schema),
        lance_path,
        mode="create",
        data_storage_version="stable",
    )
    logger.debug("Creating lance dataset at %s", lance_path)
    return dataset


def _read_spectra(
    filename: str,
    process_spectrum: Callable,
) -> Tuple[List[Dict[str, Union[str, float, int, np.ndarray]]], int]:
    """
    Get the spectra from the given file.

    Parameters
    ----------
    filename : str
        The path of the peak file to be read.
    process_spectrum : Callable
        The function to process the spectra.

    Returns
    -------
    Tuple[List[Dict[str, Union[str, float, int, np.ndarray]]], int]
        The spectra read from the given file as a list of dictionaries and
        the number of low-quality spectra.
    """
    low_quality_counter = 0
    spectra = []
    filename = os.path.abspath(filename)
    for spec in ms_io.get_spectra(filename):
        spec = process_spectrum(spec)
        if spec is None:
            low_quality_counter += 1
        else:
            spectra.append(spec)
    return spectra, low_quality_counter


class _PerChargeLockRegistry:
    """
    Lazily-created per-bucket locks so writers can serialize within a bucket's
    Lance dataset while allowing different buckets to be written concurrently.
    """

    def __init__(self) -> None:
        self._locks: Dict[object, threading.Lock] = {}
        self._guard = threading.Lock()

    def get(self, key: object) -> threading.Lock:
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock


def _write_spectra_lance(
    spectra_queue: queue.Queue,
    lance_locks: "_PerChargeLockRegistry",
    schema: pa.Schema,
    charge_to_bucket: Dict[Union[int, str], Tuple[Union[int, str], ...]],
    catch_other_charges: bool,
    auto_bucket: bool,
) -> None:
    """
    Read spectra from a queue and write to a lance dataset.

    Parameters
    ----------
    spectra_queue : queue.Queue
        Queue from which to read spectra for writing to pickle files.
    lance_locks : _PerChargeLockRegistry
        Per-charge locks to synchronize writes within each dataset.
    schema : pa.Schema
        The schema of the dataset.
    charge_to_bucket : Dict[Union[int, str], Tuple[Union[int, str], ...]]
        Mapping from each precursor charge to its bucket key (the sorted tuple
        of charges sharing that bucket).
    catch_other_charges : bool
        Whether to catch charges not in any bucket.
    auto_bucket : bool
        Whether to cluster every distinct charge separately, assigning each
        spectrum to its own per-charge bucket. Takes precedence over
        ``charge_to_bucket`` and ``catch_other_charges``.
    """
    spec_to_write = collections.defaultdict(list)
    while True:
        spec = spectra_queue.get()
        if spec is None:
            # Write remaining spectra to the dataset.
            for bucket_key in spec_to_write.keys():
                if len(spec_to_write[bucket_key]) == 0:
                    continue
                _write_to_dataset(
                    spec_to_write[bucket_key],
                    bucket_key,
                    lance_locks.get(bucket_key),
                    schema,
                    config.work_dir,
                )
                spec_to_write[bucket_key].clear()
            return
        charge = spec["precursor_charge"]
        charge = "unknown" if charge is None else charge

        # Determine bucket key
        if auto_bucket:
            # Every distinct charge is clustered in its own singleton bucket.
            bucket_key = (charge,)
        else:
            bucket_key = charge_to_bucket.get(charge, None) or (
                "other" if catch_other_charges else None
            )
        if bucket_key is not None:
            spec_to_write[bucket_key].append(spec)

            if len(spec_to_write[bucket_key]) >= _LANCE_WRITE_BATCH_SIZE:
                _write_to_dataset(
                    spec_to_write[bucket_key],
                    bucket_key,
                    lance_locks.get(bucket_key),
                    schema,
                    config.work_dir,
                )
                spec_to_write[bucket_key].clear()


def _write_to_dataset(
    spec_to_write: List[Dict],
    charge_bucket: Union[Tuple[int, ...], str],
    lock: threading.Lock,
    schema: pa.Schema,
    work_dir: str,
) -> int:
    """
    Write a list of spectra to a lance dataset.

    Parameters
    ----------
    spec_to_write : List[Dict]
        The spectra to write.
    charge_bucket : Union[Tuple[int, ...], str]
        The bucket key for the spectra.
    lock : multiprocessing.synchronize.Lock
        Lock to synchronize writing to the dataset.
    schema : pa.Schema
        The schema of the dataset.
    work_dir : str
        The directory in which the dataset is stored.
    Returns
    -------
    int
        The number of spectra written to the dataset.
    """
    # Write the spectra to the dataset.
    new_rows = pa.Table.from_pylist(spec_to_write, schema)
    bucket_str = bucket_key_to_str(charge_bucket)
    path = os.path.join(
        work_dir, "spectra", f"spectra_charge{bucket_str}.lance"
    )
    with lock:
        if not os.path.exists(path):
            _create_lance_dataset(charge_bucket, schema)
        lance.write_dataset(new_rows, path, mode="append")
    return len(new_rows)


def bucket_key_to_str(bucket_key: Union[Tuple[int, ...], str]) -> str:
    """
    Convert a bucket key to a safe string for filenames.

    Examples:
        (1,2) -> _1_2
        (3, "unknown") -> _3_unknown
        "other" -> _other

    Parameters
    ----------
    bucket_key : Union[Tuple[int, ...], str]
        The bucket key.
    Returns
    -------
    str
        The string representation of the bucket key.
    """
    if isinstance(bucket_key, set):
        # Convert set to sorted tuple for consistent ordering
        bucket_key = tuple(
            sorted(bucket_key, key=lambda x: (isinstance(x, str), x))
        )
    if isinstance(bucket_key, tuple):
        parts = [str(p) for p in bucket_key]
        return "_" + "_".join(parts)
    elif isinstance(bucket_key, str):
        return "_" + bucket_key
    else:
        raise TypeError(f"Unsupported bucket key type: {type(bucket_key)}")


def _write_cluster_info(clusters: pd.DataFrame) -> None:
    """
    Export the clustering results to a CSV file.

    Parameters
    ----------
    clusters : pd.DataFrame
        The clustering results.
    """
    with open(f"{config.output_filename}.csv", "a") as f_out:
        # Metadata.
        f_out.write(f"# falcon version {__version__}\n")
        f_out.write(f"# work_dir = {config.work_dir}\n")
        f_out.write(f"# overwrite = {config.overwrite}\n")
        f_out.write(
            f"# export_representatives = " f"{config.export_representatives}\n"
        )
        f_out.write(
            f"# precursor_tol = {config.precursor_tol[0]:.2f} "
            f"{config.precursor_tol[1]}\n"
        )
        f_out.write(f"# rt_tol = {config.rt_tol}\n")
        f_out.write(f"# fragment_tol = {config.fragment_tol:.2f}\n")
        f_out.write(f"# linkage = {config.linkage}\n")
        f_out.write(
            f"# distance_threshold = {config.distance_threshold:.3f}\n"
        )
        f_out.write(f"# min_matched_peaks = {config.min_matched_peaks}\n")
        f_out.write(f"# consensus_method = {config.consensus_method}\n")
        f_out.write(
            f"# outlier_cutoff_lower = {config.outlier_cutoff_lower:.2f}\n"
        )
        f_out.write(
            f"# outlier_cutoff_upper = {config.outlier_cutoff_upper:.2f}\n"
        )
        f_out.write(f"# batch_size = {config.batch_size}\n")
        f_out.write(
            f"# precursor_charge_buckets = "
            f"{config.precursor_charge_buckets}\n"
        )
        f_out.write(f"# min_peaks = {config.min_peaks}\n")
        f_out.write(f"# min_mz_range = {config.min_mz_range:.2f}\n")
        f_out.write(f"# min_mz = {config.min_mz:.2f}\n")
        f_out.write(f"# max_mz = {config.max_mz:.2f}\n")
        f_out.write(
            f"# remove_precursor_tol = " f"{config.remove_precursor_tol:.2f}\n"
        )
        f_out.write(f"# min_intensity = {config.min_intensity:.2f}\n")
        f_out.write(f"# max_peaks_used = {config.max_peaks_used}\n")
        f_out.write(f"# scaling = {config.scaling}\n")
        f_out.write("#\n")
        # Cluster assignments.
        clusters.to_csv(f_out, index=False, chunksize=1000000)


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

    sys.exit(main())
