import functools
import logging
import multiprocessing
import os
import shutil
import sys
import tempfile
import threading
from typing import Callable, List, Set, Tuple, Union

import joblib
import lance
import natsort
import numpy as np
import pandas as pd

from . import __version__, utils
from .spectra_io import (
    _prepare_spectra,
    _write_cluster_info,
    bucket_key_to_str,
)
from .cluster import cluster, spectrum
from .config import config
from .ms_io import ms_io

logger = logging.getLogger("falcon")

utils.set_seeds()


def main(args: Union[str, List[str], None] = None) -> int:
    utils.configure_logger()
    config.parse(args)
    _log_config()

    rm_work_dir = _setup_work_dir()
    # Abort if outputs already exist and --overwrite was not given.
    if _validate_outputs():
        logging.shutdown()
        return 1
    _warn_aggressive_outlier_cutoffs()

    process_spectrum = _build_process_spectrum()
    charge_buckets = _load_or_prepare_buckets(process_spectrum)
    bucket_datasets, cluster_results = _run_clustering(charge_buckets)
    _export_results(bucket_datasets, cluster_results)

    if rm_work_dir:
        shutil.rmtree(config.work_dir)
    logging.shutdown()
    return 0


# Config attributes dumped at debug level on startup. `precursor_tol` is logged
# separately because it is a (value, unit) pair.
_CONFIG_LOG_FIELDS = [
    "work_dir",
    "overwrite",
    "export_representatives",
    "rt_tol",
    "fragment_tol",
    "linkage",
    "distance_threshold",
    "min_matched_peaks",
    "consensus_method",
    "outlier_cutoff_lower",
    "outlier_cutoff_upper",
    "batch_size",
    "precursor_charge_buckets",
    "min_peaks",
    "min_mz_range",
    "min_mz",
    "max_mz",
    "remove_precursor_tol",
    "min_intensity",
    "max_peaks_used",
    "scaling",
]


def _log_config() -> None:
    """Log the falcon version and the active configuration (debug level)."""
    logger.info("falcon version %s", str(__version__))
    logger.debug("precursor_tol = %.2f %s", *config.precursor_tol)
    for field in _CONFIG_LOG_FIELDS:
        logger.debug("%s = %s", field, getattr(config, field))


def _setup_work_dir() -> bool:
    """
    Create the working directory and its ``spectra`` subdirectory.

    Returns
    -------
    bool
        Whether the working directory was created as a temporary directory and
        should be removed on exit.
    """
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
    return rm_work_dir


def _check_existing_output(suffix: str, description: str) -> bool:
    """
    Handle a pre-existing output file: remove it under ``--overwrite``, else
    log an error.

    Returns
    -------
    bool
        Whether the file exists and must not be overwritten (the run should
        abort).
    """
    path = f"{config.output_filename}.{suffix}"
    if not os.path.isfile(path):
        return False
    if config.overwrite:
        logger.warning(
            "Output file %s (%s) already exists and will be overwritten",
            path,
            description,
        )
        os.remove(path)
        return False
    logger.error(
        "Output file %s (%s) already exists, aborting...", path, description
    )
    return True


def _validate_outputs() -> bool:
    """
    Check both output files for pre-existing results.

    Returns
    -------
    bool
        Whether the run should abort (an output exists without --overwrite).
    """
    # Evaluate both (no short-circuit) so the .mgf file is still removed under
    # --overwrite even when the .csv check already decided to abort.
    csv_abort = _check_existing_output("csv", "cluster assignments")
    mgf_abort = _check_existing_output("mgf", "cluster representatives")
    return csv_abort or mgf_abort


def _warn_aggressive_outlier_cutoffs() -> None:
    """Warn when both averaging outlier cutoffs are set below 1."""
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


def _build_process_spectrum() -> Callable:
    """Build the configured single-spectrum preprocessing function."""
    _, min_mz, max_mz = spectrum.get_dim(
        config.min_mz, config.max_mz, config.fragment_tol
    )
    return functools.partial(
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


def _load_or_prepare_buckets(
    process_spectrum: Callable,
) -> List[Union[Set[Union[int, str]], str]]:
    """
    Return the charge buckets, reusing cached spectra when possible.

    Under ``--overwrite`` the intermediate ``spectra`` directory is cleared and
    the spectra are re-read and re-partitioned; otherwise a cached partitioning
    is reused after validating it against the current
    ``--precursor_charge_buckets`` configuration.
    """
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
        return cached["buckets"]

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
    return charge_buckets


def _run_clustering(
    charge_buckets: List[Union[Set[Union[int, str]], str]],
) -> Tuple[list, list]:
    """
    Cluster every non-empty charge bucket in a single shared worker pool.

    All non-empty charge buckets are clustered together (rather than one pool
    per bucket) so cores stay busy across bucket boundaries instead of idling at
    each bucket's end-of-pool barrier; the resulting partitions are identical to
    clustering each bucket on its own.

    Returns
    -------
    Tuple[list, list]
        The per-bucket lance datasets (non-empty only) and the matching
        clustering results, aligned positionally.
    """
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
    return bucket_datasets, cluster_results


def _export_results(bucket_datasets: list, cluster_results: list) -> None:
    """
    Assemble cluster assignments and representatives and write the outputs.

    Cluster labels from different charge buckets are offset so they do not
    overlap, the per-bucket assignments are concatenated, and the ``.csv`` (and
    optionally ``.mgf``) outputs are written on background IO threads.
    """
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



if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

    sys.exit(main())
