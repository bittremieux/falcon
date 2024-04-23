import collections
import functools
import itertools
import glob
import logging
import multiprocessing
import os
import pickle
import queue
import shutil
import sys
import tempfile
import threading
from typing import Dict, Iterator, List, Tuple, Union

import joblib
import lance
import natsort
import numba as nb
import numpy as np
import pandas as pd
import pyarrow as pa
import scipy.sparse as ss
from sklearn.random_projection import SparseRandomProjection
from spectrum_utils.spectrum import MsmsSpectrum

from . import __version__
from .cluster import cluster, spectrum
from .config import config
from .ms_io import ms_io


logger = logging.getLogger("falcon")


def main(args: Union[str, List[str]] = None) -> int:
    # Configure logging.
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            "{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : "
            "{message}",
            style="{",
        )
    )
    root.addHandler(handler)
    # Disable dependency non-critical log messages.
    logging.getLogger("faiss").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("numexpr").setLevel(logging.WARNING)

    # Load the configuration.
    config.parse(args)
    logger.info("falcon version %s", str(__version__))
    logger.debug("work_dir = %s", config.work_dir)
    logger.debug("overwrite = %s", config.overwrite)
    logger.debug("export_representatives = %s", config.export_representatives)
    logger.debug("precursor_tol = %.2f %s", *config.precursor_tol)
    logger.debug("rt_tol = %s", config.rt_tol)
    logger.debug("fragment_tol = %.2f", config.fragment_tol)
    logger.debug("eps = %.3f", config.eps)
    logger.debug("mz_interval = %d", config.mz_interval)
    logger.debug("low_dim = %d", config.low_dim)
    logger.debug("n_neighbors = %d", config.n_neighbors)
    logger.debug("n_neighbors_ann = %d", config.n_neighbors_ann)
    logger.debug("batch_size = %d", config.batch_size)
    logger.debug("n_probe = %d", config.n_probe)
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
        logging.warning(
            "Working directory %s already exists, previous "
            "results might get overwritten",
            config.work_dir,
        )
    os.makedirs(config.work_dir, exist_ok=True)
    os.makedirs(os.path.join(config.work_dir, "spectra"), exist_ok=True)
    os.makedirs(os.path.join(config.work_dir, "nn"), exist_ok=True)

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
    if config.overwrite:
        if os.path.isfile(os.path.join(config.work_dir, "spec_data.lance")):
            os.remove(os.path.join(config.work_dir, "spec_data.lance"))
        for filename in os.listdir(os.path.join(config.work_dir, "spectra")):
            os.remove(os.path.join(config.work_dir, "spectra", filename))
        for filename in os.listdir(os.path.join(config.work_dir, "nn")):
            os.remove(os.path.join(config.work_dir, "nn", filename))

    # Read the spectra from the input files and partition them based on their
    # precursor m/z.
    if not any(
        [
            filename.endswith(".pkl")
            for filename in os.listdir(
                os.path.join(config.work_dir, "spectra")
            )
        ]
    ):
        buckets = _prepare_spectra()
        joblib.dump(
            buckets, os.path.join(config.work_dir, "spectra", "buckets.joblib")
        )
    else:
        buckets = joblib.load(
            os.path.join(config.work_dir, "spectra", "buckets.joblib")
        )

    vec_len, min_mz, max_mz = spectrum.get_dim(
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

    transformation = (
        SparseRandomProjection(config.low_dim, random_state=0)
        .fit(np.zeros((1, vec_len)))
        .components_.astype(np.float32)
        .T
    )
    vectorize = functools.partial(
        spectrum.to_vector,
        transformation=transformation,
        min_mz=min_mz,
        bin_size=config.fragment_tol,
        dim=vec_len,
        norm=True,
    )

    # Cluster the spectra per precursor mass bucket.
    clusters_all, current_label, representatives = [], 0, []
    dataset = lance.dataset(os.path.join(config.work_dir, "spec_data.lance"))
    dataset.create_scalar_index(
        column="precursor_mz", index_type="BTREE", replace=True
    )
    bucket_id = 0
    for charge, bucket_limits in buckets.items():
        for min_mz, max_mz in bucket_limits:
            query = f"precursor_mz >= {min_mz} AND precursor_mz <= {max_mz}"
            spectra = dataset.scanner(filter=query).to_table().to_pandas()
            n_spectra = len(spectra)
            logger.info(
                f"Cluster {n_spectra} spectra with precursor charge {charge} in mass interval {min_mz} - {max_mz}"
            )
            dist_filename = os.path.join(
                config.work_dir, "nn", f"dist_{bucket_id}.npz"
            )
            metadata_filename = os.path.join(
                config.work_dir, "nn", f"metadata_{bucket_id}.parquet"
            )
            if not os.path.isfile(dist_filename) or not os.path.isfile(
                metadata_filename
            ):
                pairwise_dist_matrix, metadata = (
                    cluster.compute_pairwise_distances(
                        n_spectra,
                        spectra,
                        bucket_id,
                        process_spectrum,
                        vectorize,
                        config.precursor_tol[0],
                        config.precursor_tol[1],
                        config.rt_tol,
                        config.n_neighbors,
                        config.n_neighbors_ann,
                        config.batch_size,
                        config.n_probe,
                    )
                )
                if metadata is None:
                    continue
                metadata.insert(2, "precursor_charge", charge)
                logger.debug(
                    "Export pairwise distance matrix to file %s",
                    dist_filename,
                )
                ss.save_npz(dist_filename, pairwise_dist_matrix, False)
                metadata.to_parquet(metadata_filename, index=False)
            else:
                logger.debug(
                    "Load previously computed pairwise distance matrix "
                    "from file %s",
                    dist_filename,
                )
                pairwise_dist_matrix = ss.load_npz(dist_filename)
                metadata = pd.read_parquet(metadata_filename)
            # No valid spectra found with the current charge.
            if len(metadata) == 0:
                continue
            # Cluster using the pairwise distance matrix.
            clusters = cluster.generate_clusters(
                pairwise_dist_matrix,
                config.eps,
                metadata["precursor_mz"].values,
                metadata["retention_time"].values,
                config.precursor_tol[0],
                config.precursor_tol[1],
                config.rt_tol,
            )
            # Make sure that different charges have non-overlapping cluster labels.
            clusters += current_label
            # noinspection PyUnresolvedReferences
            current_label = np.amax(clusters) + 1
            # Save cluster assignments.
            metadata["cluster"] = clusters
            clusters_all.append(metadata)
            # Extract identifiers for cluster representatives (medoids).
            if config.export_representatives:
                charge_representatives = cluster.get_cluster_representatives(
                    clusters,
                    pairwise_dist_matrix.indptr,
                    pairwise_dist_matrix.indices,
                    pairwise_dist_matrix.data,
                )
                representatives.append(spectra.iloc[charge_representatives])
            bucket_id += 1

    # Export cluster memberships and representative spectra.
    clusters_all = pd.concat(clusters_all, ignore_index=True).sort_values(
        ["filename", "spectrum_id"], key=natsort.natsort_keygen()
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
        representatives = pd.concat(representatives, ignore_index=True)
        logger.info(
            "Export %d cluster representative spectra to output file " "%s",
            len(representatives),
            f"{config.output_filename}.mgf",
        )
        representatives = representatives.apply(
            cluster.df_row_to_spectrum, axis=1
        ).tolist()
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


def _prepare_spectra() -> Dict[int, Dict[int, List[Tuple[str, str]]]]:
    """
    Read the spectra from the input peak files and partition to intermediate
    files split and sorted by precursor m/z.

    Returns
    -------
    List[Tuple[int, str]]
        The number of spectra per precursor m/z bucket and the filenames of the
        intermediate files.
    """
    input_filenames = [
        fn for pattern in config.input_filenames for fn in glob.glob(pattern)
    ]
    logger.info("Read spectra from %d peak file(s)", len(input_filenames))
    # Use multiple worker processes to read the peak files.
    max_file_workers = min(len(input_filenames), multiprocessing.cpu_count())
    file_queue = queue.Queue()
    # Restrict the number of spectra simultaneously in memory to avoid
    # excessive memory requirements.
    max_spectra_in_memory = 1_000_000
    spectra_queue = queue.Queue(maxsize=max_spectra_in_memory)
    # Include sentinels at the end to stop the worker file readers.
    for filename in itertools.chain(
        input_filenames, itertools.repeat(None, max_file_workers)
    ):
        file_queue.put(filename)
    # Read the peak files and put their spectra in the queue for consumption.
    precursor_mz = collections.defaultdict(list)
    peak_readers = multiprocessing.pool.ThreadPool(
        max_file_workers,
        _read_spectra_queue,
        (file_queue, spectra_queue, precursor_mz),
    )
    # create lance dataset
    schema = pa.schema(
        [
            pa.field("identifier", pa.string()),
            pa.field("precursor_mz", pa.float32()),
            pa.field("precursor_charge", pa.int32()),
            pa.field("mz", pa.list_(pa.float32())),
            pa.field("intensity", pa.list_(pa.float32())),
            pa.field("retention_time", pa.float32()),
            pa.field("filename", pa.string()),
        ]
    )
    lance_path = os.path.join(config.work_dir, "spec_data.lance")
    lance.write_dataset(
        pa.Table.from_pylist([], schema),
        lance_path,
        mode="overwrite",
    )
    logger.debug("Created lance dataset at %s", lance_path)
    lance_lock = multiprocessing.Lock()
    # Write the spectra to a lance file.
    pkl_writers = multiprocessing.pool.ThreadPool(
        max_file_workers,
        _write_spectra_lance,
        (spectra_queue, lance_lock, schema),
    )
    peak_readers.close()
    peak_readers.join()
    # Add sentinels to indicate stopping. This needs to happen after all files
    # have been read (by joining `peak_readers`).
    for _ in range(max_file_workers):
        spectra_queue.put(None)
    pkl_writers.close()
    pkl_writers.join()
    # get dataset length
    dataset = lance.dataset(lance_path)
    n_spectra = dataset.count_rows()
    logger.debug(
        "Read %d spectra from %d peak files", n_spectra, len(input_filenames)
    )
    # Group spectra with near-identical precursor m/z values in the same mass
    # bucket before clustering.
    buckets = spectra_to_bucket(
        precursor_mz, config.precursor_tol[0], config.precursor_tol[1]
    )
    return buckets


def _read_spectra_queue(
    file_queue: queue.Queue,
    spectra_queue: queue.Queue,
    precursor_mz: Dict[int, List[float]],
) -> None:
    """
    Get the spectra from the file queue and store them in the spectra queue.

    Parameters
    ----------
    file_queue : queue.Queue
        Queue from which the file names are retrieved.
    spectra_queue : queue.Queue
        Queue in which spectra are stored.
    precursor_mz : Dict[int, List[float]]
        List in which precursor m/z values are stored by charge.
    """
    while True:
        filename = file_queue.get()
        if filename is None:
            return
        for spec in _read_spectra(
            filename,
            precursor_mz,
        ):
            spectra_queue.put(spec)


def _read_spectra(
    filename: str,
    precursor_mz: Dict[int, List[float]],
) -> Iterator[Tuple[MsmsSpectrum, str]]:
    """
    Get the spectra from the given file.

    Parameters
    ----------
    filename : str
        The path of the peak file to be read.
    precursor_mz : Dict[int, List[float]]
        Dictionary that will be filled with precursor m/z values per charge.
    mass_bucket_width : float
        The size of the mass bucket.
    bucket_tolerance : float
        The tolerance for handling border cases during bucketing.
    work_dir : str
        The directory in which the spectrum buckets will be stored.

    Returns
    -------
    Iterator[Tuple[MsmsSpectrum, str]]
        The spectra read from the given file and their bucket filenames
        (based on precursor charge and m/z).
    """
    filename = os.path.abspath(filename)
    for spec in ms_io.get_spectra(filename):
        spec.filename = filename
        precursor_mz[spec.precursor_charge].append(spec.precursor_mz)
        yield _spec_to_dict(spec)


def _spec_to_dict(spectrum: MsmsSpectrum) -> Dict[str, Union[str, int, float]]:
    """
    Convert a MsmsSpectrum object to a dictionary.

    Parameters
    ----------
    spectrum : MsmsSpectrum
        The spectrum to convert.

    Returns
    -------
    Dict[str, Union[str, int, float]]
        The spectrum as a dictionary.
    """
    return {
        "identifier": spectrum.identifier,
        "precursor_mz": spectrum.precursor_mz,
        "precursor_charge": spectrum.precursor_charge,
        "mz": spectrum.mz,
        "intensity": spectrum.intensity,
        "retention_time": spectrum.retention_time,
        "filename": spectrum.filename,
    }


@nb.njit(cache=True)
def _precursor_to_interval(
    mz: float,
    charge: int,
    interval_width: float,
    bucket_tolerance: float,
) -> int:
    """
    Convert the precursor m/z to the neutral mass and get the interval index.

    Parameters
    ----------
    mz : float
        The precursor m/z.
    charge : int
        The precursor charge.
    interval_width : float
        The width of each m/z interval.
    bucket_tolerance : float
        The tolerance for handling border cases during bucketing.

    Returns
    -------
    int
        The index of the interval to which a spectrum with the given m/z
        belongs.
    """
    hydrogen_mass = 1.00794
    neutral_mass = (mz - hydrogen_mass) * max(abs(charge), 1)
    interval = int(neutral_mass // interval_width)
    remainder = neutral_mass % interval_width
    if remainder >= interval_width - bucket_tolerance:
        interval += 1
    return interval


def _write_spectra_lance(
    spectra_queue: queue.Queue,
    lance_lock: multiprocessing.Lock,
    schema: pa.Schema,
) -> None:
    """
    Read spectra from a queue and write to a lance dataset.

    Parameters
    ----------
    spectra_queue : queue.Queue
        Queue from which to read spectra for writing to pickle files.
    lance_lock : multiprocessing.Lock
        Lock
    schema : pa.Schema
        The schema of the dataset.
    """
    spec_to_write = []
    while True:
        spec = spectra_queue.get()
        if spec is None:
            # flush remaining spectra
            if len(spec_to_write) > 0:
                _write_to_dataset(
                    spec_to_write, lance_lock, schema, config.work_dir
                )
            return
        spec_to_write.append(spec)
        if len(spec_to_write) >= 10_000:
            _write_to_dataset(
                spec_to_write, lance_lock, schema, config.work_dir
            )
            spec_to_write.clear()


def _write_to_dataset(
    spec_to_write: List[Dict[str, Union[str, int, float]]],
    lock: multiprocessing.Lock,
    schema: pa.Schema,
    work_dir: str,
):
    """
    Write a list of spectra to a lance dataset.

    Parameters
    ----------
    spec_to_write : List[Dict[str, Union[str, int, float]]]
        The spectra to write.
    lock : multiprocessing.Lock
        Lock
    schema : pa.Schema
        The schema of the dataset.
    work_dir : str
        The directory in which the dataset is stored.

    Returns
    -------
    int
        The number of spectra written to the dataset.
    """
    new_rows = pa.Table.from_pylist(spec_to_write, schema)
    path = os.path.join(work_dir, "spec_data.lance")
    with lock:
        lance.write_dataset(new_rows, path, mode="append")
    return len(new_rows)


def spectra_to_bucket(
    precursor_mz: Dict[int, List[float]],
    tol: float,
    tol_mode: str,
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Group spectra with near-identical precursor m/z values in the same mass
    bucket.

    Parameters
    ----------
    precursor_mz : Dict[int, List[float]]
        The precursor m/z values per charge.
    tol : float
        The precursor m/z tolerance.
    tol_mode : str
        The tolerance mode, either "Da" or "ppm".

    Returns
    -------
    Dict[int, List[Tuple[str, str]]]
        Bucket limits (min_mz, max_mz) per charge.
    """
    sorted_precursor_mz = {
        charge: sorted(mz_values) for charge, mz_values in precursor_mz.items()
    }
    bucket_limits = _get_bucket_limits(sorted_precursor_mz, tol, tol_mode)

    return bucket_limits


def _get_bucket_limits(
    precursor_mz: Dict[int, List[float]],
    tol: float,
    tol_mode: str,
) -> Dict[int, List[Tuple[float, float]]]:
    """
    Create mass buckets given all precursor m/z's and precursor m/z tolerance.

    Parameters
    ----------
    precursor_mz : Dict[int, List[float]]
        The precursor m/z values per charge.
    tol : float
        The precursor m/z tolerance.
    tol_mode : str
        The tolerance mode, either "Da" or "ppm".

    Returns
    -------
    Dict[int, List[Tuple[float, float]]]
        The mass bucket boundaries (min_mz, max_mz) per charge.
    """
    min_max_mz = collections.defaultdict(list)
    for charge, mz_values in precursor_mz.items():
        cluster_count = 0
        i = 0
        while i < len(mz_values):
            cluster_min = mz_values[i]
            if tol_mode == "Da":
                while (
                    i < len(mz_values) - 1
                    and mz_values[i + 1] - mz_values[i] < tol
                ):
                    i += 1
            elif tol_mode == "ppm":
                while (
                    i < len(mz_values) - 1
                    and (mz_values[i + 1] - mz_values[i])
                    / mz_values[i]
                    * 10**6
                    < tol
                ):
                    i += 1
            cluster_max = mz_values[i]
            min_max_mz[charge].append((cluster_min, cluster_max))
            cluster_count += 1
            i += 1
    return min_max_mz


def _sort_spectra_pkl(filename: str) -> int:
    """
    Sort spectra in a pickle file by their precursor m/z.

    Parameters
    ----------
    filename : str
        The pickled spectrum file name.

    Returns
    -------
    int
        The number of spectra in the file.
    """
    with open(filename, "rb+") as f_pkl:
        # Read all the spectra from the pickle file.
        spectra = []
        while True:
            try:
                spectra.append(pickle.load(f_pkl))
            except EOFError:
                break
        # Write the sorted spectra to the pickle file.
        order = _get_precursor_order(
            np.asarray([spec.precursor_mz for spec in spectra])
        )
        f_pkl.seek(0)
        pickle.dump(
            [spectra[i] for i in order],
            f_pkl,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        return len(spectra)


@nb.njit(cache=True)
def _get_precursor_order(precursor_mzs: np.ndarray) -> np.ndarray:
    """
    Get the precursor m/z sorting order.

    This should use radix sort on integers with O(n) time complexity
    internally.

    Parameters
    ----------
    precursor_mzs : np.ndarray
        The precursor m/z's for which the sorting order is to be determined.

    Returns
    -------
    np.ndarray
        The order to sort the given precursor m/z's.
    """
    # noinspection PyUnresolvedReferences
    return np.argsort(
        np.asarray(precursor_mzs * 10000).astype(np.int32), kind="mergesort"
    )


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
        f_out.write(f"# eps = {config.eps:.3f}\n")
        f_out.write(f"# mz_interval = {config.mz_interval}\n")
        f_out.write(f"# low_dim = {config.low_dim}\n")
        f_out.write(f"# n_neighbors = {config.n_neighbors}\n")
        f_out.write(f"# n_neighbors_ann = {config.n_neighbors_ann}\n")
        f_out.write(f"# batch_size = {config.batch_size}\n")
        f_out.write(f"# n_probe = {config.n_probe}\n")
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


def _find_spectra_pkl(
    filename: str, spectra_to_read: Dict[Tuple[str, str], int]
) -> List[MsmsSpectrum]:
    """
    Read spectra with the specified USIs from a pkl file.

    Parameters
    ----------
    filename : str
        Name of the pkl file from which the spectra are read.
    spectra_to_read : Dict[Tuple[str, str], int]
        Dict with as keys tuples of the filename and identifier of the spectra
        to read from the pkl file, and as values the cluster labels that will
        be assigned to the spectra.

    Returns
    -------
    Iterable[MsmsSpectrum]
        The spectra with the specified USIs.
    """
    with open(filename, "rb") as f_in:
        spectra = pickle.load(f_in)
        spectra_key_to_index = {
            (spec.filename, spec.identifier): i
            for i, spec in enumerate(spectra)
        }
        spectra_index_to_cluster = {
            spectra_key_to_index[key]: spectra_to_read[key]
            for key in (
                set(spectra_key_to_index.keys()) & set(spectra_to_read.keys())
            )
        }
        if len(spectra_index_to_cluster) != len(spectra_to_read):
            raise ValueError(
                f"{len(spectra_to_read) - len(spectra_index_to_cluster)} "
                f"spectra not found in file {filename}"
            )
        else:
            spectra_found = []
            for i, cluster_ in spectra_index_to_cluster.items():
                spec = spectra[i]
                spec.cluster = cluster_
                spectra_found.append(spec)
            return spectra_found


if __name__ == "__main__":
    sys.exit(main())
