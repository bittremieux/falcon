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
from typing import BinaryIO, Dict, Iterator, List, Tuple, Union

import joblib
import natsort
import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse as ss
from sklearn.random_projection import SparseRandomProjection
from spectrum_utils.spectrum import MsmsSpectrum

from . import __version__
from .cluster import cluster, spectrum
from .config import config
from .ms_io import ms_io


logger = logging.getLogger('falcon')


def main(args: Union[str, List[str]] = None) -> int:
    # Configure logging.
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        '{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : '
        '{message}', style='{'))
    root.addHandler(handler)
    # Disable dependency non-critical log messages.
    logging.getLogger('faiss').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('numexpr').setLevel(logging.WARNING)

    # Load the configuration.
    config.parse(args)
    logger.info('falcon version %s', str(__version__))
    logger.debug('work_dir = %s', config.work_dir)
    logger.debug('overwrite = %s', config.overwrite)
    logger.debug('export_representatives = %s', config.export_representatives)
    logger.debug('precursor_tol = %.2f %s', *config.precursor_tol)
    logger.debug('rt_tol = %s', config.rt_tol)
    logger.debug('fragment_tol = %.2f', config.fragment_tol)
    logger.debug('eps = %.3f', config.eps)
    logger.debug('mz_interval = %d', config.mz_interval)
    logger.debug('low_dim = %d', config.low_dim)
    logger.debug('n_neighbors = %d', config.n_neighbors)
    logger.debug('n_neighbors_ann = %d', config.n_neighbors_ann)
    logger.debug('batch_size = %d', config.batch_size)
    logger.debug('n_probe = %d', config.n_probe)
    logger.debug('min_peaks = %d', config.min_peaks)
    logger.debug('min_mz_range = %.2f', config.min_mz_range)
    logger.debug('min_mz = %.2f', config.min_mz)
    logger.debug('max_mz = %.2f', config.max_mz)
    logger.debug('remove_precursor_tol = %.2f', config.remove_precursor_tol)
    logger.debug('min_intensity = %.2f', config.min_intensity)
    logger.debug('max_peaks_used = %d', config.max_peaks_used)
    logger.debug('scaling = %s', config.scaling)

    rm_work_dir = False
    if config.work_dir is None:
        config.work_dir = tempfile.mkdtemp()
        rm_work_dir = True
    elif os.path.isdir(config.work_dir):
        logging.warning('Working directory %s already exists, previous '
                        'results might get overwritten', config.work_dir)
    os.makedirs(config.work_dir, exist_ok=True)
    os.makedirs(os.path.join(config.work_dir, 'spectra'), exist_ok=True)
    os.makedirs(os.path.join(config.work_dir, 'nn'), exist_ok=True)

    # Clean all intermediate and final results if "overwrite" is specified,
    # otherwise abort if the output files already exist.
    exit_exists = False
    if os.path.isfile(f'{config.output_filename}.csv'):
        if config.overwrite:
            logger.warning('Output file %s (cluster assignments) already '
                           'exists and will be overwritten',
                           f'{config.output_filename}.csv')
            os.remove(f'{config.output_filename}.csv')
        else:
            logger.error('Output file %s (cluster assignments) already '
                         'exists, aborting...',
                         f'{config.output_filename}.csv')
            exit_exists = True
    if os.path.isfile(f'{config.output_filename}.mgf'):
        if config.overwrite:
            logger.warning('Output file %s (cluster representatives) already '
                           'exists and will be overwritten',
                           f'{config.output_filename}.mgf')
            os.remove(f'{config.output_filename}.mgf')
        else:
            logger.error('Output file %s (cluster representatives) already '
                         'exists, aborting...',
                         f'{config.output_filename}.mgf')
            exit_exists = True
    if exit_exists:
        logging.shutdown()
        return 1
    if config.overwrite:
        for filename in os.listdir(os.path.join(config.work_dir, 'spectra')):
            os.remove(os.path.join(config.work_dir, 'spectra', filename))
        for filename in os.listdir(os.path.join(config.work_dir, 'nn')):
            os.remove(os.path.join(config.work_dir, 'nn', filename))

    # Read the spectra from the input files and partition them based on their
    # precursor m/z.
    if not any([filename.endswith('.pkl') for filename in os.listdir(
            os.path.join(config.work_dir, 'spectra'))]):
        buckets = _prepare_spectra()
        joblib.dump(buckets, os.path.join(config.work_dir, 'spectra',
                                          'buckets.joblib'))
    else:
        buckets = joblib.load(os.path.join(config.work_dir, 'spectra',
                                           'buckets.joblib'))

    vec_len, min_mz, max_mz = spectrum.get_dim(config.min_mz, config.max_mz,
                                               config.fragment_tol)
    process_spectrum = functools.partial(
        spectrum.process_spectrum,
        min_peaks=config.min_peaks,
        min_mz_range=config.min_mz_range,
        mz_min=min_mz,
        mz_max=max_mz,
        remove_precursor_tolerance=config.remove_precursor_tol,
        min_intensity=config.min_intensity,
        max_peaks_used=config.max_peaks_used,
        scaling=None if config.scaling == 'off' else config.scaling)

    transformation = (SparseRandomProjection(config.low_dim, random_state=0)
                      .fit(np.zeros((1, vec_len)))
                      .components_.astype(np.float32).T)
    vectorize = functools.partial(
        spectrum.to_vector, transformation=transformation, min_mz=min_mz,
        bin_size=config.fragment_tol, dim=vec_len, norm=True)

    # Cluster the spectra per charge.
    clusters_all, current_label, representative_info = [], 0, []
    for charge, (n_spectra, bucket_filenames) in buckets.items():
        logger.info('Cluster %d spectra with precursor charge %d',
                    n_spectra, charge)
        dist_filename = os.path.join(
            config.work_dir, 'nn', f'dist_{charge}.npz')
        metadata_filename = os.path.join(
            config.work_dir, 'nn', f'metadata_{charge}.parquet')
        if (not os.path.isfile(dist_filename)
                or not os.path.isfile(metadata_filename)):
            pairwise_dist_matrix, metadata = \
                cluster.compute_pairwise_distances(
                    n_spectra, bucket_filenames, process_spectrum, vectorize,
                    config.precursor_tol[0], config.precursor_tol[1],
                    config.rt_tol, config.n_neighbors, config.n_neighbors_ann,
                    config.batch_size, config.n_probe)
            metadata.insert(2, 'precursor_charge', charge)
            logger.debug('Export pairwise distance matrix to file %s',
                         dist_filename)
            ss.save_npz(dist_filename, pairwise_dist_matrix, False)
            metadata.to_parquet(metadata_filename, index=False)
        else:
            logger.debug('Load previously computed pairwise distance matrix '
                         'from file %s', dist_filename)
            pairwise_dist_matrix = ss.load_npz(dist_filename)
            metadata = pd.read_parquet(metadata_filename)
        # No valid spectra found with the current charge.
        if len(metadata) == 0:
            continue
        # Cluster using the pairwise distance matrix.
        clusters = cluster.generate_clusters(
            pairwise_dist_matrix, config.eps, metadata['precursor_mz'].values,
            metadata['retention_time'].values, config.precursor_tol[0],
            config.precursor_tol[1], config.rt_tol)
        # Make sure that different charges have non-overlapping cluster labels.
        clusters += current_label
        # noinspection PyUnresolvedReferences
        current_label = np.amax(clusters) + 1
        # Save cluster assignments.
        metadata['cluster'] = clusters
        clusters_all.append(metadata)
        # Extract identifiers for cluster representatives (medoids).
        if config.export_representatives:
            charge_representatives = cluster.get_cluster_representatives(
                clusters, pairwise_dist_matrix.indptr,
                pairwise_dist_matrix.indices, pairwise_dist_matrix.data)
            representative_info.append(metadata.iloc[charge_representatives])

    # Export cluster memberships and representative spectra.
    clusters_all = (pd.concat(clusters_all, ignore_index=True)
                    .sort_values(['filename', 'spectrum_id'],
                                 key=natsort.natsort_keygen()))
    logger.info('Export cluster assignments of %d spectra to %d unique '
                'clusters to output file %s',
                len(clusters_all), clusters_all['cluster'].nunique(),
                f'{config.output_filename}.csv')
    # Perform IO in a separate worker process.
    write_csv_worker = threading.Thread(
        target=_write_cluster_info, args=(clusters_all,), daemon=True)
    write_csv_worker.start()
    if config.export_representatives:
        representative_info = pd.concat(representative_info, ignore_index=True)
        logger.info('Export %d cluster representative spectra to output file '
                    '%s', len(representative_info),
                    f'{config.output_filename}.mgf')
        # Get the spectra corresponding to the cluster representatives.
        representative_info['pkl'] = representative_info.apply(
            lambda row: os.path.join(
                config.work_dir, 'spectra',
                f"""{row.precursor_charge}_{_precursor_to_interval(
                    row.precursor_mz, row.precursor_charge,
                    config.mz_interval)}.pkl"""),
            axis='columns')
        representatives = []
        for spectra in joblib.Parallel(n_jobs=-1, prefer='threads')(
                joblib.delayed(_find_spectra_pkl)(
                    fn, (spectra.set_index(['filename', 'spectrum_id'])
                         ['cluster'].to_dict()))
                for fn, spectra in representative_info.groupby('pkl')):
            representatives.extend(spectra)
        representatives.sort(key=lambda spec: spec.cluster)
        # Perform IO in a separate worker process.
        write_mgf_worker = threading.Thread(
            target=ms_io.write_spectra,
            args=(f'{config.output_filename}.mgf', representatives),
            daemon=True)
        write_mgf_worker.start()
        write_mgf_worker.join()
    write_csv_worker.join()

    if rm_work_dir:
        shutil.rmtree(config.work_dir)

    logging.shutdown()
    return 0


def _prepare_spectra() -> Dict[int, Tuple[int, List[str]]]:
    """
    Read the spectra from the input peak files and partition to intermediate
    files split and sorted by precursor m/z.

    Returns
    -------
    Dict[int, Tuple[int, List[str]]]
        A dictionary with per precursor charge the total number of spectra and
        a list of file names of the spectrum buckets.
    """
    input_filenames = [fn for pattern in config.input_filenames
                       for fn in glob.glob(pattern)]
    logger.info('Read spectra from %d peak file(s)', len(input_filenames))
    # Use multiple worker processes to read the peak files.
    max_file_workers = min(len(input_filenames), multiprocessing.cpu_count())
    file_queue = queue.Queue()
    # Restrict the number of spectra simultaneously in memory to avoid
    # excessive memory requirements.
    max_spectra_in_memory = 1_000_000
    spectra_queue = queue.Queue(maxsize=max_spectra_in_memory)
    # Include sentinels at the end to stop the worker file readers.
    for filename in itertools.chain(
            input_filenames, itertools.repeat(None, max_file_workers)):
        file_queue.put(filename)
    # Read the peak files and put their spectra in the queue for consumption.
    peak_readers = multiprocessing.pool.ThreadPool(
        max_file_workers, _read_spectra_queue, (file_queue, spectra_queue))
    # Write the spectra to (unsorted) pickle files.
    pkl_filehandles = {}
    lock = multiprocessing.Lock()
    pkl_writers = multiprocessing.pool.ThreadPool(
        max_file_workers, _write_spectra_pkl,
        (pkl_filehandles, spectra_queue, lock))
    peak_readers.close()
    peak_readers.join()
    # Add sentinels to indicate stopping. This needs to happen after all files
    # have been read (by joining `peak_readers`).
    for _ in range(max_file_workers):
        spectra_queue.put((None, None))
    pkl_writers.close()
    pkl_writers.join()
    # Make sure the spectra in the individual files are sorted by their
    # precursor m/z and count the number of spectra per precursor charge.
    buckets: Dict = collections.defaultdict(lambda: [0, []])
    for filename, n_spectra_file in zip(
            pkl_filehandles.keys(),
            joblib.Parallel(n_jobs=-1, backend='threading')(
                joblib.delayed(_sort_spectra_pkl)(filehandle)
                for filehandle in pkl_filehandles.values())):
        charge = os.path.basename(filename)
        charge = int(charge[:charge.index('_')])
        buckets[charge][0] += n_spectra_file
        buckets[charge][1].append(filename)
    for filehandle in pkl_filehandles.values():
        filehandle.close()
    n_spectra, n_buckets = 0, 0
    for charge, (n_spectra_charge, filenames) in buckets.items():
        n_spectra += n_spectra_charge
        n_buckets += len(filenames)
        buckets[charge] = n_spectra_charge, natsort.natsorted(filenames)
    logger.debug('%d spectra written to %d buckets by precursor charge and '
                 'precursor m/z', n_spectra, n_buckets)
    return dict(buckets)


def _read_spectra_queue(file_queue: queue.Queue, spectra_queue: queue.Queue) \
        -> None:
    """
    Get the spectra from the file queue and store them in the spectra queue.

    Parameters
    ----------
    file_queue : queue.Queue
        Queue from which the file names are retrieved.
    spectra_queue : queue.Queue
        Queue in which spectra are stored.
    """
    while True:
        filename = file_queue.get()
        if filename is None:
            return
        for spec in _read_spectra(filename, config.mz_interval,
                                  config.work_dir):
            spectra_queue.put(spec)


def _read_spectra(filename: str, mz_interval: int, work_dir: str) \
        -> Iterator[Tuple[MsmsSpectrum, str]]:
    """
    Get the spectra from the given file.

    Parameters
    ----------
    filename : str
        The path of the peak file to be read.
    mz_interval : int
        The width of each m/z interval.
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
        interval = _precursor_to_interval(
            spec.precursor_mz, spec.precursor_charge, mz_interval)
        pkl_filename = os.path.join(
            work_dir, 'spectra', f'{spec.precursor_charge}_{interval}.pkl')
        yield spec, pkl_filename


@nb.njit(cache=True)
def _precursor_to_interval(mz: float, charge: int, interval_width: int) -> int:
    """
    Convert the precursor m/z to the neutral mass and get the interval index.

    Parameters
    ----------
    mz : float
        The precursor m/z.
    charge : int
        The precursor charge.
    interval_width : int
        The width of each m/z interval.

    Returns
    -------
    int
        The index of the interval to which a spectrum with the given m/z and
        charge belongs.
    """
    hydrogen_mass, cluster_width = 1.00794, 1.0005079
    neutral_mass = (mz - hydrogen_mass) * max(abs(charge), 1)
    return round(neutral_mass / cluster_width) // interval_width


def _write_spectra_pkl(pkl_filehandles: Dict[str, BinaryIO],
                       spectra_queue: queue.Queue,
                       lock: multiprocessing.Lock) -> None:
    """
    Read spectra from a queue and write individually to a pickle file.

    Parameters
    ----------
    pkl_filehandles : Dict[str, BinaryIO]
        File handles of the pickle files to dump spectra.
    spectra_queue : queue.Queue
        Queue from which to read spectra for writing to pickle files.
    lock : multiprocessing.Lock
        Lock to avoid race conditions during file handle creation.
    """
    while True:
        spec, pkl_filename = spectra_queue.get()
        if spec is None:
            return
        lock.acquire()
        if pkl_filename not in pkl_filehandles:
            pkl_filehandles[pkl_filename] = open(pkl_filename, 'wb+')
        lock.release()
        pickle.dump(spec, pkl_filehandles[pkl_filename],
                    protocol=pickle.HIGHEST_PROTOCOL)


def _sort_spectra_pkl(filehandle: BinaryIO) -> int:
    """
    Sort spectra in a pickle file by their precursor m/z.

    Parameters
    ----------
    filehandle : BinaryIO
        The pickled spectrum file handle.

    Returns
    -------
    int
        The number of spectra in the file.
    """
    # Read all the spectra from the pickle file.
    spectra = []
    filehandle.seek(0)
    while True:
        try:
            spectra.append(pickle.load(filehandle))
        except EOFError:
            break
    # Write the sorted spectra to the pickle file.
    order = _get_precursor_order(
        np.asarray([spec.precursor_mz for spec in spectra]))
    filehandle.seek(0)
    pickle.dump([spectra[i] for i in order], filehandle,
                protocol=pickle.HIGHEST_PROTOCOL)
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
    return np.argsort(np.asarray(precursor_mzs * 10000).astype(np.int32),
                      kind='mergesort')


def _write_cluster_info(clusters: pd.DataFrame) -> None:
    """
    Export the clustering results to a CSV file.

    Parameters
    ----------
    clusters : pd.DataFrame
        The clustering results.
    """
    with open(f'{config.output_filename}.csv', 'a') as f_out:
        # Metadata.
        f_out.write(f'# falcon version {__version__}\n')
        f_out.write(f'# work_dir = {config.work_dir}\n')
        f_out.write(f'# overwrite = {config.overwrite}\n')
        f_out.write(f'# export_representatives = '
                    f'{config.export_representatives}\n')
        f_out.write(f'# precursor_tol = {config.precursor_tol[0]:.2f} '
                    f'{config.precursor_tol[1]}\n')
        f_out.write(f'# rt_tol = {config.rt_tol}\n')
        f_out.write(f'# fragment_tol = {config.fragment_tol:.2f}\n')
        f_out.write(f'# eps = {config.eps:.3f}\n')
        f_out.write(f'# mz_interval = {config.mz_interval}\n')
        f_out.write(f'# low_dim = {config.low_dim}\n')
        f_out.write(f'# n_neighbors = {config.n_neighbors}\n')
        f_out.write(f'# n_neighbors_ann = {config.n_neighbors_ann}\n')
        f_out.write(f'# batch_size = {config.batch_size}\n')
        f_out.write(f'# n_probe = {config.n_probe}\n')
        f_out.write(f'# min_peaks = {config.min_peaks}\n')
        f_out.write(f'# min_mz_range = {config.min_mz_range:.2f}\n')
        f_out.write(f'# min_mz = {config.min_mz:.2f}\n')
        f_out.write(f'# max_mz = {config.max_mz:.2f}\n')
        f_out.write(f'# remove_precursor_tol = '
                    f'{config.remove_precursor_tol:.2f}\n')
        f_out.write(f'# min_intensity = {config.min_intensity:.2f}\n')
        f_out.write(f'# max_peaks_used = {config.max_peaks_used}\n')
        f_out.write(f'# scaling = {config.scaling}\n')
        f_out.write('#\n')
        # Cluster assignments.
        clusters.to_csv(f_out, index=False, chunksize=1000000)


def _find_spectra_pkl(
        filename: str, spectra_to_read: Dict[Tuple[str, str], int]) \
        -> List[MsmsSpectrum]:
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
    with open(filename, 'rb') as f_in:
        spectra = pickle.load(f_in)
        spectra_key_to_index = {(spec.filename, spec.identifier): i
                                for i, spec in enumerate(spectra)}
        spectra_index_to_cluster = {
            spectra_key_to_index[key]: spectra_to_read[key]
            for key in (set(spectra_key_to_index.keys()) &
                        set(spectra_to_read.keys()))}
        if len(spectra_index_to_cluster) != len(spectra_to_read):
            raise ValueError(
                f'{len(spectra_to_read) - len(spectra_index_to_cluster)} '
                f'spectra not found in file {filename}')
        else:
            spectra_found = []
            for i, cluster_ in spectra_index_to_cluster.items():
                spec = spectra[i]
                spec.cluster = cluster_
                spectra_found.append(spec)
            return spectra_found


if __name__ == '__main__':
    sys.exit(main())
