import collections
import functools
import glob
import logging
import os
import pickle
import shutil
import sys
import tempfile
from typing import Dict, List, Tuple, Union

import joblib
import natsort
import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse as ss
from sklearn.utils import murmurhash3_32
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
    logger.debug('usi_pxd = %s', config.usi_pxd)
    logger.debug('precursor_tol = %.2f %s', *config.precursor_tol)
    logger.debug('rt_tol = %s', config.rt_tol)
    logger.debug('fragment_tol = %.2f', config.fragment_tol)
    logger.debug('eps = %.3f', config.eps)
    logger.debug('min_samples = %d', config.min_samples)
    logger.debug('mz_interval = %d', config.mz_interval)
    logger.debug('hash_len = %d', config.hash_len)
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

    # Pre-compute the index hash mappings.
    hash_lookup = np.asarray([murmurhash3_32(i, 0, True) % config.hash_len
                              for i in range(vec_len)], np.uint32)
    vectorize = functools.partial(
        spectrum.to_vector_parallel, dim=config.hash_len, min_mz=min_mz,
        max_mz=max_mz, bin_size=config.fragment_tol, hash_lookup=hash_lookup,
        norm=True)

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
            metadata.insert(1, 'precursor_charge', charge)
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
            pairwise_dist_matrix, config.eps, config.min_samples,
            metadata['precursor_mz'].values, config.precursor_tol[0],
            config.precursor_tol[1])
        # Make sure that different charges have non-overlapping cluster labels.
        mask_no_noise = clusters != -1
        clusters[mask_no_noise] += current_label
        current_label = (np.amax(clusters[mask_no_noise]) + 1
                         if any(mask_no_noise) else current_label)
        # Save cluster assignments.
        metadata['cluster'] = clusters
        clusters_all.append(metadata)
        # Extract identifiers for cluster representatives (medoids).
        if config.export_representatives:
            charge_repr = cluster.get_cluster_representatives(
                clusters, pairwise_dist_matrix.indptr,
                pairwise_dist_matrix.indices, pairwise_dist_matrix.data)
            if charge_repr is not None:
                representative_info.append(metadata.iloc[charge_repr])
            if config.export_include_singletons:
                representative_info.append(metadata.loc[~mask_no_noise])
            logger.debug('Extract %d cluster representative %sidentifiers',
                         len(charge_repr) if charge_repr is not None else 0,
                         f'and {(~mask_no_noise).sum()} singleton spectra '
                         if config.export_include_singletons else '')

    # Export cluster memberships and representative spectra.
    n_clusters, n_spectra_clustered = 0, 0
    for clust in clusters_all:
        clust_no_noise = clust[clust['cluster'] != -1]
        n_clusters += clust_no_noise['cluster'].nunique()
        n_spectra_clustered += len(clust_no_noise)
    if n_spectra_clustered == 0:
        logger.error('No valid spectra found for clustering')
        logging.shutdown()
        return 1
    logger.info('Export cluster assignments of %d spectra to %d unique '
                'clusters to output file %s', n_spectra_clustered, n_clusters,
                f'{config.output_filename}.csv')
    clusters_all = (pd.concat(clusters_all, ignore_index=True)
                    .sort_values('identifier', key=natsort.natsort_keygen()))
    with open(f'{config.output_filename}.csv', 'a') as f_out:
        # Metadata.
        f_out.write(f'# falcon version {__version__}\n')
        f_out.write(f'# work_dir = {config.work_dir}\n')
        f_out.write(f'# overwrite = {config.overwrite}\n')
        f_out.write(f'# export_representatives = '
                    f'{config.export_representatives}\n')
        f_out.write(f'# usi_pxd = {config.usi_pxd}\n')
        f_out.write(f'# precursor_tol = {config.precursor_tol[0]:.2f} '
                    f'{config.precursor_tol[1]}\n')
        f_out.write(f'# rt_tol = {config.rt_tol}\n')
        f_out.write(f'# fragment_tol = {config.fragment_tol:.2f}\n')
        f_out.write(f'# eps = {config.eps:.3f}\n')
        f_out.write(f'# min_samples = {config.min_samples}\n')
        f_out.write(f'# mz_interval = {config.mz_interval}\n')
        f_out.write(f'# hash_len = {config.hash_len}\n')
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
        clusters_all.to_csv(f_out, index=False, chunksize=1000000)
    if config.export_representatives:
        representative_info = pd.concat(representative_info, ignore_index=True)
        logger.info('Export %d cluster representative spectra %sto output '
                    'file %s', len(representative_info),
                    ('(including singletons) '
                     if config.export_include_singletons else ''),
                    f'{config.output_filename}.mgf')
        # Get the spectra corresponding to the cluster representatives.
        representative_info['filename'] = representative_info.apply(
            lambda row: os.path.join(
                config.work_dir, 'spectra',
                f"""{row.precursor_charge}_{_precursor_to_interval(
                    row.precursor_mz, row.precursor_charge,
                    config.mz_interval)}.pkl"""),
            axis='columns')
        representatives = []
        for spectra in joblib.Parallel(n_jobs=-1)(
                joblib.delayed(_find_spectra_pkl)(
                    fn, spectra.set_index('identifier')['cluster'].to_dict())
                for fn, spectra in representative_info.groupby('filename')):
            representatives.extend(spectra)
        representatives.sort(key=lambda spec: spec.cluster)
        ms_io.write_spectra(f'{config.output_filename}.mgf', representatives)

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
    spectra = collections.defaultdict(lambda: collections.defaultdict(list))
    for file_spectra in joblib.Parallel(n_jobs=-1)(
            joblib.delayed(_read_spectra)(filename, config.usi_pxd,
                                          config.mz_interval, config.work_dir)
            for filename in input_filenames):
        for spec, pkl_filename in file_spectra:
            spectra[spec.precursor_charge][pkl_filename].append(spec)
    # Make sure the spectra in the individual files are sorted by their
    # precursor m/z and count the number of spectra per precursor charge.
    buckets, n_spectra, n_buckets = {}, 0, 0
    for charge in sorted(spectra.keys()):
        spectra_charge = spectra[charge]
        n_spectra_charge = sum(
            joblib.Parallel(n_jobs=-1)
            (joblib.delayed(_write_spectra_pkl)(filename, file_spectra)
             for filename, file_spectra in spectra_charge.items()))
        buckets[charge] = (n_spectra_charge,
                           natsort.natsorted(spectra_charge.keys()))
        n_spectra += n_spectra_charge
        n_buckets += len(spectra_charge.keys())
    logger.debug('%d spectra written to %d buckets by precursor charge and '
                 'precursor m/z', n_spectra, n_buckets)
    return buckets


def _read_spectra(filename: str, usi_pxd: str, mz_interval: int,
                  work_dir: str) -> List[Tuple[MsmsSpectrum, str]]:
    """
    Get MS/MS spectra from the given file.

    Parameters
    ----------
    filename : str
        The path of the peak file to be read.
    usi_pxd : str
        ProteomeXchange dataset identifier to create USI references.
    mz_interval : int
        The width of each m/z interval.
    work_dir : str
        The directory in which the spectrum buckets will be stored.


    Returns
    -------
    List[Tuple[MsmsSpectrum, str]]
        The spectra read from the given file and their bucket filenames
        (based on precursor charge and m/z).
    """
    spectra = []
    for spec in ms_io.get_spectra(filename):
        spec.identifier = f'mzspec:{usi_pxd}:{spec.identifier}'
        interval = _precursor_to_interval(
            spec.precursor_mz, spec.precursor_charge, mz_interval)
        pkl_filename = os.path.join(work_dir, 'spectra',
                                    f'{spec.precursor_charge}_{interval}.pkl')
        spectra.append((spec, pkl_filename))
    return spectra


@nb.njit
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
    neutral_mass = (mz - hydrogen_mass) * charge
    return round(neutral_mass / cluster_width) // interval_width


def _write_spectra_pkl(filename: str, spectra: List[MsmsSpectrum]) -> int:
    """
    Write the spectra to a pickled file sorted by their precursor m/z.

    Parameters
    ----------
    filename : str
        The pickled spectrum file name.
    spectra : List[MsmsSpectrum]
        The spectra to write to the file.

    Returns
    -------
    int
        The number of spectra in the file.
    """
    # Radix sort on integers with O(n) time complexity.
    order = np.argsort(np.asarray([int(spec.precursor_mz * 10000)
                                   for spec in spectra], dtype=np.int32),
                       kind='stable')
    with open(filename, 'wb') as f_in:
        pickle.dump([spectra[i] for i in order], f_in, protocol=5)
    return len(spectra)


def _find_spectra_pkl(filename: str, usis_to_read: Dict[str, int]) \
        -> List[MsmsSpectrum]:
    """
    Read spectra with the specified USIs from a pkl file.

    Parameters
    ----------
    filename : str
        Name of the pkl file from which the spectra are read.
    usis_to_read : Dict[str, int]
        Dict with as keys the USIs of the spectra to read from the pkl file,
        and as values the cluster labels that will be assigned to the spectra.

    Returns
    -------
    Iterable[MsmsSpectrum]
        The spectra with the specified USIs.
    """
    spectra_found = []
    with open(filename, 'rb') as f_in:
        for spec in pickle.load(f_in):
            if spec.identifier in usis_to_read:
                spec.cluster = usis_to_read[spec.identifier]
                spectra_found.append(spec)
                if len(spectra_found) == len(usis_to_read):
                    return spectra_found
    raise ValueError(f'{len(usis_to_read) - len(spectra_found)} spectra not '
                     f'found in file {filename}')


if __name__ == '__main__':
    sys.exit(main())
