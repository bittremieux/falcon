import collections
import functools
import glob
import logging
import math
import os
import pickle
import shutil
import sys
import tempfile
from typing import Dict, List, Tuple, Union

import joblib
import natsort
import numpy as np
import pandas as pd
import scipy.sparse as ss
from sklearn.utils import murmurhash3_32
from spectrum_utils.spectrum import MsmsSpectrum

from cluster import cluster, spectrum
from config import config
from ms_io import ms_io


logger = logging.getLogger('spectrum_clustering')


def main(args: Union[str, List[str]] = None) -> int:
    # Load the configuration.
    config.parse(args)

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

    # Clean all intermediate results if "overwrite" is specified.
    if config.overwrite:
        for filename in os.listdir(os.path.join(config.work_dir, 'spectra')):
            os.remove(os.path.join(config.work_dir, 'spectra', filename))
        for filename in os.listdir(os.path.join(config.work_dir, 'nn')):
            os.remove(os.path.join(config.work_dir, 'nn', filename))
        if os.path.isfile(os.path.join(config.work_dir, 'clusters.csv')):
            os.remove(os.path.join(config.work_dir, 'clusters.csv'))
        if os.path.isfile(os.path.join(config.work_dir, 'clusters.mgf')):
            os.remove(os.path.join(config.work_dir, 'clusters.mgf'))

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
    process_spectrum = functools.partial(
        spectrum.process_spectrum,
        min_peaks=config.min_peaks,
        min_mz_range=config.min_mz_range,
        mz_min=config.min_mz,
        mz_max=config.max_mz,
        remove_precursor_tolerance=config.remove_precursor_tol,
        min_intensity=config.min_intensity,
        max_peaks_used=config.max_peaks_used,
        scaling=None if config.scaling == 'off' else config.scaling)

    # Pre-compute the index hash mappings.
    vec_len, min_mz, max_mz = spectrum.get_dim(config.min_mz, config.max_mz,
                                               config.fragment_tol)
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
            representative_info.append(
                metadata.iloc[list(cluster.get_cluster_representatives(
                    clusters, pairwise_dist_matrix))])

    # Export cluster memberships and representative spectra.
    n_clusters, n_spectra_clustered = 0, 0
    for clust in clusters_all:
        clust_no_noise = clust[clust['cluster'] != -1]
        n_clusters += clust_no_noise['cluster'].nunique()
        n_spectra_clustered += len(clust_no_noise)
    logger.info('Export cluster assignments of %d spectra to %d unique '
                'clusters', n_spectra_clustered, n_clusters)
    clusters_all = (pd.concat(clusters_all, ignore_index=True)
                    .sort_values('identifier', key=natsort.natsort_keygen()))
    clusters_all.to_csv(f'{config.output_filename}.csv', index=False)
    if config.export_representatives:
        representative_info = (
            pd.concat(representative_info, ignore_index=True)
            .sort_values(['precursor_charge', 'precursor_mz']))
        logger.debug('Export %d cluster representative spectra',
                     len(representative_info))
        # Get the spectra corresponding to the cluster representatives.
        representative_info['file_mz'] = \
            representative_info['precursor_mz'].apply(
                lambda mz: (math.floor(mz / config.mz_interval)
                            * config.mz_interval))
        representative_info['filename'] = \
            representative_info[['precursor_charge', 'file_mz']].apply(
                lambda row: os.path.join(config.work_dir, 'spectra',
                                         f'{row[0]}_{row[1]}.pkl'),
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
    logger.info('Read spectra from %d peak file(s)',
                len(input_filenames))
    filehandles = collections.defaultdict(dict)
    for file_spectra in joblib.Parallel(n_jobs=-1)(
            joblib.delayed(_read_spectra)(filename, config.usi_pxd)
            for filename in input_filenames):
        for spec in file_spectra:
            charge = spec.precursor_charge
            mz = (math.floor(spec.precursor_mz / config.mz_interval)
                  * config.mz_interval)
            filename = os.path.join(config.work_dir, 'spectra',
                                    f'{charge}_{mz}.pkl')
            if mz not in filehandles[charge]:
                filehandles[charge][mz] = open(filename, 'wb')
            pickle.dump(spec, filehandles[charge][mz], protocol=5)
            # FIXME: Add nearby spectra to neighboring files.
    for filehandles_charge in filehandles.values():
        for filehandle in filehandles_charge.values():
            filehandle.close()
    # Make sure the spectra in the individual files are sorted by their
    # precursor m/z and count the number of spectra per precursor charge.
    logger.debug('Order spectrum splits by precursor m/z')
    buckets = {}
    for charge, filehandles_charge in sorted(filehandles.items()):
        count, filenames = 0, []
        for mz, filehandles_charge_mz in sorted(filehandles_charge.items()):
            # TODO: Parallelize.
            filename = os.path.join(config.work_dir, 'spectra',
                                    f'{charge}_{mz}.pkl')
            count += _read_write_spectra_pkl(filename)
            filenames.append(filename)
        buckets[charge] = count, filenames
    return buckets


def _read_spectra(filename: str, usi_pxd: str) -> List[MsmsSpectrum]:
    """
    Get high-quality processed MS/MS spectra from the given file.

    Parameters
    ----------
    filename : str
        The path of the peak file to be read.
    usi_pxd : str
        ProteomeXchange dataset identifier to create USI references.

    Returns
    -------
    List[MsmsSpectrum]
        The spectra in the given file.
    """
    spectra = []
    for spec in ms_io.get_spectra(filename):
        spec.identifier = f'mzspec:{usi_pxd}:{spec.identifier}'
        spectra.append(spec)
    spectra.sort(key=lambda s: s.precursor_mz)
    return spectra


def _read_write_spectra_pkl(filename: str) -> int:
    """
    Read the spectra from the pickled file and write them back to the same file
    sorted by their precursor m/z.

    Parameters
    ----------
    filename : str
        The pickled spectrum file name.

    Returns
    -------
    int
        The number of spectra in the file.
    """
    spectra = []
    with open(filename, 'rb') as f:
        while True:
            try:
                spectra.append(pickle.load(f))
            except EOFError:
                break
    if len(spectra) == 0:
        os.remove(filename)
        return 0
    else:
        spectra.sort(key=lambda spec: spec.precursor_mz)
        with open(filename, 'wb') as f_out:
            pickle.dump(spectra, f_out, protocol=5)
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
            if spec.identifier in usis_to_read.keys():
                spec.cluster = usis_to_read[spec.identifier]
                spectra_found.append(spec)
                if len(spectra_found) == len(usis_to_read):
                    return spectra_found
    raise ValueError(f'{len(usis_to_read) - len(spectra_found)} spectra not '
                     f'found in file {filename}')


if __name__ == '__main__':
    # Configure logging.
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter(
        '{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : '
        '{message}', style='{'))
    root.addHandler(handler)
    # Disable dependency non-critical log messages.
    logging.getLogger('faiss').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('numexpr').setLevel(logging.WARNING)

    main()
