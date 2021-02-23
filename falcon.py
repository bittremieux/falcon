import functools
import logging
import math
import os
import pickle
import sys
from typing import Dict, List

import joblib
import natsort
import numpy as np
import pandas as pd
import scipy.sparse as ss
from sklearn.utils import murmurhash3_32

import config
from cluster import cluster, spectrum
from ms_io import ms_io


logger = logging.getLogger('spectrum_clustering')


def main():
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

    if os.path.isdir(config.work_dir):
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
        charge_count = _prepare_spectra()
        joblib.dump(charge_count, os.path.join(config.work_dir, 'spectra',
                                               'info.joblib'))
    else:
        charge_count = joblib.load(os.path.join(config.work_dir, 'spectra',
                                                'info.joblib'))

    # Pre-compute the index hash mappings.
    vec_len, min_mz, max_mz = spectrum.get_dim(config.min_mz, config.max_mz,
                                               config.fragment_mz_tolerance)
    hash_lookup = np.asarray([murmurhash3_32(i, 0, True) % config.hash_len
                              for i in range(vec_len)], np.uint32)
    vectorize = functools.partial(
        spectrum.to_vector_parallel, dim=config.hash_len, min_mz=min_mz,
        max_mz=max_mz, bin_size=config.fragment_mz_tolerance,
        hash_lookup=hash_lookup, norm=True)

    # Cluster the spectra per charge.
    clusters_all, current_label, representative_ids = [], 0, []
    for charge, n_spectra in charge_count.items():
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
                    charge, n_spectra, config.mzs, vectorize,
                    config.precursor_tol_mass, config.precursor_tol_mode,
                    config.n_neighbors, config.n_neighbors_ann,
                    config.batch_size, config.n_probe, config.work_dir)
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
            metadata['precursor_mz'].values, config.precursor_tol_mass,
            config.precursor_tol_mode)
        # Make sure that different charges have non-overlapping cluster labels.
        mask_no_noise = clusters != -1
        clusters[mask_no_noise] += current_label
        current_label = np.amax(clusters[mask_no_noise]) + 1 if any(mask_no_noise) else current_label
        # Save cluster assignments.
        metadata['cluster'] = clusters
        clusters_all.append(metadata)
        # Extract identifiers for cluster representatives (medoids).
        if config.export_representatives:
            representative_ids.extend(cluster.get_cluster_representatives(
                clusters, pairwise_dist_matrix))

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
    clusters_all.to_csv(os.path.join(config.work_dir, 'clusters.csv'),
                        index=False)
    if config.export_representatives:
        raise NotImplementedError('It is not possible to export '
                                  'representative spectra yet')
        # logger.debug('Export %d cluster representative spectra',
        #              len(representative_ids))
        # representatives = []
        # for cluster_label, representative_i in representative_ids:
        #     representative = None   # TODO: Retrieve the original spectrum.
        #     # representative.cluster = cluster_label
        #     representatives.append(representative)
        # representatives.sort(key=lambda spec: spec.cluster)
        # ms_io.write_spectra(os.path.join(config.work_dir, 'clusters.mgf'),
        #                     representatives)

    logging.shutdown()


def _prepare_spectra() -> Dict[int, int]:
    """
    Read the spectra from the input peak files and partition to intermediate
    files split and sorted by precursor m/z.

    Returns
    -------
    Dict[int, int]
        A dictionary with the number of spectra per precursor charge.
    """
    logger.info('Read spectra from %d peak file(s)', len(config.filenames))
    filehandles = {(charge, mz): open(os.path.join(config.work_dir, 'spectra',
                                                   f'{charge}_{mz}.pkl'), 'wb')
                   for charge in config.charges for mz in config.mzs}
    for file_spectra in joblib.Parallel(n_jobs=-1)(
            joblib.delayed(_read_process_spectra)(filename)
            for filename in config.filenames):
        for spec in file_spectra:
            # FIXME: Add nearby spectra to neighboring files.
            pickle.dump(
                spec,
                filehandles[(
                    spec.precursor_charge,
                    math.floor(spec.precursor_mz / config.mz_interval)
                    * config.mz_interval)],
                protocol=5)
    for filehandle in filehandles.values():
        filehandle.close()
    # Make sure the spectra in the individual files are sorted by their
    # precursor m/z and count the number of spectra per precursor charge.
    logger.debug('Order spectrum splits by precursor m/z')
    charge_count = {charge: 0 for charge in config.charges}
    for charge in config.charges:
        for count in joblib.Parallel(n_jobs=-1)(
                joblib.delayed(_read_write_spectra_pkl)(
                    os.path.join(config.work_dir, 'spectra',
                                 f'{charge}_{mz}.pkl'))
                for mz in config.mzs):
            charge_count[charge] += count
    return charge_count


def _read_process_spectra(filename: str) -> List[spectrum.MsmsSpectrumNb]:
    """
    Get high-quality processed MS/MS spectra from the given file.

    Parameters
    ----------
    filename : str
        The path of the peak file to be read.

    Returns
    -------
    List[spectrum.MsmsSpectrumNb]
        The processed spectra in the given file.
    """
    spectra = []
    for spec_raw in ms_io.get_spectra(filename):
        spec_raw.identifier = f'mzspec:{config.pxd}:{spec_raw.identifier}'
        # Discard low-quality spectra.
        spec_processed = spectrum.process_spectrum(
            spec_raw, config.min_peaks, config.min_mz_range, config.min_mz,
            config.max_mz, config.remove_precursor_tolerance,
            config.min_intensity, config.max_peaks_used, config.scaling)
        if (spec_processed is not None
                and spec_processed.precursor_charge in config.charges):
            spectra.append(spec_processed)
    spectra.sort(key=lambda spec: spec.precursor_mz)
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


if __name__ == '__main__':
    main()
