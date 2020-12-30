import copy
import logging
import os
import sys

import joblib
import natsort
import numpy as np
import pandas as pd
import scipy.sparse as ss
from sklearn.utils import murmurhash3_32

from cluster import cluster, spectrum
from config import *
from ms_io import ms_io


if __name__ == '__main__':
    # Configure logging.
    logging.captureWarnings(True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        '{asctime} {levelname} [{name}/{processName}] {module}.{funcName} : '
        '{message}', style='{'))
    root.addHandler(handler)
    # Disable dependency non-critical log messages.
    logging.getLogger('faiss').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('numexpr').setLevel(logging.WARNING)
    # Initialize the logger.
    logger = logging.getLogger('spectrum_clustering')

    if os.path.isdir(work_dir):
        logging.warning('Working directory %s already exists, previous '
                        'results might get overwritten', work_dir)
    else:
        os.makedirs(work_dir)

    # Read the spectra from the input files.
    # FIXME: Configurable input files.
    filenames = []
    spectra, spectra_raw = {charge: [] for charge in charges}, {}
    logger.info('Read spectra from %d peak file(s)', len(filenames))
    # noinspection PyProtectedMember
    for file_spectra in joblib.Parallel(n_jobs=-1)(
            joblib.delayed(ms_io._get_spectra)(filename)
            for filename in filenames):
        for spec_raw in file_spectra:
            spec_raw.identifier = f'mzspec:{pxd}:{spec_raw.identifier}'
            spectra_raw[spec_raw.identifier] = copy.copy(spec_raw)
            # Discard low-quality spectra.
            spec_processed = spectrum.process_spectrum(
                spec_raw, min_peaks, min_mz_range, min_mz, max_mz,
                remove_precursor_tolerance, min_intensity, max_peaks_used,
                scaling)
            if (spec_processed is not None
                    and spec_processed.precursor_charge in charges):
                spectra[spec_processed.precursor_charge].append(spec_processed)
    # Make sure the spectra are sorted by precursor m/z for every charge state.
    for charge, spectra_charge in spectra.items():
        spectra[charge] = sorted(
            spectra_charge, key=lambda spec: spec.precursor_mz)

    # Pre-compute index hash mappings.
    vec_len, min_mz, max_mz = spectrum.get_dim(min_mz, max_mz,
                                               fragment_mz_tolerance)
    hash_lookup = np.asarray([murmurhash3_32(i, 0, True) % hash_len
                              for i in range(vec_len)], np.uint32)

    # Cluster the spectra per charge.
    clusters_all, current_label, representatives = [], 0, []
    for charge, spectra_charge in spectra.items():
        logger.info('Cluster %d spectra with precursor charge %d',
                    len(spectra_charge), charge)
        # Convert the spectra to hashed vectors.
        vectors = np.zeros((len(spectra_charge), hash_len), np.float32)
        joblib.Parallel(n_jobs=-1, prefer='threads')(
            joblib.delayed(spectrum.to_vector)(spec, vectors[i, :], min_mz,
                                               max_mz, fragment_mz_tolerance,
                                               hash_lookup, False)
            for i, spec in enumerate(spectra_charge))

        # Cluster the vectors.
        precursor_mzs = np.asarray([spec.precursor_mz
                                    for spec in spectra_charge])
        dist_filename = os.path.join(work_dir, f'dist_{charge}.npz')
        if not os.path.isfile(dist_filename):
            pairwise_dist_matrix = cluster.compute_pairwise_distances(
                vectors, precursor_mzs, precursor_tol_mass, precursor_tol_mode,
                mz_interval, n_neighbors, n_neighbors_ann, precursor_tol_mass,
                precursor_tol_mode, batch_size, n_probe,
                os.path.join(work_dir, str(charge)))
            ss.save_npz(dist_filename, pairwise_dist_matrix)
        else:
            pairwise_dist_matrix = ss.load_npz(dist_filename)
        clusters = cluster.generate_clusters(
            pairwise_dist_matrix, eps, min_samples, precursor_mzs,
            precursor_tol_mass, precursor_tol_mode)
        # Make sure that different charges have non-overlapping cluster labels.
        mask_no_noise = clusters != -1
        clusters[mask_no_noise] += current_label
        current_label = np.amax(clusters[mask_no_noise]) + 1
        # Extract cluster representatives (medoids).
        for cluster_label, representative_i in \
                cluster.get_cluster_representatives(
                    clusters[mask_no_noise], pairwise_dist_matrix):
            representative = spectra_raw[spectra_charge[representative_i]
                                         .identifier]
            representative.cluster = cluster_label
            representatives.append(representative)
        # Save cluster assignments.
        clusters_all.append(pd.DataFrame({
            'identifier': [spec.identifier for spec in spectra_charge],
            'cluster': clusters}))

    # Export cluster memberships and representative spectra.
    clusters_all = (pd.concat(clusters_all, ignore_index=True)
                    .sort_values('identifier', key=natsort.natsort_keygen()))
    clusters_all.to_csv(os.path.join(work_dir, 'clusters.csv'), index=False)
    ms_io.write_spectra(os.path.join(work_dir, 'clusters.mgf'),
                        sorted(representatives, key=lambda spec: spec.cluster))

    logging.shutdown()
