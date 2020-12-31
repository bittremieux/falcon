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

import config
from cluster import cluster, spectrum
from ms_io import ms_io


def main():
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

    if os.path.isdir(config.work_dir):
        logging.warning('Working directory %s already exists, previous '
                        'results might get overwritten', config.work_dir)
    else:
        os.makedirs(config.work_dir)

    # Read the spectra from the input files.
    spectra, spectra_raw = {charge: [] for charge in config.charges}, {}
    logger.info('Read spectra from %d peak file(s)', len(config.filenames))
    # noinspection PyProtectedMember
    for file_spectra in joblib.Parallel(n_jobs=-1)(
            joblib.delayed(ms_io._get_spectra)(filename)
            for filename in config.filenames):
        for spec_raw in file_spectra:
            spec_raw.identifier = f'mzspec:{config.pxd}:{spec_raw.identifier}'
            # Discard low-quality spectra.
            spec_processed = spectrum.process_spectrum(
                copy.copy(spec_raw), config.min_peaks, config.min_mz_range,
                config.min_mz, config.max_mz,
                config.remove_precursor_tolerance, config.min_intensity,
                config.max_peaks_used, config.scaling)
            if (spec_processed is not None
                    and spec_processed.precursor_charge in config.charges):
                spectra_raw[spec_raw.identifier] = spec_raw
                spectra[spec_processed.precursor_charge].append(spec_processed)
    # Make sure the spectra are sorted by precursor m/z for every charge state.
    for charge, spectra_charge in spectra.items():
        spectra[charge] = sorted(
            spectra_charge, key=lambda spec: spec.precursor_mz)

    # Pre-compute index hash mappings.
    vec_len, min_mz, max_mz = spectrum.get_dim(config.min_mz, config.max_mz,
                                               config.fragment_mz_tolerance)
    hash_lookup = np.asarray([murmurhash3_32(i, 0, True) % config.hash_len
                              for i in range(vec_len)], np.uint32)

    # Cluster the spectra per charge.
    clusters_all, current_label, representatives = [], 0, []
    for charge, spectra_charge in spectra.items():
        logger.info('Cluster %d spectra with precursor charge %d',
                    len(spectra_charge), charge)
        # Convert the spectra to hashed vectors.
        vectors = np.zeros((len(spectra_charge), config.hash_len), np.float32)
        joblib.Parallel(n_jobs=-1, prefer='threads')(
            joblib.delayed(spectrum.to_vector)(
                spec, vectors[i, :], min_mz, max_mz,
                config.fragment_mz_tolerance, hash_lookup, False)
            for i, spec in enumerate(spectra_charge))

        # Cluster the vectors.
        precursor_mzs = np.asarray([spec.precursor_mz
                                    for spec in spectra_charge])
        dist_filename = os.path.join(config.work_dir, f'dist_{charge}.npz')
        if not os.path.isfile(dist_filename):
            pairwise_dist_matrix = cluster.compute_pairwise_distances(
                vectors, precursor_mzs, config.precursor_tol_mass,
                config.precursor_tol_mode, config.mz_interval,
                config.n_neighbors, config.n_neighbors_ann,
                config.precursor_tol_mass, config.precursor_tol_mode,
                config.batch_size, config.n_probe,
                os.path.join(config.work_dir, str(charge)))
            ss.save_npz(dist_filename, pairwise_dist_matrix)
        else:
            pairwise_dist_matrix = ss.load_npz(dist_filename)
        clusters = cluster.generate_clusters(
            pairwise_dist_matrix, config.eps, config.min_samples,
            precursor_mzs, config.precursor_tol_mass,
            config.precursor_tol_mode)
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
    clusters_all.to_csv(os.path.join(config.work_dir, 'clusters.csv'),
                        index=False)
    ms_io.write_spectra(os.path.join(config.work_dir, 'clusters.mgf'),
                        sorted(representatives, key=lambda spec: spec.cluster))

    logging.shutdown()


if __name__ == '__main__':
    main()
