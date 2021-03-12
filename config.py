import os

import numpy as np


# Precursor charges and m/z's considered.
mz_interval = 1
charges, mzs = (2, 3), np.arange(50, 2501, mz_interval)

# Spectrum preprocessing.
min_peaks = 5
min_mz_range = 250.
min_mz, max_mz = 101., 1500.
remove_precursor_tolerance = 0.5
min_intensity = 0.01
max_peaks_used = 50
scaling = None

# Spectrum to vector conversion.
fragment_mz_tolerance = 0.05
hash_len = 800

# Spectrum matching.
precursor_tol_mass, precursor_tol_mode = 20, 'ppm'

# NN index construction and querying.
n_neighbors, n_neighbors_ann = 64, 128
n_probe = 32
batch_size = 2**16

# DBSCAN clustering.
eps = 0.1
min_samples = 2

# Input/output.
overwrite = False
export_representatives = True
pxd = 'USI000000'
peak_dir = os.path.abspath('../data/interim')
work_dir = os.path.abspath('../data/processed')
filenames = [os.path.join(peak_dir, filename)
             for filename in os.listdir(peak_dir)
             if filename.endswith('.mgf')]
