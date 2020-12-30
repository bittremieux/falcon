# Precursor charges considered.
charges = 2, 3

# Spectrum preprocessing.
min_peaks = 5
min_mz_range = 250.
min_mz, max_mz = 101., 1500.
remove_precursor_tolerance = 0.5
min_intensity = 0.01
max_peaks_used = 50
scaling = 'rank'

# Spectrum to vector conversion.
fragment_mz_tolerance = 0.05
hash_len = 1600

# Spectrum matching.
precursor_tol_mass, precursor_tol_mode = 20, 'ppm'

# NN index construction and querying.
mz_interval = 1
n_neighbors, n_neighbors_ann = 64, 128
n_probe = 128
batch_size = 2**16

# DBSCAN clustering.
eps = 0.3
min_samples = 2

# Miscellaneous.
work_dir = '/tmp'
