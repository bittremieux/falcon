import collections
import math
from typing import List, Optional, Tuple, Iterator

import numba as nb
import numpy as np

from . import similarity


ConsensusTuple = collections.namedtuple(
    "ConsensusTuple",
    [
        "precursor_mz",  # np.float32
        "precursor_charge",  # np.int32 or np.nan
        "mz",  # np.ndarray of np.float32
        "intensity",  # np.ndarray of np.float32
        "retention_time",  # np.float32
        "cluster_id",  # np.int32
        "mz_split",  # np.int32
    ],
)


@nb.njit
def _get_cluster_group_idx(clusters: np.ndarray) -> Iterator[Tuple[int, int]]:
    """
    Get start and stop indexes for unique cluster labels.

    Parameters
    ----------
    clusters : np.ndarray
        The ordered cluster labels (noise points are -1).

    Returns
    -------
    Iterator[Tuple[int, int]]
        Tuples with the start index (inclusive) and end index (exclusive) of
        the unique cluster labels.
    """
    start_i = 0
    # Yield all noise points (-1) as singletons first
    while start_i < clusters.shape[0] and clusters[start_i] == -1:
        yield start_i, start_i + 1
        start_i += 1
    # Now yield actual clusters
    stop_i = start_i
    while stop_i < clusters.shape[0]:
        label = clusters[stop_i]
        start_i = stop_i
        while stop_i < clusters.shape[0] and clusters[stop_i] == label:
            stop_i += 1
        yield start_i, stop_i


def _get_representative_spectra(
    spectra: List[similarity.SpectrumTuple],
    labels: np.ndarray,
    rts: np.ndarray,
    order_map: np.ndarray,
    consensus_method: str,
    consensus_params: dict,
) -> List[ConsensusTuple]:
    """
    Get the representative spectra for each cluster.

    Parameters
    ----------
    spectra : List[similarity.SpectrumTuple]
        The spectra.
    labels : np.ndarray
        Cluster labels.
    rts : np.ndarray
        The retention times corresponding to the current interval indexes.
        order_map : np.ndarray
        Map to convert label indexes to pairwise distance matrix indexes.
    consensus_method : str
        The method to use for consensus spectrum computation.
    consensus_params : dict
        Additional parameters for the consensus spectrum computation.

    Returns
    -------
    List[ConsensusTuple]
        The representative spectra for each cluster.
    """
    if consensus_method == "medoid":
        (
            precursor_mzs,
            precursor_charges,
            mzs,
            intensities,
            retention_times,
            cluster_ids,
        ) = _get_cluster_medoids(
            spectra, labels, rts, order_map, **consensus_params
        )
        # create consensus spectra outside of numba
        medoids = [
            ConsensusTuple(
                precursor_mz=np.float32(precursor_mzs[i]),
                precursor_charge=(
                    np.int32(precursor_charges[i])
                    if not np.isnan(precursor_charges[i])
                    else np.nan
                ),
                mz=mzs[i].astype(np.float32),
                intensity=intensities[i].astype(np.float32),
                retention_time=np.float32(retention_times[i]),
                cluster_id=np.int32(cluster_ids[i]),
                mz_split=None,
            )
            for i in range(len(precursor_mzs))
        ]
        return medoids
    elif consensus_method == "average":
        (
            precursor_mzs,
            precursor_charges,
            mzs,
            intensities,
            retention_times,
            cluster_ids,
        ) = _get_cluster_average(
            spectra,
            labels,
            rts,
            order_map,
            **consensus_params,
        )
        # create consensus spectra outside of numba
        avg_spectra = [
            ConsensusTuple(
                precursor_mz=np.float32(precursor_mzs[i]),
                precursor_charge=(
                    np.int32(precursor_charges[i])
                    if not np.isnan(precursor_charges[i])
                    else np.nan
                ),
                mz=mzs[i].astype(np.float32),
                intensity=intensities[i].astype(np.float32),
                retention_time=np.float32(retention_times[i]),
                cluster_id=np.int32(cluster_ids[i]),
                mz_split=None,
            )
            for i in range(len(precursor_mzs))
        ]
        return avg_spectra
    else:
        raise ValueError(
            f"Unknown consensus spectrum method: {consensus_method}"
        )


@nb.njit(fastmath=True, boundscheck=False)
def _get_cluster_medoids(
    spectra: List[similarity.SpectrumTuple],
    labels: np.ndarray,
    rts: np.ndarray,
    order_map: np.ndarray,
    pdist: np.ndarray,
) -> Tuple[
    List[float],
    List[Optional[int]],
    List[np.ndarray],
    List[np.ndarray],
    List[float],
    List[int],
]:
    """
    Get the indexes of the cluster medoids.

    Parameters
    ----------
    spectra : List[similarity.SpectrumTuple]
        The spectra.
    labels : np.ndarray
        Cluster labels.
    rts: np.ndarray
        The retention times corresponding to the current interval indexes.
    pdist : np.ndarray
        Condensed pairwise distance matrix.
    order_map : np.ndarray
        Map to convert label indexes to pairwise distance matrix indexes.

    Returns
    -------
    Tuple[List[float], List[Optional[int]], List[np.ndarray], List[np.ndarray], List[float], List[int]]
        The medoid spectra for each cluster.
    """
    m = len(spectra)
    precursor_mzs = []
    precursor_charges = []
    mzs = []
    intensities = []
    retention_times = []
    cluster_ids = []

    for start_i, stop_i in _get_cluster_group_idx(labels):
        # If less than 3 spectra in cluster, use the first spectrum as medoid.
        if stop_i - start_i > 2:
            row_sum = np.zeros(stop_i - start_i, np.float32)
            for row in range(stop_i - start_i):
                for col in range(row + 1, stop_i - start_i):
                    i, j = order_map[start_i + row], order_map[start_i + col]
                    if i > j:
                        i, j = j, i
                    pdist_ij = pdist[m * i + j - ((i + 2) * (i + 1)) // 2]
                    row_sum[row] += pdist_ij
                    row_sum[col] += pdist_ij
            medoid_spec = spectra[order_map[start_i + np.argmin(row_sum)]]
            precursor_mzs.append(medoid_spec.precursor_mz)
            precursor_charges.append(medoid_spec.precursor_charge)
            mzs.append(medoid_spec.mz)
            intensities.append(medoid_spec.intensity)
            retention_times.append(rts[start_i + np.argmin(row_sum)])
            cluster_ids.append(labels[start_i + np.argmin(row_sum)])
        else:
            medoid_spec = spectra[order_map[start_i]]
            precursor_mzs.append(medoid_spec.precursor_mz)
            precursor_charges.append(medoid_spec.precursor_charge)
            mzs.append(medoid_spec.mz)
            intensities.append(medoid_spec.intensity)
            retention_times.append(rts[start_i])
            cluster_ids.append(labels[start_i])
    return (
        precursor_mzs,
        precursor_charges,
        mzs,
        intensities,
        retention_times,
        cluster_ids,
    )


def _get_cluster_average(
    spectra: List[similarity.SpectrumTuple],
    labels: np.ndarray,
    rts: np.ndarray,
    order_map: np.ndarray,
    min_mz: float,
    max_mz: float,
    bin_size: float,
    outlier_cutoff_lower: float,
    outlier_cutoff_upper: float,
) -> Tuple[
    List[float],
    List[Optional[int]],
    List[np.ndarray],
    List[np.ndarray],
    List[float],
    List[int],
]:
    """
    Get the average spectra for each cluster. The average spectrum is computed
    by binning the spectra, removing (intensity) outliers in each bin, and averaging the remaining peaks.
    Adapted from Carr, A. V. et al. Proteomics (2024).
    https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/pmic.202300234.

    Parameters
    ----------
    spectra : List[similarity.SpectrumTuple]
        The spectra.
    labels : np.ndarray
        Cluster labels.
    rts : np.ndarray
        The retention times corresponding to the current interval indexes.
    order_map : np.ndarray
        Map to convert label indexes to pairwise distance matrix indexes.
    min_mz : float
        The minimum m/z value to consider for binning.
    max_mz : float
        The maximum m/z value to consider for binning.
    bin_size : float
        The width of each bin in m/z units.
    outlier_cutoff_lower : float
        The number of standard deviations for the lower bound for outlier rejection.
    outlier_cutoff_upper : float
        The number of standard deviations for the upper bound for outlier rejection.

    Returns
    -------
    Tuple[List[float], List[Optional[int]], List[np.ndarray], List[np.ndarray], List[float], List[int]]
        The average spectra for each cluster.
    """
    precursor_mzs = []
    precursor_charges = []
    mzs = []
    intensities = []
    retention_times = []
    cluster_ids = []
    for start_i, stop_i in _get_cluster_group_idx(labels):
        if stop_i - start_i > 1:
            spectra_to_average = [
                spectra[order_map[i]] for i in range(start_i, stop_i)
            ]
            # average precursor mz
            avg_mz = np.mean(
                [spec.precursor_mz for spec in spectra_to_average], axis=0
            )
            charge = spectra_to_average[0].precursor_charge
            avg_rt = np.mean(rts[start_i:stop_i])

            # Bin the spectra
            bins_idx, bins_peaks, bins_mz = _spectrum_binning(
                spectra_to_average, min_mz, max_mz, bin_size
            )
            del spectra_to_average
            # Outlier rejection
            bins_idx, bins_peaks, bins_mz = _outlier_rejection(
                bins_idx,
                bins_peaks,
                bins_mz,
                outlier_cutoff_lower,
                outlier_cutoff_upper,
            )
            # Construct average spectrum
            avg_spectrum = _construct_average_spectrum(
                bins_idx,
                bins_peaks,
                bins_mz,
                avg_mz,
                charge,
                avg_rt,
                labels[start_i],
            )
            precursor_mzs.append(avg_spectrum[0])
            precursor_charges.append(avg_spectrum[1])
            mzs.append(avg_spectrum[2])
            intensities.append(avg_spectrum[3])
            retention_times.append(avg_spectrum[4])
            cluster_ids.append(avg_spectrum[5])
        else:
            # Single spectrum cluster
            avg_spectrum = spectra[order_map[start_i]]
            precursor_mzs.append(avg_spectrum.precursor_mz)
            precursor_charges.append(avg_spectrum.precursor_charge)
            mzs.append(avg_spectrum.mz)
            intensities.append(avg_spectrum.intensity)
            retention_times.append(rts[start_i])
            cluster_ids.append(labels[start_i])
    return (
        precursor_mzs,
        precursor_charges,
        mzs,
        intensities,
        retention_times,
        cluster_ids,
    )


@nb.njit(cache=True)
def _spectrum_binning(
    spectra: List[similarity.SpectrumTuple],
    min_mz: float,
    max_mz: float,
    bin_size: float,
) -> Tuple[List[int], nb.typed.List]:
    """
    Jointly bin multiple spectra into fixed-size bins based on m/z values.

    Parameters
    ----------
    spectra : List[similarity.SpectrumTuple]
        A list of spectra to be binned.
    min_mz : float
        The minimum m/z value to consider for binning.
    max_mz : float
        The maximum m/z value to consider for binning.
    bin_size : float
        The width of each bin in m/z units.

    Returns
    -------
    Tuple[np.ndarray, nb.typed.List]
        A tuple containing:
        - An array of integers representing the indices of the non-empty bins.
        - A Numba typed list of arrays containing the intensities for each bin.
    """
    start_dim = min_mz - (min_mz % bin_size)
    end_dim = max_mz + bin_size - (max_mz % bin_size)
    n_bins = math.ceil((end_dim - start_dim) / bin_size)

    n_spectra = len(spectra)

    bins_indices = -np.ones(n_bins, dtype=np.int32)
    bins_peaks = nb.typed.List()
    bins_mz = nb.typed.List()
    for _ in range(n_bins):
        nested_list_p = nb.typed.List.empty_list(nb.types.float32)
        nested_list_m = nb.typed.List.empty_list(nb.types.float32)
        bins_peaks.append(nested_list_p)
        bins_mz.append(nested_list_m)
    bins_spectra_count = np.zeros(n_bins, dtype=np.int32)

    for spec in spectra:
        bins_peak_presence = np.zeros(n_bins, dtype=np.int32)
        for mz, intensity in zip(spec.mz, spec.intensity):
            bin_idx = math.floor((mz - min_mz) / bin_size)
            if 0 <= bin_idx < n_bins:
                if bins_indices[bin_idx] == -1:
                    bins_indices[bin_idx] = bin_idx
                bins_peaks[bin_idx].append(intensity)
                bins_mz[bin_idx].append(mz)
                bins_peak_presence[bin_idx] = 1
        bins_spectra_count += bins_peak_presence
    # Mark peaks that appear in less than 70% of the spectra as empty for removal
    for i in range(n_bins):
        if bins_indices[i] != -1:
            if bins_spectra_count[i] < 0.7 * n_spectra:
                bins_indices[i] = -1
    # Remove empty bins
    mask = bins_indices != -1
    bins_indices = bins_indices[mask]
    bins_peaks_nb = nb.typed.List()
    bins_mz_nb = nb.typed.List()
    for i in bins_indices:
        bins_peaks_nb.append(typed_list_to_numpy(bins_peaks[i]))
        bins_mz_nb.append(typed_list_to_numpy(bins_mz[i]))

    return bins_indices, bins_peaks_nb, bins_mz_nb


@nb.njit(cache=True)
def _outlier_rejection(
    bins_indices: List[int],
    bins_peaks: nb.typed.List,
    bins_mz: nb.typed.List,
    outlier_cutoff_lower: float,
    outlier_cutoff_upper: float,
) -> Tuple[nb.typed.List]:
    """
    Remove outliers from binned spectra using the sigma clipping algorithm and
    return the averaged intensities.

    Parameters
    ----------
    bins_indices : List[int]
        The indices of the non-empty bins.
    bins_peaks : nb.typed.List
        The intensities for each bin.
    bins_mz : nb.typed.List
        The m/z values for each bin.
    outlier_cutoff_lower : float
        The number of standard deviations for the lower bound.
    outlier_cutoff_upper : float
        The number of standard deviations for the upper bound.

    Returns
    -------
    nb.typed.List
        The cleaned and averaged intensities for each bin.
    """
    n_peaks = len(bins_indices)
    cleaned_bins_peaks = np.zeros(n_peaks, dtype=np.float32)
    cleaned_bins_mz = np.zeros(n_peaks, dtype=np.float32)

    zero_peaks = np.zeros(len(bins_indices), dtype=np.bool_)
    for i in range(len(bins_indices)):
        intensities = bins_peaks[i]
        mzs = bins_mz[i]
        if len(intensities) > 2:
            clipped_p, clipped_m = _sigma_clipping(
                intensities,
                mzs,
                outlier_cutoff_lower,
                outlier_cutoff_upper,
            )
            if len(clipped_p) < 1:
                zero_peaks[i] = True
            else:
                cleaned_bins_peaks[i] = np.mean(clipped_p)
                cleaned_bins_mz[i] = np.mean(clipped_m)
        else:
            cleaned_bins_peaks[i] = np.mean(intensities)
            cleaned_bins_mz[i] = np.mean(mzs)

    # Return non-empty bins
    return (
        bins_indices[~zero_peaks],
        cleaned_bins_peaks[~zero_peaks],
        cleaned_bins_mz[~zero_peaks],
    )


@nb.njit(cache=True)
def _sigma_clipping(
    intensities: np.ndarray,
    mzs: np.ndarray,
    outlier_cutoff_lower: float,
    outlier_cutoff_upper: float,
) -> np.ndarray:
    """
    Apply sigma clipping to remove outliers from the array.

    Parameters
    ----------
    intensities : np.ndarray
        The array of intensities.
    mzs : np.ndarray
        The array of m/z values.
    outlier_cutoff_lower : float
        The number of standard deviations for the lower bound.
    outlier_cutoff_upper : float
        The number of standard deviations for the upper bound.

    Returns
    -------
    np.ndarray
        The array of intensities with outliers removed.
    """
    while len(intensities) > 2:
        med = np.median(intensities)
        std = np.std(intensities)
        if std == 0.0:
            break
        # Mask outliers
        mask = _sigma_clip(
            intensities, med, std, outlier_cutoff_lower, outlier_cutoff_upper
        )
        # Break if no outliers were found
        if np.sum(mask) == len(intensities):
            break
        intensities = intensities[mask]
        mzs = mzs[mask]
    return intensities, mzs


@nb.njit(cache=True)
def _sigma_clip(
    values: np.ndarray,
    median: float,
    std: float,
    outlier_cutoff_lower: float,
    outlier_cutoff_upper: float,
) -> np.ndarray:
    """
    Label outlier intensities.

    Parameters
    ----------
    values : np.ndarray
        The values to check.
    median : float
        The median of the values.
    std : float
        The standard deviation of the values.
    outlier_cutoff_lower : float
        The number of standard deviations below the median.
    outlier_cutoff_upper : float
        The number of standard deviations above the median.

    Returns
    -------
    np.ndarray
        mask:
            True if the value is not an outlier (keep)
            False otherwise (remove)
    """
    lower_bound = median - outlier_cutoff_lower * std
    upper_bound = median + outlier_cutoff_upper * std
    return (values >= lower_bound) & (values <= upper_bound)


@nb.njit(cache=True)
def _construct_average_spectrum(
    bins_indices: List[int],
    bins_peaks: nb.typed.List,
    bins_mz: nb.typed.List,
    avg_precursor_mz: float,
    charge: int,
    avg_rt: float,
    cluster: int,
) -> Tuple[float, Optional[int], np.ndarray, np.ndarray, float, int]:
    """
    Construct the average spectrum from the binned spectra.

    Parameters
    ----------
    bins_indices : List[int]
        The indices of the non-empty bins.
    bins_peaks : nb.typed.List
        The intensities for each bin.
    bins_mz : nb.typed.List
        The m/z values for each bin
    avg_precursor_mz : float
        The average precursor m/z.
    charge : int
        The precursor charge.
    avg_rt : float
        The average retention time.
    cluster : int
        The cluster label.

    Returns
    -------
    Tuple[float, Optional[int], np.ndarray, np.ndarray, float, int]
        The average spectrum as a tuple containing:
        - The average precursor m/z.
        - The precursor charge (or None if not available).
        - The m/z values.
        - The intensities.
        - The average retention time.
        - The cluster label.
    """
    mz = np.empty(len(bins_indices), np.float32)
    intensity = np.empty(len(bins_indices), np.float32)

    for idx, (avg_intensity, avg_mz) in enumerate(zip(bins_peaks, bins_mz)):
        mz[idx] = avg_mz
        intensity[idx] = avg_intensity
    return (
        avg_precursor_mz,
        charge if charge is not None else np.nan,
        mz,
        intensity,
        avg_rt,
        cluster,
    )


@nb.njit(cache=True)
def typed_list_to_numpy(lst: nb.typed.List) -> np.ndarray:
    """
    Convert a Numba typed list of floats to a Numpy array.

    Parameters
    ----------
    lst : nb.typed.List
        The list of floats to convert.

    Returns
    -------
    np.ndarray
        The converted array.
    """
    n = len(lst)
    arr = np.empty(n, dtype=np.float32)
    for i in range(n):
        arr[i] = lst[i]
    return arr



