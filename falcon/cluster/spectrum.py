import math
from typing import Dict, List, Optional, Tuple, Union

import faiss
import numba as nb
import numpy as np
import pandas as pd
import scipy.sparse as ss
import spectrum_utils.spectrum as sus


@nb.njit(cache=True)
def _check_spectrum_valid(
    spectrum_mz: np.ndarray, min_peaks: int, min_mz_range: float
) -> bool:
    """
    Check whether a cluster is of good enough quality to be used.

    Parameters
    ----------
    spectrum_mz : np.ndarray
        M/z peaks of the cluster whose quality is checked.
    min_peaks : int
        Minimum number of peaks the cluster has to contain.
    min_mz_range : float
        Minimum m/z range the cluster's peaks need to cover.

    Returns
    -------
    bool
        True if the cluster has enough peaks covering a wide enough mass
        range, False otherwise.
    """
    return (
        len(spectrum_mz) >= min_peaks
        and spectrum_mz[-1] - spectrum_mz[0] >= min_mz_range
    )


@nb.njit(cache=True)
def _norm_intensity(spectrum_intensity: np.ndarray) -> np.ndarray:
    """
    Normalize cluster peak intensities by their vector norm.

    Parameters
    ----------
    spectrum_intensity : np.ndarray
        The cluster peak intensities to be normalized.

    Returns
    -------
    np.ndarray
        The normalized peak intensities.
    """
    return spectrum_intensity / np.linalg.norm(spectrum_intensity)


def process_spectrum(
    spectrum: sus.MsmsSpectrum,
    min_peaks: int,
    min_mz_range: float,
    mz_min: Optional[float] = None,
    mz_max: Optional[float] = None,
    remove_precursor_tolerance: Optional[float] = None,
    min_intensity: Optional[float] = None,
    max_peaks_used: Optional[int] = None,
    scaling: Optional[str] = None,
) -> Dict[str, Union[str, int, float, np.ndarray]]:
    """
    Process a cluster.

    Processing steps include:
    - Restrict the m/z range to a minimum and maximum m/z.
    - Remove peak(s) around the precursor m/z value.
    - Remove peaks below a percentage of the base peak intensity.
    - Retain only the top most intense peaks.
    - Scale and normalize peak intensities.

    Parameters
    ----------
    spectrum : MsmsSpectrum
        The cluster to be processed.
    min_peaks : int
        Minimum number of peaks the cluster has to contain to be valid.
    min_mz_range : float
        Minimum m/z range the cluster's peaks need to cover to be valid.
    mz_min : Optional[float], optional
        Minimum m/z (inclusive). If not set no minimal m/z restriction will
        occur.
    mz_max : Optional[float], optional
        Maximum m/z (inclusive). If not set no maximal m/z restriction will
        occur.
    remove_precursor_tolerance : Optional[float], optional
        Fragment mass tolerance (in Dalton) around the precursor mass to remove
        the precursor peak. If not set, the precursor peak will not be removed.
    min_intensity : Optional[float], optional
        Remove peaks whose intensity is below `min_intensity` percentage
        of the base peak intensity. If None, no minimum intensity filter will
        be applied.
    max_peaks_used : Optional[int], optional
        Only retain the `max_peaks_used` most intense peaks. If None, all peaks
        are retained.
    scaling : {'root', 'log', 'rank'}, optional
        Method to scale the peak intensities. Potential transformation options
        are:

        - 'root': Square root-transform the peak intensities.
        - 'log':  Log2-transform (after summing the intensities with 1 to avoid
          negative values after the transformation) the peak intensities.
        - 'rank': Rank-transform the peak intensities with maximum rank
          `max_peaks_used`.
        - None: No scaling is performed.

    Returns
    -------
    Dict[str, Union[str, int, float, np.ndarray]]
        The processed spectrum as a dictionary.
    """
    none_charge = spectrum.precursor_charge is None
    spectrum = spectrum.set_mz_range(mz_min, mz_max)
    if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
        return None

    if remove_precursor_tolerance is not None:
        # Temporarily set the precursor charge to 1 to remove the precursor peak
        if none_charge:
            spectrum.precursor_charge = 1
        spectrum = spectrum.remove_precursor_peak(
            remove_precursor_tolerance, "Da", 0
        )
        if none_charge:
            spectrum.precursor_charge = None
        if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
            return None

    if min_intensity is not None or max_peaks_used is not None:
        min_intensity = 0.0 if min_intensity is None else min_intensity
        spectrum = spectrum.filter_intensity(min_intensity, max_peaks_used)
        if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
            return None

    spectrum = spectrum.scale_intensity(scaling, max_rank=max_peaks_used)
    intensity = _norm_intensity(spectrum.intensity)

    # noinspection PyUnresolvedReferences
    return {
        "identifier": spectrum.identifier,
        "precursor_mz": spectrum.precursor_mz,
        "precursor_charge": spectrum.precursor_charge,
        "mz": spectrum.mz,
        "intensity": intensity,
        "retention_time": spectrum.retention_time,
    }


@nb.njit("Tuple((u4, f4, f4))(f4, f4, f4)", cache=True)
def get_dim(
    min_mz: float, max_mz: float, bin_size: float
) -> Tuple[int, float, float]:
    """
    Compute the number of bins over the given mass range for the given bin
    size.

    Parameters
    ----------
    min_mz : float
        The minimum mass in the mass range (inclusive).
    max_mz : float
        The maximum mass in the mass range (inclusive).
    bin_size : float
        The bin size (in Da).

    Returns
    -------
        A tuple containing (i) the number of bins over the given mass range for
        the given bin size, (ii) the highest multiple of bin size lower than
        the minimum mass, (iii) the lowest multiple of the bin size greater
        than the maximum mass. These two final values are the true boundaries
        of the mass range (inclusive min, exclusive max).
    """
    start_dim = min_mz - min_mz % bin_size
    end_dim = max_mz + bin_size - max_mz % bin_size
    return math.ceil((end_dim - start_dim) / bin_size), start_dim, end_dim
