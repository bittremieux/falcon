import collections
from typing import Tuple

import numba as nb
import numpy as np
import pandas as pd


SpectrumTuple = collections.namedtuple(
    "SpectrumTuple", ["precursor_mz", "precursor_charge", "mz", "intensity"]
)


@nb.njit(fastmath=True, boundscheck=False)
def cosine_fast(
    spec: SpectrumTuple,
    spec_other: SpectrumTuple,
    fragment_mz_tolerance: float,
) -> Tuple[float, int]:
    """
    Compute the cosine similarity between the given spectra.

    Parameters
    ----------
    spec : SpectrumTuple
        Numba-compatible tuple containing information from the first spectrum.
    spec_other : SpectrumTuple
        Numba-compatible tuple containing information from the second spectrum.
    fragment_mz_tolerance : float
        The fragment m/z tolerance used to match peaks in both spectra with
        each other.

    Returns
    -------
    Tuple[float, int]
        A tuple containing the cosine similarity between both spectra and the
        number of matched peaks.
    """
    # Find the matching peaks between both spectra.
    other_peak_index = 0
    peak_matches = []
    for peak_index, (peak_mz, peak_intensity) in enumerate(
        zip(spec.mz, spec.intensity)
    ):
        # Advance while there is an excessive mass difference.
        while other_peak_index < len(spec_other.mz) - 1 and (
            peak_mz - fragment_mz_tolerance > spec_other.mz[other_peak_index]
        ):
            other_peak_index += 1
        # Match the peaks within the fragment mass window if possible.
        other_peak_i = other_peak_index
        while (
            other_peak_i < len(spec_other.mz)
            and abs(peak_mz - (spec_other.mz[other_peak_i]))
            <= fragment_mz_tolerance
        ):
            peak_matches.append(
                (
                    peak_index,
                    other_peak_i,
                    peak_intensity * spec_other.intensity[other_peak_i],
                )
            )
            other_peak_i += 1

    score = 0.0

    peak_matches.sort(key=lambda x: x[2], reverse=True)
    spec_peaks_used = np.zeros(len(spec.mz), np.bool_)
    other_spec_peaks_used = np.zeros(len(spec_other.mz), np.bool_)
    for peak_index, other_peak_index, pair_score in peak_matches:
        if (
            not spec_peaks_used[peak_index]
            and not other_spec_peaks_used[other_peak_index]
        ):
            score += pair_score
            spec_peaks_used[peak_index] = True
            other_spec_peaks_used[other_peak_index] = True

    score = max(0.0, min(score, 1.0))

    return score, spec_peaks_used.sum()


def df_row_to_spectrum_tuple(row: pd.Series) -> SpectrumTuple:
    """
    Convert a row of a DataFrame to a SpectrumTuple.

    Parameters
    ----------
    row : pd.Series
        The row of a DataFrame containing the spectrum information.

    Returns
    -------
    SpectrumTuple
        A tuple containing the spectrum information.
    """
    return SpectrumTuple(
        row["precursor_mz"],
        row["precursor_charge"],
        row["mz"],
        row["intensity"],
    )
