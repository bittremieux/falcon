import tempfile
from typing import List

import numba as nb
import numpy as np

from . import similarity


def compute_condensed_distance_matrix(
    spec_tuples: List[similarity.SpectrumTuple],
    fragment_mz_tol: float,
    min_matches: int,
) -> np.ndarray:
    """
    Compute the condensed pairwise distance matrix for the given spectra.

    Parameters
    ----------
    spec_tuples : List[similarity.SpectrumTuple]
        The spectra to compute the pairwise distance matrix for.
    fragment_mz_tolerance : float
        The fragment m/z tolerance.
    min_matches : int
        The minimum number of matched peaks to consider the spectra similar.

    Returns
    -------
    np.ndarray
        The condensed pairwise distance matrix.
    """
    n = len(spec_tuples)
    pdist_file = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    pdist_filename = pdist_file.name
    pdist_file.close()  # Close the file so that it can be opened by memmap

    condensed_dist_matrix = np.lib.format.open_memmap(
        pdist_filename,
        mode="w+",
        dtype=np.float32,
        shape=(n * (n - 1) // 2,),
    )

    _condensed_distance_matrix_parallel(
        condensed_dist_matrix,
        spec_tuples,
        fragment_mz_tol,
        min_matches,
    )

    return condensed_dist_matrix


@nb.njit(parallel=True)
def _condensed_distance_matrix_parallel(
    condensed_dist_matrix: np.ndarray,
    spec_tuples: List[similarity.SpectrumTuple],
    fragment_mz_tol: float,
    min_matches: int,
) -> None:
    """
    Compute the condensed pairwise distance matrix for the given spectra.

    Parameters
    ----------
    condensed_dist_matrix : np.ndarray
        The condensed pairwise distance matrix.
    spec_tuples : List[similarity.SpectrumTuple]
        The spectra to compute the pairwise distance matrix for.
    fragment_mz_tol : float
        The fragment m/z tolerance.
    min_matches : int
        The minimum number of matched peaks to consider the spectra similar.
    """
    n = len(spec_tuples)

    for i in nb.prange(n - 1):
        for j in range(i + 1, n):
            spec_tup1 = spec_tuples[i]
            spec_tup2 = spec_tuples[j]
            sim, n_match = similarity.cosine_fast(
                spec_tup1, spec_tup2, fragment_mz_tol
            )
            if n_match < min_matches:
                sim = 0.0
            distance = 1.0 - sim
            idx = condensed_index(i, j, n)
            condensed_dist_matrix[idx] = distance


@nb.njit
def condensed_index(i: int, j: int, n: int) -> int:
    """
    Get the index of the condensed distance matrix.

    Parameters
    ----------
    i : int
        The row index.
    j : int
        The column index.
    n : int
        The number of spectra.

    Returns
    -------
    int
        The index of the condensed distance matrix.
    """
    if n <= 0:
        raise ValueError("Number of spectra must be greater than 0")
    elif i == j:
        raise ValueError("No diagonal elements in condensed matrix")
    elif i < 0 or j < 0:
        raise ValueError("Indices must be non-negative")
    elif i >= n or j >= n:
        raise ValueError("Index out of bounds")
    if i > j:
        i, j = j, i
    return int(n * i + j - ((i + 2) * (i + 1)) // 2)
