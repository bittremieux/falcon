import os
import tempfile
from typing import List, Tuple

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
    fragment_mz_tol : float
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
    # Unlink the backing file now that it is mapped. On POSIX the inode (and its
    # data) stays alive until this memmap is released (e.g. the `del pdist` in
    # `_cluster_mz_interval`), after which the OS reclaims the space
    # automatically. Without this the temp file is never removed and leaks for
    # the lifetime of the process, accumulating one file per m/z interval and
    # potentially filling the disk on large datasets.
    os.unlink(pdist_filename)

    _condensed_distance_matrix_parallel(
        condensed_dist_matrix,
        spec_tuples,
        fragment_mz_tol,
        min_matches,
    )

    return condensed_dist_matrix


def open_shared_condensed(n: int) -> Tuple[str, np.ndarray]:
    """
    Create a named, on-disk condensed distance matrix that several processes
    can fill concurrently.

    Unlike `compute_condensed_distance_matrix`, the backing file is NOT unlinked
    here: tile workers and the finalize step open it by name in separate
    processes, so the caller is responsible for unlinking it once every opener
    is done (see `generate_clusters_multi`).

    Parameters
    ----------
    n : int
        The number of spectra in the interval.

    Returns
    -------
    Tuple[str, np.ndarray]
        The backing file path and the writable memmap of shape
        ``(n * (n - 1) // 2,)``.
    """
    pdist_file = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    pdist_filename = pdist_file.name
    pdist_file.close()
    condensed_dist_matrix = np.lib.format.open_memmap(
        pdist_filename,
        mode="w+",
        dtype=np.float32,
        shape=(n * (n - 1) // 2,),
    )
    return pdist_filename, condensed_dist_matrix


@nb.njit
def _condensed_rows(
    condensed_dist_matrix: np.ndarray,
    spec_tuples: List[similarity.SpectrumTuple],
    r0: int,
    r1: int,
    fragment_mz_tol: float,
    min_matches: int,
) -> None:
    """
    Fill the condensed-distance entries for rows ``[r0, r1)`` of the upper
    triangle (pairs ``(i, j)`` with ``r0 <= i < r1`` and ``i < j < n``).

    This computes one contiguous row-band of the same matrix produced by
    `_condensed_distance_matrix_parallel`, writing into the shared memmap at the
    canonical `condensed_index` positions. It is intentionally serial (no
    `prange`): parallelism comes from dispatching many of these as independent
    pool tasks, one worker each, so workers stay single-threaded and don't
    oversubscribe. Each pair is computed exactly once and identically to the
    full build, so the assembled matrix is bit-for-bit identical.
    """
    n = len(spec_tuples)
    for i in range(r0, r1):
        spec_tup1 = spec_tuples[i]
        for j in range(i + 1, n):
            sim, n_match = similarity.cosine_fast(
                spec_tup1, spec_tuples[j], fragment_mz_tol
            )
            if n_match < min_matches:
                sim = 0.0
            condensed_dist_matrix[condensed_index(i, j, n)] = 1.0 - sim


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
