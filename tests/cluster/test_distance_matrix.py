import pytest
import numpy as np
from falcon.cluster import distance_matrix, cluster

class TestCondensedIndex:
    def test_condensed_index_basic(self):
        """Test basic cases where the condensed index is computed correctly."""
        assert distance_matrix.condensed_index(0, 1, 5) == 0
        assert distance_matrix.condensed_index(0, 2, 5) == 1
        assert distance_matrix.condensed_index(1, 3, 5) == 5
        assert distance_matrix.condensed_index(2, 3, 5) == 7
        assert distance_matrix.condensed_index(3, 4, 5) == 9

    def test_condensed_index_swapped_inputs(self):
        """Ensure (i, j) gives the same result as (j, i)."""
        assert distance_matrix.condensed_index(2, 4, 5) == distance_matrix.condensed_index(4, 2, 5)
        assert distance_matrix.condensed_index(0, 3, 5) == distance_matrix.condensed_index(3, 0, 5)

    def test_condensed_index_invalid_diagonal(self):
        """Check that passing (i, i) raises a ValueError."""
        with pytest.raises(
            ValueError, match="No diagonal elements in condensed matrix"
        ):
            distance_matrix.condensed_index(2, 2, 5)

    def test_condensed_index_out_of_bounds(self):
        """Test cases where i or j is out of range (negative or >= n)."""
        with pytest.raises(ValueError):
            distance_matrix.condensed_index(-1, 2, 5)
        with pytest.raises(ValueError):
            distance_matrix.condensed_index(2, 5, 5)
        with pytest.raises(ValueError):
            distance_matrix.condensed_index(5, 2, 5)

    def test_condensed_index_invalid_n(self):
        """Test cases where n is invalid."""
        with pytest.raises(ValueError):
            distance_matrix.condensed_index(0, 1, 0)
        with pytest.raises(ValueError):
            distance_matrix.condensed_index(0, 1, -1)

    def test_condensed_index_large_matrix(self):
        """Test with a larger matrix to check correct indexing."""
        assert distance_matrix.condensed_index(10, 20, 100) == 954
        assert distance_matrix.condensed_index(50, 99, 100) == 3773


class TestComputeCondensedDistanceMatrix:
    def test_identical_spectra(self, spectrum_tuple_pair):
        """Identical spectra should have distance 0."""
        spec1, spec2 = spectrum_tuple_pair
        # Use distance_matrix module directly
        pdist = distance_matrix.compute_condensed_distance_matrix(
            [spec1, spec2], fragment_mz_tol=0.5, min_matches=0
        )
        assert abs(pdist[0]) < 0.01

    def test_orthogonal_spectra(self, orthogonal_spectrum_tuple_pair):
        """Non-overlapping spectra should have distance ~1."""
        spec1, spec2 = orthogonal_spectrum_tuple_pair
        pdist = distance_matrix.compute_condensed_distance_matrix(
            [spec1, spec2], fragment_mz_tol=0.05, min_matches=0
        )
        assert pdist[0] > 0.99

    def test_min_matches_filter(self, spectrum_tuple_pair):
        """Pairs with fewer matched peaks than min_matches should get distance 1."""
        spec1, spec2 = spectrum_tuple_pair
        # spec1 and spec2 have 3 peaks each; setting min_matches=100 should force dist=1
        pdist = distance_matrix.compute_condensed_distance_matrix(
            [spec1, spec2], fragment_mz_tol=0.5, min_matches=100
        )
        assert abs(pdist[0] - 1.0) < 0.01
