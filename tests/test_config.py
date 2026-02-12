"""Tests for falcon.config — P5: Configuration parsing and validation."""

import pytest

from falcon.config import Config


class TestConfigParse:
    def test_defaults(self, tmp_path):
        """Default values should match expected defaults."""
        # Create a dummy input file so the parser doesn't complain
        dummy = tmp_path / "dummy.mgf"
        dummy.write_text("")
        cfg = Config()
        cfg.parse([str(dummy), "output"])
        assert cfg.fragment_tol == 0.05
        assert cfg.min_peaks == 5
        assert cfg.min_mz_range == 250.0
        assert cfg.min_mz == 101.0
        assert cfg.max_mz == 1500.0
        assert cfg.scaling == "off"
        assert cfg.linkage == "complete"
        assert cfg.distance_threshold == 0.1
        assert cfg.batch_size == 2**15
        assert cfg.min_matched_peaks == 0
        assert cfg.consensus_method == "medoid"
        assert cfg.overwrite is False
        assert cfg.export_representatives is False

    def test_precursor_tol_ppm(self, tmp_path):
        """--precursor_tol with ppm should be parsed correctly."""
        dummy = tmp_path / "dummy.mgf"
        dummy.write_text("")
        cfg = Config()
        cfg.parse([str(dummy), "output", "--precursor_tol", "20", "ppm"])
        assert cfg.precursor_tol[0] == 20.0
        assert cfg.precursor_tol[1] == "ppm"

    def test_precursor_tol_da(self, tmp_path):
        """--precursor_tol with Da should be parsed correctly."""
        dummy = tmp_path / "dummy.mgf"
        dummy.write_text("")
        cfg = Config()
        cfg.parse([str(dummy), "output", "--precursor_tol", "0.5", "Da"])
        assert cfg.precursor_tol[0] == 0.5
        assert cfg.precursor_tol[1] == "Da"

    def test_getattr_before_parse(self):
        """Accessing config before parse() should raise RuntimeError."""
        cfg = Config()
        with pytest.raises(RuntimeError, match="not been initialized"):
            _ = cfg.fragment_tol

    def test_scaling_choices(self, tmp_path):
        """Valid scaling values should be accepted."""
        dummy = tmp_path / "dummy.mgf"
        dummy.write_text("")
        for scaling in ["off", "root", "log", "rank"]:
            cfg = Config()
            cfg.parse([str(dummy), "output", "--scaling", scaling])
            assert cfg.scaling == scaling
