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

    def test_rt_tol_default_and_override(self, tmp_path):
        """rt_tol defaults to None and parses as a float when given."""
        dummy = tmp_path / "dummy.mgf"
        dummy.write_text("")
        cfg = Config()
        cfg.parse([str(dummy), "output"])
        assert cfg.rt_tol is None
        cfg = Config()
        cfg.parse([str(dummy), "output", "--rt_tol", "30"])
        assert cfg.rt_tol == 30.0

    def test_precursor_charge_buckets_default_none(self, tmp_path):
        """Without --precursor_charge_buckets the value stays None so that
        every distinct charge is clustered separately."""
        dummy = tmp_path / "dummy.mgf"
        dummy.write_text("")
        cfg = Config()
        cfg.parse([str(dummy), "output"])
        assert cfg.precursor_charge_buckets is None

    def test_precursor_charge_buckets_override(self, tmp_path):
        """Explicit buckets are parsed and validated into sets/'other'."""
        dummy = tmp_path / "dummy.mgf"
        dummy.write_text("")
        cfg = Config()
        cfg.parse(
            [
                str(dummy),
                "output",
                "--precursor_charge_buckets",
                "[1]",
                "[2, 3]",
                "[unknown]",
                "other",
            ]
        )
        assert cfg.precursor_charge_buckets == [
            {1},
            {2, 3},
            {"unknown"},
            "other",
        ]

    def test_clustering_overrides(self, tmp_path):
        """Linkage, fragment_tol, and consensus_method parse correctly."""
        dummy = tmp_path / "dummy.mgf"
        dummy.write_text("")
        cfg = Config()
        cfg.parse(
            [
                str(dummy),
                "output",
                "--linkage",
                "average",
                "--fragment_tol",
                "0.02",
                "--consensus_method",
                "average",
            ]
        )
        assert cfg.linkage == "average"
        assert cfg.fragment_tol == 0.02
        assert cfg.consensus_method == "average"

    def test_outlier_cutoffs(self, tmp_path):
        """Outlier cutoff bounds parse as floats."""
        dummy = tmp_path / "dummy.mgf"
        dummy.write_text("")
        cfg = Config()
        cfg.parse(
            [
                str(dummy),
                "output",
                "--outlier_cutoff_lower",
                "2.0",
                "--outlier_cutoff_upper",
                "3.0",
            ]
        )
        assert cfg.outlier_cutoff_lower == 2.0
        assert cfg.outlier_cutoff_upper == 3.0

    def test_config_file_loading(self, tmp_path):
        """Settings in a config file are applied via -c."""
        dummy = tmp_path / "dummy.mgf"
        dummy.write_text("")
        cfg_file = tmp_path / "config.ini"
        cfg_file.write_text("fragment_tol = 0.1\nmin_peaks = 9\n")
        cfg = Config()
        cfg.parse([str(dummy), "output", "-c", str(cfg_file)])
        assert cfg.fragment_tol == 0.1
        assert cfg.min_peaks == 9

    def test_cli_overrides_config_file(self, tmp_path):
        """Command-line arguments take precedence over the config file."""
        dummy = tmp_path / "dummy.mgf"
        dummy.write_text("")
        cfg_file = tmp_path / "config.ini"
        cfg_file.write_text("fragment_tol = 0.1\n")
        cfg = Config()
        cfg.parse(
            [
                str(dummy),
                "output",
                "-c",
                str(cfg_file),
                "--fragment_tol",
                "0.03",
            ]
        )
        assert cfg.fragment_tol == 0.03
