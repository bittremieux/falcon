"""Tests for falcon.ms_io — P4: I/O dispatch and file reading/writing."""

import os
import math

import numpy as np
import pytest

from falcon.ms_io import ms_io


# ---------------------------------------------------------------------------
# ms_io.get_spectra dispatch
# ---------------------------------------------------------------------------

class TestGetSpectraDispatch:
    def test_unknown_extension(self, tmp_path):
        """Unsupported file extension should raise ValueError."""
        f = tmp_path / "test.xyz"
        f.write_text("dummy")
        with pytest.raises(ValueError, match="Unknown spectrum file type"):
            list(ms_io.get_spectra(str(f)))

    def test_nonexistent_file(self):
        """Non-existing file should raise ValueError."""
        with pytest.raises(ValueError, match="Non-existing peak file"):
            list(ms_io.get_spectra("/nonexistent/path/fake.mgf"))

    def test_mgf_dispatch(self, tmp_path):
        """A .mgf file should be dispatched to the MGF reader."""
        mgf = tmp_path / "test.mgf"
        mgf.write_text(
            "BEGIN IONS\n"
            "PEPMASS=500.0\n"
            "CHARGE=2+\n"
            "100.0 50\n"
            "200.0 100\n"
            "300.0 75\n"
            "END IONS\n"
        )
        spectra = list(ms_io.get_spectra(str(mgf)))
        assert len(spectra) == 1


# ---------------------------------------------------------------------------
# ms_io.write_spectra
# ---------------------------------------------------------------------------

class TestWriteSpectra:
    def test_unsupported_format(self, tmp_path):
        """Non-MGF output should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported peak file format"):
            ms_io.write_spectra(str(tmp_path / "out.mzml"), [])


# ---------------------------------------------------------------------------
# MGF round-trip
# ---------------------------------------------------------------------------

class TestMgfRoundtrip:
    def test_write_and_read(self, tmp_path):
        """Written MGF can be read back with the same values."""
        from falcon.cluster.cluster import ConsensusTuple
        from falcon.ms_io import mgf_io

        spectrum = ConsensusTuple(
            precursor_mz=np.float32(500.0),
            precursor_charge=np.int32(2),
            mz=np.array([100.0, 200.0, 300.0], dtype=np.float32),
            intensity=np.array([0.5, 1.0, 0.75], dtype=np.float32),
            retention_time=np.float32(120.0),
            cluster_id=np.int32(0),
            mz_split=np.int32(0),
        )
        out_path = str(tmp_path / "roundtrip.mgf")
        mgf_io.write_spectra(out_path, [spectrum])

        # Read back
        spectra_read = list(mgf_io.get_spectra(out_path))
        assert len(spectra_read) == 1
        s = spectra_read[0]
        assert abs(s.precursor_mz - 500.0) < 0.01
        assert len(s.mz) == 3


# ---------------------------------------------------------------------------
# MGF read specifics
# ---------------------------------------------------------------------------

class TestMgfRead:
    def test_basic_parse(self, tmp_path):
        """Parse a minimal MGF file with known peaks."""
        mgf = tmp_path / "basic.mgf"
        mgf.write_text(
            "BEGIN IONS\n"
            "TITLE=test:scan:1\n"
            "PEPMASS=500.0\n"
            "CHARGE=2+\n"
            "RTINSECONDS=120.0\n"
            "100.0 50\n"
            "200.0 100\n"
            "END IONS\n"
        )
        from falcon.ms_io import mgf_io

        spectra = list(mgf_io.get_spectra(str(mgf)))
        assert len(spectra) == 1
        assert abs(spectra[0].precursor_mz - 500.0) < 0.01
        assert spectra[0].precursor_charge == 2
        assert len(spectra[0].mz) == 2

    def test_no_charge(self, tmp_path):
        """Handle spectra without charge state."""
        mgf = tmp_path / "nocharge.mgf"
        mgf.write_text(
            "BEGIN IONS\n"
            "PEPMASS=400.0\n"
            "100.0 50\n"
            "200.0 100\n"
            "END IONS\n"
        )
        from falcon.ms_io import mgf_io

        spectra = list(mgf_io.get_spectra(str(mgf)))
        assert len(spectra) == 1
        assert spectra[0].precursor_charge is None

    def test_scan_identifier(self, tmp_path):
        """SCANS param should generate correct scan identifier."""
        mgf = tmp_path / "scans.mgf"
        mgf.write_text(
            "BEGIN IONS\n"
            "PEPMASS=500.0\n"
            "CHARGE=2+\n"
            "SCANS=42\n"
            "100.0 50\n"
            "END IONS\n"
        )
        from falcon.ms_io import mgf_io

        spectra = list(mgf_io.get_spectra(str(mgf)))
        assert len(spectra) == 1
        assert "scan:42" in spectra[0].identifier


# ---------------------------------------------------------------------------
# _scale_intensities
# ---------------------------------------------------------------------------

class TestScaleIntensities:
    def test_scale_to_1000(self):
        from falcon.ms_io.mgf_io import _scale_intensities

        intensity = np.array([0.5, 1.0, 0.25], dtype=np.float32)
        scaled = _scale_intensities(intensity)
        assert abs(np.max(scaled) - 1000.0) < 0.01
        assert abs(scaled[0] - 500.0) < 0.01
