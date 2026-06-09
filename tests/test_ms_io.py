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

    def test_missing_rt_is_nan(self, tmp_path):
        """A spectrum without RTINSECONDS should parse RT as NaN (not -1/0)."""
        mgf = tmp_path / "nort.mgf"
        mgf.write_text(
            "BEGIN IONS\n"
            "TITLE=test:scan:1\n"
            "PEPMASS=500.0\n"
            "CHARGE=2+\n"
            "100.0 50\n"
            "200.0 100\n"
            "END IONS\n"
        )
        from falcon.ms_io import mgf_io

        spectra = list(mgf_io.get_spectra(str(mgf)))
        assert len(spectra) == 1
        assert math.isnan(spectra[0].retention_time)

    def test_cluster_title_preserved(self, tmp_path):
        """A '<filename>:cluster:<id>' title should be passed through as-is."""
        mgf = tmp_path / "myrun.mgf"
        mgf.write_text(
            "BEGIN IONS\n"
            "TITLE=myrun:cluster:7\n"
            "PEPMASS=500.0\n"
            "CHARGE=2+\n"
            "100.0 50\n"
            "200.0 100\n"
            "END IONS\n"
        )
        from falcon.ms_io import mgf_io

        spectra = list(mgf_io.get_spectra(str(mgf)))
        assert len(spectra) == 1
        assert spectra[0].identifier == "myrun:cluster:7"

    def test_usi_title_preserved(self, tmp_path):
        """A valid USI title (mzspec:...) should be passed through as-is."""
        mgf = tmp_path / "run.mgf"
        mgf.write_text(
            "BEGIN IONS\n"
            "TITLE=mzspec:PXD000000:run:scan:9\n"
            "PEPMASS=500.0\n"
            "CHARGE=2+\n"
            "100.0 50\n"
            "200.0 100\n"
            "END IONS\n"
        )
        from falcon.ms_io import mgf_io

        spectra = list(mgf_io.get_spectra(str(mgf)))
        assert len(spectra) == 1
        assert spectra[0].identifier == "mzspec:PXD000000:run:scan:9"

    def test_scan_singular_fallback(self, tmp_path):
        """A 'SCAN' (singular) param should be used when 'SCANS' is absent."""
        mgf = tmp_path / "sng.mgf"
        mgf.write_text(
            "BEGIN IONS\n"
            "SCAN=77\n"
            "PEPMASS=500.0\n"
            "CHARGE=2+\n"
            "100.0 50\n"
            "200.0 100\n"
            "END IONS\n"
        )
        from falcon.ms_io import mgf_io

        spectra = list(mgf_io.get_spectra(str(mgf)))
        assert len(spectra) == 1
        assert spectra[0].identifier == "sng:scan:77"

    def test_index_fallback(self, tmp_path):
        """Without a title/scans/scan, the identifier falls back to the index."""
        mgf = tmp_path / "idx.mgf"
        mgf.write_text(
            "BEGIN IONS\n"
            "PEPMASS=500.0\n"
            "CHARGE=2+\n"
            "100.0 50\n"
            "200.0 100\n"
            "END IONS\n"
            "BEGIN IONS\n"
            "PEPMASS=600.0\n"
            "CHARGE=2+\n"
            "100.0 50\n"
            "200.0 100\n"
            "END IONS\n"
        )
        from falcon.ms_io import mgf_io

        spectra = list(mgf_io.get_spectra(str(mgf)))
        assert [s.identifier for s in spectra] == [
            "idx:index:0",
            "idx:index:1",
        ]

    def test_non_usi_title_replaced(self, tmp_path):
        """A plain title (not a USI, not a cluster title) is replaced."""
        mgf = tmp_path / "foo.mgf"
        mgf.write_text(
            "BEGIN IONS\n"
            "TITLE=just a label\n"
            "PEPMASS=500.0\n"
            "CHARGE=2+\n"
            "100.0 50\n"
            "200.0 100\n"
            "END IONS\n"
        )
        from falcon.ms_io import mgf_io

        spectra = list(mgf_io.get_spectra(str(mgf)))
        assert len(spectra) == 1
        # No scans/scan present => identifier falls back to the index.
        assert spectra[0].identifier == "foo:index:0"

    def test_filename_param_override(self, tmp_path):
        """A 'FILENAME' param overrides the on-disk file name in identifiers."""
        mgf = tmp_path / "ondisk.mgf"
        mgf.write_text(
            "BEGIN IONS\n"
            "FILENAME=original.raw\n"
            "SCANS=5\n"
            "PEPMASS=500.0\n"
            "CHARGE=2+\n"
            "100.0 50\n"
            "200.0 100\n"
            "END IONS\n"
        )
        from falcon.ms_io import mgf_io

        spectra = list(mgf_io.get_spectra(str(mgf)))
        assert len(spectra) == 1
        assert spectra[0].identifier == "original:scan:5"

    def test_rt_parsed(self, tmp_path):
        """RTINSECONDS should be parsed into the retention time."""
        mgf = tmp_path / "rt.mgf"
        mgf.write_text(
            "BEGIN IONS\n"
            "SCANS=1\n"
            "PEPMASS=500.0\n"
            "CHARGE=2+\n"
            "RTINSECONDS=123.5\n"
            "100.0 50\n"
            "200.0 100\n"
            "END IONS\n"
        )
        from falcon.ms_io import mgf_io

        spectra = list(mgf_io.get_spectra(str(mgf)))
        assert len(spectra) == 1
        assert spectra[0].retention_time == pytest.approx(123.5)

    def test_negative_charge(self, tmp_path):
        """Negative-mode charge states are parsed with their sign."""
        mgf = tmp_path / "neg.mgf"
        mgf.write_text(
            "BEGIN IONS\n"
            "SCANS=1\n"
            "PEPMASS=500.0\n"
            "CHARGE=3-\n"
            "100.0 50\n"
            "200.0 100\n"
            "END IONS\n"
        )
        from falcon.ms_io import mgf_io

        spectra = list(mgf_io.get_spectra(str(mgf)))
        assert len(spectra) == 1
        assert spectra[0].precursor_charge == -3

    def test_multiple_spectra_with_malformed_skipped(self, tmp_path):
        """A spectrum that fails to parse is skipped; valid ones still read."""
        mgf = tmp_path / "multi.mgf"
        mgf.write_text(
            "BEGIN IONS\n"
            "SCANS=1\n"
            "PEPMASS=500.0\n"
            "CHARGE=2+\n"
            "100.0 50\n"
            "200.0 100\n"
            "END IONS\n"
            # Missing PEPMASS => raises and is skipped.
            "BEGIN IONS\n"
            "SCANS=2\n"
            "CHARGE=2+\n"
            "100.0 50\n"
            "200.0 100\n"
            "END IONS\n"
            "BEGIN IONS\n"
            "SCANS=3\n"
            "PEPMASS=600.0\n"
            "CHARGE=3+\n"
            "100.0 50\n"
            "200.0 100\n"
            "END IONS\n"
        )
        from falcon.ms_io import mgf_io

        spectra = list(mgf_io.get_spectra(str(mgf)))
        assert [s.identifier for s in spectra] == [
            "multi:scan:1",
            "multi:scan:3",
        ]


# ---------------------------------------------------------------------------
# MGF write (_spectra_to_dicts) — optional fields
# ---------------------------------------------------------------------------


class TestSpectraToDicts:
    def _consensus(self, charge, rt):
        from falcon.cluster.cluster import ConsensusTuple

        return ConsensusTuple(
            precursor_mz=np.float32(500.0),
            precursor_charge=charge,
            mz=np.array([100.0, 200.0], dtype=np.float32),
            intensity=np.array([0.5, 1.0], dtype=np.float32),
            retention_time=rt,
            cluster_id=np.int32(3),
            mz_split=np.int32(0),
        )

    def test_full_fields(self):
        from falcon.ms_io import mgf_io

        dicts = list(
            mgf_io._spectra_to_dicts(
                [self._consensus(np.int32(2), np.float32(120.0))]
            )
        )
        params = dicts[0]["params"]
        assert params["title"] == "falcon:cluster:3"
        assert params["charge"] == 2
        assert params["rtinseconds"] == pytest.approx(120.0)

    def test_nan_charge_omitted(self):
        from falcon.ms_io import mgf_io

        dicts = list(
            mgf_io._spectra_to_dicts(
                [self._consensus(np.nan, np.float32(120.0))]
            )
        )
        params = dicts[0]["params"]
        assert "charge" not in params
        assert params["rtinseconds"] == pytest.approx(120.0)

    def test_nan_rt_omitted(self):
        from falcon.ms_io import mgf_io

        dicts = list(
            mgf_io._spectra_to_dicts(
                [self._consensus(np.int32(2), np.float32("nan"))]
            )
        )
        params = dicts[0]["params"]
        assert "rtinseconds" not in params
        assert params["charge"] == 2

    def test_nan_rt_roundtrip(self, tmp_path):
        """A NaN RT written to MGF is read back as NaN."""
        from falcon.ms_io import mgf_io

        out_path = str(tmp_path / "nanrt.mgf")
        mgf_io.write_spectra(
            out_path, [self._consensus(np.int32(2), np.float32("nan"))]
        )
        spectra = list(mgf_io.get_spectra(out_path))
        assert len(spectra) == 1
        assert math.isnan(spectra[0].retention_time)


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
