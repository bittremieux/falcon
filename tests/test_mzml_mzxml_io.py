"""Tests for falcon.ms_io.mzml_io and mzxml_io — P2: XML peak file readers."""

import math
import os

import numpy as np
import pytest

from falcon.ms_io import mzml_io, mzxml_io

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ---------------------------------------------------------------------------
# mzml_io._parse_spectrum
# ---------------------------------------------------------------------------


def _mzml_dict(scan=None, ion_extra=None):
    """Build a minimal Pyteomics-style mzML spectrum dict."""
    return {
        "id": "run:scan:5",
        "m/z array": np.array([100.0, 200.0], dtype=np.float32),
        "intensity array": np.array([1.0, 2.0], dtype=np.float32),
        "scanList": {"scan": [scan if scan is not None else {}]},
        "precursorList": {
            "precursor": [
                {
                    "selectedIonList": {
                        "selectedIon": [
                            {"selected ion m/z": 500.0, **(ion_extra or {})}
                        ]
                    }
                }
            ]
        },
    }


class TestMzmlParseSpectrum:
    def test_missing_rt_is_nan(self):
        """No 'scan start time' should yield a NaN retention time."""
        spec = mzml_io._parse_spectrum(_mzml_dict(ion_extra={"charge state": 2}))
        assert math.isnan(spec.retention_time)
        assert spec.precursor_mz == 500.0
        assert spec.precursor_charge == 2

    def test_rt_present(self):
        spec = mzml_io._parse_spectrum(
            _mzml_dict(
                scan={"scan start time": 6.0}, ion_extra={"charge state": 2}
            )
        )
        assert spec.retention_time == pytest.approx(6.0)

    def test_possible_charge_state_fallback(self):
        """'possible charge state' is used when 'charge state' is absent."""
        spec = mzml_io._parse_spectrum(
            _mzml_dict(ion_extra={"possible charge state": 3})
        )
        assert spec.precursor_charge == 3

    def test_no_charge(self):
        spec = mzml_io._parse_spectrum(_mzml_dict())
        assert spec.precursor_charge is None


# ---------------------------------------------------------------------------
# mzxml_io._parse_spectrum
# ---------------------------------------------------------------------------


def _mzxml_dict(extra_precursor=None, rt=None):
    """Build a minimal Pyteomics-style mzXML spectrum dict."""
    precursor = {"precursorMz": 400.0, **(extra_precursor or {})}
    d = {
        "id": "run:scan:7",
        "m/z array": np.array([100.0, 200.0], dtype=np.float32),
        "intensity array": np.array([1.0, 2.0], dtype=np.float32),
        "precursorMz": [precursor],
    }
    if rt is not None:
        d["retentionTime"] = rt
    return d


class TestMzxmlParseSpectrum:
    def test_missing_rt_is_nan(self):
        spec = mzxml_io._parse_spectrum(_mzxml_dict())
        assert math.isnan(spec.retention_time)
        assert spec.precursor_mz == 400.0
        assert spec.precursor_charge is None

    def test_rt_present(self):
        spec = mzxml_io._parse_spectrum(_mzxml_dict(rt=6.0))
        assert spec.retention_time == pytest.approx(6.0)

    def test_charge(self):
        spec = mzxml_io._parse_spectrum(
            _mzxml_dict(extra_precursor={"precursorCharge": 2})
        )
        assert spec.precursor_charge == 2


# ---------------------------------------------------------------------------
# get_spectra over minimal real files (MS-level filtering + identifiers)
#
# The fixture files in tests/data each contain one MS1 and one MS2 scan. Each
# peak list encodes a single (m/z=100.0, intensity=10.0) peak.
# ---------------------------------------------------------------------------


class TestMzmlGetSpectra:
    def test_filters_ms1_and_builds_identifier(self):
        path = os.path.join(DATA_DIR, "sample.mzml")
        spectra = list(mzml_io.get_spectra(path))
        # Only the MS2 spectrum is returned.
        assert len(spectra) == 1
        s = spectra[0]
        assert s.identifier == "sample:scan:2"
        assert s.precursor_mz == pytest.approx(400.0)
        assert s.precursor_charge == 2
        assert s.retention_time == pytest.approx(6.0)


class TestMzxmlGetSpectra:
    def test_filters_ms1_and_builds_identifier(self):
        path = os.path.join(DATA_DIR, "sample.mzxml")
        spectra = list(mzxml_io.get_spectra(path))
        assert len(spectra) == 1
        s = spectra[0]
        assert s.identifier == "sample:scan:2"
        assert s.precursor_mz == pytest.approx(400.0)
        assert s.precursor_charge == 2
