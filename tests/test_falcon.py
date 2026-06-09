"""Tests for charge-bucket functionality in falcon.falcon."""

import collections
import queue
import threading

import lance
import numpy as np
import pyarrow as pa
import pytest

from falcon.falcon import (
    _PerChargeLockRegistry,
    _discover_auto_buckets,
    _write_spectra_lance,
    _write_to_dataset,
    bucket_key_to_str,
)
from falcon.config import config as _global_config

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def work_dir(tmp_path):
    """Set config.work_dir to a temp directory for tests that call into lance writers.

    Config.__getattr__ raises RuntimeError when uninitialized, so we set the
    attribute directly in __dict__ to bypass that guard.
    """
    (tmp_path / "spectra").mkdir()
    _global_config.__dict__["work_dir"] = str(tmp_path)
    yield tmp_path
    del _global_config.__dict__["work_dir"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCHEMA = pa.schema(
    [
        pa.field("identifier", pa.string()),
        pa.field("precursor_mz", pa.float32()),
        pa.field("precursor_charge", pa.int8()),
        pa.field("mz", pa.list_(pa.float32())),
        pa.field("intensity", pa.list_(pa.float32())),
        pa.field("retention_time", pa.float32()),
    ]
)


def _make_spec(identifier, charge):
    mz = np.array([100.0, 200.0, 300.0], dtype=np.float32)
    intensity = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    intensity /= np.linalg.norm(intensity)
    return {
        "identifier": identifier,
        "precursor_mz": np.float32(500.0),
        "precursor_charge": charge,
        "mz": mz,
        "intensity": intensity,
        "retention_time": np.float32(0.0),
    }


def _run_write_worker(
    spectra,
    charge_to_bucket,
    catch_other_charges,
    work_dir,
    auto_bucket=False,
):
    """Push spectra through _write_spectra_lance and return a dict of row counts.

    Requires the ``work_dir`` fixture to have set config.work_dir already.
    """
    q = queue.Queue()
    for spec in spectra:
        q.put(spec)
    q.put(None)  # sentinel

    locks = _PerChargeLockRegistry()
    _write_spectra_lance(
        q, locks, SCHEMA, charge_to_bucket, catch_other_charges, auto_bucket
    )

    spectra_dir = work_dir / "spectra"
    counts = {}
    if spectra_dir.exists():
        for p in spectra_dir.iterdir():
            if p.suffix == ".lance":
                counts[p.name] = lance.dataset(str(p)).count_rows()
    return counts


# ---------------------------------------------------------------------------
# bucket_key_to_str
# ---------------------------------------------------------------------------


class TestBucketKeyToStr:
    def test_single_int_tuple(self):
        assert bucket_key_to_str((1,)) == "_1"

    def test_multi_int_tuple(self):
        assert bucket_key_to_str((1, 2)) == "_1_2"

    def test_tuple_with_unknown_str(self):
        assert bucket_key_to_str((3, "unknown")) == "_3_unknown"

    def test_string_other(self):
        assert bucket_key_to_str("other") == "_other"

    def test_string_unknown(self):
        assert bucket_key_to_str("unknown") == "_unknown"

    def test_set_sorted_numerically(self):
        # Sets are unordered; the function must sort them
        assert bucket_key_to_str({2, 1}) == "_1_2"

    def test_set_with_unknown_sorts_after_ints(self):
        # str sorts after int: 3 then "unknown"
        assert bucket_key_to_str({"unknown", 3}) == "_3_unknown"

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            bucket_key_to_str(42)


# ---------------------------------------------------------------------------
# _write_to_dataset
# ---------------------------------------------------------------------------


class TestWriteToDataset:
    def test_creates_dataset_at_correct_path(self, work_dir):
        lock = threading.Lock()
        _write_to_dataset(
            [_make_spec("s1", 2)], (2,), lock, SCHEMA, str(work_dir)
        )
        assert (work_dir / "spectra" / "spectra_charge_2.lance").exists()

    def test_returns_correct_row_count(self, work_dir):
        lock = threading.Lock()
        rows = [_make_spec(f"s{i}", 2) for i in range(5)]
        count = _write_to_dataset(rows, (2,), lock, SCHEMA, str(work_dir))
        assert count == 5

    def test_appends_to_existing_dataset(self, work_dir):
        lock = threading.Lock()
        _write_to_dataset(
            [_make_spec("s1", 2)], (2,), lock, SCHEMA, str(work_dir)
        )
        _write_to_dataset(
            [_make_spec("s2", 2)], (2,), lock, SCHEMA, str(work_dir)
        )
        ds = lance.dataset(
            str(work_dir / "spectra" / "spectra_charge_2.lance")
        )
        assert ds.count_rows() == 2


# ---------------------------------------------------------------------------
# Charge routing via _write_spectra_lance
# ---------------------------------------------------------------------------


class TestChargeRouting:
    def test_single_charge_bucket(self, work_dir):
        charge_to_bucket = {1: (1,)}
        counts = _run_write_worker(
            [_make_spec("s1", 1), _make_spec("s2", 1)],
            charge_to_bucket,
            catch_other_charges=False,
            work_dir=work_dir,
        )
        assert counts == {"spectra_charge_1.lance": 2}

    def test_multiple_single_charge_buckets(self, work_dir):
        charge_to_bucket = {1: (1,), 2: (2,)}
        spectra = [_make_spec("a", 1), _make_spec("b", 2), _make_spec("c", 1)]
        counts = _run_write_worker(
            spectra,
            charge_to_bucket,
            catch_other_charges=False,
            work_dir=work_dir,
        )
        assert counts["spectra_charge_1.lance"] == 2
        assert counts["spectra_charge_2.lance"] == 1

    def test_multi_charge_bucket_groups_charges_together(self, work_dir):
        # Charges 1 and 2 share a single bucket keyed (1, 2)
        charge_to_bucket = {1: (1, 2), 2: (1, 2)}
        spectra = [_make_spec("a", 1), _make_spec("b", 2)]
        counts = _run_write_worker(
            spectra,
            charge_to_bucket,
            catch_other_charges=False,
            work_dir=work_dir,
        )
        assert counts == {"spectra_charge_1_2.lance": 2}

    def test_unknown_charge_routed_to_unknown_bucket(self, work_dir):
        charge_to_bucket = {"unknown": ("unknown",)}
        spectra = [_make_spec("u1", None)]
        counts = _run_write_worker(
            spectra,
            charge_to_bucket,
            catch_other_charges=False,
            work_dir=work_dir,
        )
        assert counts == {"spectra_charge_unknown.lance": 1}

    def test_unmapped_charge_goes_to_other_when_enabled(self, work_dir):
        charge_to_bucket = {1: (1,)}  # charge 3 has no explicit bucket
        spectra = [_make_spec("x", 3)]
        counts = _run_write_worker(
            spectra,
            charge_to_bucket,
            catch_other_charges=True,
            work_dir=work_dir,
        )
        assert counts == {"spectra_charge_other.lance": 1}

    def test_unmapped_charge_dropped_when_other_disabled(self, work_dir):
        charge_to_bucket = {1: (1,)}
        spectra = [_make_spec("x", 3)]
        counts = _run_write_worker(
            spectra,
            charge_to_bucket,
            catch_other_charges=False,
            work_dir=work_dir,
        )
        assert counts == {}

    def test_mixed_routing(self, work_dir):
        charge_to_bucket = {1: (1,), 2: (2,), "unknown": ("unknown",)}
        spectra = [
            _make_spec("a", 1),
            _make_spec("b", 2),
            _make_spec("c", None),  # unknown charge
            _make_spec("d", 5),  # not in any bucket
        ]
        counts = _run_write_worker(
            spectra,
            charge_to_bucket,
            catch_other_charges=True,
            work_dir=work_dir,
        )
        assert counts["spectra_charge_1.lance"] == 1
        assert counts["spectra_charge_2.lance"] == 1
        assert counts["spectra_charge_unknown.lance"] == 1
        assert counts["spectra_charge_other.lance"] == 1


# ---------------------------------------------------------------------------
# Auto-bucket mode: every distinct charge clustered separately
# ---------------------------------------------------------------------------


class TestAutoBucketRouting:
    def test_each_charge_gets_own_bucket(self, work_dir):
        spectra = [
            _make_spec("a", 1),
            _make_spec("b", 2),
            _make_spec("c", 1),
            _make_spec("d", 3),
        ]
        counts = _run_write_worker(
            spectra,
            charge_to_bucket={},
            catch_other_charges=False,
            work_dir=work_dir,
            auto_bucket=True,
        )
        assert counts == {
            "spectra_charge_1.lance": 2,
            "spectra_charge_2.lance": 1,
            "spectra_charge_3.lance": 1,
        }

    def test_missing_charge_clustered_separately(self, work_dir):
        spectra = [_make_spec("a", 2), _make_spec("u", None)]
        counts = _run_write_worker(
            spectra,
            charge_to_bucket={},
            catch_other_charges=False,
            work_dir=work_dir,
            auto_bucket=True,
        )
        assert counts == {
            "spectra_charge_2.lance": 1,
            "spectra_charge_unknown.lance": 1,
        }

    def test_no_charge_is_dropped(self, work_dir):
        # Charges that would be dropped without 'other' are still captured.
        spectra = [_make_spec("x", 7), _make_spec("y", 13)]
        counts = _run_write_worker(
            spectra,
            charge_to_bucket={},
            catch_other_charges=False,
            work_dir=work_dir,
            auto_bucket=True,
        )
        assert counts == {
            "spectra_charge_7.lance": 1,
            "spectra_charge_13.lance": 1,
        }


# ---------------------------------------------------------------------------
# _discover_auto_buckets
# ---------------------------------------------------------------------------


class TestDiscoverAutoBuckets:
    def test_discovers_and_sorts_buckets(self, work_dir):
        spectra = [
            _make_spec("a", 3),
            _make_spec("b", 1),
            _make_spec("c", None),
            _make_spec("d", 2),
        ]
        _run_write_worker(
            spectra,
            charge_to_bucket={},
            catch_other_charges=False,
            work_dir=work_dir,
            auto_bucket=True,
        )
        spectra_dir = str(work_dir / "spectra")
        # Numeric charges sorted ascending, "unknown" last.
        assert _discover_auto_buckets(spectra_dir) == [
            (1,),
            (2,),
            (3,),
            ("unknown",),
        ]
