"""Unit tests for the parent-side cudacore collection lease reader."""

import json
import os
import socket
import tempfile
import time
import unittest
from unittest.mock import patch

from rtp_llm.utils import cudacore_lease

from rtp_llm.utils.cudacore_lease import (
    CUDACORE_LEASE_PREFIX,
    cudacore_diagnostics_dir,
    remaining_collection_seconds,
    wait_for_collection_leases,
)


class CudacoreLeaseTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self._saved_env = os.environ.pop("CUDA_COREDUMP_FILE", None)
        self.processes = {}
        self.real_process_identity = cudacore_lease._process_identity
        self.identity_patch = patch.object(
            cudacore_lease, "_process_identity", side_effect=self.processes.get
        )
        self.identity_patch.start()
        self.addCleanup(self.identity_patch.stop)

    def tearDown(self):
        if self._saved_env is not None:
            os.environ["CUDA_COREDUMP_FILE"] = self._saved_env
        self._tmp.cleanup()

    def _write_lease(self, name, pid, deadline_epoch_ms, **extra):
        self.processes.setdefault(pid, cudacore_lease._ProcessIdentity(0, 12345))
        lease = {
            "schema_version": "rtp_llm.cudacore_lease.v1",
            "host": socket.gethostname(),
            "pid": pid,
            "worker_start_id": f"{pid}-12345",
            "rank": 1,
            "created_epoch_ms": deadline_epoch_ms - 30000,
            "deadline_epoch_ms": deadline_epoch_ms,
            "window_ms": 30000,
        }
        lease.update(extra)
        with open(os.path.join(self.dir, name), "w", encoding="utf-8") as handle:
            json.dump(lease, handle)

    def test_remaining_seconds_uses_the_workers_original_deadline(self):
        now = time.time()
        self._write_lease(f"{CUDACORE_LEASE_PREFIX}a", 1234, int((now + 12.0) * 1000))

        remaining = remaining_collection_seconds([1234], self.dir, now=now)

        self.assertAlmostEqual(remaining, 12.0, places=1)

    def test_stale_foreign_and_invalid_leases_are_ignored(self):
        now = time.time()
        self._write_lease(f"{CUDACORE_LEASE_PREFIX}expired", 1234, int((now - 5.0) * 1000))
        self._write_lease(f"{CUDACORE_LEASE_PREFIX}foreign", 9999, int((now + 30.0) * 1000))
        with open(os.path.join(self.dir, f"{CUDACORE_LEASE_PREFIX}broken"), "w") as handle:
            handle.write("{not json")
        with open(os.path.join(self.dir, "cudacore_incident.other"), "w") as handle:
            handle.write("{}")

        self.assertEqual(remaining_collection_seconds([1234], self.dir, now=now), 0.0)

    def test_takes_the_largest_window_across_workers(self):
        now = time.time()
        self._write_lease(f"{CUDACORE_LEASE_PREFIX}a", 1, int((now + 3.0) * 1000))
        self._write_lease(f"{CUDACORE_LEASE_PREFIX}b", 2, int((now + 9.0) * 1000))

        self.assertAlmostEqual(
            remaining_collection_seconds([1, 2], self.dir, now=now), 9.0, places=1
        )

    def test_wait_returns_immediately_without_a_lease(self):
        started = time.time()
        waited = wait_for_collection_leases([4242], self.dir, reason="test")
        self.assertEqual(waited, 0.0)
        self.assertLess(time.time() - started, 0.5)

    def test_wait_is_bounded_by_the_lease_deadline(self):
        now = time.time()
        self._write_lease(f"{CUDACORE_LEASE_PREFIX}a", 4242, int((now + 0.4) * 1000))

        started = time.time()
        waited = wait_for_collection_leases([4242], self.dir, reason="test")
        elapsed = time.time() - started

        self.assertGreaterEqual(waited, 0.3)
        self.assertLess(elapsed, 5.0)

    def test_missing_directory_is_not_an_error(self):
        self.assertEqual(
            remaining_collection_seconds([1], os.path.join(self.dir, "nope")), 0.0
        )
        self.assertEqual(
            wait_for_collection_leases([1], os.path.join(self.dir, "nope")), 0.0
        )

    def test_outer_manager_recognizes_rank_grandchild_but_not_unrelated_worker(self):
        now = time.time()
        self.processes[100] = cudacore_lease._ProcessIdentity(1, 1000)
        self.processes[200] = cudacore_lease._ProcessIdentity(100, 2000)
        self.processes[300] = cudacore_lease._ProcessIdentity(200, 3000)
        self._write_lease(
            CUDACORE_LEASE_PREFIX + "rank", 300, int((now + 9) * 1000),
            worker_start_id="300-3000",
        )
        self.assertAlmostEqual(remaining_collection_seconds([100], self.dir, now), 9, places=1)
        self.assertAlmostEqual(remaining_collection_seconds([200], self.dir, now), 9, places=1)
        self.processes[400] = cudacore_lease._ProcessIdentity(1, 4000)
        self.assertEqual(remaining_collection_seconds([400], self.dir, now), 0)

    def test_reused_pid_foreign_host_and_invalid_window_are_ignored(self):
        now = time.time()
        for extra in (
            {"worker_start_id": "1234-old"},
            {"host": "another-host"},
            {"schema_version": "unknown"},
            {"window_ms": 60000},
            {"pid": True},
            {"deadline_epoch_ms": "not-an-integer"},
        ):
            with self.subTest(extra=extra):
                self._write_lease(CUDACORE_LEASE_PREFIX + "bad", 1234,
                                  int((now + 9) * 1000))
                path = os.path.join(self.dir, CUDACORE_LEASE_PREFIX + "bad")
                with open(path) as handle:
                    lease = json.load(handle)
                lease.update(extra)
                with open(path, "w") as handle:
                    json.dump(lease, handle)
                self.assertEqual(remaining_collection_seconds([1234], self.dir, now), 0)

    def test_wrong_json_shapes_do_not_escape_cleanup(self):
        self.processes[1234] = cudacore_lease._ProcessIdentity(0, 12345)
        for value in ([], None, "text", 42):
            with self.subTest(value=value):
                with open(os.path.join(self.dir, CUDACORE_LEASE_PREFIX + "bad"), "w") as handle:
                    json.dump(value, handle)
                self.assertEqual(remaining_collection_seconds([1234], self.dir), 0)

    def test_monotonic_lease_is_independent_of_wall_clock(self):
        now = time.time()
        mono = time.monotonic()
        self._write_lease(
            CUDACORE_LEASE_PREFIX + "mono", 1234, int((now + 5) * 1000),
            created_mono_ms=int(mono * 1000) - 25000,
            deadline_mono_ms=int(mono * 1000) + 5000,
        )
        with patch.object(cudacore_lease.time, "monotonic", return_value=mono):
            self.assertAlmostEqual(
                remaining_collection_seconds([1234], self.dir, now + 3600), 5, places=1
            )

    def test_parent_hard_cap_uses_monotonic_and_materializes_pid_iterable(self):
        self.processes[1234] = cudacore_lease._ProcessIdentity(0, 12345)
        clock = [1000.0]
        seen_roots = []

        def remaining(roots, *args):
            seen_roots.append(set(roots))
            return 100.0

        with patch.object(cudacore_lease, "_remaining_for_roots", side_effect=remaining), \
             patch.object(cudacore_lease.time, "monotonic", side_effect=lambda: clock[0]), \
             patch.object(cudacore_lease.time, "time", return_value=100.0), \
             patch.object(cudacore_lease.time, "sleep", side_effect=lambda n: clock.__setitem__(0, clock[0] + n)):
            waited = wait_for_collection_leases(iter([1234]), self.dir)
        self.assertEqual(waited, 35.0)
        self.assertTrue(all(roots == {1234} for roots in seen_roots))

    def test_real_proc_stat_reader_matches_current_process(self):
        identity = self.real_process_identity(os.getpid())
        self.assertIsNotNone(identity)
        self.assertEqual(identity.parent_pid, os.getppid())
        self.assertGreater(identity.start_ticks, 0)

    def test_diagnostics_dir_follows_the_coredump_file_parent(self):
        os.environ["CUDA_COREDUMP_FILE"] = os.path.join(self.dir, "prefill.%h.%p.%t")
        self.assertEqual(cudacore_diagnostics_dir(), self.dir)

        os.environ["CUDA_COREDUMP_FILE"] = "|/usr/bin/cat > /tmp/out"
        self.assertTrue(cudacore_diagnostics_dir().endswith("cudacore_diagnostics"))


if __name__ == "__main__":
    unittest.main()
