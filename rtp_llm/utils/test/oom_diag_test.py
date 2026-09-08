import os
import pickle
import shutil
import tempfile
import threading
import time
import unittest
from unittest import mock

import torch

from rtp_llm.utils import oom_diag


def _reset_module_state() -> None:
    oom_diag._installed = False
    oom_diag._oom_fired = False
    oom_diag._last_observer_dump = None
    oom_diag._dump_counter = 0


def _sample_snapshot() -> dict:
    mib = 1024**2
    return {
        "allocator_settings": {"max_split_size": 64 * mib},
        "device_traces": [[{"action": "alloc"}, {"action": "free_completed"}]],
        "segments": [
            {
                "device": 2,
                "address": 0x1000,
                "total_size": 4 * mib,
                "allocated_size": mib,
                "active_size": mib,
                "requested_size": 768 * 1024,
                "stream": 7,
                "segment_type": "large",
                "segment_pool_id": (0, 1),
                "is_expandable": False,
                "frames": [],
                "blocks": [
                    {
                        "address": 0x1000,
                        "size": mib,
                        "requested_size": 768 * 1024,
                        "state": "active_allocated",
                        "frames": [
                            {
                                "name": "model_forward",
                                "filename": "model.py",
                                "line": 42,
                            }
                        ],
                    },
                    {
                        "address": 0x1000 + mib,
                        "size": 3 * mib,
                        "requested_size": 0,
                        "state": "inactive",
                        "frames": [],
                    },
                ],
            }
        ],
    }


class OomDiagUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="oom_diag_test_")
        self.saved_record = os.environ.get(oom_diag._RECORD_ENV)
        self.saved_log_dir = os.environ.get(oom_diag._LOG_DIR_ENV)
        self.saved_retention = {
            name: os.environ.get(name)
            for name in (
                oom_diag._DUMP_MAX_COUNT_ENV,
                oom_diag._DUMP_MAX_AGE_SECONDS_ENV,
                oom_diag._DUMP_MAX_BYTES_ENV,
            )
        }
        os.environ.pop(oom_diag._RECORD_ENV, None)
        for name in self.saved_retention:
            os.environ.pop(name, None)
        os.environ[oom_diag._LOG_DIR_ENV] = self.tmp
        _reset_module_state()

    def tearDown(self) -> None:
        if self.saved_record is None:
            os.environ.pop(oom_diag._RECORD_ENV, None)
        else:
            os.environ[oom_diag._RECORD_ENV] = self.saved_record
        if self.saved_log_dir is None:
            os.environ.pop(oom_diag._LOG_DIR_ENV, None)
        else:
            os.environ[oom_diag._LOG_DIR_ENV] = self.saved_log_dir
        for name, value in self.saved_retention.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        _reset_module_state()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _cuda_diagnostics(self):
        return (
            mock.patch.object(torch.cuda, "current_device", return_value=2),
            mock.patch.object(torch.cuda, "device_count", return_value=3),
            mock.patch.object(
                torch.cuda, "mem_get_info", return_value=(1024**3, 8 * 1024**3)
            ),
            mock.patch.object(
                torch.cuda,
                "memory_stats",
                return_value={"allocated_bytes.all.current": 3 * 1024**3},
            ),
            mock.patch.object(
                torch.cuda, "memory_summary", return_value="FULL ALLOCATOR TABLE"
            ),
            mock.patch.object(
                torch.cuda.memory, "_snapshot", return_value=_sample_snapshot()
            ),
        )

    def test_install_preserves_oom_observer_and_is_idempotent(self) -> None:
        os.environ[oom_diag._RECORD_ENV] = "1"
        with mock.patch.object(
            torch.cuda.memory, "_record_memory_history"
        ) as record, mock.patch.object(
            torch._C, "_cuda_attach_out_of_memory_observer", create=True
        ) as attach, mock.patch.object(
            torch.cuda, "current_device", return_value=3
        ):
            oom_diag.install_oom_dump()
            oom_diag.install_oom_dump()

        record.assert_called_once_with(
            enabled="all",
            context="all",
            stacks="all",
            max_entries=oom_diag._MAX_TRACE_ENTRIES,
        )
        attach.assert_called_once_with(oom_diag._oom_observer)

    def test_install_is_disabled_by_default(self) -> None:
        with mock.patch.object(
            torch.cuda.memory, "_record_memory_history"
        ) as record, mock.patch.object(
            torch._C, "_cuda_attach_out_of_memory_observer", create=True
        ) as attach:
            oom_diag.install_oom_dump()
        record.assert_not_called()
        attach.assert_not_called()

    def test_dump_writes_all_device_allocator_summary(self) -> None:
        patches = self._cuda_diagnostics()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            output_path = oom_diag.dump_oom_diagnostics(
                tag="fatal_gpu_oom",
                alloc_size=256 * 1024**2,
                exception="CUDA out of memory; original marker",
                cpp_backtrace="CUDAGraph.cpp:208 original backtrace",
            )

        self.assertIsNotNone(output_path)
        text = open(output_path, encoding="utf-8").read()
        self.assertIn("tag=fatal_gpu_oom", text)
        self.assertIn("CUDA out of memory; original marker", text)
        self.assertIn("CUDAGraph.cpp:208 original backtrace", text)
        self.assertIn("logical_devices=[0, 1, 2]", text)
        self.assertEqual(text.count("FULL ALLOCATOR TABLE"), 3)
        self.assertIn("allocated_bytes.all.current=3221225472 (3.00 GiB)", text)
        self.assertIn("segments=1 blocks=2", text)
        self.assertIn("usage=ACTIVE_ALLOCATED", text)
        self.assertIn("usage=CACHED_FREE", text)
        self.assertIn("model_forward at model.py:42", text)

    def test_observer_remains_one_shot_but_manual_dumps_repeat(self) -> None:
        patches = self._cuda_diagnostics()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            oom_diag._oom_observer(2, 4096, 8 * 1024**3, 1024**3)
            first_path = oom_diag._last_observer_dump
            oom_diag._oom_observer(2, 8192, 8 * 1024**3, 512 * 1024**2)
            reused_path = oom_diag.dump_oom_diagnostics(
                tag="fatal_gpu_oom", device=2, reuse_observer_dump=True
            )
            manual_path = oom_diag.dump_oom_diagnostics(tag="allocator_dump", device=2)

        self.assertEqual(reused_path, first_path)
        self.assertNotEqual(manual_path, first_path)
        self.assertEqual(len(list(oom_diag._out_dir().glob("oom_allocator_*.log"))), 2)

    def test_repeated_manual_dumps_use_incrementing_ids(self) -> None:
        patches = self._cuda_diagnostics()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            first = oom_diag.dump_oom_diagnostics(device=2)
            second = oom_diag.dump_oom_diagnostics(device=2)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertTrue(first.endswith("_n000000.log"))
        self.assertTrue(second.endswith("_n000001.log"))

    def test_dump_correlation_id_is_in_filename_and_log(self) -> None:
        patches = self._cuda_diagnostics()
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            output_path = oom_diag.dump_oom_diagnostics(
                tag="allocator_dump",
                device=2,
                dump_correlation_id="opaque-id-123",
            )

        self.assertIsNotNone(output_path)
        self.assertIn("_cidopaque-id-123_", output_path)
        self.assertIn("dump_id=opaque-id-123", open(output_path).read())

    def test_suffix_sanitizes_every_external_component(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"WORLD_RANK": "../../rank", "FRONTEND_SERVER_ID": "../server"},
        ):
            suffix = oom_diag._suffix(
                "../tag/name", -1, 3, dump_correlation_id="../../dump/id"
            )

        self.assertNotIn("/", suffix)
        self.assertNotIn("..", suffix)
        self.assertIn("_ciddump-id_", suffix)

    def test_retention_enforces_age_count_and_bytes_for_dump_groups(self) -> None:
        output_dir = oom_diag._out_dir()

        def write_group(suffix: str, modified_at: float, sizes: tuple[int, ...]):
            paths = []
            for index, size in enumerate(sizes):
                prefix = "oom_allocator_" if index == 0 else "oom_allocator_snapshot_"
                extension = ".log" if index == 0 else ".pickle"
                path = output_dir / f"{prefix}{suffix}{extension}"
                path.write_bytes(b"x" * size)
                os.utime(path, (modified_at, modified_at))
                paths.append(path)
            return paths

        os.environ[oom_diag._DUMP_MAX_COUNT_ENV] = "2"
        os.environ[oom_diag._DUMP_MAX_AGE_SECONDS_ENV] = "100"
        os.environ[oom_diag._DUMP_MAX_BYTES_ENV] = "10"
        expired = write_group("expired", 800.0, (1, 1))
        oldest_retained = write_group("oldest", 920.0, (4, 4))
        over_byte_budget = write_group("middle", 950.0, (8,))
        newest = write_group("newest", 980.0, (8,))

        with oom_diag._dump_lock, oom_diag._retention_lock(output_dir):
            oom_diag._prune_dump_files(output_dir, now=1000.0)

        for path in expired + oldest_retained + over_byte_budget:
            self.assertFalse(path.exists())
        self.assertTrue(newest[0].exists())

    def test_retention_counts_correlated_fanout_as_one_group(self) -> None:
        output_dir = oom_diag._out_dir()
        old = output_dir / "oom_allocator_old.log"
        old.write_bytes(b"old")
        os.utime(old, (100.0, 100.0))
        fanout = []
        for rank in range(4):
            path = output_dir / (
                "oom_allocator_allocator_dump_cidrequest-1_"
                f"r{rank}_s0_d0_pid{rank}_n000000.log"
            )
            path.write_bytes(b"new")
            os.utime(path, (200.0 + rank, 200.0 + rank))
            fanout.append(path)
        os.environ[oom_diag._DUMP_MAX_COUNT_ENV] = "1"

        with oom_diag._dump_lock, oom_diag._retention_lock(output_dir):
            oom_diag._prune_dump_files(output_dir, now=250.0)

        self.assertFalse(old.exists())
        self.assertTrue(all(path.exists() for path in fanout))

    def test_retention_does_not_prune_active_locked_publication(self) -> None:
        output_dir = oom_diag._out_dir()
        suffix = "allocator_dump_cidactive-1_r0_s0_d0_pid1_n000000"
        artifact = output_dir / f"oom_allocator_{suffix}.log"
        marker = output_dir / f"{oom_diag._IN_PROGRESS_PREFIX}{suffix}"
        artifact.write_bytes(b"too large")
        marker_fd = os.open(marker, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        oom_diag.fcntl.flock(marker_fd, oom_diag.fcntl.LOCK_EX | oom_diag.fcntl.LOCK_NB)
        os.utime(artifact, (1.0, 1.0))
        os.utime(marker, (1.0, 1.0))
        os.environ[oom_diag._DUMP_MAX_AGE_SECONDS_ENV] = "10"
        os.environ[oom_diag._DUMP_MAX_BYTES_ENV] = "1"

        try:
            with oom_diag._dump_lock, oom_diag._retention_lock(output_dir):
                oom_diag._prune_dump_files(output_dir, now=100.0)
        finally:
            oom_diag.fcntl.flock(marker_fd, oom_diag.fcntl.LOCK_UN)
            os.close(marker_fd)

        self.assertTrue(marker.exists())
        self.assertTrue(artifact.exists())

    def test_retention_removes_stale_crash_marker_and_artifacts(self) -> None:
        output_dir = oom_diag._out_dir()
        suffix = "allocator_dump_cidstale-1_r0_s0_d0_pid1_n000000"
        artifact = output_dir / f"oom_allocator_{suffix}.log"
        marker = output_dir / f"{oom_diag._IN_PROGRESS_PREFIX}{suffix}"
        artifact.write_bytes(b"stale")
        marker.touch()
        os.utime(artifact, (1.0, 1.0))
        os.utime(marker, (1.0, 1.0))
        os.environ[oom_diag._DUMP_MAX_AGE_SECONDS_ENV] = "10"

        with oom_diag._dump_lock, oom_diag._retention_lock(output_dir):
            oom_diag._prune_dump_files(output_dir, now=100.0)

        self.assertFalse(marker.exists())
        self.assertFalse(artifact.exists())

    def test_concurrent_dumps_are_serialized(self) -> None:
        active = 0
        max_active = 0
        guard = threading.Lock()
        start = threading.Barrier(3)

        def snapshot():
            nonlocal active, max_active
            with guard:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with guard:
                active -= 1
            return {"segments": []}

        def dump():
            start.wait()
            oom_diag.dump_oom_diagnostics(device=0)

        with mock.patch.object(
            torch.cuda, "device_count", return_value=1
        ), mock.patch.object(
            torch.cuda, "mem_get_info", return_value=(1, 2)
        ), mock.patch.object(
            torch.cuda, "memory_stats", return_value={}
        ), mock.patch.object(
            torch.cuda, "memory_summary", return_value="summary"
        ), mock.patch.object(
            torch.cuda.memory, "_snapshot", side_effect=snapshot
        ):
            threads = [threading.Thread(target=dump) for _ in range(2)]
            for thread in threads:
                thread.start()
            start.wait()
            for thread in threads:
                thread.join()

        self.assertEqual(max_active, 1)
        self.assertEqual(len(list(oom_diag._out_dir().glob("oom_allocator_*.log"))), 2)

    def test_diagnostic_failure_is_contained(self) -> None:
        with mock.patch.object(
            torch.cuda, "current_device", return_value=0
        ), mock.patch.object(
            torch.cuda, "device_count", return_value=1
        ), mock.patch.object(
            torch.cuda, "mem_get_info", side_effect=RuntimeError("query failed")
        ), mock.patch.object(
            torch.cuda, "memory_stats", return_value={}
        ), mock.patch.object(
            torch.cuda, "memory_summary", return_value="summary"
        ), mock.patch.object(
            torch.cuda.memory, "_snapshot", return_value={}
        ), mock.patch.object(
            oom_diag.tempfile,
            "NamedTemporaryFile",
            side_effect=OSError("dump file failed"),
        ), self.assertLogs(
            oom_diag._LOG, level="ERROR"
        ) as logs:
            result = oom_diag.dump_oom_diagnostics()

        self.assertIsNone(result)
        self.assertIn("dump file failed", "\n".join(logs.output))

    def test_suffix_includes_rank_device_pid_and_counter(self) -> None:
        suffix = oom_diag._suffix("test", 7, 12)
        self.assertTrue(suffix.startswith("test_r"))
        self.assertIn("_s", suffix)
        self.assertIn("_d7_", suffix)
        self.assertIn(f"_pid{os.getpid()}_", suffix)
        self.assertTrue(suffix.endswith("_n000012"))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA or HIP required")
class OomDiagAcceleratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="oom_diag_accelerator_test_")
        self.saved_record = os.environ.get(oom_diag._RECORD_ENV)
        self.saved_log_dir = os.environ.get(oom_diag._LOG_DIR_ENV)
        os.environ[oom_diag._RECORD_ENV] = "1"
        os.environ[oom_diag._LOG_DIR_ENV] = self.tmp
        _reset_module_state()
        torch.cuda.memory._record_memory_history(enabled=None)

    def tearDown(self) -> None:
        torch.cuda.memory._record_memory_history(enabled=None)
        if self.saved_record is None:
            os.environ.pop(oom_diag._RECORD_ENV, None)
        else:
            os.environ[oom_diag._RECORD_ENV] = self.saved_record
        if self.saved_log_dir is None:
            os.environ.pop(oom_diag._LOG_DIR_ENV, None)
        else:
            os.environ[oom_diag._LOG_DIR_ENV] = self.saved_log_dir
        _reset_module_state()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_observer_writes_text_and_memory_viz_snapshot(self) -> None:
        oom_diag.install_oom_dump()
        tensor = torch.zeros(8192, dtype=torch.float32, device="cuda")
        try:
            oom_diag._oom_observer(
                device=torch.cuda.current_device(),
                alloc_size=4096,
                device_total=80 * 1024**3,
                device_free=1024,
            )
        finally:
            del tensor
            torch.cuda.empty_cache()

        logs = list(oom_diag._out_dir().glob("oom_allocator_*.log"))
        snapshots = list(oom_diag._out_dir().glob("oom_allocator_snapshot_*.pickle"))
        self.assertEqual(len(logs), 1)
        self.assertEqual(len(snapshots), 1)
        text = logs[0].read_text(encoding="utf-8")
        self.assertIn("[TORCH MEMORY SUMMARY - ALL LOGICAL DEVICES]", text)
        self.assertIn("BLOCK[", text)
        self.assertIn("allocation_frames=", text)
        with snapshots[0].open("rb") as snapshot_file:
            snapshot = pickle.load(snapshot_file)
        self.assertIn("segments", snapshot)


if __name__ == "__main__":
    unittest.main()
