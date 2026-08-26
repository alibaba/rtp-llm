import os
import pickle
import shutil
import tempfile
import unittest
from unittest import mock

import torch

from rtp_llm.utils import oom_diag


def _reset_module_state() -> None:
    oom_diag._installed = False
    oom_diag._oom_fired = False


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
            },
            {
                "device": 2,
                "address": 0x800000,
                "total_size": 2 * mib,
                "allocated_size": 2 * mib,
                "active_size": 2 * mib,
                "requested_size": 2 * mib,
                "stream": 9,
                "segment_type": "small",
                "segment_pool_id": (0, 1),
                "is_expandable": True,
                "frames": [],
                "blocks": [
                    {
                        "address": 0x800000,
                        "size": 2 * mib,
                        "requested_size": 2 * mib,
                        "state": "active_awaiting_free",
                        "frames": [],
                    }
                ],
            },
        ],
    }


class OriginalOomError(RuntimeError):
    pass


class OomDiagUnitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="oom_diag_test_")
        self.saved_record = os.environ.get(oom_diag._RECORD_ENV)
        self.saved_log_dir = os.environ.get(oom_diag._LOG_DIR_ENV)
        os.environ.pop(oom_diag._RECORD_ENV, None)
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
        _reset_module_state()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_install_is_disabled_by_default(self) -> None:
        with mock.patch.object(torch.cuda.memory, "_record_memory_history") as record:
            oom_diag.install_oom_dump()
        record.assert_not_called()
        self.assertFalse(oom_diag._installed)

    def test_install_enables_history_and_observer_once(self) -> None:
        os.environ[oom_diag._RECORD_ENV] = "1"
        with (
            mock.patch.object(torch.cuda.memory, "_record_memory_history") as record,
            mock.patch.object(
                torch._C, "_cuda_attach_out_of_memory_observer"
            ) as attach,
            mock.patch.object(torch.cuda, "current_device", return_value=3),
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

    def test_dump_directory_follows_service_log_path(self) -> None:
        service_log_dir = os.path.join(self.tmp, "service_logs")
        os.environ[oom_diag._LOG_DIR_ENV] = service_log_dir
        self.assertEqual(str(oom_diag._out_dir()), service_log_dir)
        os.environ[oom_diag._LOG_DIR_ENV] = self.tmp

    def test_dump_logs_and_writes_full_human_readable_summary(self) -> None:
        stats = {
            "allocated_bytes.all.current": 3 * 1024**3,
            "reserved_bytes.all.current": 4 * 1024**3,
            "inactive_split_bytes.all.current": 512 * 1024**2,
            "num_alloc_retries": 7,
            "num_ooms": 2,
        }
        with (
            mock.patch.object(torch.cuda, "current_device", return_value=2),
            mock.patch.object(
                torch.cuda, "mem_get_info", return_value=(1024**3, 8 * 1024**3)
            ),
            mock.patch.object(torch.cuda, "memory_stats", return_value=stats),
            mock.patch.object(
                torch.cuda, "memory_summary", return_value="FULL ALLOCATOR TABLE"
            ),
            mock.patch.object(
                torch.cuda.memory, "_snapshot", return_value=_sample_snapshot()
            ),
            self.assertLogs(oom_diag._LOG, level="ERROR") as logs,
        ):
            oom_diag.dump_oom_diagnostics(
                tag="normal_engine_step",
                alloc_size=256 * 1024**2,
                exception="CUDA out of memory; original stack marker",
                cpp_backtrace="CUDAGraph.cpp:208 original backtrace marker",
            )

        files = list(oom_diag._out_dir().glob("oom_allocator_normal_engine_step_*.log"))
        self.assertEqual(len(files), 1)
        text = files[0].read_text(encoding="utf-8")
        self.assertIn("CUDA out of memory; original stack marker", text)
        self.assertIn("CUDAGraph.cpp:208 original backtrace marker", text)
        self.assertIn("failed_alloc_bytes=268435456 (256.00 MiB)", text)
        self.assertIn("allocated_bytes.all.current=3221225472 (3.00 GiB)", text)
        self.assertIn("[TORCH MEMORY SUMMARY]\nFULL ALLOCATOR TABLE", text)
        self.assertIn(
            "[TORCH ALLOCATOR SEGMENTS AND BLOCKS - FULL, NOT TRUNCATED]", text
        )
        self.assertIn("segments=2 blocks=3", text)
        self.assertIn("BLOCK[000000.000000]", text)
        self.assertIn("BLOCK[000000.000001]", text)
        self.assertIn("BLOCK[000001.000000]", text)
        self.assertIn("usage=ACTIVE_ALLOCATED", text)
        self.assertIn("usage=CACHED_FREE", text)
        self.assertIn("usage=ACTIVE_AWAITING_FREE", text)
        self.assertIn("size=3145728 (3.00 MiB)", text)
        self.assertIn("model_forward at model.py:42", text)
        self.assertIn("FULL ALLOCATOR TABLE", "\n".join(logs.output))

    def _assert_dump_failure_preserves_original_exception(self) -> None:
        original = OriginalOomError("original CUDA OOM marker")
        inner_exception = None
        inner_traceback = None
        outer_exception = None
        outer_traceback = None
        try:
            try:
                raise original
            except OriginalOomError as caught:
                inner_exception = caught
                inner_traceback = caught.__traceback__
                oom_diag.dump_oom_diagnostics(exception=str(caught))
                raise
        except OriginalOomError as caught:
            outer_exception = caught
            outer_traceback = caught.__traceback__

        self.assertIs(inner_exception, original)
        self.assertIs(outer_exception, original)
        self.assertIs(outer_traceback, inner_traceback)
        self.assertEqual(str(outer_exception), "original CUDA OOM marker")

    def test_allocator_snapshot_failure_does_not_replace_original_exception(
        self,
    ) -> None:
        with (
            mock.patch.object(torch.cuda, "current_device", return_value=0),
            mock.patch.object(torch.cuda, "mem_get_info", return_value=(1, 2)),
            mock.patch.object(torch.cuda, "memory_stats", return_value={}),
            mock.patch.object(torch.cuda, "memory_summary", return_value="summary"),
            mock.patch.object(
                torch.cuda.memory,
                "_snapshot",
                side_effect=RuntimeError("allocator snapshot failed"),
            ),
            self.assertLogs(oom_diag._LOG, level="ERROR") as logs,
        ):
            self._assert_dump_failure_preserves_original_exception()
        self.assertIn("allocator snapshot failed", "\n".join(logs.output))

    def test_dump_file_failure_does_not_replace_original_exception(self) -> None:
        with (
            mock.patch.object(torch.cuda, "current_device", return_value=0),
            mock.patch.object(torch.cuda, "mem_get_info", return_value=(1, 2)),
            mock.patch.object(torch.cuda, "memory_stats", return_value={}),
            mock.patch.object(torch.cuda, "memory_summary", return_value="summary"),
            mock.patch.object(
                torch.cuda.memory, "_snapshot", return_value={"segments": []}
            ),
            mock.patch.object(
                oom_diag.Path, "open", side_effect=OSError("dump file open failed")
            ),
            self.assertLogs(oom_diag._LOG, level="ERROR") as logs,
        ):
            self._assert_dump_failure_preserves_original_exception()
        self.assertIn("dump file open failed", "\n".join(logs.output))

    def test_dump_is_one_shot(self) -> None:
        with (
            mock.patch.object(torch.cuda, "current_device", return_value=0),
            mock.patch.object(torch.cuda, "mem_get_info", return_value=(1, 2)),
            mock.patch.object(torch.cuda, "memory_stats", return_value={}),
            mock.patch.object(torch.cuda, "memory_summary", return_value="summary"),
            mock.patch.object(
                torch.cuda.memory, "_snapshot", return_value={"segments": []}
            ),
        ):
            oom_diag.dump_oom_diagnostics(tag="first")
            oom_diag.dump_oom_diagnostics(tag="second")

        self.assertEqual(len(list(oom_diag._out_dir().glob("oom_allocator_*.log"))), 1)

    def test_suffix_is_process_safe(self) -> None:
        suffix = oom_diag._suffix("test", 7)
        self.assertIn("_r", suffix)
        self.assertIn("_s", suffix)
        self.assertIn("_d7_", suffix)
        self.assertIn(f"_pid{os.getpid()}_", suffix)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class OomDiagCudaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="oom_diag_cuda_test_")
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
            oom_diag._oom_observer(0, 4096, 80 * 1024**3, 1024)
        finally:
            del tensor
            torch.cuda.empty_cache()

        markers = list(
            oom_diag._out_dir().glob("oom_allocator_torch_allocator_oom_*.log")
        )
        snapshots = list(oom_diag._out_dir().glob("oom_allocator_snapshot_*.pickle"))
        self.assertEqual(len(markers), 1)
        self.assertEqual(len(snapshots), 1)
        marker_text = markers[0].read_text(encoding="utf-8")
        self.assertIn("[TORCH MEMORY SUMMARY]", marker_text)
        self.assertIn(
            "[TORCH ALLOCATOR SEGMENTS AND BLOCKS - FULL, NOT TRUNCATED]", marker_text
        )
        self.assertIn("BLOCK[", marker_text)
        self.assertIn("allocation_frames=", marker_text)
        self.assertIn("oom_diag_test.py", marker_text)
        with snapshots[0].open("rb") as snapshot_file:
            snapshot = pickle.load(snapshot_file)
        self.assertIn("segments", snapshot)
        expected_blocks = sum(
            len(segment.get("blocks", [])) for segment in snapshot["segments"]
        )
        self.assertEqual(marker_text.count("  BLOCK["), expected_blocks)


if __name__ == "__main__":
    unittest.main()
