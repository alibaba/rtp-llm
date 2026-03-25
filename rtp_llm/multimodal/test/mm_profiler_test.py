import os
import shutil
import tempfile
import threading
from unittest import TestCase, main
from unittest.mock import MagicMock, patch

from rtp_llm.multimodal.mm_profiler import MMProfiler, _build_top_operations


class MMProfilerTest(TestCase):

    def setUp(self):
        self.profiler = MMProfiler()
        self.tmp_dir = tempfile.mkdtemp(prefix="mm_profiler_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # start_profile
    # ------------------------------------------------------------------

    def test_start_profile_returns_started(self):
        result = self.profiler.start_profile(count=5, output_path=self.tmp_dir)
        self.assertEqual(result["status"], "started")
        self.assertEqual(result["target_count"], 5)
        self.assertEqual(result["output_path"], self.tmp_dir)

    def test_start_profile_creates_output_dir(self):
        out = os.path.join(self.tmp_dir, "sub", "dir")
        self.profiler.start_profile(count=1, output_path=out)
        self.assertTrue(os.path.isdir(out))

    def test_start_profile_rejects_double_start(self):
        self.profiler.start_profile(count=5, output_path=self.tmp_dir)
        result = self.profiler.start_profile(count=3, output_path=self.tmp_dir)
        self.assertEqual(result["status"], "error")
        self.assertIn("already in progress", result["message"])

    def test_start_profile_stores_config(self):
        self.profiler.start_profile(
            count=1,
            output_path=self.tmp_dir,
            record_shapes=False,
            with_stack=True,
            profile_memory=False,
        )
        self.assertEqual(self.profiler._profile_cfg["record_shapes"], False)
        self.assertEqual(self.profiler._profile_cfg["with_stack"], True)
        self.assertEqual(self.profiler._profile_cfg["profile_memory"], False)

    # ------------------------------------------------------------------
    # end_profile
    # ------------------------------------------------------------------

    def test_end_profile_no_session(self):
        result = self.profiler.end_profile()
        self.assertEqual(result["status"], "error")
        self.assertIn("No profiling session", result["message"])

    def test_end_profile_while_armed(self):
        self.profiler.start_profile(count=10, output_path=self.tmp_dir)
        result = self.profiler.end_profile()
        self.assertEqual(result["status"], "stopped_early")
        self.assertEqual(result["profiled_count"], 0)
        self.assertFalse(self.profiler._armed)

    def test_end_profile_after_completion(self):
        self.profiler.start_profile(count=1, output_path=self.tmp_dir)
        self._run_profiled_request()

        self.assertTrue(self.profiler._finished)
        result = self.profiler.end_profile()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["profiled_count"], 1)

    def test_end_profile_generates_summary(self):
        self.profiler.start_profile(count=1, output_path=self.tmp_dir)
        self._run_profiled_request(
            averages=_FakeKeyAverages(
                [
                    _FakeEvent("conv2d", count=4, cuda_total=2000),
                ]
            )
        )

        result = self.profiler.end_profile()
        self.assertIn("summary", result["files"])
        self.assertIn("top_operations", result["files"])
        self.assertTrue(os.path.isfile(result["files"]["summary"]))
        self.assertTrue(os.path.isfile(result["files"]["top_operations"]))

    # ------------------------------------------------------------------
    # get_status
    # ------------------------------------------------------------------

    def test_get_status_idle(self):
        s = self.profiler.get_status()
        self.assertFalse(s["is_profiling"])
        self.assertEqual(s["profiled_count"], 0)
        self.assertEqual(s["target_count"], 0)
        self.assertFalse(s["finished"])

    def test_get_status_armed(self):
        self.profiler.start_profile(count=10, output_path=self.tmp_dir)
        s = self.profiler.get_status()
        self.assertTrue(s["is_profiling"])
        self.assertEqual(s["target_count"], 10)
        self.assertEqual(s["output_path"], self.tmp_dir)

    def test_get_status_after_profile(self):
        self.profiler.start_profile(count=2, output_path=self.tmp_dir)
        self._run_profiled_request()

        s = self.profiler.get_status()
        self.assertTrue(s["is_profiling"])
        self.assertEqual(s["profiled_count"], 1)

    def test_get_status_finished(self):
        self.profiler.start_profile(count=1, output_path=self.tmp_dir)
        self._run_profiled_request()

        s = self.profiler.get_status()
        self.assertFalse(s["is_profiling"])
        self.assertTrue(s["finished"])

    # ------------------------------------------------------------------
    # on_request_complete (proxy no-op hook)
    # ------------------------------------------------------------------

    def test_on_request_complete_is_noop(self):
        self.profiler.on_request_complete()
        self.assertEqual(self.profiler._profiled_count, 0)
        self.assertFalse(self.profiler._armed)

    # ------------------------------------------------------------------
    # profile_request context manager
    # ------------------------------------------------------------------

    @patch("torch.cuda.is_available", return_value=False)
    def test_profile_request_noop_when_not_armed(self, _mock_cuda):
        with self.profiler.profile_request():
            pass
        self.assertEqual(self.profiler._profiled_count, 0)

    @patch("torch.cuda.is_available", return_value=False)
    def test_profile_request_profiles_when_armed(self, _mock_cuda):
        self.profiler.start_profile(count=5, output_path=self.tmp_dir)
        with patch("torch.profiler.profile") as mock_prof:
            mock_ctx = MagicMock()
            mock_prof.return_value = mock_ctx
            with self.profiler.profile_request():
                pass
        self.assertEqual(self.profiler._profiled_count, 1)
        self.assertTrue(self.profiler._armed)
        mock_ctx.export_chrome_trace.assert_called_once()

    @patch("torch.cuda.is_available", return_value=False)
    def test_profile_request_auto_stops_at_target(self, _mock_cuda):
        self.profiler.start_profile(count=2, output_path=self.tmp_dir)
        for _ in range(2):
            self._run_profiled_request()

        self.assertFalse(self.profiler._armed)
        self.assertTrue(self.profiler._finished)
        self.assertEqual(self.profiler._profiled_count, 2)

    @patch("torch.cuda.is_available", return_value=False)
    def test_profile_request_skips_after_target_reached(self, _mock_cuda):
        self.profiler.start_profile(count=1, output_path=self.tmp_dir)
        self._run_profiled_request()
        self.assertTrue(self.profiler._finished)

        with self.profiler.profile_request():
            pass
        self.assertEqual(self.profiler._profiled_count, 1)

    @patch("torch.cuda.is_available", return_value=False)
    def test_profile_request_exports_trace_file(self, _mock_cuda):
        self.profiler.start_profile(count=1, output_path=self.tmp_dir)
        with patch("torch.profiler.profile") as mock_prof:
            mock_ctx = MagicMock()
            mock_prof.return_value = mock_ctx
            with self.profiler.profile_request():
                pass

        expected = os.path.join(self.tmp_dir, "timeline_0.json")
        mock_ctx.export_chrome_trace.assert_called_once_with(expected)

    @patch("torch.cuda.is_available", return_value=False)
    def test_profile_request_handles_export_error(self, _mock_cuda):
        self.profiler.start_profile(count=1, output_path=self.tmp_dir)
        with patch("torch.profiler.profile") as mock_prof:
            mock_ctx = MagicMock()
            mock_ctx.export_chrome_trace.side_effect = RuntimeError("export failed")
            mock_prof.return_value = mock_ctx
            with self.profiler.profile_request():
                pass
        self.assertEqual(self.profiler._profiled_count, 1)

    # ------------------------------------------------------------------
    # Full lifecycle
    # ------------------------------------------------------------------

    def test_full_lifecycle(self):
        result = self.profiler.start_profile(count=2, output_path=self.tmp_dir)
        self.assertEqual(result["status"], "started")

        self._run_profiled_request()
        self.assertEqual(self.profiler._profiled_count, 1)
        self.assertTrue(self.profiler._armed)

        self._run_profiled_request()
        self.assertEqual(self.profiler._profiled_count, 2)
        self.assertFalse(self.profiler._armed)
        self.assertTrue(self.profiler._finished)

        result = self.profiler.end_profile()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["profiled_count"], 2)

    def test_can_restart_after_completion(self):
        self.profiler.start_profile(count=1, output_path=self.tmp_dir)
        self._run_profiled_request()
        self.profiler.end_profile()

        result = self.profiler.start_profile(count=3, output_path=self.tmp_dir)
        self.assertEqual(result["status"], "started")
        self.assertEqual(result["target_count"], 3)

    # ------------------------------------------------------------------
    # _build_top_operations
    # ------------------------------------------------------------------

    def test_build_top_operations(self):
        averages = _FakeKeyAverages(
            [
                _FakeEvent(
                    "op_a",
                    count=10,
                    cpu_total=1000,
                    cuda_total=5000,
                    self_cpu=800,
                    self_cuda=4500,
                ),
                _FakeEvent(
                    "op_b",
                    count=5,
                    cpu_total=500,
                    cuda_total=2000,
                    self_cpu=400,
                    self_cuda=1800,
                ),
            ]
        )
        ops = _build_top_operations(averages)
        self.assertEqual(len(ops), 2)
        self.assertEqual(ops[0]["name"], "op_a")
        self.assertGreater(ops[0]["cuda_time_total_us"], ops[1]["cuda_time_total_us"])
        self.assertEqual(ops[0]["count"], 10)
        self.assertAlmostEqual(ops[0]["cpu_time_avg_us"], 100.0)

    # ------------------------------------------------------------------
    # Thread-safety smoke test
    # ------------------------------------------------------------------

    def test_concurrent_start_only_one_wins(self):
        results = []
        barrier = threading.Barrier(10)

        def try_start():
            barrier.wait()
            r = self.profiler.start_profile(count=5, output_path=self.tmp_dir)
            results.append(r)

        threads = [threading.Thread(target=try_start) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        started = [r for r in results if r["status"] == "started"]
        errors = [r for r in results if r["status"] == "error"]
        self.assertEqual(len(started), 1)
        self.assertEqual(len(errors), 9)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_profiled_request(self, averages=None):
        """Run one request through profile_request with mocked torch profiler."""
        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.profiler.profile"
        ) as mock_prof:
            mock_ctx = MagicMock()
            if averages is not None:
                mock_ctx.key_averages.return_value = averages
            mock_prof.return_value = mock_ctx
            with self.profiler.profile_request():
                pass


# ======================================================================
# Fake objects for testing
# ======================================================================


class _FakeEvent:
    def __init__(
        self, key, count=1, cpu_total=0, cuda_total=0, self_cpu=0, self_cuda=0
    ):
        self.key = key
        self.count = count
        self.cpu_time_total = cpu_total
        self.cuda_time_total = cuda_total
        self.self_cpu_time_total = self_cpu
        self.self_cuda_time_total = self_cuda


class _FakeKeyAverages(list):
    def __init__(self, events=None):
        super().__init__(events or [])

    def table(self, sort_by=None, row_limit=None):
        return "fake summary table"


if __name__ == "__main__":
    main()
