"""Core performance behavior contracts."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from rtp_llm.test.perf_test.decode.executor import DecodeCaseExecutor
from rtp_llm.test.perf_test.decode.model import (
    DecodeBatchMeasurement,
    DecodeCaseSpec,
    DecodeProfilingConfig,
    ExactBatchPolicy,
    TargetTPOTPolicy,
)
from rtp_llm.test.perf_test.decode.profiling import DecodeProfiler
from rtp_llm.test.perf_test.plan import case_id
from rtp_llm.test.perf_test.prefill.profile import PrefillProfiler
from rtp_llm.test.perf_test.prefill.trace import TraceCapture, TraceClient
from rtp_llm.test.perf_test.test_support import prefill_profile_formal_bytes
from rtp_llm.test.perf_test.test_support import (
    prefill_profile_formal_bytes as _formal_bytes,
)
from rtp_llm.test.perf_test.test_support import prefill_profile_setup
from rtp_llm.test.perf_test.test_support import prefill_profile_setup as _setup

decode_profiling_TOPOLOGY = {"tp": 1, "dp": 1, "pp": 1, "ep": 1}


def decode_profiling_exact_case(*, batch=4, profile=False):
    return DecodeCaseSpec(
        workload_kind="grid",
        input_lengths=(128,) * batch,
        output_length=16,
        batch_policy=ExactBatchPolicy(strategy="fixed", size=batch),
        measure_runs=2,
        warmup_runs=1,
        generate_config={"top_p": 0.8, "nested": {"value": 1}},
        profiling=DecodeProfilingConfig(enabled=profile),
    )


def decode_profiling_tpot_case(*, profile=True):
    return replace(
        decode_profiling_exact_case(batch=16, profile=profile),
        batch_policy=TargetTPOTPolicy("target_tpot", 16, 4, 10.0),
    )


def decode_profiling_case_payload(case):
    identity = case.to_identity()
    return {"case_id": case_id("decode", identity), **identity}


def decode_profiling_measurement(batch, tpot=1.0):
    return DecodeBatchMeasurement(
        batch_size=batch,
        input_lengths=(128,) * batch,
        worker_allocations=(),
        measurements=tuple(
            (
                {
                    "run": run,
                    "worker_allocations": [],
                    "throughput_tokens_per_second": 100.0,
                    "requests": [
                        {
                            "request_index": index,
                            "success": True,
                            "input_length": 128,
                            "output_length": 16,
                            "latency_ms": {
                                "wait": 1.0,
                                "prefill": 2.0,
                                "decode": 15.0,
                                "tpot": tpot,
                                "total": 17.0,
                            },
                        }
                        for index in range(batch)
                    ],
                }
                for run in range(2)
            )
        ),
    )


class DefaultExecutorTest(unittest.TestCase):

    def test_profile_nsys_launch_is_scoped_and_preserves_default_launch(self):
        from dataclasses import replace

        from rtp_llm.test.perf_test.execution.runtime import create_maga_handle
        from rtp_llm.test.perf_test.plan import freeze, thaw
        from rtp_llm.test.perf_test.prefill_profile_test import _setup

        with tempfile.TemporaryDirectory() as directory:
            plan = _setup(Path(directory) / "run")
            config = thaw(plan.resolved_config)
            config["workload"]["profiling"] = {
                "backend": "nsys",
                "nsys_path": "/opt/nsys",
            }
            config["engine"]["args"]["torch_cuda_profiler_dir"] = "/formal"
            plan = replace(plan, resolved_config=freeze(config))
            manager = Mock()
            manager.return_value.port = 123
            with patch.dict(
                "sys.modules",
                {
                    "rtp_llm.test.utils.maga_server_manager": SimpleNamespace(
                        MagaServerManager=manager
                    )
                },
            ):
                handle = create_maga_handle(
                    plan, profile_directory=directory, nsys_session="unique"
                )
                options = manager.call_args.kwargs
                self.assertEqual(
                    [
                        "/opt/nsys",
                        "launch",
                        "--session-new=unique",
                        "--show-output=true",
                        "--trace=cuda,nvtx,osrt",
                        "--wait=all",
                    ],
                    options["process_prefix"],
                )
                self.assertIn(
                    "--torch_cuda_profiler_dir=" + directory, options["smoke_args_str"]
                )
                self.assertNotIn("/formal", options["smoke_args_str"])
                handle.stop()
                manager.return_value.stop_server.assert_called_once()
                create_maga_handle(plan)
                self.assertNotIn("process_prefix", manager.call_args.kwargs)
                self.assertIn("/formal", manager.call_args.kwargs["smoke_args_str"])


class PrefillProfileTest(unittest.TestCase):

    def test_selected_case_is_reconstructed_into_isolated_profile_session(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            plan = prefill_profile_setup(root)
            before = prefill_profile_formal_bytes(root)
            runner = Mock()
            runner.run_case.side_effect = lambda case: {
                "case_id": case["case_id"],
                "status": "ok",
                "metrics": {"latency_ms": 1.0},
                "errors": [],
            }

            def factory(profile_dir):
                (profile_dir / "trace.json").write_text(
                    '{"traceEvents": [{"name": "kernel"}]}'
                )
                original = runner.run_case.side_effect
                runner.run_case.side_effect = lambda case: dict(
                    original(case), trace_files=["trace.json"]
                )
                return runner

            profiler = PrefillProfiler(factory)
            outcome = profiler.run(
                root, [plan.cases[0].case_id], session_id="profile-001"
            )
            self.assertEqual("completed", outcome["status"])
            reconstructed = runner.run_case.call_args.args[0]
            self.assertEqual(plan.cases[0].case_id, reconstructed["case_id"])
            self.assertEqual(512, reconstructed["input_len"])
            self.assertTrue(
                (root / "profiles" / "profile-001" / "profile_result.json").is_file()
            )
            self.assertEqual(before, prefill_profile_formal_bytes(root))

    def test_failed_profile_keeps_formal_artifacts_unchanged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            plan = prefill_profile_setup(root)
            before = prefill_profile_formal_bytes(root)
            runner = Mock()
            runner.run_case.return_value = {
                "case_id": plan.cases[0].case_id,
                "status": "invalid_reuse",
                "metrics": None,
                "errors": [{"code": "invalid_reuse", "message": "mismatch"}],
            }
            outcome = PrefillProfiler(lambda _profile_dir: runner).run(
                root, [plan.cases[0].case_id], session_id="failed"
            )
            self.assertEqual("failed", outcome["status"])
            self.assertEqual(before, prefill_profile_formal_bytes(root))


class TraceTest(unittest.TestCase):

    def test_kineto_all_ranks_and_measure_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            post = Mock(return_value=Mock(json=lambda: {"status": "ok"}))
            capture = TraceCapture(root, "case", port=123, tp_size=2, post=post)
            client = TraceClient(Mock(), capture)
            client.request(phase="seed")
            post.assert_not_called()
            client.request(phase="measure")
            client.request(phase="measure")
            self.assertEqual(1, post.call_count)
            self.assertEqual(
                "http://127.0.0.1:123/start_profile", post.call_args.args[0]
            )
            for rank in range(2):
                (root / f"case_wr{rank}_1.json").write_text(
                    json.dumps({"traceEvents": [{"name": "kernel"}]})
                )
            self.assertEqual(2, len(capture.finish()))

    def test_nsys_stop_exports_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def command(argv, **kwargs):
                if argv[1] == "stop":
                    (root / "case.nsys-rep").write_bytes(b"report")
                return Mock(returncode=0, stdout="", stderr="")

            run = Mock(side_effect=command)
            capture = TraceCapture(
                root, "case", backend="nsys", nsys_session="session", run=run
            )
            capture.start()
            self.assertEqual(["case.nsys-rep"], capture.finish())
            self.assertEqual(
                ["start", "stop"], [call.args[0][1] for call in run.call_args_list]
            )


class DecodeProfilingTest(unittest.TestCase):

    def test_tpot_profiles_only_selected_batch_after_search(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.json"
            batch_executor = Mock()
            events = []

            def measure(_case, batch):
                events.append(("measure", batch))
                return decode_profiling_measurement(batch, 8.0 if batch <= 8 else 12.0)

            batch_executor.measure.side_effect = measure

            def profile(_case, batch, **_kwargs):
                events.append(("profile", batch))
                trace.write_text(json.dumps({"traceEvents": []}))

            batch_executor.profile.side_effect = profile
            profiler = DecodeProfiler(
                batch_executor,
                run_root=Path(directory),
                log_path=None,
                sleep=lambda _seconds: None,
                artifact_discoverer=lambda _log, _offset, _name: (str(trace),),
            )
            result = DecodeCaseExecutor(
                batch_executor, topology=decode_profiling_TOPOLOGY, profiler=profiler
            ).execute(decode_profiling_case_payload(decode_profiling_tpot_case()))
            self.assertEqual([("measure", 8), ("measure", 12), ("profile", 8)], events)
            self.assertEqual("ok", result["status"])
            manifest = result["metrics"]["profiling"]
            self.assertEqual("completed", manifest["status"])
            self.assertEqual([128] * 8, manifest["input_lengths"])
            self.assertEqual(8, manifest["batch_size"])
            self.assertEqual(
                [str(Path(directory) / "traces/trace.json")], manifest["artifacts"]
            )
            self.assertIn("decode_in128_out16_bs8_", manifest["trace_name"])
            self.assertEqual(
                manifest,
                json.loads((Path(directory) / manifest["manifest_path"]).read_text()),
            )

    def test_profile_failure_does_not_change_valid_performance_status(self):
        with tempfile.TemporaryDirectory() as directory:
            batch_executor = Mock()
            batch_executor.measure.return_value = decode_profiling_measurement(4)
            batch_executor.profile.side_effect = RuntimeError(
                "profile failed\n" + "x" * 600
            )
            profiler = DecodeProfiler(
                batch_executor, run_root=Path(directory), log_path=None
            )
            result = DecodeCaseExecutor(
                batch_executor, topology=decode_profiling_TOPOLOGY, profiler=profiler
            ).execute(
                decode_profiling_case_payload(decode_profiling_exact_case(profile=True))
            )
            self.assertEqual("request_failed", result["status"])
            self.assertIn("requires shutdown", result["errors"][0]["message"])
            self.assertFalse(profiler.session_reusable)
            self.assertEqual(4, result["metrics"]["global_batch_size"])
            manifest = result["metrics"]["profiling"]
            self.assertEqual("failed", manifest["status"])
            self.assertEqual([], manifest["artifacts"])
            self.assertEqual(500, len(manifest["error"]))
            self.assertNotIn("\n", manifest["error"])
            self.assertEqual(
                manifest,
                json.loads((Path(directory) / manifest["manifest_path"]).read_text()),
            )

    def test_flush_timeout_writes_failed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            profiler = DecodeProfiler(
                Mock(),
                run_root=Path(directory),
                log_path=None,
                monotonic=Mock(side_effect=[0.0, 0.0, 2.0]),
                sleep=lambda _seconds: None,
                artifact_discoverer=lambda _log, _offset, _name: (),
            )
            case = replace(
                decode_profiling_exact_case(batch=1, profile=True),
                profiling=DecodeProfilingConfig(enabled=True, flush_timeout_seconds=1),
            )
            manifest = profiler.profile(case, batch_size=1)
            self.assertEqual("failed", manifest["status"])
            self.assertIn("trace flush timeout", manifest["error"])
            self.assertTrue((Path(directory) / manifest["manifest_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
