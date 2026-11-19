"""Performance framework behavior and failure regressions."""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from rtp_llm.test.perf_test.config_schema import EngineConfig
from rtp_llm.test.perf_test.decode.executor import (
    DecodeBatchExecutor,
    DecodeCaseExecutor,
)
from rtp_llm.test.perf_test.decode.model import (
    DecodeCaseSpec,
    DecodeProfilingConfig,
    ExactBatchPolicy,
)
from rtp_llm.test.perf_test.decode.profiling import DecodeProfiler
from rtp_llm.test.perf_test.plan import case_id
from rtp_llm.test.perf_test.prefill.profile import (
    PrefillProfiler,
    ProfiledRunner,
    TraceCapture,
    TraceClient,
)
from rtp_llm.test.perf_test.storage import atomic_write_json
from rtp_llm.test.perf_test.tests.support.execution import default_executor_measurement
from rtp_llm.test.perf_test.tests.support.prefill import prefill_profile_formal_bytes
from rtp_llm.test.perf_test.tests.support.prefill import prefill_profile_setup
from rtp_llm.test.perf_test.tests.support.prefill import prefill_profile_setup as _setup


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
                case_dir = profile_dir / plan.cases[0].case_id
                case_dir.mkdir()
                (case_dir / "trace_wr0_1.json").write_text(
                    '{"traceEvents": [{"name": "kernel"}]}'
                )
                original = runner.run_case.side_effect
                runner.run_case.side_effect = lambda case: dict(
                    original(case), trace_files=[case["case_id"] + "/trace_wr0_1.json"]
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
            manifest = json.loads(
                (root / "profiles/profile-001/manifest.json").read_text()
            )
            self.assertEqual(2, manifest["schema_version"])
            self.assertEqual("prefill", manifest["mode"])
            artifact = manifest["cases"][0]["artifacts"][0]
            self.assertEqual(0, artifact["rank"])
            self.assertEqual("chrome_trace", artifact["format"])
            self.assertEqual(
                f"traces/profile-001/{plan.cases[0].case_id}/trace_wr0_1.json",
                artifact["path"],
            )
            self.assertTrue((root / artifact["path"]).is_file())
            with self.assertRaisesRegex(Exception, "already exists"):
                profiler.run(root, [plan.cases[0].case_id], session_id="profile-001")
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

    def test_prefill_runner_retains_completed_capture_under_its_case(self):
        from rtp_llm.test.perf_test.plan import thaw

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = prefill_profile_setup(root / "run")
            trace_dir = root / "run/traces/capture"
            trace_dir.mkdir(parents=True)
            case = dict(thaw(plan.cases[0].identity), case_id=plan.cases[0].case_id)
            name = case["case_id"] + "_wr0_1.json"
            payload = '{"traceEvents": [{"name": "kernel"}]}'
            (trace_dir / name).write_text(payload)
            capture = Mock(requires_session_restart=False, commands=[])
            capture.finish.return_value = [name]
            runner = Mock()
            runner.run_case.return_value = {"case_id": case["case_id"], "status": "ok"}
            with patch(
                "rtp_llm.test.perf_test.prefill.profile.TraceCapture",
                return_value=capture,
            ), patch(
                "rtp_llm.test.perf_test.prefill.runner.PrefillRunner",
                return_value=runner,
            ):
                result = ProfiledRunner(
                    Mock(), Mock(), trace_dir, thaw(plan.resolved_config), "capture"
                ).run_case(case)
            self.assertEqual([case["case_id"] + "/" + name], result["trace_files"])
            self.assertEqual(
                payload, (trace_dir / result["trace_files"][0]).read_text()
            )
            self.assertFalse((trace_dir / name).exists())
            capture.abort.assert_called_once()

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

    def test_fixed_batch_profiles_after_formal_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.json"
            batch_executor = Mock()
            events = []

            def measure(_case, batch):
                events.append(("measure", batch))
                return decode_profiling_measurement(batch, 8.0)

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
            ).execute(
                decode_profiling_case_payload(
                    decode_profiling_exact_case(batch=8, profile=True)
                )
            )
            self.assertEqual([("measure", 8), ("profile", 8)], events)
            self.assertEqual("ok", result["status"])
            reference = result["metrics"]["profiling"]
            self.assertEqual({"status", "manifest_path", "error"}, set(reference))
            self.assertEqual("completed", reference["status"])
            manifest = json.loads(
                (Path(directory) / reference["manifest_path"]).read_text()
            )
            self.assertEqual(2, manifest["schema_version"])
            self.assertEqual("decode", manifest["mode"])
            self.assertEqual("kineto", manifest["backend"])
            artifact = manifest["cases"][0]["artifacts"][0]
            self.assertEqual("decode", artifact["phase"])
            self.assertTrue(
                artifact["path"].startswith("traces/" + manifest["session_id"] + "/")
            )
            self.assertTrue((Path(directory) / artifact["path"]).is_file())
            self.assertEqual(result["case_id"], manifest["cases"][0]["case_id"])

        # Model TP ranks: only rank zero sees the request's gen_timeline flag.
        # Each nonzero rank needs a fresh all-rank RPC before every round.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transport = Mock()
            armed = set()
            captures = {0: 0, 1: 0}
            events = []

            def start_profile(**kwargs):
                events.append("arm")
                self.assertTrue(kwargs["enable_all_ranks"])
                armed.update((0, 1))

            def round_requests(**kwargs):
                events.append("round")
                self.assertTrue(kwargs["generate_config"]["gen_timeline"])
                armed.add(0)
                for rank in tuple(armed):
                    captures[rank] += 1
                    name = kwargs["generate_config"]["profile_trace_name"]
                    (root / f"{name}_wr{rank}_{captures[rank]}.json").write_text(
                        json.dumps({"traceEvents": []})
                    )
                    armed.remove(rank)
                return ({"success": True, "request_index": 0},)

            transport.start_profile.side_effect = start_profile
            transport.run_round.side_effect = round_requests
            batch = DecodeBatchExecutor(
                transport,
                engine=EngineConfig(
                    max_seq_len=4096, max_batch_tokens=8192, concurrency_limit=16
                ),
                dp_size=1,
            )
            case = replace(
                decode_profiling_exact_case(batch=1, profile=True),
                profiling=DecodeProfilingConfig(enabled=True, runs=3),
            )
            profiler = DecodeProfiler(
                batch,
                run_root=root,
                log_path=None,
                trace_directory=root,
                world_size=2,
                monotonic=Mock(side_effect=[0.0, 0.0, 2.0]),
                sleep=lambda _seconds: None,
            )
            result = profiler.profile(case, batch_size=1)
            self.assertEqual(["arm", "round"] * 3, events)
            self.assertEqual({0: 3, 1: 3}, captures)
            self.assertEqual("completed", result["status"])
            self.assertTrue(profiler.session_reusable)
            self.assertFalse(armed)
            manifest = json.loads((root / result["manifest_path"]).read_text())
            self.assertEqual(6, len(manifest["cases"][0]["artifacts"]))

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
            self.assertEqual(500, len(manifest["error"]))
            self.assertNotIn("\n", manifest["error"])
            stored = json.loads(
                (Path(directory) / manifest["manifest_path"]).read_text()
            )
            self.assertEqual("failed", stored["status"])
            self.assertEqual([], stored["cases"][0]["artifacts"])
        for failure in ("copy", "mkdir", "manifest"):
            with self.subTest(
                failure=failure
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                trace = root / "source.json"
                batch_executor = Mock()
                batch_executor.measure.return_value = decode_profiling_measurement(4)
                batch_executor.profile.side_effect = (
                    lambda *args, **kwargs: trace.write_text(
                        json.dumps({"traceEvents": []})
                    )
                )
                if failure == "mkdir":
                    (root / "traces").write_text("occupied")

                def persist(path, payload):
                    if failure == "copy" and "traces" in path.parts:
                        raise OSError("trace copy disk full")
                    if failure == "manifest" and path.name == "manifest.json":
                        raise OSError("profile manifest disk full")
                    atomic_write_json(path, payload)

                profiler = DecodeProfiler(
                    batch_executor,
                    run_root=root,
                    log_path=None,
                    artifact_discoverer=lambda *_args: (str(trace),),
                )
                executor = DecodeCaseExecutor(
                    batch_executor,
                    topology=decode_profiling_TOPOLOGY,
                    profiler=profiler,
                )
                with patch(
                    "rtp_llm.test.perf_test.decode.profiling.atomic_write_json",
                    side_effect=persist,
                ):
                    result = executor.execute(
                        decode_profiling_case_payload(
                            decode_profiling_exact_case(profile=True)
                        )
                    )
                self.assertEqual("ok", result["status"])
                self.assertTrue(profiler.session_reusable)
                manifest = result["metrics"]["profiling"]
                self.assertEqual("failed", manifest["status"])
                self.assertIn(
                    "Not a directory" if failure == "mkdir" else "disk full",
                    manifest["error"],
                )
                if failure != "manifest":
                    stored = json.loads((root / manifest["manifest_path"]).read_text())
                    self.assertEqual("failed", stored["status"])
                    self.assertEqual([], stored["cases"][0]["artifacts"])
                next_result = executor.execute(
                    decode_profiling_case_payload(decode_profiling_exact_case())
                )
                self.assertEqual("ok", next_result["status"])
                self.assertEqual(2, batch_executor.measure.call_count)

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
            # One completed file cannot prove a two-round/all-rank capture finished.
            for runs, world_size in ((2, 1), (1, 2)):
                trace = Path(directory) / "partial_wr0_1.json"
                trace.write_text(json.dumps({"traceEvents": []}))
                batch = Mock()
                partial = DecodeProfiler(
                    batch,
                    run_root=Path(directory),
                    log_path=None,
                    world_size=world_size,
                    monotonic=Mock(side_effect=[0.0, 0.0, 2.0]),
                    sleep=lambda _s: None,
                    artifact_discoverer=lambda *_args: (str(trace),),
                )
                config = replace(case.profiling, runs=runs, enable_all_ranks=True)
                result = partial.profile(replace(case, profiling=config), batch_size=1)
                self.assertEqual("failed", result["status"])
                self.assertFalse(partial.session_reusable)


class DefaultExecutorTest(unittest.TestCase):

    def test_profile_nsys_launch_is_scoped_and_preserves_default_launch(self):
        from dataclasses import replace

        from rtp_llm.test.perf_test.errors import PlanningError
        from rtp_llm.test.perf_test.execution.runtime import create_maga_handle
        from rtp_llm.test.perf_test.plan import freeze, thaw
        from rtp_llm.test.perf_test.tests.support.prefill import (
            prefill_profile_setup as _setup,
        )

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
                with self.assertRaisesRegex(PlanningError, "torch_cuda_profiler_dir"):
                    create_maga_handle(
                        plan, profile_directory=directory, nsys_session="unique"
                    )
                manager.assert_not_called()
                del config["engine"]["args"]["torch_cuda_profiler_dir"]
                plan = replace(plan, resolved_config=freeze(config))
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
                self.assertNotIn(
                    "--torch_cuda_profiler_dir",
                    manager.call_args.kwargs["smoke_args_str"],
                )
                self.assertNotIn(
                    "TORCH_CUDA_PROFILER_DIR", manager.call_args.kwargs["env_args"]
                )


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


def decode_profiling_case_payload(case):
    identity = case.to_identity()
    return {"case_id": case_id("decode", identity), **identity}


def decode_profiling_measurement(batch, tpot=1.0):
    return default_executor_measurement(
        batch,
        (128,) * batch,
        16,
        runs=2,
        latency={
            "wait": 1.0,
            "prefill": 2.0,
            "decode": 15.0,
            "tpot": tpot,
            "total": 17.0,
        },
    )


if __name__ == "__main__":
    unittest.main()
