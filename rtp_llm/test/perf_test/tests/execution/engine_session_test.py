"""Performance framework behavior and failure regressions."""

import json
import signal
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test.entry import plan_from_source
from rtp_llm.test.perf_test.errors import EngineLifecycleError, PlanningError
from rtp_llm.test.perf_test.execution.distributed.member import run_member
from rtp_llm.test.perf_test.execution.engine_session import (
    EngineInterrupted,
    EngineSession,
)
from rtp_llm.test.perf_test.execution.runtime import (
    HttpPrefillClient,
    build_engine_launch_spec,
)
from rtp_llm.test.perf_test.plan import thaw
from rtp_llm.test.perf_test.server import configure_scheduler
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.tests.support.execution import (
    engine_session_FakeHandle,
    runtime_adapter_plan,
)
from rtp_llm.test.perf_test.tests.support.fixtures import PERF_TEST_ROOT


class EngineSessionTest(unittest.TestCase):

    def _session(self, handle, **kwargs):
        launcher = Mock(return_value=handle)
        timelines = Mock(return_value=["traces/timeline.json"])
        coredumps = Mock(return_value={"count": 1, "files": ["core.123"]})
        session = EngineSession(
            launcher=launcher,
            readiness_timeout_seconds=30,
            timeline_collector=timelines,
            coredump_collector=coredumps,
            install_signal_handlers=False,
            **kwargs
        )
        return (session, launcher, timelines, coredumps)

    def test_success_exposes_client_and_collects_diagnostics_on_teardown(self):
        handle = engine_session_FakeHandle()
        (session, launcher, timelines, coredumps) = self._session(handle)
        with session as active:
            self.assertIs(handle.client, active.client)
            self.assertEqual("/tmp/process.log", active.log_path)
            active.assert_healthy()
        launcher.assert_called_once_with()
        self.assertEqual([30], handle.wait_ready_calls)
        self.assertEqual(1, handle.stop_calls)
        timelines.assert_called_once_with()
        coredumps.assert_called_once_with()
        self.assertEqual(("traces/timeline.json",), session.diagnostics.timeline_files)
        self.assertEqual(1, session.diagnostics.coredumps["count"])

    def test_startup_failure_stops_and_preserves_diagnostics(self):
        handle = engine_session_FakeHandle(ready=False)
        handle.exit_code = 17
        (session, launcher, timelines, coredumps) = self._session(handle)
        with self.assertRaisesRegex(EngineLifecycleError, "failed readiness") as error:
            session.start()
        launcher.assert_called_once_with()
        self.assertEqual(1, handle.stop_calls)
        timelines.assert_called_once_with()
        coredumps.assert_called_once_with()
        self.assertEqual(17, error.exception.diagnostics.exit_code)
        session.close()
        self.assertEqual(1, handle.stop_calls)

    def test_interrupt_closes_session_and_carries_signal(self):
        handle = engine_session_FakeHandle()
        (session, _, _, _) = self._session(handle)
        with self.assertRaises(EngineInterrupted) as error:
            with session:
                try:
                    session.handle_signal(signal.SIGINT)
                except EngineInterrupted:
                    self.assertEqual(0, handle.stop_calls)
                    raise
        self.assertEqual(signal.SIGINT, error.exception.signal_number)
        self.assertEqual(1, handle.stop_calls)
        session.close()
        self.assertEqual(1, handle.stop_calls)

    def test_multiple_cases_share_exactly_one_startup(self):
        handle = engine_session_FakeHandle()
        (session, launcher, _, _) = self._session(handle)
        with session:
            session.start()
            for _case in range(5):
                self.assertIs(handle.client, session.client)
                session.assert_healthy()
        launcher.assert_called_once_with()
        self.assertEqual([30], handle.wait_ready_calls)
        self.assertEqual(1, handle.stop_calls)


class RuntimeAdapterTest(unittest.TestCase):

    def test_launch_topology_is_accepted_by_engine_parallelism_setup(self):
        from rtp_llm.config.server_config_setup import set_parallelism_config
        from rtp_llm.ops import ParallelismConfig

        for topology in (dict(tp=1, dp=2), dict(tp=2, dp=1), dict(tp=2, dp=1, ep=1)):
            with self.subTest(topology=topology):
                spec = build_engine_launch_spec(runtime_adapter_plan(topology=topology))
                options = dict((arg[2:].split("=", 1) for arg in spec.argv))
                world_size = int(options["world_size"])
                for rank in range(world_size):
                    config = ParallelismConfig()
                    for name in (
                        "tp_size",
                        "dp_size",
                        "pp_size",
                        "ep_size",
                        "world_size",
                    ):
                        setattr(config, name, int(options[name]))
                    config.local_world_size = world_size
                    set_parallelism_config(config, world_rank=rank)
                    self.assertEqual(int(options["ep_size"]), config.ep_size)
                    self.assertEqual(rank % config.tp_size, config.tp_rank)
                    self.assertEqual(
                        rank // config.tp_size % config.dp_size, config.dp_rank
                    )
                    self.assertEqual(
                        rank // (config.tp_size * config.dp_size), config.pp_rank
                    )

    def test_launch_spec_is_derived_only_from_resolved_plan(self):
        spec = build_engine_launch_spec(runtime_adapter_plan())
        self.assertEqual((3, 5), spec.device_ids)
        self.assertEqual("false", spec.environment["TOKENIZERS_PARALLELISM"])
        self.assertEqual("2", spec.environment["WORLD_SIZE"])
        self.assertEqual("1", spec.environment["USE_BATCH_DECODE_SCHEDULER"])
        self.assertIn("--model_type=qwen_2", spec.argv)
        self.assertIn("--checkpoint_path=/models/qwen", spec.argv)
        self.assertIn("--tokenizer_path=/models/qwen-tokenizer", spec.argv)
        self.assertIn("--tp_size=2", spec.argv)
        self.assertIn("--ep_size=2", spec.argv)
        self.assertIn("--pp_size=1", spec.argv)
        self.assertIn("--max_batch_tokens_size=8192", spec.argv)
        self.assertIn("--enable_cuda_graph=0", spec.argv)

    def test_launch_spec_preserves_pipeline_topology(self):
        plan = runtime_adapter_plan()
        config = thaw(plan.resolved_config)
        config["model"]["topology"] = {"tp": 1, "dp": 1, "pp": 2, "ep": 1}
        with self.assertRaisesRegex(PlanningError, "topology.pp=1"):
            build_engine_launch_spec(replace(plan, resolved_config=config))

    def test_pipeline_size_cannot_be_overridden_in_engine_args(self):
        plan = runtime_adapter_plan()
        config = thaw(plan.resolved_config)
        config["engine"]["args"]["pp_size"] = 2
        with self.assertRaisesRegex(PlanningError, "pp_size.*duplicates"):
            build_engine_launch_spec(replace(plan, resolved_config=config))

    def test_http_prefill_client_projects_server_timing_and_reuse(self):
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "aux_info": {
                "input_len": 128,
                "output_len": 1,
                "reuse_len": 64,
                "first_token_cost_time": 12.5,
                "wait_time": 1.25,
            }
        }
        post = Mock(return_value=response)
        client = HttpPrefillClient(
            "http://127.0.0.1:18000", post=post, timeout_seconds=30
        )
        result = client.request(
            phase="measure",
            case_id="case-a",
            request_index=0,
            run_index=1,
            prompt="hello",
            input_ids=[1, 2],
            output_length=1,
        )
        self.assertEqual("http://127.0.0.1:18000", post.call_args.args[0])
        self.assertEqual(
            {
                "success": True,
                "input_len": 128,
                "output_len": 1,
                "reuse_len": 64,
                "latency_ms": 12.5,
                "ttft_ms": 12.5,
                "wait_time_ms": 1.25,
            },
            result,
        )
        request = post.call_args.kwargs
        self.assertEqual("hello", request["json"]["prompt"])
        self.assertEqual(1, request["json"]["generate_config"]["max_new_tokens"])
        self.assertNotIn("input_ids", request["json"])
        self.assertEqual(30, request["timeout"])
        response.json.return_value = {"status": "ok", "aux_info": {}}
        for seconds in (10, 7200):
            with self.subTest(seconds=seconds):
                client = HttpPrefillClient(
                    "http://127.0.0.1:18000", post=post, timeout_seconds=seconds
                )
                client.configure_scheduler(1)
                self.assertEqual(min(seconds, 300), post.call_args.kwargs["timeout"])
                client.request(
                    phase="warmup",
                    case_id="case-a",
                    request_index=0,
                    run_index=None,
                    prompt="hello",
                    input_ids=[1],
                    output_length=1,
                )
                self.assertEqual(seconds, post.call_args.kwargs["timeout"])
        from rtp_llm.test.perf_test.prefill.runner import PrefillRunner

        self.assertEqual(
            ("request_failed", "deadline exceeded"),
            PrefillRunner._validate(
                {"success": False, "error": "deadline exceeded"},
                input_len=1,
                expected_reuse=0,
            ),
        )


class ExecutionSchedulerTest(unittest.TestCase):

    def test_requires_explicit_success_acknowledgement(self):
        response = Mock()
        response.json.return_value = {"status": "error", "error": "bad mode"}
        post = Mock(return_value=response)
        with self.assertRaisesRegex(RuntimeError, "scheduler rejected"):
            configure_scheduler(1234, batch_size=2, mode="decode", post=post)
        response.raise_for_status.assert_called_once_with()


class MemberLaunchTest(unittest.TestCase):

    def make_plan(self, directory):
        source = json.loads(
            (PERF_TEST_ROOT / "examples/prefill_local.json").read_text()
        )
        source["model"]["topology"] = {"tp": 2, "dp": 1, "pp": 1, "ep": 1}
        source["execution"] = {
            "backend": "ssh",
            "hosts": [{"ssh_target": "gpu", "advertise_host": "192.0.2.11"}],
            "gpus_per_machine": 2,
            "source": {
                "build_policy": "reuse",
                "worktree": "/frozen/source",
                "ref": "a" * 40,
            },
        }
        with patch(
            "rtp_llm.test.perf_test.ssh.source.resolve_source",
            side_effect=lambda value, **kwargs: dict(value),
        ):
            (plan, frozen_source) = plan_from_source(
                source,
                source_dir=directory,
                result_directory=str(directory / "controller"),
            )
        return (plan, frozen_source)

    def test_launch_spec_has_only_this_members_device_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            (plan, _source) = self.make_plan(Path(directory))
            self.assertEqual((0, 1), build_engine_launch_spec(plan).device_ids)

    def test_locked_physical_devices_reach_server_without_mutating_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (plan, source) = self.make_plan(root)
            original = plan.to_dict()
            descriptor = dict(
                run_id="member-test",
                member_id="0",
                member_index=0,
                members=[dict(hostname="192.0.2.11", port=12333)],
                world_rank=0,
                local_world_size=2,
                devices=[0, 1],
                start_port=12333,
                log_dir=str(root / "logs"),
                result_directory=str(root / "worker"),
            )
            configs = SimpleNamespace(
                parallelism_config=object(),
                prefill_cp_config=object(),
                model_args=SimpleNamespace(tokenizer_path="/m"),
            )
            handle = Mock(tokenizer_path="/m", log_file_path=None)
            handle.start_server.return_value = True
            with patch.dict("os.environ", {"RTP_LLM_GPU_LOCKED_DEVICES": "3,5"}), patch(
                "rtp_llm.server.server_args.server_args.setup_args",
                return_value=configs,
            ), patch(
                "rtp_llm.config.server_config_setup.set_parallelism_config"
            ), patch(
                "rtp_llm.test.perf_test.execution.distributed.rank_group.start_local_rank_group",
                return_value=handle,
            ) as start, patch(
                "rtp_llm.test.perf_test.execution.distributed.prefill.execute_master_prefill",
                return_value=SimpleNamespace(state=RunState.COMPLETED),
            ):
                self.assertEqual(
                    0, run_member(plan, source_config=source, descriptor=descriptor)
                )
                environment = start.call_args.kwargs["engine_environment"]
                self.assertEqual("3,5", environment["CUDA_VISIBLE_DEVICES"])
                self.assertEqual("3,5", environment["HIP_VISIBLE_DEVICES"])
            self.assertEqual(original, plan.to_dict())
            handle.stop_server.assert_called_once()


if __name__ == "__main__":
    unittest.main()
