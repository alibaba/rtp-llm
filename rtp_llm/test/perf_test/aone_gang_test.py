import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test.aone.config_schema import parse_execution
from rtp_llm.test.perf_test.aone.gang_worker import _merge_workload, load_gang_plan
from rtp_llm.test.perf_test.aone.plan import plan_resources
from rtp_llm.test.perf_test.capabilities import validate_aone_capability
from rtp_llm.test.perf_test.config_resolver import load_resolved_config
from rtp_llm.test.perf_test.execution.aone_bazel import create_default_aone_executor
from rtp_llm.test.perf_test.execution.aone_gang import (
    BROKER_TOKEN_ENV,
    GangAOneExecutor,
)
from rtp_llm.test.perf_test.plan import build_execution_plan, thaw
from rtp_llm.test.perf_test.results import RESULT_SCHEMA_VERSION, parse_result_envelope
from rtp_llm.test.perf_test.run_directory import (
    initialize_run_directory,
    inspect_run_directory,
)
from rtp_llm.test.perf_test.state import RunState


def _gang_source():
    return {
        "schema_version": 1,
        "model": {
            "label": "gang",
            "type": "registered",
            "checkpoint": "/m",
            "tokenizer": "/m",
            "topology": dict(TOPOLOGY),
        },
        "engine": {
            "args": {"reuse_cache": 1},
            "max_seq_len": 4097,
            "max_batch_tokens": 8192,
            "concurrency_limit": 1,
        },
        "workload": {
            "kind": "cache_grid",
            "max_input_len": 2048,
            "cache_alignment": 64,
            "commit_tail_tokens": 64,
            "compute_steps": [256],
            "batch_sizes": [1],
        },
        "execution": {
            "backend": "aone",
            "resource": "daily",
            "gpu_type": "H20",
            "max_parallel": MACHINE_COUNT,
            "gpus_per_machine": LOCAL_WORLD_SIZE,
        },
        "output": {},
        "regression": {},
        "publish": {},
    }


def _cases():
    return [
        {
            "input_len": 512,
            "cache_len": 0,
            "compute_len": 512,
            "batch_size": 1,
            "measure_runs": 1,
            "seed": 1,
            "prefix_policy": "shared_seed",
        },
        {
            "input_len": 1024,
            "cache_len": 512,
            "compute_len": 512,
            "batch_size": 1,
            "measure_runs": 1,
            "seed": 1,
            "prefix_policy": "shared_seed",
        },
    ]


def _setup_run(root, source=None):
    source = _gang_source() if source is None else source
    root.parent.mkdir(parents=True, exist_ok=True)
    config_path = root.parent / "submit-host-only.json"
    config_path.write_text(json.dumps(source))
    plan = build_execution_plan(
        resource_planner=plan_resources,
        capability_validator=validate_aone_capability,
        mode="prefill",
        action="run",
        resolved=load_resolved_config(config_path, execution_parser=parse_execution),
        cases=_cases(),
        result_directory=str(root),
    )
    initialize_run_directory(
        root, source_config=source, plan=plan, run_id=RUN_ID, code_version="rev"
    )
    config_path.unlink()
    return plan


def _case_result(case, status="ok"):
    identity = thaw(case.identity)
    ok = status == "ok"
    return {
        "case_id": case.case_id,
        "identity": identity,
        "status": status,
        "runs": 1,
        "errors": [] if ok else [{"code": status, "message": "remote case failed"}],
        "metrics": (
            {
                "input_length": identity["input_len"],
                "expected_reuse_length": identity["cache_len"],
                "observed_reuse_length": identity["cache_len"],
                "compute_length": identity["compute_len"],
                "latency_ms": 10.0,
                "throughput_tokens_per_second": 1000.0,
                "validation": {"shape": True, "reuse": True, "timing": True},
                "seed": 1,
                "prefix_policy": {"kind": "shared_seed"},
                "measurements": [{"run": 0, "latency_ms": 10.0}],
            }
            if ok
            else None
        ),
    }


def _envelope(plan, cases, status):
    successful = sum((case["status"] == "ok" for case in cases))
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": RUN_ID,
        "mode": "prefill",
        "backend": "aone",
        "status": status,
        "config_fingerprint": plan.config_fingerprint,
        "workload_fingerprint": plan.workload_fingerprint,
        "model": {
            "label": "gang",
            "type": "registered",
            "checkpoint": "/m",
            "topology": dict(TOPOLOGY),
        },
        "summary": {
            "planned_cases": len(cases),
            "successful_cases": successful,
            "failed_cases": len(cases) - successful,
        },
        "cases": cases,
    }


def _write_master_run_tree(outputs, plan, status="completed"):
    cases = [_case_result(case) for case in plan.cases]
    return _write_master_run_tree_with_cases(outputs, plan, cases, status)


def _write_master_run_tree_with_cases(outputs, plan, cases, status):
    run = outputs / "run"
    (run / "results").mkdir(parents=True)
    (run / "results" / "cases.jsonl").write_text(
        "".join((json.dumps(case) + "\n" for case in cases)), encoding="utf-8"
    )
    (run / "results" / "result.json").write_text(
        json.dumps(_envelope(plan, cases, status)), encoding="utf-8"
    )
    (run / "results" / "summary.json").write_text(
        json.dumps({"run_id": RUN_ID, "status": status}), encoding="utf-8"
    )
    (run / "reports").mkdir()
    (run / "reports" / "master_report.json").write_text("{}", encoding="utf-8")
    (run / "traces").mkdir()
    (run / "traces" / "master_trace.json").write_text("{}", encoding="utf-8")
    (run / "profiles").mkdir()
    (run / "profiles" / "master_profile.json").write_text("{}", encoding="utf-8")
    return run


def _write_testlogs_tree(base, plan, *, status="completed", master=True):
    for index in range(MACHINE_COUNT):
        shard = base / f"shard_{index + 1}_of_{MACHINE_COUNT}"
        outputs = shard / "test.outputs"
        outputs.mkdir(parents=True)
        (shard / "test.log").write_text(f"shard {index} log\n", encoding="utf-8")
        (outputs / "gang_worker.json").write_text(
            json.dumps({"shard_index": index, "state": "completed"}), encoding="utf-8"
        )
        if index == 0 and master:
            _write_master_run_tree(outputs, plan, status=status)


TOPOLOGY = {"tp": 8, "dp": 2, "pp": 1, "ep": 16}

MACHINE_COUNT = 2

LOCAL_WORLD_SIZE = 8

RUN_ID = "run-gang"


class BrokerProxyTest(unittest.TestCase):
    def test_token_bearing_requests_ignore_all_ambient_proxy_settings(self):
        import requests

        from rtp_llm.test.perf_test.aone.gang_worker import BrokerClient

        for uppercase in (True, False):
            for scheme in ("http", "https"):
                with self.subTest(uppercase=uppercase, scheme=scheme):
                    proxy_env = {
                        name.upper() if uppercase else name: "http://proxy.invalid:8080"
                        for name in ("http_proxy", "https_proxy", "all_proxy")
                    }
                    sent = []

                    def send(session, request, **kwargs):
                        sent.append((request, kwargs))
                        response = requests.Response()
                        response.status_code = 200
                        response._content = b'{"status": "ok"}'
                        return response

                    with patch.dict("os.environ", proxy_env, clear=True), patch(
                        "requests.sessions.Session.send",
                        autospec=True,
                        side_effect=send,
                    ):
                        broker = BrokerClient(
                            f"{scheme}://broker.invalid:18000",
                            "test-secret",
                            "run-test",
                        )
                        broker.heartbeat(0)
                    self.assertEqual(1, len(sent))
                    request, settings = sent[0]
                    self.assertEqual({}, settings["proxies"])
                    self.assertEqual(
                        "Bearer test-secret", request.headers["Authorization"]
                    )
                    self.assertEqual(
                        f"{scheme}://broker.invalid:18000/v1/runs/run-test/heartbeat",
                        request.url,
                    )
                    self.assertEqual({"shard_index": 0}, json.loads(request.body))


class GangExecutorTest(unittest.TestCase):
    def test_master_collection_rejects_results_from_another_run_or_plan(self):
        mutations = {
            "run_id": lambda result: result.update(run_id="other-run"),
            "config": lambda result: result.update(config_fingerprint="a" * 64),
            "workload": lambda result: result.update(workload_fingerprint="b" * 64),
            "case": lambda result: result["cases"][0].update(case_id="other-case"),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                root = workspace / "run"
                plan = _setup_run(root)
                outputs = workspace / "outputs"
                _write_master_run_tree(outputs, plan)
                path = outputs / "run/results/result.json"
                result = json.loads(path.read_text())
                mutate(result)
                path.write_text(json.dumps(result))
                executor = GangAOneExecutor(
                    workspace_root=workspace,
                    resource="daily",
                    gpu_type="H20",
                    max_parallel=2,
                )
                from rtp_llm.test.perf_test.errors import CollectionError

                with self.assertRaises(CollectionError):
                    executor._collect_master_tree(root, outputs)
                self.assertFalse((root / "results/result.json").exists())

    def test_gang_collection_retains_nonmaster_decode_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / "outputs"
            trace_dir = outputs / "TEST_OUTPUT_run_1"
            trace_dir.mkdir(parents=True)
            (trace_dir / "decode_in8_out2_bs2_case_1.json").write_text(
                json.dumps({"traceEvents": []}), encoding="utf-8"
            )
            GangAOneExecutor._copy_worker_traces(outputs, root / "traces", 1)
            self.assertTrue(
                (root / "traces/shard_2/decode_in8_out2_bs2_case_1.json").is_file()
            )

    def _gang_flow(
        self,
        workspace,
        status="completed",
        remote_exit=0,
        member_failure=False,
        missing_master=False,
    ):
        root = workspace / "runs/gang"
        plan = _setup_run(root)
        base = workspace / "bazel-testlogs/runs/gang/workload/aone_package/gang"
        _write_testlogs_tree(base, plan, status=status, master=not missing_master)
        trace = base / "shard_2_of_2/test.outputs/TEST_OUTPUT_worker/decode_trace.json"
        trace.parent.mkdir()
        trace.write_text(json.dumps({"traceEvents": []}))
        if status == "partial":
            result = base / "shard_1_of_2/test.outputs/run/results/result.json"
            cases = [
                _case_result(plan.cases[0]),
                _case_result(plan.cases[1], "request_failed"),
            ]
            result.write_text(json.dumps(_envelope(plan, cases, status)))
            (result.parent / "cases.jsonl").write_text(
                "".join((json.dumps(case) + "\n" for case in cases))
            )
        calls = []
        brokers = []
        from rtp_llm.test.perf_test.aone.broker import run_gang_broker

        def broker_factory(*args, **kwargs):
            broker = run_gang_broker(*args, **kwargs)
            brokers.append(broker)
            return broker

        def run(command, **kwargs):
            calls.append((command, kwargs))
            if "test" in command:
                broker = brokers[0]
                for index in range(2):
                    broker.register_member(
                        shard_index=index,
                        hostname=f"host-{index}",
                        port=18000,
                        address=f"127.0.0.{index + 1}",
                    )
                broker.record_result(shard_index=0, state="completed")
                broker.record_result(
                    shard_index=1, state="failed" if member_failure else "completed"
                )
            return SimpleNamespace(returncode=remote_exit if "test" in command else 0)

        def executor(**kwargs):
            return GangAOneExecutor(**kwargs, run_command=run, broker_host="127.0.0.1")

        with patch(
            "rtp_llm.test.perf_test.execution.aone_gang.run_gang_broker",
            side_effect=broker_factory,
        ), patch(
            "rtp_llm.test.perf_test.execution.aone_gang._workspace_root",
            return_value=workspace,
        ), patch(
            "rtp_llm.test.perf_test.execution.aone_gang.GangAOneExecutor",
            side_effect=executor,
        ):
            outcome = create_default_aone_executor(plan)
        return (root, plan, outcome, calls)

    def test_failed_member_overrides_successful_bazel_and_master_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan, outcome, _ = self._gang_flow(
                Path(directory), member_failure=True
            )
            self.assertEqual(RunState.FAILED, outcome.state)
            result = parse_result_envelope(
                json.loads((root / "results/result.json").read_text())
            )
            self.assertEqual(RunState.FAILED, result.status)

    def test_missing_master_still_collects_other_members_logs_and_traces(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan, outcome, _ = self._gang_flow(
                Path(directory), missing_master=True
            )
            self.assertEqual(RunState.FAILED, outcome.state)
            self.assertTrue((root / "workload/gang_shards/shard_2/test.log").is_file())
            self.assertTrue((root / "traces/shard_2/decode_trace.json").is_file())

    def test_gang_submission_preserves_package_inputs_artifacts_and_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            (root, plan, outcome, calls) = self._gang_flow(Path(directory))
            self.assertEqual(RunState.COMPLETED, outcome.state)
            self.assertEqual(0, outcome.exit_code)
            self.assertEqual("formal", outcome.report["report_kind"])
            self.assertEqual(2, len(outcome.collection.cases))
            package = root / "workload/aone_package"
            gang = load_gang_plan(package / "gang_plan.json")
            (root / "remote-inputs").mkdir()
            merged = _merge_workload(
                package / "resolved_config.json",
                package / "resolved_workload.json",
                root / "remote-inputs",
            )
            loaded = load_resolved_config(merged, execution_parser=parse_execution)
            self.assertEqual(plan.config_fingerprint, loaded.config_fingerprint)
            self.assertEqual(16, gang["world_size"])
            self.assertEqual(2, gang["machine_count"])
            self.assertNotIn(
                "shard_count",
                json.loads((package / "resolved_config.json").read_text())["execution"],
            )
            self.assertTrue((root / "traces/master_trace.json").is_file())
            self.assertTrue((root / "profiles/master_profile.json").is_file())
            self.assertTrue((root / "workload/gang_shards/shard_2/test.log").is_file())
            self.assertTrue(
                all(
                    (
                        stage["status"] == "completed"
                        for stage in inspect_run_directory(root)
                        .manifest["stages"]
                        .values()
                    )
                )
            )
            (command, kwargs) = calls[1]
            token = kwargs["env"][BROKER_TOKEN_ENV]
            self.assertNotIn(token, " ".join(command))
            for path in root.rglob("*"):
                if path.is_file():
                    self.assertNotIn(token, path.read_text(errors="replace"))

    def test_nonzero_gang_exit_preserves_partial_results_and_overrides_completion(self):
        for status in ("partial", "completed"):
            with self.subTest(
                master_status=status
            ), tempfile.TemporaryDirectory() as directory:
                (root, plan, outcome, _) = self._gang_flow(
                    Path(directory), status=status, remote_exit=3
                )
                self.assertEqual(
                    RunState.PARTIAL if status == "partial" else RunState.FAILED,
                    outcome.state,
                )
                self.assertNotEqual(0, outcome.exit_code)
                self.assertEqual(
                    "failed",
                    inspect_run_directory(root).manifest["stages"]["submit"]["status"],
                )
                persisted = parse_result_envelope(
                    json.loads((root / "results/result.json").read_text())
                )
                self.assertEqual(outcome.state, persisted.status)
                self.assertEqual(2, len(persisted.cases))
                self.assertEqual("ok", persisted.cases[0].status)
                if status == "completed":
                    self.assertIn("Bazel gang test exited 3", str(outcome.failure))


from rtp_llm.test.perf_test.aone.broker import BrokerMember
from rtp_llm.test.perf_test.aone.gang_worker import (
    EXIT_RENDEZVOUS_ERROR,
    GangBrokerFailure,
    _default_decode_worker,
    _default_master_executor,
    _default_prefill_master_executor,
    _default_server_runner,
    build_worker_context,
    run_gang_worker,
)


class GangWorkerTest(unittest.TestCase):

    def test_worker_acknowledges_bazel_machine_sharding(self):
        status = self.root / "shard.status"
        with patch.dict("os.environ", {"TEST_SHARD_STATUS_FILE": str(status)}):
            self.assertEqual(0, self._run(0, lambda *args, **kwargs: 0))
        self.assertTrue(status.is_file())

    def test_broker_token_is_not_in_rank_process_environment_or_failure_artifacts(self):
        import os

        secret = "test-broker-secret-do-not-persist"

        def launch(context):
            self.assertNotIn(BROKER_TOKEN_ENV, os.environ)
            return self.handle

        with patch.dict(
            "os.environ",
            {
                BROKER_TOKEN_ENV: secret,
                "TEST_SHARD_INDEX": "0",
                "TEST_TOTAL_SHARDS": "2",
            },
        ):
            code = run_gang_worker(
                self.config_path,
                self.plan_path,
                broker=self.broker,
                server_runner=launch,
                test_outputs_dir=self.root / "outputs",
                master_executor=lambda *args, **kwargs: 0,
            )
        self.assertEqual(0, code)
        self.assertNotIn(secret, (self.root / "outputs/gang_worker.json").read_text())

    def test_decode_worker_resolves_its_own_tokenizer(self):
        self._example("decode")
        handle = Mock(tokenizer_path=None)
        with patch(
            "rtp_llm.test.perf_test.aone.gang_worker._resolve_model_paths",
            return_value={"tokenizer": "/local/tokenizer"},
        ) as resolve, patch(
            "rtp_llm.test.perf_test.test_util.create_query",
            return_value={8: "prompt"},
        ) as create, patch(
            "rtp_llm.test.perf_test.multi_node.decode_sync.serve_decode_worker"
        ) as serve:
            _default_decode_worker(
                handle,
                context=build_worker_context(self.gang, self.members, 1),
                config_path=str(self.config_path),
            )
        resolve.assert_called_once()
        self.assertEqual("/local/tokenizer", create.call_args.kwargs["tokenizer_path"])
        serve.assert_called_once()

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.gang = {
            "schema_version": 1,
            "run_id": "gang-core",
            "config_fingerprint": "fp",
            "workload_fingerprint": "wf",
            "world_size": 2,
            "machine_count": 2,
            "local_world_size": 1,
            "gpus_per_machine": 1,
            "gpus_per_host": 8,
            "gpu_type": "H20_CU13",
            "topology": {"tp": 2, "dp": 1, "pp": 1, "ep": 1},
            "master_shard_index": 0,
            "port_range": {"start": 18000, "count": 1},
        }
        self.members = [
            BrokerMember(index, f"10.0.0.{index}", f"host-{index}", 18000)
            for index in range(2)
        ]
        self.context = build_worker_context(self.gang, self.members, 0)
        self.handle = Mock(
            tokenizer_path="/m",
            log_file_path=str(self.root / "server.log"),
            log_dir=str(self.root / "logs"),
        )
        self.handle.start_server.return_value = True
        self.broker = Mock(gang_state="ready")
        self.broker.members.return_value = self.members
        self.plan_path = self.root / "gang.json"
        self.plan_path.write_text(json.dumps(self.gang))
        self.config_path = self.root / "config.json"
        self.config_path.write_text("{}")
        self.addCleanup(self._assert_no_heartbeats)

    def _assert_no_heartbeats(self):
        self.assertEqual(
            [],
            [
                t
                for t in threading.enumerate()
                if t.name.startswith("gang-worker-heartbeat") and t.is_alive()
            ],
        )

    def _example(self, mode):
        source = _gang_source()
        source["model"]["topology"] = {"tp": 2, "dp": 1, "pp": 1, "ep": 1}
        source["execution"]["gpus_per_machine"] = 1
        if mode == "decode":
            source["workload"] = {
                "kind": "grid",
                "input_lengths": [8],
                "output_lengths": [2],
                "batch": {"strategy": "fixed", "sizes": [1]},
                "warmup_runs": 0,
                "measure_runs": 1,
                "profiling": {"enabled": False},
            }
        source["engine"].update(
            max_seq_len=64, max_batch_tokens=64, concurrency_limit=1
        )
        if mode == "prefill":
            source["workload"].update(
                max_input_len=16,
                cache_alignment=4,
                commit_tail_tokens=4,
                compute_steps=[4],
                cache_lengths=[0],
                batch_sizes=[1],
            )
        else:
            source["workload"].update(
                input_lengths=[8],
                output_lengths=[2],
                warmup_runs=0,
                measure_runs=1,
                profiling={"enabled": False},
            )
        self.config_path.write_text(json.dumps(source))
        return source

    def test_model_startup_propagates_args_rank_layout_and_disables_cross_host_all_reduce(
        self,
    ):
        self._example("prefill")
        context = build_worker_context(self.gang, self.members, 1)
        runner = Mock()
        runner.LocalRankGroupContext.side_effect = lambda **kwargs: SimpleNamespace(
            **kwargs
        )
        configs = SimpleNamespace(
            parallelism_config=object(), prefill_cp_config=object()
        )
        with patch.dict(
            "os.environ",
            {
                "PERF_TEST_CONFIG_PATH": str(self.config_path),
                "TEST_UNDECLARED_OUTPUTS_DIR": str(self.root),
            },
        ), patch(
            "rtp_llm.test.perf_test.aone.gang_worker._load_local_server_runner",
            return_value=runner,
        ), patch(
            "rtp_llm.test.perf_test.aone.gang_worker._resolve_model_paths",
            return_value={
                "type": "qwen_2",
                "checkpoint": "/resolved/model",
                "tokenizer": "/resolved/tokenizer",
            },
        ), patch(
            "rtp_llm.server.server_args.server_args.setup_args", return_value=configs
        ) as setup, patch(
            "rtp_llm.config.server_config_setup.set_parallelism_config"
        ):
            _default_server_runner(context)
            import os

            self.assertEqual("1", os.environ["FT_DISABLE_CUSTOM_AR"])
        (args, kwargs) = runner.start_local_rank_group.call_args
        self.assertEqual(1, args[0].world_rank)
        self.assertEqual("0", context.cuda_visible_devices)
        self.assertIn("host-1", context.gang_config_string)
        self.assertIn("--tp_size=2", kwargs["engine_argv"])
        self.assertIn("--reuse_cache=1", kwargs["engine_argv"])
        self.assertIn("--checkpoint_path=/resolved/model", kwargs["engine_argv"])
        self.assertIn("--tokenizer_path=/resolved/tokenizer", kwargs["engine_argv"])
        self.assertEqual(list(kwargs["engine_argv"]), setup.call_args.args[0])
        self.assertEqual("2", kwargs["engine_environment"]["WORLD_SIZE"])

    def test_prefill_master_executes_cases_probe_and_finalizes_owned_results(self):
        source = self._example("prefill")
        client = Mock()

        def request(**kwargs):
            return {
                "success": True,
                "input_len": len(kwargs["input_ids"]),
                "output_len": 1,
                "reuse_len": (
                    4
                    if kwargs["case_id"].startswith("cache_probe_")
                    and kwargs["phase"] == "measure"
                    else 0
                ),
                "latency_ms": 10.0,
            }

        client.request.side_effect = request
        session = SimpleNamespace(client=client, close=Mock())
        output = self.root / "prefill"
        with patch(
            "rtp_llm.test.perf_test.aone.gang_worker._resolve_model_paths",
            return_value=source["model"],
        ), patch(
            "rtp_llm.test.perf_test.test_util._load_tokenizer",
            return_value=SimpleNamespace(encode=lambda text: text.split()),
        ), patch(
            "rtp_llm.test.perf_test.multi_node.prefill_runtime.ExternalPrefillSession",
            return_value=session,
        ):
            code = _default_prefill_master_executor(
                self.handle,
                context=self.context,
                config_path=str(self.config_path),
                result_directory=str(output),
            )
        self.assertEqual(0, code)
        parsed = parse_result_envelope(
            json.loads((output / "results/result.json").read_text())
        )
        self.assertEqual(RunState.COMPLETED, parsed.status)
        self.assertTrue(parsed.cases)
        self.assertTrue(all((case.status == "ok" for case in parsed.cases)))
        self.assertTrue((output / "checkpoints/prefill_cache_probe.json").is_file())
        session.close.assert_called_once()

    def test_decode_master_executes_and_finalizes_aone_envelope(self):
        self._example("decode")
        transport = Mock()
        transport.run_round.return_value = (
            {
                "request_index": 0,
                "input_length": 8,
                "output_length": 2,
                "success": True,
                "latency_ms": {
                    "wait": 0.0,
                    "prefill": 2.0,
                    "decode": 8.0,
                    "tpot": 4.0,
                    "total": 10.0,
                },
            },
        )
        output = self.root / "decode"
        with patch(
            "rtp_llm.test.perf_test.test_util.create_query", return_value={8: "prompt"}
        ), patch(
            "rtp_llm.test.perf_test.multi_node.decode_runtime.DecodeHttpTransport",
            return_value=transport,
        ) as factory:
            code = _default_master_executor(
                self.handle,
                context=self.context,
                config_path=str(self.config_path),
                result_directory=str(output),
            )
        self.assertEqual(0, code)
        parsed = parse_result_envelope(
            json.loads((output / "results/result.json").read_text())
        )
        self.assertEqual("decode", parsed.mode)
        self.assertEqual("aone", parsed.backend)
        self.assertTrue(parsed.cases)
        self.assertTrue(all((case.status == "ok" for case in parsed.cases)))
        self.assertEqual("http://127.0.0.1:18000", factory.call_args.kwargs["endpoint"])
        transport.configure_scheduler.assert_called()

    def test_rank_batch_decode_master_uses_cluster_transport(self):
        self._check_decode_master_cluster(
            {
                "kind": "grid",
                "input_lengths": [8],
                "batch": {"strategy": "fixed", "sizes": [1], "unit": "per_dp_rank"},
            }
        )

    def test_global_distribution_and_target_master_use_cluster_transport(self):
        for workload in (
            {
                "kind": "grid",
                "input_lengths": [8],
                "batch": {"strategy": "fixed", "sizes": [2], "unit": "global"},
            },
            {
                "kind": "grid",
                "input_lengths": [8],
                "batch": {
                    "strategy": "target_tpot",
                    "max_size": 4,
                    "candidate_step": 1,
                    "target_tpot_ms": 10,
                },
            },
            {
                "kind": "distribution",
                "distribution_file": "not-on-worker.json",
                "batch": {"strategy": "fixed", "sizes": [2]},
            },
        ):
            with self.subTest(kind=workload["kind"], batch=workload["batch"]):
                self._check_decode_master_cluster(workload)

    def _check_decode_master_cluster(self, workload):
        source = self._example("decode")
        source["model"]["topology"] = {"tp": 1, "dp": 2, "pp": 1, "ep": 2}
        source["workload"] = workload
        self.config_path.write_text(json.dumps(source))
        self.gang["topology"] = source["model"]["topology"]
        context = build_worker_context(self.gang, self.members, 0)
        plan = SimpleNamespace(cases=[SimpleNamespace(identity={"input_lengths": [8]})])
        cluster = Mock()
        with patch(
            "rtp_llm.test.perf_test.multi_node.decode_runtime.build_external_decode_plan",
            return_value=(plan, source),
        ), patch(
            "rtp_llm.test.perf_test.test_util.create_query", return_value={8: "prompt"}
        ), patch(
            "rtp_llm.test.perf_test.multi_node.decode_sync.ClusterDecodeTransport",
            return_value=cluster,
        ), patch(
            "rtp_llm.test.perf_test.multi_node.decode_runtime.execute_master_decode",
            return_value=SimpleNamespace(state=RunState.COMPLETED),
        ) as execute:
            code = _default_master_executor(
                self.handle,
                context=context,
                config_path=str(self.config_path),
                result_directory=str(self.root / "run"),
            )
        self.assertEqual(0, code)
        cluster.connect_workers.assert_called_once()
        cluster.close.assert_called_once()
        self.assertIs(cluster, execute.call_args.kwargs["transport"])

    def _run(self, rank, executor=None):
        with patch.dict(
            "os.environ", {"TEST_SHARD_INDEX": str(rank), "TEST_TOTAL_SHARDS": "2"}
        ):
            return run_gang_worker(
                self.config_path,
                self.plan_path,
                broker=self.broker,
                server_runner=lambda context: self.handle,
                test_outputs_dir=self.root / "outputs",
                master_executor=executor,
            )

    def test_master_failure_votes_failed_stops_server_and_preserves_partial_artifacts(
        self,
    ):

        def execute(handle, *, result_directory, **kwargs):
            output = Path(result_directory) / "partial.json"
            output.parent.mkdir(parents=True)
            output.write_text("{}")
            return 7

        self.assertEqual(7, self._run(0, execute))
        self.assertEqual("failed", self.broker.report.call_args.args[1])
        self.assertTrue((self.root / "outputs/run/partial.json").is_file())
        self.handle.stop_server.assert_called_once()
        self.assertEqual(
            "failed",
            json.loads((self.root / "outputs/gang_worker.json").read_text())["state"],
        )

    def test_waiting_worker_terminates_on_gang_failure_and_stops_server(self):
        self.broker.members.side_effect = [
            self.members,
            GangBrokerFailure("member_failed"),
        ]
        self.assertEqual(EXIT_RENDEZVOUS_ERROR, self._run(1))
        self.assertEqual(2, self.broker.members.call_count)
        self.handle.start_server.assert_called_once()
        self.handle.stop_server.assert_called_once()

    def test_rank_batch_decode_worker_measures_before_completion_vote(self):
        self._check_decode_worker_dispatch("per_dp_rank")

    def test_global_decode_worker_measures_before_completion_vote(self):
        self._check_decode_worker_dispatch("global")

    def _check_decode_worker_dispatch(self, unit):
        source = self._example("decode")
        source["model"]["topology"] = {"tp": 1, "dp": 2, "pp": 1, "ep": 2}
        source["workload"]["batch"] = {
            "strategy": "fixed",
            "sizes": [1],
            "unit": unit,
        }
        self.config_path.write_text(json.dumps(source))
        self.gang["topology"] = source["model"]["topology"]
        self.plan_path.write_text(json.dumps(self.gang))
        with patch(
            "rtp_llm.test.perf_test.aone.gang_worker._default_decode_worker"
        ) as measure, patch(
            "rtp_llm.test.perf_test.aone.gang_worker._run_waiter",
            return_value=0,
        ) as waiter:
            self.assertEqual(0, self._run(1))
        measure.assert_called_once()
        waiter.assert_called_once()


class GangPackageTest(unittest.TestCase):
    def test_machine_shards_request_full_hosts_even_with_partial_gpu_use(self):
        from rtp_llm.test.perf_test.aone.package import generate_gang_package

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _gang_source()
            source["execution"]["gpus_per_machine"] = 4
            source["execution"]["max_parallel"] = 4
            plan = _setup_run(root / "run", source)
            package = generate_gang_package(root / "generated", plan, run_id=RUN_ID)
            actions = []
            namespace = {
                "native": SimpleNamespace(
                    py_test=lambda **kwargs: actions.append(kwargs)
                ),
                "load": lambda *args: None,
                "test_suite": lambda **kwargs: None,
            }
            exec((Path(__file__).parent / "aone_defs.bzl").read_text(), namespace)
            exec(package.build_file.read_text(), namespace)
            self.assertEqual(("//generated:gang",), package.targets)
            self.assertEqual(1, len(actions))
            action = actions[0]
            self.assertEqual(4, action["shard_count"])
            self.assertEqual(
                {"gpu": "H20", "gpu_count": "8"}, action["exec_properties"]
            )
            self.assertEqual(
                "//rtp_llm/test/perf_test:aone/gang_worker.py", action["main"]
            )
            self.assertTrue(
                {
                    "resolved_config.json",
                    "resolved_workload.json",
                    "gang_plan.json",
                }.issubset(action["data"])
            )
            self.assertIn(BROKER_TOKEN_ENV, action["env_inherit"])
            self.assertNotIn(BROKER_TOKEN_ENV, " ".join(action["args"]))
            payload = load_gang_plan(package.shard_plan)
            self.assertEqual(
                (16, 4, 4, 8),
                tuple(
                    payload[key]
                    for key in (
                        "world_size",
                        "machine_count",
                        "local_world_size",
                        "gpus_per_host",
                    )
                ),
            )


class FrozenGangInputsTest(unittest.TestCase):
    def test_accepted_workloads_reach_worker_startup_with_identical_plans(self):
        from rtp_llm.test.perf_test.aone import gang_worker
        from rtp_llm.test.perf_test.aone.package import generate_gang_package
        from rtp_llm.test.perf_test.cli import parse_cli
        from rtp_llm.test.perf_test.entry import build_request_plan

        examples = Path(__file__).parent / "examples"
        shapes = (
            ("decode_distribution_local.json", "decode", 2),
            ("prefill_random_batch_local.json", "prefill", 12),
            ("prefill_traffic_distribution_local.json", "prefill", 12),
            ("prefill_local.json", "prefill", None),
        )
        for example, mode, count in shapes:
            for named in (
                (False, True)
                if "batch" in example or "traffic" in example
                else (False,)
            ):
                with self.subTest(
                    example=example, named=named
                ), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    source = json.loads((examples / example).read_text())
                    source["model"]["topology"].update(
                        tp=8,
                        dp=2 if mode == "decode" else 1,
                        ep=16 if mode == "decode" else 1,
                    )
                    source["execution"] = dict(
                        backend="aone",
                        resource="daily",
                        gpu_type="H20",
                        gpus_per_machine=8 if mode == "decode" else 4,
                        max_parallel=4,
                    )
                    workload = source["workload"]
                    spec = (
                        workload
                        if mode == "decode"
                        else workload.get("traffic_request_batches", {})
                    )
                    if "distribution_file" in spec:
                        external = root / "external.json"
                        external.write_text(
                            (examples / spec["distribution_file"]).read_text()
                        )
                        spec["distribution_file"] = str(external)
                    if named:
                        key = (
                            "traffic_request_batches"
                            if "traffic" in example
                            else "random_request_batches"
                        )
                        workload["grids"] = [
                            dict(name="sampled", **{key: workload.pop(key)})
                        ]
                    source_path = root / "source.json"
                    source_path.write_text(json.dumps(source))
                    original, _ = build_request_plan(
                        parse_cli(
                            [
                                f"--mode={mode}",
                                "--action=run",
                                f"--config={source_path}",
                            ]
                        )
                    )
                    if count is not None:
                        self.assertEqual(count, len(original.cases))
                    package = generate_gang_package(
                        root / "package", original, run_id="frozen"
                    )
                    source_path.unlink()
                    (root / "external.json").unlink(missing_ok=True)
                    outputs = root / "outputs"
                    context = SimpleNamespace(
                        gang_plan=json.loads(package.shard_plan.read_text())
                    )
                    seen = []

                    def startup(config_path, *_args, **_kwargs):
                        plan, _ = gang_worker._build_rank_group_plan(
                            str(config_path), context, outputs
                        )
                        self.assertEqual(
                            original.config_fingerprint, plan.config_fingerprint
                        )
                        self.assertEqual(
                            original.workload_fingerprint, plan.workload_fingerprint
                        )
                        self.assertEqual(original.cases, plan.cases)
                        self.assertEqual(original.resolved_config, plan.resolved_config)
                        if mode == "decode":
                            worker_context = SimpleNamespace(
                                gang_plan=context.gang_plan,
                                local_world_size=original.resource_plan.local_world_size,
                                world_rank=original.resource_plan.local_world_size,
                                members=[
                                    SimpleNamespace(shard_index=0, hostname="master")
                                ],
                            )
                            with patch(
                                "rtp_llm.test.perf_test.test_util.create_query",
                                return_value={},
                            ) as queries, patch(
                                "rtp_llm.test.perf_test.multi_node.decode_sync.serve_decode_worker"
                            ) as serve:
                                gang_worker._default_decode_worker(
                                    SimpleNamespace(tokenizer_path="/tokenizer"),
                                    context=worker_context,
                                    config_path=str(config_path),
                                )
                            self.assertEqual(
                                sorted(
                                    {
                                        n
                                        for case in plan.cases
                                        for n in case.identity["input_lengths"]
                                    }
                                ),
                                queries.call_args.kwargs["input_len_list"],
                            )
                            serve.assert_called_once()
                        for field in ("config_fingerprint", "workload_fingerprint"):
                            expected = context.gang_plan[field]
                            context.gang_plan[field] = "0" * 64
                            with self.assertRaisesRegex(Exception, field + " mismatch"):
                                gang_worker._build_rank_group_plan(
                                    str(config_path), context, outputs
                                )
                            context.gang_plan[field] = expected
                        seen.append(plan)
                        return 0

                    with patch.dict(
                        "os.environ",
                        {
                            "TEST_UNDECLARED_OUTPUTS_DIR": str(outputs),
                            "PERF_TEST_BROKER_URL": "http://localhost:1234",
                            "PERF_TEST_BROKER_TOKEN": "test",
                            "PERF_TEST_BROKER_RUN_ID": "frozen",
                        },
                    ), patch.object(
                        gang_worker, "run_gang_worker", side_effect=startup
                    ):
                        code = gang_worker.main(
                            [
                                "--resolved-config",
                                str(package.resolved_config),
                                "--resolved-workload",
                                str(package.resolved_workload),
                                "--gang-plan",
                                str(package.shard_plan),
                            ],
                            broker_factory=Mock(),
                        )
                    self.assertEqual(0, code)
                    self.assertEqual(1, len(seen))


if __name__ == "__main__":
    unittest.main()
