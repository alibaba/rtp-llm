"""Core performance behavior contracts."""

import ast
import copy
import itertools
import json
import logging
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.test.perf_test.decode.transport import DecodeHttpTransport
from rtp_llm.test.perf_test.errors import PartialRunError, PlanningError
from rtp_llm.test.perf_test.multi_node import multi_benchmark
from rtp_llm.test.perf_test.multi_node.decode_runtime import (
    build_external_decode_plan,
    execute_master_decode,
    run_external_rank,
)
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.test_support import (
    fixture_engine,
    fixture_execution,
    fixture_model,
)


def decode_multi_node_decode_source(*, inputs=(128,), outputs=(16, 64), batch=None):
    return {
        "schema_version": 1,
        "model": fixture_model(
            label="external-decode", topology={"tp": 1, "dp": 2, "pp": 1, "ep": 1}
        ),
        "engine": fixture_engine(),
        "workload": {
            "kind": "grid",
            "input_lengths": list(inputs),
            "output_lengths": list(outputs),
            "batch": batch or {"strategy": "fixed", "sizes": [1, 4]},
            "measure_runs": 1,
            "warmup_runs": 0,
        },
        "execution": fixture_execution(devices=[0, 1]),
        "output": {},
        "regression": {},
        "publish": {},
    }


def decode_multi_node_write_decode_config(root, *, dp=2):
    source = decode_multi_node_decode_source()
    source["model"]["topology"]["dp"] = dp
    source["execution"]["devices"] = list(range(dp))
    source["output"]["result_root"] = str(root / "results")
    path = root / f"decode-dp{dp}.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    return path


decode_multi_node_TOPOLOGY = {"tp": 1, "dp": 2, "pp": 1, "ep": 1}


def decode_multi_node_marker(plan, run_id, state):
    return {
        "state": state,
        "run_id": run_id,
        "world_rank": 0,
        "config_fingerprint": plan.config_fingerprint,
        "workload_fingerprint": plan.workload_fingerprint,
    }


def decode_multi_node_persist_run(plan, source, root, state, run_id):

    class Transport:

        def configure_scheduler(self, _allocations):
            pass

        def run_round(self, *, input_lengths, output_length, **_kwargs):
            return tuple(
                (
                    {
                        "request_index": index,
                        "success": state == "completed",
                        "input_length": length,
                        "output_length": output_length,
                        "error": "request failed",
                        "latency_ms": {
                            "wait": 1.0,
                            "prefill": 2.0,
                            "decode": 15.0,
                            "tpot": 1.0,
                            "total": 17.0,
                        },
                    }
                    for (index, length) in enumerate(input_lengths)
                )
            )

    plan = replace(plan, result_directory=str(root))
    with patch.dict(os.environ, {"PERF_TEST_RUN_ID": run_id}):
        with patch(
            "rtp_llm.test.perf_test.multi_node.decode_runtime.DecodeHttpTransport",
            return_value=Transport(),
        ):
            execute_master_decode(
                plan,
                source_config=source,
                endpoint="http://test",
                prompt_for_length=str,
                server_log_path=None,
                profiler_directory=root / "traces",
            )


class DecodeMultiNodeTest(unittest.TestCase):

    def test_matching_marker_without_persisted_artifacts_fails_before_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = decode_multi_node_write_decode_config(root)
            (plan, _source) = build_external_decode_plan(
                path, "unused", decode_multi_node_TOPOLOGY
            )
            task = {
                "perf_test_config": str(path),
                "tp_size": 1,
                "dp_size": 2,
                "ep_size": 1,
                "ip_lists": "node0",
                "run_user": "test",
                "ssh_port": 2222,
                "ft_sub_dir": "perf",
                "sub_cmd": "test",
            }
            actions = []

            def run(args, *, env, stdout, **_kwargs):
                actions.append(env["SUB_CMD"])
                if env["SUB_CMD"] == "test":
                    stdout.write(
                        "PERF_TEST_DECODE_RESULT "
                        + json.dumps(
                            decode_multi_node_marker(
                                plan, env["PERF_TEST_RUN_ID"], "completed"
                            )
                        )
                        + "\n"
                    )
                return subprocess.CompletedProcess(
                    args, 7 if env["SUB_CMD"] == "test" else 0
                )

            with patch.object(
                multi_benchmark, "logger", logging.getLogger("decode-launcher-test")
            ):
                with patch.object(multi_benchmark.subprocess, "run", side_effect=run):
                    with self.assertRaisesRegex(RuntimeError, "persisted"):
                        multi_benchmark.execute_task(task, str(root), {}, False, 0)
            self.assertEqual(["test", "kill", "copy"], actions)

    def test_completion_requires_current_master_and_persisted_matching_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (plan, source) = build_external_decode_plan(
                decode_multi_node_write_decode_config(root),
                str(root / "run"),
                decode_multi_node_TOPOLOGY,
            )
            log = root / "log"
            run_id = "current-attempt"
            marker = decode_multi_node_marker(plan, run_id, "completed")
            for field, value in (
                ("run_id", "stale"),
                ("world_rank", 1),
                ("config_fingerprint", "other"),
                ("workload_fingerprint", "other"),
            ):
                log.write_text(
                    "PERF_TEST_DECODE_RESULT "
                    + json.dumps({**marker, field: value})
                    + "\n"
                )
                self.assertIsNone(
                    multi_benchmark.extract_decode_state(
                        str(log), plan=plan, run_id=run_id
                    )
                )
            log.write_text("PERF_TEST_DECODE_RESULT " + json.dumps(marker) + "\n")
            self.assertEqual(
                "completed",
                multi_benchmark.extract_decode_state(
                    str(log), plan=plan, run_id=run_id
                ),
            )
            with self.assertRaisesRegex(RuntimeError, "persisted"):
                multi_benchmark.verify_collected_decode_run(
                    root / "run", plan=plan, run_id=run_id, state="completed"
                )
            decode_multi_node_persist_run(
                plan, source, root / "run", "completed", run_id
            )
            multi_benchmark.verify_collected_decode_run(
                root / "run", plan=plan, run_id=run_id, state="completed"
            )
            for artifact, field, value in (
                ("manifest.json", "run_id", "stale"),
                ("results/result.json", "run_id", "stale"),
                ("results/result.json", "config_fingerprint", "other"),
                ("results/result.json", "status", "partial"),
                ("manifest.json", None, []),
            ):
                path = root / "run" / artifact
                original = path.read_text()
                changed = json.loads(original)
                if field is None:
                    changed = value
                else:
                    changed[field] = value
                path.write_text(json.dumps(changed))
                with self.subTest(artifact=artifact, field=field):
                    with self.assertRaisesRegex(RuntimeError, "persisted"):
                        multi_benchmark.verify_collected_decode_run(
                            root / "run", plan=plan, run_id=run_id, state="completed"
                        )
                path.write_text(original)

    def test_runtime_topology_mismatch_precedes_model_setup_or_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = decode_multi_node_write_decode_config(root)
            parallelism = SimpleNamespace(
                tp_size=1, dp_size=1, pp_size=1, ep_size=1, world_rank=0
            )
            configs = SimpleNamespace(
                parallelism_config=parallelism,
                prefill_cp_config=None,
                server_config=SimpleNamespace(start_port=12333),
                model_args=SimpleNamespace(model_type="registered"),
            )
            effects = []

            def configure(_configs):
                effects.append("model-download")
                raise RuntimeError("model setup reached")

            script = Path(__file__).parent / "multi_node/local_server_runner.py"
            entry = ast.parse(script.read_text()).body[-1]
            compiled = compile(
                ast.Module(body=entry.body, type_ignores=[]), str(script), "exec"
            )
            namespace = dict(
                os=os,
                pathlib=__import__("pathlib"),
                time=__import__("time"),
                PyEnvConfigs=object,
                setup_args=lambda: configs,
                setup_and_configure_server=configure,
                set_parallelism_config=lambda *_args, **_kwargs: effects.append(
                    "topology"
                ),
                build_external_decode_plan=build_external_decode_plan,
            )
            with patch.dict(os.environ, {"PERF_TEST_CONFIG_PATH": str(path)}):
                with self.assertRaisesRegex(PlanningError, "runtime topology"):
                    exec(compiled, namespace)
            self.assertEqual(["topology"], effects)
            self.assertFalse(list(root.glob("TEST_OUTPUT*")))

    def test_launcher_retains_remote_output_when_early_diagnostic_copy_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {
                "perf_test_config": str(decode_multi_node_write_decode_config(root)),
                "tp_size": 1,
                "dp_size": 2,
                "ep_size": 1,
                "ip_lists": "node0",
                "run_user": "test",
                "ssh_port": 2222,
                "ft_sub_dir": "perf",
                "sub_cmd": "test",
            }
            actions = []

            def run(args, *, env, check, **_kwargs):
                actions.append(env["SUB_CMD"])
                code = 7 if env["SUB_CMD"] in ("test", "copy") else 0
                if check and code:
                    raise subprocess.CalledProcessError(code, args)
                return subprocess.CompletedProcess(args, code)

            with patch.object(
                multi_benchmark, "logger", logging.getLogger("decode-launcher-test")
            ):
                with patch.object(multi_benchmark.subprocess, "run", side_effect=run):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "diagnostic collection failed; remote output retained",
                    ):
                        multi_benchmark.execute_task(task, str(root), {}, True, 0)
            self.assertEqual(["test", "kill", "copy"], actions)
            self.assertFalse((root / "Task_/run").exists())

    def test_launcher_snapshots_configs_and_collects_finalized_runs_but_propagates_rank_failure(
        self,
    ):
        for state, kind in itertools.product(
            ("completed", "partial"), ("grid", "distribution")
        ):
            with self.subTest(
                state=state, kind=kind
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = decode_multi_node_decode_source()
                if kind == "distribution":
                    source["workload"] = {
                        "kind": "distribution",
                        "distribution_file": "requests.json",
                        "output_length": 16,
                        "batch": {"strategy": "fixed", "sizes": [1, 4]},
                        "measure_runs": 1,
                        "warmup_runs": 0,
                    }
                (root / "requests.json").write_text(
                    json.dumps({"sequence_lengths": [32, 64, 96, 128]})
                )
                path = root / "config.json"
                path.write_text(json.dumps(source))
                (expected, _) = build_external_decode_plan(
                    path, "unused", decode_multi_node_TOPOLOGY
                )
                task = {
                    "perf_test_config": str(path),
                    "tp_size": 1,
                    "dp_size": 2,
                    "ep_size": 1,
                    "ip_lists": "node0",
                    "run_user": "test",
                    "ssh_port": 2222,
                    "ft_sub_dir": "perf",
                    "sub_cmd": "test",
                }
                actions = []
                delivered = []

                def run(args, *, env, stdout, check, **_kwargs):
                    action = env["SUB_CMD"]
                    actions.append(action)
                    code = 0
                    if action == "test":
                        path.write_text("{}")
                        (delivered_plan, delivered_source) = build_external_decode_plan(
                            env["PERF_TEST_CONFIG_PATH"],
                            "unused",
                            decode_multi_node_TOPOLOGY,
                        )
                        delivered.append(
                            (delivered_plan, delivered_source, env["PERF_TEST_RUN_ID"])
                        )
                        stdout.write(
                            "PERF_TEST_DECODE_RESULT "
                            + json.dumps(
                                decode_multi_node_marker(
                                    delivered_plan, env["PERF_TEST_RUN_ID"], state
                                )
                            )
                            + "\n"
                        )
                        code = 137
                    if action == "copy":
                        decode_multi_node_persist_run(
                            delivered[0][0],
                            delivered[0][1],
                            Path(env["TASK_OUTPUT_DIR"]) / "run",
                            state,
                            delivered[0][2],
                        )
                    if check and code:
                        raise subprocess.CalledProcessError(code, args)
                    return subprocess.CompletedProcess(args, code)

                with patch.object(
                    multi_benchmark, "logger", logging.getLogger("decode-launcher-test")
                ):
                    with patch.object(
                        multi_benchmark.subprocess, "run", side_effect=run
                    ):
                        if state == "partial":
                            with self.assertRaises(PartialRunError):
                                multi_benchmark.execute_task(
                                    task, str(root), {}, True, 0
                                )
                        else:
                            with self.assertRaisesRegex(RuntimeError, "rank failure"):
                                multi_benchmark.execute_task(
                                    task, str(root), {}, True, 0
                                )
                self.assertEqual(["test", "kill", "copy", "clean"], actions)
                self.assertEqual(
                    expected.workload_fingerprint, delivered[0][0].workload_fingerprint
                )
                self.assertEqual(
                    [case.case_id for case in expected.cases],
                    [case.case_id for case in delivered[0][0].cases],
                )
                self.assertEqual(
                    state,
                    json.loads((root / "Task_/run/results/result.json").read_text())[
                        "status"
                    ],
                )

    def test_runner_delivers_same_config_path_to_every_rank_and_propagates_failure(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config source.json"
            config.write_text("{}")
            bin_dir = root / "bin"
            bin_dir.mkdir()
            calls = root / "calls.jsonl"
            for tool in ("ssh", "scp"):
                path = bin_dir / tool
                path.write_text(
                    f"#!{sys.executable}\nimport json, os, sys\nwith open(os.environ['FAKE_CALLS'], 'a') as stream:\n stream.write(json.dumps([os.path.basename(sys.argv[0]), sys.argv[1:]]) + '\\n')\nif os.path.basename(sys.argv[0]) == 'scp' and os.environ.get('FAKE_FAIL') == '1': sys.exit(7)\n"
                )
                path.chmod(493)
            env = dict(
                PATH=f"{bin_dir}:{os.environ['PATH']}",
                SUB_CMD="test",
                IP_LISTS="node0,node1",
                RUN_USER="test",
                SSH_PORT="2222",
                OPEN_SOURCE_REF="main",
                TP_SIZE="1",
                DP_SIZE="2",
                EP_SIZE="2",
                START_PORT="12333",
                MODEL_TYPE="registered",
                TOKENIZER_PATH="/m",
                CHECKPOINT_PATH="/m",
                PERF_TEST_CONFIG_PATH=str(config),
                PERF_TEST_RUN_ID="delivery_test",
                FAKE_CALLS=str(calls),
            )
            script = Path(__file__).parent / "multi_node/multi_runner.sh"
            completed = subprocess.run(
                ["bash", str(script)], env=env, capture_output=True, text=True
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
            recorded = [json.loads(line) for line in calls.read_text().splitlines()]
            copies = [
                args
                for (tool, args) in recorded
                if tool == "scp" and str(config) in args
            ]
            self.assertEqual(2, len(copies))
            for args in copies:
                self.assertTrue(
                    args[-1].endswith(":/tmp/rtp_llm_perf_delivery_test/config.json")
                )
            environments = [
                shlex.split(args[-1])
                for (tool, args) in recorded
                if tool == "ssh" and "bash /tmp/multi_local_executor.sh" in args[-1]
            ]
            self.assertEqual(2, len(environments))
            for remote_command in environments:
                self.assertIn(
                    "PERF_TEST_CONFIG_PATH=/tmp/rtp_llm_perf_delivery_test/config.json",
                    remote_command,
                )
                self.assertIn(
                    "GANG_CONFIG_STRING=name:test_part0,ip:node0,port:12333;name:test_part1,ip:node1,port:12333",
                    remote_command,
                )
            completed = subprocess.run(
                ["bash", str(script)],
                env={**env, "FAKE_FAIL": "1"},
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(0, completed.returncode)

    def test_master_executes_grid_and_writes_standard_artifacts_and_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = decode_multi_node_write_decode_config(root)
            source = json.loads(path.read_text())
            source["workload"]["profiling"] = {"enabled": True}
            path.write_text(json.dumps(source))
            (plan, source) = build_external_decode_plan(
                path, str(root / "run"), decode_multi_node_TOPOLOGY
            )
            log = root / "server.log"
            requests = []

            class Response:
                status_code = 200

                def __init__(self, body):
                    self.body = body

                def json(self):
                    return self.body

            def post(url, *, json, **_kwargs):
                requests.append((url, copy.deepcopy(json)))
                if url.endswith("/start_profile"):
                    trace = root / (json["trace_name"] + ".json")
                    trace.write_text('{"traceEvents": []}')
                    with log.open("a") as stream:
                        stream.write(f"profiler trace saved: {trace}\n")
                    return Response({"status": "ok"})
                if url.endswith("/update_scheduler_info"):
                    return Response({"status": "ok"})
                return Response(
                    {
                        "aux_info": {
                            "input_len": int(json["prompt"].split("-")[1]),
                            "output_len": json["generate_config"]["max_new_tokens"],
                            "wait_time": 1.0,
                            "first_token_cost_time": 3.0,
                            "cost_time": 18.0,
                        }
                    }
                )

            def transport(**kwargs):
                return DecodeHttpTransport(**kwargs, post=post)

            with patch(
                "rtp_llm.test.perf_test.multi_node.decode_runtime.DecodeHttpTransport",
                side_effect=transport,
            ):
                result = execute_master_decode(
                    plan,
                    source_config=source,
                    endpoint="http://external:12333",
                    prompt_for_length=lambda length: f"prompt-{length}",
                    server_log_path=str(log),
                    profiler_directory=root / "traces",
                )
            run = root / "run"
            self.assertEqual(RunState.COMPLETED, result.state)
            self.assertEqual(
                "completed", json.loads((run / "manifest.json").read_text())["state"]
            )
            envelope = json.loads((run / "results/result.json").read_text())
            self.assertEqual(4, envelope["summary"]["successful_cases"])
            self.assertEqual(4, len(list((run / "results/cases").glob("*.json"))))
            self.assertEqual(
                4, len(list((run / "profiles").glob("*/profile_manifest.json")))
            )
            self.assertEqual(
                "formal",
                json.loads((run / "reports/decode_report.json").read_text())[
                    "report_kind"
                ],
            )
            self.assertTrue((run / "source_config.json").is_file())
            self.assertTrue((run / "resolved_config.json").is_file())
            self.assertTrue((run / "execution_plan.json").is_file())
            self.assertTrue((run / "results/summary.json").is_file())
            self.assertTrue(
                all((url.startswith("http://external:12333") for (url, _) in requests))
            )
            measured_outputs = {
                payload["generate_config"]["max_new_tokens"]
                for (url, payload) in requests
                if url == "http://external:12333"
            }
            self.assertEqual({16, 64}, measured_outputs)
            self.assertTrue(
                all(
                    (
                        case["metrics"]["topology"] == decode_multi_node_TOPOLOGY
                        for case in envelope["cases"]
                    )
                )
            )

    def test_worker_rank_only_waits_and_does_not_write_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            events = []

            def execute():
                root.mkdir()
                events.append("execute")

            result = run_external_rank(
                world_rank=1,
                wait_for_master=lambda: events.append("wait"),
                execute_master=execute,
            )
            self.assertIsNone(result)
            self.assertEqual(["wait"], events)
            self.assertFalse(root.exists())

    def test_rank_zero_executes_once_and_returns_result(self):
        events = []

        def execute():
            events.append("execute")
            return "complete"

        result = run_external_rank(
            world_rank=0,
            wait_for_master=lambda: events.append("wait"),
            execute_master=execute,
        )
        self.assertEqual("complete", result)
        self.assertEqual(["execute"], events)


if __name__ == "__main__":
    unittest.main()
