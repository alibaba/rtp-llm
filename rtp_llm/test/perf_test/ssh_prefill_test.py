"""Core performance behavior contracts."""

import ast
import contextlib
import dataclasses
import io
import json
import logging
import os
import shlex
import subprocess
import tempfile
import typing
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.test.perf_test.decode.scheduler import requires_distributed_decode
from rtp_llm.test.perf_test.decode.transport import DecodeHttpTransport
from rtp_llm.test.perf_test.multi_node import multi_benchmark, prefill_runtime
from rtp_llm.test.perf_test.multi_node.decode_runtime import run_external_rank
from rtp_llm.test.perf_test.multi_node.decode_sync import validate_decode_layout
from rtp_llm.test.perf_test.test_support import (
    prefill_external_aux,
    prefill_external_CASE_CACHE_LEN,
    prefill_external_CASE_INPUT_LEN,
    prefill_external_FakeModelServer,
    prefill_external_Tokenizer,
    prefill_external_TOPOLOGY,
    prefill_external_write_prefill_config,
)


class SshPrefillTest(unittest.TestCase):

    def test_waiter_reads_explicit_or_process_environment_without_class_attributes(
        self,
    ):
        script = Path(__file__).parent / "multi_node/local_server_runner.py"
        function = next(
            n
            for n in ast.parse(script.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "wait_master_done"
        )
        parsed = Mock(side_effect=RuntimeError("gang parsed"))
        namespace = dict(
            vars(typing),
            os=os,
            PyEnvConfigs=type("NoClassConfig", (), {}),
            members_from_test_env=parsed,
        )
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(script), "exec"),
            namespace,
        )
        for supplied, expected in (
            ({"GANG_CONFIG_STRING": "explicit"}, "explicit"),
            ({}, "process"),
            (None, "process"),
        ):
            with self.subTest(supplied=supplied), patch.dict(
                os.environ, {"GANG_CONFIG_STRING": "process"}
            ):
                with self.assertRaisesRegex(RuntimeError, "gang parsed"):
                    namespace["wait_master_done"](supplied)
                parsed.assert_called_with(expected)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "no gang config"):
                namespace["wait_master_done"]()

    def test_external_prefill_uses_same_deadline_for_http_and_dashsc(self):
        from rtp_llm.test.perf_test.execution.runtime import HttpPrefillClient

        for options, seconds in (({}, 900), ({"timeout_seconds": 1234}, 1234)):
            with self.subTest(options=options):
                post = Mock(
                    return_value=SimpleNamespace(
                        status_code=200, json=lambda: {"aux_info": {}}
                    )
                )

                def http(endpoint, **kwargs):
                    return HttpPrefillClient(endpoint, post=post, **kwargs)

                with patch.object(
                    prefill_runtime, "HttpPrefillClient", side_effect=http
                ), patch(
                    "rtp_llm.test.perf_test.prefill.dashsc_transport.DashScPrefillClient"
                ) as dashsc:
                    session = prefill_runtime.ExternalPrefillSession(
                        "http://localhost:18000",
                        topology={},
                        transport="http_text",
                        **options,
                    )
                    session.client.request(
                        phase="probe",
                        case_id="probe",
                        request_index=0,
                        run_index=0,
                        prompt="hello",
                        input_ids=[1],
                        output_length=1,
                    )
                    self.assertEqual(seconds, post.call_args.kwargs["timeout"])
                    prefill_runtime.ExternalPrefillSession(
                        "http://localhost:18000",
                        topology={},
                        transport="dashsc_input_ids",
                        **options,
                    )
                    self.assertEqual(
                        seconds, dashsc.call_args.kwargs["timeout_seconds"]
                    )

    def test_ssh_main_projects_engine_contract_before_host_startup(self):
        from rtp_llm.test.perf_test.execution.runtime import build_engine_launch_spec
        from rtp_llm.test.perf_test.multi_node.decode_runtime import (
            build_external_decode_plan,
        )
        from rtp_llm.test.perf_test.multi_node.gang_config import (
            build_gang_config_string,
        )
        from rtp_llm.test.perf_test.plan import ExecutionPlan, thaw

        script = Path(__file__).parent / "multi_node/local_server_runner.py"
        tree = ast.parse(script.read_text())
        definitions = [
            node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        ]
        entry = compile(
            ast.Module(body=tree.body[-1].body, type_ignores=[]), str(script), "exec"
        )
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                example = (
                    Path(__file__).parent
                    / "examples"
                    / (
                        "prefill_local.json"
                        if mode == "prefill"
                        else "decode_grid_local.json"
                    )
                )
                source = json.loads(
                    (
                        prefill_external_write_prefill_config(root)
                        if mode == "prefill"
                        else example
                    ).read_text()
                )
                source["model"]["topology"] = {"tp": 1, "dp": 2, "pp": 1, "ep": 2}
                source["engine"]["max_seq_len"] = 8192
                source["engine"]["max_batch_tokens"] = 65536
                source["engine"]["args"] = {
                    "reuse_cache": "true",
                    "seq_size_per_block": 64,
                    "extra_data_path": "space 'quoted' literal",
                }
                if mode == "prefill":
                    source["workload"].update(
                        max_input_len=192,
                        cache_lengths=[64],
                        commit_tail_tokens=64,
                        compute_steps=[64],
                    )
                source["engine"]["env"] = {
                    "SSH_CUSTOM": "custom value",
                    "TORCH_CUDA_PROFILER_DIR": "/tmp/shared",
                    "TEST_UNDECLARED_OUTPUTS_DIR": "/tmp/shared",
                    "MAGA_SERVER_WORK_DIR": "/tmp/shared",
                    "USE_BATCH_DECODE_SCHEDULER": "0",
                    "FAKE_BALANCE_EXPERT": "0",
                    "WORLD_RANK": "99",
                    "LOCAL_WORLD_SIZE": "99",
                    "WORLD_SIZE": "99",
                    "TP_SIZE": "99",
                    "DP_SIZE": "99",
                    "PP_SIZE": "99",
                    "EP_SIZE": "99",
                    "GANG_CONFIG_STRING": "wrong-hosts",
                    "START_PORT": "9999",
                    "CUDA_VISIBLE_DEVICES": "99",
                    "LOCAL_RANK": "99",
                }
                runtime_environment = {
                    "WORLD_RANK": "1",
                    "LOCAL_WORLD_SIZE": "1",
                    "WORLD_SIZE": "2",
                    "TP_SIZE": "1",
                    "DP_SIZE": "2",
                    "PP_SIZE": "1",
                    "EP_SIZE": "2",
                    "START_PORT": "12333",
                    "CUDA_VISIBLE_DEVICES": "3",
                    "LOCAL_RANK": "0",
                }
                config = root / "config.json"
                config.write_text(json.dumps(source))
                parallelism = SimpleNamespace(
                    tp_size=1,
                    dp_size=2,
                    pp_size=1,
                    ep_size=2,
                    world_rank=1,
                    local_world_size=1,
                )
                configs = SimpleNamespace(
                    parallelism_config=parallelism,
                    prefill_cp_config=None,
                    server_config=SimpleNamespace(start_port=12333),
                    model_args=SimpleNamespace(
                        tokenizer_path=source["model"].get(
                            "tokenizer", source["model"]["checkpoint"]
                        ),
                        ckpt_path=source["model"]["checkpoint"],
                        model_type=source["model"]["type"],
                    ),
                    distribute_config=SimpleNamespace(gang_config_string="gang"),
                )
                stages = []
                captured = {}

                def setup(args=None):
                    stages.append(("parse", dict(os.environ)))
                    return configs

                def configure(configs):
                    stages.append(("configure", dict(os.environ)))
                    configs.model_args.ckpt_path = "/downloaded/checkpoint"
                    configs.model_args.tokenizer_path = "/downloaded/tokenizer"

                def manager(**kwargs):
                    captured.update(kwargs)
                    captured["environment"] = dict(os.environ)
                    return SimpleNamespace(
                        start_server=lambda: True, stop_server=lambda: None
                    )

                dispatch = {}
                serve_worker = Mock()
                namespace = dict(
                    vars(typing),
                    os=os,
                    pathlib=__import__("pathlib"),
                    time=__import__("time"),
                    sys=__import__("sys"),
                    json=json,
                    logging=logging,
                    shlex=shlex,
                    dataclass=dataclasses.dataclass,
                    replace=dataclasses.replace,
                    thaw=thaw,
                    PyEnvConfigs=SimpleNamespace,
                    ExecutionPlan=ExecutionPlan,
                    setup_args=setup,
                    setup_and_configure_server=configure,
                    set_parallelism_config=lambda *_a, **_k: None,
                    build_engine_launch_spec=build_engine_launch_spec,
                    build_external_prefill_plan=prefill_runtime.build_external_prefill_plan,
                    build_external_decode_plan=build_external_decode_plan,
                    requires_distributed_decode=requires_distributed_decode,
                    validate_decode_layout=validate_decode_layout,
                    DecodeHttpTransport=DecodeHttpTransport,
                    create_query=lambda **_: {},
                    serve_decode_worker=serve_worker,
                    build_gang_config_string=build_gang_config_string,
                    members_from_test_env=lambda _: [
                        SimpleNamespace(
                            name="test_part0", ip="host0", server_port=12333
                        ),
                        SimpleNamespace(
                            name="test_part1", ip="host1", server_port=12333
                        ),
                    ],
                    MagaServerManager=manager,
                    fetch_remote_file_to_local=lambda p: p,
                    run_external_rank=lambda **kwargs: dispatch.update(kwargs),
                )
                exec(
                    compile(
                        ast.Module(body=definitions, type_ignores=[]),
                        str(script),
                        "exec",
                    ),
                    namespace,
                )
                with patch.dict(
                    os.environ,
                    {
                        "PERF_TEST_CONFIG_PATH": str(config),
                        "PERF_TEST_MODE": mode,
                        "PERF_TEST_RUN_ID": "contract",
                        "WORLD_RANK": "1",
                        "LOCAL_WORLD_SIZE": "1",
                        "USE_BATCH_DECODE_SCHEDULER": "0",
                        "FAKE_BALANCE_EXPERT": "0",
                        "UPLOAD_OSS_PATH": "",
                        **runtime_environment,
                    },
                ), patch.object(os, "setpgrp"):
                    with patch.object(
                        os.path, "abspath", side_effect=lambda p: str(root / p)
                    ):
                        exec(entry, namespace)
                if mode == "decode":
                    self.assertIsNotNone(dispatch["execute_worker"])
                    dispatch["execute_worker"]()
                    serve_worker.assert_called_once()
                    self.assertEqual(1, serve_worker.call_args.kwargs["world_rank"])
                argv = shlex.split(captured.get("smoke_args_str", ""))
                self.assertIn("--reuse_cache=true", argv)
                self.assertIn("--seq_size_per_block=64", argv)
                self.assertIn("--extra_data_path=space 'quoted' literal", argv)
                self.assertIn("--checkpoint_path=/downloaded/checkpoint", argv)
                self.assertIn("--tokenizer_path=/downloaded/tokenizer", argv)
                self.assertIn(f"--model_type={source['model']['type']}", argv)
                self.assertEqual("custom value", captured["env_args"]["SSH_CUSTOM"])
                for stage, env in stages:
                    for key, value in runtime_environment.items():
                        self.assertEqual(value, env[key], (stage, key))
                    self.assertEqual("1", env["USE_BATCH_DECODE_SCHEDULER"], stage)
                    self.assertEqual("1", env["FAKE_BALANCE_EXPERT"], stage)
                for key in ("USE_BATCH_DECODE_SCHEDULER", "FAKE_BALANCE_EXPERT"):
                    self.assertEqual("1", captured["env_args"][key])
                env = {**captured["environment"], **captured["env_args"]}
                for key, value in runtime_environment.items():
                    self.assertEqual(value, env[key], key)
                self.assertEqual("12333", captured["port"])
                self.assertEqual("custom value", env["SSH_CUSTOM"])
                self.assertEqual("1", env["WORLD_RANK"])
                self.assertEqual("1", env["LOCAL_WORLD_SIZE"])
                self.assertEqual(
                    "name:test_part0,ip:host0,port:12333;name:test_part1,ip:host1,port:12333",
                    env["GANG_CONFIG_STRING"],
                )
                self.assertEqual(
                    str(root / "TEST_OUTPUT_contract_1"), env["TORCH_CUDA_PROFILER_DIR"]
                )
                self.assertEqual(
                    str(root / "TEST_OUTPUT_contract_1"),
                    env["TEST_UNDECLARED_OUTPUTS_DIR"],
                )
                self.assertEqual(os.getcwd(), env["MAGA_SERVER_WORK_DIR"])

    def test_rank_group_preserves_gang_runtime_assignments_from_custom_env(self):
        script = Path(__file__).parent / "multi_node/local_server_runner.py"
        definitions = [
            node
            for node in ast.parse(script.read_text()).body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        ]
        runtime_environment = {
            "WORLD_RANK": "2",
            "WORLD_SIZE": "4",
            "LOCAL_WORLD_SIZE": "2",
            "LOCAL_RANK": "0",
            "TP_SIZE": "2",
            "DP_SIZE": "2",
            "PP_SIZE": "1",
            "EP_SIZE": "4",
            "GANG_CONFIG_STRING": "established-hosts",
            "START_PORT": "23456",
            "CUDA_VISIBLE_DEVICES": "4,5",
        }
        stages = []
        captured = {}

        def manager(**kwargs):
            captured.update(kwargs)
            captured["environment"] = dict(os.environ)
            return SimpleNamespace()

        namespace = dict(
            vars(typing),
            os=os,
            pathlib=__import__("pathlib"),
            shlex=shlex,
            logging=logging,
            dataclass=dataclasses.dataclass,
            PyEnvConfigs=SimpleNamespace,
            ExecutionPlan=SimpleNamespace,
            MagaServerManager=manager,
            setup_and_configure_server=lambda _: stages.append(dict(os.environ)),
        )
        exec(
            compile(ast.Module(body=definitions, type_ignores=[]), str(script), "exec"),
            namespace,
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                **runtime_environment,
                "TORCH_CUDA_PROFILER_DIR": "/tmp/stale",
                "TEST_UNDECLARED_OUTPUTS_DIR": "/tmp/stale",
                "MAGA_SERVER_WORK_DIR": "/tmp/stale",
            },
        ), patch.object(os, "setpgrp"):
            context = namespace["LocalRankGroupContext"](
                port=23456,
                world_rank=2,
                log_dir_path=directory,
                profiler_directory=Path(directory),
                plan=SimpleNamespace(
                    resolved_config={
                        "engine": {
                            "max_seq_len": 1024,
                            "max_batch_tokens": 1024,
                            "concurrency_limit": 2,
                        }
                    }
                ),
            )
            configs = SimpleNamespace(
                model_args=SimpleNamespace(
                    ckpt_path="/local/model",
                    tokenizer_path="/local/tokenizer",
                    model_type="test",
                )
            )
            namespace["start_local_rank_group"](
                context,
                configs,
                None,
                engine_argv=["--reuse_cache=true"],
                engine_environment={
                    **{key: "conflict" for key in runtime_environment},
                    "GANG_CUSTOM": "kept",
                    "TORCH_CUDA_PROFILER_DIR": "/tmp/shared",
                    "TEST_UNDECLARED_OUTPUTS_DIR": "/tmp/shared",
                    "MAGA_SERVER_WORK_DIR": "/tmp/shared",
                    "USE_BATCH_DECODE_SCHEDULER": "1",
                    "FAKE_BALANCE_EXPERT": "1",
                },
            )
        for key, value in runtime_environment.items():
            self.assertEqual(value, stages[0][key], key)
            self.assertEqual(
                value, {**captured["environment"], **captured["env_args"]}[key], key
            )
        child_env = {**captured["environment"], **captured["env_args"]}
        self.assertEqual(directory, child_env["TORCH_CUDA_PROFILER_DIR"])
        self.assertEqual(directory, child_env["TEST_UNDECLARED_OUTPUTS_DIR"])
        self.assertEqual(os.getcwd(), child_env["MAGA_SERVER_WORK_DIR"])
        self.assertEqual("23456", captured["port"])
        self.assertEqual("kept", stages[0]["GANG_CUSTOM"])
        self.assertEqual("kept", captured["env_args"]["GANG_CUSTOM"])
        self.assertEqual("1", captured["env_args"]["USE_BATCH_DECODE_SCHEDULER"])

    def test_ssh_lifecycle_preserves_spaced_build_arguments(self):
        script = Path(__file__).parent / "multi_node/multi_runner.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "environment.json"
            executor = root / "capture.py"
            executor.write_text(
                "import json, os\nfrom pathlib import Path\nPath(os.environ['CAPTURE']).write_text(json.dumps(dict(os.environ)))\n"
            )
            shell = 'scp() { :; }\nssh() {\n  local command="${@: -1}"\n  command="${command/bash \\/tmp\\/multi_local_executor.sh/python3 $EXECUTOR}"\n  /bin/bash -c "$command"\n}\nexport -f scp ssh\nsource "$1"\n'
            arguments = " --jobs 64 --verbose_failures --config=cuda13 "
            for action in ("build", "kill", "clean"):
                with self.subTest(action=action):
                    capture.unlink(missing_ok=True)
                    result = subprocess.run(
                        ["bash", "-c", shell, "bash", str(script)],
                        env={
                            "PATH": os.environ["PATH"],
                            "IP_LISTS": "node0",
                            "RUN_USER": "test",
                            "SSH_PORT": "22",
                            "OPEN_SOURCE_REF": "main",
                            "SUB_CMD": action,
                            "BAZEL_BUILD_ARGS": arguments,
                            "CAPTURE": str(capture),
                            "EXECUTOR": str(executor),
                            "QUOTED_VALUE": "space 'quote' ; $literal\nsecond line",
                        },
                        capture_output=True,
                        text=True,
                    )
                    self.assertTrue(capture.exists(), result.stderr)
                    remote = json.loads(capture.read_text())
                    self.assertEqual(arguments, remote["BAZEL_BUILD_ARGS"])
                    self.assertEqual(
                        "space 'quote' ; $literal\nsecond line", remote["QUOTED_VALUE"]
                    )

    def test_prefill_preflight_selects_mode_and_rejects_bad_topology(self):
        with tempfile.TemporaryDirectory() as directory:
            path = prefill_external_write_prefill_config(Path(directory))
            task = {
                "perf_test_config": str(path),
                "tp_size": 1,
                "dp_size": 2,
                "pp_size": 1,
                "ep_size": 2,
            }
            (plan, _source) = multi_benchmark.validate_benchmark(task)
            self.assertEqual("prefill", plan.mode)
            self.assertEqual("local", plan.backend)
            task["dp_size"] = 1
            with self.assertRaisesRegex(Exception, "runtime topology"):
                multi_benchmark.validate_benchmark(task)

    def test_prefill_terminal_marker_binds_run_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = prefill_external_write_prefill_config(root)
            (plan, _source) = multi_benchmark.validate_benchmark(
                {
                    "perf_test_config": str(path),
                    "tp_size": 1,
                    "dp_size": 2,
                    "ep_size": 2,
                }
            )
            marker = {
                "state": "completed",
                "run_id": "this-run",
                "world_rank": 0,
                "config_fingerprint": plan.config_fingerprint,
                "workload_fingerprint": plan.workload_fingerprint,
            }
            log = root / "test.log"
            log.write_text("PERF_TEST_PREFILL_RESULT " + json.dumps(marker) + "\n")
            self.assertEqual(
                "completed",
                multi_benchmark.extract_run_state(
                    str(log), plan=plan, run_id="this-run"
                ),
            )
            self.assertIsNone(
                multi_benchmark.extract_run_state(
                    str(log), plan=plan, run_id="other-run"
                )
            )

    def test_external_prefill_master_completes_against_http_service(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (plan, source) = prefill_runtime.build_external_prefill_plan(
                prefill_external_write_prefill_config(root),
                root / "run",
                prefill_external_TOPOLOGY,
            )
            with prefill_external_FakeModelServer(
                [
                    prefill_external_aux(16, 0, 5.0),
                    prefill_external_aux(16, 8, 5.0),
                    prefill_external_aux(prefill_external_CASE_INPUT_LEN, 0, 5.0),
                    prefill_external_aux(
                        prefill_external_CASE_INPUT_LEN,
                        prefill_external_CASE_CACHE_LEN,
                        12.5,
                    ),
                ]
            ) as server:
                with patch(
                    "rtp_llm.test.perf_test.test_util._load_tokenizer",
                    return_value=prefill_external_Tokenizer(),
                ):
                    result = prefill_runtime.execute_master_prefill(
                        plan,
                        source_config=source,
                        endpoint=server.endpoint,
                        tokenizer_path="/m",
                        server_log_path=None,
                        profiler_directory=root,
                    )
            self.assertEqual("completed", result.state.value)
            self.assertEqual(
                "completed",
                json.loads((root / "run/results/result.json").read_text())["status"],
            )
            self.assertTrue((root / "run/results/summary.json").is_file())

    def test_traffic_prefill_delivers_self_contained_frozen_config(self):
        examples = Path(__file__).parent / "examples"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {
                "perf_test_config": str(
                    examples / "prefill_traffic_distribution_local.json"
                ),
                "tp_size": 1,
                "dp_size": 1,
                "ep_size": 1,
                "ip_lists": "node0",
                "run_user": "test",
                "ssh_port": 22,
                "ft_sub_dir": "perf",
                "sub_cmd": "test",
            }
            captured = {}
            (original, _) = multi_benchmark.validate_benchmark(task)
            from rtp_llm.test.perf_test.config_resolver import load_resolved_config

            local = load_resolved_config(task["perf_test_config"])
            self.assertEqual(local.workload_fingerprint, original.workload_fingerprint)

            def run(args, *, env, **_kwargs):
                (remote, _) = prefill_runtime.build_external_prefill_plan(
                    env["PERF_TEST_CONFIG_PATH"],
                    root / "remote-run",
                    {"tp": 1, "dp": 1, "pp": 1, "ep": 1},
                    **(
                        {
                            "resolved_workload_path": env[
                                "PERF_TEST_RESOLVED_WORKLOAD_PATH"
                            ]
                        }
                        if env.get("PERF_TEST_RESOLVED_WORKLOAD_PATH")
                        else {}
                    ),
                )
                self.assertEqual(
                    original.workload_fingerprint, remote.workload_fingerprint
                )
                self.assertEqual(original.config_fingerprint, remote.config_fingerprint)
                self.assertEqual(original.cases, remote.cases)
                self.assertEqual(
                    original.resolved_config["workload"],
                    remote.resolved_config["workload"],
                )
                source = json.loads(Path(env["PERF_TEST_CONFIG_PATH"]).read_text())
                captured.update(source["workload"])
                self.assertEqual("prefill", env["PERF_TEST_MODE"])
                return subprocess.CompletedProcess(args, 7)

            with patch.object(
                multi_benchmark, "logger", logging.getLogger("ssh-prefill")
            ):
                with patch.object(multi_benchmark.subprocess, "run", side_effect=run):
                    with patch.object(multi_benchmark, "multi_kill_script"):
                        with patch.object(
                            multi_benchmark, "multi_clean_script"
                        ), patch.object(multi_benchmark, "multi_copy_script"):
                            with self.assertRaisesRegex(
                                RuntimeError, "did not finalize"
                            ):
                                multi_benchmark.execute_task(
                                    task, str(root), {}, False, 0
                                )
            self.assertEqual(12, len(captured["request_batches"]))
            self.assertNotIn("traffic_request_batches", captured)

    def test_public_only_remote_install_uses_public_requirements(self):
        script = Path(__file__).parent / "multi_node/multi_local_executor.sh"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deps = root / "deps"
            deps.mkdir()
            (deps / "requirements_lock_torch_gpu_cuda13.txt").write_text("")
            fake_python = root / "python"
            fake_python.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$ARGS_FILE"\n')
            fake_python.chmod(493)
            args_file = root / "args.txt"
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    'source "$1"; cd "$2"; install_requirements',
                    "bash",
                    str(script),
                    str(root),
                ],
                env={
                    **os.environ,
                    "BUILD_FROM_SCRATCH": "2",
                    "PERF_TEST_PYTHON_BIN": str(fake_python),
                    "ARGS_FILE": str(args_file),
                },
                capture_output=True,
                text=True,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn(
                "deps/requirements_lock_torch_gpu_cuda13.txt", args_file.read_text()
            )
            self.assertIn("--no-deps", args_file.read_text().splitlines())

    def test_rank_zero_dispatches_prefill_and_preserves_master_failure(self):
        script = Path(__file__).parent / "multi_node/local_server_runner.py"
        entry = ast.parse(script.read_text()).body[-1]
        policy = next(
            (
                node
                for node in ast.parse(script.read_text()).body
                if isinstance(node, ast.FunctionDef)
                and node.name == "_host_engine_environment"
            )
        )
        compiled = compile(
            ast.Module(body=[policy, *entry.body], type_ignores=[]), str(script), "exec"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = prefill_external_write_prefill_config(root)
            parallelism = SimpleNamespace(
                tp_size=1,
                dp_size=2,
                pp_size=1,
                ep_size=2,
                world_rank=0,
                local_world_size=2,
            )
            py_config = SimpleNamespace(
                parallelism_config=parallelism,
                prefill_cp_config=None,
                server_config=SimpleNamespace(start_port=12333),
                model_args=SimpleNamespace(
                    tokenizer_path="/m",
                    ckpt_path="/m",
                    model_type=json.loads(config.read_text())["model"]["type"],
                ),
                distribute_config=SimpleNamespace(gang_config_string=""),
            )
            events = []
            handle = SimpleNamespace(
                tokenizer_path="/m",
                log_file_path=None,
                start_server=lambda: True,
                stop_server=lambda: events.append("stop"),
            )

            def master(*_args, **_kwargs):
                events.append("prefill")
                raise RuntimeError("master failed")

            namespace = {
                "os": os,
                "Dict": typing.Dict,
                "pathlib": __import__("pathlib"),
                "time": __import__("time"),
                "sys": __import__("sys"),
                "json": json,
                "logging": logging,
                "PyEnvConfigs": SimpleNamespace,
                "setup_args": lambda *_args: py_config,
                "thaw": __import__(
                    "rtp_llm.test.perf_test.plan", fromlist=["thaw"]
                ).thaw,
                "replace": dataclasses.replace,
                "build_engine_launch_spec": __import__(
                    "rtp_llm.test.perf_test.execution.runtime",
                    fromlist=["build_engine_launch_spec"],
                ).build_engine_launch_spec,
                "set_parallelism_config": lambda *_args, **_kwargs: None,
                "build_external_prefill_plan": prefill_runtime.build_external_prefill_plan,
                "LocalRankGroupContext": lambda **kwargs: SimpleNamespace(**kwargs),
                "start_local_rank_group": lambda *_args, **_kwargs: handle,
                "run_external_rank": run_external_rank,
                "execute_master_prefill": master,
                "try_upload_log": lambda *_args: None,
            }
            with patch.dict(
                os.environ,
                {
                    "PERF_TEST_CONFIG_PATH": str(config),
                    "PERF_TEST_MODE": "prefill",
                    "PERF_TEST_RUN_ID": "test-run",
                    "UPLOAD_OSS_PATH": "",
                },
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "master failed"):
                        exec(compiled, namespace)
            self.assertEqual(["prefill", "stop"], events)


if __name__ == "__main__":
    unittest.main()
