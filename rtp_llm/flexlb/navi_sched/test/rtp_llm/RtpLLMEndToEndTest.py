import argparse
import asyncio
import json
import math
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


class RtpLLMEndToEndTest:
    def __init__(self, args):
        self._args = args
        self._test_dir = Path(__file__).resolve().parent
        self._navi_sched_dir = self._test_dir.parent.parent
        self._rtp_source = Path(args.rtp_source).resolve()
        self._flexlb_source = Path(
            args.flexlb_source or args.rtp_source
        ).resolve()
        self._build_dir = Path(args.build_dir).resolve()
        self._runtime_dir = Path(args.runtime_dir).resolve()
        self._navi_master_process = None
        self._flexlb_master_process = None
        self._rtp_processes = {}
        self._log_files = []

    def run(self):
        self._prepare_runtime()
        try:
            self._start_masters()
            for index in range(self._args.agent_count):
                self._start_rtp(index, f"server_{index}")
            online_state = self._wait_for_agents_online(self._args.agent_count)
            if self._args.agent_count > 1:
                result = self._run_multi_node_test(online_state)
                print(json.dumps(result, sort_keys=True))
            else:
                self._run_single_node_test()
        finally:
            self._stop_all_rtp()
            self._stop_masters()
            self._close_logs()
        self._check_ports_closed()

    def _run_single_node_test(self):
        batch_result = self._run_batch_test()
        training_workload = self._run_latency_workload("train", 10)
        training_workload["grouped_batches"] = (
            self._run_grouped_latency_workload(
                "train", self._args.latency_training_grouped_repeats
            )
        )
        first = self._infer(self._cache_prompt(), "rtp_e2e_request_0", 20)
        second = self._infer(self._cache_prompt(), "rtp_e2e_request_1", 20)
        self._check_response(first)
        self._check_response(second)
        reused_tokens = self._reuse_tokens(second)
        if reused_tokens <= 0:
            raise RuntimeError(f"RTP-LLM did not reuse KV cache: {second}")
        route_switch_result = self._run_route_switch_test()
        cache_state = self._wait_for_master_state(
            self._agent_cache_ready,
            "Master did not receive RTP-LLM cache and request state",
        )
        model_revision = cache_state["nodes"][0]["latency_model_revision"]
        expected_training_samples = (
            batch_result["completed_batches"]
            + training_workload["grouped_batches"]
        )
        training_state = self._wait_for_latency_state(
            minimum_samples=expected_training_samples
        )
        training_sample_count = len(training_state["samples"])
        training_parameters = training_state["parameters"]
        self._stop_rtp(0)
        self._wait_for_master_state(
            self._agent_stopped,
            "Master did not observe RTP-LLM Agent shutdown",
        )
        self._wait_for_rtp_ports_closed(0)
        self._start_rtp(0, "server_0_evaluation")
        restart_state = self._wait_for_master_state(
            lambda state: self._agents_online(state, 1, None)
            and state["nodes"][0].get("latency_model_revision", 0)
            >= model_revision
            and state["nodes"][0].get("latency_parameter_count", 0) == 11,
            "RTP-LLM Agent did not restore its latency model",
        )
        restarted = self._infer(
            "After restart, answer that the connection works.",
            "rtp_e2e_restart_0",
            12,
        )
        self._check_response(restarted)
        evaluation_workload = self._run_latency_workload("evaluation", 2)
        evaluation_workload["grouped_batches"] = (
            self._run_grouped_latency_workload("evaluation", 1)
        )
        evaluation_state = self._wait_for_latency_state(
            minimum_samples=(
                training_sample_count
                + evaluation_workload["grouped_batches"]
            )
        )
        if evaluation_state["parameters"] != training_parameters:
            raise RuntimeError("latency model changed during frozen evaluation")
        fitting_report = self._latency_error_report(
            evaluation_state, training_sample_count
        )
        print(json.dumps({
            "batch_result": batch_result,
            "cache_state": cache_state,
            "generated_text": self._response_text(first),
            "restart_state": restart_state,
            "restart_text": self._response_text(restarted),
            "reused_tokens": reused_tokens,
            "route_switch_result": route_switch_result,
            "training_workload": training_workload,
            "evaluation_workload": evaluation_workload,
            "latency_fitting_report": fitting_report,
        }, sort_keys=True))

    def _run_latency_workload(self, phase, rounds):
        phrases = [
            "Measure prefill latency with deterministic tokens. ",
            "Cache reuse changes the amount of prompt computation. ",
            "A varied prompt gives the nonlinear model useful coverage. ",
            "Long context evaluation must include both hits and misses. ",
        ]
        repeat_counts = [4, 24, 72, 160]
        requests = 0
        reused = 0
        for round_index in range(rounds):
            for case_index, (phrase, repeat_count) in enumerate(
                zip(phrases, repeat_counts)
            ):
                response = self._infer(
                    f"{phase} case {case_index}. " + phrase * repeat_count,
                    f"rtp_fit_{phase}_{round_index}_{case_index}",
                    1,
                )
                self._check_response(response)
                requests += 1
                reused += int(self._reuse_tokens(response) > 0)
        return {"requests": requests, "requests_with_cache_reuse": reused}

    def _run_grouped_latency_workload(self, phase, repeats):
        sys.path[:0] = [
            path for path in self._rtp_python_paths() if path not in sys.path
        ]
        from rtp_llm.config.generate_config import GenerateConfig
        from rtp_llm.cpp.model_rpc.model_rpc_client import ModelRpcClient
        from rtp_llm.utils.base_model_datatypes import GenerateInput
        import torch

        phase_id = 1 if phase == "train" else 2

        async def run_request(client, request_id, batch_id, batch_size, length):
            generate_input = GenerateInput(
                request_id=request_id,
                token_ids=torch.tensor(
                    [1 + index % 100 for index in range(length)],
                    dtype=torch.int32,
                ),
                mm_inputs=[],
                generate_config=GenerateConfig(
                    can_use_pd_separation=False,
                    max_new_tokens=1,
                    timeout_ms=30000,
                    top_k=1,
                    using_hf_sampling=False,
                ),
                group_id=batch_id,
                group_size=batch_size,
            )
            async for _ in client.enqueue(generate_input):
                pass

        async def run():
            client = ModelRpcClient(
                [f"127.0.0.1:{self._rpc_port(0)}"],
                {},
                max_rpc_timeout_ms=30000,
            )
            completed = 0
            try:
                for repeat in range(repeats):
                    for case_index, (batch_size, length) in enumerate(
                        ((2, 64), (3, 256), (4, 768))
                    ):
                        batch_id = (
                            phase_id * 100000 + repeat * 100 + case_index + 1
                        )
                        await asyncio.gather(*[
                            run_request(
                                client,
                                phase_id * 10000000 + batch_id * 10 + index,
                                batch_id,
                                batch_size,
                                length,
                            )
                            for index in range(batch_size)
                        ])
                        completed += 1
                return completed
            finally:
                await client.close()

        return asyncio.run(run())

    def _run_route_switch_test(self):
        results = []
        routes = (
            ("navi_sched", ""),
            ("flexlb", f"127.0.0.1:{self._args.flexlb_port}"),
            ("navi_sched", ""),
        )
        for route_name, master_address in routes:
            state = self._switch_master_address(master_address)
            response = self._infer(
                f"Runtime route is {route_name}.",
                f"rtp_route_{route_name}_{state['generation']}",
                8,
            )
            self._check_response(response)
            results.append({
                "route": route_name,
                "master_address": master_address,
                "generation": state["generation"],
                "text": self._response_text(response),
            })
        return results

    def _switch_master_address(self, master_address):
        result = subprocess.run(
            [
                "curl", "--fail", "--silent", "--show-error",
                "--request", "POST",
                "--header", "Content-Type: application/json",
                "--data", json.dumps({
                    "update": True,
                    "masterAddress": master_address,
                }, separators=(",", ":")),
                f"http://127.0.0.1:{self._control_port()}/route-mode",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        state = json.loads(result.stdout)
        actual_address = state.get("masterAddress", "")
        if not state.get("accepted") or actual_address != master_address:
            raise RuntimeError(
                f"master address update did not take effect: {state}"
            )
        time.sleep(self._args.route_switch_settle_seconds)
        return state

    def _run_batch_test(self):
        sys.path[:0] = [
            path for path in self._rtp_python_paths()
            if path not in sys.path
        ]
        from rtp_llm.config.generate_config import GenerateConfig
        from rtp_llm.cpp.model_rpc.model_rpc_client import ModelRpcClient
        from rtp_llm.utils.base_model_datatypes import GenerateInput
        import torch

        async def run_request(client, request_id, batch_id, batch_size):
            generate_input = GenerateInput(
                request_id=request_id,
                token_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
                mm_inputs=[],
                generate_config=GenerateConfig(
                    can_use_pd_separation=False,
                    max_new_tokens=2,
                    timeout_ms=30000,
                    top_k=1,
                    using_hf_sampling=False,
                ),
                group_id=batch_id,
                group_size=batch_size,
            )
            output_count = 0
            async for outputs in client.enqueue(generate_input):
                output_count += len(outputs.generate_outputs)
            if output_count == 0:
                raise RuntimeError(
                    f"grouped RTP request {request_id} returned no output"
                )
            return output_count

        async def run():
            client = ModelRpcClient(
                [f"127.0.0.1:{self._rpc_port(0)}"],
                {},
                max_rpc_timeout_ms=30000,
            )
            try:
                grouped = await asyncio.gather(
                    run_request(client, 910001, 920001, 2),
                    run_request(client, 910002, 920001, 2),
                )
                timeout_start = time.monotonic()
                partial = await run_request(client, 910003, 920002, 2)
                timeout_ms = (time.monotonic() - timeout_start) * 1000
                late_start = time.monotonic()
                late = await run_request(client, 910004, 920002, 2)
                late_ms = (time.monotonic() - late_start) * 1000
                return {
                    "completed_batches": 3,
                    "grouped_output_counts": grouped,
                    "late_output_count": late,
                    "late_request_ms": round(late_ms, 3),
                    "partial_output_count": partial,
                    "partial_request_ms": round(timeout_ms, 3),
                }
            finally:
                await client.close()

        return asyncio.run(run())

    def _run_multi_node_test(self, online_state):
        responses = []
        for index in range(max(4, self._args.agent_count * 2)):
            response = self._infer(
                f"Multi-node routing request {index}.",
                f"rtp_multi_request_{index}",
                8,
            )
            self._check_response(response)
            responses.append(response)
        self._stop_rtp(0)
        remaining_state = self._wait_for_agents_online(
            self._args.agent_count - 1,
            excluded_rpc_port=self._rpc_port(0),
        )
        self._wait_for_rtp_ports_closed(0)
        failover = self._infer(
            "Answer after one inference node stopped.",
            "rtp_multi_failover",
            8,
            frontend_index=1,
        )
        self._check_response(failover, excluded_rpc_port=self._rpc_port(0))
        self._start_rtp(0, "server_0_restart")
        restarted_state = self._wait_for_agents_online(self._args.agent_count)
        return {
            "agent_count": self._args.agent_count,
            "initial_state": online_state,
            "remaining_state": remaining_state,
            "restarted_state": restarted_state,
            "route_count": len(responses),
            "failover_text": self._response_text(failover),
        }

    def _prepare_runtime(self):
        if self._runtime_dir.exists():
            shutil.rmtree(self._runtime_dir)
        self._runtime_dir.mkdir(parents=True, exist_ok=True)
        names = ["navi_master", "flexlb_master", "server_0_evaluation"]
        names.extend(f"server_{index}" for index in range(self._args.agent_count))
        for name in names:
            path = self._runtime_dir / name
            path.mkdir(parents=True, exist_ok=True)

    def _start_masters(self):
        self._start_navi_master()
        self._start_flexlb_master()

    def _start_navi_master(self):
        master_dir = self._runtime_dir / "navi_master"
        config_path = master_dir / "master.json"
        config_path.write_text(json.dumps({
            "scheduler_name": "random",
            "thread_num": self._args.thread_num,
            "local_nodes": [
                {
                    "biz_name": self._args.biz_name,
                    "ip": "127.0.0.1",
                    "grpc_port": self._agent_port(index),
                    "support_heartbeat": True,
                }
                for index in range(self._args.agent_count)
            ],
        }), encoding="utf-8")
        env = os.environ.copy()
        python_paths = self._rtp_python_paths()
        if env.get("PYTHONPATH"):
            python_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        env["LD_LIBRARY_PATH"] = self._library_path(env)
        env.update({
            "NAVI_SCHED_MASTER_BIZ_NAME": self._args.master_biz_name,
            "NAVI_SCHED_MASTER_CONFIG_FILE": str(config_path),
            "NAVI_SCHED_MASTER_ENABLE_KMON": "0",
            "NAVI_SCHED_MASTER_GRPC_PORT": str(self._args.master_port),
            "NAVI_SCHED_MASTER_LOG_FILE": str(master_dir / "navi_sched.log"),
            "NAVI_SCHED_MASTER_NAME": "rtp_llm_e2e_master",
            "NAVI_SCHED_MASTER_SNAPSHOT_DIR": "navi_sched",
        })
        self._navi_master_process = self._start_process(
            [sys.executable, "-m", "navi_sched.master_process"],
            master_dir,
            self._runtime_dir / "navi_master.log",
            env,
        )
        self._wait_for_port(
            self._navi_master_process,
            self._args.master_port,
            "NaviSched Master",
        )
        self._wait_for_port(
            self._navi_master_process,
            self._control_port(),
            "NaviSched route control",
        )

    def _start_flexlb_master(self):
        flexlb_dir = self._flexlb_source / "rtp_llm" / "flexlb"
        jar = flexlb_dir / "flexlb-api" / "target" / (
            "flexlb-api-1.0.0-SNAPSHOT.jar"
        )
        if not jar.is_file():
            raise RuntimeError(f"FlexLB JAR does not exist: {jar}")
        domain = "rtp-navi-e2e-pdfusion"
        worker_addresses = ",".join(
            f"127.0.0.1:{self._frontend_port(index)}"
            for index in range(self._args.agent_count)
        )
        service_config = {
            "service_id": (
                "aigc.text-generation.generation.navi_sched_e2e"
            ),
            "load_balance": True,
            "role_endpoints": [{
                "group": "local",
                "pd_fusion_endpoint": {
                    "address": domain,
                    "protocol": "http",
                    "path": "/",
                },
            }],
        }
        env = os.environ.copy()
        java_home = Path(self._args.java_home).resolve()
        env.update({
            f"DOMAIN_ADDRESS:{domain}": worker_addresses,
            "HIPPO_ROLE": "navi_sched_real_e2e",
            "MODEL_SERVICE_CONFIG": json.dumps(
                service_config, separators=(",", ":")
            ),
            "OTEL_EXPORTER_OTLP_ENDPOINT": "none",
            "OTEL_TRACE_SKIP_PATTERN": ".*",
            "PATH": f"{java_home / 'bin'}{os.pathsep}{env.get('PATH', '')}",
        })
        command = [
            str(java_home / "bin" / "java"),
            "-jar", str(jar),
            f"--server.port={self._args.flexlb_port}",
            f"--management.server.port={self._args.flexlb_management_port}",
            "--spring.profiles.active=test",
        ]
        self._flexlb_master_process = self._start_process(
            command,
            self._runtime_dir / "flexlb_master",
            self._runtime_dir / "flexlb_master.log",
            env,
        )
        self._wait_for_port(
            self._flexlb_master_process,
            self._args.flexlb_port,
            "FlexLB Master",
        )

    def _start_rtp(self, index, runtime_name):
        env = self._rtp_env(index, runtime_name)
        command = [sys.executable, "-m", "rtp_llm.start_server"]
        process = self._start_process(
            command,
            self._runtime_dir / runtime_name,
            self._runtime_dir / f"{runtime_name}.log",
            env,
        )
        self._rtp_processes[index] = process
        self._wait_for_port(
            process,
            self._frontend_port(index),
            "RTP-LLM frontend",
        )
        self._wait_for_port(
            process,
            self._rpc_port(index),
            "RTP-LLM model RPC server",
        )
        self._wait_for_port(
            process,
            self._agent_port(index),
            "NaviSched Agent",
        )

    def _rtp_env(self, index, runtime_name):
        env = os.environ.copy()
        runtime_dir = self._runtime_dir / runtime_name
        python_paths = self._rtp_python_paths()
        if env.get("PYTHONPATH"):
            python_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        env["LD_LIBRARY_PATH"] = self._library_path(env)
        env.update({
            "ACT_TYPE": self._args.act_type,
            "CHECKPOINT_PATH": str(Path(self._args.model).resolve()),
            "CUDA_VISIBLE_DEVICES": self._visible_gpus(index),
            "DEVICE_RESERVE_MEMORY_BYTES": str(
                self._args.device_reserve_memory_bytes
            ),
            "DISABLE_FLASH_INFER": "1" if self._args.disable_flash_infer else "0",
            "DP_SIZE": "1",
            "ENABLE_COMM_OVERLAP": "0",
            "ENABLE_FMHA": "on",
            "EP_SIZE": "1",
            "FRONTEND_SERVER_COUNT": "1",
            "FT_DISABLE_CUSTOM_AR": "1",
            "KERNEL_SEQ_SIZE_PER_BLOCK": str(
                self._args.kernel_seq_size_per_block
            ),
            "LOG_LEVEL": "INFO",
            "LOG_PATH": str(runtime_dir),
            "LOAD_METHOD": "scratch",
            "LOAD_PYTHON_MODEL": "1" if self._args.load_python_model else "0",
            "MAX_SEQ_LEN": str(self._args.max_seq_len),
            "MODEL_SERVICE_CONFIG": json.dumps({
                "service_id": (
                    "aigc.text-generation.generation.navi_sched_e2e"
                ),
                "master_endpoint": {
                    "type": "VipServer",
                    "address": f"127.0.0.1:{self._args.flexlb_port}",
                    "protocol": "http",
                    "path": "/",
                },
                "use_local": True,
            }, separators=(",", ":")),
            "MODEL_TYPE": self._args.model_type,
            "NCCL_DISABLE_ABORT": "1",
            "RESERVER_RUNTIME_MEM_MB": str(self._args.runtime_memory_mb),
            "REUSE_CACHE": "1",
            "QUANTIZATION": self._args.quantization,
            "RTP_LLM_MASTER_TYPE": "navi_sched",
            "RTP_LLM_NAVI_SCHED_AGENT_GIG_PORT": str(self._agent_port(index)),
            "RTP_LLM_NAVI_SCHED_BIZ_NAME": self._args.biz_name,
            "RTP_LLM_NAVI_SCHED_LATENCY_FIT_BATCH_SIZE": (
                "100000" if "evaluation" in runtime_name else "4"
            ),
            "RTP_LLM_NAVI_SCHED_LATENCY_HISTORY_LIMIT": "2000",
            "RTP_LLM_NAVI_SCHED_LATENCY_MODEL_STATE_FILE": str(
                self._runtime_dir / f"agent_{index}_latency_model.state"
            ),
            "RTP_LLM_NAVI_SCHED_LATENCY_PERSIST_INTERVAL": "1",
            "RTP_LLM_NAVI_SCHED_MASTER_BIZ_NAME": self._args.master_biz_name,
            "RTP_LLM_NAVI_SCHED_MASTER_ADDRESS": (
                f"127.0.0.1:{self._args.master_port}"
            ),
            "RTP_LLM_NAVI_SCHED_TARGET_ROLE": "PREFILL",
            "SHUTDOWN_TIMEOUT": "10",
            "SEQ_SIZE_PER_BLOCK": str(self._args.seq_size_per_block),
            "START_PORT": str(self._frontend_port(index)),
            "TOKENIZER_PATH": str(Path(self._args.model).resolve()),
            "TP_SIZE": str(self._args.tp_size),
            "USE_RPC_MODEL": "1",
            "WARM_UP": "0",
            "LOCAL_WORLD_SIZE": str(self._args.tp_size),
            "WORLD_SIZE": str(self._args.tp_size),
        })
        return env

    def _rtp_python_paths(self):
        paths = []
        if self._args.navi_python_dir:
            paths.append(str(Path(self._args.navi_python_dir).resolve()))
        paths.append(str(self._rtp_source))
        paths.append(str(self._build_dir))
        return paths

    def _visible_gpus(self, index):
        if self._args.gpus:
            gpus = [value.strip() for value in self._args.gpus.split(",")]
        else:
            gpus = [str(self._args.gpu + offset) for offset in range(
                self._args.agent_count * self._args.tp_size
            )]
        begin = index * self._args.tp_size
        end = begin + self._args.tp_size
        if len(gpus) < end:
            raise ValueError(
                f"agent {index} requires {self._args.tp_size} GPUs, "
                f"but --gpus only provides {len(gpus)}"
            )
        return ",".join(gpus[begin:end])

    def _frontend_port(self, index):
        return self._args.frontend_port + index * self._args.port_stride

    def _rpc_port(self, index):
        return self._args.rpc_port + index * self._args.port_stride

    def _agent_port(self, index):
        return self._args.agent_port + index * self._args.port_stride

    def _control_port(self):
        return self._args.master_port + 1

    def _library_path(self, env):
        cuda_home = Path(self._args.cuda_home).resolve()
        paths = [
            "/usr/local/nvidia/lib64",
            str(self._build_dir),
            str(self._build_dir / "rtp_llm" / "libs"),
            str(self._rtp_source / ".navi_sched" / "cpp" / "lib"),
            str(Path(sys.prefix) / "lib"),
            self._torch_library_path(),
            str(cuda_home / "targets" / "x86_64-linux" / "lib"),
            str(cuda_home / "extras" / "CUPTI" / "lib64"),
            "/usr/local/PPU_SDK/targets/x86_64-linux/lib",
        ]
        if env.get("LD_LIBRARY_PATH"):
            paths.append(env["LD_LIBRARY_PATH"])
        return os.pathsep.join(paths)

    @staticmethod
    def _torch_library_path():
        import torch

        return str(Path(torch.__file__).resolve().parent / "lib")

    def _start_process(self, command, cwd, log_path, env=None):
        log_file = open(log_path, "w", encoding="utf-8")
        self._log_files.append(log_file)
        return subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def _cache_prompt(self):
        # Keep enough prompt tokens to populate at least one complete cache block.
        repeat_count = max(80, math.ceil(self._args.seq_size_per_block / 8) + 1)
        return (
            "KV cache scheduling must use exact framework block keys. "
            * repeat_count
        )

    def _infer(self, prompt, request_id, max_tokens, frontend_index=0):
        if self._args.prefill_only:
            max_tokens = 1
        body = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "model": "navi_sched_e2e",
            "request_id": request_id,
            "stream": False,
            "temperature": 0.1,
            "top_k": 1,
            "max_tokens": max_tokens,
        }).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{self._frontend_port(frontend_index)}"
            "/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Request-Id": request_id,
            },
            method="POST",
        )
        with urllib.request.urlopen(
            request, timeout=self._args.timeout_seconds
        ) as response:
            if response.status != 200:
                raise RuntimeError(f"RTP-LLM returned HTTP {response.status}")
            return json.load(response)

    def _check_response(self, response, excluded_rpc_port=None):
        if not self._response_text(response):
            raise RuntimeError(f"RTP-LLM returned no generated text: {response}")
        role_addresses = response.get("aux_info", {}).get("role_addrs", [])
        selected_ports = {
            address.get("grpc_port") for address in role_addresses
        }
        valid_ports = {
            self._rpc_port(index) for index in range(self._args.agent_count)
        }
        if not selected_ports.intersection(valid_ports):
            raise RuntimeError(
                f"RTP-LLM did not use the NaviSched-selected endpoint: {response}"
            )
        if excluded_rpc_port in selected_ports:
            raise RuntimeError(
                f"RTP-LLM selected offline endpoint {excluded_rpc_port}: "
                f"{response}"
            )

    @staticmethod
    def _response_text(response):
        choices = response.get("choices") or []
        if not choices:
            return ""
        return str(choices[0].get("message", {}).get("content", ""))

    @staticmethod
    def _reuse_tokens(response):
        aux_info = response.get("aux_info", {})
        usage = response.get("usage", {})
        details = usage.get("prompt_tokens_details") or {}
        return max(
            int(aux_info.get("reuse_len", 0) or 0),
            int(details.get("cached_tokens", 0) or 0),
        )

    def _wait_for_latency_state(self, minimum_samples):
        state_path = self._runtime_dir / "agent_0_latency_model.state"
        deadline = time.monotonic() + self._args.timeout_seconds
        last_state = None
        while time.monotonic() < deadline:
            try:
                last_state = self._read_latency_state(state_path)
            except (OSError, ValueError, struct.error):
                last_state = None
            if (
                last_state is not None
                and len(last_state["samples"]) >= minimum_samples
            ):
                return last_state
            time.sleep(0.1)
        count = len(last_state["samples"]) if last_state else 0
        raise RuntimeError(
            f"latency state has {count} samples, expected {minimum_samples}"
        )

    @staticmethod
    def _read_latency_state(path):
        content = path.read_bytes()
        offset = 0

        def read_varuint():
            nonlocal offset
            value = 0
            shift = 0
            while offset < len(content) and shift <= 63:
                current = content[offset]
                offset += 1
                value |= (current & 0x7F) << shift
                if current < 0x80:
                    return value
                shift += 7
            raise ValueError("invalid latency state varint")

        def read_int64():
            value = read_varuint()
            return (value >> 1) ^ -(value & 1)

        def read_fixed(format_string):
            nonlocal offset
            value = struct.unpack_from("<" + format_string, content, offset)[0]
            offset += struct.calcsize(format_string)
            return value

        magic_size = read_varuint()
        magic = content[offset:offset + magic_size].decode()
        offset += magic_size
        version = read_fixed("i")
        formula = read_fixed("i")
        binary_version = read_int64()
        revision = read_int64()
        parameter_count = read_varuint()
        parameters = [read_fixed("d") for _ in range(parameter_count)]
        adam_moment_1_count = read_varuint()
        adam_moment_1 = [
            read_fixed("d") for _ in range(adam_moment_1_count)
        ]
        adam_moment_2_count = read_varuint()
        adam_moment_2 = [
            read_fixed("d") for _ in range(adam_moment_2_count)
        ]
        optimizer_step = read_int64()
        sample_count = read_varuint()
        samples = []
        for _ in range(sample_count):
            sample = {
                "batch_count": read_int64(),
                "batch_size": read_int64(),
                "cache_hit_tokens": read_int64(),
                "cache_miss_tokens": read_int64(),
                "cache_miss_token_square_sum": read_fixed("d"),
                "cache_hit_miss_token_product_sum": read_fixed("d"),
                "capacity_split": read_fixed("?"),
                "waiting_batch_size": read_int64(),
                "waiting_cache_hit_tokens": read_int64(),
                "waiting_cache_miss_tokens": read_int64(),
                "waiting_cache_miss_token_square_sum": read_fixed("d"),
                "waiting_cache_hit_miss_token_product_sum": read_fixed("d"),
                "preceding_request_count": read_int64(),
                "latency_ms": read_fixed("d"),
                "event_time_us": read_int64(),
            }
            if sample["capacity_split"]:
                raise ValueError("RTP-LLM latency sample unexpectedly uses capacity split")
            samples.append(sample)
        if (
            magic != "navi_sched_latency_state"
            or version != 2
            or formula != 1
            or binary_version != 8
            or len(parameters) != 11
            or len(adam_moment_1) != len(parameters)
            or len(adam_moment_2) != len(parameters)
            or optimizer_step <= 0
            or offset != len(content)
        ):
            raise ValueError("invalid latency model state")
        return {
            "formula": formula,
            "binary_version": binary_version,
            "revision": revision,
            "parameters": parameters,
            "adam_moment_1": adam_moment_1,
            "adam_moment_2": adam_moment_2,
            "optimizer_step": optimizer_step,
            "samples": samples,
        }

    @staticmethod
    def _latency_prediction(sample, parameters):
        scale = 1024.0
        inputs = (
            1.0,
            float(sample["batch_size"]),
            sample["cache_hit_tokens"] / scale,
            sample["cache_miss_tokens"] / scale,
            sample["cache_miss_token_square_sum"] / (scale * scale),
            sample["cache_hit_miss_token_product_sum"] / (scale * scale),
            float(sample["batch_count"]),
        )
        linear = sum(
            value * parameters[index] for index, value in enumerate(inputs)
        ) / 320.0
        scaled_linear = linear * (parameters[9] + 1.0)
        nonlinear = scaled_linear + math.sqrt(scaled_linear * scaled_linear + 4.0)
        total_prompt = (
            sample["cache_hit_tokens"] + sample["cache_miss_tokens"]
        ) / scale
        return (parameters[7] / 0.005 + parameters[8] / 0.02 * nonlinear +
                parameters[10] * total_prompt)

    @classmethod
    def _latency_error_report(cls, state, sample_start):
        samples = state["samples"][sample_start:]
        if not samples:
            raise RuntimeError("frozen evaluation produced no latency samples")
        details = []
        for sample in samples:
            prediction = cls._latency_prediction(sample, state["parameters"])
            actual = sample["latency_ms"]
            details.append({
                "batch_size": sample["batch_size"],
                "cache_hit_tokens": sample["cache_hit_tokens"],
                "cache_miss_tokens": sample["cache_miss_tokens"],
                "actual_ms": round(actual, 6),
                "predicted_ms": round(prediction, 6),
                "error_ms": round(prediction - actual, 6),
            })
        errors = [item["error_ms"] for item in details]
        absolute_errors = [abs(value) for value in errors]
        positive = [
            abs(item["error_ms"]) / item["actual_ms"] * 100.0
            for item in details
            if item["actual_ms"] > 0.0
        ]
        return {
            "evaluation_sample_count": len(details),
            "training_sample_count": sample_start,
            "model_revision": state["revision"],
            "parameters": state["parameters"],
            "mae_ms": sum(absolute_errors) / len(absolute_errors),
            "rmse_ms": math.sqrt(
                sum(value * value for value in errors) / len(errors)
            ),
            "max_absolute_error_ms": max(absolute_errors),
            "mape_percent": sum(positive) / len(positive) if positive else None,
            "samples": details,
        }

    def _master_snapshot(self):
        snapshot_dir = (
            self._runtime_dir
            / "navi_master"
            / "navi_sched"
            / "rtp_llm_e2e_master"
        )
        snapshots = list(snapshot_dir.glob("*_snapshot"))
        if not snapshots:
            return None
        try:
            latest = max(snapshots, key=lambda path: path.stat().st_mtime_ns)
            content = latest.read_text(encoding="utf-8")
        except OSError:
            # The snapshot writer can briefly expose a truncated file between updates.
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return self._parse_master_snapshot(content)

    @staticmethod
    def _parse_master_snapshot(content):
        if "Role\n" not in content or "\nAgents\n" not in content:
            return None
        state = {"nodes": []}
        finished = re.search(r"^  finished_requests: (\d+)$", content, re.MULTILINE)
        state["agent_request_finished_count"] = (
            int(finished.group(1)) if finished else 0
        )
        integer_fields = {
            "schedule_round_count": "schedule_round_count",
            "schedule_round_request_count": "schedule_round_request_count",
            "schedule_round_requests_min": "schedule_round_requests_min",
            "schedule_round_requests_max": "schedule_round_requests_max",
        }
        for output_key, snapshot_key in integer_fields.items():
            match = re.search(
                rf"^  {snapshot_key}: (\d+)$", content, re.MULTILINE
            )
            state[output_key] = int(match.group(1)) if match else 0
        mean = re.search(
            r"^  schedule_round_requests_mean: ([0-9.eE+-]+)$",
            content,
            re.MULTILINE,
        )
        state["schedule_round_requests_mean"] = (
            float(mean.group(1)) if mean else 0.0
        )
        histogram = re.search(
            r"^  schedule_round_request_count_histogram:(.*)$",
            content,
            re.MULTILINE,
        )
        state["schedule_round_request_count_histogram"] = {
            request_count: int(round_count)
            for request_count, round_count in re.findall(
                r"(\d+):(\d+)", histogram.group(1) if histogram else ""
            )
        }
        for block in content.split("\n  Agent\n")[1:]:
            def value(pattern, default=None):
                match = re.search(pattern, block, re.MULTILINE)
                return match.group(1) if match else default

            address = value(r"^    server_address: (.+)$", "")
            cache = re.search(
                r"^    cache_blocks: total=(\d+), cached=(\d+)$",
                block,
                re.MULTILINE,
            )
            requests = re.search(
                r"^    requests: inflight=(\d+), waiting_for_batch=(\d+), "
                r"queued_for_execution=(\d+), running=(\d+)$",
                block,
                re.MULTILINE,
            )
            latency = re.search(
                r"^    latency_model: formula=([^,]+), binary_version=(-?\d+), "
                r"revision=(\d+), parameter_count=(\d+)$",
                block,
                re.MULTILINE,
            )
            node = {
                "biz_name": value(r"^    biz_name: (.+)$", ""),
                "online": value(r"^    online: (true|false)$", "false") == "true",
                "server_address": address,
                "server_port": int(address.rsplit(":", 1)[1]) if ":" in address else 0,
                "cached_block_count": int(cache.group(2)) if cache else 0,
                "inflight_requests": int(requests.group(1)) if requests else 0,
                "latency_model_revision": int(latency.group(3)) if latency else 0,
                "latency_parameter_count": int(latency.group(4)) if latency else 0,
            }
            state["nodes"].append(node)
        return state

    def _wait_for_master_state(self, predicate, error_message):
        deadline = time.monotonic() + self._args.timeout_seconds
        last_state = None
        while time.monotonic() < deadline:
            self._check_process(self._navi_master_process, "NaviSched Master")
            last_state = self._master_snapshot()
            if last_state is not None and predicate(last_state):
                return last_state
            time.sleep(0.2)
        raise RuntimeError(f"{error_message}: {last_state}")

    def _wait_for_agents_online(self, count, excluded_rpc_port=None):
        return self._wait_for_master_state(
            lambda state: self._agents_online(
                state, count, excluded_rpc_port
            ),
            f"Master did not connect to {count} RTP-LLM Agents",
        )

    def _agents_online(self, state, count, excluded_rpc_port):
        nodes = state.get("nodes", [])
        expected_ports = {
            self._rpc_port(index) for index in range(self._args.agent_count)
        }
        if excluded_rpc_port is not None:
            expected_ports.discard(excluded_rpc_port)
        actual_ports = {
            node.get("server_port")
            for node in nodes
            if node.get("online") and node.get("biz_name") == self._args.biz_name
        }
        return len(nodes) == count and actual_ports == expected_ports

    @staticmethod
    def _agent_cache_ready(state):
        nodes = state.get("nodes", [])
        return len(nodes) == 1 and all([
            nodes[0].get("cached_block_count", 0) > 0,
            nodes[0].get("inflight_requests") == 0,
            state.get("agent_request_finished_count", 0) > 0,
            nodes[0].get("latency_model_revision", 0) >= 2,
            nodes[0].get("latency_parameter_count", 0) == 11,
        ])

    @staticmethod
    def _agent_stopped(state):
        return not state.get("nodes")

    def _wait_for_port(self, process, port, process_name):
        deadline = time.monotonic() + self._args.timeout_seconds
        while time.monotonic() < deadline:
            self._check_process(process, process_name)
            if self._port_is_open(port):
                return
            time.sleep(0.2)
        raise RuntimeError(f"{process_name} did not listen on port {port}")

    @staticmethod
    def _check_process(process, name):
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"{name} exited with code {return_code}")

    def _stop_rtp(self, index):
        process = self._rtp_processes.pop(index, None)
        if process is None or process.poll() is not None:
            return
        process.send_signal(signal.SIGTERM)
        self._wait_for_agent_stop_or_process(process, self._rpc_port(index))
        self._stop_process(process, "RTP-LLM", signal_sent=True)

    def _wait_for_agent_stop_or_process(self, process, rpc_port):
        deadline = time.monotonic() + min(30, self._args.timeout_seconds)
        while time.monotonic() < deadline and process.poll() is None:
            state = self._master_snapshot()
            if state is not None and all(
                node.get("server_port") != rpc_port
                for node in state.get("nodes", [])
            ):
                return
            time.sleep(0.2)

    def _stop_all_rtp(self):
        for index in list(self._rtp_processes):
            self._stop_rtp(index)

    def _stop_masters(self):
        self._stop_process(self._flexlb_master_process, "FlexLB Master")
        self._flexlb_master_process = None
        self._stop_process(self._navi_master_process, "NaviSched Master")
        self._navi_master_process = None

    @staticmethod
    def _stop_process(process, name, signal_sent=False):
        if process is None or process.poll() is not None:
            return
        if not signal_sent:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
            print(f"{name} required SIGKILL after graceful shutdown timeout")

    def _wait_for_rtp_ports_closed(self, index):
        ports = [
            self._frontend_port(index),
            self._rpc_port(index),
            self._agent_port(index),
        ]
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if not any(self._port_is_open(port) for port in ports):
                return
            time.sleep(0.2)
        raise RuntimeError(f"RTP-LLM ports remain open after shutdown: {ports}")

    def _check_ports_closed(self):
        ports = [
            self._args.master_port,
            self._control_port(),
            self._args.flexlb_port,
            self._args.flexlb_management_port,
            self._args.flexlb_port + 2,
        ]
        for index in range(self._args.agent_count):
            ports.extend([
                self._frontend_port(index),
                self._rpc_port(index),
                self._agent_port(index),
            ])
        for port in ports:
            if self._port_is_open(port):
                raise RuntimeError(f"port {port} remains open after shutdown")

    @staticmethod
    def _port_is_open(port):
        with socket.socket() as sock:
            sock.settimeout(0.2)
            return sock.connect_ex(("127.0.0.1", port)) == 0

    def _close_logs(self):
        for log_file in self._log_files:
            log_file.close()
        self._log_files.clear()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rtp-source",
        default=str(Path.home() / "RTP-LLM" / "github-opensource"),
    )
    parser.add_argument("--flexlb-source", default="")
    parser.add_argument("--build-dir")
    parser.add_argument("--navi-python-dir", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--agent-count", type=int, default=1)
    parser.add_argument("--port-stride", type=int, default=1000)
    parser.add_argument(
        "--runtime-dir",
        default=str(
            Path(__file__).resolve().parent.parent.parent
            / "tmp"
            / "rtp_llm_e2e"
        ),
    )
    parser.add_argument("--cuda-home", default="/usr/local/cuda-12.9")
    parser.add_argument("--master-port", type=int, default=19010)
    parser.add_argument("--flexlb-port", type=int, default=17001)
    parser.add_argument("--flexlb-management-port", type=int, default=17002)
    parser.add_argument("--agent-port", type=int, default=19100)
    parser.add_argument("--frontend-port", type=int, default=38530)
    parser.add_argument("--rpc-port", type=int, default=38531)
    parser.add_argument("--master-biz-name", default="master")
    parser.add_argument("--biz-name", default="model_backend")
    parser.add_argument("--model-type", default="qwen_2")
    parser.add_argument("--act-type", default="BF16")
    parser.add_argument("--quantization", default="")
    parser.add_argument("--load-python-model", action="store_true")
    parser.add_argument("--thread-num", type=int, default=64)
    parser.add_argument(
        "--java-home",
        default="/opt/taobao/install/ajdk21_21.0.6.0.6",
    )
    parser.add_argument("--route-switch-settle-seconds", type=float, default=1.0)
    parser.add_argument(
        "--latency-training-grouped-repeats", type=int, default=3
    )
    parser.add_argument("--max-seq-len", type=int, default=5120)
    parser.add_argument("--seq-size-per-block", type=int, default=64)
    parser.add_argument(
        "--kernel-seq-size-per-block",
        type=int,
        default=0,
        help=(
            "attention kernel page size; 0 uses --seq-size-per-block"
        ),
    )
    parser.add_argument("--runtime-memory-mb", type=int, default=2048)
    parser.add_argument(
        "--device-reserve-memory-bytes",
        type=int,
        default=-2048000000,
    )
    parser.add_argument(
        "--disable-flash-infer",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--prefill-only", action="store_true")
    args = parser.parse_args()
    if args.tp_size < 1:
        parser.error("--tp-size must be positive")
    if args.latency_training_grouped_repeats < 1:
        parser.error("--latency-training-grouped-repeats must be positive")
    if args.kernel_seq_size_per_block < 0:
        parser.error("--kernel-seq-size-per-block cannot be negative")
    if (
        args.kernel_seq_size_per_block
        and args.seq_size_per_block % args.kernel_seq_size_per_block != 0
    ):
        parser.error(
            "--seq-size-per-block must be divisible by "
            "--kernel-seq-size-per-block"
        )
    if args.build_dir is None:
        args.build_dir = str(Path(args.rtp_source) / "bazel-out" / "k8-opt" / "bin")
    return args


def main():
    RtpLLMEndToEndTest(parse_args()).run()


if __name__ == "__main__":
    main()
