"""Four PP smoke configurations: basic PP, mixed parallelism/MTP and two PD handoffs.

Each configuration starts only the servers under test. Numerical comparisons,
cache backend matrices and scheduler state transitions belong to functional tests.

Set PP_QWEN35_MOE_CHECKPOINT_PATH or CHECKPOINT_PATH when running the suite.
Matching *_TOKENIZER_PATH overrides are optional. All cases reuse the same MoE
checkpoint; only the mixed-parallel case enables expert parallelism.
"""

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import psutil
import requests

from rtp_llm.config.py_config_modules import MIN_WORKER_INFO_PORT_NUM, ServerConfig
from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager
from rtp_llm.test.utils.port_util import PortsContext

REQUEST_TIMEOUT = int(os.environ.get("PP_REQUEST_TIMEOUT", "300"))
VARIANTS = {
    "pdfusion_pp4_tp1": {
        "pp": 4,
        "tp": 1,
        "dp": 1,
        "ep": 1,
        "sp": 0,
        "model_type": "qwen35_moe",
        "family": "PP_QWEN35_MOE",
    },
    "pdfusion_pp2_tp2_dp2_mtp": {
        "pp": 2,
        "tp": 2,
        "dp": 2,
        "ep": 4,
        "sp": 3,
        "model_type": "qwen35_moe",
        "family": "PP_QWEN35_MOE",
    },
    "sym_mtp3": {
        "prefill_pp": 2,
        "decode_pp": 2,
        "sp": 3,
        "model_type": "qwen35_moe",
        "family": "PP_QWEN35_MOE",
    },
    "pp2_pp1_mtp3": {
        "prefill_pp": 2,
        "decode_pp": 1,
        "sp": 3,
        "model_type": "qwen35_moe",
        "family": "PP_QWEN35_MOE",
    },
}


def selected_cases():
    names = os.environ.get("PP_TEST_CASES", "pdfusion_pp4_tp1")
    names = [name.strip() for name in names.split(",") if name.strip()]
    if not names or set(names) - VARIANTS.keys():
        raise ValueError(
            f"Invalid PP cases: {names}; choose from {', '.join(VARIANTS)}"
        )
    return list(dict.fromkeys(names))


def base_smoke_args(block_size):
    # The target reserves all GPUs; each server supplies its own topology.
    args = iter(shlex.split(os.environ["SMOKE_ARGS"]))
    stripped = []
    for token in args:
        if token in (
            "--world_size",
            "--tp_size",
            "--dp_size",
            "--pp_size",
            "--ep_size",
            "--seq_size_per_block",
            "--role_type",
            "--reuse_cache",
            "--sp_type",
            "--sp_model_type",
            "--sp_checkpoint_path",
            "--sp_act_type",
            "--gen_num_per_cycle",
        ):
            next(args)
        else:
            stripped.append(token)
    return stripped + ["--seq_size_per_block", str(block_size), "--reuse_cache", "0"]


def reserve_server_port(test_case):
    context = PortsContext(num_ports=200)
    ports = context.__enter__()
    test_case.addCleanup(context.__exit__, None, None, None)
    return str(ports[0] + 100)


class PPSmokeTest(unittest.TestCase):
    case_name = "pdfusion_pp4_tp1"

    def setUp(self):
        self.variant = VARIANTS[self.case_name]
        self.model_type = self.variant["model_type"]
        family = self.variant["family"]
        self.checkpoint = os.environ.get(
            f"{family}_CHECKPOINT_PATH", os.environ.get("CHECKPOINT_PATH")
        )
        self.assertTrue(
            self.checkpoint, f"Set {family}_CHECKPOINT_PATH or CHECKPOINT_PATH"
        )
        self.tokenizer_path = os.environ.get(
            f"{family}_TOKENIZER_PATH",
            os.environ.get("TOKENIZER_PATH", self.checkpoint),
        )
        self.outputs = []
        self.routing_evidence = []
        self.shutdown_evidence = {}
        self.logs = []

    def id(self):
        return f"{super().id()}[{self.case_name}]"

    def shortDescription(self):
        return self.case_name

    def speculative_args(self):
        if not self.variant["sp"]:
            return ["--sp_type", "none"]
        return [
            "--sp_type",
            "mtp",
            "--sp_model_type",
            f"{self.model_type}_mtp",
            "--sp_checkpoint_path",
            self.checkpoint,
            "--sp_act_type",
            "BF16",
            "--gen_num_per_cycle",
            str(self.variant["sp"]),
        ]

    def start_server(self, server):
        try:
            self.assertTrue(
                server.start_server(
                    model_path=self.checkpoint,
                    model_type=self.model_type,
                    tokenizer_path=self.tokenizer_path,
                ),
                f"{self.case_name} failed to start: {server.log_file_path}",
            )
        finally:
            self.logs.append(server.log_file_path)

    def generate(self, port, prompt, tokens, label, role_addrs=None, dp_rank=0):
        started = time.monotonic()
        output = {
            "label": label,
            "port": port,
            "dp_rank": dp_rank,
            "max_new_tokens": tokens,
        }
        self.outputs.append(output)
        config = {
            "is_streaming": False,
            "max_new_tokens": tokens,
            "min_new_tokens": tokens,
            "top_k": 1,
            "top_p": 1.0,
            "random_seed": 1234,
            "return_output_ids": True,
            "aux_info": True,
        }
        if role_addrs is not None:
            config["role_addrs"] = role_addrs
        response = requests.post(
            f"http://127.0.0.1:{port}/",
            json={"prompt": prompt, "generate_config": config},
            timeout=REQUEST_TIMEOUT,
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        output.update(elapsed_seconds=time.monotonic() - started, result=result)
        self.assertEqual(result.get("error_code", 0), 0, result)
        self.assertIs(result["finished"], True, result)
        self.assertEqual(len(result["output_ids"][0]), tokens, result)
        return result

    def assert_mtp_executed(self, results):
        # Acceptance depends on the prompt. Smoke checks execution, while
        # deterministic executor tests cover accepted/rejected draft branches.
        for result in results:
            # Prefill-only requests do not enter the speculative decode loop.
            if not result["output_ids"][0]:
                continue
            aux = result["aux_info"]
            self.assertGreater(aux["speculative_draft_rounds"], 0, aux)
            self.assertEqual(
                len(aux["speculative_accepted_tokens_per_pos"]),
                self.variant["sp"],
                aux,
            )

    def dp_role_addrs(self, port, dp_rank):
        # Pin the backend: choosing a frontend port does not pin DP routing.
        backend = ServerConfig()
        backend.start_port = port
        backend.set_local_rank(dp_rank * self.variant["tp"])
        return [
            {
                "role": "PDFUSION",
                "ip": "127.0.0.1",
                "http_port": backend.http_port,
                "grpc_port": backend.rpc_server_port,
            }
        ]

    def worker_status(self, port):
        response = requests.get(
            f"http://127.0.0.1:{port}/worker_status", timeout=5
        )
        self.assertEqual(response.status_code, 200, response.text)
        status = {int(item["dp_rank"]): item for item in response.json()["results"]}
        self.assertEqual(set(status), set(range(self.variant["dp"])), status)
        return status

    def wait_frontends(self, port):
        # start_server waits only for the main port. Each DP frontend starts
        # after its own backend is ready, so wait for all of them before traffic.
        pending = {
            int(port) + rank * self.variant["tp"] * MIN_WORKER_INFO_PORT_NUM
            for rank in range(self.variant["dp"])
        }
        deadline = time.monotonic() + 60
        while pending and time.monotonic() < deadline:
            for frontend_port in list(pending):
                try:
                    response = requests.get(
                        f"http://127.0.0.1:{frontend_port}/frontend_health", timeout=2
                    )
                    if response.status_code == 200:
                        pending.remove(frontend_port)
                except requests.RequestException:
                    pass
            if pending:
                time.sleep(0.1)
        self.assertFalse(pending, f"DP frontends failed to become ready: ports={pending}")

    @staticmethod
    def finished_requests(status):
        return {
            (rank, int(task["request_id"])): task
            for rank, item in status.items()
            for task in item["finished_task_list"]
        }

    def run_dp_requests(
        self, server, cases, label, concurrent=False, wait_for_idle=False
    ):
        before = set(self.finished_requests(self.worker_status(server.port)))
        barrier = Barrier(len(cases)) if concurrent else None
        completed = {}

        def collect_completions():
            status = self.worker_status(server.port)
            completed.update(
                (key, task)
                for key, task in self.finished_requests(status).items()
                if key not in before
            )
            return {
                rank: item["running_task_info"]
                for rank, item in status.items()
                if item["running_task_info"]
            }

        def send(index, case):
            rank, prompt, tokens = case
            if barrier is not None:
                barrier.wait(timeout=30)
            return self.generate(
                server.port,
                prompt,
                tokens,
                f"{label},dp={rank},request={index}",
                self.dp_role_addrs(server.port, rank),
                rank,
            )

        if concurrent:
            with ThreadPoolExecutor(max_workers=len(cases)) as pool:
                futures = [pool.submit(send, i, case) for i, case in enumerate(cases)]
                # Worker history expires after five seconds. Collect short
                # requests while longer requests in the same burst still run.
                while not all(future.done() for future in futures):
                    collect_completions()
                    time.sleep(0.05)
                results = [future.result() for future in futures]
        else:
            results = [send(i, case) for i, case in enumerate(cases)]

        # Compare completions for the whole burst. Per-request comparisons
        # would see each other's completions or serialize the traffic.
        expected = Counter(rank for rank, _, _ in cases)
        deadline = time.monotonic() + 5
        while True:
            running = collect_completions()
            actual = Counter(rank for rank, _ in completed)
            ready = actual == expected and (not wait_for_idle or not running)
            if ready or time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        self.routing_evidence.append(
            {
                "phase": label,
                "expected": dict(expected),
                "actual": dict(actual),
                "completed": sorted(completed),
                "running": running,
            }
        )
        self.assertEqual(actual, expected, label)
        if wait_for_idle:
            # Establish the next phase's idle-DP condition, not an executor
            # drain contract: worker_status only describes RPC request records.
            self.assertFalse(running, f"{label}: DP did not become idle: {running}")
        for task in completed.values():
            self.assertEqual(
                int(task.get("error_info", {}).get("error_code", 0)), 0, task
            )
        self.assert_mtp_executed(results)
        return results

    def run_mixed_requests(self, server):
        self.wait_frontends(server.port)
        counting = "Continue counting the positive integers: 1, 2, 3,"
        # Each side must progress while the other side participates with fake work.
        self.run_dp_requests(
            server, [(0, counting, 32)], "dp0_only", wait_for_idle=True
        )
        self.run_dp_requests(
            server, [(1, counting, 32)], "dp1_only", wait_for_idle=True
        )
        # Prefill-only DP0 must join the MTP collectives used by generating DP1.
        self.run_dp_requests(
            server,
            [
                (0, "Explain how a compiler translates a program.", 0),
                (1, counting, 48),
            ],
            "prefill_only_with_generation",
            concurrent=True,
        )
        # Both replicas resume generation and receive multiple requests.
        self.run_dp_requests(
            server,
            [
                (0, "Explain how a compiler translates a program.", 64),
                (0, counting, 32),
                (1, "Describe how a distributed database handles a write request.", 48),
                (1, "List the first several months of the year:", 16),
            ],
            "concurrent_generation",
            concurrent=True,
        )
        self.assert_graceful_shutdown(server)

    def run_pdfusion(self, gpu_ids):
        variant = self.variant
        pp, tp, dp = (variant[key] for key in ("pp", "tp", "dp"))
        world_size = pp * tp * dp
        self.assertGreaterEqual(len(gpu_ids), world_size)
        args = (
            base_smoke_args(2048)
            + [
                "--pp_size", str(pp),
                "--tp_size", str(tp),
                "--dp_size", str(dp),
                "--ep_size", str(variant["ep"]),
                "--world_size", str(world_size),
                "--role_type", "PDFUSION",
                "--worker_info_port_num", str(MIN_WORKER_INFO_PORT_NUM),
            ]
            + self.speculative_args()
        )
        env = {
            "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids[:world_size]),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "RTP_LLM_STREAM_ASYNC": "0",
            "RTP_LLM_DEVICE_INPUT": "0",
        }
        if dp > 1:
            args.extend(
                [
                    "--shutdown_timeout", "30",
                    "--frontend_pre_stop_drain_seconds", "0",
                    "--dash_sc_grpc_pre_stop_drain_seconds", "0",
                    "--backend_post_frontend_drain_seconds", "0",
                    "--pre_stop_drain_signal", "0",
                ]
            )
            env.update(
                FT_SERVER_TEST="1",
                DASH_SC_GRPC_PRE_STOP_DRAIN_SECONDS="0",
                RTP_LLM_STOP_TIMEOUT_MS="30000",
                RTP_LLM_DEFERRED_GROUP_SHUTDOWN_HEADROOM_SECONDS="10",
            )
        server = MagaServerManager(
            port=reserve_server_port(self),
            env_args=env,
            role_name=self.case_name,
            smoke_args_str=shlex.join(args),
        )
        try:
            self.start_server(server)
            if dp > 1:
                self.run_mixed_requests(server)
            else:
                self.generate(
                    server.port, "Count the positive integers: 1, 2, 3,", 32, "basic_pp"
                )
        finally:
            server.stop_server()

    def run_pd(self, gpu_ids):
        prefill_pp = self.variant["prefill_pp"]
        decode_pp = self.variant["decode_pp"]
        self.assertGreaterEqual(len(gpu_ids), prefill_pp + decode_pp)
        prefill_port = reserve_server_port(self)
        decode_port = reserve_server_port(self)
        service_config = ServiceRoute(
            service_id="test",
            role_endpoints=[
                GroupEndPoint(
                    group="default",
                    prefill_endpoint=EndPoint(
                        type="Vipserver",
                        address=f"127.0.0.1:{prefill_port}",
                        protocol="http",
                        path="/",
                    ),
                    decode_endpoint=EndPoint(
                        type="Vipserver",
                        address=f"127.0.0.1:{decode_port}",
                        protocol="http",
                        path="/",
                    ),
                )
            ],
            use_local=True,
        ).model_dump_json()

        def make_server(role, port, peer_port, devices, pp):
            return MagaServerManager(
                env_args={
                    "CUDA_VISIBLE_DEVICES": ",".join(devices),
                    "WORLD_SIZE": str(pp),
                    "LOCAL_WORLD_SIZE": str(pp),
                    "MODEL_SERVICE_CONFIG": service_config,
                    "REMOTE_SERVER_PORT": peer_port,
                    "REMOTE_RPC_SERVER_IP": "localhost",
                    "RTP_LLM_STREAM_ASYNC": "0",
                },
                port=port,
                role_name=f"{self.case_name}_{role.lower()}",
                smoke_args_str=shlex.join(
                    base_smoke_args(2048)
                    + [
                        "--pp_size", str(pp),
                        "--tp_size", "1",
                        "--dp_size", "1",
                        "--ep_size", "1",
                        "--world_size", str(pp),
                        "--cache_store_rdma_mode", "0",
                        "--use_local", "1",
                        "--load_cache_timeout_ms", "120000",
                        "--role_type", role,
                    ]
                    + self.speculative_args()
                ),
            )

        prefill = make_server(
            "PREFILL", prefill_port, decode_port, gpu_ids[:prefill_pp], prefill_pp
        )
        decode = make_server(
            "DECODE",
            decode_port,
            prefill_port,
            gpu_ids[prefill_pp : prefill_pp + decode_pp],
            decode_pp,
        )
        try:
            self.start_server(decode)
            self.start_server(prefill)
            cases = [
                ("Count the positive integers in order: 1, 2, 3,", 32),
                ("Briefly explain what a distributed system is:", 64),
            ]
            # Check repeated PD handoffs; concurrent requests are covered by
            # the mixed-parallel case in the same suite.
            results = []
            for index, (prompt, tokens) in enumerate(cases):
                result = self.generate(
                    prefill.port, prompt, tokens, f"pd_handoff_{index}"
                )
                self.assertTrue(result["aux_info"]["pd_sep"], result)
                results.append(result)
            self.assert_mtp_executed(results)
        finally:
            prefill.stop_server()
            decode.stop_server()

    def assert_graceful_shutdown(self, server):
        with server._state_lock:
            process = server._server_process
        self.assertIsNotNone(process)
        self.assertIsNone(process.poll(), "server exited before shutdown")
        children = psutil.Process(process.pid).children(recursive=True)
        started = time.monotonic()
        process.terminate()
        try:
            try:
                exit_code = process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                self.fail(f"PP shutdown timed out: {server.log_file_path}")
            self.assertEqual(exit_code, 0, server.log_file_path)
            _, alive = psutil.wait_procs(children, timeout=5)
            self.assertFalse(
                alive, f"PP shutdown left children alive: {[p.pid for p in alive]}"
            )
            log = Path(server.log_file_path).read_text(errors="replace")
            failure = re.search(
                r"Force killing process |Timed out waiting for .* process group to exit"
                r"|engine stop failed during backend shutdown"
                r"|\*\*\* SIG(?:SEGV|FPE|ILL|ABRT|BUS).*received by PID",
                log,
            )
            self.assertIsNone(
                failure, f"shutdown failure: {failure}; log={server.log_file_path}"
            )
            self.shutdown_evidence.update(
                scenario="after_requests",
                elapsed_seconds=time.monotonic() - started,
                exit_code=exit_code,
            )
        finally:
            # After the parent exits, stop_server cannot discover orphaned children.
            if process.poll() is not None:
                _, alive = psutil.wait_procs(children, timeout=0)
                for child in alive:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                psutil.wait_procs(alive, timeout=5)

    def test_selected_case(self):
        report = {
            "case": self.case_name,
            "model_type": self.model_type,
            "checkpoint": self.checkpoint,
            "topology": self.variant,
            "outputs": self.outputs,
            "routing": self.routing_evidence,
            "shutdown": self.shutdown_evidence,
            "logs": self.logs,
            "passed": False,
        }
        try:
            gpu_ids = [str(gpu_id) for gpu_id in get_gpu_ids()]
            if "prefill_pp" in self.variant:
                self.run_pd(gpu_ids)
            else:
                self.run_pdfusion(gpu_ids)
            report["passed"] = True
        except Exception as error:
            report["error"] = str(error)
            raise
        finally:
            output_dir = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "."))
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{self.case_name}.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name in selected_cases():
        case = PPSmokeTest("test_selected_case")
        case.case_name = name
        suite.addTest(case)
    return suite


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", help="Comma-separated PP cases; overrides PP_TEST_CASES"
    )
    parser.add_argument("--list-cases", action="store_true", help="List cases and exit")
    options, unittest_args = parser.parse_known_args()
    if options.list_cases:
        for name, topology in VARIANTS.items():
            print(f"{name}: {topology}")
        sys.exit(0)
    if options.cases is not None:
        os.environ["PP_TEST_CASES"] = options.cases
    unittest.main(argv=[sys.argv[0]] + unittest_args)
