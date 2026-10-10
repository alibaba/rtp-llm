"""PP smoke cases selected with --cases or PP_TEST_CASES (default: sym).

PP=1 supplies matching generation references. The retained cases cover distinct
PD routes, middle stages, MTP step/cache paths, DP recovery and resident prefixes.
The MTP DP recovery case also checks shutdown once; PPExecutor/PPScheduler unit
tests cover fake-result consumption, slot draining and idle scheduler wakeup.

Pass CHECKPOINT_PATH / MODEL_TYPE for an individual target, or
PP_QWEN3_CHECKPOINT_PATH and PP_QWEN35_DENSE_CHECKPOINT_PATH for mixed suites.
SP_MODEL_TYPE / SP_CHECKPOINT_PATH / SP_TYPE retain the draft-model overrides.
--list-cases lists the catalogue without starting a server.
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

import psutil
import requests

from rtp_llm.config.py_config_modules import MIN_WORKER_INFO_PORT_NUM, ServerConfig
from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager
from rtp_llm.test.utils.port_util import PortsContext

# Draft model registry name for the MTP variants, overridable per checkpoint family.
SP_MODEL_TYPE = os.environ.get(
    "SP_MODEL_TYPE", os.environ.get("PD_SP_MODEL_TYPE", "qwen35_dense_mtp")
)
REQUEST_TIMEOUT = int(os.environ.get("PP_REQUEST_TIMEOUT", "300"))

# /** Keep one representative for each PD stage/TP routing path. */
VARIANTS = {
    "sym": {"prefill_pp": 2, "prefill_tp": 1, "decode_pp": 2, "decode_tp": 1},
    "sym_mtp1": {
        "prefill_pp": 2,
        "prefill_tp": 1,
        "decode_pp": 2,
        "decode_tp": 1,
        "sp": 1,
    },
    "sym_mtp3": {
        "prefill_pp": 2,
        "prefill_tp": 1,
        "decode_pp": 2,
        "decode_tp": 1,
        "sp": 3,
    },
    "asym": {"prefill_pp": 2, "prefill_tp": 1, "decode_pp": 2, "decode_tp": 2},
    "conv": {"prefill_pp": 2, "prefill_tp": 2, "decode_pp": 2, "decode_tp": 1},
    # Requires a model with PP support and a sparse MLA backend with CP prefill.
    "cp_sharded": {
        "prefill_pp": 2,
        "prefill_tp": 2,
        "decode_pp": 2,
        "decode_tp": 1,
        "prefill_cp": 2,
        "kv_cache_sharded": True,
    },
    "cp_full": {
        "prefill_pp": 2,
        "prefill_tp": 2,
        "decode_pp": 2,
        "decode_tp": 1,
        "prefill_cp": 2,
        "kv_cache_sharded": False,
    },
    "cp_full_decode_tp2": {
        "prefill_pp": 2,
        "prefill_tp": 2,
        "decode_pp": 2,
        "decode_tp": 2,
        "prefill_cp": 2,
        "kv_cache_sharded": False,
    },
    "cp_full_alltoall": {
        "prefill_pp": 2,
        "prefill_tp": 2,
        "decode_pp": 2,
        "decode_tp": 1,
        "prefill_cp": 2,
        "cp_rotate_method": "ALLTOALL",
    },
    "pp1_cp_full": {
        "prefill_pp": 1,
        "prefill_tp": 2,
        "decode_pp": 1,
        "decode_tp": 1,
        "prefill_cp": 2,
        "kv_cache_sharded": False,
    },
    "pp1_cp_full_tp2": {
        "prefill_pp": 1,
        "prefill_tp": 2,
        "decode_pp": 1,
        "decode_tp": 2,
        "prefill_cp": 2,
        "kv_cache_sharded": False,
    },
    # P runs ordinary prefill; D uses PREFILL_CP to import whole MLA KV blocks.
    # PP=1 on both sides permits classic DeepSeek-V2 (MODEL_TYPE=deepseek2),
    # which supports neither PP nor CP prefill.
    "pp1_mla_cp_import": {
        "prefill_pp": 1,
        "prefill_tp": 2,
        "decode_pp": 1,
        "decode_tp": 2,
        "decode_prefill_cp": True,
    },
    "pp1_pp2": {"prefill_pp": 1, "prefill_tp": 1, "decode_pp": 2, "decode_tp": 1},
    "pp2_pp1_tp2": {"prefill_pp": 2, "prefill_tp": 2, "decode_pp": 1, "decode_tp": 2},
}

# A PDFUSION case uses one server. PD cases above retain their two-server layout.
VARIANTS.update(
    {
        "pdfusion_mtp_regression": {"pp": 2, "tp": 2, "dp": 1, "ep": 1},
        "fake_pp2_tp2_dp2": {"pp": 2, "tp": 2, "dp": 2, "ep": 4, "sp": 0},
        "fake_pp2_tp2_dp2_mtp": {"pp": 2, "tp": 2, "dp": 2, "ep": 4, "sp": 3},
        "multi_task_prompt_pp2_tp2": {"pp": 2, "tp": 2, "dp": 1, "ep": 1, "sp": 0},
        "multi_task_prompt_pp2_dp2": {"pp": 2, "tp": 1, "dp": 2, "ep": 2, "sp": 0},
        "multi_task_prompt_pp2_pd": {
            "pp": 2,
            "tp": 1,
            "dp": 1,
            "ep": 1,
            "sp": 0,
            "pd": True,
        },
        "multi_task_prompt_pp2_cp2": {
            "pp": 2,
            "tp": 2,
            "dp": 1,
            "ep": 1,
            "sp": 0,
            "cp": 2,
            "pd": True,
            "decode_tp": 2,
        },
        "multi_task_prompt_pp2_mtp": {
            "pp": 2,
            "tp": 1,
            "dp": 1,
            "ep": 1,
            "sp": 1,
            "block_size": 2048,
        },
        "multi_task_prompt_pp2_pd_mtp": {
            "pp": 2,
            "tp": 1,
            "dp": 1,
            "ep": 1,
            "sp": 1,
            "block_size": 2048,
            "pd": True,
        },
    }
)

VARIANTS["pdfusion_pp4_tp1"] = {"pp": 4, "tp": 1, "dp": 1, "ep": 1, "sp": 0}
VARIANTS["fake_pp2_tp2_dp2_mtp"]["graceful_shutdown"] = True


def selected_cases():
    names = os.environ.get("PP_TEST_CASES", os.environ.get("PD_VARIANT", "sym"))
    names = [name.strip() for name in names.split(",") if name.strip()]
    unknown = set(names) - VARIANTS.keys()
    if not names or unknown:
        raise ValueError(
            f"Invalid PP cases: {names}; choose from {', '.join(VARIANTS)}"
        )
    return list(dict.fromkeys(names))


def speculative_args(checkpoint, steps):
    if not steps:
        return ["--sp_type", "none"]
    sp_type = os.environ.get("SP_TYPE", "mtp")
    if sp_type not in ("mtp", "eagle", "dspark"):
        raise ValueError(
            f"PP speculative cases require SP_TYPE=mtp/eagle/dspark, got {sp_type}"
        )
    return [
        "--sp_type",
        sp_type,
        "--sp_model_type",
        SP_MODEL_TYPE,
        "--sp_checkpoint_path",
        os.environ.get("SP_CHECKPOINT_PATH", checkpoint),
        "--sp_act_type",
        os.environ.get("SP_ACT_TYPE", "BF16"),
        "--gen_num_per_cycle",
        str(steps),
    ]


def base_smoke_args(default_seq_size_per_block=16):
    # SMOKE_ARGS carries the BUILD-level GPU reservation; strip its parallelism
    # flags so each server sets its own. Preserve the original block sizes:
    # PD uses 16 to exercise block routing; the Qwen3.5 MTP regression uses 2048.
    seq_size_per_block = int(
        os.environ.get(
            "PP_SEQ_SIZE_PER_BLOCK",
            os.environ.get("PD_SEQ_SIZE_PER_BLOCK", str(default_seq_size_per_block)),
        )
    )
    args = shlex.split(os.environ["SMOKE_ARGS"])
    stripped = []
    skip_next = False
    for token in args:
        if skip_next:
            skip_next = False
            continue
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
            skip_next = True
            continue
        stripped.append(token)
    return stripped + ["--seq_size_per_block", str(seq_size_per_block)]


def reserve_server_port(test_case):
    context = PortsContext(num_ports=200)
    ports = context.__enter__()
    test_case.addCleanup(context.__exit__, None, None, None)
    return str(ports[0] + 100)


class DPRequestRouting:
    def dp_role_addrs(self, port, tp, busy_dp):
        # /** Frontend ports do not pin DP routing; select the backend explicitly. */
        backend = ServerConfig()
        backend.start_port = port
        backend.set_local_rank(busy_dp * tp)
        return [{
            "role": "PDFUSION",
            "ip": "127.0.0.1",
            "http_port": backend.http_port,
            "grpc_port": backend.rpc_server_port,
        }]

    def worker_status(self, port, dp):
        response = requests.get(f"http://127.0.0.1:{port}/worker_status", timeout=5)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIn("results", body, body)
        results = body["results"]
        status = {int(item["dp_rank"]): item for item in results}
        self.assertEqual(set(status), set(range(dp)), results)
        return status

    def finished_requests(self, status):
        return {
            (rank, int(task["request_id"])): task
            for rank, item in status.items()
            for task in item["finished_task_list"]
        }

    def generate_on_dp(
        self, port, variant, busy_dp, tokens, label,
        prompt="Continue counting the positive integers: 1, 2, 3,",
        task_id=None,
    ):
        before = self.worker_status(port, variant["dp"])
        result = self.generate(
            port, prompt, tokens, label,
            self.dp_role_addrs(port, variant["tp"], busy_dp), task_id,
        )

        # /** The frontend assigns request IDs; inspect completion deltas to
        # verify the actual destination rather than echoed routing arguments. */
        previous = set(self.finished_requests(before))
        # /** Responses can precede metadata cleanup. Wait for completion and
        # idle replicas before the next request, without sampling a RUNNING phase. */
        deadline = time.monotonic() + 5
        while True:
            status = self.worker_status(port, variant["dp"])
            finished = self.finished_requests(status)
            added = finished.keys() - previous
            running = {
                rank: item["running_task_info"]
                for rank, item in status.items() if item["running_task_info"]
            }
            if (added and not running) or time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        self.assertEqual(
            [rank for rank, _ in added], [busy_dp],
            f"{label}: expected one completed request on DP {busy_dp}, got {added}",
        )
        self.assertFalse(running, f"{label}: requests did not drain: {running}")
        for key in added:
            self.assertEqual(
                int(finished[key].get("error_info", {}).get("error_code", 0)), 0,
                finished[key],
            )
        return result


class PPModelTest(unittest.TestCase):
    def setUp(self):
        variant = VARIANTS[self.case_name]
        dense = variant.get("sp", 0) > 0 or self.case_name == "pdfusion_mtp_regression"
        family = "PP_QWEN35_DENSE" if dense else "PP_QWEN3"
        self.checkpoint = os.environ.get(
            f"{family}_CHECKPOINT_PATH", os.environ.get("CHECKPOINT_PATH")
        )
        self.model_type = os.environ.get(
            "MODEL_TYPE", os.environ.get("PD_MODEL_TYPE", "qwen35_dense" if dense else "qwen_3")
        )
        self.tokenizer_path = os.environ.get(
            f"{family}_TOKENIZER_PATH", os.environ.get("TOKENIZER_PATH", self.checkpoint)
        )

    def assert_draft_accepted(self, results, label, propose_step=None):
        speculative = [
            result["aux_info"] for result in results
            if len(result["output_ids"][0]) > 1
        ]
        if propose_step is not None:
            for aux in speculative:
                self.assertEqual(
                    len(aux["speculative_accepted_tokens_per_pos"]), propose_step, label,
                )
        self.assertTrue(
            any(
                aux["speculative_draft_rounds"] > 0
                and sum(aux["speculative_accepted_tokens_per_pos"]) > 0
                for aux in speculative
            ),
            f"{label}: no draft token was accepted",
        )
        if propose_step is not None and propose_step > 1:
            self.assertTrue(
                any(
                    aux["speculative_accepted_tokens_per_pos"][-1] > 0
                    for aux in speculative
                ),
                f"{label}: the multi-step draft chain never reached acceptance",
            )


class PPTopologyTest(DPRequestRouting, PPModelTest):
    case_name = "sym"

    def id(self):
        return f"{super().id()}[{self.case_name}]"

    def shortDescription(self):
        return self.case_name

    def generate(self, port, prompt, max_new_tokens, label="", role_addrs=None, task_id=None):
        started = time.monotonic()
        output = {"label": label, "port": port, "max_new_tokens": max_new_tokens}
        self.outputs.append(output)
        generate_config = {
            "is_streaming": False,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": max_new_tokens,
            "top_k": 1,
            "top_p": 1.0,
            "random_seed": 1234,
            "return_output_ids": True,
            "aux_info": True,
        }
        if role_addrs is not None:
            generate_config["role_addrs"] = role_addrs
        if task_id is not None:
            generate_config["task_id"] = task_id
        response = requests.post(
            f"http://127.0.0.1:{port}/",
            json={
                "prompt": prompt,
                "generate_config": generate_config,
            },
            timeout=REQUEST_TIMEOUT,
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        output.update(elapsed_seconds=time.monotonic() - started, result=result)
        self.assertTrue(result["finished"], result)
        self.assertEqual(len(result["output_ids"][0]), max_new_tokens, result)
        return result

    def run_baseline(self, checkpoint, devices, tp_size, block_size=16):
        """Target-only PP=1 reference for the PDFUSION cases."""
        args = (
            base_smoke_args(default_seq_size_per_block=block_size)
            + [
                "--tp_size",
                str(tp_size),
                "--dp_size",
                "1",
                "--pp_size",
                "1",
                "--world_size",
                str(tp_size),
                "--ep_size",
                "1",
                "--reuse_cache",
                "0",
                "--role_type",
                "PDFUSION",
            ]
            + speculative_args(checkpoint, 0)
        )
        server = MagaServerManager(
            port=reserve_server_port(self),
            env_args={
                "CUDA_VISIBLE_DEVICES": devices,
                "WORLD_SIZE": str(tp_size),
                "RTP_LLM_STREAM_ASYNC": "0",
            },
            role_name=f"{self.case_name}_baseline",
            smoke_args_str=shlex.join(args),
        )
        try:
            self.assertTrue(
                server.start_server(
                    model_path=checkpoint,
                    model_type=self.model_type,
                    tokenizer_path=self.tokenizer_path,
                ),
                f"baseline failed to start: {server.log_file_path}",
            )
            return [
                self.generate(server.port, prompt, tokens)
                for prompt, tokens in self.cases()
            ]
        finally:
            server.stop_server()

    def run_pd(
        self,
        checkpoint,
        prefill_devices,
        decode_devices,
        prefill_pp,
        prefill_tp,
        decode_pp,
        decode_tp,
        prefill_cp=1,
        kv_cache_sharded=False,
        decode_prefill_cp=False,
        sp=0,
        cp_rotate_method="ALL_GATHER",
        label="actual",
    ):
        prefill_port = reserve_server_port(self)
        decode_port = reserve_server_port(self)
        group_endpoint = GroupEndPoint(
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
        service_config = ServiceRoute(
            service_id="test", role_endpoints=[group_endpoint], use_local=True
        ).model_dump_json()

        common_pd = (
            "--dp_size 1 --ep_size 1 "
            "--cache_store_rdma_mode 0 --use_local 1 --load_cache_timeout_ms 120000 "
            "--reuse_cache 0"
        )
        prefill_ws = prefill_pp * prefill_tp
        decode_ws = decode_pp * decode_tp
        """
        CP is asymmetric: prefill runs the selected rotation; decode
        declares the peer as CP (PREFILL_CP) and mirrors the CP shape config
        (prefill_cp_size / kv_cache_sharded) from its own settings.
        """
        # Both sides need the draft model and matching draft precision.
        sp_args = speculative_args(checkpoint, sp)
        prefill_args = f"--pp_size {prefill_pp} --tp_size {prefill_tp} --world_size {prefill_ws} {common_pd}"
        if prefill_cp > 1:
            prefill_args += (
                f" --cp_rotate_method {cp_rotate_method} --prefill_cp_size {prefill_cp}"
            )
            if kv_cache_sharded:
                prefill_args += " --prefill_cp_kv_cache_sharded 1"
        decode_args = f"--pp_size {decode_pp} --tp_size {decode_tp} --world_size {decode_ws} {common_pd}"
        if prefill_cp > 1 or decode_prefill_cp:
            decode_args += " --cp_rotate_method PREFILL_CP"
            if kv_cache_sharded:
                decode_args += (
                    f" --prefill_cp_size {prefill_cp} --prefill_cp_kv_cache_sharded 1"
                )
        prefill = MagaServerManager(
            env_args={
                "CUDA_VISIBLE_DEVICES": prefill_devices,
                "WORLD_SIZE": str(prefill_ws),
                "MODEL_SERVICE_CONFIG": service_config,
                "REMOTE_SERVER_PORT": str(decode_port),
                "REMOTE_RPC_SERVER_IP": "localhost",
                "RTP_LLM_STREAM_ASYNC": "0",
            },
            port=prefill_port,
            role_name=f"{self.case_name}_{label}_prefill_pp{prefill_pp}",
            smoke_args_str=shlex.join(
                base_smoke_args(default_seq_size_per_block=2048 if sp else 16)
                + shlex.split(prefill_args)
                + sp_args
                + ["--role_type", "PREFILL"]
            ),
        )
        decode = MagaServerManager(
            env_args={
                "CUDA_VISIBLE_DEVICES": decode_devices,
                "WORLD_SIZE": str(decode_ws),
                "MODEL_SERVICE_CONFIG": service_config,
                "REMOTE_SERVER_PORT": str(prefill_port),
                "REMOTE_RPC_SERVER_IP": "localhost",
                "RTP_LLM_STREAM_ASYNC": "0",
            },
            port=decode_port,
            role_name=f"{self.case_name}_{label}_decode_pp{decode_pp}",
            smoke_args_str=shlex.join(
                base_smoke_args(default_seq_size_per_block=2048 if sp else 16)
                + shlex.split(decode_args)
                + sp_args
                + ["--role_type", "DECODE"]
            ),
        )
        try:
            self.assertTrue(
                decode.start_server(
                    model_path=checkpoint,
                    model_type=self.model_type,
                    tokenizer_path=self.tokenizer_path,
                ),
                f"decode failed to start: {decode.log_file_path}",
            )
            self.assertTrue(
                prefill.start_server(
                    model_path=checkpoint,
                    model_type=self.model_type,
                    tokenizer_path=self.tokenizer_path,
                ),
                f"prefill failed to start: {prefill.log_file_path}",
            )
            # PD entrance is the prefill instance (decode_entrance=false).
            return [
                self.generate(
                    prefill.port,
                    prompt,
                    tokens,
                    label=f"pd_{label}_p{prefill_pp}_d{decode_pp}",
                )
                for prompt, tokens in self.cases()
            ]
        finally:
            prefill.stop_server()
            decode.stop_server()

    def cases(self):
        return [
            ("The capital of France is", 1),
            ("Count the positive integers in order: 1, 2, 3,", 32),
            ("Briefly explain what a distributed system is:", 64),
        ]

    def wait_frontends(self, ports):
        pending = set(ports)
        deadline = time.monotonic() + 60
        while pending and time.monotonic() < deadline:
            for port in list(pending):
                try:
                    response = requests.get(
                        f"http://127.0.0.1:{port}/frontend_health", timeout=2
                    )
                    if response.status_code == 200:
                        pending.remove(port)
                except requests.RequestException:
                    pass
            if pending:
                time.sleep(0.1)
        self.assertFalse(
            pending, f"DP frontends failed to become ready: ports={pending}"
        )

    def assert_graceful_shutdown(self, server, variant):
        """Signal the parent and require every backend to finish the complete stop path."""
        with server._state_lock:
            process = server._server_process
        self.assertIsNotNone(process)
        self.assertIsNone(process.poll(), "server exited before shutdown was requested")
        children = psutil.Process(process.pid).children(recursive=True)
        started = time.monotonic()
        process.terminate()
        try:
            try:
                exit_code = process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                self.fail(f"PP shutdown did not finish within 90s: {server.log_file_path}")
            self.assertEqual(exit_code, 0, server.log_file_path)
            _, alive = psutil.wait_procs(children, timeout=5)
            self.assertFalse(
                alive, f"PP shutdown left children alive: {[p.pid for p in alive]}"
            )
            log = Path(server.log_file_path).read_text(errors="replace")
            completed = Counter(
                tuple(map(int, match))
                for match in re.findall(
                    r"PP shutdown completed: pp_rank=(\d+), tp_rank=(\d+), dp_rank=(\d+)",
                    log,
                )
            )
            expected = Counter(
                (pp_rank, tp_rank, dp_rank)
                for pp_rank in range(variant["pp"])
                for tp_rank in range(variant["tp"])
                for dp_rank in range(variant["dp"])
            )
            self.assertEqual(completed, expected, server.log_file_path)
            backend_stops = log.count("BackendManager stopped successfully")
            self.assertEqual(backend_stops, sum(expected.values()), server.log_file_path)
            failure = re.search(
                r"Force killing process |Timed out waiting for .* process group to exit"
                r"|engine stop failed during backend shutdown"
                r"|\*\*\* SIG(?:SEGV|FPE|ILL|ABRT|BUS).*received by PID",
                log,
            )
            self.assertIsNone(failure, f"shutdown failure: {failure}; log={server.log_file_path}")
            self.shutdown_evidence.update(
                scenario="idle_after_dp_recovery",
                backend_stops=backend_stops,
                elapsed_seconds=time.monotonic() - started,
                exit_code=exit_code,
                completed_ranks=sorted(completed),
                log_file=server.log_file_path,
            )
        finally:
            # Once the parent exits, stop_server cannot discover orphaned children.
            # Assert graceful exit above before cleaning up a failing case.
            if process.poll() is not None:
                _, alive = psutil.wait_procs(children, timeout=0)
                for child in alive:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                psutil.wait_procs(alive, timeout=5)

    def run_pdfusion(self, checkpoint, gpu_ids, variant):
        pp, tp, dp = (variant[key] for key in ("pp", "tp", "dp"))
        world_size = pp * tp * dp
        self.assertGreaterEqual(len(gpu_ids), world_size)
        args = (
            base_smoke_args(default_seq_size_per_block=variant.get("block_size", 2048 if variant.get("sp") else 16))
            + [
                "--pp_size",
                str(pp),
                "--tp_size",
                str(tp),
                "--dp_size",
                str(dp),
                "--ep_size",
                str(variant["ep"]),
                "--world_size",
                str(world_size),
                "--role_type",
                "PDFUSION",
                "--reuse_cache",
                "0",
                "--worker_info_port_num",
                str(MIN_WORKER_INFO_PORT_NUM),
            ]
            + speculative_args(checkpoint, variant["sp"])
        )
        env = {
            "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids[:world_size]),
            "WORLD_SIZE": str(world_size),
            "LOCAL_WORLD_SIZE": str(world_size),
            "RTP_LLM_STREAM_ASYNC": "0",
            "RTP_LLM_DEVICE_INPUT": "0",
        }
        if variant.get("graceful_shutdown"):
            args.extend(
                [
                    "--shutdown_timeout",
                    "30",
                    "--frontend_pre_stop_drain_seconds",
                    "0",
                    "--dash_sc_grpc_pre_stop_drain_seconds",
                    "0",
                    "--backend_post_frontend_drain_seconds",
                    "0",
                    "--pre_stop_drain_signal",
                    "0",
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
            self.assertTrue(
                server.start_server(
                    model_path=checkpoint,
                    model_type=self.model_type,
                    tokenizer_path=self.tokenizer_path,
                ),
                f"{self.case_name} failed to start: {server.log_file_path}",
            )
            if dp == 1:
                outputs = [
                    self.generate(server.port, prompt, tokens, self.case_name)
                    for prompt, tokens in self.cases()
                ]
                return outputs

            ports = [
                int(server.port) + rank * tp * MIN_WORKER_INFO_PORT_NUM for rank in range(dp)
            ]
            self.wait_frontends(ports)
            outputs = []
            # /** Returning to each previously idle replica checks recovery in both directions. */
            for busy_dp in (0, 1, 0, 1):
                result = self.generate_on_dp(
                    server.port, variant, busy_dp, 64,
                    f"request={len(outputs)},dp={busy_dp}",
                )
                if outputs:
                    self.assertEqual(result["output_ids"], outputs[0]["output_ids"])
                outputs.append(result)
            if variant["sp"]:
                self.assert_draft_accepted(outputs, self.case_name, variant["sp"])
            if variant.get("graceful_shutdown"):
                self.assert_graceful_shutdown(server, variant)
            return outputs
        finally:
            server.stop_server()

    def check_pd_variant(self, checkpoint, gpu_ids, variant):
        decode_tp = variant["decode_tp"]
        prefill_tp = variant["prefill_tp"]
        prefill_pp = variant["prefill_pp"]
        decode_pp = variant["decode_pp"]
        decode_prefill_cp = variant.get("decode_prefill_cp", False)
        sp = variant.get("sp", 0)
        if decode_prefill_cp:
            # MLA-only: with MHA the slice plan assumes a rotating prefill.
            self.assertNotEqual(
                self.model_type,
                "qwen_3",
                f"case {self.case_name} needs an MLA checkpoint (MODEL_TYPE=deepseek2)",
            )
        prefill_gpus = prefill_pp * prefill_tp
        decode_gpus = decode_pp * decode_tp
        need = prefill_gpus + decode_gpus
        self.assertGreaterEqual(len(gpu_ids), need, f"need {need} GPUs, got {gpu_ids}")
        baseline_variant = dict(variant, prefill_pp=1, decode_pp=1)
        baseline = self.run_pd(
            checkpoint,
            prefill_devices=",".join(gpu_ids[:prefill_tp]),
            decode_devices=",".join(gpu_ids[prefill_gpus : prefill_gpus + decode_tp]),
            label="baseline",
            **baseline_variant,
        )
        actual = self.run_pd(
            checkpoint,
            prefill_devices=",".join(gpu_ids[:prefill_gpus]),
            decode_devices=",".join(gpu_ids[prefill_gpus : prefill_gpus + decode_gpus]),
            **variant,
        )
        for (prompt, _), base, got in zip(self.cases(), baseline, actual):
            self.assertEqual(
                got["output_ids"],
                base["output_ids"],
                f"PP PD diverges from the matching PP=1 PD baseline on: {prompt[:40]}",
            )
        if sp:
            self.assert_draft_accepted(actual, self.case_name, sp)

    def test_selected_topology(self):
        checkpoint = self.checkpoint
        self.assertTrue(checkpoint, "Pass --test_env=CHECKPOINT_PATH=<checkpoint>")
        variant = VARIANTS[self.case_name]
        gpu_ids = [str(x) for x in get_gpu_ids()]
        self.outputs = []
        self.shutdown_evidence = {}
        report = {
            "case": self.case_name,
            "model_type": self.model_type,
            "checkpoint": checkpoint,
            "sp_type": (
                os.environ.get("SP_TYPE", "mtp") if variant.get("sp") else "none"
            ),
            "sp_model_type": SP_MODEL_TYPE if variant.get("sp") else None,
            "topology": variant,
            "outputs": self.outputs,
            "shutdown": self.shutdown_evidence,
            "passed": False,
        }
        try:
            if "prefill_pp" in variant:
                self.check_pd_variant(checkpoint, gpu_ids, variant)
            elif variant["dp"] > 1:
                self.run_pdfusion(checkpoint, gpu_ids, variant)
            else:
                self.assertGreaterEqual(len(gpu_ids), variant["pp"] * variant["tp"])
                baseline = self.run_baseline(
                    checkpoint,
                    ",".join(gpu_ids[: variant["tp"]]),
                    variant["tp"],
                    block_size=variant.get("block_size", 2048 if variant.get("sp") else 16),
                )
                actual = self.run_pdfusion(checkpoint, gpu_ids, variant)
                self.assertEqual(
                    [item["output_ids"] for item in actual],
                    [item["output_ids"] for item in baseline],
                    f"{self.case_name} differs from target-only PP=1 baseline",
                )
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


class MtpPPTest(PPModelTest):
    case_name = "pdfusion_mtp_regression"

    def generate(self, server, prompt, max_new_tokens, sampling_options=None):
        generate_config = {
            "is_streaming": False,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": max_new_tokens,
            "top_k": 1,
            "top_p": 1.0,
            "random_seed": 1234,
            "return_output_ids": True,
            "aux_info": True,
        }
        generate_config.update(sampling_options or {})
        response = requests.post(
            f"http://127.0.0.1:{server.port}/",
            json={
                "prompt": prompt,
                "generate_config": generate_config,
            },
            timeout=180,
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["finished"], result)
        self.assertEqual(len(result["output_ids"]), 1, result)
        self.assertEqual(len(result["output_ids"][0]), max_new_tokens, result)
        self.assertGreater(result["aux_info"]["iter_count"], 0, result)
        return result

    def run_variant(self, checkpoint, propose_step, reuse_cache=False):
        variant = f"pp2_mtp{propose_step}_reuse{int(reuse_cache)}"
        topology = VARIANTS["pdfusion_mtp_regression"]
        world_size = topology["pp"] * topology["tp"] * topology["dp"]
        gpu_ids = [str(x) for x in get_gpu_ids()]
        self.assertGreaterEqual(len(gpu_ids), world_size)
        args = base_smoke_args(default_seq_size_per_block=2048)
        for parallelism in ("pp", "tp", "dp", "ep"):
            args += [f"--{parallelism}_size", str(topology[parallelism])]
        args += [
            "--world_size",
            str(world_size),
            "--role_type",
            "PDFUSION",
            "--reuse_cache",
            str(int(reuse_cache)),
        ]
        args += speculative_args(checkpoint, propose_step)
        server = MagaServerManager(
            port=reserve_server_port(self),
            env_args={
                "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids[:world_size]),
                "WORLD_SIZE": str(world_size),
                "LOCAL_WORLD_SIZE": str(world_size),
                "RTP_LLM_STREAM_ASYNC": "0",
            },
            role_name=variant,
            smoke_args_str=shlex.join(args),
        )
        cases = [
            ("The capital of France is", 1),
            ("Count the positive integers in order: 1, 2, 3,", 64),
            ("The quick brown fox jumps over the lazy dog. " * 220 + "Continue:", 64),
        ]
        prefix = cases[-1][0]
        # Repeat the full prompt, then change only its continuation. Reused MTP
        # KV must not retain the successor token from the previous request.
        cases += [
            (prefix, 64),
            (prefix + " Write a poem:", 32),
            (prefix + " Write a recipe:", 32),
            (
                "Continue this pattern: red blue red blue red blue",
                32,
                {
                    "repetition_penalty": 1.2,
                    "presence_penalty": 0.3,
                    "frequency_penalty": 0.2,
                    "no_repeat_ngram_size": 3,
                },
            ),
        ]
        outputs = {}
        output_dir = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "."))
        try:
            self.assertTrue(
                server.start_server(
                    model_path=checkpoint,
                    model_type=self.model_type,
                    tokenizer_path=self.tokenizer_path,
                ),
                f"{variant} failed to start: {server.log_file_path}",
            )
            outputs["serial"] = []
            for case in cases:
                outputs["serial"].append(self.generate(server, *case))
            self.assertEqual(outputs["serial"][2]["aux_info"]["reuse_len"], 0)
            if reuse_cache:
                for result in outputs["serial"][3:6]:
                    self.assertGreater(result["aux_info"]["reuse_len"], 0, result)
            else:
                for result in outputs["serial"]:
                    self.assertEqual(result["aux_info"]["reuse_len"], 0, result)
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(self.generate, server, *case)
                    for case in (cases[4], cases[-1])
                ]
                outputs["concurrent"] = [future.result() for future in futures]
            if reuse_cache:
                result = outputs["concurrent"][0]
                self.assertGreater(result["aux_info"]["reuse_len"], 0, result)
        finally:
            server.stop_server()
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{variant}.json").write_text(
                json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return outputs

    def test_pdfusion_mtp_matches_target_generation(self):
        checkpoint = self.checkpoint
        self.assertTrue(
            checkpoint, "Pass --test_env=CHECKPOINT_PATH=<Qwen3.5-27B checkpoint>"
        )
        baseline = self.run_variant(checkpoint, 0)
        variants = ((step, reuse) for step in (1, 3) for reuse in (False, True))
        for propose_step, reuse_cache in variants:
            with self.subTest(propose_step=propose_step, reuse_cache=reuse_cache):
                actual = self.run_variant(
                    checkpoint, propose_step, reuse_cache=reuse_cache
                )
                for mode in ("serial", "concurrent"):
                    self.assertEqual(
                        [result["output_ids"] for result in actual[mode]],
                        [result["output_ids"] for result in baseline[mode]],
                        f"{mode}: MTP {propose_step} differs from target-only PP",
                    )
                self.assert_draft_accepted(actual["serial"], f"MTP {propose_step}", propose_step)


# System prompts long enough to span multiple KV blocks at the smoke block size
# (seq_size_per_block=16), so the resident prefix yields reusable whole blocks.
MULTI_TASK_PROMPTS = [
    {
        "task_id": "translator",
        "prompt": (
            "You are a professional translator. Translate the user's text into French. "
            "Preserve the original meaning, tone, and punctuation as faithfully as possible. "
            "Output only the translation, without any explanation or extra commentary.\n"
        ),
    },
    {
        "task_id": "counter",
        "prompt": (
            "You are a precise sequence assistant. Continue the given numeric or patterned "
            "sequence exactly, without skipping, repeating, or reordering any element. "
            "Output only the continuation, with no surrounding words or explanation.\n"
        ),
    },
]

# Each case sends a user prompt under a task_id; updatePrefix prepends the resident
# system-prompt tokens, which must be reused from the startup-built KV.
MULTI_TASK_CASES = [
    ("translator", "The capital of France is", 8),
    ("counter", "Count the positive integers in order: 1, 2, 3,", 16),
]

# Long prompts for models requiring large seq_size_per_block (e.g. Qwen3.5 MTP needs 2048).
# Each prompt must exceed block_size tokens so at least one complete block is registered
# as resident KV after dropLastPartialBlock.
_LONG_TRANSLATOR = (
    "You are a professional translator specializing in technical and literary texts. "
    "Translate the user's text into French while preserving the original meaning, tone, "
    "and punctuation as faithfully as possible. Output only the translation without any "
    "explanation or extra commentary. Maintain formal register unless the source is clearly "
    "informal. Keep proper nouns untranslated unless a standard French equivalent exists. "
    "Preserve paragraph structure and line breaks from the original text. When encountering "
    "ambiguous phrases choose the interpretation that best fits the surrounding context. "
)
_LONG_COUNTER = (
    "You are a precise sequence assistant that continues numeric or patterned sequences "
    "exactly without skipping repeating or reordering any element. Output only the "
    "continuation with no surrounding words or explanation. Verify each element against "
    "the established pattern before emitting it. If the pattern is arithmetic maintain "
    "the common difference. If geometric maintain the common ratio. For alternating or "
    "composite patterns identify all sub-patterns and extend each one correctly. "
)
MULTI_TASK_PROMPTS_LONG = [
    {"task_id": "translator", "prompt": _LONG_TRANSLATOR * 30},
    {"task_id": "counter", "prompt": _LONG_COUNTER * 30},
]
MULTI_TASK_CASES_LONG = [
    ("translator", "The capital of France is", 8),
    ("counter", "Count the positive integers in order: 1, 2, 3,", 16),
]


class MultiTaskPromptPPTest(DPRequestRouting, PPModelTest):
    """PP multi-task system prompt: build resident KV at startup, reuse it per request.

    PP>1 exercises preRun's synchronous pipeline completion; the PP=1 baseline
    uses the same constructor with local execution. Both servers share the same
    multi_task_prompt config and the same task_id requests, so both prepend identical
    prefix tokens. Matching greedy output plus reuse_len>0 verifies the PP startup build
    produced correct, reusable resident KV on every stage.
    """

    case_name = "multi_task_prompt_pp2_tp2"

    def id(self):
        return f"{super().id()}[{self.case_name}]"

    def request_cases(self, block_size):
        """Switch A/B/A, then distinguish resident prefixes with identical user tokens."""
        cases = MULTI_TASK_CASES_LONG if block_size > 16 else MULTI_TASK_CASES
        requests_by_task = [cases[0], cases[1], cases[0]]
        # /** Long-prefix fixtures can emit a shared reasoning preamble before task-dependent tokens. */
        probe_tokens = 128 if block_size > 16 else 8
        requests_by_task.extend(
            (task_id, "Input: One, two, three.\nOutput:", probe_tokens)
            for task_id, _, _ in cases
        )
        return requests_by_task

    def shortDescription(self):
        return self.case_name

    def start_server(
        self, checkpoint, gpu_ids, pp, tp, dp, ep, role_name, cp=1, sp=0, block_size=16
    ):
        world_size = pp * tp * dp
        self.assertGreaterEqual(
            len(gpu_ids), world_size, f"need {world_size} GPUs, got {gpu_ids}"
        )
        output_dir = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "."))
        output_dir.mkdir(parents=True, exist_ok=True)
        prompts = MULTI_TASK_PROMPTS_LONG if block_size > 16 else MULTI_TASK_PROMPTS
        prompt_file = output_dir / f"{role_name}_multi_task_prompt.json"
        prompt_file.write_text(
            json.dumps(prompts, ensure_ascii=False), encoding="utf-8"
        )
        args = (
            base_smoke_args(default_seq_size_per_block=block_size)
            + [
                "--pp_size",
                str(pp),
                "--tp_size",
                str(tp),
                "--dp_size",
                str(dp),
                "--ep_size",
                str(ep),
                "--world_size",
                str(world_size),
                "--role_type",
                "PDFUSION",
                "--reuse_cache",
                "1",
                "--multi_task_prompt",
                str(prompt_file.resolve()),
            ]
            + speculative_args(checkpoint, sp)
        )
        if cp > 1:
            args += ["--prefill_cp_size", str(cp), "--cp_rotate_method", "ALL_GATHER"]
        server = MagaServerManager(
            port=reserve_server_port(self),
            env_args={
                "CUDA_VISIBLE_DEVICES": ",".join(gpu_ids[:world_size]),
                "WORLD_SIZE": str(world_size),
                "LOCAL_WORLD_SIZE": str(world_size),
                "RTP_LLM_STREAM_ASYNC": "0",
                "RTP_LLM_DEVICE_INPUT": "0",
            },
            role_name=role_name,
            smoke_args_str=shlex.join(args),
        )
        self.assertTrue(
            server.start_server(
                model_path=checkpoint,
                model_type=self.model_type,
                tokenizer_path=self.tokenizer_path,
            ),
            f"{role_name} failed to start: {server.log_file_path}",
        )
        return server

    def generate(self, port, prompt, max_new_tokens, label="", role_addrs=None, task_id=None):
        generate_config = {
            "is_streaming": False,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": max_new_tokens,
            "top_k": 1,
            "top_p": 1.0,
            "random_seed": 1234,
            "return_output_ids": True,
            "aux_info": True,
            "task_id": task_id,
        }
        if role_addrs is not None:
            generate_config["role_addrs"] = role_addrs
        response = requests.post(
            f"http://127.0.0.1:{port}/",
            json={"prompt": prompt, "generate_config": generate_config},
            timeout=REQUEST_TIMEOUT,
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["finished"], result)
        self.assertEqual(len(result["output_ids"][0]), max_new_tokens, result)
        return result

    def run_with_prompt(
        self, checkpoint, gpu_ids, pp, tp, dp, ep, role_name, cp=1, sp=0, block_size=16
    ):
        server = self.start_server(
            checkpoint,
            gpu_ids,
            pp,
            tp,
            dp,
            ep,
            role_name,
            cp=cp,
            sp=sp,
            block_size=block_size,
        )
        cases = self.request_cases(block_size)
        try:
            outputs = []
            for rank in range(dp):
                for task_id, prompt, tokens in cases:
                    if dp > 1:
                        result = self.generate_on_dp(
                            server.port, {"tp": tp, "dp": dp}, rank, tokens,
                            f"{role_name},dp={rank},task={task_id}", prompt, task_id,
                        )
                    else:
                        result = self.generate(server.port, prompt, tokens, task_id=task_id)
                    outputs.append(result)
            return outputs
        finally:
            server.stop_server()

    def run_pd_with_prompt(
        self,
        checkpoint,
        gpu_ids,
        pp,
        tp,
        role_name,
        cp=1,
        decode_tp=None,
        sp=0,
        block_size=16,
    ):
        """Start separate PREFILL and DECODE servers (both PP>1) with multi_task_prompt."""
        if decode_tp is None:
            decode_tp = tp
        prefill_ws = pp * tp
        decode_ws = pp * decode_tp
        total_gpus = prefill_ws + decode_ws
        self.assertGreaterEqual(len(gpu_ids), total_gpus, f"PD needs {total_gpus} GPUs")
        output_dir = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "."))
        output_dir.mkdir(parents=True, exist_ok=True)
        prompts = MULTI_TASK_PROMPTS_LONG if block_size > 16 else MULTI_TASK_PROMPTS
        prompt_file = output_dir / f"{role_name}_multi_task_prompt.json"
        prompt_file.write_text(
            json.dumps(prompts, ensure_ascii=False), encoding="utf-8"
        )
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
        common = (
            f"--pp_size {pp} --dp_size 1 --ep_size 1"
            " --cache_store_rdma_mode 0 --use_local 1 --load_cache_timeout_ms 120000"
            " --reuse_cache 1"
            f" --multi_task_prompt {prompt_file.resolve()}"
        )
        prefill_devices = ",".join(gpu_ids[:prefill_ws])
        decode_devices = ",".join(gpu_ids[prefill_ws : prefill_ws + decode_ws])
        cp_args = (
            ["--prefill_cp_size", str(cp), "--cp_rotate_method", "ALL_GATHER"]
            if cp > 1
            else []
        )
        prefill = MagaServerManager(
            env_args={
                "CUDA_VISIBLE_DEVICES": prefill_devices,
                "WORLD_SIZE": str(prefill_ws),
                "MODEL_SERVICE_CONFIG": service_config,
                "REMOTE_SERVER_PORT": str(decode_port),
                "REMOTE_RPC_SERVER_IP": "localhost",
                "RTP_LLM_STREAM_ASYNC": "0",
                "RTP_LLM_DEVICE_INPUT": "0",
            },
            port=prefill_port,
            role_name=f"{role_name}_prefill",
            smoke_args_str=shlex.join(
                base_smoke_args(default_seq_size_per_block=block_size)
                + shlex.split(common)
                + ["--tp_size", str(tp), "--world_size", str(prefill_ws)]
                + ["--role_type", "PREFILL"]
                + cp_args
                + speculative_args(checkpoint, sp)
            ),
        )
        decode = MagaServerManager(
            env_args={
                "CUDA_VISIBLE_DEVICES": decode_devices,
                "WORLD_SIZE": str(decode_ws),
                "MODEL_SERVICE_CONFIG": service_config,
                "REMOTE_SERVER_PORT": str(prefill_port),
                "REMOTE_RPC_SERVER_IP": "localhost",
                "RTP_LLM_STREAM_ASYNC": "0",
                "RTP_LLM_DEVICE_INPUT": "0",
            },
            port=decode_port,
            role_name=f"{role_name}_decode",
            smoke_args_str=shlex.join(
                base_smoke_args(default_seq_size_per_block=block_size)
                + shlex.split(common)
                + ["--tp_size", str(decode_tp), "--world_size", str(decode_ws)]
                + ["--role_type", "DECODE"]
                + (["--cp_rotate_method", "PREFILL_CP"] if cp > 1 else [])
                + speculative_args(checkpoint, sp)
            ),
        )
        try:
            self.assertTrue(
                decode.start_server(
                    model_path=checkpoint,
                    model_type=self.model_type,
                    tokenizer_path=self.tokenizer_path,
                ),
                f"PD decode failed to start: {decode.log_file_path}",
            )
            self.assertTrue(
                prefill.start_server(
                    model_path=checkpoint,
                    model_type=self.model_type,
                    tokenizer_path=self.tokenizer_path,
                ),
                f"PD prefill failed to start: {prefill.log_file_path}",
            )
            return [
                self.generate(prefill.port, prompt, tokens, task_id=task_id)
                for task_id, prompt, tokens in self.request_cases(block_size)
            ]
        finally:
            prefill.stop_server()
            decode.stop_server()

    def test_pp_multi_task_prompt_matches_pp1(self):
        checkpoint = self.checkpoint
        self.assertTrue(checkpoint, "Pass --test_env=CHECKPOINT_PATH=<checkpoint>")
        variant = VARIANTS[self.case_name]
        pp, tp, dp = variant["pp"], variant["tp"], variant.get("dp", 1)
        ep = variant.get("ep", 1)
        cp = variant.get("cp", 1)
        sp = variant.get("sp", 0)
        block_size = variant.get("block_size", 2048 if variant.get("sp") else 16)
        is_pd = variant.get("pd", False)
        cases = self.request_cases(block_size)
        gpu_ids = [str(x) for x in get_gpu_ids()]
        baseline = self.run_with_prompt(
            checkpoint,
            gpu_ids,
            1,
            tp,
            1,
            1,
            f"{self.case_name}_pp1_baseline",
            cp=1 if is_pd else cp,
            sp=sp,
            block_size=block_size,
        )
        self.assertNotEqual(
            baseline[-2]["output_ids"], baseline[-1]["output_ids"],
            "identical task outputs cannot detect a switched resident prefix",
        )
        if is_pd:
            actual = self.run_pd_with_prompt(
                checkpoint,
                gpu_ids,
                pp,
                tp,
                self.case_name,
                cp=cp,
                decode_tp=variant.get("decode_tp"),
                sp=sp,
                block_size=block_size,
            )
        else:
            actual = self.run_with_prompt(
                checkpoint,
                gpu_ids,
                pp,
                tp,
                dp,
                ep,
                self.case_name,
                cp=cp,
                sp=sp,
                block_size=block_size,
            )
        report = {
            "case": self.case_name,
            "model_type": self.model_type,
            "checkpoint": checkpoint,
            "topology": variant,
            "baseline": baseline,
            "actual": actual,
            "passed": False,
        }
        try:
            self.assertEqual(len(baseline), len(cases))
            self.assertEqual(len(actual), dp * len(cases))
            seen_tasks = set()
            for index, got in enumerate(actual):
                task_id, prompt, _ = cases[index % len(cases)]
                base = baseline[index % len(cases)]
                task_key = (index // len(cases), task_id)
                got_reuse = got["aux_info"]["reuse_len"]
                base_reuse = base["aux_info"]["reuse_len"]
                self.assertGreaterEqual(
                    got_reuse,
                    block_size,
                    f"task {task_id!r}: reuse_len={got_reuse} < one full block ({block_size}); "
                    f"resident prefix was not reused",
                )
                if task_key not in seen_tasks:
                    self.assertEqual(
                        got_reuse,
                        base_reuse,
                        f"task {task_id!r}: PP={pp} reuse_len={got_reuse} != "
                        f"PP=1 baseline reuse_len={base_reuse}",
                    )
                    seen_tasks.add(task_key)
                self.assertEqual(
                    got["output_ids"],
                    base["output_ids"],
                    f"PP={pp} multi_task_prompt diverges from PP=1 baseline on task "
                    f"{task_id!r} ({prompt[:40]!r})",
                )
            report["passed"] = True
        finally:
            output_dir = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "."))
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{self.case_name}.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name in selected_cases():
        if name == "pdfusion_mtp_regression":
            case = MtpPPTest("test_pdfusion_mtp_matches_target_generation")
        elif name in (
            "multi_task_prompt_pp2_tp2",
            "multi_task_prompt_pp2_dp2",
            "multi_task_prompt_pp2_pd",
            "multi_task_prompt_pp2_cp2",
            "multi_task_prompt_pp2_mtp",
            "multi_task_prompt_pp2_pd_mtp",
        ):
            case = MultiTaskPromptPPTest("test_pp_multi_task_prompt_matches_pp1")
            case.case_name = name
        else:
            case = PPTopologyTest("test_selected_topology")
            case.case_name = name
        suite.addTest(case)
    return suite


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases", help="Comma-separated PP cases; overrides PP_TEST_CASES / PD_VARIANT"
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
