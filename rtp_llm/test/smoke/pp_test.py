"""PP topology smoke cases, selected with --cases or PP_TEST_CASES.

--list-cases lists the available PD and PDFUSION cases without starting a server.
PD_VARIANT remains an alias for selecting one of the original PD cases.
pdfusion_mtp_regression retains the MTP step/cache-reuse/concurrency matrix.
Its defaults remain qwen35_dense and block size 2048; PD defaults remain
qwen_3 and block size 16.
MODEL_TYPE, CHECKPOINT_PATH and TOKENIZER_PATH select the target model;
SP_MODEL_TYPE, SP_CHECKPOINT_PATH and SP_TYPE select the optional draft model.
PD_MODEL_TYPE / PD_SP_MODEL_TYPE remain supported as fallbacks.
For Qwen3.5 dense, use MODEL_TYPE=qwen35_dense and PP_SEQ_SIZE_PER_BLOCK=2048.
DP cases configure EP=TP*DP; expert communication is exercised only by MoE models.
"""

import argparse
import json
import logging
import os
import re
import shlex
import sys
import time
import unittest
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from rtp_llm.config.py_config_modules import MIN_WORKER_INFO_PORT_NUM, ServerConfig
from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager

# Overridable for MLA checkpoints (deepseek2); CP mode A and MLA mode B need one.
MODEL_TYPE = os.environ.get("MODEL_TYPE", os.environ.get("PD_MODEL_TYPE", "qwen_3"))
# Draft model registry name for the MTP variants, overridable per checkpoint family.
SP_MODEL_TYPE = os.environ.get(
    "SP_MODEL_TYPE", os.environ.get("PD_SP_MODEL_TYPE", "qwen35_dense_mtp")
)
REQUEST_TIMEOUT = int(os.environ.get("PP_REQUEST_TIMEOUT", "300"))

# Variant table: each side's (pp, tp) width. decode_gpus = decode_pp*decode_tp.
#   sym:              prefill pp2tp1 / decode pp2tp1 - symmetric PP stage routing
#   sym_mtp1..4: sym layout + MTP, proposal width 1..4. P hands off s0+d1
#                     only; width>2 pads on D, so sp>=2 exercises that path.
#   asym:             prefill pp2tp1 / decode pp2tp2 - decode TP finer, sub-slice read
#   conv:             prefill pp2tp2 / decode pp2tp1 - prefill TP finer, peer assembly
#   pp2_tp2:          prefill pp2tp2 / decode pp2tp2 - TP>1 both sides, no CP (control)
#   pp1_tp2:          prefill pp1tp2 / decode pp1tp2 - pp=1 flat path with TP>1
#   pp1_tp1:          prefill pp1tp1 / decode pp1tp1 - flat single-worker direct path
#   pp1_asym:         prefill pp1tp1 / decode pp1tp2 - flat, decode TP finer
#   pp1_conv:         prefill pp1tp2 / decode pp1tp1 - flat, prefill TP finer
#   cp_sharded:       prefill pp2tp2(cp=2,sharded) / decode pp2tp1 - CP mode A
#   cp_full:          prefill pp2tp2(cp=2,full) / decode pp2tp1 - CP mode B
#   cp_full_decode_tp2: prefill pp2tp2(cp=2,full) / decode pp2tp2 - CP mode B + decode TP>1
#   pp1_cp_full:      prefill pp1tp2(cp=2,full) / decode pp1tp1 - flat path + CP mode B
#   pp1_cp_full_tp2:  prefill pp1tp2(cp=2,full) / decode pp1tp2 - flat + CP mode B + decode TP>1
#   pp1_mla_cp_import: prefill pp1tp2 / decode pp1tp2 - MLA only: decode-side PREFILL_CP flag
#   pp2_pp1:          prefill pp2tp1 / decode pp1tp1 - decode rank pulls across both stage groups
#   pp1_pp2:          prefill pp1tp1 / decode pp2tp1 - each decode stage pulls a partial range
#   pp2_pp1_tp2:      prefill pp2tp2 / decode pp1tp2 - partial ranges with TP>1 on both sides
#
# Note: CP mode A (sharded) and MLA mode B need an MLA checkpoint: run with
# PD_MODEL_TYPE=deepseek2 and CHECKPOINT_PATH=<DeepSeek-V2 dir>. Classic MLA has
# no rotating prefill attention impl ("can not find mla type"), so MLA mode B is
# covered by pp1_mla_cp_import: decode declares PREFILL_CP while the prefill
# stays plain - whole-block MLA KV makes the two layouts byte-identical.
VARIANTS = {
    "sym": {"prefill_pp": 2, "prefill_tp": 1, "decode_pp": 2, "decode_tp": 1},
    "sym_mtp1": {
        "prefill_pp": 2,
        "prefill_tp": 1,
        "decode_pp": 2,
        "decode_tp": 1,
        "sp": 1,
    },
    "sym_mtp2": {
        "prefill_pp": 2,
        "prefill_tp": 1,
        "decode_pp": 2,
        "decode_tp": 1,
        "sp": 2,
    },
    "sym_mtp3": {
        "prefill_pp": 2,
        "prefill_tp": 1,
        "decode_pp": 2,
        "decode_tp": 1,
        "sp": 3,
    },
    "sym_mtp4": {
        "prefill_pp": 2,
        "prefill_tp": 1,
        "decode_pp": 2,
        "decode_tp": 1,
        "sp": 4,
    },
    "asym": {"prefill_pp": 2, "prefill_tp": 1, "decode_pp": 2, "decode_tp": 2},
    "conv": {"prefill_pp": 2, "prefill_tp": 2, "decode_pp": 2, "decode_tp": 1},
    "pp2_tp2": {"prefill_pp": 2, "prefill_tp": 2, "decode_pp": 2, "decode_tp": 2},
    "pp1_tp2": {"prefill_pp": 1, "prefill_tp": 2, "decode_pp": 1, "decode_tp": 2},
    "pp1_tp1": {"prefill_pp": 1, "prefill_tp": 1, "decode_pp": 1, "decode_tp": 1},
    "pp1_asym": {"prefill_pp": 1, "prefill_tp": 1, "decode_pp": 1, "decode_tp": 2},
    "pp1_conv": {"prefill_pp": 1, "prefill_tp": 2, "decode_pp": 1, "decode_tp": 1},
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
    "pp1_mla_cp_import": {
        "prefill_pp": 1,
        "prefill_tp": 2,
        "decode_pp": 1,
        "decode_tp": 2,
        "decode_prefill_cp": True,
    },
    "pp2_pp1": {"prefill_pp": 2, "prefill_tp": 1, "decode_pp": 1, "decode_tp": 1},
    "pp1_pp2": {"prefill_pp": 1, "prefill_tp": 1, "decode_pp": 2, "decode_tp": 1},
    "pp2_pp1_tp2": {"prefill_pp": 2, "prefill_tp": 2, "decode_pp": 1, "decode_tp": 2},
}

# A PDFUSION case uses one server. PD cases above retain their two-server layout.
VARIANTS.update(
    {
        "pdfusion_mtp_regression": {"pp": 2, "tp": 2, "dp": 1, "ep": 1},
        "pdfusion_pp2_tp2": {"pp": 2, "tp": 2, "dp": 1, "ep": 1, "sp": 0},
        "pdfusion_pp2_tp2_mtp": {"pp": 2, "tp": 2, "dp": 1, "ep": 1, "sp": 3},
        "fake_pp2_tp2_dp2": {"pp": 2, "tp": 2, "dp": 2, "ep": 4, "sp": 0},
        "fake_pp2_tp2_dp2_mtp": {"pp": 2, "tp": 2, "dp": 2, "ep": 4, "sp": 3},
        "multi_task_prompt_pp2": {"pp": 2, "tp": 1, "dp": 1, "ep": 1, "sp": 0},
        "multi_task_prompt_pp2_tp2": {"pp": 2, "tp": 2, "dp": 1, "ep": 1, "sp": 0},
        "multi_task_prompt_pp2_dp2": {"pp": 2, "tp": 1, "dp": 2, "ep": 2, "sp": 0},
    }
)


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


class FakeBatchProgress:
    """Read PP completions and actual request destinations from the console log."""

    def __init__(self, path):
        self.path = path
        self.offset = 0
        self.counts = Counter()
        self.requests = []

    def poll(self):
        with open(self.path, "rb") as log:
            log.seek(self.offset)
            # Bound each poll even while debug output is still being appended.
            end = os.fstat(log.fileno()).st_size
            while self.offset < end:
                line = log.readline()
                if not line.endswith(b"\n"):
                    break
                self.offset = log.tell()
                match = re.search(rb"PP fake batch completed: dp_rank=(\d+)", line)
                if match:
                    self.counts[int(match[1])] += 1
                match = re.search(
                    rb"\[RANK (\d+)\].*\[rtp_llm/cpp/model_rpc/LocalRpcServer.cc:\d+\]"
                    rb".*receive request (\d+)",
                    line,
                )
                if match:
                    self.requests.append(
                        {"world_rank": int(match[1]), "request_id": int(match[2])}
                    )
        return self.counts.copy()


class PPTopologyTest(unittest.TestCase):
    case_name = "sym"

    def id(self):
        return f"{super().id()}[{self.case_name}]"

    def shortDescription(self):
        return self.case_name

    def generate(self, port, prompt, max_new_tokens, label="", role_addrs=None):
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

    def run_baseline(self, checkpoint, devices, tp_size):
        # Same TP as the PD decode side so TP numerics cancel out; the
        # comparison then isolates the PP + PD transfer difference only.
        args = (
            base_smoke_args()
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
                    model_type=MODEL_TYPE,
                    tokenizer_path=os.environ.get("TOKENIZER_PATH", checkpoint),
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
    ):
        prefill_port = MagaServerManager.get_free_port()
        decode_port = MagaServerManager.get_free_port()
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
        CP is asymmetric: prefill runs the rotation (ALL_GATHER); decode
        declares the peer as CP (PREFILL_CP) and mirrors the CP shape config
        (prefill_cp_size / kv_cache_sharded) from its own settings.
        """
        # Both sides need the draft model and matching draft precision.
        sp_args = speculative_args(checkpoint, sp)
        prefill_args = f"--pp_size {prefill_pp} --tp_size {prefill_tp} --world_size {prefill_ws} {common_pd}"
        if prefill_cp > 1:
            prefill_args += (
                f" --cp_rotate_method ALL_GATHER --prefill_cp_size {prefill_cp}"
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
            role_name=f"{self.case_name}_prefill_pp{prefill_pp}",
            smoke_args_str=shlex.join(
                base_smoke_args()
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
            role_name=f"{self.case_name}_decode_pp{decode_pp}",
            smoke_args_str=shlex.join(
                base_smoke_args()
                + shlex.split(decode_args)
                + sp_args
                + ["--role_type", "DECODE"]
            ),
        )
        try:
            self.assertTrue(
                decode.start_server(
                    model_path=checkpoint,
                    model_type=MODEL_TYPE,
                    tokenizer_path=os.environ.get("TOKENIZER_PATH", checkpoint),
                ),
                f"decode failed to start: {decode.log_file_path}",
            )
            self.assertTrue(
                prefill.start_server(
                    model_path=checkpoint,
                    model_type=MODEL_TYPE,
                    tokenizer_path=os.environ.get("TOKENIZER_PATH", checkpoint),
                ),
                f"prefill failed to start: {prefill.log_file_path}",
            )
            # PD entrance is the prefill instance (decode_entrance=false).
            return [
                self.generate(prefill.port, prompt, tokens)
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

    def wait_fake_progress(self, progress, before, ranks, minimum):
        deadline = time.monotonic() + 30
        while True:
            counts = progress.poll()
            if all(counts[rank] - before[rank] >= minimum for rank in ranks):
                self.fake_progress.append(dict(counts))
                return
            if time.monotonic() >= deadline:
                self.fail(
                    f"PP fake round trips stopped: ranks={ranks}, before={dict(before)}, "
                    f"after={dict(counts)}, expected >= {minimum} new completions; "
                    f"log={progress.path}"
                )
            time.sleep(0.1)

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

    def generate_on_dp(self, port, progress, variant, busy_dp, tokens, label):
        # A frontend knows every DP backend; its HTTP port does not pin routing.
        backend = ServerConfig()
        backend.start_port = port
        backend.set_local_rank(busy_dp * variant["tp"])
        role_addr = {
            "role": "PDFUSION",
            "ip": "127.0.0.1",
            "http_port": backend.http_port,
            "grpc_port": backend.rpc_server_port,
        }
        before = progress.poll()
        request_index = len(progress.requests)
        during_request = before
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(
                self.generate,
                port,
                "Continue counting the positive integers: 1, 2, 3,",
                tokens,
                label,
                [role_addr],
            )
            while not pending.done():
                counts = progress.poll()
                if not pending.done():
                    during_request = counts
                time.sleep(0.1)
            pending.result()

        progress.poll()
        received = progress.requests[request_index:]
        self.assertEqual(
            [request["world_rank"] for request in received],
            [busy_dp * variant["tp"]],
            f"{label}: request did not reach the selected DP; received={received}",
        )
        idle_progress = {
            rank: during_request[rank] - before[rank]
            for rank in range(variant["dp"])
            if rank != busy_dp
        }
        self.outputs[-1].update(
            backend_request=received[0], idle_fake_completions=idle_progress
        )
        for rank, completed in idle_progress.items():
            self.assertGreaterEqual(
                completed,
                variant["pp"] + 1,
                f"{label}: idle DP {rank} did not complete enough fake batches "
                f"while the real request was pending: {idle_progress}",
            )

    def run_pdfusion(self, checkpoint, gpu_ids, variant):
        pp, tp, dp = (variant[key] for key in ("pp", "tp", "dp"))
        world_size = pp * tp * dp
        self.assertGreaterEqual(len(gpu_ids), world_size)
        args = (
            base_smoke_args()
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
        if dp > 1:
            # The completion marker is emitted after receiving the fake result.
            # Console output keeps all ranks' evidence in this server's process.log.
            env.update(FT_SERVER_TEST="1", LOG_LEVEL="DEBUG")
            # initLogger reloads alog.conf after LOG_LEVEL was applied. Set the
            # console level in that config as well so completion markers survive.
            base_config = Path(__file__).resolve().parents[2] / "config/alog.conf"
            output_dir = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "."))
            output_dir.mkdir(parents=True, exist_ok=True)
            log_config = output_dir / f"{self.case_name}.alog.conf"
            log_config.write_text(
                re.sub(
                    r"(?m)^alog.logger.console=[^,\n]+",
                    "alog.logger.console=DEBUG",
                    base_config.read_text(),
                )
            )
            args.extend(["--ft_alog_conf_path", str(log_config.resolve())])
        server = MagaServerManager(
            env_args=env,
            role_name=self.case_name,
            smoke_args_str=shlex.join(args),
        )
        try:
            self.assertTrue(
                server.start_server(
                    model_path=checkpoint,
                    model_type=MODEL_TYPE,
                    tokenizer_path=os.environ.get("TOKENIZER_PATH", checkpoint),
                ),
                f"{self.case_name} failed to start: {server.log_file_path}",
            )
            if dp == 1:
                return [
                    self.generate(server.port, prompt, tokens, self.case_name)
                    for prompt, tokens in self.cases()
                ]

            progress = FakeBatchProgress(server.log_file_path)
            ports = [
                server.port + rank * tp * MIN_WORKER_INFO_PORT_NUM for rank in range(dp)
            ]
            self.wait_frontends(ports)
            self.wait_fake_progress(progress, Counter(), list(range(dp)), pp + 1)
            # Exercise both busy/idle directions, then return to each DP after idle.
            for cycle in range(2):
                for busy_dp in range(dp):
                    for tokens in (64, 128):
                        self.generate_on_dp(
                            server.port,
                            progress,
                            variant,
                            busy_dp,
                            tokens,
                            f"cycle={cycle},busy_dp={busy_dp},tokens={tokens}",
                        )
                    # Once the real request drains, every DP must keep completing
                    # fake batches. The next request then checks recovery from idle.
                    before = progress.poll()
                    self.wait_fake_progress(progress, before, list(range(dp)), pp + 1)
        finally:
            server.stop_server()

    def check_pd_variant(self, checkpoint, gpu_ids, variant):
        decode_tp = variant["decode_tp"]
        prefill_tp = variant["prefill_tp"]
        prefill_pp = variant["prefill_pp"]
        decode_pp = variant["decode_pp"]
        prefill_cp = variant.get("prefill_cp", 1)
        kv_cache_sharded = variant.get("kv_cache_sharded", False)
        decode_prefill_cp = variant.get("decode_prefill_cp", False)
        sp = variant.get("sp", 0)
        if decode_prefill_cp:
            # MLA-only: with MHA the slice plan assumes a rotating prefill.
            self.assertNotEqual(
                MODEL_TYPE,
                "qwen_3",
                f"case {self.case_name} needs an MLA checkpoint (MODEL_TYPE=deepseek2)",
            )
        prefill_gpus = prefill_pp * prefill_tp
        decode_gpus = decode_pp * decode_tp
        # Baseline and prefill run strictly sequentially and share the first
        # max(decode_tp, prefill_gpus) GPUs; decode gets the rest.
        shared = max(decode_tp, prefill_gpus)
        need = shared + decode_gpus
        self.assertGreaterEqual(len(gpu_ids), need, f"need {need} GPUs, got {gpu_ids}")
        baseline = self.run_baseline(
            checkpoint, ",".join(gpu_ids[:decode_tp]), decode_tp
        )
        actual = self.run_pd(
            checkpoint,
            prefill_devices=",".join(gpu_ids[:prefill_gpus]),
            decode_devices=",".join(gpu_ids[shared : shared + decode_gpus]),
            prefill_pp=prefill_pp,
            prefill_tp=prefill_tp,
            decode_pp=decode_pp,
            decode_tp=decode_tp,
            prefill_cp=prefill_cp,
            kv_cache_sharded=kv_cache_sharded,
            decode_prefill_cp=decode_prefill_cp,
            sp=sp,
        )
        for (prompt, _), base, got in zip(self.cases(), baseline, actual):
            if (prefill_tp != 1 and prefill_tp != decode_tp) or prefill_cp > 1:
                # Prefill TP or CP differs from the baseline (CP changes the
                # attention reduction order), so numerics can flip a greedy
                # token mid-sequence; a routing error instead garbles token 1.
                # Require the first token to match, report the overlap.
                self.assertEqual(
                    got["output_ids"][0][:1],
                    base["output_ids"][0][:1],
                    f"first token diverges on: {prompt[:40]}",
                )
                same = sum(
                    1
                    for a, b in zip(got["output_ids"][0], base["output_ids"][0])
                    if a == b
                )
                total = len(base["output_ids"][0])
                print(f"[overlap] {prompt[:30]!r}: {same}/{total} tokens equal")
            else:
                self.assertEqual(
                    got["output_ids"],
                    base["output_ids"],
                    f"PD diverges from PDFUSION baseline on: {prompt[:40]}",
                )
        if sp:
            # Speculative decoding must actually engage: some multi-token case
            # has to accept at least one draft (iter_count < emitted tokens).
            self.assertTrue(
                any(
                    got["aux_info"]["iter_count"] < len(got["output_ids"][0])
                    for (prompt, tokens), got in zip(self.cases(), actual)
                    if tokens > 1
                ),
                f"MTP sp={sp} accepted no draft token on any multi-token case",
            )

    def test_selected_topology(self):
        checkpoint = os.environ.get("CHECKPOINT_PATH")
        self.assertTrue(checkpoint, "Pass --test_env=CHECKPOINT_PATH=<checkpoint>")
        variant = VARIANTS[self.case_name]
        gpu_ids = [str(x) for x in get_gpu_ids()]
        self.outputs = []
        self.fake_progress = []
        report = {
            "case": self.case_name,
            "model_type": MODEL_TYPE,
            "checkpoint": checkpoint,
            "sp_type": (
                os.environ.get("SP_TYPE", "mtp") if variant.get("sp") else "none"
            ),
            "sp_model_type": SP_MODEL_TYPE if variant.get("sp") else None,
            "topology": variant,
            "outputs": self.outputs,
            "fake_completions": self.fake_progress,
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
                    checkpoint, ",".join(gpu_ids[: variant["tp"]]), variant["tp"]
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


class MtpPPTest(unittest.TestCase):
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
                    model_type=os.environ.get(
                        "MODEL_TYPE", os.environ.get("PD_MODEL_TYPE", "qwen35_dense")
                    ),
                    tokenizer_path=os.environ.get("TOKENIZER_PATH", checkpoint),
                ),
                f"{variant} failed to start: {server.log_file_path}",
            )
            outputs["serial"] = []
            for case in cases:
                outputs["serial"].append(self.generate(server, *case))
            if reuse_cache:
                for result in outputs["serial"][3:6]:
                    self.assertGreater(result["aux_info"]["reuse_len"], 0, result)
            else:
                for result in outputs["serial"]:
                    self.assertEqual(result["aux_info"]["reuse_len"], 0, result)
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(self.generate, server, *case) for case in cases[1:]
                ]
                outputs["concurrent"] = [future.result() for future in futures]
        finally:
            server.stop_server()
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{variant}.json").write_text(
                json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return outputs

    def test_pdfusion_mtp_matches_target_generation(self):
        checkpoint = os.environ.get("CHECKPOINT_PATH")
        self.assertTrue(
            checkpoint, "Pass --test_env=CHECKPOINT_PATH=<Qwen3.5-27B checkpoint>"
        )
        baseline = self.run_variant(checkpoint, 0)
        variants = ((step, reuse) for step in (1, 3, 4) for reuse in (False, True))
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
                self.assertTrue(
                    any(
                        result["aux_info"]["iter_count"] < len(result["output_ids"][0])
                        for result in actual["serial"][1:]
                    ),
                    f"MTP {propose_step} did not accept any draft tokens",
                )


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


class MultiTaskPromptPPTest(unittest.TestCase):
    """PP multi-task system prompt: build resident KV at startup, reuse it per request.

    PP>1 exercises the direct pipeline build (buildSystemPromptsDirect); the PP=1
    baseline exercises the proven non-PP preRun build. Both servers share the same
    multi_task_prompt config and the same task_id requests, so both prepend identical
    prefix tokens. Matching greedy output plus reuse_len>0 verifies the PP startup build
    produced correct, reusable resident KV on every stage.
    """

    case_name = "multi_task_prompt_pp2"

    def id(self):
        return f"{super().id()}[{self.case_name}]"

    def shortDescription(self):
        return self.case_name

    def start_server(self, checkpoint, gpu_ids, pp, tp, dp, ep, role_name):
        world_size = pp * tp * dp
        self.assertGreaterEqual(
            len(gpu_ids), world_size, f"need {world_size} GPUs, got {gpu_ids}"
        )
        output_dir = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "."))
        output_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = output_dir / f"{role_name}_multi_task_prompt.json"
        prompt_file.write_text(
            json.dumps(MULTI_TASK_PROMPTS, ensure_ascii=False), encoding="utf-8"
        )
        args = (
            base_smoke_args()
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
            + speculative_args(checkpoint, 0)
        )
        server = MagaServerManager(
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
                model_type=MODEL_TYPE,
                tokenizer_path=os.environ.get("TOKENIZER_PATH", checkpoint),
            ),
            f"{role_name} failed to start: {server.log_file_path}",
        )
        return server

    def generate_with_task(self, server, task_id, prompt, max_new_tokens):
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
        response = requests.post(
            f"http://127.0.0.1:{server.port}/",
            json={"prompt": prompt, "generate_config": generate_config},
            timeout=REQUEST_TIMEOUT,
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["finished"], result)
        self.assertEqual(len(result["output_ids"][0]), max_new_tokens, result)
        return result

    def run_with_prompt(self, checkpoint, gpu_ids, pp, tp, dp, ep, role_name):
        server = self.start_server(checkpoint, gpu_ids, pp, tp, dp, ep, role_name)
        try:
            return [
                self.generate_with_task(server, task_id, prompt, tokens)
                for task_id, prompt, tokens in MULTI_TASK_CASES
            ]
        finally:
            server.stop_server()

    def test_pp_multi_task_prompt_matches_pp1(self):
        checkpoint = os.environ.get("CHECKPOINT_PATH")
        self.assertTrue(checkpoint, "Pass --test_env=CHECKPOINT_PATH=<checkpoint>")
        variant = VARIANTS[self.case_name]
        pp, tp, dp = variant["pp"], variant["tp"], variant.get("dp", 1)
        ep = variant.get("ep", 1)
        gpu_ids = [str(x) for x in get_gpu_ids()]
        baseline = self.run_with_prompt(
            checkpoint, gpu_ids, 1, tp, dp, ep, f"{self.case_name}_pp1_baseline"
        )
        actual = self.run_with_prompt(
            checkpoint, gpu_ids, pp, tp, dp, ep, self.case_name
        )
        report = {
            "case": self.case_name,
            "model_type": MODEL_TYPE,
            "checkpoint": checkpoint,
            "topology": variant,
            "baseline": baseline,
            "actual": actual,
            "passed": False,
        }
        try:
            for (task_id, prompt, _), base, got in zip(
                MULTI_TASK_CASES, baseline, actual
            ):
                got_reuse = got["aux_info"]["reuse_len"]
                base_reuse = base["aux_info"]["reuse_len"]
                self.assertGreaterEqual(
                    got_reuse,
                    16,
                    f"task {task_id!r}: reuse_len={got_reuse} < one full block (16); "
                    f"resident prefix was not reused",
                )
                self.assertEqual(
                    got_reuse,
                    base_reuse,
                    f"task {task_id!r}: PP={pp} reuse_len={got_reuse} != "
                    f"PP=1 baseline reuse_len={base_reuse}",
                )
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
            "multi_task_prompt_pp2",
            "multi_task_prompt_pp2_tp2",
            "multi_task_prompt_pp2_dp2",
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
