import os
import shlex
import unittest

import requests

from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager

MODEL_TYPE = "qwen_3"

# Variant table: each side's (pp, tp) width. decode_gpus = decode_pp*decode_tp.
#   sym:     prefill pp2tp1 / decode pp2tp1 - symmetric PP stage routing
#   asym:    prefill pp2tp1 / decode pp2tp2 - decode TP finer, sub-slice read
#   conv:    prefill pp2tp2 / decode pp2tp1 - prefill TP finer, peer assembly
#   pp1_tp2: prefill pp1tp2 / decode pp1tp2 - pp=1 flat path with TP>1
VARIANTS = {
    "sym": {"prefill_pp": 2, "prefill_tp": 1, "decode_pp": 2, "decode_tp": 1},
    "asym": {"prefill_pp": 2, "prefill_tp": 1, "decode_pp": 2, "decode_tp": 2},
    "conv": {"prefill_pp": 2, "prefill_tp": 2, "decode_pp": 2, "decode_tp": 1},
    "pp1_tp2": {"prefill_pp": 1, "prefill_tp": 2, "decode_pp": 1, "decode_tp": 2},
}


def base_smoke_args():
    # SMOKE_ARGS carries the BUILD-level GPU reservation; strip its parallelism
    # flags so each server sets its own.
    args = shlex.split(os.environ["SMOKE_ARGS"])
    stripped = []
    skip_next = False
    for token in args:
        if skip_next:
            skip_next = False
            continue
        if token in ("--world_size", "--tp_size", "--dp_size", "--pp_size"):
            skip_next = True
            continue
        stripped.append(token)
    return stripped


class PdPPTest(unittest.TestCase):
    def generate(self, port, prompt, max_new_tokens):
        response = requests.post(
            f"http://127.0.0.1:{port}/",
            json={
                "prompt": prompt,
                "generate_config": {
                    "is_streaming": False,
                    "max_new_tokens": max_new_tokens,
                    "min_new_tokens": max_new_tokens,
                    "top_k": 1,
                    "top_p": 1.0,
                    "random_seed": 1234,
                    "return_output_ids": True,
                    "aux_info": True,
                },
            },
            timeout=300,
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertTrue(result["finished"], result)
        self.assertEqual(len(result["output_ids"][0]), max_new_tokens, result)
        return result

    def run_baseline(self, checkpoint, devices, tp_size):
        # Same TP as the PD decode side so TP numerics cancel out; the
        # comparison then isolates the PP + PD transfer difference only.
        args = base_smoke_args() + [
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
        ]
        server = MagaServerManager(
            env_args={
                "CUDA_VISIBLE_DEVICES": devices,
                "WORLD_SIZE": str(tp_size),
                "RTP_LLM_STREAM_ASYNC": "0",
            },
            role_name="pdfusion_pp1",
            smoke_args_str=shlex.join(args),
        )
        try:
            self.assertTrue(
                server.start_server(
                    model_path=checkpoint,
                    model_type=MODEL_TYPE,
                    tokenizer_path=checkpoint,
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
        prefill_args = f"--pp_size {prefill_pp} --tp_size {prefill_tp} --world_size {prefill_ws} {common_pd}"
        decode_args = f"--pp_size {decode_pp} --tp_size {decode_tp} --world_size {decode_ws} {common_pd}"
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
            role_name=f"prefill_pp{prefill_pp}",
            smoke_args_str=shlex.join(
                base_smoke_args()
                + shlex.split(prefill_args)
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
            role_name=f"decode_pp{decode_pp}",
            smoke_args_str=shlex.join(
                base_smoke_args() + shlex.split(decode_args) + ["--role_type", "DECODE"]
            ),
        )
        try:
            self.assertTrue(
                decode.start_server(
                    model_path=checkpoint,
                    model_type=MODEL_TYPE,
                    tokenizer_path=checkpoint,
                ),
                f"decode failed to start: {decode.log_file_path}",
            )
            self.assertTrue(
                prefill.start_server(
                    model_path=checkpoint,
                    model_type=MODEL_TYPE,
                    tokenizer_path=checkpoint,
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

    def test_pd_matches_pdfusion_baseline(self):
        checkpoint = os.environ.get("CHECKPOINT_PATH")
        self.assertTrue(
            checkpoint, "Pass --test_env=CHECKPOINT_PATH=<Qwen3 checkpoint>"
        )
        pd_variant = os.environ.get("PD_VARIANT", "sym")
        variant = VARIANTS[pd_variant]
        gpu_ids = [str(x) for x in get_gpu_ids()]
        decode_tp = variant["decode_tp"]
        prefill_tp = variant["prefill_tp"]
        prefill_pp = variant["prefill_pp"]
        decode_pp = variant["decode_pp"]
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
        )
        for (prompt, _), base, got in zip(self.cases(), baseline, actual):
            if prefill_tp != 1 and prefill_tp != decode_tp:
                # Prefill TP differs from the baseline, so numerics can flip a
                # greedy token mid-sequence; a routing error instead garbles
                # token 1. Require the first token to match, report the overlap.
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


if __name__ == "__main__":
    unittest.main()
