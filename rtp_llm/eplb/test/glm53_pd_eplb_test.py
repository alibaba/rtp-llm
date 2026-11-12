import concurrent.futures
import json
import os
import re
import time
import unittest
from pathlib import Path

import numpy as np
import requests
from transformers import PreTrainedTokenizerFast

from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
from rtp_llm.test.smoke.eplb_smoke_checks import check_eplb_applied
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


def role_environments(mode, devices, prefill_port, decode_port):
    common = {
        "CONCURRENCY_LIMIT": "64",
        "LOAD_CACHE_TIMEOUT_MS": "900000",
        "MAX_SEQ_LEN": "8192",
        "MAX_BATCH_TOKENS_SIZE": "8192",
        "FRONTEND_SERVER_COUNT": "1",
        "ACT_TYPE": "BF16",
        "LOAD_METHOD": "scratch",
        "QUANTIZATION": "FP8_PER_BLOCK_NO_MOE",
        "MOE_STRATEGY": "mega_moe_se",
        "ENABLE_FUSE_KERNELS": "1",
        "FP8_KV_CACHE": "1",
        "KERNEL_SEQ_SIZE_PER_BLOCK": "64",
        "SEQ_SIZE_PER_BLOCK": "64",
        "PREFILL_CP_KV_CACHE_SHARDED": "1",
        "ENABLE_GLM52_SHARED_INDEXER_KV_CACHE": "1",
        "USE_DEEPEP_MOE": "0",
        "USE_DEEPEP_LOW_LATENCY": "0",
        "USE_DEEPEP_INTERNODE": "0",
        "WARM_UP": "0",
        "THINK_MODE": "1",
        "THINK_START_TAG": "<think>",
        "THINK_END_TAG": "</think>",
        "RTP_LLM_MAX_TOKENS_EXCLUDE_THINKING": "0",
        "RTP_LLM_ENABLE_MTP_INDEXER_SHARE": "1",
        "OMP_NUM_THREADS": "8",
        "DG_JIT_CPP_STANDARD": "20",
        "CC": "/opt/rh/gcc-toolset-13/root/usr/bin/gcc",
        "CXX": "/opt/rh/gcc-toolset-13/root/usr/bin/g++",
        "CUDAHOSTCXX": "/opt/rh/gcc-toolset-13/root/usr/bin/g++",
        "NVCC_PREPEND_FLAGS": "-ccbin=/opt/rh/gcc-toolset-13/root/usr/bin/g++",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,large_segment_size_mb:1024",
        "KV_CACHE_MEM_MB": "4096",
        "REDUNDANT_EXPERT": "0",
        "EPLB_MODE": mode,
        "RTP_LLM_GLM5_MEGA_GATE": "1",
        "RTP_LLM_GLM5_GATE_PACK": "1",
        "EPLB_METRICS_SNAPSHOT_MAX_FILES": "64",
        "REMOTE_RPC_SERVER_IP": "localhost",
        "CACHE_STORE_RDMA_MODE": "1",
        "DECODE_ENTRANCE": "0",
        "FRONTEND_PRE_STOP_DRAIN_SECONDS": "0",
        "SHUTDOWN_TIMEOUT": "5",
    }
    prefill = dict(
        common,
        **{
            "ROLE_TYPE": "PREFILL",
            "TP_SIZE": "4",
            "DP_SIZE": "1",
            "EP_SIZE": "4",
            "WORLD_SIZE": "4",
            "CUDA_VISIBLE_DEVICES": ",".join(devices[:4]),
            "CP_ROTATE_METHOD": "ALL_GATHER",
            "PREFILL_CP_SIZE": "4",
            "RTP_LLM_REQUIRE_CP_SHARDED_KV_CACHE": "1",
            "GLM5_CP_OPT": "1",
            "USE_GATHER_PATH": "1",
            "GLM5_PREFILL_INDEXER_TOPK_BACKEND": "topk_v3_tie_break",
            "CP_FORCE_SINGLE_PREFILL": "0",
            "MAX_CONTEXT_BATCH_SIZE": "1",
            "REUSE_CACHE": "1",
            "ENABLE_DEVICE_CACHE": "1",
            "ENABLE_MEMORY_CACHE": "1",
            "MEMORY_CACHE_SIZE_MB": "4096",
            "RESERVER_RUNTIME_MEM_MB": "80728",
            "RTP_LLM_CP_ACTUAL_TO_PADDED_INDEX_OPT": "0",
            "RTP_LLM_CP_RESTORE_INDEX_CPU_OPT": "0",
            "RTP_LLM_PIN_HOST_BLOCK_POOL": "1",
            "RTP_LLM_ASYNC_HOST_CACHE_REGISTRATION": "1",
            "RTP_LLM_ASYNC_PREPARE_CACHE": "1",
            "RTP_LLM_HOST_BLOCK_POOL_PREFAULT_THREADS": "16",
            "ENABLE_GPU_PREFIX_TREE": "0",
            "ENABLE_PREFIX_TREE_MEMORY_CACHE": "0",
            "ENABLE_DSV4_STATE_BLOCK_INDEPENDENT_EVICTION": "0",
            "TORCH_CUDA_POOL_PREALLOC_MB": "0",
            "TORCH_CUDA_POOL_PREALLOC_SAFETY_MB": "4096",
            "DG_MEGA_MOE_NVLINK_BARRIER_TIMEOUT_SECS": "300",
            "DSV4_STARTUP_REAL_WARMUP": "1",
            "RTP_LLM_STARTUP_REAL_WARMUP_TOKEN_LENS": "2",
            "EPLB_UPDATE_TIME": "2",
            "EPLB_STATS_WINDOW_SIZE": "2",
            "EPLB_METRICS_REPORT_INTERVAL": "2",
            "EPLB_REARRANGE_AFTER_PREFILL": "1",
            "REMOTE_SERVER_PORT": str(decode_port),
        },
    )
    decode = dict(
        common,
        **{
            "ROLE_TYPE": "DECODE",
            "TP_SIZE": "1",
            "DP_SIZE": "2",
            "EP_SIZE": "2",
            "WORLD_SIZE": "2",
            "CUDA_VISIBLE_DEVICES": ",".join(devices[4:6]),
            "CP_ROTATE_METHOD": "PREFILL_CP",
            "PREFILL_CP_SIZE": "4",
            "REUSE_CACHE": "0",
            "RESERVER_RUNTIME_MEM_MB": "24480",
            "KV_CACHE_MEM_MB": "16384",
            "ENABLE_CUDA_GRAPH": "1",
            "DECODE_CAPTURE_CONFIG": "16,32,40,48,64",
            "RTP_LLM_STREAM_ASYNC": "1",
            "RTP_LLM_DROP_BROAD_SYNC": "1",
            "RTP_LLM_DEVICE_INPUT": "1",
            "RTP_LLM_GLM5_CMP": "1",
            "GLM5_INDEXER_TOPK_BACKEND": "topk_v3",
            "ENABLE_FP32_LM_HEAD": "0",
            "CUDA_LAUNCH_BLOCKING": "0",
            "RTP_LLM_FORCE_SP_PREFILL_CUDA_GRAPH": "1",
            "DISABLE_SP_PREFILL_CUDA_GRAPH": "0",
            "RESERVE_BLOCK_RATIO": "8",
            "RTP_LLM_DSA_MLA_HOST_CACHE_MB": "8192",
            "RTP_LLM_HOST_BLOCK_POOL_PREFAULT_THREADS": "16",
            "RTP_LLM_ASYNC_HOST_CACHE_REGISTRATION": "1",
            "DG_MEGA_MOE_NVLINK_BARRIER_TIMEOUT_SECS": "30",
            "EPLB_UPDATE_TIME": "32",
            "EPLB_STATS_WINDOW_SIZE": "16",
            "EPLB_METRICS_REPORT_INTERVAL": "16",
            "REMOTE_SERVER_PORT": str(prefill_port),
        },
    )
    prefill["MODEL_SERVICE_CONFIG"] = ServiceRoute(
        service_id="glm53_eplb_test",
        use_local=True,
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
    ).model_dump_json()
    return {"prefill": prefill, "decode": decode}


class Glm53PdEplbTest(unittest.TestCase):
    def test_full_model_pd_rearrangement(self):
        checkpoint = os.environ["EPLB_GLM5_CHECKPOINT"]
        output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        self.assertEqual(len(devices), 6)
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(Path(checkpoint) / "tokenizer.json"),
            chat_template=(Path(checkpoint) / "chat_template.jinja").read_text(),
            eos_token="<|endoftext|>",
            pad_token="<|endoftext|>",
        )
        prompts = [
            "What is the capital of France? Answer in one sentence.",
            "Write a Python function that adds two integers, then explain it.",
            "Explain why the sky looks blue in about 100 words.",
            "What is 17 plus 25? Show the calculation.",
            "Name three animals that live in the ocean.",
            "Translate good morning into French and Spanish.",
        ]
        reference = {}
        records = []
        for mode in os.environ.get("EPLB_GLM5_MODES", "NONE,EPLB").split(","):
            ports = {
                role: MagaServerManager.get_free_port()
                for role in ("prefill", "decode")
            }
            envs = role_environments(mode, devices, ports["prefill"], ports["decode"])
            (output / f"{mode}_config.json").write_text(json.dumps(envs, indent=2))
            servers = {
                role: MagaServerManager(
                    role_name=f"{mode.lower()}_{role}", env_args=env, port=ports[role]
                )
                for role, env in envs.items()
            }
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    started = {
                        role: pool.submit(
                            server.start_server,
                            model_path=checkpoint,
                            model_type="glm_5",
                            timeout=1800,
                        )
                        for role, server in servers.items()
                    }
                    for role, future in started.items():
                        self.assertTrue(
                            future.result(), f"{mode} {role} failed to start"
                        )
                for repeat in range(2):
                    for index, prompt in enumerate(prompts):
                        rendered = tokenizer.apply_chat_template(
                            [{"role": "user", "content": prompt}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        start = time.perf_counter()
                        response = requests.post(
                            f"http://127.0.0.1:{servers['prefill'].port}/",
                            json={
                                "prompt": rendered,
                                "generate_config": {
                                    "max_new_tokens": 128,
                                    "top_k": 1,
                                    "top_p": 1.0,
                                    "random_seed": 123,
                                    "is_streaming": False,
                                    "return_logits": True,
                                    "return_output_ids": True,
                                    "return_input_ids": True,
                                },
                            },
                            timeout=900,
                        )
                        if not response.ok:
                            (
                                output / f"{mode}_{repeat}_{index}_http_error.txt"
                            ).write_text(response.text)
                        response.raise_for_status()
                        result = response.json()
                        name = f"{mode}_{repeat}_{index}.json"
                        (output / name).write_text(
                            json.dumps(result, ensure_ascii=False)
                        )
                        records.append(
                            {
                                "mode": mode,
                                "repeat": repeat,
                                "prompt": prompt,
                                "artifact": name,
                                "wall_ms": (time.perf_counter() - start) * 1000,
                            }
                        )
                        (output / "requests.json").write_text(
                            json.dumps(records, indent=2)
                        )
                        logits = np.asarray(result["logits"], dtype=np.float32)
                        with self.subTest(
                            mode=mode, repeat=repeat, prompt=index, check="generation"
                        ):
                            self.assertNotIn("error", result, result)
                            self.assertTrue(result["finished"])
                            self.assertTrue(result["output_ids"])
                            self.assertTrue(result["aux_info"]["pd_sep"])
                            text = result["response"].strip()
                            self.assertTrue(text)
                            self.assertNotRegex(text, r"(?s)^(.{1,32})\1{7,}$")
                            if index == 0:
                                self.assertIn("Paris", text)
                            self.assertGreater(logits.size, 100000)
                            self.assertTrue(np.isfinite(logits).all())
                        if index not in reference:
                            reference[index] = result
                        else:
                            expected = reference[index]
                            with self.subTest(
                                mode=mode, repeat=repeat, prompt=index, check="tokens"
                            ):
                                self.assertEqual(
                                    result["input_ids"], expected["input_ids"]
                                )
                                self.assertEqual(
                                    result["output_ids"], expected["output_ids"]
                                )
                            with self.subTest(
                                mode=mode,
                                repeat=repeat,
                                prompt=index,
                                check="bitwise_logits",
                            ):
                                np.testing.assert_allclose(
                                    logits,
                                    np.asarray(expected["logits"], dtype=np.float32),
                                    rtol=0,
                                    atol=0,
                                )
                if mode in ("EPLB", "ALL"):
                    for role, size in (("prefill", 4), ("decode", 2)):
                        check_eplb_applied(output / f"{mode.lower()}_{role}_logs", size)
                    port = servers["prefill"].port
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompts[0]}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    probe = requests.post(
                        f"http://127.0.0.1:{port}/",
                        json={
                            "prompt": prompt,
                            "generate_config": {
                                "max_new_tokens": 128,
                                "top_k": 1,
                                "is_streaming": False,
                                "return_output_ids": True,
                            },
                        },
                        timeout=900,
                    )
                    probe.raise_for_status()
                    result = probe.json()
                    (output / f"{mode}_post_migration.json").write_text(
                        json.dumps(result)
                    )
                    self.assertTrue(result["finished"])
                    self.assertTrue(result["output_ids"])
                    self.assertIn("Paris", result["response"])
            finally:
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    list(
                        pool.map(lambda server: server.stop_server(), servers.values())
                    )
            for role, size in (("prefill", 4), ("decode", 2)):
                logs_dir = output / f"{mode.lower()}_{role}_logs"
                snapshots = [
                    json.loads(path.read_text())
                    for path in logs_dir.rglob("eplb_snapshot/*.json")
                ]
                self.assertTrue(snapshots, f"No {role} load snapshots")
                self.assertTrue(
                    any(item["total_activations"] > 0 for item in snapshots)
                )
                logs = "\n".join(
                    path.read_text(errors="replace") for path in logs_dir.rglob("*.log")
                )
                if mode == "EPLB":
                    applied = re.findall(
                        r"\[EPLB\] applied .*?update=(\d+) changed_slots=\[([^\]]*)\] ep_rank=(\d+)",
                        logs,
                    )
                    changed_ranks = {
                        int(rank)
                        for _, slots, rank in applied
                        if any(int(value.strip()) > 0 for value in slots.split(","))
                    }
                    self.assertEqual(changed_ranks, set(range(size)), (role, applied))
                else:
                    self.assertNotIn("[EPLB] applied", logs)


if __name__ == "__main__":
    unittest.main()
