"""Full V4 Flash DSpark P/D accuracy with default warmup and real host KV.

RTP_DSV4_TEST_CHECKPOINT must point to a locally staged checkpoint.
The overlap prefill path and native top-k are used without test hooks.
DSpark exposes accepted token IDs; full-vocabulary logit parity is covered by
the GLM/DSA smoke and attention tensor parity by the V4 GPU tests.
"""

import json
import logging
import os
from pathlib import Path
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class Dsv4SparseKVOffloadAccuracyTest(unittest.TestCase):
    def test_full_model_dspark_pd(self):
        hot_cache_mode = os.getenv("RTP_TEST_SPARSE_KV_OFFLOAD_HOT_CACHE_MODE", "static")
        self.assertIn(hot_cache_mode, ("static", "dynamic"))
        output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        model = os.environ.get("RTP_DSV4_TEST_CHECKPOINT")
        self.assertTrue(
            model,
            "Set RTP_DSV4_TEST_CHECKPOINT to a locally staged V4 Flash checkpoint",
        )
        devices = list(get_gpu_ids())
        self.assertGreaterEqual(len(devices), 4)
        ports = [MagaServerManager.get_free_port() for _ in range(2)]
        route = ServiceRoute(
            service_id="v4_final_smoke",
            use_local=True,
            role_endpoints=[
                GroupEndPoint(
                    group="default",
                    prefill_endpoint=EndPoint(
                        type="Vipserver",
                        address=f"127.0.0.1:{ports[0]}",
                        protocol="http",
                        path="/",
                    ),
                    decode_endpoint=EndPoint(
                        type="Vipserver",
                        address=f"127.0.0.1:{ports[1]}",
                        protocol="http",
                        path="/",
                    ),
                )
            ],
        )
        common = (
            " --load_method scratch --world_size 2 --ep_size 2 --act_type BF16"
            " --fp8_kv_cache 1 --seq_size_per_block 256 --kernel_seq_size_per_block 128"
            " --max_seq_len 4096 --max_context_batch_size 1 --concurrency_limit 4"
            " --kv_cache_mem_mb 4096 --reserver_runtime_mem_mb 4096 --test_block_num 128"
            " --dsv4_fixed_pool_blocks 32 --dsv4_hca_state_pool_blocks 32"
            " --use_deepep_moe 1 --moe_strategy mega_moe_se --cache_store_rdma_mode 0"
            " --use_local 1 --reuse_cache 0 --enable_memory_cache 0 --enable_disk_cache 0"
            " --enable_remote_cache 0 --load_cache_timeout_ms 180000"
            " --frontend_server_count 1 --think_mode 0 --enable_fp32_lm_head 0"
            " --prefill_cp_kv_cache_sharded 1 --sp_type dspark --gen_num_per_cycle 3"
            f" --sp_model_type deepseek_v4_dspark --sp_checkpoint_path {model} --sp_act_type bf16"
        )

        def manager(role, graph, offload):
            label = f"v4_{role}_graph_{int(graph)}_offload_{int(offload)}"
            env = dict(
                LOAD_PYTHON_MODEL="1",
                FT_SERVER_TEST="1",
                WORLD_SIZE="2",
                LOG_LEVEL="INFO",
                MODEL_SERVICE_CONFIG=route.model_dump_json(),
                REMOTE_RPC_SERVER_IP="localhost",
                REMOTE_SERVER_PORT=ports[1 - role],
                DSV4_USE_FRAMEWORK_KV="1",
                DSV4_PREFILL_CP_OVERLAP="1",
                DG_JIT_CPP_STANDARD="20",
                OMP_NUM_THREADS="8",
                PYTORCH_ALLOC_CONF="expandable_segments:True",
                MODEL_EVAL_VARIANT="fp8_kv_cache",
                DSV4_BF16_VLLM="0",
                DSV4_COMPRESSOR_FAST="1",
                FRONTEND_PRE_STOP_DRAIN_SECONDS="0",
                DASH_SC_GRPC_PRE_STOP_DRAIN_SECONDS="0",
            )
            if role == 0:
                cli = (
                    common
                    + " --role_type PREFILL --tp_size 2 --cp_rotate_method ALL_GATHER --use_deepep_low_latency 0 --enable_cuda_graph 0"
                )
            else:
                cli = (
                    common
                    + f" --role_type DECODE --tp_size 1 --dp_size 2 --cp_rotate_method PREFILL_CP --prefill_cp_size 2 --use_deepep_low_latency 1 --enable_cuda_graph {int(graph)} --decode_capture_config 1,2"
                )
                if offload:
                    cli += " --enable_sparse_kv_offload 1 --sparse_kv_offload_host_mb 1024 --sparse_kv_offload_test_hbm_blocks 2"
                    cli += f" --sparse_kv_offload_hot_cache_mode {hot_cache_mode}"
            result = MagaServerManager(
                env_args=env,
                device_ids=devices[2 * role : 2 * (role + 1)],
                port=ports[role],
                role_name=label,
                smoke_args_str=cli,
            )
            self.addCleanup(result.stop_server)
            return result

        session = requests.Session()
        session.trust_env = False
        self.addCleanup(session.close)
        prefill = manager(0, False, False)
        baseline = None
        summary = []
        prompts = [
            "The capital of France is",
            "The record is " + "red green blue black white yellow. " * 300,
        ]
        for graph, offload in [
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ]:
            decode = manager(1, graph, offload)
            label = f"graph_{int(graph)}_offload_{int(offload)}"
            try:
                starting = [prefill, decode] if baseline is None else [decode]
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [
                        executor.submit(
                            m.start_server,
                            model_path=model,
                            model_type="deepseek_v4",
                            timeout=1800,
                        )
                        for m in starting
                    ]
                    for future in futures:
                        self.assertTrue(future.result())
                startup = Path(decode.log_file_path).read_text(errors="replace")
                if offload:
                    self.assertIn(f"hot_cache_mode={hot_cache_mode}", startup)
                    self.assertIn("decode warmup runtime=", startup)
                    self.assertRegex(
                        startup, r"sparse KV offload tag=csa_kv H=2 P=[1-9]"
                    )
                if graph:
                    self.assertIn("capture success for batch size", startup)

                results = []
                metrics = []
                for i, prompt in enumerate(prompts + prompts[1:] * 2 + prompts[:1]):
                    if graph and i == 4:
                        session.post(
                            f"http://127.0.0.1:{decode.port}/set_log_level",
                            json={"log_level": "DEBUG"},
                            timeout=10,
                        ).raise_for_status()
                    response = session.post(
                        f"http://127.0.0.1:{prefill.port}/",
                        json=dict(
                            prompt=prompt,
                            generate_config=dict(
                                max_new_tokens=32,
                                min_new_tokens=32,
                                top_k=1,
                                top_p=1.0,
                                random_seed=42,
                                reuse_cache=False,
                                can_use_pd_separation=True,
                                return_output_ids=True,
                                aux_info=True,
                            ),
                        ),
                        timeout=600,
                    )
                    (output / f"{label}_{i}.json").write_text(response.text)
                    response.raise_for_status()
                    result = response.json()
                    self.assertNotIn("error_code", result)
                    self.assertTrue(result.get("finished"))
                    self.assertTrue(result["aux_info"]["pd_sep"])
                    self.assertEqual(len(result["output_ids"][0]), 32)
                    self.assertGreater(
                        result["aux_info"]["speculative_draft_rounds"], 0
                    )
                    results.append(result["output_ids"])
                    metrics.append(result["aux_info"])
                self.assertEqual(results[0], results[-1])
                self.assertEqual(results[1], results[2])
                self.assertEqual(results[1], results[3])
                (output / f"{label}_accuracy.json").write_text(
                    json.dumps(
                        dict(
                            token_ids_equal=baseline is None or results == baseline,
                            requests=len(results),
                            request_metrics=metrics,
                        ),
                        indent=2,
                    )
                )
                if baseline is None:
                    baseline = results
                else:
                    self.assertEqual(results, baseline)
                if graph:
                    self.assertIn(
                        "batch size used in replay:",
                        Path(decode.log_file_path).read_text(errors="replace"),
                    )
                summary.append(dict(mode=label, requests=len(results), ids=results))
                (output / "summary.json").write_text(json.dumps(summary, indent=2))
                logging.info(
                    "PASS V4 %s: full 43-layer DSpark P/D, identical greedy outputs",
                    label,
                )
            finally:
                decode.stop_server()
        prefill.stop_server()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
