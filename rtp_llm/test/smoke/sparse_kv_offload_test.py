"""Native sparse MLA off/on comparison with real prefill/decode servers."""

import json
import logging
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class SparseKVOffloadModelTest(unittest.TestCase):
    def _compare_model(self, checkpoint, model_type, fp8, layer_count=None):
        indexshare = checkpoint.startswith("GLM-5.3")
        role_size = 1
        devices = list(get_gpu_ids())
        self.assertGreaterEqual(len(devices), 2 * role_size)
        ports = [MagaServerManager.get_free_port() for _ in range(2)]
        route = ServiceRoute(
            service_id="sparse_offload",
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
            f"--tp_size {role_size} --dp_size 1 --ep_size {role_size} --world_size {role_size} --act_type BF16"
            " --warm_up 0 --seq_size_per_block 256 --kernel_seq_size_per_block 128"
            " --max_seq_len 4096 --max_context_batch_size 1 --concurrency_limit 2"
            " --frontend_server_count 1"
            " --kv_cache_mem_mb 512 --reuse_cache 0 --enable_memory_cache 0"
            " --enable_disk_cache 0 --enable_remote_cache 0 --use_local 1"
            " --cache_store_rdma_mode 0 --load_cache_timeout_ms 120000"
            " --use_deepep_moe 0 --use_all_gather 1"
            f" --fp8_kv_cache {int(fp8)}"
        )
        if layer_count is not None:
            common += f" --hack_layer_num {layer_count}"
        model_path = str(
            Path(os.environ.get("RTP_TEST_MODEL_ROOT", "/mnt/nas1/hf")) / checkpoint
        )
        output_dir = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "."))
        session = requests.Session()
        session.trust_env = False
        self.addCleanup(session.close)

        def manager(role, args, label, drain_before_moe=True):
            result = MagaServerManager(
                env_args={
                    "LOAD_PYTHON_MODEL": "1",
                    "FT_SERVER_TEST": "1",
                    "WORLD_SIZE": str(role_size),
                    "LOG_LEVEL": "INFO",
                    "GRPC_CLIENT_CHANNEL_BACKUP_POLL_INTERVAL_MS": "500",
                    "REMOTE_RPC_SERVER_IP": "localhost",
                    "REMOTE_SERVER_PORT": ports[1 - role],
                    "MODEL_SERVICE_CONFIG": route.model_dump_json(),
                    "RTP_LLM_DSA_MLA_DRAIN_BEFORE_MOE": str(int(drain_before_moe)),
                    **(
                        {
                            "RTP_SPARSE_TEST_STABLE_TOPK": "1",
                            "PYTHONPATH": str(
                                Path(__file__).with_name("offload_test_hooks")
                            )
                            + os.pathsep
                            + os.environ.get("PYTHONPATH", ""),
                        }
                        if indexshare
                        else {}
                    ),
                },
                device_ids=devices[role * role_size : (role + 1) * role_size],
                port=ports[role],
                role_name=f"offload_{checkpoint}_{label}",
                smoke_args_str=common + args,
            )
            self.addCleanup(result.stop_server)
            return result

        prefill = manager(0, " --role_type PREFILL --enable_cuda_graph 0", "prefill")
        baseline = None
        # Distinct prompts and repeated requests also exercise block reuse.
        prompts = [
            "The sequence is " + "one two three four five six seven eight. " * 100,
            "The record is "
            + "red green blue black white yellow. " * (400 if indexshare else 140),
        ]
        modes = (
            (False, False, True),
            (False, True, True),
            (True, False, True),
            (True, True, True),
        )
        if indexshare:
            modes += ((True, True, False),)
        for graph, offload, drain_before_moe in modes:
            label = f"graph_{int(graph)}_offload_{int(offload)}"
            if not drain_before_moe:
                label += "_drain_0"
            args = (
                f" --role_type DECODE --enable_cuda_graph {int(graph)}"
                " --decode_capture_config 1,2"
            )
            if fp8:
                args += " --moe_strategy fp8_per_block_no_dp_masked"
            if offload:
                args += (
                    " --enable_sparse_kv_offload 1 --sparse_kv_offload_host_mb 32"
                    " --sparse_kv_offload_test_hbm_blocks 2"
                )
            decode = manager(1, args, label, drain_before_moe)
            try:
                servers = [decode] if baseline is not None else [prefill, decode]
                with ThreadPoolExecutor(max_workers=2) as executor:
                    starts = [
                        executor.submit(
                            server.start_server,
                            model_path=model_path,
                            model_type=model_type,
                            timeout=1800,
                        )
                        for server in servers
                    ]
                    for started in starts:
                        self.assertTrue(started.result())
                response = session.post(
                    f"http://127.0.0.1:{decode.port}/set_log_level",
                    json={"log_level": "INFO"},
                    timeout=10,
                )
                response.raise_for_status()
                log_path = Path(decode.log_file_path)
                startup = log_path.read_text(errors="replace")
                offset = len(startup)
                if indexshare:
                    self.assertIn("SPARSE_OFFLOAD_TEST_STABLE_TOPK", startup)
                if offload:
                    self.assertRegex(
                        startup, r"sparse KV offload tag=default H=2 P=[1-9]"
                    )
                if graph:
                    self.assertIn("capture success for batch size", startup)
                outputs = []
                for index, prompt in enumerate(prompts + prompts[:1]):
                    # DEBUG adds device synchronization; use it only for the
                    # final request that proves graph replay in the log.
                    if graph and index == 2:
                        response = session.post(
                            f"http://127.0.0.1:{decode.port}/set_log_level",
                            json={"log_level": "DEBUG"},
                            timeout=10,
                        )
                        response.raise_for_status()
                    response = session.post(
                        f"http://127.0.0.1:{prefill.port}/",
                        json={
                            "prompt": prompt,
                            "generate_config": {
                                "max_new_tokens": 12,
                                "top_k": 1,
                                "top_p": 1.0,
                                "random_seed": 42,
                                "reuse_cache": False,
                                "can_use_pd_separation": True,
                                "return_input_ids": True,
                                "return_output_ids": True,
                                "aux_info": True,
                                **(
                                    {
                                        "min_new_tokens": 12,
                                        "return_logits": True,
                                        "select_tokens_id": [
                                            1024 * i for i in range(1, 16)
                                        ],
                                    }
                                    if indexshare
                                    else {}
                                ),
                            },
                        },
                        timeout=600,
                    )
                    (output_dir / f"{checkpoint}_{label}_{index}.http.txt").write_text(
                        response.text
                    )
                    response.raise_for_status()
                    actual = response.json()
                    self.assertNotIn("error_code", actual)
                    self.assertIs(actual.get("finished"), True)
                    self.assertTrue(actual["aux_info"]["pd_sep"])
                    self.assertGreater(actual["aux_info"]["input_len"], 3 * 256)
                    if indexshare and index == 1:
                        self.assertGreater(actual["aux_info"]["input_len"], 2048)
                    self.assertTrue(actual["output_ids"][0])
                    if indexshare:
                        self.assertEqual(len(actual["output_ids"][0]), 12)
                        self.assertEqual(len(actual["logits"][0]), 15)
                        outputs.append((actual["output_ids"], actual["logits"]))
                    else:
                        outputs.append(actual["output_ids"])
                self.assertEqual(outputs[0], outputs[2])
                if baseline is None:
                    baseline = outputs
                else:
                    self.assertEqual(outputs, baseline, label)
                if graph:
                    self.assertRegex(
                        log_path.read_text(errors="replace")[offset:],
                        r"batch size used in replay: 1 \(graph key [12]\)",
                    )
                (output_dir / f"{checkpoint}_{label}.json").write_text(
                    json.dumps(outputs)
                )
                logging.info("%s %s: identical greedy output", checkpoint, label)
            finally:
                decode.stop_server()
        prefill.stop_server()

    def test_glm_fp8(self):
        self._compare_model("GLM-5-FP8-4layer", "glm_5", True)

    def test_glm_bf16(self):
        self._compare_model("GLM-5-BF16-4layer", "glm_5", False)

    def test_deepseek_v32(self):
        self._compare_model("DeepSeek-V3.2-4layer", "deepseek_v32", True)

    def test_glm53(self):
        self._compare_model("GLM-5.3-ms-4ed051d0", "glm_5", True, layer_count=7)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
