"""Native GLM NextN P/D comparison across MTP, offload and CUDA graphs."""

import json
import logging
import os
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class Glm5MtpOffloadTest(unittest.TestCase):
    def test_native_pd(self):
        role_size = int(os.getenv("RTP_GLM_MTP_ROLE_SIZE", "1"))
        layers = int(os.getenv("RTP_GLM_MTP_LAYERS", "7"))
        stable = os.getenv("RTP_GLM_MTP_STABLE_TOPK", "1") == "1"
        comparison = os.getenv("RTP_GLM_MTP_COMPARE_IDS", "strict")
        self.assertIn(comparison, ("strict", "record"))
        kv_budget_mb = int(
            os.getenv("RTP_GLM_MTP_KV_MB", "16384" if layers == 78 else "2048")
        )
        model_path = str(
            Path(os.getenv("RTP_TEST_MODEL_ROOT", "/mnt/nas1/hf"))
            / "GLM-5.3-ms-4ed051d0"
        )
        devices = list(get_gpu_ids())
        self.assertGreaterEqual(len(devices), 2 * role_size)
        output = Path(os.getenv("TEST_UNDECLARED_OUTPUTS_DIR", "."))
        output.mkdir(parents=True, exist_ok=True)
        ports = [MagaServerManager.get_free_port() for _ in range(2)]
        route = ServiceRoute(
            service_id="glm_mtp_offload",
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
            f" --tp_size 1 --dp_size {role_size} --ep_size {role_size} --world_size {role_size}"
            " --act_type BF16 --fp8_kv_cache 1 --load_method scratch --warm_up 0"
            " --seq_size_per_block 256 --kernel_seq_size_per_block 128 --max_seq_len 4096"
            f" --max_context_batch_size 1 --concurrency_limit {2 * role_size}"
            f" --frontend_server_count 1 --kv_cache_mem_mb {kv_budget_mb} --reserver_runtime_mem_mb 2048"
            " --reuse_cache 0 --enable_memory_cache 0 --enable_disk_cache 0 --enable_remote_cache 0"
            " --use_local 1 --cache_store_rdma_mode 0 --load_cache_timeout_ms 180000"
        )
        if layers != 78:
            common += f" --hack_layer_num {layers}"
        summaries = []
        baseline = None
        prefill = None
        prefill_mtp = None
        prompts = [
            "The capital of France is",
            "The record is " + "red green blue black white yellow. " * 340,
        ]
        session = requests.Session()
        session.trust_env = False
        self.addCleanup(session.close)

        def make_manager(role, mtp, graph, offload, label):
            env = {
                "LOAD_PYTHON_MODEL": "1",
                "FT_SERVER_TEST": "1",
                "WORLD_SIZE": str(role_size),
                "LOG_LEVEL": "INFO",
                "REMOTE_RPC_SERVER_IP": "localhost",
                "REMOTE_SERVER_PORT": ports[1 - role],
                "MODEL_SERVICE_CONFIG": route.model_dump_json(),
                "RTP_LLM_DSA_MLA_DRAIN_BEFORE_MOE": "1",
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            }
            if stable:
                env.update(
                    RTP_SPARSE_TEST_STABLE_TOPK="1",
                    PYTHONPATH=str(Path(__file__).with_name("offload_test_hooks"))
                    + os.pathsep
                    + os.getenv("PYTHONPATH", ""),
                )
            cli = (
                common
                + f" --role_type {'PREFILL' if role == 0 else 'DECODE'} --enable_cuda_graph {int(graph)} --decode_capture_config 1,2"
            )
            if role_size > 1:
                cli += f" --use_deepep_moe 1 --use_deepep_low_latency {int(role == 1)} --use_all_gather 0"
                cli += " --moe_strategy " + (
                    "fp8_per_block_ep_normal"
                    if role == 0
                    else "fp8_per_block_ep_low_latency"
                )
            else:
                cli += " --use_deepep_moe 0 --use_all_gather 1 --moe_strategy fp8_per_block_no_dp_masked"
            if mtp:
                cli += f" --sp_type eagle --sp_model_type glm_5_mtp --sp_checkpoint_path {model_path} --gen_num_per_cycle 3"
            if offload:
                cli += " --enable_sparse_kv_offload 1 --sparse_kv_offload_host_mb 1024 --sparse_kv_offload_test_hbm_blocks 2"
            manager = MagaServerManager(
                env_args=env,
                device_ids=devices[role * role_size : (role + 1) * role_size],
                port=ports[role],
                role_name=label,
                smoke_args_str=cli,
            )
            self.addCleanup(manager.stop_server)
            return manager

        def start(managers):
            with ThreadPoolExecutor(max_workers=len(managers)) as executor:
                futures = [
                    executor.submit(
                        manager.start_server,
                        model_path=model_path,
                        model_type="glm_5",
                        timeout=1800,
                    )
                    for manager in managers
                ]
                while not all(future.done() for future in futures):
                    for manager in managers:
                        if (
                            manager.log_file_path
                            and Path(manager.log_file_path).exists()
                        ):
                            log = Path(manager.log_file_path).read_text(
                                errors="replace"
                            )
                            if any(
                                marker in log
                                for marker in (
                                    "start server error:",
                                    "Multi-rank startup failed:",
                                    "CUDA out of memory",
                                    "CUDA error:",
                                )
                            ):
                                for server in managers:
                                    server.stop_server()
                                self.fail(
                                    f"startup failed; see {manager.log_file_path}"
                                )
                    time.sleep(2)
                for future in futures:
                    self.assertTrue(future.result())

        modes = [
            (False, False, False),
            (True, False, False),
            (True, False, True),
            (True, True, False),
            (True, True, True),
        ]
        resume = int(os.getenv("RTP_GLM_MTP_RESUME_MODE", "0"))
        if resume:
            summaries = json.loads((output / "glm-mtp-summary.json").read_text())
            self.assertEqual(len(summaries), resume)
            for result in summaries:
                self.assertEqual(
                    (result["role_size"], result["layers"], result["stable_topk"]),
                    (role_size, layers, stable),
                )
            baseline = [
                json.loads((output / f"mtp_0_graph_0_offload_0_{i}.json").read_text())[
                    "output_ids"
                ]
                for i in range(3 + 2 * role_size)
            ]
        for mtp, graph, offload in modes[resume:]:
            label = f"mtp_{int(mtp)}_graph_{int(graph)}_offload_{int(offload)}"
            if prefill_mtp != mtp:
                if prefill is not None:
                    prefill.stop_server()
                prefill = make_manager(0, mtp, False, False, f"prefill_{int(mtp)}")
                prefill_mtp = mtp
                pending = [prefill]
            else:
                pending = []
            decode = make_manager(1, mtp, graph, offload, label)
            try:
                start(pending + [decode])

                def generate(item):
                    index, prompt = item
                    response = session.post(
                        f"http://127.0.0.1:{prefill.port}/",
                        json={
                            "prompt": prompt,
                            "generate_config": {
                                "max_new_tokens": 32,
                                "min_new_tokens": 32,
                                "top_k": 1,
                                "top_p": 1.0,
                                "random_seed": 42,
                                "reuse_cache": False,
                                "can_use_pd_separation": True,
                                "return_output_ids": True,
                                "aux_info": True,
                            },
                        },
                        timeout=600,
                    )
                    (output / f"{label}_{index}.json").write_text(response.text)
                    response.raise_for_status()
                    result = response.json()
                    self.assertNotIn("error_code", result)
                    self.assertTrue(result.get("finished"))
                    self.assertTrue(result["aux_info"]["pd_sep"])
                    self.assertEqual(len(result["output_ids"][0]), 32)
                    return result

                serial = [generate(item) for item in enumerate(prompts + prompts[:1])]
                with ThreadPoolExecutor(max_workers=2 * role_size) as executor:
                    concurrent = list(
                        executor.map(generate, enumerate(prompts * role_size, start=3))
                    )
                ids = [result["output_ids"] for result in serial + concurrent]
                self.assertEqual(ids[0], ids[2])
                if baseline is None:
                    baseline = ids
                mismatches = [
                    dict(
                        request=i,
                        first_token=next(
                            j
                            for j, (x, y) in enumerate(zip(actual[0], expected[0]))
                            if x != y
                        ),
                    )
                    for i, (actual, expected) in enumerate(zip(ids, baseline))
                    if actual != expected
                ]
                (output / f"{label}_comparison.json").write_text(
                    json.dumps(
                        dict(
                            comparison=comparison,
                            matched_requests=len(ids) - len(mismatches),
                            mismatches=mismatches,
                        ),
                        indent=2,
                    )
                )
                if comparison == "strict":
                    self.assertEqual(ids, baseline, label)
                if graph:
                    session.post(
                        f"http://127.0.0.1:{decode.port}/set_log_level",
                        json={"log_level": "DEBUG"},
                        timeout=10,
                    ).raise_for_status()
                    generate((100, prompts[0]))
                log = Path(decode.log_file_path).read_text(errors="replace")
                if offload:
                    self.assertRegex(log, r"sparse KV offload tag=default H=2 P=[1-9]")
                if graph:
                    self.assertTrue(
                        "capture success for batch size" in log, decode.log_file_path
                    )
                    self.assertTrue(
                        "batch size used in replay:" in log, decode.log_file_path
                    )
                if mtp:
                    self.assertTrue(
                        all(
                            result["aux_info"]["speculative_draft_rounds"] > 0
                            for result in serial
                        )
                    )
                    if layers == 78:
                        self.assertTrue(
                            any(
                                result["aux_info"]["iter_count"] < 32
                                for result in serial
                            )
                        )
                summaries.append(
                    dict(
                        mode=label,
                        role_size=role_size,
                        layers=layers,
                        stable_topk=stable,
                        requests=len(ids),
                        comparison=comparison,
                        matched_requests=len(ids) - len(mismatches),
                        mismatches=mismatches,
                        results=[result["aux_info"] for result in serial],
                        log=decode.log_file_path,
                    )
                )
                (output / "glm-mtp-summary.json").write_text(
                    json.dumps(summaries, indent=2)
                )
                logging.info(
                    "PASS %s: %d/%d identical greedy outputs",
                    label,
                    len(ids) - len(mismatches),
                    len(ids),
                )
            finally:
                decode.stop_server()
        prefill.stop_server()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
