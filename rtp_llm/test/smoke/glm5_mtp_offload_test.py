"""Native GLM NextN P/D comparison across MTP, offload and CUDA graphs."""

import json
import logging
import os
import re
import tempfile
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
        hot_cache_mode = os.getenv("RTP_TEST_SPARSE_KV_OFFLOAD_HOT_CACHE_MODE", "static")
        self.assertIn(hot_cache_mode, ("static", "dynamic"))
        role_size = int(os.getenv("RTP_GLM_MTP_ROLE_SIZE", "1"))
        layers = int(os.getenv("RTP_GLM_MTP_LAYERS", "4"))
        stable = os.getenv("RTP_GLM_MTP_STABLE_TOPK", "0") == "1"
        comparison = os.getenv("RTP_GLM_MTP_COMPARE_IDS", "strict")
        self.assertIn(comparison, ("strict", "record"))
        kv_budget_mb = int(
            os.getenv("RTP_GLM_MTP_KV_MB", "16384" if layers == 78 else "2048")
        )
        model_root = Path(os.getenv("RTP_TEST_MODEL_ROOT", "/mnt/nas1/hf"))
        model_path = str(
            model_root / os.getenv("RTP_GLM_MTP_CHECKPOINT", "GLM-5-FP8-4layer")
        )
        devices = list(get_gpu_ids())
        self.assertGreaterEqual(len(devices), 2 * role_size)
        output = Path(os.getenv("TEST_UNDECLARED_OUTPUTS_DIR", "."))
        output.mkdir(parents=True, exist_ok=True)
        draft_source = Path(
            os.getenv(
                "RTP_GLM_MTP_DRAFT_CHECKPOINT", str(model_root / "GLM-5-BF16-4layer")
            )
        )
        draft_config = json.loads((draft_source / "config.json").read_text())
        draft_weights = json.loads(
            (draft_source / "model.safetensors.index.json").read_text()
        )["weight_map"]
        mtp_layers = sorted(
            {
                int(match.group(1))
                for key in draft_weights
                if (
                    match := re.fullmatch(r"model\.layers\.(\d+)\.eh_proj\.weight", key)
                )
            }
        )
        self.assertTrue(mtp_layers, f"No real NextN weights in {draft_source}")
        # Cropped checkpoints can retain the original NextN layer number.
        # Give the draft loader its original offset without copying weights
        # or changing the target's four-layer configuration. Expose only the
        # shards needed by NextN and its global weights to the fast loader.
        draft_dir = tempfile.TemporaryDirectory(prefix="glm-mtp-", dir=output)
        self.addCleanup(draft_dir.cleanup)
        draft_path = draft_dir.name
        draft_config["num_hidden_layers"] = mtp_layers[0] or 1
        draft_config["num_nextn_predict_layers"] = len(mtp_layers)
        draft_weights = {
            key: shard
            for key, shard in draft_weights.items()
            if not key.startswith("model.layers.")
            or any(key.startswith(f"model.layers.{layer}.") for layer in mtp_layers)
        }
        draft_shards = set(draft_weights.values())
        for source in draft_source.iterdir():
            if source.name in ("config.json", "model.safetensors.index.json"):
                continue
            if source.suffix == ".safetensors" and source.name not in draft_shards:
                continue
            (Path(draft_path) / source.name).symlink_to(source.resolve())
        (Path(draft_path) / "config.json").write_text(json.dumps(draft_config))
        (Path(draft_path) / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": draft_weights})
        )
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
            " --act_type BF16 --fp8_kv_cache 1"
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
        mtp_baseline = None
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
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
                "FRONTEND_PRE_STOP_DRAIN_SECONDS": "0",
                "DASH_SC_GRPC_PRE_STOP_DRAIN_SECONDS": "0",
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
                cli += " --use_deepep_moe 0 --use_all_gather 1"
                if role == 1:
                    # This single-rank FP8 backend supports both eager and
                    # graph; auto uses a graph-unsafe hybrid MoE on SM100.
                    cli += " --moe_strategy fp8_per_block_no_dp_masked"
            if mtp:
                cli += (
                    f" --sp_type eagle --sp_model_type glm_5_mtp --sp_checkpoint_path {draft_path} --gen_num_per_cycle 3"
                    # Quantize the real BF16 NextN checkpoint through the
                    # loader, matching the target's FP8 MoE compute contract.
                    " --sp_quantization FP8_PER_BLOCK"
                )
            if offload:
                cli += " --enable_sparse_kv_offload 1 --sparse_kv_offload_host_mb 1024 --sparse_kv_offload_test_hbm_blocks 2"
                cli += f" --sparse_kv_offload_hot_cache_mode {hot_cache_mode}"
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
            if resume > 1:
                mtp_baseline = [
                    json.loads(
                        (output / f"mtp_1_graph_0_offload_0_{i}.json").read_text()
                    )["output_ids"]
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
                runtime_log_offset = len(Path(decode.log_file_path).read_text(errors="replace"))

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
                # Compare offload with the same speculative algorithm. Native
                # FP8 FlashMLA changes split-K with query batch geometry; MTP
                # verification can therefore differ from one-token decode even
                # when Q and selected KV bytes are identical. Keep the serial
                # MTP/non-MTP control strict and record the concurrent control
                # separately, without accepting offload-induced differences.
                if mtp:
                    self.assertEqual(ids[:3], baseline[:3], label)
                    if mtp_baseline is None:
                        mtp_baseline = ids
                    expected_ids = mtp_baseline
                else:
                    expected_ids = baseline

                def differences(actual_ids, reference_ids):
                    return [
                        dict(
                            request=i,
                            first_token=next(
                                j
                                for j, (x, y) in enumerate(zip(actual[0], expected[0]))
                                if x != y
                            ),
                        )
                        for i, (actual, expected) in enumerate(
                            zip(actual_ids, reference_ids)
                        )
                        if actual != expected
                    ]

                mismatches = differences(ids, expected_ids)
                non_mtp_mismatches = differences(ids, baseline) if mtp else []
                (output / f"{label}_comparison.json").write_text(
                    json.dumps(
                        dict(
                            comparison=comparison,
                            matched_requests=len(ids) - len(mismatches),
                            mismatches=mismatches,
                            reference="mtp_1_graph_0_offload_0" if mtp else label,
                            non_mtp_mismatches=non_mtp_mismatches,
                        ),
                        indent=2,
                    )
                )
                if comparison == "strict":
                    self.assertEqual(ids, expected_ids, label)
                if graph:
                    session.post(
                        f"http://127.0.0.1:{decode.port}/set_log_level",
                        json={"log_level": "DEBUG"},
                        timeout=10,
                    ).raise_for_status()
                    generate((100, prompts[0]))
                log = Path(decode.log_file_path).read_text(errors="replace")
                if offload:
                    self.assertIn("decode warmup runtime=", log)
                    self.assertRegex(log, r"sparse KV offload tag=default H=2 P=[1-9]")
                    self.assertIn(f"hot_cache_mode={hot_cache_mode}", log)
                if offload and hot_cache_mode == "dynamic":
                    capacities = [
                        (int(active), int(maximum))
                        for active, maximum in re.findall(
                            r"sparse KV hot cache active_rows=(\d+) max_rows=(\d+)",
                            log[runtime_log_offset:],
                        )
                    ]
                    self.assertTrue(any(0 < active < maximum for active, maximum in capacities), capacities)
                    self.assertTrue(any(active == 0 for active, _ in capacities), capacities)
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
                        non_mtp_mismatches=non_mtp_mismatches,
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
