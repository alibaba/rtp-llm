"""Deployment smoke for sparse MLA PD, captured-batch shrink and EAGLE.

These are opt-in, checkpoint-backed tests, not mocked model tests. A missing
checkpoint/backend is an error. EAGLE requires a draft matching its target.
"""

import json
import os
import shlex
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch

from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class GenerationContract:
    model_type = "glm_5"
    default_checkpoint = "/mnt/nas1/hf/GLM-5-FP8-4layer"
    requires_cuda = True

    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "GPU smoke requires a working GPU")
        if self.requires_cuda:
            self.assertIsNone(
                torch.version.hip, "This target requires CUDA attention kernels"
            )
        self.devices = list(get_gpu_ids())
        self.checkpoint = os.environ.get("CHECKPOINT_PATH", self.default_checkpoint)
        config = json.loads((Path(self.checkpoint) / "config.json").read_text())
        self.config = config.get("text_config", config)
        self.outputs = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", "."))
        self.outputs.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.trust_env = False
        # Cold accelerator JIT can exceed the normal request deadline. Keep
        # correctness checks unchanged; allow explicit cold-cache smoke runs.
        self.request_timeout = int(
            os.environ.get("NEWLOADER_SMOKE_REQUEST_TIMEOUT_S", "300")
        )
        self.assertGreater(self.request_timeout, 0)
        self.addCleanup(self.session.close)

    def manager(self, role, *, devices=None, port=None, env=None, extra=""):
        manager = MagaServerManager(
            env_args={
                "LOAD_PYTHON_MODEL": "1",
                "USE_NEW_LOADER": "1",
                "REQUIRE_WEIGHT_UPDATE": "0",
                "LOAD_METHOD": "scratch",
                "LOG_LEVEL": "DEBUG",
                "FT_SERVER_TEST": "1",
                "WORLD_SIZE": "1",
                **(env or {}),
            },
            device_ids=devices or self.devices[:1],
            port=port,
            role_name=role,
            smoke_args_str=(
                "--tp_size 1 --ep_size 1 --dp_size 1 --world_size 1 "
                "--role_type PDFUSION "
                "--use_new_loader true --require_weight_update false --load_method scratch "
                "--act_type BF16 --max_seq_len 2048 --max_context_batch_size 4 "
                "--concurrency_limit 4 --seq_size_per_block 64 --kernel_seq_size_per_block 64 "
                "--kv_cache_mem_mb 2048 --use_local true --use_local_preprocess true "
                "--cache_store_rdma_mode 0 --load_cache_timeout_ms 120000 "
                "--reuse_cache true --frontend_server_count 1 --warm_up false "
                "--shutdown_timeout 10 --frontend_pre_stop_drain_seconds 0 "
                "--dash_sc_grpc_pre_stop_drain_seconds 0 " + extra
            ),
        )
        self.addCleanup(manager.stop_server)
        return manager

    def start(self, manager):
        self.assertTrue(
            manager.start_server(self.checkpoint, self.model_type, timeout=3600)
        )
        log = Path(manager.log_file_path).read_text(errors="replace")
        self.assertIn(f"loader route for model_type={self.model_type}: newloader", log)
        # Request DEBUG after startup too: server initialization can reset it.
        response = self.session.post(
            f"http://127.0.0.1:{manager.port}/set_log_level",
            json={"log_level": "DEBUG"},
            timeout=30,
        )
        response.raise_for_status()

    def request(self, manager, label, prompt, *, pd=False):
        response = self.session.post(
            f"http://127.0.0.1:{manager.port}/",
            json={
                "prompt": prompt,
                "generate_config": {
                    "max_new_tokens": 24,
                    "min_new_tokens": 8,
                    "top_k": 1,
                    "top_p": 1.0,
                    "random_seed": 42,
                    "reuse_cache": False,
                    "can_use_pd_separation": pd,
                    "return_output_ids": True,
                    "return_input_ids": True,
                    "aux_info": True,
                },
            },
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        result = response.json()
        (self.outputs / f"{label}.json").write_text(json.dumps(result))
        self.assertNotIn("error", result)
        self.assertTrue(result["finished"])
        self.assertEqual(len(result["output_ids"]), 1)
        self.assertGreaterEqual(len(result["output_ids"][0]), 8)
        self.assertEqual(result["aux_info"]["pd_sep"], pd)
        return result

    def equal_tokens(self, expected, actual):
        self.assertEqual(expected["input_ids"], actual["input_ids"])
        self.assertEqual(expected["output_ids"], actual["output_ids"])


class SparseMlaPDSmoke(GenerationContract, unittest.TestCase):
    def test_indexer_cache_transfer_matches_fused_service(self):
        self.assertGreaterEqual(torch.cuda.device_count(), 2)
        self.assertGreater(
            self.config["index_topk"], 0, "Checkpoint must enable the sparse indexer"
        )
        from transformers import AutoTokenizer

        # Exceed top-k so sparse selection is meaningful, not an all-token
        # shortcut that can hide an invalid indexer cache after P/D transfer.
        tokenizer = AutoTokenizer.from_pretrained(
            self.checkpoint, trust_remote_code=True, local_files_only=True
        )
        topk = self.config["index_topk"]
        prompt = "Summarize this log: "
        while len(tokenizer.encode(prompt)) <= topk + 128:
            prompt += "The server validates its sparse cache. " * 32
        second_prompt = "Write three key points instead.\n" + prompt
        sequence_capacity = len(tokenizer.encode(prompt)) + 256
        pd_args = (
            "--enable_cuda_graph false --fp8_kv_cache 1 "
            f"--max_seq_len {sequence_capacity}"
        )
        baseline = self.manager("sparse_mla_fused", extra=pd_args)
        try:
            self.start(baseline)
            expected = self.request(baseline, "sparse_fused", prompt)
            self.assertGreater(len(expected["input_ids"][0]), topk)
            second_expected = self.request(
                baseline, "sparse_fused_second", second_prompt
            )
        finally:
            baseline.stop_server()
        prefill_port = MagaServerManager.get_free_port()
        decode_port = MagaServerManager.get_free_port()
        route = ServiceRoute(
            service_id="newloader_sparse_contract",
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
        )
        managers = []
        for index, (role, port, peer) in enumerate(
            (
                ("PREFILL", prefill_port, decode_port),
                ("DECODE", decode_port, prefill_port),
            )
        ):
            managers.append(
                self.manager(
                    f"sparse_mla_{role.lower()}",
                    devices=self.devices[index : index + 1],
                    port=port,
                    env={
                        "MODEL_SERVICE_CONFIG": route.model_dump_json(),
                        "REMOTE_RPC_SERVER_IP": "localhost",
                        "REMOTE_SERVER_PORT": peer,
                    },
                    extra=f"--role_type {role} {pd_args}",
                )
            )
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(self.start, manager) for manager in managers]
                for future in futures:
                    future.result()
            actual = self.request(managers[0], "sparse_pd", prompt, pd=True)
            self.equal_tokens(expected, actual)
            repeated = self.request(
                managers[0], "sparse_pd_second_transfer", second_prompt, pd=True
            )
            self.equal_tokens(second_expected, repeated)
        finally:
            for manager in managers:
                manager.stop_server()


class MlaGraphShrinkSmoke(GenerationContract, unittest.TestCase):
    model_type = "deepseek2"
    default_checkpoint = "/mnt/nas1/hf/DeepSeek-V2-Lite-Chat"

    def test_shrinking_graph_metadata_and_real_generation(self):
        # Exercise actual native parameter buffers and CUDA graph replay, not
        # a mocked planner. Then check real model generation in a padded graph.
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        params = rtp_llm_ops.FlashInferMlaAttnParams()
        empty = torch.empty(0, dtype=torch.int32)
        lengths = torch.tensor([3, 5, 7, 9], dtype=torch.int32)
        blocks = torch.tensor([[10], [20], [30], [40]], dtype=torch.int32)

        def fill(active, fixed):
            params.fill_params(
                empty,
                lengths[:active],
                torch.ones(active, dtype=torch.int32),
                blocks[:active],
                64,
                forbid_realloc=fixed,
                planned_batch_size=4,
            )

        fill(4, False)
        slots, positions = params.slot_mapping.view(4), params.positions_d.view(4)
        pointers = (slots.data_ptr(), positions.data_ptr())
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            seen_slots, seen_positions = slots.clone(), positions.clone()
        for active in (2, 0, 1, 4):
            fill(active, True)
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(
                seen_slots.tolist(),
                [10 * (i + 1) * 64 + int(lengths[i]) for i in range(active)]
                + [-1] * (4 - active),
            )
            self.assertEqual(
                seen_positions.tolist(), lengths[:active].tolist() + [0] * (4 - active)
            )
            if active:
                self.assertEqual(
                    (params.slot_mapping.data_ptr(), params.positions_d.data_ptr()),
                    pointers,
                )
        prompt = "Explain why the sky looks blue in three sentences."
        baseline = self.manager("mla_eager", extra="--enable_cuda_graph false")
        try:
            self.start(baseline)
            expected = self.request(baseline, "mla_eager", prompt)
        finally:
            baseline.stop_server()
        captured = self.manager(
            "mla_graph",
            extra="--enable_cuda_graph true --decode_capture_config 4",
        )
        try:
            self.start(captured)
            offset = Path(captured.log_file_path).stat().st_size
            actual = self.request(captured, "mla_graph_padded_single_request", prompt)
            self.equal_tokens(expected, actual)
            with Path(captured.log_file_path).open("rb") as log:
                log.seek(offset)
                runtime = log.read().decode(errors="replace")
            self.assertRegex(
                runtime,
                r"using CUDA graph forward, is_target_verify=0, is_prefill=0, graph_bs=4",
            )
        finally:
            captured.stop_server()


class EagleContract(GenerationContract):
    """Actual EAGLE service, never EAGLE-3 or a DSpARK draft substitution."""

    requires_cuda = False
    draft_model_type = "qwen35_dense_mtp"

    def manager(self, role, *, devices=None, port=None, env=None, extra=""):
        if torch.version.hip is not None:
            # AITER batch-prefill ships vectorized kernels for page size 16;
            # the generic CUDA smoke default (64) is not portable to this path.
            extra += " --seq_size_per_block 16 --kernel_seq_size_per_block 16 --use_triton_pa 1"
        return super().manager(role, devices=devices, port=port, env=env, extra=extra)

    def test_eagle_matches_target_and_accepts_draft_tokens(self):
        draft = os.environ.get("EAGLE_CHECKPOINT_PATH", self.checkpoint)
        self.assertTrue((Path(draft) / "config.json").is_file())
        graph_mode = os.environ.get("EAGLE_ENABLE_CUDA_GRAPH", "0")
        self.assertIn(graph_mode, ("0", "1"))
        graph_args = (
            "--enable_cuda_graph true --decode_capture_config '1,2,4' "
            if graph_mode == "1"
            else "--enable_cuda_graph false "
        )
        prompts = [
            "Count from one to ten:",
            "Write a Python function that adds two integers.",
        ]
        baseline = self.manager("eagle_target", extra="--enable_cuda_graph false")
        try:
            self.start(baseline)
            expected = [
                self.request(baseline, f"eagle_target_{i}", prompt)
                for i, prompt in enumerate(prompts)
            ]
        finally:
            baseline.stop_server()
        speculative = self.manager(
            "eagle_draft",
            extra=(
                graph_args + "--sp_type eagle --sp_act_type BF16 "
                "--gen_num_per_cycle 3 --sp_model_type "
                + self.draft_model_type
                + " --sp_checkpoint_path "
                + shlex.quote(draft)
            ),
        )
        try:
            self.start(speculative)
            log = Path(speculative.log_file_path).read_text(errors="replace")
            self.assertIn(
                f"loader route for model_type={self.draft_model_type}: newloader", log
            )
            total_accepted = 0
            request_log_offset = Path(speculative.log_file_path).stat().st_size
            for i, prompt in enumerate(prompts):
                actual = self.request(speculative, f"eagle_speculative_{i}", prompt)
                self.equal_tokens(expected[i], actual)
                aux = actual["aux_info"]
                rounds = aux["speculative_draft_rounds"]
                accepted = aux["speculative_accepted_tokens_per_pos"]
                self.assertGreater(rounds, 1)
                self.assertTrue(accepted)
                self.assertTrue(all(0 <= n <= rounds for n in accepted))
                total_accepted += sum(accepted)
            self.assertGreater(total_accepted, 0, "Draft must accept real tokens")
            if graph_mode == "1":
                with Path(speculative.log_file_path).open("rb") as handle:
                    handle.seek(request_log_offset)
                    self.assertIn(
                        "using CUDA graph forward",
                        handle.read().decode(errors="replace"),
                    )
        finally:
            speculative.stop_server()


class Qwen35DenseEagleSmoke(EagleContract, unittest.TestCase):
    model_type = "qwen35_dense"
    default_checkpoint = "/mnt/nas1/hf/Qwen3.5-0.8B"


class Qwen35MoeEagleSmoke(EagleContract, unittest.TestCase):
    model_type = "qwen35_moe"
    draft_model_type = "qwen35_moe_mtp"
    default_checkpoint = "/mnt/nas1/hf/Qwen3.5-35B-A3B-FP8"


class Qwen3NextEagleSmoke(EagleContract, unittest.TestCase):
    model_type = "qwen3_next"
    draft_model_type = "qwen3_next_mtp"
    default_checkpoint = "/mnt/nas1/hf/Qwen3-Next-80B-A3B-Instruct-FP8"


class Qwen2EagleSmoke(EagleContract, unittest.TestCase):
    model_type = "qwen_2"
    draft_model_type = "qwen_2-mtp"
    default_checkpoint = "/mnt/nas1/mtp_reg/qwen2_14b_target"

    def setUp(self):
        super().setUp()
        self.assertTrue(
            os.environ.get("EAGLE_CHECKPOINT_PATH"),
            "Qwen2 requires EAGLE_CHECKPOINT_PATH for its separately trained draft",
        )


if __name__ == "__main__":
    unittest.main()
