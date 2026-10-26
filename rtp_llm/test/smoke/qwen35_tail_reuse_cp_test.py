"""Manual real-checkpoint Qwen3.5 CP2 prefill tail-reuse acceptance."""

from __future__ import annotations

import json
import os
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch
from transformers import AutoTokenizer

from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class Qwen35TailReuseCPTest(unittest.TestCase):
    full_layer = False
    cp_size = 2

    def setUp(self):
        self.model_path = Path(os.environ["T09_QWEN35_CHECKPOINT"])
        self.assertTrue((self.model_path / "config.json").is_file())
        config = json.loads((self.model_path / "config.json").read_text())
        self.assertEqual(
            config["text_config"]["layer_types"][:3],
            ["linear_attention", "linear_attention", "linear_attention"],
        )
        self.full_layer = (
            self.full_layer or os.environ.get("T10_QWEN35_CP_FULL_LAYER") == "1"
        )
        self.layer_count = 4 if self.full_layer else 3
        if self.full_layer:
            self.assertEqual(config["text_config"]["layer_types"][3], "full_attention")
        self.output_dir = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        self.probe_dir = self.output_dir / "linear_cache_probe"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, local_files_only=True
        )
        self.session = requests.Session()
        self.session.trust_env = False
        self.addCleanup(self.session.close)
        self._start_servers()

    def _start_servers(self):
        cards = list(get_gpu_ids())
        self.assertGreaterEqual(len(cards), self.cp_size + 1)
        prefill_port = MagaServerManager.get_free_port()
        decode_port = MagaServerManager.get_free_port()
        route = ServiceRoute(
            service_id=f"qwen35_tail_reuse_cp{self.cp_size}",
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
        common_env = {
            "LOAD_PYTHON_MODEL": "1",
            "LOG_LEVEL": "INFO",
            "FT_SERVER_TEST": "1",
            "REMOTE_RPC_SERVER_IP": "localhost",
            "MODEL_SERVICE_CONFIG": route.model_dump_json(),
        }
        common_args = (
            "--act_type BF16 --seq_size_per_block 256"
            f" --kernel_seq_size_per_block {128 if self.cp_size == 2 else 64}"
            f" --hack_layer_num {self.layer_count} --max_seq_len 2048 --max_context_batch_size 1"
            " --concurrency_limit 1 --reuse_cache 1 --load_method fastsafetensors"
            " --kv_cache_mem_mb 4096 --reserver_runtime_mem_mb 12288"
            " --cache_store_rdma_mode 0 --use_local 1 --load_cache_timeout_ms 120000"
            " --warm_up 0 --enable_device_cache 1 --enable_memory_cache 0"
            " --enable_disk_cache 0 --enable_remote_cache 0"
            f" --enable_flashinfer_trtllm_gen {0 if self.cp_size == 2 else 1}"
        )
        hook_config = self.output_dir / "qwen35_cp_hook_config.json"
        hook_config.write_text(
            json.dumps(
                {
                    "case": "qwen35_cp_linear_cache",
                    "function_hooks": [
                        {
                            "target": "rtp_llm.models_py.model_desc.qwen3_next.Qwen3NextModel.forward",
                            "before": "before_model_forward",
                            "after": "after_model_forward",
                        }
                    ],
                }
            )
        )
        if os.environ.get("T10_GDN_NUMERIC_PROBE") == "1":
            config = json.loads(hook_config.read_text())
            module = "rtp_llm.models_py.model_desc.qwen3_next."
            config["function_hooks"].extend(
                [
                    {
                        "target": module + "Qwen3NextDecoderLayer.forward",
                        "before": "before_decoder",
                        "after": "after_decoder",
                    },
                    {
                        "target": module + "Qwen3NextGatedDeltaNet._input_project",
                        "after": "after_input_project",
                    },
                    {
                        "target": module + "causal_conv1d_fn",
                        "before": "before_conv",
                        "after": "after_conv",
                    },
                    {
                        "target": module + "chunk_gated_delta_rule",
                        "before": "before_gdn_chunk",
                        "after": "after_gdn_chunk",
                    },
                ]
            )
            hook_config.write_text(json.dumps(config))
        probe_file = Path(__file__).with_name("qwen35_tail_reuse_cp_probe.py")
        self.prefill = MagaServerManager(
            env_args={
                **common_env,
                "WORLD_SIZE": str(self.cp_size),
                "REMOTE_SERVER_PORT": decode_port,
                "RTP_HOT_HOOK": "1",
                "RTP_HOT_HOOK_FILE": str(probe_file),
                "RTP_HOT_HOOK_CONFIG": str(hook_config),
                "RTP_HOT_HOOK_DUMP_DIR": str(self.probe_dir),
                "T09_CACHE_PROBE_DIR": str(self.probe_dir),
                "T09_CACHE_PROBE_ROLE": "prefill",
                "T09_CACHE_PROBE_LAYERS": str(self.layer_count),
                "T10_GDN_NUMERIC_PROBE": os.environ.get("T10_GDN_NUMERIC_PROBE", "0"),
            },
            device_ids=cards[: self.cp_size],
            port=prefill_port,
            role_name=f"qwen35_tail_cp{self.cp_size}_prefill",
            smoke_args_str=(
                common_args
                + f" --tp_size {self.cp_size} --ep_size {self.cp_size} --dp_size 1 --world_size {self.cp_size}"
                " --role_type PREFILL --enable_cuda_graph 0 --enable_prefill_partial_tail_reuse 1"
                + (
                    " --moe_strategy fp8_per_block_pure_cp --use_all_gather 1"
                    " --prefill_cp_kv_cache_sharded 1 --prefill_cp_size 2 --cp_rotate_method ALL_GATHER"
                    if self.cp_size == 2
                    else ""
                )
            ),
        )
        self.decode = MagaServerManager(
            env_args={
                **common_env,
                "WORLD_SIZE": "1",
                "REMOTE_SERVER_PORT": prefill_port,
            },
            device_ids=cards[self.cp_size : self.cp_size + 1],
            port=decode_port,
            role_name=f"qwen35_tail_cp{self.cp_size}_decode",
            smoke_args_str=(
                common_args + " --tp_size 1 --ep_size 1 --dp_size 1 --world_size 1"
                " --enable_flashinfer_trtllm_gen 0"
                " --role_type DECODE --enable_cuda_graph 0"
                + (
                    " --prefill_cp_kv_cache_sharded 1 --prefill_cp_size 2 --cp_rotate_method PREFILL_CP"
                    if self.cp_size == 2
                    else ""
                )
            ),
        )
        self.addCleanup(self.prefill.stop_server)
        self.addCleanup(self.decode.stop_server)
        with ThreadPoolExecutor(max_workers=2) as executor:
            starts = [
                executor.submit(
                    manager.start_server,
                    model_path=str(self.model_path),
                    model_type="qwen35_moe",
                    timeout=3600,
                )
                for manager in (self.prefill, self.decode)
            ]
            for role, start in zip(("prefill", "decode"), starts):
                self.assertTrue(start.result(), f"{role} startup failed")

    def _extend_prompt(self, prompt: str, length: int) -> str:
        current = len(self.tokenizer.encode(prompt))
        while current < length:
            prompt += " record"
            next_length = len(self.tokenizer.encode(prompt))
            self.assertEqual(next_length, current + 1)
            current = next_length
        self.assertEqual(current, length)
        return prompt

    def _request(
        self, label: str, prompt: str, reuse: bool = True
    ) -> tuple[dict, dict[str, dict]]:
        prior_probes = set(self.probe_dir.glob("prefill_rank*_*.pt"))
        query = {
            "prompt": prompt,
            "generate_config": {
                "max_new_tokens": 1,
                "ignore_eos": True,
                "top_k": 1,
                "top_p": 1.0,
                "random_seed": 7,
                "reuse_cache": reuse,
                "enable_device_cache": True,
                "enable_memory_cache": False,
                "enable_disk_cache": False,
                "enable_remote_cache": False,
                "can_use_pd_separation": self.cp_size == 2,
                "return_input_ids": True,
                "return_output_ids": True,
                "aux_info": True,
            },
        }
        started = time.time_ns()
        response = self.session.post(
            f"http://127.0.0.1:{self.prefill.port}/", json=query, timeout=1200
        )
        ended = time.time_ns()
        (self.output_dir / f"{label}.http.txt").write_text(response.text)
        response.raise_for_status()
        actual = response.json()
        self.assertNotIn("error_code", actual, actual)
        self.assertIs(actual.get("finished"), True)
        self.assertTrue(actual["output_ids"][0])
        record = {
            "input_len": len(actual["input_ids"][0]),
            "reuse_len": actual["aux_info"].get("reuse_len"),
            "pd_sep": actual["aux_info"].get("pd_sep"),
            "request_wall_ms": (ended - started) / 1e6,
            "output_ids": actual["output_ids"],
        }
        (self.output_dir / f"{label}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        probe_files = set(self.probe_dir.glob("prefill_rank*_*.pt")) - prior_probes
        probes: dict[str, dict] = {}
        for path in probe_files:
            data = torch.load(path, weights_only=False)
            key = f"{data['phase']}_rank{data['rank']}"
            self.assertNotIn(key, probes, f"{label}: duplicate probe {key}")
            probes[key] = data
        return record, probes

    def _assert_owner_copy(
        self, source: dict[str, dict], reused: dict[str, dict]
    ) -> None:
        self.assertEqual(set(source), {f"after_rank{r}" for r in range(self.cp_size)})
        self.assertEqual(
            set(reused),
            {
                f"{phase}_rank{r}"
                for r in range(self.cp_size)
                for phase in ("before", "after")
            },
        )
        for rank in range(self.cp_size):
            a = source[f"after_rank{rank}"]
            b_before = reused[f"before_rank{rank}"]
            b_after = reused[f"after_rank{rank}"]
            self.assertEqual((a["prefix"], a["end_position"]), (0, 1300))
            self.assertEqual(
                (b_before["prefix"], b_before["end_position"]), (1300, 2000)
            )
            self.assertEqual((b_after["prefix"], b_after["end_position"]), (1300, 2000))
            for layer in range(3):
                key = (layer, "linear")
                source_row = a["rows"][key]
                before_row = b_before["rows"][key]
                after_row = b_after["rows"][key]
                points = source_row["save_positions"].reshape(-1)
                blocks = source_row["save_block_ids"].reshape(-1)
                selected = blocks[points == 1300]
                self.assertEqual(
                    selected.numel(), 1, f"rank{rank} layer{layer}: source save"
                )
                source_block = int(selected[0])
                self.assertGreater(source_block, 0)
                pages = (
                    source_row["seq_size_per_block"]
                    // source_row["kernel_seq_size_per_block"]
                )
                slot = (1300 - 1) // source_row["seq_size_per_block"]
                destination = before_row["runtime_table"][
                    0, slot * pages : (slot + 1) * pages
                ]
                self.assertEqual(destination.numel(), pages)
                self.assertTrue((destination >= 0).all())
                self.assertNotEqual(int(destination[0]) // pages, source_block)
                for offset, destination_id in enumerate(destination.tolist()):
                    source_id = source_block * pages + offset
                    source_value = source_row["values"][source_row["ids"] == source_id]
                    copied_value = before_row["values"][
                        before_row["ids"] == destination_id
                    ]
                    self.assertEqual(source_value.shape[0], 1)
                    self.assertEqual(copied_value.shape[0], 1)
                    self.assertTrue(
                        torch.equal(
                            source_value.view(torch.uint8),
                            copied_value.view(torch.uint8),
                        ),
                        f"rank{rank} layer{layer}: owner copy differs",
                    )
                immutable_ids = source_row["ids"]
                if self.cp_size == 1:
                    # Saved LINEAR checkpoints are tree-owned. A's transient
                    # runtime block may be released after publication.
                    saved = blocks[blocks > 0].to(torch.long)
                    immutable_ids = torch.unique(
                        (saved[:, None] * pages + torch.arange(pages)).reshape(-1)
                    )
                source_indices = torch.searchsorted(after_row["ids"], immutable_ids)
                expected_indices = torch.searchsorted(source_row["ids"], immutable_ids)
                self.assertTrue((source_indices < after_row["ids"].numel()).all())
                self.assertTrue((expected_indices < source_row["ids"].numel()).all())
                self.assertTrue(
                    torch.equal(after_row["ids"][source_indices], immutable_ids)
                )
                self.assertTrue(
                    torch.equal(source_row["ids"][expected_indices], immutable_ids)
                )
                self.assertTrue(
                    torch.equal(
                        after_row["values"][source_indices].view(torch.uint8),
                        source_row["values"][expected_indices].view(torch.uint8),
                    ),
                    f"rank{rank} layer{layer}: source mutated",
                )
                save_points = after_row["save_positions"].reshape(-1)
                save_blocks = after_row["save_block_ids"].reshape(-1)
                # CP2 uses a 512-token complete-key stride here; 1536 is a
                # runtime block and 1792 is not a complete-key endpoint.
                saved = save_blocks[save_points == 2000]
                self.assertEqual(
                    saved.numel(), 1, f"rank{rank} layer{layer}: missing tail 2000"
                )
                self.assertGreater(int(saved[0]), 0)
                self.assertFalse(((save_points == 1556) | (save_points == 1812)).any())

    def _assert_full_source_immutable(self, source, reused):
        for rank in range(self.cp_size):
            a = source[f"after_rank{rank}"]["rows"]
            before = reused[f"before_rank{rank}"]["rows"]
            after = reused[f"after_rank{rank}"]["rows"]
            full_keys = [key for key in a if key[0] == 3]
            self.assertEqual(len(full_keys), 1)
            key = full_keys[0]
            self.assertNotEqual(key[1], "linear")
            src, dst, end = a[key], before[key], after[key]
            base = src["seq_size_per_block"]
            kernel = src["kernel_seq_size_per_block"]
            self.assertEqual(base % kernel, 0)
            pages_per_block = base // kernel
            tail_slot = ((1300 - 1) // base) // self.cp_size
            # FULL publishes dense runtime resources. Only LINEAR needs
            # selected state destinations; an absent FULL destination is not
            # evidence of missing KV production.
            for row in (src, dst, end):
                saved = row["save_block_ids"]
                if saved is not None:
                    self.assertFalse((saved >= 0).any())
            for global_block in range((1300 + base - 1) // base):
                if global_block % self.cp_size != rank:
                    continue
                slot = global_block // self.cp_size
                valid = min(base, 1300 - global_block * base)
                for offset in range((valid + kernel - 1) // kernel):
                    page = slot * pages_per_block + offset
                    count = min(kernel, valid - offset * kernel)
                    source_id = int(src["runtime_table"][0, page])
                    destination_id = int(dst["runtime_table"][0, page])
                    self.assertGreaterEqual(source_id, 0)
                    self.assertGreaterEqual(destination_id, 0)
                    if slot < tail_slot:
                        self.assertEqual(source_id, destination_id)
                    else:
                        # CP continuation replaces the common logical tail ID
                        # on every rank, including a rank whose local tail is full.
                        self.assertNotEqual(source_id, destination_id)
                    values = []
                    for row, physical_id in (
                        (src, source_id),
                        (dst, destination_id),
                        (end, source_id),
                    ):
                        value = row["values"][row["ids"] == physical_id]
                        self.assertEqual(value.shape[0], 1)
                        values.append(
                            value[:, :, :, :count].contiguous().view(torch.uint8)
                        )
                    self.assertTrue(
                        torch.equal(values[0], values[1]),
                        f"rank{rank}: FULL restored prefix differs",
                    )
                    self.assertTrue(
                        torch.equal(values[0], values[2]),
                        f"rank{rank}: FULL published source mutated",
                    )

    def _assert_hidden_suffix(self, reused, cold, *, label="same_split", required=True):
        def rows(probes):
            result = {}
            for rank in range(self.cp_size):
                probe = probes[f"after_rank{rank}"]
                hidden = probe["hidden_states"]
                positions = probe["position_ids"]
                self.assertIsNotNone(hidden)
                self.assertIsNotNone(positions)
                self.assertEqual(positions.numel() % hidden.shape[0], 0)
                axes = positions.numel() // hidden.shape[0]
                self.assertIn(axes, (1, 3))
                positions = positions.reshape(hidden.shape[0], axes)
                # Text-only Qwen3.5 mRoPE stores interleaved equal T/H/W axes.
                self.assertTrue(
                    torch.equal(positions, positions[:, :1].expand_as(positions))
                )
                positions = positions[:, 0]
                for pos, value in zip(positions.tolist(), hidden):
                    if 1300 <= pos < 2000:
                        self.assertNotIn(pos, result)
                        result[pos] = value.float()
            self.assertEqual(set(result), set(range(1300, 2000)))
            return torch.stack([result[pos] for pos in range(1300, 2000)])

        actual, expected = rows(reused), rows(cold)
        delta = (actual - expected).abs()
        outside = delta > 1e-2 + 1e-2 * expected.abs()
        (self.output_dir / f"full_linear_{label}_hidden_comparison.json").write_text(
            json.dumps(
                {
                    "tokens": 700,
                    "rtol": 1e-2,
                    "atol": 1e-2,
                    "maxabs": float(delta.max()),
                    "meanabs": float(delta.mean()),
                    "fraction_outside_original_tolerance": float(
                        outside.float().mean()
                    ),
                    "passes_original_tolerance": not bool(outside.any()),
                    "required_reference": required,
                    "reference": (
                        "independent whole-model no-save private runtime: 1300 then 700"
                        if label == "same_split"
                        else "unsplit cold whole-model: 2000"
                    ),
                },
                indent=2,
            )
            + "\n"
        )

        if required:
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_cp2_g1300_to_2000_prefill(self):
        prompt_a = self._extend_prompt("Qwen35 CP2 tail reuse:", 1300)
        prompt_b = self._extend_prompt(prompt_a, 2000)
        self.assertEqual(
            self.tokenizer.encode(prompt_b)[:1300], self.tokenizer.encode(prompt_a)
        )
        source, source_probes = self._request("source_1300", prompt_a)
        reused, reused_probes = self._request("reused_2000", prompt_b)
        self.assertEqual(source["input_len"], 1300)
        self.assertEqual(reused["input_len"], 2000)
        self.assertIs(reused["pd_sep"], False)
        self.assertEqual(reused["reuse_len"], 1300)
        self._assert_owner_copy(source_probes, reused_probes)
        if self.full_layer:
            self._assert_full_source_immutable(source_probes, reused_probes)
        if self.full_layer:
            reference_dir = self.output_dir / "private_reference"
            references = {}
            for rank in range(self.cp_size):
                path = reference_dir / f"rank{rank}_G1300_N2000.pt"
                reference = torch.load(path, weights_only=False)
                self.assertEqual(reference["production_cache_bytes_copied"], 0)
                self.assertEqual(reference["selected_save_slots"], 0)
                self.assertEqual(
                    (reference["prefix"], reference["end_position"]), (1300, 2000)
                )
                references[f"after_rank{rank}"] = reference
        cold, cold_probes = self._request("cold_2000", prompt_b, reuse=False)
        self.assertEqual(cold["input_len"], 2000)
        self.assertEqual(cold["reuse_len"], 0)
        self.assertEqual(cold["output_ids"], reused["output_ids"])
        if self.full_layer:
            # Preserve the independent unsplit diagnostic and its original
            # tolerance. GDN no-save 1300+700 vs 2000 already differs before
            # any save/restore; the strict recovery oracle above shares the
            # two forward boundaries and owns a completely separate KV pool.
            self._assert_hidden_suffix(
                reused_probes,
                cold_probes,
                label="unsplit_cold_diagnostic",
                required=False,
            )
            self._assert_hidden_suffix(reused_probes, references)


class Qwen35FullLinearCPTest(Qwen35TailReuseCPTest):
    full_layer = True


class Qwen35FullLinearNonCPTest(Qwen35TailReuseCPTest):
    full_layer = True
    cp_size = 1

    def test_noncp_g1300_to_2000_prefill(self):
        self.test_cp2_g1300_to_2000_prefill()


if __name__ == "__main__":
    unittest.main()
