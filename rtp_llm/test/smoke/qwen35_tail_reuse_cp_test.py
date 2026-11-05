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
        self.both_off = os.environ.get("T10_COMMON_PREFIX_BOTH_OFF") == "1"
        if self.both_off and (
            self.cp_size != 1
            or self._testMethodName != "test_internal_f800_full_linear"
            or os.environ.get("T10_QWEN35_HEAD_PROJECTION") == "1"
            or os.environ.get("T10_GDN_NUMERIC_PROBE") == "1"
        ):
            raise AssertionError(
                "both-off requires the non-CP F800 fixture without numerical hooks"
            )
        self.head_projection = (
            self._testMethodName == "test_internal_f800_full_linear"
            and os.environ.get("T10_QWEN35_HEAD_PROJECTION") == "1"
        )
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
                "T10_FORK_CP_SIZE": str(self.cp_size),
                "T10_GDN_NUMERIC_PROBE": os.environ.get("T10_GDN_NUMERIC_PROBE", "0"),
                "T10_QWEN35_HEAD_PROJECTION": "1" if self.head_projection else "0",
                "T10_COMMON_PREFIX_BOTH_OFF": "1" if self.both_off else "0",
                "T10_INTERNAL_FORK_STATE_REFERENCE": (
                    "1"
                    if self._testMethodName == "test_internal_f800_full_linear"
                    else "0"
                ),
            },
            device_ids=cards[: self.cp_size],
            port=prefill_port,
            role_name=f"qwen35_tail_cp{self.cp_size}_prefill",
            smoke_args_str=(
                common_args
                + f" --tp_size {self.cp_size} --ep_size {self.cp_size} --dp_size 1 --world_size {self.cp_size}"
                " --role_type PREFILL --enable_cuda_graph 0"
                + f" --enable_prefill_partial_tail_reuse {0 if self.both_off else 1}"
                + (
                    f" --enable_prefill_common_prefix_recovery {0 if self.both_off else 1}"
                    if self._testMethodName.startswith("test_internal_")
                    else ""
                )
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
        self,
        label: str,
        prompt: str,
        reuse: bool = True,
        *,
        pd_separation: bool | None = None,
        max_new_tokens: int = 1,
    ) -> tuple[dict, dict[str, dict]]:
        prior_probes = set(self.probe_dir.glob("prefill_rank*_*.pt"))
        query = {
            "prompt": prompt,
            "generate_config": {
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
                "top_k": 1,
                "top_p": 1.0,
                "random_seed": 7,
                "reuse_cache": reuse,
                "enable_device_cache": True,
                "enable_memory_cache": False,
                "enable_disk_cache": False,
                "enable_remote_cache": False,
                "can_use_pd_separation": (
                    self.cp_size == 2 if pd_separation is None else pd_separation
                ),
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
            "prefill_reuse_len": actual["aux_info"].get("prefill_total_reuse_len"),
            "decode_reuse_len": actual["aux_info"].get("decode_total_reuse_len"),
            "role_addrs": actual["aux_info"].get("role_addrs", []),
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

    def _assert_hidden_suffix(
        self, reused, cold, *, label="same_split", required=True, start=1300
    ):
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
                    if start <= pos < 2000:
                        self.assertNotIn(pos, result)
                        result[pos] = value.float()
            self.assertEqual(set(result), set(range(start, 2000)))
            return torch.stack([result[pos] for pos in range(start, 2000)])

        actual, expected = rows(reused), rows(cold)
        delta = (actual - expected).abs()
        outside = delta > 1e-2 + 1e-2 * expected.abs()
        (self.output_dir / f"full_linear_{label}_hidden_comparison.json").write_text(
            json.dumps(
                {
                    "tokens": 2000 - start,
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
                        else (
                            f"independent private A0→1300, B{512 if self.cp_size == 2 else 768}→2000, C800→2000"
                            if label == "internal_f800_same_boundary"
                            else "unsplit cold whole-model: 2000"
                        )
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

    def _assert_head_projection(self, reference, *, rank, start, end):
        if not self.head_projection:
            return
        projection = reference.get("resolved_head_projection")
        self.assertIsNotNone(
            projection, f"rank{rank} G{start} resolved-head projection absent"
        )
        self.assertFalse(projection["native_cpp_logits_capture"])
        self.assertEqual(projection["model_layers"], 4)
        self.assertEqual(projection["tp_size"], self.cp_size)
        self.assertEqual(projection["tp_rank"], rank)
        self.assertEqual(projection["atol"], 1e-2)
        self.assertEqual(projection["rtol"], 1e-2)
        rows = reference["hidden_states"].shape[0]
        self.assertEqual(projection["gathered_rank_rows"][rank], rows)
        self.assertEqual(len(projection["gathered_rank_rows"]), self.cp_size)
        self.assertEqual(projection["gathered_tp_hidden"], self.cp_size > 1)
        self.assertEqual(projection["absolute_positions"], list(range(start, end)))
        self.assertEqual(
            projection["excluded_padding_rows"],
            sum(projection["gathered_rank_rows"]) - (end - start),
        )
        self.assertEqual(
            projection["compared_element_count"],
            (end - start)
            * (projection["local_vocab_range"][1] - projection["local_vocab_range"][0]),
        )
        (self.output_dir / f"qwen35_head_rank{rank}_G{start}_N{end}.json").write_text(
            json.dumps(projection, indent=2) + "\n"
        )
        self.assertTrue(projection["passes_original_tolerance"], projection)

    def _assert_complete_head_shards(self, references, *, start, end):
        if not self.head_projection:
            return
        self.assertEqual(len(references), self.cp_size)
        projections = [
            reference["resolved_head_projection"] for reference in references
        ]
        vocab = projections[0]["full_vocab_size"]
        self.assertTrue(
            all(projection["full_vocab_size"] == vocab for projection in projections)
        )
        ranges = sorted(projection["local_vocab_range"] for projection in projections)
        cursor = 0
        for begin, finish in ranges:
            self.assertEqual(begin, cursor)
            self.assertGreater(finish, begin)
            cursor = finish
        self.assertEqual(cursor, vocab)
        self.assertEqual(
            sum(projection["compared_element_count"] for projection in projections),
            (end - start) * vocab,
        )

    def _test_internal_f800_both_off(self, prompt_a, prompt_b, prompt_c):
        source, a_probes = self._request("both_off_a1300", prompt_a)
        b, b_probes = self._request("both_off_b2000", prompt_b)
        c, c_probes = self._request("both_off_c2000", prompt_c)
        for record, length, reuse in (
            (source, 1300, 0),
            (b, 2000, 768),
            (c, 2000, 768),
        ):
            self.assertEqual(
                (record["input_len"], record["reuse_len"]), (length, reuse)
            )
            self.assertIs(record["pd_sep"], False)
        self.assertEqual(set(a_probes), {"after_rank0"})
        for probes in (b_probes, c_probes):
            self.assertEqual(set(probes), {"before_rank0", "after_rank0"})
        snapshots = [a_probes["after_rank0"]]
        for probes in (b_probes, c_probes):
            snapshots.extend((probes["before_rank0"], probes["after_rank0"]))
        for snapshot, start, end in zip(
            snapshots, (0, 768, 768, 768, 768), (1300, 2000, 2000, 2000, 2000)
        ):
            self.assertEqual(
                (snapshot["prefix"], snapshot["end_position"]), (start, end)
            )
            self.assertEqual(len(snapshot["rows"]), 4)
            for row in snapshot["rows"].values():
                points, blocks = row["save_positions"], row["save_block_ids"]
                if points is None or blocks is None:
                    self.assertIsNone(points)
                    self.assertIsNone(blocks)
                    continue
                self.assertEqual(points.shape, blocks.shape)
                self.assertFalse(
                    bool((blocks[(points == 800) | (points == 2000)] > 0).any())
                )

        def values_at(row, ids):
            selected = torch.searchsorted(row["ids"], ids)
            self.assertTrue((selected < row["ids"].numel()).all())
            self.assertTrue(torch.equal(row["ids"][selected], ids))
            return row["values"][selected]

        a_rows = snapshots[0]["rows"]
        self.assertEqual(sum(key[1] == "linear" for key in a_rows), 3)
        for key, source_row in a_rows.items():
            base = source_row["seq_size_per_block"]
            kernel = source_row["kernel_seq_size_per_block"]
            pages = base // kernel
            if key[1] == "linear":
                slot = (768 - 1) // base
                ids = source_row["runtime_table"][
                    0, slot * pages : (slot + 1) * pages
                ].long()
            else:
                ids = source_row["runtime_table"][0, : 768 // kernel].long()
            self.assertGreater(ids.numel(), 0)
            self.assertTrue((ids > 0).all())
            original = values_at(source_row, ids).contiguous().view(torch.uint8)
            for snapshot in snapshots[1:]:
                actual = (
                    values_at(snapshot["rows"][key], ids).contiguous().view(torch.uint8)
                )
                self.assertTrue(
                    torch.equal(original, actual), f"{key}: G768 source bytes changed"
                )

        cold_b, _ = self._request("both_off_b2000_cold", prompt_b, reuse=False)
        cold_c, _ = self._request("both_off_c2000_cold", prompt_c, reuse=False)
        for actual, cold in ((b, cold_b), (c, cold_c)):
            self.assertEqual((cold["input_len"], cold["reuse_len"]), (2000, 0))
            self.assertEqual(actual["output_ids"], cold["output_ids"])
        (self.output_dir / "qwen35_both_off_summary.json").write_text(
            json.dumps(
                {
                    "flags": {"F": 0, "N": 0},
                    "A": source,
                    "B": b,
                    "C": c,
                    "cold_B": cold_b,
                    "cold_C": cold_c,
                    "source_layers": len(a_rows),
                },
                indent=2,
            )
            + "\n"
        )

    def test_internal_f800_full_linear(self):
        base = "Qwen35 FULL+LINEAR internal F800:"
        prefix = self._extend_prompt(base, 800)
        prompt_a = self._extend_prompt(prefix, 1300)
        source_ids = self.tokenizer.encode(prompt_a)
        branches = []
        for suffix in (" branch", " amber", " violet", " silver", " orange"):
            candidate = self.tokenizer.encode(prefix + suffix)
            if (
                len(candidate) > 800
                and candidate[:800] == source_ids[:800]
                and candidate[800] != source_ids[800]
                and all(candidate[800] != token for token, _ in branches)
            ):
                branches.append(
                    (candidate[800], self._extend_prompt(prefix + suffix, 2000))
                )
            if len(branches) == 2:
                break
        self.assertEqual(len(branches), 2, "two strict FULL+LINEAR F800 branches")

        if self.both_off:
            self._test_internal_f800_both_off(prompt_a, branches[0][1], branches[1][1])
            return

        source, source_probes = self._request("full_linear_f800_source1300", prompt_a)
        reused_b, b_probes = self._request("full_linear_f800_b2000", branches[0][1])
        b_resume = 512 if self.cp_size == 2 else 768
        self.assertEqual((source["input_len"], source["reuse_len"]), (1300, 0))
        self.assertEqual(
            (reused_b["input_len"], reused_b["reuse_len"]), (2000, b_resume)
        )
        self.assertEqual(
            set(source_probes), {f"after_rank{rank}" for rank in range(self.cp_size)}
        )
        self.assertEqual(
            set(b_probes),
            {
                f"{phase}_rank{rank}"
                for rank in range(self.cp_size)
                for phase in ("before", "after")
            },
        )
        for rank in range(self.cp_size):
            a_rows = source_probes[f"after_rank{rank}"]["rows"]
            b_rows = b_probes[f"after_rank{rank}"]["rows"]
            self.assertEqual(set(a_rows), set(b_rows))
            for key, row in b_rows.items():
                if key[0] == 3:
                    blocks = row["save_block_ids"]
                    if blocks is not None:
                        self.assertFalse((blocks >= 0).any(), "FULL sparse F")
                    continue
                points = row["save_positions"].reshape(-1)
                blocks = row["save_block_ids"].reshape(-1)
                for endpoint in (800, 2000):
                    selected = blocks[points == endpoint]
                    self.assertEqual(selected.numel(), 1, f"rank{rank} {key}: F/N")
                    self.assertGreater(int(selected[0]), 0)
                self.assertNotEqual(
                    int(blocks[points == 800][0]), int(blocks[points == 2000][0])
                )

        reference_dir = self.output_dir / "private_fork_reference"
        head_references = {(0, 1300): [], (b_resume, 2000): [], (800, 2000): []}
        for rank in range(self.cp_size):
            private_a = torch.load(
                reference_dir / f"rank{rank}_G0_N1300.pt", weights_only=False
            )
            private_b = torch.load(
                reference_dir / f"rank{rank}_G{b_resume}_N2000.pt",
                weights_only=False,
            )
            for reference, expected in (
                (private_a, (0, 1300)),
                (private_b, (b_resume, 2000)),
            ):
                self.assertEqual(reference["rank"], rank)
                self.assertEqual(
                    (reference["prefix"], reference["end_position"]), expected
                )
                self.assertEqual(reference["cp_size"], self.cp_size)
                self.assertEqual(reference["production_cache_bytes_copied"], 0)
                self.assertTrue(reference["cache_store_isolation"])
                self.assertTrue(
                    all(
                        not entry["reference_store_inputs"]
                        and not entry["reference_store_writer"]
                        for entry in reference["cache_store_isolation"]
                    )
                )
                self._assert_head_projection(
                    reference, rank=rank, start=expected[0], end=expected[1]
                )
                head_references[expected].append(reference)
            b_rows = b_probes[f"after_rank{rank}"]["rows"]
            self.assertEqual(
                set(private_b["state_rows"]),
                {key for key in b_rows if key[0] != 3},
            )
            self.assertEqual(set(private_b["private_selected_block_ids"]), {"linear"})
            for key, expected in private_b["state_rows"].items():
                row = b_rows[key]
                pages = row["seq_size_per_block"] // row["kernel_seq_size_per_block"]
                points = row["save_positions"].reshape(-1)
                blocks = row["save_block_ids"].reshape(-1)
                selected = blocks[points == 800]
                self.assertEqual(selected.numel(), 1)
                ids = int(selected[0]) * pages + torch.arange(pages)
                positions = torch.searchsorted(row["ids"], ids)
                self.assertTrue((positions < row["ids"].numel()).all())
                self.assertTrue(torch.equal(row["ids"][positions], ids))
                actual = row["values"][positions]
                self.assertEqual(
                    actual.shape, expected.shape, f"rank{rank} {key}: F800 state shape"
                )
                self.assertTrue(
                    torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)),
                    f"rank{rank} {key}: F800 private same-boundary state bytes differ",
                )

        reused_c, c_probes = self._request("full_linear_f800_c2000", branches[1][1])
        self.assertEqual((reused_c["input_len"], reused_c["reuse_len"]), (2000, 800))
        private_c_probes = {}
        for rank in range(self.cp_size):
            for phase in ("before", "after"):
                self.assertEqual(c_probes[f"{phase}_rank{rank}"]["prefix"], 800)
            private_c = torch.load(
                reference_dir / f"rank{rank}_G800_N2000.pt", weights_only=False
            )
            self.assertEqual(private_c["rank"], rank)
            self.assertEqual(
                (private_c["prefix"], private_c["end_position"]), (800, 2000)
            )
            self.assertEqual(private_c["private_stage"], 2)
            self.assertEqual(private_c["cp_size"], self.cp_size)
            self.assertEqual(private_c["production_cache_bytes_copied"], 0)
            self._assert_head_projection(private_c, rank=rank, start=800, end=2000)
            head_references[(800, 2000)].append(private_c)
            self.assertTrue(
                all(
                    not entry["reference_store_inputs"]
                    and not entry["reference_store_writer"]
                    for entry in private_c["cache_store_isolation"]
                )
            )
            if self.cp_size == 2:
                full_copies = [
                    (column, valid)
                    for tag, column, _, _, valid in private_c["private_seed_copies"]
                    if tag == "full"
                ]
                self.assertEqual(
                    full_copies,
                    (
                        [(0, 128), (1, 128), (2, 128), (3, 128)]
                        if rank == 0
                        else [(0, 128), (1, 128), (2, 32)]
                    ),
                )
            private_c_probes[f"after_rank{rank}"] = private_c
        for (start, end), references in head_references.items():
            self._assert_complete_head_shards(references, start=start, end=end)
        self._assert_hidden_suffix(
            c_probes,
            private_c_probes,
            label="internal_f800_same_boundary",
            start=800,
        )
        for rank in range(self.cp_size):
            a_rows = source_probes[f"after_rank{rank}"]["rows"]
            b_rows = b_probes[f"after_rank{rank}"]["rows"]
            c_rows = c_probes[f"after_rank{rank}"]["rows"]
            for key, source_row in a_rows.items():
                if key[0] == 3:
                    continue
                pages = (
                    source_row["seq_size_per_block"]
                    // source_row["kernel_seq_size_per_block"]
                )
                saved = source_row["save_block_ids"].reshape(-1)
                ids = torch.unique(
                    (saved[saved > 0, None] * pages + torch.arange(pages)).reshape(-1)
                )
                source_positions = torch.searchsorted(source_row["ids"], ids)
                self.assertTrue((source_positions < source_row["ids"].numel()).all())
                self.assertTrue(torch.equal(source_row["ids"][source_positions], ids))
                original = source_row["values"][source_positions]
                for rows in (b_rows, c_rows):
                    row = rows[key]
                    positions = torch.searchsorted(row["ids"], ids)
                    self.assertTrue((positions < row["ids"].numel()).all())
                    self.assertTrue(torch.equal(row["ids"][positions], ids))
                    self.assertTrue(
                        torch.equal(
                            original.view(torch.uint8),
                            row["values"][positions].view(torch.uint8),
                        ),
                        f"rank{rank} {key}: A selected state changed",
                    )
            full_key = next(key for key in a_rows if key[0] == 3)
            source_full = a_rows[full_key]
            produced_full = b_rows[full_key]
            before_full = c_probes[f"before_rank{rank}"]["rows"][full_key]
            after_full = c_rows[full_key]
            base = source_full["seq_size_per_block"]
            kernel = source_full["kernel_seq_size_per_block"]
            pages = base // kernel
            for global_block in range((800 + base - 1) // base):
                if global_block % self.cp_size != rank:
                    continue
                slot = global_block // self.cp_size
                valid = min(base, 800 - global_block * base)
                for offset in range((valid + kernel - 1) // kernel):
                    page = slot * pages + offset
                    count = min(kernel, valid - offset * kernel)
                    source_id = int(source_full["runtime_table"][0, page])
                    # CP2 restores the shared FULL prefix from A even where
                    # B recomputed those pages after its G512 continuation.
                    producer = (
                        source_full
                        if self.cp_size == 2 or global_block * base < b_resume
                        else produced_full
                    )
                    producer_id = int(producer["runtime_table"][0, page])
                    restored_id = int(before_full["runtime_table"][0, page])
                    self.assertGreater(source_id, 0)
                    self.assertGreater(producer_id, 0)
                    self.assertGreater(restored_id, 0)
                    source_value = source_full["values"][
                        source_full["ids"] == source_id
                    ]
                    producer_value = producer["values"][producer["ids"] == producer_id]
                    restored_value = before_full["values"][
                        before_full["ids"] == restored_id
                    ]
                    source_after = after_full["values"][after_full["ids"] == source_id]
                    self.assertEqual(source_value.shape[0], 1)
                    self.assertEqual(producer_value.shape[0], 1)
                    self.assertEqual(restored_value.shape[0], 1)
                    self.assertEqual(source_after.shape[0], 1)
                    produced_bytes = (
                        producer_value[..., :count, :].contiguous().view(torch.uint8)
                    )
                    self.assertTrue(
                        torch.equal(
                            produced_bytes,
                            restored_value[..., :count, :]
                            .contiguous()
                            .view(torch.uint8),
                        ),
                        f"rank{rank} FULL page {page}: C restored producer prefix differs",
                    )
                    self.assertTrue(
                        torch.equal(
                            source_value[..., :count, :].contiguous().view(torch.uint8),
                            source_after[..., :count, :].contiguous().view(torch.uint8),
                        ),
                        f"rank{rank} FULL page {page}: A source mutated",
                    )
        cold_f, cold_f_probes = self._request(
            "full_linear_f800_independent_state", prefix, reuse=False
        )
        self.assertEqual((cold_f["input_len"], cold_f["reuse_len"]), (800, 0))
        cold_state_diagnostic = {}
        for rank in range(self.cp_size):
            independent_rows = cold_f_probes[f"after_rank{rank}"]["rows"]
            b_rows = b_probes[f"after_rank{rank}"]["rows"]
            for key, produced in b_rows.items():
                if key[0] == 3:
                    continue
                independent = independent_rows[key]
                selected_values = []
                for row in (produced, independent):
                    pages = (
                        row["seq_size_per_block"] // row["kernel_seq_size_per_block"]
                    )
                    if row["save_positions"] is None:
                        # Cold F has no optional save; use its real terminal block.
                        slot = (800 - 1) // row["seq_size_per_block"]
                        ids = (
                            row["runtime_table"]
                            .reshape(-1)[slot * pages : (slot + 1) * pages]
                            .long()
                        )
                    else:
                        points = row["save_positions"].reshape(-1)
                        blocks = row["save_block_ids"].reshape(-1)
                        saved = blocks[points == 800]
                        self.assertEqual(
                            saved.numel(), 1, f"rank{rank} {key}: F state save"
                        )
                        self.assertGreater(int(saved[0]), 0)
                        ids = int(saved[0]) * pages + torch.arange(pages)
                    self.assertEqual(ids.numel(), pages)
                    self.assertTrue((ids > 0).all())
                    positions = torch.searchsorted(row["ids"], ids)
                    self.assertTrue((positions < row["ids"].numel()).all())
                    self.assertTrue(torch.equal(row["ids"][positions], ids))
                    selected_values.append(row["values"][positions])
                produced_state, cold_state = selected_values
                finite = torch.isfinite(produced_state) & torch.isfinite(cold_state)
                cold_state_diagnostic[f"rank{rank}:{key}"] = {
                    "byte_equal": bool(
                        torch.equal(
                            produced_state.view(torch.uint8),
                            cold_state.view(torch.uint8),
                        )
                    ),
                    "byte_mismatch": int(
                        (
                            produced_state.view(torch.uint8)
                            != cold_state.view(torch.uint8)
                        ).sum()
                    ),
                    "finite_common_maxabs": (
                        float(
                            (
                                produced_state[finite].double()
                                - cold_state[finite].double()
                            )
                            .abs()
                            .max()
                        )
                        if finite.any()
                        else None
                    ),
                    "nonfinite_pattern_mismatch": int(
                        (
                            torch.isfinite(produced_state) != torch.isfinite(cold_state)
                        ).sum()
                    ),
                }
        self.assertTrue(cold_state_diagnostic, "no F800 LINEAR state rows")
        (self.output_dir / "full_linear_f800_independent_state.json").write_text(
            json.dumps(
                {
                    "reference": "cold 0-to-800 runtime state; different compute boundary",
                    "acceptance": "diagnostic only; same-boundary private whole-model oracle is separate",
                    "layers": cold_state_diagnostic,
                },
                indent=2,
            )
            + "\n"
        )
        cold_c, cold_probes = self._request(
            "full_linear_f800_c_cold", branches[1][1], reuse=False
        )
        self.assertEqual(cold_c["reuse_len"], 0)
        self.assertEqual(reused_c["output_ids"], cold_c["output_ids"])
        self._assert_hidden_suffix(
            c_probes,
            cold_probes,
            label="internal_f800_unsplit_cold_diagnostic",
            required=False,
            start=800,
        )


class Qwen35FullLinearNonCPTest(Qwen35FullLinearCPTest):
    cp_size = 1

    def test_noncp_g1300_to_2000_prefill(self):
        self.test_cp2_g1300_to_2000_prefill()


if __name__ == "__main__":
    unittest.main()
