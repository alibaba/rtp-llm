"""Four-layer checkpoint save, short-tail restore, and direct state oracle.

This manual acceptance target requires T07_FOUR_LAYER_CHECKPOINT to name a
six-shard four-layer checkpoint and two CUDA13 GPUs. The prefill process
records each layer's last hidden state and a read-only snapshot of the actual
tagged physical cache blocks. Text and token equality are secondary checks.
"""

from __future__ import annotations

import json
import logging
import os
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch
from transformers import AutoTokenizer

from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
    DSV4_KERNEL_ROW_TAGS,
    DSV4_TAGS,
)
from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class DeepSeekV4TailReuseFourLayerTest(unittest.TestCase):
    def setUp(self):
        self.model_path = Path(os.environ["T07_FOUR_LAYER_CHECKPOINT"])
        self.assertTrue((self.model_path / "model.safetensors.index.json").is_file())
        config = json.loads((self.model_path / "config.json").read_text())
        self.assertEqual(config["num_hidden_layers"], 4)
        self.assertEqual(config["compress_ratios"], [0, 0, 4, 128])
        self.output_dir = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        self.probe_dir = self.output_dir / "physical_cache_probe"
        self.trace_dir = self.output_dir / "layer_traces"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, local_files_only=True
        )
        self.session = requests.Session()
        self.session.trust_env = False
        self.addCleanup(self.session.close)
        self._start_servers()

    def _start_servers(self):
        cards = list(get_gpu_ids())
        self.assertGreaterEqual(len(cards), 2)
        prefill_port = MagaServerManager.get_free_port()
        decode_port = MagaServerManager.get_free_port()
        route = ServiceRoute(
            service_id="v4_tail_reuse_four_layer",
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
            "DSV4_MOE_STRATEGY": "mega",
            "DSV4_CHUNK_TOKENS": "2048",
            "WORLD_SIZE": "1",
            "REMOTE_RPC_SERVER_IP": "localhost",
            "MODEL_SERVICE_CONFIG": route.model_dump_json(),
        }
        common_args = (
            "--tp_size 1 --ep_size 1 --dp_size 1 --world_size 1"
            " --act_type BF16 --fp8_kv_cache 1 --seq_size_per_block 256"
            " --kernel_seq_size_per_block 128 --max_seq_len 2048"
            " --max_context_batch_size 1 --concurrency_limit 1 --reuse_cache 1"
            " --load_method fastsafetensors --kv_cache_mem_mb 4096"
            " --reserver_runtime_mem_mb 12288 --cache_store_rdma_mode 0"
            " --use_local 1 --load_cache_timeout_ms 120000 --warm_up 0"
            " --enable_device_cache 1 --enable_memory_cache 0"
            " --enable_disk_cache 0 --enable_remote_cache 0"
        )
        if self._testMethodName.startswith("test_internal_"):
            # The F800 fixture keeps several branches live; 2 GiB still
            # supplies thousands of blocks while fitting a shared GPU.
            common_args = common_args.replace(
                "--kv_cache_mem_mb 4096", "--kv_cache_mem_mb 2048"
            )
        ac02 = self._testMethodName == "test_internal_ac02_g1100_f1300"
        if ac02:
            # Long C retains content through F1300, while sparse state at
            # canonical 1280 is outside its active tail. C's later N1100
            # checkpoint is therefore B's farthest jointly ready state.
            common_args += " --linear_step 8"
        f800_writer_fault = (
            self._testMethodName == "test_internal_f800_selected_writer_fault_cleanup"
        )
        function_hooks = [
            {
                "target": "rtp_llm.models_py.modules.dsv4.prefill.forward.forward_layers",
                "after": "after_forward_layers",
            },
            {
                "target": "rtp_llm.models_py.modules.dsv4.decode.forward.forward_decode",
                "after": "after_decode_forward",
            },
        ]
        if f800_writer_fault:
            function_hooks[0]["before"] = "before_forward_layers_fault_scope"
            function_hooks.extend(
                [
                    {
                        "target": "rtp_llm.models_py.modules.dsv4.fp8.sparse_save.iter_sparse_save_slots",
                        "replace": "replace_sparse_save_slots_for_f800_fault",
                    },
                    {
                        "target": "rtp_llm.models_py.modules.dsv4.fp8.compressor.run_save_partial_states",
                        "after": "after_f800_selected_state_writer",
                    },
                ]
            )
        cost_probe = os.environ.get("T10_F800_COST_PROBE") == "1"
        if cost_probe:
            self.assertEqual(
                self._testMethodName,
                "test_internal_f800_later_request_uses_exact_saved_state",
                "cost instrumentation requires the ordinary non-CP F800 fixture",
            )
            function_hooks[0]["before"] = "before_forward_layers_cost"
            function_hooks.extend(
                [
                    {
                        "target": "rtp_llm.models_py.modules.dsv4.fp8.sparse_save.iter_sparse_save_slots",
                        "replace": "replace_sparse_save_slots_for_cost",
                    },
                    {
                        "target": "rtp_llm.models_py.modules.dsv4.fp8.compressor.run_save_partial_states",
                        "before": "before_selected_writer_for_cost",
                        "after": "after_selected_writer_for_cost",
                    },
                    {
                        "target": "rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton.quantize_and_insert_k_cache",
                        "before": "before_selected_writer_for_cost",
                        "after": "after_selected_writer_for_cost",
                    },
                ]
            )
        hook_config = self.output_dir / "tail_reuse_hook_config.json"
        hook_config.write_text(
            json.dumps(
                {
                    "case": "v4_four_layer_cache",
                    "function_hooks": function_hooks,
                }
            )
        )
        probe_file = Path(__file__).with_name("deepseek_v4_tail_reuse_probe.py")
        self.assertTrue(probe_file.is_file())
        f1300_private = self._testMethodName == "test_internal_f1300_source2000"
        private_source_end = "1100" if ac02 else ("2000" if f1300_private else "1300")
        private_fork_at = "1300" if ac02 or f1300_private else "800"
        private_b_resume = "1100" if ac02 else ("1280" if f1300_private else "768")
        self.prefill = MagaServerManager(
            env_args={
                **common_env,
                "REMOTE_SERVER_PORT": decode_port,
                "MOEDBG": "1",
                "MOEDBG_ALL_LAYERS": "1",
                "MOEDBG_MAX_SEQ": "2000",
                "MOEDBG_DIR": str(self.trace_dir),
                "MOEDBG_CASE": "four_layer",
                "RTP_HOT_HOOK": "1",
                "RTP_HOT_HOOK_FILE": str(probe_file),
                "RTP_HOT_HOOK_CONFIG": str(hook_config),
                "RTP_HOT_HOOK_DUMP_DIR": str(self.probe_dir),
                "T07_CACHE_PROBE_DIR": str(self.probe_dir),
                "T07_CACHE_PROBE_ROLE": "prefill",
                "T10_F800_COST_PROBE": "1" if cost_probe else "0",
                "ENABLE_MODEL_INPUTS_LOG": "1" if cost_probe else "0",
                "T10_PRIVATE_FORK_REFERENCE": (
                    "1" if self._testMethodName.startswith("test_internal_") else "0"
                ),
                "T10_PRIVATE_SOURCE_END": private_source_end,
                "T10_PRIVATE_FORK_AT": private_fork_at,
                "T10_PRIVATE_B_RESUME": private_b_resume,
                "T10_PRIVATE_SKIP_FIRST_B": "1" if ac02 else "0",
                "T10_F800_WRITER_FAULT_ARM_FILE": (
                    str(self.probe_dir / "f800_writer_fault.arm")
                    if f800_writer_fault
                    else ""
                ),
                "RTP_LLM_PREFIX_CACHE_OBSERVABILITY": (
                    "true"
                    if ac02 or f800_writer_fault
                    else "false"
                ),
            },
            device_ids=cards[:1],
            port=prefill_port,
            role_name="v4_tail_prefill",
            smoke_args_str=(
                common_args
                + " --role_type PREFILL --enable_cuda_graph 0"
                + " --enable_prefill_partial_tail_reuse "
                + (
                    "0"
                    if self._testMethodName
                    == "test_internal_f800_only_later_request_uses_exact_saved_state"
                    else "1"
                )
                + (
                    " --enable_prefill_common_prefix_recovery 1"
                    if self._testMethodName.startswith("test_internal_")
                    else ""
                )
            ),
        )
        self.decode = MagaServerManager(
            env_args={
                **common_env,
                "REMOTE_SERVER_PORT": prefill_port,
                "RTP_HOT_HOOK": "1",
                "RTP_HOT_HOOK_FILE": str(probe_file),
                "RTP_HOT_HOOK_CONFIG": str(hook_config),
                "RTP_HOT_HOOK_DUMP_DIR": str(self.probe_dir),
                "T07_CACHE_PROBE_DIR": str(self.probe_dir),
                "T07_CACHE_PROBE_ROLE": "decode",
            },
            device_ids=cards[1:2],
            port=decode_port,
            role_name="v4_tail_decode",
            smoke_args_str=common_args + " --role_type DECODE --enable_cuda_graph 0",
        )
        self.addCleanup(self.prefill.stop_server)
        self.addCleanup(self.decode.stop_server)
        with ThreadPoolExecutor(max_workers=2) as executor:
            starts = [
                executor.submit(
                    manager.start_server,
                    model_path=str(self.model_path),
                    model_type="deepseek_v4",
                    timeout=3600,
                )
                for manager in (self.prefill, self.decode)
            ]
            for role, start in zip(("prefill", "decode"), starts):
                self.assertTrue(start.result(), f"{role} server startup failed")

    def _extend_prompt(self, prompt: str, length: int) -> str:
        current = len(self.tokenizer.encode(prompt))
        while current < length:
            prompt += " record"
            next_length = len(self.tokenizer.encode(prompt))
            self.assertEqual(next_length, current + 1, "one-token prompt extension")
            current = next_length
        self.assertEqual(current, length)
        return prompt

    def _fork_prompts(self, label: str, fork_at: int, source_len: int = 1300):
        prefix = self._extend_prompt(f"Tail reuse {label} branch:", fork_at)
        prefix_ids = self.tokenizer.encode(prefix)
        for left, right in ((" red", " blue"), (" alpha", " beta"), (" east", " west")):
            left_ids = self.tokenizer.encode(prefix + left)
            right_ids = self.tokenizer.encode(prefix + right)
            if (
                len(left_ids) == fork_at + 1
                and len(right_ids) == fork_at + 1
                and left_ids[:fork_at] == right_ids[:fork_at] == prefix_ids
                and left_ids[fork_at] != right_ids[fork_at]
            ):
                prompt_a = self._extend_prompt(prefix + left, source_len)
                prompt_b = self._extend_prompt(prefix + right, 2000)
                a_ids = self.tokenizer.encode(prompt_a)
                b_ids = self.tokenizer.encode(prompt_b)
                first_difference = next(
                    index for index, (a, b) in enumerate(zip(a_ids, b_ids)) if a != b
                )
                self.assertEqual(first_difference, fork_at)
                return prompt_a, prompt_b, a_ids, b_ids
        self.fail(f"{label}: no one-token fork at position {fork_at}")

    def _request(self, label: str, prompt: str, reuse: bool, expect_error=False):
        before_probe = set(self.probe_dir.glob("prefill_*.pt"))
        before_decode = set(self.probe_dir.glob("decode_*.json"))
        before_trace = set(self.trace_dir.glob("four_layer/rank*_step*.pt"))
        query = {
            "prompt": prompt,
            "generate_config": {
                # PrefillRpcServer uses P→D only when max_new_tokens > 1.
                "max_new_tokens": 2,
                "ignore_eos": True,
                "top_k": 1,
                "top_p": 1.0,
                "random_seed": 7,
                "reuse_cache": reuse,
                "enable_device_cache": True,
                "enable_memory_cache": False,
                "enable_disk_cache": False,
                "enable_remote_cache": False,
                "can_use_pd_separation": True,
                "return_input_ids": True,
                "return_output_ids": True,
                "aux_info": True,
            },
        }
        request_start_ns = time.time_ns()
        response = self.session.post(
            f"http://127.0.0.1:{self.prefill.port}/", json=query, timeout=1200
        )
        response_end_ns = time.time_ns()
        (self.output_dir / f"{label}.http.txt").write_text(response.text)
        if expect_error:
            try:
                error = response.json()
            except ValueError:
                error = {}
            if not isinstance(error, dict):
                error = {}
            self.assertTrue(
                response.status_code >= 400
                or error.get("error_code") not in (None, 0)
                or error.get("finished") is False,
                f"{label}: injected writer exception was not surfaced: "
                f"HTTP {response.status_code} {response.text[:500]}",
            )
            return response
        response.raise_for_status()
        actual = response.json()
        self.assertNotIn("error_code", actual, actual)
        self.assertIs(actual.get("finished"), True)
        self.assertEqual(
            len(actual["input_ids"][0]), len(self.tokenizer.encode(prompt))
        )
        self.assertTrue(actual["output_ids"][0])
        probes = set(self.probe_dir.glob("prefill_*.pt")) - before_probe
        decode_files = set(self.probe_dir.glob("decode_*.json")) - before_decode
        traces = set(self.trace_dir.glob("four_layer/rank*_step*.pt")) - before_trace
        self.assertEqual(len(probes), 1, f"{label}: physical cache probe files")
        self.assertEqual(len(traces), 1, f"{label}: per-layer trace files")
        self.assertTrue(decode_files, f"{label}: P→D decode table probe missing")
        probe = torch.load(next(iter(probes)), weights_only=False)
        trace = torch.load(next(iter(traces)), weights_only=False)
        decode_tables = [json.loads(path.read_text()) for path in sorted(decode_files)]
        (self.output_dir / f"{label}.json").write_text(
            json.dumps(
                {
                    "input_len": len(actual["input_ids"][0]),
                    "request_start_unix_ns": request_start_ns,
                    "response_end_unix_ns": response_end_ns,
                    "prefill_reuse_len": actual["aux_info"].get(
                        "prefill_total_reuse_len"
                    ),
                    "output_ids": actual["output_ids"],
                    "trace_extra": {
                        key: str(value) for key, value in trace["extra"].items()
                    },
                    "physical_group_count": len(probe["rows"]),
                    "decode_tables": decode_tables,
                },
                indent=2,
            )
        )
        return actual, probe, trace, decode_tables

    def _assert_registered_source(
        self, key, source, after, source_end, endpoint, label
    ):
        if key[1] in DSV4_KERNEL_ROW_TAGS:
            # Compressed FULL still has dense prefix coverage in raw-token
            # coordinates: it publishes runtime pages, never selected snapshots.
            self.assertFalse((source["save_block_ids"] >= 0).any(), f"{label} {key}")
            self.assertFalse(
                (after["current_save_block_ids"] >= 0).any(), f"{label} {key}"
            )
            self.assertEqual(
                set(source["source_ids"].tolist()), set(source["table_ids"].tolist())
            )
        else:
            # DSV4 SWA and compression STATE use OPAQUE_STATE ring payloads;
            # they still require precise endpoint snapshots.
            saved = source["save_block_ids"][source["save_positions"] == source_end]
            self.assertEqual(saved.numel(), 1, f"{label} {key}: A short save")
            self.assertGreater(int(saved[0]), 0, f"{label} {key}: A save ID")
            self.assertTrue(
                set(source["source_ids"].tolist()) - set(source["table_ids"].tolist()),
                f"{label} {key}: short-point row absent",
            )
            saved_end = after["current_save_block_ids"][
                after["current_save_positions"] == endpoint
            ]
            self.assertEqual(saved_end.numel(), 1, f"{label} {key}: B short save")
            self.assertGreater(int(saved_end[0]), 0, f"{label} {key}: B save ID")
        for suffix in ("ids", "values", "scales"):
            name = f"source_{suffix}"
            if name in source:
                self.assertTrue(
                    torch.equal(source[name], after[name]),
                    f"{label} {key}: published source mutated: {name}",
                )

    def _assert_pair(
        self,
        label,
        end_position,
        expected_resume,
        source,
        after,
        reused,
        resumed_trace,
        decode_tables,
        require_source_tail=True,
        source_end=1300,
        private_reference=False,
        source_device_stable=True,
    ):
        self.assertEqual(source["end_position"], source_end)
        self.assertEqual(after["end_position"], end_position)
        self.assertEqual(set(source["rows"]), set(after["rows"]))
        self.assertEqual({tag for _, tag in source["rows"]}, set(DSV4_TAGS))
        for key, before_row in source["rows"].items():
            if require_source_tail:
                self._assert_registered_source(
                    key, before_row, after["rows"][key], source_end, end_position, label
                )
            elif source_device_stable:
                for suffix in ("ids", "values", "scales"):
                    name = f"source_{suffix}"
                    if name in before_row:
                        self.assertTrue(
                            torch.equal(before_row[name], after["rows"][key][name]),
                            f"{label} {key}: A source changed: {name}",
                        )

        self.assertIs(reused["aux_info"]["pd_sep"], True)
        reuse_len = reused["aux_info"].get("prefill_total_reuse_len")
        self.assertEqual(reuse_len, expected_resume)
        self.assertEqual(
            int(resumed_trace["extra"]["seq_len_total"]), end_position - expected_resume
        )
        self.assertTrue(decode_tables, f"{label}: P→D decode table absent")
        expected_base_keys = (end_position + 255) // 256
        for decode in decode_tables:
            self.assertEqual(set(decode["tables"]), set(DSV4_TAGS))
            for tag in ("csa_kv", "hca_kv", "indexer_kv"):
                self.assertEqual(
                    len(decode["tables"][tag]["physical_ids"]),
                    expected_base_keys,
                    f"{label}: decode {tag} physical table width",
                )

        reference_file = (
            self.probe_dir
            / f"{'private' if private_reference else 'direct'}_reference_{end_position}_g{expected_resume}.pt"
        )
        if not reference_file.is_file():
            error_file = self.probe_dir / "direct_reference_error.txt"
            self.fail(
                "independent tagged-pool reference missing: "
                + (error_file.read_text() if error_file.is_file() else "no hook error")
            )
        direct = torch.load(reference_file, weights_only=False)
        compared = {}
        comparisons = []
        for layer in range(4):
            name = f"layer{layer:02d}_last"
            resumed = resumed_trace["tensors"][name]
            reference = direct["layers"][layer].float()
            self.assertEqual(resumed.shape, reference.shape, name)
            self.assertTrue(torch.isfinite(resumed).all(), name)
            diff = (resumed - reference).abs()
            compared[name] = float(diff.max())
            comparisons.append((name, resumed, reference))
        for name, direct_key in (
            ("lm_last_hidden", "final_last"),
            ("lm_logits_last", "logits_last"),
        ):
            resumed = resumed_trace["tensors"][name]
            reference = direct[direct_key].float()
            self.assertEqual(resumed.shape, reference.shape, name)
            diff = (resumed - reference).abs()
            compared[name] = float(diff.max())
            comparisons.append((name, resumed, reference))
        (self.output_dir / f"numeric_diffs_{label}.json").write_text(
            json.dumps(compared, indent=2) + "\n"
        )
        for name, resumed, reference in comparisons:
            with self.subTest(scenario=label, tensor=name):
                torch.testing.assert_close(resumed, reference, rtol=0.002, atol=0.002)

    def test_independent_g1300_branches_with_direct_state_oracle(self):
        first_blocks = set()
        for label, end_position in (
            ("b2000", 2000),
            ("b1301", 1301),
            ("b1544", 1544),
            ("b1320", 1320),
        ):
            prompt_a = self._extend_prompt(f"Tail reuse {label} state test:", 1300)
            prompt_b = self._extend_prompt(prompt_a, end_position)
            tokens_a = self.tokenizer.encode(prompt_a)
            tokens_b = self.tokenizer.encode(prompt_b)
            self.assertEqual(tokens_b[:1300], tokens_a)
            first_block = tuple(tokens_a[:256])
            self.assertNotIn(
                first_block, first_blocks, f"{label}: cache namespace reused"
            )
            first_blocks.add(first_block)

            _, source, _, _ = self._request(f"{label}_source_1300", prompt_a, True)
            reused, after, resumed_trace, decode_tables = self._request(
                f"{label}_reused_{end_position}", prompt_b, True
            )
            self._assert_pair(
                label,
                end_position,
                1300,
                source,
                after,
                reused,
                resumed_trace,
                decode_tables,
            )
            if end_position == 2000:
                # Normal full checkpoints alias runtime blocks and need not
                # appear in the optional sparse-save tensor. Consume each
                # point from the completed B path to prove publication.
                for full_point in (1536, 1792):
                    shorter_end = full_point + 1
                    shorter = self._extend_prompt(prompt_a, shorter_end)
                    self.assertEqual(
                        self.tokenizer.encode(shorter), tokens_b[:shorter_end]
                    )
                    resumed, probe, trace, _ = self._request(
                        f"{label}_published_{full_point}", shorter, True
                    )
                    self.assertEqual(probe["end_position"], shorter_end)
                    self.assertIs(resumed["aux_info"]["pd_sep"], True)
                    self.assertEqual(
                        resumed["aux_info"]["prefill_total_reuse_len"], full_point
                    )
                    self.assertEqual(int(trace["extra"]["seq_len_total"]), 1)

    def test_exact_forks_with_direct_state_oracle(self):
        first_blocks = set()
        for label, fork_at, expected_resume in (
            ("fork800", 800, 768),
            ("fork1024", 1024, 1024),
        ):
            prompt_a, prompt_b, tokens_a, tokens_b = self._fork_prompts(label, fork_at)
            self.assertEqual(tokens_a[:fork_at], tokens_b[:fork_at])
            self.assertNotEqual(tokens_a[fork_at], tokens_b[fork_at])
            first_block = tuple(tokens_a[:256])
            self.assertNotIn(
                first_block, first_blocks, f"{label}: cache namespace reused"
            )
            first_blocks.add(first_block)

            _, source, _, _ = self._request(f"{label}_source_1300", prompt_a, True)
            reused, after, resumed_trace, decode_tables = self._request(
                f"{label}_reused_2000", prompt_b, True
            )
            self._assert_pair(
                label,
                2000,
                expected_resume,
                source,
                after,
                reused,
                resumed_trace,
                decode_tables,
            )

    def test_internal_f1300_source2000(self):
        label = "internal_f1300"
        fork_at = 1300
        prompt_a, prompt_b, tokens_a, tokens_b = self._fork_prompts(
            label, fork_at, source_len=2000
        )
        prefix = self._extend_prompt(f"Tail reuse {label} branch:", fork_at)
        prompt_c = None
        for third in (" green", " amber", " violet", " silver"):
            candidate = self.tokenizer.encode(prefix + third)
            if (
                len(candidate) == fork_at + 1
                and candidate[:fork_at] == tokens_a[:fork_at]
                and candidate[fork_at] not in (tokens_a[fork_at], tokens_b[fork_at])
            ):
                prompt_c = self._extend_prompt(prefix + third, 2000)
                break
        self.assertIsNotNone(prompt_c, "no third one-token branch at F1300")
        _, source, _, _ = self._request(f"{label}_source_2000", prompt_a, True)
        reused_b, after_b, trace_b, decode_b = self._request(
            f"{label}_b_2000", prompt_b, True
        )
        self._assert_pair(
            label,
            2000,
            1280,
            source,
            after_b,
            reused_b,
            trace_b,
            decode_b,
            source_end=2000,
            private_reference=True,
        )
        for key, row in after_b["rows"].items():
            positions = row["current_save_positions"].reshape(-1)
            destinations = row["current_save_block_ids"].reshape(-1)
            if key[1] in DSV4_KERNEL_ROW_TAGS:
                self.assertFalse((destinations >= 0).any(), f"{key}: FULL copy")
                continue
            selected_f = destinations[positions == fork_at]
            selected_n = destinations[positions == 2000]
            self.assertEqual(selected_f.numel(), 1, f"{key}: F endpoint")
            self.assertEqual(selected_n.numel(), 1, f"{key}: N endpoint")
            self.assertGreater(int(selected_f[0]), 0)
            self.assertGreater(int(selected_n[0]), 0)
            self.assertNotEqual(int(selected_f[0]), int(selected_n[0]))
        reused_c, after_c, trace_c, decode_c = self._request(
            f"{label}_c_2000", prompt_c, True
        )
        self.assertEqual(reused_c["aux_info"].get("prefill_total_reuse_len"), fork_at)
        self._assert_pair(
            f"{label}_c_reference",
            2000,
            fork_at,
            source,
            after_c,
            reused_c,
            trace_c,
            decode_c,
            source_end=2000,
        )
        private_f = torch.load(
            self.probe_dir / "private_f1300_runtime.pt", weights_only=False
        )
        self.assertEqual(private_f["production_cache_bytes_copied"], 0)
        state_diffs = {}
        for key, saved in after_b["rows"].items():
            selected = saved.get("selected_f1300_values")
            if selected is None:
                continue
            independent = private_f["rows"][key]["values"]
            self.assertEqual(selected.shape, independent.shape, f"{key}: F state shape")
            state_diffs[str(key)] = bool(torch.equal(selected, independent))
        self.assertTrue(state_diffs, "no state-constrained F1300 save rows")
        self.assertTrue(all(state_diffs.values()), state_diffs)
        (self.output_dir / "f1300_state_bytes.json").write_text(
            json.dumps(state_diffs, indent=2) + "\n"
        )
        cold_c, _, _, _ = self._request(f"{label}_c_cold", prompt_c, False)
        self.assertEqual(cold_c["aux_info"].get("prefill_total_reuse_len"), 0)
        self.assertEqual(reused_c["output_ids"], cold_c["output_ids"])

    def test_internal_ac02_g1100_f1300(self):
        label = "internal_ac02"
        fork_a = 800
        fork_c = 1300
        prefix = self._extend_prompt(f"Tail reuse {label} branch:", fork_a)

        def branch(base, position, choices):
            ids = self.tokenizer.encode(base)
            self.assertEqual(len(ids), position)
            options = {}
            for name, suffix in choices.items():
                candidate = base + suffix
                candidate_ids = self.tokenizer.encode(candidate)
                self.assertEqual(len(candidate_ids), position + 1, name)
                self.assertEqual(candidate_ids[:position], ids, name)
                options[name] = (candidate, candidate_ids[position])
            self.assertEqual(
                len({token for _, token in options.values()}), len(options)
            )
            return options

        first = branch(prefix, fork_a, {"a": " red", "c": " blue"})
        prompt_a = self._extend_prompt(first["a"][0], 2000)
        c_common = self._extend_prompt(first["c"][0], fork_c)
        prompt_c1100 = self._extend_prompt(first["c"][0], 1100)
        second = branch(
            c_common,
            fork_c,
            {"c": " green", "b": " amber"},
        )
        prompt_c2000 = self._extend_prompt(second["c"][0], 2000)
        prompt_b = self._extend_prompt(second["b"][0], 2000)
        tokens_a = self.tokenizer.encode(prompt_a)
        tokens_c = self.tokenizer.encode(prompt_c2000)
        tokens_b = self.tokenizer.encode(prompt_b)
        self.assertEqual(tokens_a[:fork_a], tokens_b[:fork_a])
        self.assertNotEqual(tokens_a[fork_a], tokens_b[fork_a])
        self.assertEqual(tokens_c[:fork_c], tokens_b[:fork_c])
        self.assertNotEqual(tokens_c[fork_c], tokens_b[fork_c])
        self.assertEqual(self.tokenizer.encode(prompt_c1100), tokens_c[:1100])

        source_c1100, source, _, _ = self._request(
            f"{label}_c_ready_1100", prompt_c1100, True
        )
        self.assertEqual(source_c1100["aux_info"].get("prefill_total_reuse_len"), 0)
        source_c2000, _, _, _ = self._request(
            f"{label}_c_source_2000", prompt_c2000, True
        )
        self.assertEqual(source_c2000["aux_info"].get("prefill_total_reuse_len"), 1100)
        source_a, _, _, _ = self._request(f"{label}_a_source_2000", prompt_a, True)
        self.assertLess(source_a["aux_info"].get("prefill_total_reuse_len", 0), fork_a)

        reused_b, after_b, trace_b, decode_b = self._request(
            f"{label}_b_2000", prompt_b, True
        )
        self._assert_pair(
            label,
            2000,
            1100,
            source,
            after_b,
            reused_b,
            trace_b,
            decode_b,
            source_end=1100,
            private_reference=True,
        )
        for key, row in after_b["rows"].items():
            positions = row["current_save_positions"].reshape(-1)
            destinations = row["current_save_block_ids"].reshape(-1)
            self.assertEqual(int((positions == fork_a).sum()), 0, f"{key}: earlier F")
            if key[1] in DSV4_KERNEL_ROW_TAGS:
                self.assertFalse((destinations >= 0).any(), f"{key}: FULL copy")
                continue
            selected_f = destinations[positions == fork_c]
            selected_n = destinations[positions == 2000]
            self.assertEqual(selected_f.numel(), 1, f"{key}: F endpoint")
            self.assertEqual(selected_n.numel(), 1, f"{key}: N endpoint")
            self.assertGreater(int(selected_f[0]), 0)
            self.assertGreater(int(selected_n[0]), 0)
            self.assertNotEqual(int(selected_f[0]), int(selected_n[0]))

        private_f = torch.load(
            self.probe_dir / "private_f1300_runtime.pt", weights_only=False
        )
        self.assertEqual(private_f["production_cache_bytes_copied"], 0)
        state_diffs = {}
        for key, row in after_b["rows"].items():
            selected = row.get("selected_f1300_values")
            if selected is None:
                continue
            independent = private_f["rows"][key]["values"]
            self.assertEqual(selected.shape, independent.shape, f"{key}: F shape")
            state_diffs[str(key)] = bool(torch.equal(selected, independent))
        self.assertTrue(state_diffs, "no F1300 state-constrained rows")
        self.assertTrue(all(state_diffs.values()), state_diffs)
        (self.output_dir / "ac02_f1300_state_bytes.json").write_text(
            json.dumps(state_diffs, indent=2) + "\n"
        )
        cold_b, _, _, _ = self._request(f"{label}_b_cold", prompt_b, False)
        self.assertEqual(cold_b["aux_info"].get("prefill_total_reuse_len"), 0)
        self.assertEqual(reused_b["output_ids"], cold_b["output_ids"])

    def test_internal_f800_later_request_uses_exact_saved_state(self):
        self._test_internal_f800_later_request_uses_exact_saved_state(tail_enabled=True)

    def test_internal_f800_only_later_request_uses_exact_saved_state(self):
        self._test_internal_f800_later_request_uses_exact_saved_state(
            tail_enabled=False
        )

    def test_internal_f800_selected_writer_fault_cleanup(self):
        label = "internal_f800_writer_fault"
        prompt_a, prompt_b, tokens_a, tokens_b = self._fork_prompts(label, 800)
        self.assertEqual(tokens_a[:800], tokens_b[:800])
        self._request(f"{label}_source_1300", prompt_a, True)

        arm_file = self.probe_dir / "f800_writer_fault.arm"
        marker_file = arm_file.with_suffix(".json")
        self.assertFalse(arm_file.exists())
        self.assertFalse(marker_file.exists())
        arm_file.write_text("B:G768:N2000:F800\n")
        try:
            failed = self._request(
                f"{label}_b_failed", prompt_b, True, expect_error=True
            )
            failure = {"http_status": failed.status_code, "body": failed.text[:1000]}
        except requests.RequestException as error:
            failure = {"transport_error": str(error)}

        self.assertFalse(arm_file.exists(), "F800 writer fault was not consumed")
        self.assertTrue(
            marker_file.is_file(), "real F800 writer completion marker missing"
        )
        marker = json.loads(marker_file.read_text())
        self.assertEqual(
            (marker["resume"], marker["end_position"], marker["fork"]),
            (768, 2000, 800),
        )
        self.assertEqual(marker["writer"], "compressor.run_save_partial_states")
        self.assertIs(marker["real_writer_returned"], True)
        self.assertIs(marker["cuda_synchronized"], True)
        self.assertTrue(marker["source_unchanged"])
        self.assertTrue(
            all(marker["source_unchanged"].values()),
            f"A source changed during failed B: {marker['source_unchanged']}",
        )

        # The Python exception occurs before NormalExecutor marks the forward
        # complete, so the existing fatal engine path cannot retry in place.
        # Stop only this fixture's server tree and read its finalized log.
        old_prefill, old_decode = self.prefill, self.decode
        old_prefill.stop_server()
        old_decode.stop_server()
        prefill_log = Path(old_prefill.log_file_path).read_text(errors="replace")
        self.assertIn(
            "injected F800 selected-state writer post-CUDA failure", prefill_log
        )
        self.assertIn("SIGABRT", prefill_log, "backend fatal exit was not observed")
        observations = []
        for line in prefill_log.splitlines():
            marker_text = "event=prefix_cache_observation payload="
            if marker_text not in line:
                continue
            payload, _ = json.JSONDecoder().raw_decode(line.split(marker_text, 1)[1])
            observations.append(payload)
        self.assertTrue(observations, "prefix observation was not active")
        f_publications = [
            event
            for event in observations
            if event.get("kind") == "publication"
            and event.get("internal_fork_pos") == 800
            and event.get("insert_attempted") is True
        ]
        self.assertFalse(f_publications, "failed B published its F800 point")
        (self.output_dir / "f800_fatal_forward_evidence.json").write_text(
            json.dumps(
                {
                    "failed_request": failure,
                    "writer_marker": marker,
                    "backend_sigabrt": True,
                    "observations": observations,
                    "f800_insertions": f_publications,
                    "publication_boundary": "NormalExecutor::model_->forward precedes markComplete and completeRegisteredSave",
                },
                indent=2,
            )
            + "\n"
        )

        # MagaServerManager cannot restart after stop_server sets its stop bit.
        # Fresh managers preserve the route/config while using separate logs
        # and probe paths, so no prior DEVICE point or file can satisfy B.
        self.probe_dir = self.output_dir / "physical_cache_probe_fresh"
        self.trace_dir = self.output_dir / "layer_traces_fresh"
        fresh = []
        for old, role in ((old_prefill, "prefill"), (old_decode, "decode")):
            env = dict(old._env_args)
            env["RTP_HOT_HOOK_DUMP_DIR"] = str(self.probe_dir)
            env["T07_CACHE_PROBE_DIR"] = str(self.probe_dir)
            env["T10_F800_WRITER_FAULT_ARM_FILE"] = ""
            if role == "prefill":
                env["MOEDBG_DIR"] = str(self.trace_dir)
                env["T10_F800_FATAL_COLD_REFERENCE"] = "1"
            manager = MagaServerManager(
                env_args=env,
                device_ids=old._device_ids,
                port=old.port,
                role_name=f"v4_tail_{role}_fresh",
                smoke_args_str=old._smoke_args_str,
            )
            self.addCleanup(manager.stop_server)
            fresh.append(manager)
        self.prefill, self.decode = fresh
        with ThreadPoolExecutor(max_workers=2) as executor:
            starts = [
                executor.submit(
                    manager.start_server,
                    model_path=str(self.model_path),
                    model_type="deepseek_v4",
                    timeout=3600,
                )
                for manager in fresh
            ]
            for role, start in zip(("prefill", "decode"), starts):
                self.assertTrue(start.result(), f"fresh {role} startup failed")

        cold, _, trace, decode_tables = self._request(
            f"{label}_b_fresh_cold", prompt_b, False
        )
        self.assertEqual(cold["aux_info"].get("prefill_total_reuse_len"), 0)
        self.assertIs(cold["aux_info"]["pd_sep"], True)
        self.assertTrue(decode_tables)
        independent = torch.load(
            self.probe_dir / "direct_reference_2000_g0.pt", weights_only=False
        )
        differences = {}
        for layer in range(4):
            name = f"layer{layer:02d}_last"
            actual = trace["tensors"][name]
            expected = independent["layers"][layer].float()
            differences[name] = float((actual - expected).abs().max())
            torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)
        for name, reference in (
            ("lm_last_hidden", "final_last"),
            ("lm_logits_last", "logits_last"),
        ):
            actual = trace["tensors"][name]
            expected = independent[reference].float()
            differences[name] = float((actual - expected).abs().max())
            torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)
        (self.output_dir / "f800_fresh_cold_numerical.json").write_text(
            json.dumps(
                {
                    "reuse_len": 0,
                    "output_ids": cold["output_ids"],
                    "maxabs": differences,
                },
                indent=2,
            )
            + "\n"
        )

    def _assert_f800_cost(self):
        cost_path = self.probe_dir / "f800_cost.json"
        self.assertTrue(
            cost_path.is_file(), "production B cost instrumentation missing"
        )
        cost = json.loads(cost_path.read_text())
        samples = cost["samples"]
        for purpose in ("internal_f", "tail_n", "original", "runtime"):
            selected = [row for row in samples if row["purpose"] == purpose]
            self.assertTrue(selected, f"missing writer cost: {purpose}")
            self.assertTrue(all(row["cuda_event_ms"] >= 0 for row in selected))
        f_rows = [row for row in samples if row["purpose"] == "internal_f"]
        self.assertEqual(len(f_rows), 7, "four SWA and three compressor F writers")
        self.assertTrue(all(row["selected_rows"] > 0 for row in f_rows))

        # Existing native logger records the mapping after successful C++ copy
        # completion. Wait for its asynchronous chunk flush, never infer copies
        # from destination occupancy alone.
        records = []
        log_dir = self.output_dir / "v4_tail_prefill_logs" / "model_inputs"
        deadline = time.monotonic() + 15
        matching = []
        while time.monotonic() < deadline:
            records = []
            for path in sorted(log_dir.glob("*.pt")):
                chunk = torch.load(path, map_location="cpu", weights_only=False)
                records.extend(chunk["records"])
            matching = [
                record
                for record in records
                if record["record_type"] == "gpt_model_inputs_snapshot"
                and record["prefix_lengths"] is not None
                and record["prefix_lengths"].reshape(-1).tolist() == [768]
                and record["input_lengths"].reshape(-1).tolist() == [1232]
            ]
            if matching:
                break
            time.sleep(0.25)
        self.assertEqual(len(matching), 1, "unique successful B native copy record")
        self.assertFalse([r for r in records if r["record_type"] == "model_inputs_gap"])
        record = matching[0]
        self.assertEqual(record["dropped_before"], 0)
        tags = record["kv_cache_group_tags"]
        positions = record["cache_save_positions"].reshape(-1)
        ids = record["cache_save_block_ids"]
        self.assertEqual(ids.shape, (len(tags), 1, positions.numel()))
        destinations = {
            tag: {
                int(block)
                for block in ids[row, 0, positions == 800].tolist()
                if block >= 0
            }
            for row, tag in enumerate(tags)
        }
        copies = []
        mapping = record["kv_cache_update_mapping"]
        for tag_row, source, destination in [] if mapping is None else mapping.tolist():
            tag = tags[tag_row]
            layouts = [row for row in cost["physical_layouts"] if row["tag"] == tag]
            self.assertTrue(layouts, f"missing physical copy layout: {tag}")
            copies.append(
                {
                    "tag": tag,
                    "source": source,
                    "destination": destination,
                    "internal_f": destination in destinations[tag],
                    "one_way_copied_bytes": sum(
                        row["kv_block_stride_bytes"] + row["scale_block_stride_bytes"]
                        for row in layouts
                    ),
                }
            )
        cost["executed_copy_mappings"] = copies
        cost["f_one_way_copied_bytes"] = sum(
            row["one_way_copied_bytes"] for row in copies if row["internal_f"]
        )
        cost["f_occupied_tensor_storage_bytes"] = sum(
            len(destinations[row["tag"]])
            * (row["kv_tensor_storage_bytes"] + row["scale_tensor_storage_bytes"])
            for row in cost["physical_layouts"]
        )
        cost["f_occupied_stride_bytes"] = sum(
            len(destinations[row["tag"]])
            * (row["kv_block_stride_bytes"] + row["scale_block_stride_bytes"])
            for row in cost["physical_layouts"]
        )
        cost["copy_accounting"] = (
            "Completed native mapping rows times actual per-layer block strides, including padding and scale planes; "
            "one-way copied bytes, not read+write DRAM traffic; zero means no F copy mapping in this execution. "
            "Tensor storage includes raw-layout padding. Pool occupancy is distinct from copy bytes and from device allocator peak."
        )
        cost["writer_totals"] = {
            purpose: {
                "cuda_event_ms_sum": sum(
                    row["cuda_event_ms"] for row in samples if row["purpose"] == purpose
                ),
                "logical_written_bytes": sum(
                    row["logical_written_bytes"]
                    for row in samples
                    if row["purpose"] == purpose
                ),
            }
            for purpose in ("internal_f", "tail_n", "original", "runtime")
        }
        (self.output_dir / "f800_cost_summary.json").write_text(
            json.dumps(cost, indent=2) + "\n"
        )

    def _test_internal_f800_later_request_uses_exact_saved_state(self, tail_enabled):
        label = "internal_f800"
        fork_at = 800
        prompt_a, prompt_b, tokens_a, tokens_b = self._fork_prompts(label, fork_at)
        prefix = self._extend_prompt(f"Tail reuse {label} branch:", fork_at)
        prompt_c = None
        for third in (" green", " amber", " violet", " silver"):
            candidate = self.tokenizer.encode(prefix + third)
            if (
                len(candidate) == fork_at + 1
                and candidate[:fork_at] == tokens_a[:fork_at]
                and candidate[fork_at] not in (tokens_a[fork_at], tokens_b[fork_at])
            ):
                prompt_c = self._extend_prompt(prefix + third, 2000)
                break
        self.assertIsNotNone(prompt_c, "no third one-token branch at F800")
        tokens_c = self.tokenizer.encode(prompt_c)
        self.assertEqual(tokens_c[:fork_at], tokens_a[:fork_at])
        self.assertNotIn(tokens_c[fork_at], (tokens_a[fork_at], tokens_b[fork_at]))
        _, source, _, _ = self._request(f"{label}_source_1300", prompt_a, True)
        reused_b, after_b, trace_b, decode_b = self._request(
            f"{label}_b_2000", prompt_b, True
        )
        self._assert_pair(
            label,
            2000,
            768,
            source,
            after_b,
            reused_b,
            trace_b,
            decode_b,
            require_source_tail=tail_enabled,
        )
        for key, row in after_b["rows"].items():
            positions = row["current_save_positions"].reshape(-1)
            destinations = row["current_save_block_ids"].reshape(-1)
            if key[1] in DSV4_KERNEL_ROW_TAGS:
                self.assertFalse((destinations >= 0).any(), f"{key}: FULL copy")
                continue
            for endpoint in (800, 2000) if tail_enabled else (800,):
                selected = destinations[positions == endpoint]
                self.assertEqual(selected.numel(), 1, f"{key}: F/N endpoint")
                self.assertGreater(int(selected[0]), 0, f"{key}: F/N destination")
            if tail_enabled:
                self.assertNotEqual(
                    int(destinations[positions == 800][0]),
                    int(destinations[positions == 2000][0]),
                    f"{key}: F and N must have independent state",
                )
            else:
                self.assertEqual(
                    int((positions == 2000).sum()), 0, f"{key}: tail disabled"
                )

        reused_c, after_c, trace_c, decode_c = self._request(
            f"{label}_c_2000", prompt_c, True
        )
        self.assertEqual(reused_c["aux_info"].get("prefill_total_reuse_len"), 800)
        self.assertEqual(int(trace_c["extra"]["seq_len_total"]), 1200)
        self.assertTrue(decode_c)
        private_f = torch.load(
            self.probe_dir / "private_f800_runtime.pt", weights_only=False
        )
        self.assertEqual(private_f["production_cache_bytes_copied"], 0)
        state_diffs = {}
        for key, saved in after_b["rows"].items():
            selected = saved.get("selected_f800_values")
            if selected is None:
                continue
            independent = private_f["rows"][key]["values"]
            self.assertEqual(selected.shape, independent.shape, f"{key}: F state shape")
            delta = (selected.float() - independent.float()).abs()
            state_diffs[str(key)] = {
                "maxabs": float(delta.max()),
                "equal_bytes": bool(torch.equal(selected, independent)),
            }
        (self.output_dir / "f800_state_diffs.json").write_text(
            json.dumps(state_diffs, indent=2) + "\n"
        )
        self.assertTrue(state_diffs, "no state-constrained F800 save rows")
        for key, difference in state_diffs.items():
            self.assertTrue(
                difference["equal_bytes"], f"{key}: private F state differs"
            )
        private_b = torch.load(
            self.probe_dir / "private_reference_2000_g768.pt", weights_only=False
        )
        self.assertEqual(private_b["production_cache_bytes_copied"], 0)
        self.assertEqual(private_b["selected_save_slots"], 1)
        self._assert_pair(
            f"{label}_c_reference",
            2000,
            800,
            source,
            after_c,
            reused_c,
            trace_c,
            decode_c,
            require_source_tail=tail_enabled,
        )
        private = torch.load(
            self.probe_dir / "direct_reference_2000_g800.pt", weights_only=False
        )
        self.assertEqual(private["production_cache_bytes_copied"], 0)
        self.assertEqual(private["selected_save_slots"], 0)
        self.assertEqual(
            private["provenance"],
            "private_no_store_A0_to_1300_B768_to_2000_then_C800_to_2000",
        )
        for key, source_row in source["rows"].items():
            for suffix in ("ids", "values", "scales"):
                name = f"source_{suffix}"
                if name in source_row:
                    self.assertTrue(
                        torch.equal(source_row[name], after_c["rows"][key][name]),
                        f"{key}: A source changed after C: {name}",
                    )
        cold_c, _, _, _ = self._request(f"{label}_c_cold", prompt_c, False)
        self.assertEqual(cold_c["aux_info"].get("prefill_total_reuse_len"), 0)
        self.assertEqual(reused_c["output_ids"], cold_c["output_ids"])
        if os.environ.get("T10_F800_COST_PROBE") == "1":
            self._assert_f800_cost()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
