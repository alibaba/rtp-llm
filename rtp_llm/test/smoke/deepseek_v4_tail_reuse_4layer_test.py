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
        hook_config = self.output_dir / "tail_reuse_hook_config.json"
        hook_config.write_text(
            json.dumps(
                {
                    "case": "v4_four_layer_cache",
                    "function_hooks": [
                        {
                            "target": "rtp_llm.models_py.modules.dsv4.prefill.forward.forward_layers",
                            "after": "after_forward_layers",
                        },
                        {
                            "target": "rtp_llm.models_py.modules.dsv4.decode.forward.forward_decode",
                            "after": "after_decode_forward",
                        },
                    ],
                }
            )
        )
        probe_file = Path(__file__).with_name("deepseek_v4_tail_reuse_probe.py")
        self.assertTrue(probe_file.is_file())
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
            },
            device_ids=cards[:1],
            port=prefill_port,
            role_name="v4_tail_prefill",
            smoke_args_str=(
                common_args
                + " --role_type PREFILL --enable_cuda_graph 0"
                + " --enable_prefill_partial_tail_reuse 1"
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

    def _fork_prompts(self, label: str, fork_at: int):
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
                prompt_a = self._extend_prompt(prefix + left, 1300)
                prompt_b = self._extend_prompt(prefix + right, 2000)
                a_ids = self.tokenizer.encode(prompt_a)
                b_ids = self.tokenizer.encode(prompt_b)
                first_difference = next(
                    index for index, (a, b) in enumerate(zip(a_ids, b_ids)) if a != b
                )
                self.assertEqual(first_difference, fork_at)
                return prompt_a, prompt_b, a_ids, b_ids
        self.fail(f"{label}: no one-token fork at position {fork_at}")

    def _request(self, label: str, prompt: str, reuse: bool):
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
    ):
        self.assertEqual(source["end_position"], 1300)
        self.assertEqual(after["end_position"], end_position)
        self.assertEqual(set(source["rows"]), set(after["rows"]))
        self.assertEqual({tag for _, tag in source["rows"]}, set(DSV4_TAGS))
        for key, before_row in source["rows"].items():
            self._assert_registered_source(
                key, before_row, after["rows"][key], 1300, end_position, label
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
            self.probe_dir / f"direct_reference_{end_position}_g{expected_resume}.pt"
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
