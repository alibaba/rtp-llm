"""Manual full Kimi Linear KDA/FULL interior F800 acceptance on real weights."""

from __future__ import annotations

import hashlib
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


class KimiLinearCommonPrefixTest(unittest.TestCase):
    def setUp(self):
        self.model_path = Path(os.environ["T10_KIMI_LINEAR_CHECKPOINT"])
        config_bytes = (self.model_path / "config.json").read_bytes()
        config = json.loads(config_bytes)
        self.assertEqual(config["num_hidden_layers"], 27)
        linear = config["linear_attn_config"]
        self.assertEqual(
            linear["kda_layers"], [i for i in range(1, 28) if i % 4 != 0 and i != 27]
        )
        self.assertEqual(linear["full_attn_layers"], [4, 8, 12, 16, 20, 24, 27])
        index = json.loads(
            (self.model_path / "model.safetensors.index.json").read_text()
        )
        shards = set(index["weight_map"].values())
        self.assertEqual(len(shards), 20)
        self.assertTrue(all((self.model_path / name).is_file() for name in shards))
        self.output_dir = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        self.numerical_extension = os.environ.get("T10_KIMI_NUMERICAL_EXTENSION") == "1"
        self.both_off = os.environ.get("T10_COMMON_PREFIX_BOTH_OFF") == "1"
        if self.both_off and self.numerical_extension:
            raise AssertionError(
                "both-off cannot run the F800 private numerical reference"
            )
        self.probe_dir = self.output_dir / "kimi_cache_probe"
        (self.output_dir / "checkpoint_manifest.json").write_text(
            json.dumps(
                {
                    "path": str(self.model_path),
                    "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                    "layers": config["num_hidden_layers"],
                    "shards": len(shards),
                    "weight_bytes": index["metadata"].get("total_size"),
                },
                indent=2,
            )
            + "\n"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, local_files_only=True
        )
        self._make_prompts()
        self.session = requests.Session()
        self.session.trust_env = False
        self.addCleanup(self.session.close)
        self._start_servers()

    def _extend_prompt(self, prompt: str, length: int) -> str:
        current = self.tokenizer.encode(prompt)
        while len(current) < length:
            for suffix in (" record", " blue", " and", " one", "."):
                candidate = prompt + suffix
                tokens = self.tokenizer.encode(candidate)
                if (
                    len(tokens) == len(current) + 1
                    and tokens[: len(current)] == current
                ):
                    prompt, current = candidate, tokens
                    break
            else:
                self.fail(f"tokenizer cannot extend stable prompt at {len(current)}")
        self.assertEqual(len(current), length)
        return prompt

    def _make_prompts(self):
        self.prefix = self._extend_prompt("Kimi Linear exact interior fork:", 800)
        self.prompt_a = self._extend_prompt(self.prefix + " record", 1300)
        source_ids = self.tokenizer.encode(self.prompt_a)
        branches = []
        for suffix in (" amber", " violet", " silver", " orange", " green", " blue"):
            prompt = self.prefix + suffix
            ids = self.tokenizer.encode(prompt)
            if (
                len(ids) > 800
                and ids[:800] == source_ids[:800]
                and ids[800] != source_ids[800]
                and all(ids[800] != first for first, _ in branches)
            ):
                branches.append((ids[800], self._extend_prompt(prompt, 2000)))
            if len(branches) == 2:
                break
        self.assertEqual(len(branches), 2, "two strict F800 token branches")
        self.prompt_b, self.prompt_c = branches[0][1], branches[1][1]
        ids_b, ids_c = self.tokenizer.encode(self.prompt_b), self.tokenizer.encode(
            self.prompt_c
        )
        self.assertEqual(len(source_ids), 1300)
        self.assertEqual((len(ids_b), len(ids_c)), (2000, 2000))
        self.assertEqual(ids_b[:800], source_ids[:800])
        self.assertEqual(ids_c[:800], source_ids[:800])
        self.assertEqual(ids_b[:800], ids_c[:800])
        self.assertNotEqual(ids_b[800], source_ids[800])
        self.assertNotEqual(ids_c[800], source_ids[800])
        self.assertNotEqual(ids_b[800], ids_c[800])

    def _start_servers(self):
        cards = list(get_gpu_ids())
        self.assertGreaterEqual(len(cards), 2)
        prefill_port = MagaServerManager.get_free_port()
        decode_port = MagaServerManager.get_free_port()
        route = ServiceRoute(
            service_id="kimi_linear_common_prefix",
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
            "TRITON_AUTOTUNE_CACHE_MODE": "cached",
        }
        common_args = (
            "--act_type BF16 --ssm_state_dtype fp32 --seq_size_per_block 256"
            " --kernel_seq_size_per_block 64 --linear_step 1 --max_seq_len 2048"
            " --max_context_batch_size 1 --concurrency_limit 1 --reuse_cache 1"
            " --load_method fastsafetensors --kv_cache_mem_mb 4096"
            " --reserver_runtime_mem_mb 8192 --cache_store_rdma_mode 0 --use_local 1"
            " --load_cache_timeout_ms 120000 --warm_up 0 --enable_cuda_graph 0"
            " --enable_device_cache 1 --enable_memory_cache 0 --enable_disk_cache 0"
            " --enable_remote_cache 0 --enable_flashinfer_trtllm_gen 0"
            " --tp_size 1 --ep_size 1 --dp_size 1 --world_size 1"
        )
        hook_config = self.output_dir / "kimi_hook.json"
        hook_config.write_text(
            json.dumps(
                {
                    "case": "kimi_linear_common_prefix",
                    "function_hooks": [
                        {
                            "target": "rtp_llm.models_py.model_desc.kimi_linear.KimiLinearModel.forward",
                            "before": "before_model_forward",
                            "after": "after_model_forward",
                        }
                    ],
                }
            )
        )
        self.prefill = MagaServerManager(
            env_args={
                **common_env,
                "WORLD_SIZE": "1",
                "REMOTE_SERVER_PORT": decode_port,
                "RTP_HOT_HOOK": "1",
                "RTP_HOT_HOOK_FILE": str(
                    Path(__file__).with_name("kimi_linear_common_prefix_probe.py")
                ),
                "RTP_HOT_HOOK_CONFIG": str(hook_config),
                "RTP_HOT_HOOK_DUMP_DIR": str(self.probe_dir),
                "T10_KIMI_PROBE_DIR": str(self.probe_dir),
                "T10_KIMI_PROBE_ROLE": "prefill",
                "T10_KIMI_NUMERICAL_EXTENSION": (
                    "1" if self.numerical_extension else "0"
                ),
                "T10_COMMON_PREFIX_BOTH_OFF": "1" if self.both_off else "0",
            },
            device_ids=cards[:1],
            port=prefill_port,
            role_name="kimi_linear_common_prefix_prefill",
            smoke_args_str=(
                common_args
                + " --role_type PREFILL"
                + f" --enable_prefill_partial_tail_reuse {0 if self.both_off else 1}"
                + f" --enable_prefill_common_prefix_recovery {0 if self.both_off else 1}"
            ),
        )
        self.decode = MagaServerManager(
            env_args={
                **common_env,
                "WORLD_SIZE": "1",
                "REMOTE_SERVER_PORT": prefill_port,
            },
            device_ids=cards[1:2],
            port=decode_port,
            role_name="kimi_linear_common_prefix_decode",
            smoke_args_str=common_args + " --role_type DECODE",
        )
        self.addCleanup(self.prefill.stop_server)
        self.addCleanup(self.decode.stop_server)
        with ThreadPoolExecutor(max_workers=2) as executor:
            starts = [
                executor.submit(
                    manager.start_server,
                    model_path=str(self.model_path),
                    model_type="kimi_linear",
                    timeout=3600,
                )
                for manager in (self.prefill, self.decode)
            ]
            for role, start in zip(("prefill", "decode"), starts):
                self.assertTrue(start.result(), f"{role} startup failed")

    def _request(self, label: str, prompt: str, *, reuse: bool = True):
        prior = set(self.probe_dir.glob("prefill_rank*_*.pt"))
        query = {
            "prompt": prompt,
            "generate_config": {
                # PrefillRpcServer only sends generation to DECODE above one token.
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
        started = time.time_ns()
        response = self.session.post(
            f"http://127.0.0.1:{self.prefill.port}/", json=query, timeout=1800
        )
        elapsed_ms = (time.time_ns() - started) / 1e6
        (self.output_dir / f"{label}.http.txt").write_text(response.text)
        response.raise_for_status()
        actual = response.json()
        self.assertNotIn("error_code", actual, actual)
        self.assertIs(actual.get("finished"), True)
        self.assertEqual(len(actual["output_ids"]), 1)
        self.assertEqual(len(actual["output_ids"][0]), 2)
        record = {
            "input_len": len(actual["input_ids"][0]),
            "prefill_reuse_len": actual["aux_info"].get("prefill_total_reuse_len"),
            "decode_reuse_len": actual["aux_info"].get("decode_total_reuse_len"),
            "aggregate_reuse_len": actual["aux_info"].get("reuse_len"),
            "pd_sep": actual["aux_info"].get("pd_sep"),
            "request_wall_ms": elapsed_ms,
            "output_ids": actual["output_ids"],
        }
        (self.output_dir / f"{label}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        probes = {}
        for path in set(self.probe_dir.glob("prefill_rank*_*.pt")) - prior:
            data = torch.load(path, weights_only=False)
            key = f"{data['phase']}_rank{data['rank']}"
            self.assertNotIn(key, probes)
            probes[key] = data
        return record, probes

    def _values_at(self, row, ids):
        ids = ids.to(torch.long)
        positions = torch.searchsorted(row["ids"], ids)
        self.assertTrue((positions < row["ids"].numel()).all())
        self.assertTrue(torch.equal(row["ids"][positions], ids))
        return row["values"][positions]

    def _selected_ids(self, row, endpoint):
        points, blocks = row["save_positions"], row["save_block_ids"]
        self.assertIsNotNone(points)
        self.assertIsNotNone(blocks)
        selected = blocks[points == endpoint]
        self.assertEqual(selected.numel(), 1)
        self.assertGreater(int(selected[0]), 0)
        pages = row["seq_size_per_block"] // row["kernel_seq_size_per_block"]
        return int(selected[0]) * pages + torch.arange(pages)

    def _ready_source_ids(self, row, endpoint):
        """Use selected state when present, otherwise the original runtime alias."""
        points, blocks = row["save_positions"], row["save_block_ids"]
        if points is not None and blocks is not None and points.numel():
            selected = blocks[points == endpoint]
            if selected.numel() == 1 and int(selected[0]) > 0:
                return self._selected_ids(row, endpoint)
        pages = row["seq_size_per_block"] // row["kernel_seq_size_per_block"]
        slot = (endpoint - 1) // row["seq_size_per_block"]
        ids = row["runtime_table"][0, slot * pages : (slot + 1) * pages].long()
        self.assertEqual(ids.numel(), pages)
        self.assertTrue((ids > 0).all())
        return ids

    def _assert_hidden(self, produced, private, start):
        self.assertEqual(produced["prefix"], start)
        self.assertEqual(private["prefix"], start)
        end = produced["end_position"]
        self.assertEqual(private["end_position"], end)
        actual = produced["hidden_states"].float()
        expected = private["hidden_states"].float()
        self.assertEqual(actual.shape, expected.shape)
        positions = produced["position_ids"].reshape(-1)
        self.assertEqual(
            produced["position_ids_source"],
            "sequential_single_request_from_prefix_and_input_lengths",
        )
        self.assertEqual(
            produced["position_ids_source"], private["position_ids_source"]
        )
        self.assertTrue(torch.equal(positions, private["position_ids"].reshape(-1)))
        self.assertTrue(torch.equal(positions, torch.arange(start, end)))
        delta = (actual - expected).abs()
        unequal = actual != expected
        (
            self.output_dir / f"private_G{start}_N{end}_hidden_comparison.json"
        ).write_text(
            json.dumps(
                {
                    "tokens": end - start,
                    "atol": 0,
                    "rtol": 0,
                    "maxabs": float(delta.max()),
                    "fraction_unequal": float(unequal.float().mean()),
                    "private_no_store": True,
                },
                indent=2,
            )
            + "\n"
        )
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def _assert_numerical_extension(self, private, start, end):
        if not self.numerical_extension:
            return
        projection = private.get("resolved_head_projection")
        self.assertIsNotNone(projection)
        self.assertFalse(projection["native_cpp_logits_capture"])
        self.assertEqual(projection["model_layers"], 27)
        self.assertEqual((projection["tp_size"], projection["tp_rank"]), (1, 0))
        self.assertEqual((projection["atol"], projection["rtol"]), (0, 0))
        self.assertEqual(projection["absolute_positions"], list(range(start, end)))
        self.assertEqual(projection["excluded_padding_rows"], 0)
        self.assertEqual(
            projection["compared_element_count"],
            (end - start)
            * (projection["local_vocab_range"][1] - projection["local_vocab_range"][0]),
        )
        layers = private.get("canonical_layer_comparisons")
        self.assertIsNotNone(layers)
        self.assertEqual(set(layers), set(range(27)))
        for layer_id, comparison in layers.items():
            self.assertEqual(
                comparison["source_components"], ["hidden_states", "residual"]
            )
            self.assertEqual(comparison["shape"][0], end - start)
            self.assertEqual((comparison["atol"], comparison["rtol"]), (0, 0))
            self.assertEqual(
                comparison["compared_element_count"],
                (end - start) * comparison["shape"][1],
            )
            self.assertTrue(
                comparison["passes_original_tolerance"],
                f"layer {layer_id}: {comparison}",
            )
        (self.output_dir / f"kimi_numerical_G{start}_N{end}.json").write_text(
            json.dumps(
                {"resolved_head_projection": projection, "canonical_layers": layers},
                indent=2,
            )
            + "\n"
        )
        self.assertTrue(projection["passes_original_tolerance"], projection)

    def _test_both_off(self):
        a, a_probes = self._request("both_off_a1300", self.prompt_a)
        b, b_probes = self._request("both_off_b2000", self.prompt_b)
        c, c_probes = self._request("both_off_c2000", self.prompt_c)
        for record, length, reuse in ((a, 1300, 0), (b, 2000, 768), (c, 2000, 768)):
            self.assertEqual(
                (record["input_len"], record["prefill_reuse_len"]), (length, reuse)
            )
            self.assertIs(record["pd_sep"], True)
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
            self.assertEqual(len(snapshot["rows"]), 27)
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

        a_rows = snapshots[0]["rows"]
        self.assertEqual(sum(key[1] == "linear" for key in a_rows), 20)
        self.assertEqual(sum(key[1] == "full" for key in a_rows), 7)
        for key, source_row in a_rows.items():
            if key[1] == "linear":
                ids = self._ready_source_ids(source_row, 768)
            else:
                ids = source_row["runtime_table"][
                    0, : 768 // source_row["kernel_seq_size_per_block"]
                ].long()
            self.assertGreater(ids.numel(), 0)
            self.assertTrue((ids > 0).all())
            original = self._values_at(source_row, ids).contiguous().view(torch.uint8)
            for snapshot in snapshots[1:]:
                actual = (
                    self._values_at(snapshot["rows"][key], ids)
                    .contiguous()
                    .view(torch.uint8)
                )
                self.assertTrue(
                    torch.equal(original, actual), f"{key}: G768 source bytes changed"
                )

        cold_b, _ = self._request("both_off_b2000_cold", self.prompt_b, reuse=False)
        cold_c, _ = self._request("both_off_c2000_cold", self.prompt_c, reuse=False)
        for actual, cold in ((b, cold_b), (c, cold_c)):
            self.assertEqual((cold["input_len"], cold["prefill_reuse_len"]), (2000, 0))
            self.assertIs(cold["pd_sep"], True)
            self.assertEqual(actual["output_ids"], cold["output_ids"])
        (self.output_dir / "kimi_both_off_summary.json").write_text(
            json.dumps(
                {
                    "flags": {"F": 0, "N": 0},
                    "A": a,
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

    def test_internal_f800_full_kda(self):
        if self.both_off:
            self._test_both_off()
            return
        a, a_probes = self._request("a1300", self.prompt_a)
        b, b_probes = self._request("b2000", self.prompt_b)
        c, c_probes = self._request("c2000", self.prompt_c)
        self.assertEqual((a["input_len"], a["prefill_reuse_len"]), (1300, 0))
        self.assertEqual((b["input_len"], b["prefill_reuse_len"]), (2000, 768))
        self.assertEqual((c["input_len"], c["prefill_reuse_len"]), (2000, 800))
        self.assertTrue(all(record["pd_sep"] for record in (a, b, c)))
        self.assertEqual(set(a_probes), {"after_rank0"})
        self.assertEqual(set(b_probes), {"before_rank0", "after_rank0"})
        self.assertEqual(set(c_probes), {"before_rank0", "after_rank0"})
        a_rows = a_probes["after_rank0"]["rows"]
        b_rows = b_probes["after_rank0"]["rows"]
        c_before = c_probes["before_rank0"]["rows"]
        c_after = c_probes["after_rank0"]["rows"]
        self.assertEqual(set(a_rows), set(b_rows))
        self.assertEqual(set(a_rows), set(c_before))
        self.assertEqual(len(a_rows), 27)
        linear_keys = {key for key in a_rows if key[1] == "linear"}
        full_keys = {key for key in a_rows if key[1] == "full"}
        self.assertEqual(len(linear_keys), 20)
        self.assertEqual(len(full_keys), 7)
        reference_dir = self.output_dir / "private_reference"
        private_a = torch.load(reference_dir / "rank0_G0_N1300.pt", weights_only=False)
        private_b = torch.load(
            reference_dir / "rank0_G768_N2000.pt", weights_only=False
        )
        private_c = torch.load(
            reference_dir / "rank0_G800_N2000.pt", weights_only=False
        )
        for stage, reference in enumerate((private_a, private_b, private_c)):
            self.assertEqual(reference["private_stage"], stage)
            self.assertEqual(reference["production_cache_bytes_copied"], 0)
            self.assertTrue(reference["cache_store_isolation"])
            self.assertTrue(
                all(
                    not row["reference_store_inputs"]
                    and not row["reference_store_writer"]
                    for row in reference["cache_store_isolation"]
                )
            )
        self.assertEqual(set(private_b["state_rows"]), linear_keys)
        self.assertEqual(set(private_a["state_rows"]), linear_keys)
        self._assert_hidden(a_probes["after_rank0"], private_a, 0)
        self._assert_hidden(b_probes["after_rank0"], private_b, 768)
        self._assert_hidden(c_probes["after_rank0"], private_c, 800)
        for reference, start, end in (
            (private_a, 0, 1300),
            (private_b, 768, 2000),
            (private_c, 800, 2000),
        ):
            self._assert_numerical_extension(reference, start, end)
        for key in linear_keys:
            source_ids = self._ready_source_ids(a_rows[key], 768)
            source = self._values_at(a_rows[key], source_ids)
            self.assertTrue(
                torch.equal(
                    source.view(torch.uint8),
                    private_a["state_rows"][key].view(torch.uint8),
                ),
                f"{key}: original G768 runtime state differs from independent private model",
            )
            for rows in (b_rows, c_before, c_after):
                self.assertTrue(
                    torch.equal(
                        source.view(torch.uint8),
                        self._values_at(rows[key], source_ids).view(torch.uint8),
                    ),
                    f"{key}: A source bytes changed",
                )
            f_ids = self._selected_ids(b_rows[key], 800)
            n_ids = self._selected_ids(b_rows[key], 2000)
            self.assertFalse(
                torch.isin(f_ids, n_ids).any(), f"{key}: F/N share destination"
            )
            produced_f = self._values_at(b_rows[key], f_ids)
            expected_f = private_b["state_rows"][key]
            self.assertEqual(produced_f.shape, expected_f.shape)
            self.assertTrue(
                torch.equal(produced_f.view(torch.uint8), expected_f.view(torch.uint8)),
                f"{key}: F800 conv+SSM differs from independent private model",
            )
            self.assertTrue(
                torch.equal(
                    produced_f.view(torch.uint8),
                    self._values_at(c_after[key], f_ids).view(torch.uint8),
                ),
                f"{key}: B F800 source mutated during C",
            )
            pages = (
                b_rows[key]["seq_size_per_block"]
                // b_rows[key]["kernel_seq_size_per_block"]
            )
            slot = (800 - 1) // b_rows[key]["seq_size_per_block"]
            runtime_ids = c_before[key]["runtime_table"][
                0, slot * pages : (slot + 1) * pages
            ].long()
            self.assertTrue((runtime_ids > 0).all())
            self.assertFalse(torch.isin(runtime_ids, f_ids).any())
            self.assertTrue(
                torch.equal(
                    produced_f.view(torch.uint8),
                    self._values_at(c_before[key], runtime_ids).view(torch.uint8),
                ),
                f"{key}: C restored F800 bytes differ",
            )
        for key in full_keys:
            row = a_rows[key]
            blocks = b_rows[key]["save_block_ids"]
            if blocks is not None:
                self.assertFalse((blocks > 0).any(), f"{key}: FULL unexpected snapshot")
            pages = (1300 + row["kernel_seq_size_per_block"] - 1) // row[
                "kernel_seq_size_per_block"
            ]
            ids = row["runtime_table"][0, :pages].long()
            self.assertTrue((ids > 0).all())
            original = self._values_at(row, ids)
            for rows in (b_rows, c_after):
                self.assertTrue(
                    torch.equal(
                        original.view(torch.uint8),
                        self._values_at(rows[key], ids).view(torch.uint8),
                    ),
                    f"{key}: FULL A source bytes changed",
                )
        cold, _ = self._request("c2000_cold", self.prompt_c, reuse=False)
        self.assertEqual(cold["prefill_reuse_len"], 0)
        self.assertEqual(c["output_ids"], cold["output_ids"])


if __name__ == "__main__":
    unittest.main()
