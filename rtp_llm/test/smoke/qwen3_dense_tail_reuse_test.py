"""Real pure-FULL model reuse at an interior token, without sparse writers."""

from __future__ import annotations

import hashlib
import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch
from transformers import AutoTokenizer

from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
from rtp_llm.test.smoke import qwen35_tail_reuse_cp_test as hybrid_smoke
from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class Qwen3DenseTailReuseTest(unittest.TestCase):
    cp_size = 1
    # Share request recording/token-length checks, not the hybrid model oracle.
    _request = hybrid_smoke.Qwen35TailReuseCPTest._request
    _extend_prompt = hybrid_smoke.Qwen35TailReuseCPTest._extend_prompt

    def setUp(self):
        self.model_path = Path(os.environ["T10_QWEN3_CHECKPOINT"])
        config_bytes = (self.model_path / "config.json").read_bytes()
        config = json.loads(config_bytes)
        self.assertEqual(config["model_type"], "qwen3")
        self.layer_count = config["num_hidden_layers"]
        self.output_dir = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        (self.output_dir / "checkpoint_config_manifest.json").write_text(
            json.dumps(
                {
                    "checkpoint": str(self.model_path),
                    "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                    "config": config,
                },
                indent=2,
            )
        )
        self.probe_dir = self.output_dir / "dense_cache_probe"
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
        prefill_port, decode_port = (
            MagaServerManager.get_free_port(),
            MagaServerManager.get_free_port(),
        )
        route = ServiceRoute(
            service_id=f"qwen3_dense_tail_cp{self.cp_size}",
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
        env = {
            "LOAD_PYTHON_MODEL": "1",
            "LOG_LEVEL": "INFO",
            "FT_SERVER_TEST": "1",
            "REMOTE_RPC_SERVER_IP": "localhost",
            "MODEL_SERVICE_CONFIG": route.model_dump_json(),
        }
        common = (
            "--act_type BF16 --seq_size_per_block 256"
            f" --kernel_seq_size_per_block {128 if self.cp_size == 2 else 64}"
            " --max_seq_len 1024 --max_context_batch_size 1 --concurrency_limit 1"
            " --reuse_cache 1 --load_method fastsafetensors --kv_cache_mem_mb 512"
            " --reserver_runtime_mem_mb 2048 --cache_store_rdma_mode 0 --use_local 1"
            " --load_cache_timeout_ms 120000 --warm_up 0 --enable_cuda_graph 0"
            " --enable_device_cache 1 --enable_memory_cache 0 --enable_disk_cache 0"
            " --enable_remote_cache 0"
            f" --enable_flashinfer_trtllm_gen {0 if self.cp_size == 2 else 1}"
        )
        hook_config = self.output_dir / "dense_hook.json"
        hook_config.write_text(
            json.dumps(
                {
                    "case": "pure_full_dense_tail",
                    "function_hooks": [
                        {
                            "target": "rtp_llm.models_py.model_desc.qwen3.Qwen3Model.forward",
                            "before": "before_model_forward",
                            "after": "after_model_forward",
                        }
                    ],
                }
            )
        )
        if os.environ.get("T10_DENSE_NUMERIC_PROBE") == "1":
            hooks = json.loads(hook_config.read_text())
            hooks["function_hooks"].append(
                {
                    "target": "rtp_llm.models_py.modules.factory.attention.cuda_cp_impl.prefill_mha.allgather_cp_impl.PCPAllGatherAttnOp.forward",
                    "after": "after_cp_attention",
                }
            )
            hook_config.write_text(json.dumps(hooks))
        self.prefill = MagaServerManager(
            env_args={
                **env,
                "WORLD_SIZE": str(self.cp_size),
                "REMOTE_SERVER_PORT": decode_port,
                "RTP_HOT_HOOK": "1",
                "RTP_HOT_HOOK_FILE": str(
                    Path(__file__).with_name("qwen3_dense_tail_reuse_probe.py")
                ),
                "RTP_HOT_HOOK_CONFIG": str(hook_config),
                "RTP_HOT_HOOK_DUMP_DIR": str(self.probe_dir),
                "T10_DENSE_PROBE_DIR": str(self.probe_dir),
            },
            device_ids=cards[: self.cp_size],
            port=prefill_port,
            role_name="qwen3_dense_prefill",
            smoke_args_str=(
                common
                + f" --tp_size {self.cp_size} --ep_size 1 --dp_size 1 --world_size {self.cp_size}"
                " --role_type PREFILL"
                + (
                    " --enable_prefill_common_prefix_recovery 1"
                    if self._testMethodName
                    == "test_internal_flag_full_already_coverable"
                    else " --enable_prefill_partial_tail_reuse 1"
                )
                + (
                    " --use_all_gather 1 --prefill_cp_kv_cache_sharded 1"
                    " --prefill_cp_size 2 --cp_rotate_method ALL_GATHER"
                    if self.cp_size == 2
                    else ""
                )
            ),
        )
        self.decode = MagaServerManager(
            env_args={**env, "WORLD_SIZE": "1", "REMOTE_SERVER_PORT": prefill_port},
            device_ids=cards[self.cp_size : self.cp_size + 1],
            port=decode_port,
            role_name="qwen3_dense_decode",
            smoke_args_str=common
            + " --tp_size 1 --ep_size 1 --dp_size 1 --world_size 1 --role_type DECODE",
        )
        for server in (self.prefill, self.decode):
            self.addCleanup(server.stop_server)
        with ThreadPoolExecutor(max_workers=2) as executor:
            starts = [
                executor.submit(
                    server.start_server,
                    model_path=str(self.model_path),
                    model_type="qwen_3",
                    timeout=1200,
                )
                for server in (self.prefill, self.decode)
            ]
            for start in starts:
                self.assertTrue(start.result(), "Qwen3 startup failed")

    def _assert_no_sparse_writer(self, probes):
        self.assertEqual(
            set(probes),
            {
                f"{phase}_rank{rank}"
                for rank in range(self.cp_size)
                for phase in ("before", "after")
            },
        )
        for probe in probes.values():
            self.assertEqual(probe["model_class"], "Qwen3Model")
            self.assertFalse(probe["supports_sparse_cache_save"])
            self.assertEqual(len(probe["rows"]), self.layer_count)
            for row in probe["rows"].values():
                if row["save_block_ids"] is not None:
                    self.assertFalse(
                        (row["save_block_ids"] >= 0).any(),
                        "FULL must not allocate sparse snapshot destinations",
                    )

    def _page(self, row, page_id):
        values = row["values"][row["ids"] == page_id]
        self.assertEqual(values.shape[0], 1)
        self.assertEqual(values.ndim, 5)
        return values[0]

    def _assert_cp_logical_ids(self, source, reused, prefix):
        for key, source_row in source["after_rank0"]["rows"].items():
            base = source_row["seq_size_per_block"]
            pages = base // source_row["kernel_seq_size_per_block"]
            logical_span = base * self.cp_size
            for probes, phase, endpoint in (
                (source, "after", 700),
                (reused, "before", 900),
                (reused, "after", 900),
            ):
                columns = ((endpoint + logical_span - 1) // logical_span) * pages
                reference = probes[f"{phase}_rank0"]["rows"][key]["runtime_table"][
                    0, :columns
                ]
                self.assertEqual(reference.numel(), columns)
                self.assertTrue((reference > 0).all())
                for rank in range(1, self.cp_size):
                    other = probes[f"{phase}_rank{rank}"]["rows"][key]["runtime_table"][
                        0, :columns
                    ]
                    self.assertTrue(
                        torch.equal(reference, other),
                        f"{phase} {key}: CP logical IDs differ across ranks",
                    )
            if prefix % logical_span:
                column = (prefix // logical_span) * pages
                for rank in range(self.cp_size):
                    old = source[f"after_rank{rank}"]["rows"][key]["runtime_table"][
                        0, column
                    ]
                    new = reused[f"before_rank{rank}"]["rows"][key]["runtime_table"][
                        0, column
                    ]
                    # Includes an owner with no valid prefix, e.g. rank1/G256.
                    self.assertNotEqual(
                        int(old),
                        int(new),
                        f"rank{rank} {key}: shared CP tail was not privatized",
                    )

    def _assert_alias_cow_and_immutable(self, source, reused, prefix, source_end=700):
        if self.cp_size == 2:
            self._assert_cp_logical_ids(source, reused, prefix)
        for rank in range(self.cp_size):
            source_rows = source[f"after_rank{rank}"]["rows"]
            before = reused[f"before_rank{rank}"]
            after = reused[f"after_rank{rank}"]
            self.assertEqual((before["prefix"], before["end_position"]), (prefix, 900))
            for key, src in source_rows.items():
                dst, end = before["rows"][key], after["rows"][key]
                base, kernel = (
                    src["seq_size_per_block"],
                    src["kernel_seq_size_per_block"],
                )
                pages = base // kernel
                for global_block in range((source_end + base - 1) // base):
                    if global_block % self.cp_size != rank:
                        continue
                    slot = global_block // self.cp_size
                    for offset in range(pages):
                        token_start = global_block * base + offset * kernel
                        count = min(kernel, source_end - token_start)
                        if count <= 0:
                            continue
                        index = slot * pages + offset
                        source_id = int(src["runtime_table"][0, index])
                        source_value = self._page(src, source_id)[:, :, :count]
                        # Published dense source owns the original runtime pages.
                        self.assertTrue(
                            torch.equal(
                                source_value.contiguous().view(torch.uint8),
                                self._page(end, source_id)[:, :, :count]
                                .contiguous()
                                .view(torch.uint8),
                            ),
                            f"{key}: source valid KV mutated at {token_start}",
                        )
                        reused_count = min(count, prefix - token_start)
                        if reused_count <= 0:
                            continue
                        destination_id = int(dst["runtime_table"][0, index])
                        if global_block // self.cp_size < prefix // (
                            base * self.cp_size
                        ):
                            self.assertEqual(
                                source_id,
                                destination_id,
                                "complete dense history must alias",
                            )
                        else:
                            self.assertNotEqual(
                                source_id,
                                destination_id,
                                "shared partial tail needs private COW",
                            )
                        self.assertTrue(
                            torch.equal(
                                source_value[:, :, :reused_count]
                                .contiguous()
                                .view(torch.uint8),
                                self._page(dst, destination_id)[:, :, :reused_count]
                                .contiguous()
                                .view(torch.uint8),
                            ),
                            f"{key}: restored prefix differs at {token_start}",
                        )

    def _assert_hidden(
        self,
        actual,
        cold,
        prefix,
        label,
        *,
        endpoint=900,
        required=True,
        reference="unsplit cold Qwen3",
    ):
        def collect(probes):
            rows = {}
            for rank in range(self.cp_size):
                p = probes[f"after_rank{rank}"]
                positions = p["position_ids"].reshape(-1)
                hidden = p["hidden_states"]
                self.assertEqual(len(positions), len(hidden))
                for position, value in zip(positions.tolist(), hidden):
                    if prefix <= position < endpoint:
                        self.assertNotIn(position, rows)
                        rows[position] = value.float()
            self.assertEqual(set(rows), set(range(prefix, endpoint)))
            return torch.stack([rows[position] for position in range(prefix, endpoint)])

        actual_hidden, cold_hidden = collect(actual), collect(cold)
        delta = (actual_hidden - cold_hidden).abs()
        (self.output_dir / f"{label}_hidden.json").write_text(
            json.dumps(
                {
                    "prefix": prefix,
                    "tokens": endpoint - prefix,
                    "reference": reference,
                    "required_reference": required,
                    "passes_original_tolerance": bool(
                        torch.all(delta <= 1e-2 + 1e-2 * cold_hidden.abs())
                    ),
                    "fraction_outside_original_tolerance": float(
                        (delta > 1e-2 + 1e-2 * cold_hidden.abs()).float().mean()
                    ),
                    "rtol": 1e-2,
                    "atol": 1e-2,
                    "maxabs": float(delta.max()),
                    "meanabs": float(delta.mean()),
                },
                indent=2,
            )
        )
        if required:
            torch.testing.assert_close(actual_hidden, cold_hidden, rtol=1e-2, atol=1e-2)

    def _private_reference(self, prefix, endpoint):
        references = {}
        for rank in range(self.cp_size):
            path = (
                self.output_dir
                / "private_reference"
                / f"rank{rank}_G{prefix}_N{endpoint}.pt"
            )
            reference = torch.load(path, weights_only=False)
            self.assertEqual(
                (reference["prefix"], reference["end_position"]), (prefix, endpoint)
            )
            self.assertEqual(reference["source_endpoint"], 700)
            self.assertEqual(reference["layer_count"], self.layer_count)
            manifest = json.loads(
                (self.output_dir / "checkpoint_config_manifest.json").read_text()
            )
            self.assertEqual(
                reference["checkpoint_config_sha256"], manifest["config_sha256"]
            )
            self.assertEqual(reference["production_cache_bytes_copied"], 0)
            self.assertEqual(reference["selected_save_slots"], 0)
            self.assertTrue(reference["cache_store_isolation"])
            for group in reference["cache_store_isolation"]:
                self.assertFalse(group["reference_store_inputs"])
                self.assertFalse(group["reference_store_writer"])
            references[f"after_rank{rank}"] = reference
        return references

    def test_interior_fork_and_tail_extension(self):
        prefix = self._extend_prompt("Dense FULL reuse source:", 650)
        source_prompt = self._extend_prompt(prefix, 700)
        branch_prompt = self._extend_prompt(prefix + " branch", 900)
        extension_prompt = self._extend_prompt(source_prompt, 900)
        prefix256 = self._extend_prompt("Dense FULL reuse source:", 256)
        branch256 = self._extend_prompt(prefix256 + " branch", 900)
        source_ids = self.tokenizer.encode(source_prompt)
        branch_ids = self.tokenizer.encode(branch_prompt)
        self.assertEqual(source_ids[:650], branch_ids[:650])
        self.assertNotEqual(source_ids[650], branch_ids[650])
        branch256_ids = self.tokenizer.encode(branch256)
        self.assertEqual(source_ids[:256], branch256_ids[:256])
        self.assertNotEqual(source_ids[256], branch256_ids[256])
        source, source_probes = self._request("source700", source_prompt)
        self.assertEqual((source["input_len"], source["reuse_len"]), (700, 0))
        self._assert_no_sparse_writer(source_probes)
        if self.cp_size == 2:
            self._assert_hidden(
                source_probes,
                self._private_reference(0, 700),
                0,
                "source700_independent",
                endpoint=700,
                reference="independent no-store A700 whole-model",
            )
        # Do reused requests before cold runs can add another branch to the tree.
        results = []
        for label, prompt, expected_g in (
            ("fork650", branch_prompt, 650),
            ("extend700", extension_prompt, 700),
            # CP2: raw base256 boundary is inside logical512. Rank0's
            # complete local block must still follow the shared COW ID.
            ("fork256", branch256, 256),
        ):
            result, probes = self._request(label, prompt)
            self.assertEqual(
                (result["input_len"], result["reuse_len"]), (900, expected_g)
            )
            self.assertIs(result["pd_sep"], False)
            self._assert_no_sparse_writer(probes)
            self._assert_alias_cow_and_immutable(source_probes, probes, expected_g)
            results.append((label, prompt, expected_g, result, probes))
        for label, prompt, expected_g, result, probes in results:
            cold, cold_probes = self._request(label + "_cold", prompt, reuse=False)
            self.assertEqual(cold["reuse_len"], 0)
            self.assertEqual(result["output_ids"], cold["output_ids"])
            if self.cp_size == 1:
                # GEN cold/reference is already exact on this path. Keep its
                # original full-cold numeric gate and avoid reference replays.
                self._assert_hidden(probes, cold_probes, expected_g, label)
            else:
                self._assert_hidden(
                    probes,
                    cold_probes,
                    expected_g,
                    label + "_cold_diagnostic",
                    required=False,
                )
                self._assert_hidden(
                    probes,
                    self._private_reference(expected_g, 900),
                    expected_g,
                    label + "_independent",
                    reference="independent no-store A700 then forked dense prefix",
                )

    def test_internal_flag_full_already_coverable(self):
        self.assertEqual(self.cp_size, 1)
        prefix = self._extend_prompt("Dense FULL internal-only fork:", 650)
        source_prompt = self._extend_prompt(prefix, 768)
        source_ids = self.tokenizer.encode(source_prompt)
        branches = []
        for suffix in (" branch", " amber", " violet", " silver", " orange"):
            candidate = self.tokenizer.encode(prefix + suffix)
            if (
                len(candidate) > 650
                and candidate[:650] == source_ids[:650]
                and candidate[650] != source_ids[650]
                and all(candidate[650] != other[0] for other in branches)
            ):
                branches.append(
                    (candidate[650], self._extend_prompt(prefix + suffix, 900))
                )
            if len(branches) == 2:
                break
        self.assertEqual(len(branches), 2, "two strict FULL branches at F650")
        source, source_probes = self._request("full_internal_source768", source_prompt)
        self.assertEqual((source["input_len"], source["reuse_len"]), (768, 0))
        self._assert_no_sparse_writer(source_probes)
        results = {"A": source}
        for label, prompt in (("b", branches[0][1]), ("c", branches[1][1])):
            result, probes = self._request(f"full_internal_{label}900", prompt)
            self.assertEqual((result["input_len"], result["reuse_len"]), (900, 650))
            self._assert_no_sparse_writer(probes)
            self._assert_alias_cow_and_immutable(
                source_probes, probes, 650, source_end=768
            )
            cold, cold_probes = self._request(
                f"full_internal_{label}900_cold", prompt, reuse=False
            )
            self.assertEqual(cold["reuse_len"], 0)
            self.assertEqual(result["output_ids"], cold["output_ids"])
            self._assert_hidden(probes, cold_probes, 650, f"full_internal_{label}900")
            results[label.upper()] = result
            results[label.upper() + "_cold"] = cold
        (self.output_dir / "full_internal_summary.json").write_text(
            json.dumps({"flags": {"F": 1, "N": 0}, "requests": results}, indent=2)
            + "\n"
        )


class Qwen3DenseCommonPrefixPDTest(Qwen3DenseTailReuseTest):
    def _request(self, label, prompt, reuse=True):
        record, probes = hybrid_smoke.Qwen35TailReuseCPTest._request(
            self,
            label,
            prompt,
            reuse,
            pd_separation=True,
            max_new_tokens=2,
        )
        self.assertIs(record["pd_sep"], True, "actual P-to-D execution is required")
        self.assertEqual(len(record["output_ids"][0]), 2)
        self.assertIsInstance(record["prefill_reuse_len"], int)
        self.assertIsInstance(record["decode_reuse_len"], int)
        # Single-group FULL does not add transferred blocks to decode reuse
        # accounting (DecodeRpcServer::markLoadedCacheReuse). Verify the actual
        # serving endpoint instead of treating that counter as transfer proof.
        self.assertTrue(
            any(
                role["role"] == "DECODE" and role["http_port"] == self.decode.port
                for role in record["role_addrs"]
            ),
            "response must come from this test's DECODE service",
        )
        record["aggregate_reuse_len"] = record["reuse_len"]
        record["reuse_len"] = record["prefill_reuse_len"]
        (self.output_dir / f"{label}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        return record, probes


class Qwen3DenseTailReuseCPTest(Qwen3DenseTailReuseTest):
    cp_size = 2


if __name__ == "__main__":
    unittest.main()
