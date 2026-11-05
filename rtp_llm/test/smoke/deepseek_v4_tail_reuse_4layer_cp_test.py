"""Real CP2 P/D tail-reuse checks with rank-local physical and numeric evidence."""

from __future__ import annotations

import json
import logging
import os
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
    DSV4_KERNEL_ROW_TAGS,
    DSV4_TAGS,
)
from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
from rtp_llm.test.smoke.deepseek_v4_tail_reuse_4layer_test import (
    DeepSeekV4TailReuseFourLayerTest,
)
from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class DeepSeekV4TailReuseFourLayerCP2Test(DeepSeekV4TailReuseFourLayerTest):
    cp_size = 2

    def _start_servers(self):
        self.decode_world1 = os.environ.get("T07_CP_D_WORLD1") == "1"
        swap_cards = os.environ.get("T07_CP_SWAP_CARDS") == "1"
        cards = list(get_gpu_ids())
        required_cards = 4 if swap_cards or not self.decode_world1 else 3
        self.assertGreaterEqual(len(cards), required_cards)
        if swap_cards:
            cards[:4] = cards[2:4] + cards[:2]
        prefill_port = MagaServerManager.get_free_port()
        decode_port = MagaServerManager.get_free_port()
        route = ServiceRoute(
            service_id="v4_tail_reuse_four_layer_cp2",
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
            "DSV4_USE_MEGA_MOE_SE": "0",
            "DSV4_USE_MEGA_MOE_FUSED": "0",
            "DSV4_CHUNK_TOKENS": "2048",
            "WORLD_SIZE": "2",
            "REMOTE_RPC_SERVER_IP": "localhost",
            "MODEL_SERVICE_CONFIG": route.model_dump_json(),
            "T07_CACHE_PROBE_CP_SIZE": "2",
        }
        common_args = (
            "--tp_size 2 --ep_size 2 --dp_size 1 --world_size 2"
            " --act_type BF16 --fp8_kv_cache 1 --seq_size_per_block 256"
            " --kernel_seq_size_per_block 128 --max_seq_len 2048"
            " --max_context_batch_size 1 --concurrency_limit 1 --reuse_cache 1"
            " --load_method fastsafetensors --kv_cache_mem_mb 4096"
            " --reserver_runtime_mem_mb 12288 --cache_store_rdma_mode 0"
            " --use_local 1 --load_cache_timeout_ms 120000 --warm_up 0"
            " --enable_device_cache 1 --enable_memory_cache 0"
            " --enable_disk_cache 0 --enable_remote_cache 0"
            " --prefill_cp_kv_cache_sharded 1 --prefill_cp_size 2"
        )
        internal_fork = self._testMethodName.startswith("test_cp2_internal_")
        if internal_fork:
            common_args = common_args.replace(
                "--kv_cache_mem_mb 4096", "--kv_cache_mem_mb 2048"
            )
        decode_args = (
            common_args
            + " --role_type DECODE --enable_cuda_graph 0 --cp_rotate_method PREFILL_CP"
            + " --tp_size 1 --dp_size 2 --load_method scratch"
        )
        if self.decode_world1:
            decode_args = (
                common_args.replace(
                    "--tp_size 2 --ep_size 2 --dp_size 1 --world_size 2",
                    "--tp_size 1 --ep_size 1 --dp_size 1 --world_size 1",
                ).replace("--load_method fastsafetensors", "--load_method scratch")
                + " --role_type DECODE --enable_cuda_graph 0 --cp_rotate_method PREFILL_CP"
            )
        hook_config = self.output_dir / "tail_reuse_cp_hook_config.json"
        hook_config.write_text(
            json.dumps(
                {
                    "case": "v4_four_layer_cp_cache",
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
        self.prefill = MagaServerManager(
            env_args={
                **common_env,
                "REMOTE_SERVER_PORT": decode_port,
                "MOEDBG": "1",
                "MOEDBG_ALL_LAYERS": "1",
                "MOEDBG_MAX_SEQ": "2000",
                "MOEDBG_FULL_THRESHOLD": "5000000",
                "MOEDBG_NAME_REGEX": r"^(layer\d\d_(tail128|last)|lm_last_hidden|lm_logits_last)$",
                "MOEDBG_DIR": str(self.trace_dir),
                "MOEDBG_CASE": "four_layer_cp2",
                "RTP_HOT_HOOK": "1",
                "RTP_HOT_HOOK_FILE": str(probe_file),
                "RTP_HOT_HOOK_CONFIG": str(hook_config),
                "RTP_HOT_HOOK_DUMP_DIR": str(self.probe_dir),
                "T07_CACHE_PROBE_DIR": str(self.probe_dir),
                "T07_CACHE_PROBE_ROLE": "prefill",
                "T10_PRIVATE_FORK_REFERENCE": (
                    "1"
                    if self._testMethodName == "test_cp2_internal_f800_numerical"
                    else "0"
                ),
            },
            device_ids=cards[:2],
            port=prefill_port,
            role_name="v4_tail_cp2_prefill",
            smoke_args_str=(
                common_args
                + " --role_type PREFILL --enable_cuda_graph 0"
                + " --cp_rotate_method ALL_GATHER --enable_prefill_partial_tail_reuse 1"
                + (
                    " --enable_prefill_common_prefix_recovery 1"
                    if internal_fork
                    else ""
                )
            ),
        )
        self.decode = MagaServerManager(
            env_args={
                **common_env,
                "WORLD_SIZE": "1" if self.decode_world1 else "2",
                "REMOTE_SERVER_PORT": prefill_port,
                "RTP_HOT_HOOK": "1",
                "RTP_HOT_HOOK_FILE": str(probe_file),
                "RTP_HOT_HOOK_CONFIG": str(hook_config),
                "RTP_HOT_HOOK_DUMP_DIR": str(self.probe_dir),
                "T07_CACHE_PROBE_DIR": str(self.probe_dir),
                "T07_CACHE_PROBE_ROLE": "decode",
            },
            device_ids=cards[2:3] if self.decode_world1 else cards[2:4],
            port=decode_port,
            role_name="v4_tail_cp2_decode",
            smoke_args_str=decode_args,
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
                self.assertTrue(start.result(), f"{role} CP2 server startup failed")

    def _request_cp(self, label: str, prompt: str):
        before_probe = set(self.probe_dir.glob("prefill_rank*_*.pt"))
        before_decode = set(self.probe_dir.glob("decode_rank*_*.json"))
        before_trace = set(self.trace_dir.glob("four_layer_cp2/rank*_step*.pt"))
        query = {
            "prompt": prompt,
            "generate_config": {
                "max_new_tokens": 2,
                "ignore_eos": True,
                "top_k": 1,
                "top_p": 1.0,
                "random_seed": 7,
                "reuse_cache": True,
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

        probe_files = set(self.probe_dir.glob("prefill_rank*_*.pt")) - before_probe
        trace_files = (
            set(self.trace_dir.glob("four_layer_cp2/rank*_step*.pt")) - before_trace
        )
        decode_files = set(self.probe_dir.glob("decode_rank*_*.json")) - before_decode
        self.assertEqual(
            len(probe_files), 2, f"{label}: one physical snapshot per CP rank"
        )
        self.assertEqual(len(trace_files), 2, f"{label}: one model trace per CP rank")
        self.assertTrue(decode_files, f"{label}: P→D decode table missing")
        probes = {
            int(data["rank"]): data
            for data in (torch.load(path, weights_only=False) for path in probe_files)
        }
        traces = {
            int(data["extra"]["cp_rank"]): data
            for data in (torch.load(path, weights_only=False) for path in trace_files)
        }
        decodes = [json.loads(path.read_text()) for path in sorted(decode_files)]
        self.assertEqual(set(probes), {0, 1})
        self.assertEqual(set(traces), {0, 1})
        decode_ranks = {item["rank"] for item in decodes}
        if self.decode_world1:
            self.assertEqual(decode_ranks, {0})
        else:
            self.assertTrue(decode_ranks <= {0, 1})
        record = {
            "input_len": len(actual["input_ids"][0]),
            "request_start_unix_ns": request_start_ns,
            "response_end_unix_ns": response_end_ns,
            "prefill_reuse_len": actual["aux_info"].get("prefill_total_reuse_len"),
            "pd_sep": actual["aux_info"].get("pd_sep"),
            "output_ids": actual["output_ids"],
            "prefill_rank_capture_unix_ns": {
                rank: data["captured_unix_ns"] for rank, data in probes.items()
            },
            "prefill_rank_trace_extra": {
                rank: {
                    key: str(value)
                    for key, value in trace["extra"].items()
                    if key
                    in (
                        "cp_size",
                        "cp_rank",
                        "seq_len_full",
                        "prefix_length",
                        "seq_len_total",
                    )
                }
                for rank, trace in traces.items()
            },
            "decode_tables": decodes,
        }
        (self.output_dir / f"{label}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        return actual, probes, traces, decodes, record

    def _assert_cp_pair(self, label, source_end, end_position, source, after):
        _, before_rows, _, _, source_record = source
        actual, after_rows, traces, decodes, after_record = after
        self.assertIs(actual["aux_info"]["pd_sep"], True)
        self.assertEqual(actual["aux_info"]["prefill_total_reuse_len"], source_end)
        self.assertLess(
            max(source_record["prefill_rank_capture_unix_ns"].values()),
            after_record["request_start_unix_ns"],
            "both A ranks must finish before B can use the short point",
        )
        self.assertEqual(set(before_rows), {0, 1})
        self.assertEqual(set(after_rows), {0, 1})
        for rank in (0, 1):
            a = before_rows[rank]
            b = after_rows[rank]
            self.assertEqual(a["end_position"], source_end)
            self.assertEqual(b["end_position"], end_position)
            self.assertEqual(set(a["rows"]), set(b["rows"]))
            self.assertEqual({tag for _, tag in a["rows"]}, set(DSV4_TAGS))
            self.assertLessEqual(
                b["captured_unix_ns"], after_record["response_end_unix_ns"]
            )
            trace = traces[rank]
            self.assertEqual(int(trace["extra"]["cp_size"]), 2)
            self.assertEqual(int(trace["extra"]["prefix_length"]), source_end)
            self.assertEqual(
                int(trace["extra"]["seq_len_full"]), end_position - source_end
            )
            self.assertEqual(int(trace["extra"]["seq_len_total"]), end_position)
            for key, before_row in a["rows"].items():
                self._assert_registered_source(
                    key,
                    before_row,
                    b["rows"][key],
                    source_end,
                    end_position,
                    f"{label} rank{rank}",
                )
            if source_end == 1025:
                for tag in ("csa_state", "indexer_state"):
                    row = b["rows"][(2, tag)]
                    physical = int(row["seq_size_per_block"])
                    last = (source_end - 1) // physical
                    self.assertGreater(last, 0)
                    table = row["runtime_table"].reshape(-1)
                    self.assertGreater(
                        int(table[last - 1]), 0, f"rank{rank} {tag}: old history slot"
                    )
                    self.assertGreater(
                        int(table[last]), 0, f"rank{rank} {tag}: short history slot"
                    )

        copy_map = {}
        for key in before_rows[0]["rows"]:
            per_rank = []
            for rank in (0, 1):
                a = before_rows[rank]["rows"][key]
                b = after_rows[rank]["rows"][key]
                physical = int(a["seq_size_per_block"])
                pages = physical // int(a["kernel_seq_size_per_block"])
                slot = (source_end - 1) // physical
                if key[1] in DSV4_KERNEL_ROW_TAGS:
                    slot //= self.cp_size
                column = slot * pages
                a_id = int(a["runtime_table"][0, column]) // pages
                b_id = int(b["runtime_table"][0, column]) // pages
                if key[1] in DSV4_KERNEL_ROW_TAGS:
                    save_id = a_id  # FULL publication retains runtime ownership.
                else:
                    save = a["save_block_ids"][a["save_positions"] == source_end]
                    save_id = int(save[0])
                if b_id > 0:
                    self.assertNotEqual(
                        b_id,
                        save_id,
                        f"rank{rank} {key}: cached short source reused for write",
                    )
                    per_rank.append(
                        {"a_runtime": a_id, "a_save": save_id, "b_write": b_id}
                    )
            if key[1] in ("csa_kv", "hca_kv", "indexer_kv"):
                self.assertEqual(
                    len(per_rank),
                    2,
                    f"{key}: all CP ranks must copy the FULL tail block",
                )
            if per_rank:
                self.assertEqual(
                    len(per_rank), 2, f"{key}: all CP ranks must copy the logical block"
                )
                self.assertEqual(
                    per_rank[0]["b_write"],
                    per_rank[1]["b_write"],
                    f"{key}: CP owner ID",
                )
                copy_map[str(key)] = per_rank
        self.assertTrue(copy_map, "no CP copy destinations observed")
        (self.output_dir / f"copy_map_{label}.json").write_text(
            json.dumps(copy_map, indent=2) + "\n"
        )

        decode_ranks = {item["rank"] for item in decodes}
        if self.decode_world1:
            self.assertEqual(decode_ranks, {0})
        else:
            self.assertTrue(decode_ranks <= {0, 1})
        expected_keys = (end_position + 255) // 256
        for decode in decodes:
            self.assertEqual(set(decode["tables"]), set(DSV4_TAGS))
            for tag in ("csa_kv", "hca_kv", "indexer_kv"):
                self.assertEqual(
                    len(decode["tables"][tag]["physical_ids"]), expected_keys
                )

        numeric = {}
        owner_ranks = []
        for rank in (0, 1):
            path = (
                self.probe_dir
                / f"direct_reference_rank{rank}_{end_position}_g{source_end}.pt"
            )
            if not path.is_file():
                error_path = self.probe_dir / f"direct_reference_error_rank{rank}.txt"
                self.fail(
                    error_path.read_text()
                    if error_path.is_file()
                    else f"rank{rank}: direct reference missing"
                )
            direct = torch.load(path, weights_only=False)
            trace = traces[rank]
            comparisons = []
            for layer in range(4):
                name = f"layer{layer:02d}_tail128"
                resumed = trace["tensors"][name]
                reference = direct["layers"][layer].float()
                comparisons.append((name, resumed, reference))
            for name, direct_key in (
                ("lm_last_hidden", "final_last"),
                ("lm_logits_last", "logits_last"),
            ):
                resumed = trace["tensors"][name]
                reference = direct[direct_key].float()
                comparisons.append((name, resumed, reference))
            if direct["final_last"].numel():
                owner_ranks.append(rank)
            for name, resumed, reference in comparisons:
                self.assertEqual(resumed.shape, reference.shape, f"rank{rank} {name}")
                self.assertTrue(torch.isfinite(resumed).all(), f"rank{rank} {name}")
                diff = (resumed - reference).abs()
                numeric[f"rank{rank}_{name}"] = (
                    float(diff.max()) if diff.numel() else 0.0
                )
                with self.subTest(rank=rank, tensor=name):
                    torch.testing.assert_close(
                        resumed, reference, rtol=0.002, atol=0.002
                    )
        self.assertEqual(
            len(owner_ranks), 1, "exactly one CP rank must own final prompt token"
        )
        (self.output_dir / f"numeric_diffs_{label}.json").write_text(
            json.dumps(numeric, indent=2) + "\n"
        )

    def _run_cp_case(self, label: str, source_end: int, end_position: int):
        prompt_a = self._extend_prompt(
            f"Tail reuse CP2 {label} state test:", source_end
        )
        prompt_b = self._extend_prompt(prompt_a, end_position)
        self.assertEqual(
            self.tokenizer.encode(prompt_b)[:source_end],
            self.tokenizer.encode(prompt_a),
        )
        source = self._request_cp(f"{label}_source_{source_end}", prompt_a)
        after = self._request_cp(f"{label}_reused_{end_position}", prompt_b)
        self._assert_cp_pair(label, source_end, end_position, source, after)

    def test_cp2_g1300_to_2000(self):
        self._run_cp_case("cp2_1300_2000", 1300, 2000)

    def test_cp2_g1300_to_1320(self):
        self._run_cp_case("cp2_1300_1320", 1300, 1320)

    def test_cp2_g1025_to_1026(self):
        self._run_cp_case("cp2_1025_1026", 1025, 1026)

    def _run_cp2_internal_f800(self):
        label = "cp2_internal_f800"
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
        self.assertIsNotNone(prompt_c, "no third CP2 one-token branch at F800")

        source = self._request_cp(f"{label}_source_1300", prompt_a)
        reused_b = self._request_cp(f"{label}_b_2000", prompt_b)
        self.assertEqual(reused_b[0]["aux_info"].get("prefill_total_reuse_len"), 512)
        self.assertEqual(set(reused_b[1]), {0, 1})
        for rank in (0, 1):
            self.assertEqual(int(reused_b[2][rank]["extra"]["seq_len_full"]), 1488)
            self.assertEqual(int(reused_b[2][rank]["extra"]["seq_len_total"]), 2000)
            for key, after in reused_b[1][rank]["rows"].items():
                before = source[1][rank]["rows"][key]
                for suffix in ("ids", "values", "scales"):
                    name = f"source_{suffix}"
                    if name in before:
                        self.assertTrue(
                            torch.equal(before[name], after[name]),
                            f"rank{rank} {key}: A source changed after B",
                        )
                if key[1] in DSV4_KERNEL_ROW_TAGS:
                    self.assertFalse((after["current_save_block_ids"] >= 0).any())
                    continue
                positions = after["current_save_positions"].reshape(-1)
                blocks = after["current_save_block_ids"].reshape(-1)
                for endpoint in (800, 2000):
                    selected = blocks[positions == endpoint]
                    self.assertEqual(selected.numel(), 1, f"rank{rank} {key}: F/N")
                    self.assertGreater(int(selected[0]), 0)
                self.assertNotEqual(
                    int(blocks[positions == 800][0]),
                    int(blocks[positions == 2000][0]),
                    f"rank{rank} {key}: F and N must be distinct",
                )

        reused_c = self._request_cp(f"{label}_c_2000", prompt_c)
        self.assertEqual(reused_c[0]["aux_info"].get("prefill_total_reuse_len"), 800)
        for rank in (0, 1):
            self.assertEqual(int(reused_c[2][rank]["extra"]["seq_len_full"]), 1200)
            self.assertEqual(int(reused_c[2][rank]["extra"]["seq_len_total"]), 2000)
            for key, after in reused_c[1][rank]["rows"].items():
                before = source[1][rank]["rows"][key]
                for suffix in ("ids", "values", "scales"):
                    name = f"source_{suffix}"
                    if name in before:
                        self.assertTrue(
                            torch.equal(before[name], after[name]),
                            f"rank{rank} {key}: A source changed after C",
                        )
        return source, reused_b, reused_c

    def test_cp2_internal_f800_later_request(self):
        self._run_cp2_internal_f800()

    def test_cp2_internal_f800_numerical(self):
        _, reused_b, reused_c = self._run_cp2_internal_f800()
        numeric = {}
        owner_ranks = []
        for rank in (0, 1):
            private_a = torch.load(
                self.probe_dir / f"private_reference_rank{rank}_1300_g0.pt",
                weights_only=False,
            )
            self.assertEqual(private_a["production_cache_bytes_copied"], 0)
            self.assertEqual(private_a["selected_save_slots"], 1)
            private_f = torch.load(
                self.probe_dir / f"private_f800_runtime_rank{rank}.pt",
                weights_only=False,
            )
            self.assertEqual(private_f["production_cache_bytes_copied"], 0)
            for key, row in reused_b[1][rank]["rows"].items():
                if key[1] in DSV4_KERNEL_ROW_TAGS:
                    continue
                selected = row.get("selected_f800_values")
                self.assertIsNotNone(selected, f"rank{rank} {key}: missing F row")
                self.assertTrue(
                    torch.equal(selected, private_f["rows"][key]["values"]),
                    f"rank{rank} {key}: private F state differs",
                )
            private_b = torch.load(
                self.probe_dir / f"private_reference_rank{rank}_2000_g512.pt",
                weights_only=False,
            )
            self.assertEqual(private_b["production_cache_bytes_copied"], 0)
            self.assertEqual(private_b["selected_save_slots"], 1)
            direct = torch.load(
                self.probe_dir / f"direct_reference_rank{rank}_2000_g800.pt",
                weights_only=False,
            )
            self.assertEqual(direct["production_cache_bytes_copied"], 0)
            self.assertEqual(direct["selected_save_slots"], 0)
            self.assertEqual(
                direct["provenance"],
                "private_no_store_A0_to_1300_B512_to_2000_then_C800_to_2000",
            )
            if direct["final_last"].numel():
                owner_ranks.append(rank)
            trace = reused_c[2][rank]
            comparisons = [
                (f"layer{layer:02d}_tail128", direct["layers"][layer].float())
                for layer in range(4)
            ]
            comparisons.extend(
                (
                    ("lm_last_hidden", direct["final_last"].float()),
                    ("lm_logits_last", direct["logits_last"].float()),
                )
            )
            for name, reference in comparisons:
                resumed = trace["tensors"][name]
                self.assertEqual(resumed.shape, reference.shape, f"rank{rank} {name}")
                self.assertTrue(torch.isfinite(resumed).all(), f"rank{rank} {name}")
                difference = (resumed - reference).abs()
                numeric[f"rank{rank}_{name}"] = (
                    float(difference.max()) if difference.numel() else 0.0
                )
                with self.subTest(rank=rank, tensor=name):
                    torch.testing.assert_close(
                        resumed, reference, rtol=0.002, atol=0.002
                    )
        self.assertEqual(len(owner_ranks), 1, "one CP rank owns the last prompt token")
        (self.output_dir / "cp2_f800_numeric_diffs.json").write_text(
            json.dumps(numeric, indent=2) + "\n"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
