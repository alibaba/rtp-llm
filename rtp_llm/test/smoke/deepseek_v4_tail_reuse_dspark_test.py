"""Manual full 43-layer Flash-DSpARK P/D tail-reuse acceptance."""

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

_CAPTURE_LAYERS = (0, 40, 41, 42)


class _FlashDSparkTailReuseTest(unittest.TestCase):
    cp_size = 1

    def setUp(self):
        self.model_path = Path(os.environ["T07_FLASH_DSPARK_CHECKPOINT"])
        index = self.model_path / "model.safetensors.index.json"
        if not index.is_file():
            mounts = [
                line
                for line in Path("/proc/mounts").read_text().splitlines()
                if "3fs" in line.lower()
            ]
            self.fail(
                f"full Flash-DSpARK checkpoint index missing: {index}; "
                f"3fs mounts: {mounts}"
            )
        config = json.loads((self.model_path / "config.json").read_text())
        self.assertEqual(config["num_hidden_layers"], 43)
        self.assertEqual(config["dspark_target_layer_ids"], [40, 41, 42])
        self.assertEqual(len(config["compress_ratios"]), 46)
        self.assertEqual(config["compress_ratios"][-3:], [0, 0, 0])
        shards = set(json.loads(index.read_text())["weight_map"].values())
        self.assertEqual(len(shards), 48)
        self.assertTrue(all((self.model_path / name).is_file() for name in shards))
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
        self.assertGreaterEqual(len(cards), self.cp_size + 1)
        prefill_port = MagaServerManager.get_free_port()
        decode_port = MagaServerManager.get_free_port()
        route = ServiceRoute(
            service_id=f"v4_flash_dspark_tail_reuse_cp{self.cp_size}",
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
            "REMOTE_RPC_SERVER_IP": "localhost",
            "MODEL_SERVICE_CONFIG": route.model_dump_json(),
            "T07_CACHE_PROBE_CP_SIZE": str(self.cp_size),
        }
        internal_fork = self._testMethodName.startswith("test_internal_")
        common_args = (
            " --act_type BF16 --fp8_kv_cache 1 --seq_size_per_block 256"
            " --kernel_seq_size_per_block 128 --max_seq_len 2048"
            " --max_context_batch_size 1 --concurrency_limit 1 --reuse_cache 1"
            " --load_method fastsafetensors --kv_cache_mem_mb 16384"
            " --reserver_runtime_mem_mb 24576 --cache_store_rdma_mode 0"
            " --use_local 1 --load_cache_timeout_ms 120000 --warm_up 0"
            " --enable_device_cache 1 --enable_memory_cache 0"
            " --enable_disk_cache 0 --enable_remote_cache 0"
            f" --sp_type dspark --sp_model_type deepseek_v4_dspark --sp_checkpoint_path {self.model_path}"
            " --sp_act_type BF16 --gen_num_per_cycle 3"
        )
        if internal_fork:
            # The three short-lived F branches need much less cache than the
            # legacy full-model capacity setting. Leave room for all 45
            # target checkpoint shards on a shared GPU.
            common_args = common_args.replace(
                "--kv_cache_mem_mb 16384", "--kv_cache_mem_mb 4096"
            )
            common_args = common_args.replace(
                "--reserver_runtime_mem_mb 24576", "--reserver_runtime_mem_mb 8192"
            )
        cp_args = (
            " --prefill_cp_kv_cache_sharded 1 --prefill_cp_size 2"
            if self.cp_size == 2
            else ""
        )
        hook_config = self.output_dir / "tail_reuse_flash_dspark_hook_config.json"
        hook_config.write_text(
            json.dumps(
                {
                    "case": "v4_flash_dspark_tail_reuse",
                    "function_hooks": [
                        {
                            "target": "rtp_llm.models_py.modules.dsv4.prefill.forward.forward_layers",
                            "before": "before_forward_layers",
                            "after": "after_forward_layers",
                        },
                        {
                            "target": "rtp_llm.models_py.model_desc.deepseek_v4_dspark_model."
                            "DeepSeekV4DSparkModel.forward_commit",
                            "before": "before_forward_commit",
                            "after": "after_forward_commit",
                        },
                        {
                            "target": "rtp_llm.models_py.model_desc.deepseek_v4_dspark_model."
                            "DeepSeekV4DSparkModel.forward_propose",
                            "before": "before_forward_propose",
                            "after": "after_forward_propose",
                        },
                        {
                            "target": "rtp_llm.models_py.model_desc.deepseek_v4_model.forward_decode",
                            "after": "after_forward_decode",
                        },
                    ],
                }
            )
        )
        probe_file = Path(__file__).with_name("deepseek_v4_tail_reuse_dspark_probe.py")
        self.assertTrue(probe_file.is_file())
        prefill_args = (
            f"--tp_size {self.cp_size} --ep_size {self.cp_size} --dp_size 1"
            f" --world_size {self.cp_size}"
            + common_args
            + cp_args
            + " --role_type PREFILL --enable_cuda_graph 0"
            + (" --cp_rotate_method ALL_GATHER" if self.cp_size == 2 else "")
            + " --enable_prefill_partial_tail_reuse 1"
            + (" --enable_prefill_common_prefix_recovery 1" if internal_fork else "")
        )
        decode_args = (
            "--tp_size 1 --ep_size 1 --dp_size 1 --world_size 1"
            + common_args
            + cp_args
            + " --role_type DECODE --enable_cuda_graph 0"
            + (" --cp_rotate_method PREFILL_CP" if self.cp_size == 2 else "")
        )
        self.prefill = MagaServerManager(
            env_args={
                **common_env,
                "WORLD_SIZE": str(self.cp_size),
                "REMOTE_SERVER_PORT": decode_port,
                "MOEDBG": "1",
                "MOEDBG_ALL_LAYERS": "1",
                "MOEDBG_MAX_SEQ": "2000",
                "MOEDBG_FULL_THRESHOLD": "5000000",
                "MOEDBG_NAME_REGEX": r"^(layer(00|40|41|42)_(tail128|last)|lm_last_hidden|lm_logits_last)$",
                "MOEDBG_DIR": str(self.trace_dir),
                "MOEDBG_CASE": (
                    "dspark_target_cp2" if self.cp_size == 2 else "dspark_target"
                ),
                "RTP_HOT_HOOK": "1",
                "RTP_HOT_HOOK_FILE": str(probe_file),
                "RTP_HOT_HOOK_CONFIG": str(hook_config),
                "RTP_HOT_HOOK_DUMP_DIR": str(self.probe_dir),
                "T07_CACHE_PROBE_DIR": str(self.probe_dir),
                "T07_CACHE_PROBE_ROLE": "prefill",
                "T10_PRIVATE_FORK_REFERENCE": "1" if internal_fork else "0",
            },
            device_ids=cards[: self.cp_size],
            port=prefill_port,
            role_name="v4_flash_dspark_prefill",
            smoke_args_str=prefill_args,
        )
        self.decode = MagaServerManager(
            env_args={
                **common_env,
                "WORLD_SIZE": "1",
                "REMOTE_SERVER_PORT": prefill_port,
                "RTP_HOT_HOOK": "1",
                "RTP_HOT_HOOK_FILE": str(probe_file),
                "RTP_HOT_HOOK_CONFIG": str(hook_config),
                "RTP_HOT_HOOK_DUMP_DIR": str(self.probe_dir),
                "T07_CACHE_PROBE_DIR": str(self.probe_dir),
                "T07_CACHE_PROBE_ROLE": "decode",
            },
            device_ids=cards[self.cp_size : self.cp_size + 1],
            port=decode_port,
            role_name="v4_flash_dspark_decode",
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
                self.assertTrue(
                    start.result(), f"{role} full DSpARK server startup failed"
                )

    def _extend_prompt(self, prompt: str, length: int) -> str:
        current = len(self.tokenizer.encode(prompt))
        while current < length:
            prompt += " record"
            next_length = len(self.tokenizer.encode(prompt))
            self.assertEqual(next_length, current + 1, "one-token prompt extension")
            current = next_length
        self.assertEqual(current, length)
        return prompt

    def _request(self, label, prompt, reuse=True):
        target_pattern = "prefill_rank*_*.pt" if self.cp_size == 2 else "prefill_*.pt"
        trace_case = "dspark_target_cp2" if self.cp_size == 2 else "dspark_target"
        before_target = set(self.probe_dir.glob(target_pattern))
        before_trace = set((self.trace_dir / trace_case).glob("rank*_step*.pt"))
        before_commit = set(self.probe_dir.glob("dspark_commit_prefill_rank*_*.pt"))
        before_propose = set(self.probe_dir.glob("dspark_decode_propose_*.pt"))
        before_decode_commit = set(self.probe_dir.glob("dspark_decode_commit_*.pt"))
        decode_pattern = "decode_rank*_*.json" if self.cp_size == 2 else "decode_*.json"
        before_decode_table = set(self.probe_dir.glob(decode_pattern))
        query = {
            "prompt": prompt,
            "generate_config": {
                "max_new_tokens": 8,
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
        start_ns = time.time_ns()
        response = self.session.post(
            f"http://127.0.0.1:{self.prefill.port}/", json=query, timeout=1800
        )
        end_ns = time.time_ns()
        (self.output_dir / f"{label}.http.txt").write_text(response.text)
        response.raise_for_status()
        actual = response.json()
        self.assertNotIn("error_code", actual, actual)
        self.assertIs(actual.get("finished"), True)
        self.assertEqual(
            len(actual["input_ids"][0]), len(self.tokenizer.encode(prompt))
        )
        self.assertTrue(actual["output_ids"][0])
        aux = actual["aux_info"]
        self.assertGreater(
            aux["speculative_draft_rounds"], 0, f"{label}: no DSpARK round"
        )
        self.assertTrue(
            aux["speculative_accepted_tokens_per_pos"],
            f"{label}: no acceptance counters",
        )
        target_files = set(self.probe_dir.glob(target_pattern)) - before_target
        trace_files = (
            set((self.trace_dir / trace_case).glob("rank*_step*.pt")) - before_trace
        )
        commit_files = (
            set(self.probe_dir.glob("dspark_commit_prefill_rank*_*.pt")) - before_commit
        )
        propose_files = (
            set(self.probe_dir.glob("dspark_decode_propose_*.pt")) - before_propose
        )
        decode_commit_files = (
            set(self.probe_dir.glob("dspark_decode_commit_*.pt")) - before_decode_commit
        )
        decode_table_files = (
            set(self.probe_dir.glob(decode_pattern)) - before_decode_table
        )
        self.assertEqual(
            len(target_files), self.cp_size, f"{label}: target physical per rank"
        )
        self.assertEqual(
            len(trace_files), self.cp_size, f"{label}: target trace per rank"
        )
        self.assertEqual(
            len(commit_files), self.cp_size, f"{label}: draft commit per rank"
        )
        self.assertTrue(propose_files, f"{label}: D draft proposal absent")
        self.assertTrue(decode_commit_files, f"{label}: D draft tail commit absent")
        self.assertTrue(decode_table_files, f"{label}: P→D target tables absent")
        targets = {
            int(data["rank"]): data
            for data in (torch.load(path, weights_only=False) for path in target_files)
        }
        traces = {
            int(data["extra"]["cp_rank"]): data
            for data in (torch.load(path, weights_only=False) for path in trace_files)
        }
        commits = {
            int(data["rank"]): data
            for data in (torch.load(path, weights_only=False) for path in commit_files)
        }
        expected_ranks = set(range(self.cp_size))
        self.assertEqual(set(targets), expected_ranks)
        self.assertEqual(set(traces), expected_ranks)
        self.assertEqual(set(commits), expected_ranks)
        proposal_times = []
        commit_times = []
        for file in sorted(propose_files | decode_commit_files):
            state = torch.load(file, weights_only=False)
            self.assertTrue(all(state["physical_equal"].values()), file.name)
            self.assertLessEqual(state["hidden_maxabs"], 0.002, file.name)
            if "logits_maxabs" in state:
                self.assertLessEqual(state["logits_maxabs"], 0.002, file.name)
            if "draft_swa_table" in state:
                proposal_times.append(state["captured_unix_ns"])
                pages = (
                    state["seq_size_per_block"] // state["kernel_seq_size_per_block"]
                )
                last_column = (
                    (len(actual["input_ids"][0]) - 1) // state["seq_size_per_block"]
                ) * pages
                table = state["draft_swa_table"].reshape(-1)
                self.assertGreater(
                    table.numel(), last_column, f"{label}: draft P→D tail table"
                )
                self.assertGreater(
                    int(table[last_column]), 0, f"{label}: draft P→D tail key"
                )
            else:
                commit_times.append(state["captured_unix_ns"])
        self.assertTrue(proposal_times, f"{label}: proposal probe evidence absent")
        self.assertTrue(commit_times, f"{label}: decode commit probe evidence absent")
        self.assertLess(
            max(row["captured_unix_ns"] for row in targets.values()),
            min(proposal_times),
            f"{label}: target capture must precede D proposal",
        )
        self.assertLess(
            max(row["captured_unix_ns"] for row in commits.values()),
            min(proposal_times),
            f"{label}: all P draft commits must precede D proposal",
        )
        decodes = [json.loads(path.read_text()) for path in decode_table_files]
        self.assertEqual({data["rank"] for data in decodes}, {0})
        record = {
            "request_start_unix_ns": start_ns,
            "response_end_unix_ns": end_ns,
            "input_len": len(actual["input_ids"][0]),
            "prefill_reuse_len": actual["aux_info"].get("prefill_total_reuse_len"),
            "pd_sep": actual["aux_info"].get("pd_sep"),
            "target_capture_ns": {
                rank: row["captured_unix_ns"] for rank, row in targets.items()
            },
            "draft_commit_ns": {
                rank: row["captured_unix_ns"] for rank, row in commits.items()
            },
            "decode_proposals": len(propose_files),
            "decode_commits": len(decode_commit_files),
            "first_decode_proposal_unix_ns": min(proposal_times),
            "last_decode_commit_unix_ns": max(commit_times),
        }
        (self.output_dir / f"{label}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        return actual, targets, traces, commits, decodes, record

    def _assert_target_pair(self, label, source_end, end_position, source, after):
        actual, rows, traces, _, decodes, record = after
        a_rows = source[1]
        self.assertIs(actual["aux_info"]["pd_sep"], True)
        self.assertEqual(actual["aux_info"]["prefill_total_reuse_len"], source_end)
        self.assertLess(
            max(source[5]["target_capture_ns"].values()),
            record["request_start_unix_ns"],
        )
        for decode in decodes:
            self.assertEqual(set(decode["tables"]), set(DSV4_TAGS))
            for tag in DSV4_KERNEL_ROW_TAGS:
                physical_ids = decode["tables"][tag]["physical_ids"]
                self.assertEqual(len(physical_ids), (end_position + 255) // 256)
                self.assertTrue(
                    all(block_id > 0 for block_id in physical_ids),
                    f"{tag}: padding key",
                )
        numeric = {}
        owner = []
        copy_map = {}
        for rank in range(self.cp_size):
            a = a_rows[rank]
            b = rows[rank]
            self.assertEqual(a["end_position"], source_end)
            self.assertEqual(b["end_position"], end_position)
            self.assertLessEqual(b["captured_unix_ns"], record["response_end_unix_ns"])
            self.assertEqual(set(a["rows"]), set(b["rows"]))
            self.assertEqual({tag for _, tag in a["rows"]}, set(DSV4_TAGS))
            for key, old in a["rows"].items():
                current = b["rows"][key]
                save = old["save_block_ids"][old["save_positions"] == source_end]
                saved_end = current["current_save_block_ids"][
                    current["current_save_positions"] == end_position
                ]
                self.assertEqual(
                    save.numel(), 1, f"rank{rank} {key}: source short point"
                )
                self.assertEqual(
                    saved_end.numel(), 1, f"rank{rank} {key}: new short point"
                )
                physical = int(old["seq_size_per_block"])
                pages = physical // int(old["kernel_seq_size_per_block"])
                slot = (source_end - 1) // physical
                if key[1] in DSV4_KERNEL_ROW_TAGS:
                    slot //= self.cp_size
                    # FULL rows publish the stable runtime alias. They have no
                    # redundant sparse destination at either endpoint.
                    self.assertEqual(int(save[0]), -1)
                    self.assertEqual(int(saved_end[0]), -1)
                    source_ids = old["runtime_table"].reshape(-1)[
                        slot * pages : (slot + 1) * pages
                    ]
                    self.assertEqual(source_ids.numel(), pages)
                    self.assertTrue(bool((source_ids > 0).all()))
                    self.assertTrue(
                        set(source_ids.tolist()) <= set(old["source_ids"].tolist())
                    )
                    source_id = int(source_ids[0]) // pages
                    self.assertTrue(bool((source_ids // pages == source_id).all()))
                else:
                    self.assertGreater(int(save[0]), 0)
                    self.assertGreater(int(saved_end[0]), 0)
                    source_id = int(save[0])
                for suffix in ("ids", "values", "scales"):
                    name = f"source_{suffix}"
                    if name in old:
                        self.assertTrue(
                            torch.equal(old[name], current[name]),
                            f"rank{rank} {key}: source",
                        )
                write_id = int(current["runtime_table"][0, slot * pages]) // pages
                if write_id > 0:
                    self.assertNotEqual(write_id, source_id, f"rank{rank} {key}: alias")
                    copy_map.setdefault(str(key), {})[rank] = write_id
            direct_path = self.probe_dir / (
                f"direct_reference_rank{rank}_{end_position}_g{source_end}.pt"
                if self.cp_size == 2
                else f"direct_reference_{end_position}_g{source_end}.pt"
            )
            self.assertTrue(
                direct_path.is_file(), f"rank{rank}: target direct reference missing"
            )
            direct = torch.load(direct_path, weights_only=False)
            trace = traces[rank]
            if self.cp_size == 2:
                self.assertEqual(int(trace["extra"]["prefix_length"]), source_end)
                self.assertEqual(int(trace["extra"]["seq_len_total"]), end_position)
            else:
                self.assertEqual(
                    int(trace["extra"]["seq_len_full"]), end_position - source_end
                )
            pairs = []
            for layer in _CAPTURE_LAYERS:
                name = f"layer{layer:02d}_{'tail128' if self.cp_size == 2 else 'last'}"
                pairs.append((name, direct["layers"][layer]))
            pairs.extend(
                (
                    ("lm_last_hidden", direct["final_last"]),
                    ("lm_logits_last", direct["logits_last"]),
                )
            )
            if direct["final_last"].numel():
                owner.append(rank)
            for name, expected in pairs:
                observed = trace["tensors"][name].float()
                expected = expected.float()
                self.assertEqual(observed.shape, expected.shape, f"rank{rank} {name}")
                diff = (observed - expected).abs()
                numeric[f"rank{rank}_{name}"] = (
                    float(diff.max()) if diff.numel() else 0.0
                )
                with self.subTest(mode="dspark_target", rank=rank, tensor=name):
                    torch.testing.assert_close(
                        observed, expected, rtol=0.002, atol=0.002
                    )
            aux_path = self.probe_dir / f"dspark_aux_rank{rank}.pt"
            self.assertTrue(
                aux_path.is_file(), f"rank{rank}: target aux direct reference missing"
            )
            aux = torch.load(aux_path, weights_only=False)
            self.assertEqual(tuple(aux["capture_layers"]), (40, 41, 42))
            self.assertLessEqual(aux["maxabs"], 0.002)
        self.assertEqual(len(owner), 1, "one rank owns final target token")
        self.assertTrue(copy_map, "target restored copy destinations absent")
        if self.cp_size == 2:
            full_keys = {
                str(key) for key in a_rows[0]["rows"] if key[1] in DSV4_KERNEL_ROW_TAGS
            }
            self.assertTrue(
                full_keys <= set(copy_map), "FULL target CP copy destinations absent"
            )
            for key, ids in copy_map.items():
                self.assertEqual(set(ids), {0, 1}, f"{key}: all ranks copy")
                self.assertEqual(ids[0], ids[1], f"{key}: CP owner destination")
        (self.output_dir / f"target_numeric_diffs_{label}.json").write_text(
            json.dumps(numeric, indent=2) + "\n"
        )
        (self.output_dir / f"target_copy_map_{label}.json").write_text(
            json.dumps(copy_map, indent=2) + "\n"
        )

    def _assert_draft_pair(self, label, source_end, end_position, source, after):
        before = source[3]
        current = after[3]
        self.assertLess(
            max(source[5]["draft_commit_ns"].values()),
            after[5]["request_start_unix_ns"],
        )
        owner_ids = {}
        for rank in range(self.cp_size):
            a = before[rank]
            b = current[rank]
            self.assertEqual(a["end_position"], source_end)
            self.assertEqual(b["end_position"], end_position)
            self.assertEqual(b["resume"], source_end)
            self.assertEqual(set(a["rows"]), {(layer, "swa_kv") for layer in range(3)})
            self.assertEqual(set(b["rows"]), set(a["rows"]))
            self.assertTrue(
                all(b["physical_equal"].values()), f"rank{rank}: draft write"
            )
            self.assertTrue(
                all(b["source_immutable"].values()), f"rank{rank}: draft source"
            )
            self.assertTrue(
                all(b["restored_copy_equal"].values()), f"rank{rank}: draft restore"
            )
            self.assertLessEqual(
                b["hidden_maxabs"], 0.002, f"rank{rank}: commit projection"
            )
            for key, old in a["rows"].items():
                new = b["rows"][key]
                save = old["save_block_ids"][old["save_positions"] == source_end]
                saved_end = new["save_block_ids"][new["save_positions"] == end_position]
                self.assertEqual(save.numel(), 1, f"rank{rank} {key}: A draft save")
                self.assertEqual(
                    saved_end.numel(), 1, f"rank{rank} {key}: B draft save"
                )
                self.assertGreater(int(save[0]), 0)
                self.assertGreater(int(saved_end[0]), 0)
                self.assertTrue(
                    set(old["ids"].tolist())
                    - set(old["runtime_table"].reshape(-1).tolist()),
                    f"rank{rank} {key}: A draft optional row not captured",
                )
                self.assertTrue(
                    torch.equal(old["ids"], new["source_ids"]), f"rank{rank} {key}: IDs"
                )
                self.assertTrue(
                    torch.equal(old["values"], new["source_values"]),
                    f"rank{rank} {key}: bytes",
                )
                if "scales" in old:
                    self.assertTrue(
                        torch.equal(old["scales"], new["source_scales"]), key
                    )
                physical = int(old["seq_size_per_block"])
                pages = physical // int(old["kernel_seq_size_per_block"])
                slot = (source_end - 1) // physical
                write_id = int(new["runtime_table"].reshape(-1)[slot * pages]) // pages
                self.assertGreater(write_id, 0, f"rank{rank} {key}: draft write ID")
                self.assertNotEqual(
                    write_id, int(save[0]), f"rank{rank} {key}: draft source alias"
                )
                owner_ids.setdefault(str(key), {})[rank] = write_id
        if self.cp_size == 2:
            for key, ids in owner_ids.items():
                self.assertEqual(set(ids), {0, 1}, f"{key}: all draft CP ranks copy")
                self.assertEqual(ids[0], ids[1], f"{key}: draft CP owner")
        (self.output_dir / f"draft_copy_map_{label}.json").write_text(
            json.dumps(owner_ids, indent=2) + "\n"
        )

    def _run_case(self, label, source_end, end_position):
        prompt_a = self._extend_prompt(
            f"Full Flash-DSpARK {label} state test:", source_end
        )
        prompt_b = self._extend_prompt(prompt_a, end_position)
        self.assertEqual(
            self.tokenizer.encode(prompt_b)[:source_end],
            self.tokenizer.encode(prompt_a),
        )
        source = self._request(f"{label}_source_{source_end}", prompt_a)
        after = self._request(f"{label}_reused_{end_position}", prompt_b)
        self._assert_target_pair(label, source_end, end_position, source, after)
        self._assert_draft_pair(label, source_end, end_position, source, after)

    def _run_internal_f800_to_2000(self):
        label = (
            "dspark_cp2_internal_f800" if self.cp_size == 2 else "dspark_internal_f800"
        )
        base = f"Full Flash-DSpARK {label} branch:"
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
        self.assertEqual(len(branches), 2, "two strict full-DSpARK F800 branches")

        source = self._request(f"{label}_source_1300", prompt_a)
        self.assertEqual(source[0]["aux_info"].get("prefill_total_reuse_len"), 0)
        reused_b = self._request(f"{label}_b_2000", branches[0][1])
        expected_g = 512 if self.cp_size == 2 else 768
        self.assertEqual(
            reused_b[0]["aux_info"].get("prefill_total_reuse_len"), expected_g
        )
        selected_ids = {}
        for rank in range(self.cp_size):
            row_b = reused_b[1][rank]
            self.assertEqual(row_b["end_position"], 2000)
            private_name = (
                f"private_f800_runtime_rank{rank}.pt"
                if self.cp_size == 2
                else "private_f800_runtime.pt"
            )
            private_f = torch.load(self.probe_dir / private_name, weights_only=False)
            self.assertEqual(private_f["production_cache_bytes_copied"], 0)
            for key, row in row_b["rows"].items():
                old = source[1][rank]["rows"][key]
                for suffix in ("ids", "values", "scales"):
                    name = f"source_{suffix}"
                    if name in old:
                        self.assertTrue(
                            torch.equal(old[name], row[name]),
                            f"rank{rank} {key}: A source changed after B",
                        )
                positions = row["current_save_positions"].reshape(-1)
                blocks = row["current_save_block_ids"].reshape(-1)
                if key[1] in DSV4_KERNEL_ROW_TAGS:
                    self.assertFalse(
                        (blocks >= 0).any(), f"rank{rank} {key}: FULL save"
                    )
                    continue
                for endpoint in (800, 2000):
                    selected = blocks[positions == endpoint]
                    self.assertEqual(selected.numel(), 1, f"rank{rank} {key}: F/N")
                    self.assertGreater(int(selected[0]), 0)
                    identity = (key, endpoint)
                    if identity in selected_ids:
                        self.assertEqual(int(selected[0]), selected_ids[identity])
                    selected_ids[identity] = int(selected[0])
                self.assertNotEqual(
                    int(blocks[positions == 800][0]), int(blocks[positions == 2000][0])
                )
                for field in ("values", "scales"):
                    if field in private_f["rows"][key]:
                        self.assertTrue(
                            torch.equal(
                                row[f"selected_f800_{field}"],
                                private_f["rows"][key][field],
                            ),
                            f"rank{rank} {key}: private F {field} differs",
                        )

        reused_c = self._request(f"{label}_c_2000", branches[1][1])
        self.assertEqual(reused_c[0]["aux_info"].get("prefill_total_reuse_len"), 800)
        compared = {}
        owners = []
        for rank in range(self.cp_size):
            direct_name = (
                f"direct_reference_rank{rank}_2000_g800.pt"
                if self.cp_size == 2
                else "direct_reference_2000_g800.pt"
            )
            direct = torch.load(self.probe_dir / direct_name, weights_only=False)
            self.assertEqual(direct["production_cache_bytes_copied"], 0)
            self.assertEqual(direct["selected_save_slots"], 0)
            trace = reused_c[2][rank]
            if self.cp_size == 2:
                self.assertEqual(int(trace["extra"]["prefix_length"]), 800)
                self.assertEqual(int(trace["extra"]["seq_len_total"]), 2000)
            pairs = [
                (
                    f"layer{layer:02d}_{'tail128' if self.cp_size == 2 else 'last'}",
                    direct["layers"][layer],
                )
                for layer in _CAPTURE_LAYERS
            ]
            pairs.extend(
                (
                    ("lm_last_hidden", direct["final_last"]),
                    ("lm_logits_last", direct["logits_last"]),
                )
            )
            if direct["final_last"].numel():
                owners.append(rank)
            for name, expected in pairs:
                observed = trace["tensors"][name].float()
                expected = expected.float()
                self.assertEqual(observed.shape, expected.shape, f"rank{rank} {name}")
                diff = (observed - expected).abs()
                compared[f"rank{rank}_{name}"] = (
                    float(diff.max()) if diff.numel() else 0.0
                )
                with self.subTest(rank=rank, tensor=name):
                    torch.testing.assert_close(
                        observed, expected, rtol=0.002, atol=0.002
                    )
            aux = torch.load(
                self.probe_dir / f"dspark_aux_rank{rank}.pt", weights_only=False
            )
            self.assertEqual(tuple(aux["capture_layers"]), (40, 41, 42))
            self.assertLessEqual(aux["maxabs"], 0.002)
            for request in (reused_b, reused_c):
                self.assertIs(request[0]["aux_info"]["pd_sep"], True)
                commit = request[3][rank]
                for field in (
                    "physical_equal",
                    "source_immutable",
                    "restored_copy_equal",
                ):
                    self.assertTrue(commit[field], f"rank{rank}: empty {field}")
                    self.assertTrue(all(commit[field].values()), f"rank{rank}: {field}")
                self.assertLessEqual(commit["hidden_maxabs"], 0.002)
            for key, row in reused_c[1][rank]["rows"].items():
                old = source[1][rank]["rows"][key]
                for field in ("source_ids", "source_values", "source_scales"):
                    if field in old:
                        self.assertTrue(
                            torch.equal(old[field], row[field]),
                            f"rank{rank} {key}: A source changed after C",
                        )
        self.assertEqual(len(owners), 1, "one rank owns final target token")
        (self.output_dir / f"{label}_numeric_diffs.json").write_text(
            json.dumps(compared, indent=2) + "\n"
        )
        for name, reused, prompt in (
            ("b", reused_b, branches[0][1]),
            ("c", reused_c, branches[1][1]),
        ):
            cold = self._request(f"{label}_{name}_cold", prompt, reuse=False)
            self.assertEqual(cold[0]["aux_info"].get("prefill_total_reuse_len"), 0)
            self.assertEqual(reused[0]["output_ids"], cold[0]["output_ids"])


class FlashDSparkNonCPTest(_FlashDSparkTailReuseTest):
    def test_g1300_to_1320(self):
        self._run_case("dspark_noncp_1300_1320", 1300, 1320)

    def test_g1300_to_2000(self):
        self._run_case("dspark_noncp_1300_2000", 1300, 2000)

    def test_internal_f800_to_2000(self):
        self._run_internal_f800_to_2000()


class FlashDSparkCP2Test(_FlashDSparkTailReuseTest):
    cp_size = 2

    def test_internal_f800_to_2000(self):
        self._run_internal_f800_to_2000()

    def test_g1300_to_2000(self):
        self._run_case("dspark_cp2_1300_2000", 1300, 2000)

    def test_g1025_to_1026(self):
        self._run_case("dspark_cp2_1025_1026", 1025, 1026)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
