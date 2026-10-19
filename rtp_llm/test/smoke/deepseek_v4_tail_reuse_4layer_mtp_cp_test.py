"""Real CP2 P/D MTP tail-reuse checks for target and draft state."""

from __future__ import annotations

import json
import logging
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
from rtp_llm.test.smoke.deepseek_v4_tail_reuse_4layer_cp_test import (
    DeepSeekV4TailReuseFourLayerCP2Test,
)
from rtp_llm.test.smoke.deepseek_v4_tail_reuse_4layer_mtp_test import (
    DeepSeekV4TailReuseFourLayerMTPTest,
)
from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class DeepSeekV4TailReuseFourLayerMTPCP2Test(DeepSeekV4TailReuseFourLayerCP2Test):
    def setUp(self):
        # The MTP setup validates the filtered four-layer target plus one-layer
        # draft checkpoint; dynamic dispatch starts the CP2 topology below.
        DeepSeekV4TailReuseFourLayerMTPTest.setUp(self)

    def _start_servers(self):
        self.decode_world1 = True
        cards = list(get_gpu_ids())
        self.assertGreaterEqual(len(cards), 3)
        prefill_port = MagaServerManager.get_free_port()
        decode_port = MagaServerManager.get_free_port()
        route = ServiceRoute(
            service_id="v4_tail_reuse_four_layer_mtp_cp2",
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
            "REMOTE_RPC_SERVER_IP": "localhost",
            "MODEL_SERVICE_CONFIG": route.model_dump_json(),
            "T07_CACHE_PROBE_CP_SIZE": "2",
        }
        common_args = (
            " --act_type BF16 --fp8_kv_cache 1 --seq_size_per_block 256"
            " --kernel_seq_size_per_block 128 --max_seq_len 2048"
            " --max_context_batch_size 1 --concurrency_limit 1 --reuse_cache 1"
            " --load_method fastsafetensors --kv_cache_mem_mb 2048"
            " --reserver_runtime_mem_mb 8192 --cache_store_rdma_mode 0"
            " --use_local 1 --load_cache_timeout_ms 120000 --warm_up 0"
            " --enable_device_cache 1 --enable_memory_cache 0"
            " --enable_disk_cache 0 --enable_remote_cache 0"
            f" --sp_type mtp --sp_model_type deepseek_v4_mtp --sp_checkpoint_path {self.model_path}"
            " --sp_act_type bf16 --gen_num_per_cycle 3"
        )
        hook_config = self.output_dir / "tail_reuse_mtp_cp2_hook_config.json"
        hook_config.write_text(
            json.dumps(
                {
                    "case": "v4_four_layer_mtp_cp2_cache",
                    "function_hooks": [
                        {
                            "target": "rtp_llm.models_py.modules.dsv4.prefill.forward.forward_layers",
                            "before": "before_forward_layers",
                            "after": "after_forward_layers",
                        },
                        {
                            "target": "rtp_llm.models_py.modules.dsv4.decode.forward.forward_decode",
                            "before": "before_forward_decode",
                            "after": "after_forward_decode",
                        },
                    ],
                }
            )
        )
        probe_file = Path(__file__).with_name(
            "deepseek_v4_tail_reuse_4layer_mtp_cp_probe.py"
        )
        self.assertTrue(probe_file.is_file())
        self.prefill = MagaServerManager(
            env_args={
                **common_env,
                "WORLD_SIZE": "2",
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
            },
            device_ids=cards[:2],
            port=prefill_port,
            role_name="v4_tail_mtp_cp2_prefill",
            smoke_args_str=(
                "--tp_size 2 --ep_size 2 --dp_size 1 --world_size 2"
                + common_args
                + " --role_type PREFILL --enable_cuda_graph 0"
                + " --cp_rotate_method ALL_GATHER --prefill_cp_kv_cache_sharded 1"
                + " --prefill_cp_size 2 --enable_prefill_partial_tail_reuse 1"
            ),
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
            device_ids=cards[2:3],
            port=decode_port,
            role_name="v4_tail_mtp_cp2_decode",
            smoke_args_str=(
                "--tp_size 1 --ep_size 1 --dp_size 1 --world_size 1"
                + common_args
                + " --role_type DECODE --enable_cuda_graph 0"
                + " --cp_rotate_method PREFILL_CP --prefill_cp_kv_cache_sharded 1"
                + " --prefill_cp_size 2"
            ),
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
                self.assertTrue(start.result(), f"{role} MTP CP2 server startup failed")

    def _request_mtp_cp(self, label: str, prompt: str):
        draft_dir = self.probe_dir / "mtp_draft_prefill"
        trace_dir = self.trace_dir / "mtp_draft_prefill_cp2"
        before_prefill = set(draft_dir.glob("prefill_rank*_*.pt"))
        before_trace = set(trace_dir.glob("rank*_step*.pt"))
        before_decode = set(self.probe_dir.glob("mtp_draft_decode_*.pt"))
        target = self._request_cp(label, prompt)
        draft_files = set(draft_dir.glob("prefill_rank*_*.pt")) - before_prefill
        trace_files = set(trace_dir.glob("rank*_step*.pt")) - before_trace
        decode_files = set(self.probe_dir.glob("mtp_draft_decode_*.pt")) - before_decode
        self.assertEqual(len(draft_files), 2, f"{label}: draft physical per CP rank")
        self.assertEqual(len(trace_files), 2, f"{label}: draft trace per CP rank")
        self.assertTrue(decode_files, f"{label}: D draft continuation missing")
        draft = {
            int(data["rank"]): data
            for data in (torch.load(path, weights_only=False) for path in draft_files)
        }
        traces = {
            int(data["extra"]["cp_rank"]): data
            for data in (torch.load(path, weights_only=False) for path in trace_files)
        }
        self.assertEqual(set(draft), {0, 1})
        self.assertEqual(set(traces), {0, 1})
        checks = []
        for file in sorted(decode_files):
            data = torch.load(file, weights_only=False)
            self.assertTrue(data["source"], f"{label}: D draft source empty")
            self.assertTrue(
                all(data["physical_equal"].values()), f"{label}: D draft cache"
            )
            for name, diff in data["compared"].items():
                self.assertLessEqual(diff, 0.002, f"{label}: D draft {name}")
            checks.append({"file": file.name, "compared": data["compared"]})
        (self.output_dir / f"{label}_draft_decode_oracles.json").write_text(
            json.dumps(checks, indent=2) + "\n"
        )
        return target, draft, traces

    def _assert_draft_cp_pair(self, label, source_end, end_position, source, after):
        _, before_rows, _ = source
        _, after_rows, traces = after
        after_record = after[0][4]
        self.assertLess(
            max(row["captured_unix_ns"] for row in before_rows.values()),
            after_record["request_start_unix_ns"],
            "both A draft ranks must finish before B",
        )
        self.assertEqual(set(before_rows), {0, 1})
        self.assertEqual(set(after_rows), {0, 1})
        copy_map = {}
        numeric = {}
        final_owner = []
        for rank in (0, 1):
            a = before_rows[rank]
            b = after_rows[rank]
            self.assertEqual(a["end_position"], source_end)
            self.assertEqual(b["end_position"], end_position)
            self.assertEqual(set(a["rows"]), {(0, "swa_kv")})
            self.assertEqual(set(b["rows"]), {(0, "swa_kv")})
            self.assertLessEqual(
                b["captured_unix_ns"], after_record["response_end_unix_ns"]
            )
            row = a["rows"][(0, "swa_kv")]
            current = b["rows"][(0, "swa_kv")]
            save = row["save_block_ids"][row["save_positions"] == source_end]
            saved_end = current["current_save_block_ids"][
                current["current_save_positions"] == end_position
            ]
            self.assertEqual(save.numel(), 1, f"rank{rank}: A draft short save")
            self.assertEqual(saved_end.numel(), 1, f"rank{rank}: B draft short save")
            save_id = int(save[0])
            self.assertGreater(save_id, 0)
            self.assertGreater(int(saved_end[0]), 0)
            self.assertTrue(
                set(row["source_ids"].tolist()) - set(row["table_ids"].tolist()),
                f"rank{rank}: draft optional source row absent",
            )
            for suffix in ("ids", "values", "scales"):
                name = f"source_{suffix}"
                if name in row:
                    self.assertTrue(
                        torch.equal(row[name], current[name]),
                        f"rank{rank}: draft source mutated: {name}",
                    )
            physical = int(row["seq_size_per_block"])
            pages = physical // int(row["kernel_seq_size_per_block"])
            column = ((source_end - 1) // physical) * pages
            a_runtime = int(row["runtime_table"][0, column]) // pages
            b_write = int(current["runtime_table"][0, column]) // pages
            self.assertGreater(b_write, 0, f"rank{rank}: draft runtime destination")
            self.assertNotEqual(
                b_write, save_id, f"rank{rank}: draft source reused for write"
            )
            copy_map[rank] = {
                "a_runtime": a_runtime,
                "a_save": save_id,
                "b_write": b_write,
            }
            trace = traces[rank]
            self.assertEqual(int(trace["extra"]["cp_size"]), 2)
            self.assertEqual(int(trace["extra"]["prefix_length"]), source_end)
            self.assertEqual(
                int(trace["extra"]["seq_len_full"]), end_position - source_end
            )
            self.assertEqual(int(trace["extra"]["seq_len_total"]), end_position)
            path = (
                self.probe_dir
                / "mtp_draft_prefill"
                / f"direct_reference_rank{rank}_{end_position}_g{source_end}.pt"
            )
            if not path.is_file():
                error = path.parent / f"direct_reference_error_rank{rank}.txt"
                self.fail(
                    error.read_text()
                    if error.is_file()
                    else f"rank{rank}: draft reference missing"
                )
            direct = torch.load(path, weights_only=False)
            pairs = [
                ("layer00_tail128", direct["layers"][0]),
                ("lm_last_hidden", direct["final_last"]),
                ("lm_logits_last", direct["logits_last"]),
            ]
            if direct["final_last"].numel():
                final_owner.append(rank)
            for name, expected in pairs:
                actual = trace["tensors"][name].float()
                expected = expected.float()
                self.assertEqual(actual.shape, expected.shape, f"rank{rank} {name}")
                self.assertTrue(torch.isfinite(actual).all(), f"rank{rank} {name}")
                diff = (actual - expected).abs()
                numeric[f"rank{rank}_{name}"] = (
                    float(diff.max()) if diff.numel() else 0.0
                )
                with self.subTest(mode="mtp_cp2_draft", rank=rank, tensor=name):
                    torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)
        self.assertEqual(
            copy_map[0]["b_write"], copy_map[1]["b_write"], "draft CP owner"
        )
        self.assertEqual(len(final_owner), 1, "one draft rank owns the final token")
        (self.output_dir / f"draft_copy_map_{label}.json").write_text(
            json.dumps(copy_map, indent=2) + "\n"
        )
        (self.output_dir / f"draft_numeric_diffs_{label}.json").write_text(
            json.dumps(numeric, indent=2) + "\n"
        )

    def _run_mtp_cp_case(self, label: str, source_end: int, end_position: int):
        prompt_a = self._extend_prompt(
            f"Tail reuse MTP CP2 {label} state test:", source_end
        )
        prompt_b = self._extend_prompt(prompt_a, end_position)
        self.assertEqual(
            self.tokenizer.encode(prompt_b)[:source_end],
            self.tokenizer.encode(prompt_a),
        )
        source = self._request_mtp_cp(f"{label}_source_{source_end}", prompt_a)
        after = self._request_mtp_cp(f"{label}_reused_{end_position}", prompt_b)
        self._assert_cp_pair(label, source_end, end_position, source[0], after[0])
        self._assert_draft_cp_pair(label, source_end, end_position, source, after)

    def test_mtp_cp2_g1300_to_1320(self):
        self._run_mtp_cp_case("mtp_cp2_1300_1320", 1300, 1320)

    def test_mtp_cp2_g1300_to_2000(self):
        self._run_mtp_cp_case("mtp_cp2_1300_2000", 1300, 2000)

    def test_mtp_cp2_g1025_to_1026(self):
        self._run_mtp_cp_case("mtp_cp2_1025_1026", 1025, 1026)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
