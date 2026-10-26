"""Four-rank DSV4 prefill tail reuse with physical and numeric evidence.

This bounded four-GPU target exercises the PREFILL role's local route with
max_new_tokens=1. It does not exercise P-to-D cache transfer or decode keys.
"""

from __future__ import annotations

import json
import logging
import time
import unittest
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


class DeepSeekV4TailReuseFourLayerCP4Test(DeepSeekV4TailReuseFourLayerTest):
    cp_size = 4

    def _start_servers(self):
        cards = list(get_gpu_ids())
        self.assertGreaterEqual(len(cards), self.cp_size)
        prefill_port = MagaServerManager.get_free_port()
        unused_decode_port = MagaServerManager.get_free_port()
        route = ServiceRoute(
            service_id="v4_tail_reuse_four_layer_cp4_prefill",
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
                        address=f"127.0.0.1:{unused_decode_port}",
                        protocol="http",
                        path="/",
                    ),
                )
            ],
        )
        hook_config = self.output_dir / "tail_reuse_cp4_hook_config.json"
        hook_config.write_text(
            json.dumps(
                {
                    "case": "v4_four_layer_cp4_cache",
                    "function_hooks": [
                        {
                            "target": "rtp_llm.models_py.modules.dsv4.prefill.forward.forward_layers",
                            "after": "after_forward_layers",
                        },
                    ],
                }
            )
        )
        probe_file = Path(__file__).with_name("deepseek_v4_tail_reuse_probe.py")
        self.prefill = MagaServerManager(
            env_args={
                "LOAD_PYTHON_MODEL": "1",
                "LOG_LEVEL": "INFO",
                "FT_SERVER_TEST": "1",
                "DSV4_MOE_STRATEGY": "mega",
                "DSV4_USE_MEGA_MOE_SE": "0",
                "DSV4_USE_MEGA_MOE_FUSED": "0",
                "DSV4_CHUNK_TOKENS": "2048",
                "WORLD_SIZE": "4",
                "REMOTE_RPC_SERVER_IP": "localhost",
                "REMOTE_SERVER_PORT": unused_decode_port,
                "MODEL_SERVICE_CONFIG": route.model_dump_json(),
                "MOEDBG": "1",
                "MOEDBG_ALL_LAYERS": "1",
                "MOEDBG_MAX_SEQ": "2000",
                "MOEDBG_FULL_THRESHOLD": "5000000",
                "MOEDBG_NAME_REGEX": r"^(layer\d\d_(tail128|last)|lm_last_hidden|lm_logits_last)$",
                "MOEDBG_DIR": str(self.trace_dir),
                "MOEDBG_CASE": "four_layer_cp4",
                "RTP_HOT_HOOK": "1",
                "RTP_HOT_HOOK_FILE": str(probe_file),
                "RTP_HOT_HOOK_CONFIG": str(hook_config),
                "RTP_HOT_HOOK_DUMP_DIR": str(self.probe_dir),
                "T07_CACHE_PROBE_DIR": str(self.probe_dir),
                "T07_CACHE_PROBE_ROLE": "prefill",
                "T07_CACHE_PROBE_CP_SIZE": "4",
            },
            device_ids=cards[: self.cp_size],
            port=prefill_port,
            role_name="v4_tail_cp4_prefill",
            smoke_args_str=(
                "--tp_size 4 --ep_size 4 --dp_size 1 --world_size 4"
                " --act_type BF16 --fp8_kv_cache 1 --seq_size_per_block 256"
                " --kernel_seq_size_per_block 128 --max_seq_len 2048"
                " --max_context_batch_size 1 --concurrency_limit 1 --reuse_cache 1"
                " --load_method fastsafetensors --kv_cache_mem_mb 4096"
                " --reserver_runtime_mem_mb 12288 --cache_store_rdma_mode 0"
                " --use_local 1 --warm_up 0 --enable_device_cache 1"
                " --enable_memory_cache 0 --enable_disk_cache 0 --enable_remote_cache 0"
                " --prefill_cp_kv_cache_sharded 1 --prefill_cp_size 4"
                " --role_type PREFILL --enable_cuda_graph 0"
                " --cp_rotate_method ALL_GATHER --enable_prefill_partial_tail_reuse 1"
            ),
        )
        self.addCleanup(self.prefill.stop_server)
        self.assertTrue(
            self.prefill.start_server(
                model_path=str(self.model_path), model_type="deepseek_v4", timeout=3600
            )
        )

    def _request_cp4(self, label: str, prompt: str):
        before_probe = set(self.probe_dir.glob("prefill_rank*_*.pt"))
        before_trace = set(self.trace_dir.glob("four_layer_cp4/rank*_step*.pt"))
        query = {
            "prompt": prompt,
            "generate_config": {
                # The PREFILL role uses its local route for one generated token.
                "max_new_tokens": 1,
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
        self.assertIs(actual["aux_info"]["pd_sep"], False)
        self.assertEqual(
            len(actual["input_ids"][0]), len(self.tokenizer.encode(prompt))
        )
        self.assertTrue(actual["output_ids"][0])

        probe_files = set(self.probe_dir.glob("prefill_rank*_*.pt")) - before_probe
        trace_files = (
            set(self.trace_dir.glob("four_layer_cp4/rank*_step*.pt")) - before_trace
        )
        self.assertEqual(len(probe_files), self.cp_size)
        self.assertEqual(len(trace_files), self.cp_size)
        probes = {
            int(data["rank"]): data
            for data in (torch.load(path, weights_only=False) for path in probe_files)
        }
        traces = {
            int(data["extra"]["cp_rank"]): data
            for data in (torch.load(path, weights_only=False) for path in trace_files)
        }
        ranks = set(range(self.cp_size))
        self.assertEqual(set(probes), ranks)
        self.assertEqual(set(traces), ranks)
        record = {
            "input_len": len(actual["input_ids"][0]),
            "request_start_unix_ns": request_start_ns,
            "response_end_unix_ns": response_end_ns,
            "pd_sep": actual["aux_info"]["pd_sep"],
            "local_reuse_len": actual["aux_info"].get("local_reuse_len"),
            "output_ids": actual["output_ids"],
            "prefill_rank_capture_unix_ns": {
                rank: data["captured_unix_ns"] for rank, data in probes.items()
            },
        }
        (self.output_dir / f"{label}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        return actual, probes, traces, record

    def test_cp4_g1300_to_2000_local_prefill(self):
        prompt_a = self._extend_prompt("Tail reuse CP4 state test:", 1300)
        prompt_b = self._extend_prompt(prompt_a, 2000)
        self.assertEqual(
            self.tokenizer.encode(prompt_b)[:1300], self.tokenizer.encode(prompt_a)
        )
        a_actual, before, _, a_record = self._request_cp4("cp4_source_1300", prompt_a)
        b_actual, after, traces, b_record = self._request_cp4(
            "cp4_reused_2000", prompt_b
        )
        self.assertEqual(a_actual["aux_info"].get("local_reuse_len"), 0)
        self.assertEqual(b_actual["aux_info"].get("local_reuse_len"), 1300)
        self.assertLess(
            max(a_record["prefill_rank_capture_unix_ns"].values()),
            b_record["request_start_unix_ns"],
            "all four A ranks must finish before B can reuse the short leaf",
        )

        copy_map = {}
        for rank in range(self.cp_size):
            a = before[rank]
            b = after[rank]
            self.assertEqual(a["end_position"], 1300)
            self.assertEqual(b["end_position"], 2000)
            self.assertEqual(set(a["rows"]), set(b["rows"]))
            self.assertEqual({tag for _, tag in a["rows"]}, set(DSV4_TAGS))
            extra = traces[rank]["extra"]
            self.assertEqual(int(extra["cp_size"]), self.cp_size)
            self.assertEqual(int(extra["prefix_length"]), 1300)
            self.assertEqual(int(extra["seq_len_full"]), 700)
            self.assertEqual(int(extra["seq_len_total"]), 2000)
            for key, before_row in a["rows"].items():
                after_row = b["rows"][key]
                save = before_row["save_block_ids"][
                    before_row["save_positions"] == 1300
                ]
                self.assertEqual(save.numel(), 1, f"rank{rank} {key}: A short save")
                self.assertGreater(int(save[0]), 0, f"rank{rank} {key}: A save ID")
                self.assertTrue(
                    set(before_row["source_ids"].tolist())
                    - set(before_row["table_ids"].tolist()),
                    f"rank{rank} {key}: A short source row not captured",
                )
                for suffix in ("ids", "values", "scales"):
                    name = f"source_{suffix}"
                    if name in before_row:
                        self.assertTrue(
                            torch.equal(before_row[name], after_row[name]),
                            f"rank{rank} {key}: A source mutated: {name}",
                        )

                physical = int(before_row["seq_size_per_block"])
                pages = physical // int(before_row["kernel_seq_size_per_block"])
                slot = (1300 - 1) // physical
                if key[1] in DSV4_KERNEL_ROW_TAGS:
                    slot //= self.cp_size
                column = slot * pages
                self.assertLess(column, after_row["runtime_table"].shape[1])
                destination = int(after_row["runtime_table"][0, column]) // pages
                if destination <= 0:
                    continue
                self.assertNotEqual(
                    destination,
                    int(save[0]),
                    f"rank{rank} {key}: source reused for write",
                )
                copy_map.setdefault(str(key), []).append(
                    {
                        "rank": rank,
                        "a_runtime": int(before_row["runtime_table"][0, column])
                        // pages,
                        "a_save": int(save[0]),
                        "b_write": destination,
                    }
                )
        self.assertTrue(copy_map, "no copy destinations observed")
        for key, copies in copy_map.items():
            self.assertEqual(len(copies), self.cp_size, f"{key}: all ranks must copy")
            self.assertEqual(len({row["a_save"] for row in copies}), 1, key)
            self.assertEqual(len({row["b_write"] for row in copies}), 1, key)
        (self.output_dir / "copy_map_cp4_1300_2000.json").write_text(
            json.dumps(copy_map, indent=2) + "\n"
        )

        numeric = {}
        owner_ranks = []
        for rank in range(self.cp_size):
            path = self.probe_dir / f"direct_reference_rank{rank}_2000_g1300.pt"
            if not path.is_file():
                error = self.probe_dir / f"direct_reference_error_rank{rank}.txt"
                self.fail(
                    error.read_text()
                    if error.is_file()
                    else f"rank{rank}: direct reference missing"
                )
            direct = torch.load(path, weights_only=False)
            trace = traces[rank]
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
            if direct["final_last"].numel():
                owner_ranks.append(rank)
            for name, reference in comparisons:
                resumed = trace["tensors"][name]
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
            len(owner_ranks), 1, "one rank must own the final prompt token"
        )
        (self.output_dir / "numeric_diffs_cp4_1300_2000.json").write_text(
            json.dumps(numeric, indent=2) + "\n"
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
