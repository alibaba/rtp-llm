"""Manual real P/D MTP tail-reuse acceptance with separate target/draft oracles."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch
from transformers import AutoTokenizer

from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
from rtp_llm.test.smoke.deepseek_v4_tail_reuse_4layer_test import (
    DeepSeekV4TailReuseFourLayerTest,
)
from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class DeepSeekV4TailReuseFourLayerMTPTest(DeepSeekV4TailReuseFourLayerTest):
    def setUp(self):
        self.model_path = Path(os.environ["T07_FOUR_LAYER_MTP_CHECKPOINT"])
        self.assertTrue((self.model_path / "model.safetensors.index.json").is_file())
        config = json.loads((self.model_path / "config.json").read_text())
        self.assertEqual(config["num_hidden_layers"], 4)
        self.assertEqual(config["compress_ratios"], [0, 0, 4, 128, 0])
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
            service_id="v4_tail_reuse_four_layer_mtp",
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
            " --load_method fastsafetensors --kv_cache_mem_mb 2048"
            " --reserver_runtime_mem_mb 8192 --cache_store_rdma_mode 0"
            " --use_local 1 --load_cache_timeout_ms 120000 --warm_up 0"
            " --enable_device_cache 1 --enable_memory_cache 0"
            " --enable_disk_cache 0 --enable_remote_cache 0"
            f" --sp_type mtp --sp_model_type deepseek_v4_mtp --sp_checkpoint_path {self.model_path}"
            " --sp_act_type bf16 --gen_num_per_cycle 3"
        )
        hook_config = self.output_dir / "tail_reuse_mtp_hook_config.json"
        hook_config.write_text(
            json.dumps(
                {
                    "case": "v4_four_layer_mtp_cache",
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
            "deepseek_v4_tail_reuse_4layer_mtp_probe.py"
        )
        self.assertTrue(probe_file.is_file())
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
            role_name="v4_tail_mtp_prefill",
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
            role_name="v4_tail_mtp_decode",
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
                self.assertTrue(start.result(), f"{role} MTP server startup failed")

    def _request_mtp(self, label: str, prompt: str):
        before = set(self.probe_dir.glob("mtp_draft_decode_*.pt"))
        draft_dir = self.probe_dir / "mtp_draft_prefill"
        before_prefill = set(draft_dir.glob("prefill_*.pt"))
        before_trace = set(
            (self.trace_dir / "mtp_draft_prefill").glob("rank*_step*.pt")
        )
        response, probe, trace, decode_tables = self._request(label, prompt, True)
        draft_prefills = set(draft_dir.glob("prefill_*.pt")) - before_prefill
        draft_traces = (
            set((self.trace_dir / "mtp_draft_prefill").glob("rank*_step*.pt"))
            - before_trace
        )
        self.assertEqual(
            len(draft_prefills), 1, f"{label}: MTP draft prefill physical snapshot"
        )
        self.assertEqual(
            len(draft_traces), 1, f"{label}: MTP draft prefill layer trace"
        )
        self._draft_prefill_by_label[label] = (
            torch.load(next(iter(draft_prefills)), weights_only=False),
            torch.load(next(iter(draft_traces)), weights_only=False),
        )
        draft_files = sorted(set(self.probe_dir.glob("mtp_draft_decode_*.pt")) - before)
        self.assertTrue(draft_files, f"{label}: no real MTP draft decode forward")
        checks = []
        for file in draft_files:
            draft = torch.load(file, weights_only=False)
            self.assertTrue(draft["source"], f"{label}: empty draft physical snapshot")
            self.assertTrue(
                all(draft["physical_equal"].values()), f"{label}: draft cache"
            )
            for name, diff in draft["compared"].items():
                self.assertLessEqual(diff, 0.002, f"{label}: {name}")
            checks.append({"file": file.name, "compared": draft["compared"]})
        (self.output_dir / f"{label}_draft_oracles.json").write_text(
            json.dumps(checks, indent=2) + "\n"
        )
        return response, probe, trace, decode_tables

    def _assert_draft_prefill_pair(self, source, after, trace):
        self.assertEqual(source["end_position"], 1300)
        self.assertEqual(after["end_position"], 1320)
        self.assertEqual(set(source["rows"]), {(0, "swa_kv")})
        for key, before_row in source["rows"].items():
            current = after["rows"][key]
            save_ids = before_row["save_block_ids"][
                before_row["save_positions"] == 1300
            ]
            self.assertEqual(save_ids.numel(), 1, "draft N1300 save point")
            self.assertGreaterEqual(int(save_ids[0]), 0)
            self.assertTrue(
                set(before_row["source_ids"].tolist())
                - set(before_row["table_ids"].tolist()),
                "draft short save source was not captured",
            )
            end_ids = current["current_save_block_ids"][
                current["current_save_positions"] == 1320
            ]
            self.assertEqual(end_ids.numel(), 1, "draft N1320 save point")
            self.assertGreaterEqual(int(end_ids[0]), 0)
            for suffix in ("ids", "values", "scales"):
                name = f"source_{suffix}"
                if name in before_row:
                    self.assertTrue(
                        torch.equal(before_row[name], current[name]),
                        f"draft source mutated: {key} {name}",
                    )
        self.assertEqual(int(trace["extra"]["prefix_lengths"][0]), 1300)
        self.assertEqual(int(trace["extra"]["positions"][0]), 1300)
        self.assertEqual(int(trace["extra"]["positions"][-1]), 1319)
        self.assertEqual(int(trace["extra"]["input_ids"].numel()), 20)
        path = self.probe_dir / "mtp_draft_prefill/direct_reference_1320_g1300.pt"
        if not path.is_file():
            error = self.probe_dir / "mtp_draft_prefill/direct_reference_error.txt"
            self.fail(
                error.read_text()
                if error.is_file()
                else "MTP draft direct reference absent"
            )
        direct = torch.load(path, weights_only=False)
        compared = {}
        for name, expected in (
            ("layer00_last", direct["layers"][0]),
            ("lm_last_hidden", direct["final_last"]),
            ("lm_logits_last", direct["logits_last"]),
        ):
            actual = trace["tensors"][name].float()
            expected = expected.float()
            self.assertEqual(actual.shape, expected.shape, name)
            compared[name] = float((actual - expected).abs().max())
            with self.subTest(mode="mtp_draft_prefill", tensor=name):
                torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)
        (self.output_dir / "mtp_draft_prefill_numeric_diffs.json").write_text(
            json.dumps(compared, indent=2) + "\n"
        )

    def test_mtp_g1300_to_1320_target_and_draft(self):
        self._draft_prefill_by_label = {}
        prompt_a = self._extend_prompt("Tail reuse MTP independent state test:", 1300)
        prompt_b = self._extend_prompt(prompt_a, 1320)
        self.assertEqual(
            self.tokenizer.encode(prompt_b)[:1300], self.tokenizer.encode(prompt_a)
        )
        _, source, _, _ = self._request_mtp("mtp_source_1300", prompt_a)
        reused, after, trace, decode_tables = self._request_mtp(
            "mtp_reused_1320", prompt_b
        )
        self._assert_pair(
            "mtp_reused_1320", 1320, 1300, source, after, reused, trace, decode_tables
        )
        self._assert_draft_prefill_pair(
            self._draft_prefill_by_label["mtp_source_1300"][0],
            self._draft_prefill_by_label["mtp_reused_1320"][0],
            self._draft_prefill_by_label["mtp_reused_1320"][1],
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
