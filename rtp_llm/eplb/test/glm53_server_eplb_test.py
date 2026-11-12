import json
import os
import re
import shlex
import time
import unittest
from pathlib import Path

import numpy as np
import requests
from transformers import PreTrainedTokenizerFast

from rtp_llm.test.smoke.eplb_smoke_checks import check_eplb_applied
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


class Glm53ServerEplbTest(unittest.TestCase):
    def test_full_model_runtime_rearrangement(self):
        checkpoint = os.environ["EPLB_GLM5_CHECKPOINT"]
        output = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(Path(checkpoint) / "tokenizer.json"),
            chat_template=(Path(checkpoint) / "chat_template.jinja").read_text(),
            eos_token="<|endoftext|>",
            pad_token="<|endoftext|>",
        )
        prompts = [
            "What is the capital of France? Answer in one sentence.",
            "Write a Python function that adds two integers, then explain it.",
            "Explain why the sky looks blue in about 100 words.",
        ]
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for prompt in prompts
        ]
        records = []
        reference = {}
        modes = os.environ.get("EPLB_GLM5_MODES", "NONE,EPLB").split(",")
        mtp = os.environ.get("EPLB_GLM5_MTP", "0") == "1"
        world = int(os.environ.get("EPLB_GLM5_WORLD", "8"))
        tp = int(os.environ.get("EPLB_GLM5_TP", "1"))
        self.assertEqual(world % tp, 0)
        self.assertEqual(len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")), world)
        for mode in modes:
            args = [
                "--frontend_server_count",
                "1",
                "--shutdown_timeout",
                "5",
                "--tp_size",
                str(tp),
                "--dp_size",
                str(world // tp),
                "--ep_size",
                str(world),
                "--world_size",
                str(world),
                "--use_all_gather",
                "0",
                "--use_deepep_moe",
                "0",
                "--use_deepep_low_latency",
                "0",
                "--act_type",
                "BF16",
                "--fp8_kv_cache",
                "1",
                "--quantization",
                "FP8_PER_BLOCK_NO_MOE",
                "--enable_glm52_shared_indexer_kv_cache",
                "1",
                "--moe_strategy",
                "mega_moe_se",
                "--load_method",
                "scratch",
                "--enable_cuda_graph",
                os.environ.get("EPLB_GLM5_GRAPH", "1"),
                "--decode_capture_config",
                "1,2,4",
                "--warm_up",
                "0",
                "--max_seq_len",
                "2048",
                "--concurrency_limit",
                "4",
                "--max_batch_tokens_size",
                "2048",
                "--seq_size_per_block",
                "64",
                "--kernel_seq_size_per_block",
                "64",
                "--reserver_runtime_mem_mb",
                "8192",
                "--kv_cache_mem_mb",
                "2048",
                "--load_cache_timeout_ms",
                "600000",
                "--reuse_cache",
                "0",
                "--redundant_expert",
                os.environ.get("EPLB_GLM5_REDUNDANT", "0"),
                "--eplb_mode",
                mode,
                "--eplb_update_time",
                "32",
                "--eplb_stats_window_size",
                "16",
                "--eplb_metrics_report_interval",
                "16",
                "--eplb_metrics_snapshot_max_files",
                "64",
            ]
            if mtp:
                args.extend(
                    [
                        "--sp_model_type",
                        "glm_5_mtp",
                        "--sp_type",
                        "eagle",
                        "--sp_checkpoint_path",
                        checkpoint,
                        "--sp_act_type",
                        "BF16",
                        "--gen_num_per_cycle",
                        "3",
                    ]
                )
            server = MagaServerManager(
                role_name=mode.lower(),
                env_args={
                    "RTP_LLM_STREAM_ASYNC": "1",
                    "RTP_LLM_GLM5_CMP": "1",
                    "RTP_LLM_GLM5_MEGA_GATE": "1",
                    "RTP_LLM_GLM5_GATE_PACK": "1",
                    "FRONTEND_PRE_STOP_DRAIN_SECONDS": "0",
                },
                smoke_args_str=shlex.join(args),
            )
            try:
                self.assertTrue(
                    server.start_server(
                        model_path=checkpoint, model_type="glm_5", timeout=1800
                    )
                )
                for repeat in range(2):
                    for index, prompt in enumerate(rendered):
                        for max_tokens in (1, 64):
                            started = time.perf_counter()
                            response = requests.post(
                                f"http://127.0.0.1:{server.port}/",
                                json={
                                    "prompt": prompt,
                                    "generate_config": {
                                        "max_new_tokens": max_tokens,
                                        "top_k": 1,
                                        "top_p": 1.0,
                                        "random_seed": 123,
                                        "is_streaming": False,
                                        "return_logits": True,
                                        "return_output_ids": True,
                                        "return_input_ids": True,
                                    },
                                },
                                timeout=600,
                            )
                            response.raise_for_status()
                            result = response.json()
                            path = output / f"{mode}_{repeat}_{index}_{max_tokens}.json"
                            path.write_text(json.dumps(result, ensure_ascii=False))
                            if (
                                max_tokens == 1
                                and os.environ.get("EPLB_GLM5_PREFIX_PROBE") == "1"
                            ):
                                probe = requests.post(
                                    f"http://127.0.0.1:{server.port}/",
                                    json={
                                        "prompt": prompt + result["response"],
                                        "generate_config": {
                                            "max_new_tokens": 1,
                                            "top_k": 1,
                                            "top_p": 1.0,
                                            "is_streaming": False,
                                            "return_logits": True,
                                            "return_output_ids": True,
                                            "return_input_ids": True,
                                        },
                                    },
                                    timeout=600,
                                )
                                probe.raise_for_status()
                                (
                                    output
                                    / f"{mode}_{repeat}_{index}_prefix_probe.json"
                                ).write_text(
                                    json.dumps(probe.json(), ensure_ascii=False)
                                )
                            records.append(
                                {
                                    "mode": mode,
                                    "repeat": repeat,
                                    "prompt": prompts[index],
                                    "max_tokens": max_tokens,
                                    "artifact": path.name,
                                    "wall_ms": (time.perf_counter() - started) * 1000,
                                }
                            )
                            (output / "requests.json").write_text(
                                json.dumps(records, indent=2)
                            )
                            self.assertNotIn("error", result, result)
                            self.assertTrue(result["finished"], result)
                            self.assertTrue(result["output_ids"], result)
                            logits = np.asarray(result["logits"], dtype=np.float32)
                            self.assertGreater(
                                logits.size, 100000, "Missing full vocabulary logits"
                            )
                            self.assertTrue(
                                np.isfinite(logits).all(), "Nonfinite logits"
                            )
                            if max_tokens > 1:
                                text = result["response"].strip()
                                self.assertTrue(text)
                                self.assertNotRegex(text, r"(?s)^(.{1,32})\1{7,}$")
                                if index == 0:
                                    self.assertIn("Paris", text)
                            key = (index, max_tokens)
                            if key not in reference:
                                reference[key] = result
                            else:
                                expected = reference[key]
                                self.assertEqual(
                                    result["input_ids"], expected["input_ids"]
                                )
                                self.assertEqual(
                                    result["output_ids"], expected["output_ids"]
                                )
                                np.testing.assert_allclose(
                                    logits,
                                    np.asarray(expected["logits"], dtype=np.float32),
                                    rtol=0,
                                    atol=0,
                                )
                if mode in ("EPLB", "ALL"):
                    check_eplb_applied(output / f"{mode.lower()}_logs", world)
                    port = server.port
                    prompt = rendered[0]
                    probe = requests.post(
                        f"http://127.0.0.1:{port}/",
                        json={
                            "prompt": prompt,
                            "generate_config": {
                                "max_new_tokens": 128,
                                "top_k": 1,
                                "is_streaming": False,
                                "return_output_ids": True,
                            },
                        },
                        timeout=900,
                    )
                    probe.raise_for_status()
                    result = probe.json()
                    (output / f"{mode}_post_migration.json").write_text(
                        json.dumps(result)
                    )
                    self.assertTrue(result["finished"])
                    self.assertTrue(result["output_ids"])
                    self.assertIn("Paris", result["response"])
            finally:
                server.stop_server()
                (output / f"{mode}_config.json").write_text(
                    json.dumps(
                        {
                            "checkpoint": checkpoint,
                            "args": args,
                            "mtp": mtp,
                        },
                        indent=2,
                    )
                )
            log_dir = output / f"{mode.lower()}_logs"
            logs = "\n".join(
                p.read_text(errors="replace") for p in log_dir.rglob("*.log")
            )
            snapshots = [
                json.loads(p.read_text()) for p in log_dir.rglob("eplb_snapshot/*.json")
            ]
            self.assertTrue(snapshots, "No native load snapshots")
            self.assertTrue(any(s["total_activations"] > 0 for s in snapshots))
            self.assertTrue(all(s["model_type"] == "target" for s in snapshots))
            if mode in ("EPLB", "ALL"):
                applied = re.findall(
                    r"\[EPLB\] applied .*?update=(\d+) changed_slots=\[([^\]]*)\] ep_rank=(\d+)",
                    logs,
                )
                self.assertEqual(
                    {int(rank) for _, _, rank in applied}, set(range(world))
                )
                for rank in range(world):
                    updates = [
                        int(n)
                        for n, slots, r in applied
                        if int(r) == rank
                        and any(int(x.strip()) > 0 for x in slots.split(","))
                    ]
                    self.assertGreaterEqual(len(set(updates)), 2, (rank, applied))
            else:
                self.assertNotIn("[EPLB] applied", logs)


if __name__ == "__main__":
    unittest.main()
