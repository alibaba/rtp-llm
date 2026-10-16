"""Hopper graph contract: real weights, existing golden, Engram and cache reuse.

DSpark and PD are not exercised by this PDFUSION contract.
CHECKPOINT_PATH is supplied by the test environment, not a personal BUILD path.
"""

import json
import logging
import os
import shutil
import unittest
from pathlib import Path

from rtp_llm.test.smoke.deepseek_v41_contract_test import DeepSeekV41ContractTest


class DeepSeekV41Sm90ContractTest(DeepSeekV41ContractTest):
    require_graph_capture = True
    moe_strategy = "sm90_humming"
    max_new_tokens = 32
    short_prompt = (
        "Write the numbers from one to twenty, separated by commas: one, two,"
    )
    extra_server_env = {
        "MOE_STRATEGY": "auto",
        "USE_DEEPEP_MOE": "1",
        "USE_ALL_GATHER": "0",
        "USE_DEEPEP_LOW_LATENCY": "0",
        "USE_DEEPEP_INTERNODE": "0",
        "ENABLE_CUDA_GRAPH": "1",
        "DSV41_SM90_DECODE_GRAPH": "1",
        "DSV41_SM90_DECODE_GRAPH_MAX_ROWS": "4",
        "DSV41_SM90_NATIVE_A8": "1",
        "DSV41_SM90_A8_PROMOTION_GROUP": "32",
        "DSV41_ENGRAM_HOST_LOOKUP": "uva",
        "DSV41_ENGRAM_PREFETCH": "1",
        "DSV4_MOE_CHUNK_TOKENS": "4096",
    }

    def test_graph_engram_and_prefix_reuse(self):
        super().test_graph_engram_and_prefix_reuse()
        # Opt-in local long-context regression, using the same managed server
        # and independent full-prefill oracle as the committed short smoke.
        # This does not replace the original golden or prefix-cache assertions.
        target = int(os.environ.get("DSV41_SM90_LONG_SMOKE_TOKENS", "0"))
        if not target:
            return
        self.assertGreaterEqual(target, 32768)
        unit = "The archive stores books, maps, and letters from a long journey.\n"
        tokens = self.tokenizer.encode(unit * (target // 8 + 1))[: target - 64]
        prompt = self.tokenizer.decode(tokens, skip_special_tokens=False)
        prompt += (
            "\nWrite the numbers from one to twenty, separated by commas: one, two,"
        )
        length = len(self.tokenizer.encode(prompt))
        self.assertGreaterEqual(length, target - 128)
        self.assertLessEqual(length, target)
        original_output = self.output_dir
        self.output_dir = original_output / "long_context"
        self.output_dir.mkdir(exist_ok=True)
        try:
            first = self.request("long_context_uncached", prompt, False)
            self.assert_decode_matches_full_prefill(first)
            self.assert_same(
                first, self.request("long_context_repeat_uncached", prompt, False)
            )
        finally:
            self.output_dir = original_output
        self.assert_decode_graph_replayed_and_record_coverage()

    def start_servers(self):
        import torch

        if not torch.version.cuda or int(torch.version.cuda.split(".")[0]) != 13:
            raise RuntimeError("SM90 baseline smoke requires the CUDA13 runtime")
        if torch.cuda.device_count() != 8 or any(
            torch.cuda.get_device_capability(rank) != (9, 0) for rank in range(8)
        ):
            raise RuntimeError("SM90 TP8/EP8 smoke requires eight locked SM90 GPUs")
        # Avoid loading hundreds of GiB before discovering a known tmpfs limit.
        raw = json.loads((Path(self.model_path) / "config.json").read_text())
        config = raw.get("text_config", raw)
        rows = sum(config["engram_num_embeddings"])
        dim = int(config["engram_head_dim"])
        required = rows * (dim + dim // 32)
        capacity = shutil.disk_usage("/dev/shm").total
        if capacity < required:
            raise RuntimeError(
                f"Engram requires {required / 1024**3:.2f} GiB shared table storage, "
                f"but /dev/shm capacity is {capacity / 1024**3:.2f} GiB; "
                "increase container shared memory (256 GiB recommended). "
                "No model forward has run."
            )
        super().start_servers()
        startup = Path(self.graph_server.log_file_path).read_text(errors="replace")
        self.assertIn("native FP8 WGMMA", startup)
        self.assertNotIn("E8M0 decoded for BF16 MMA", startup)
        self.assertIn("SM90 Engram UVA prefetch enabled", startup)
        self.assertNotIn("Engram shared CPU lookup enabled", startup)

    def request(self, label, prompt, reuse):
        actual = super().request(label, prompt, reuse)
        if prompt == self.short_prompt:
            # Reuse the committed SM100 golden without changing it. PD/DSpark
            # metadata differs, so compare the shared greedy token contract.
            golden_path = (
                Path(__file__).parent
                / "data/model/deepseek_v41/full_feature_cp2_dp2.json"
            )
            golden = json.loads(golden_path.read_text())["short_uncached"]
            self.assertEqual(prompt, golden["query"]["prompt"])
            expected = golden["output_ids"][0]
            self.assertEqual(actual["output_ids"][0][: len(expected)], expected)
        return actual

    def assert_decode_graph_replayed_and_record_coverage(self):
        super().assert_decode_graph_replayed_and_record_coverage()
        (self.output_dir / "v41_sm90_coverage.json").write_text(
            json.dumps(
                {
                    "requests": self.observed,
                    "backend": "sm90_humming",
                    "execution": "decode_cuda_graph",
                    "server_args": self.server_args(),
                    "native_a8_requested": True,
                    "engram": "uva_prefetch",
                    "engram_checkpoint_layers": True,
                    "reference": "committed SM100 golden token prefix",
                    "dspark_vision_pd": "NOT_RUN",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Importing the base class must not also launch the SM100 test case.
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        DeepSeekV41Sm90ContractTest
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
