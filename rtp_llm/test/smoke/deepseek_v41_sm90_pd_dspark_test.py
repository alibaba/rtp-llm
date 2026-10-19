"""Real P/D DSpark contract, reusing the shared generation and routing checks."""

import concurrent.futures
import json
import logging
import os
import unittest
from pathlib import Path

from rtp_llm.test.smoke import deepseek_v41_contract_test as contract
from rtp_llm.test.smoke import deepseek_v41_dspark_contract_test as dspark
from rtp_llm.test.smoke import deepseek_v41_sm90_pd_contract_test as pd


class DeepSeekV41Sm90PDDSparkTest(
    dspark.DeepSeekV41DSparkContractTest, pd.DeepSeekV41Sm90PDContractTest
):
    def setUp(self):
        from rtp_llm.test.perf_test.deepseek_v41_sm90_role_test import (
            configure_jit_cache,
        )

        configure_jit_cache()
        self.mode = os.environ.get("DSV41_PD_SMOKE_MODE", "dspark_graph")
        self.assertIn(self.mode, ("dspark_graph", "normal_graph", "dspark_eager"))
        self.speculative = self.mode != "normal_graph"
        self.decode_cuda_graph = self.mode != "dspark_eager"
        self.require_graph_capture = self.decode_cuda_graph
        if not self.speculative:
            self.graph_target_verify = 0
            self.graph_is_prefill = 0
        raw = json.loads(
            (Path(os.environ["CHECKPOINT_PATH"]) / "config.json").read_text()
        )
        self.target_layers = raw.get("text_config", raw)["num_hidden_layers"]
        if self.mode != "dspark_graph":
            self.assertEqual(self.target_layers, 24)
        # CED deliberately supports only the full architecture. Reduced-layer
        # fixtures validate P/D DSpark determinism against the existing golden.
        super().setUp()
        (self.output_dir / "dspark_fixture.json").write_text(
            json.dumps(
                {
                    "mode": self.mode,
                    "target_layers": self.target_layers,
                    "ced_compaction_required": self.target_layers == 40,
                    "full_model_acceptance_valid": self.target_layers == 40,
                },
                indent=2,
            )
        )

    def server_args(self):
        args = super().server_args()
        return args if self.speculative else args + " --sp_type none"

    def request(self, label, prompt, reuse):
        if self.target_layers == 40:
            return super().request(label, prompt, reuse)
        # A draft trained against layers 37-39 is not calibrated to 21-23.
        # Preserve real rejection/fallback; never force acceptance in a fixture.
        actual = contract.DeepSeekV41ContractTest.request(self, label, prompt, reuse)
        aux = actual["aux_info"]
        if not self.speculative:
            self.assertEqual(aux.get("speculative_draft_rounds", 0), 0)
            return actual
        rounds = aux["speculative_draft_rounds"]
        accepted = aux["speculative_accepted_tokens_per_pos"]
        self.assertGreater(rounds, 0)
        self.assertTrue(accepted)
        self.assertTrue(all(0 <= n <= rounds for n in accepted))
        observed = next(item for item in self.observed if item["label"] == label)
        observed.update(
            speculative_draft_rounds=rounds,
            speculative_accepted_tokens_per_pos=accepted,
        )
        return actual

    def test_graph_engram_and_prefix_reuse(self):
        if self.target_layers == 40:
            return super().test_graph_engram_and_prefix_reuse()
        # Explicit reduced-model contract, matching the existing 24-layer
        # fixture boundary. No full-prefill, CED or release accuracy claim.
        self.assertEqual(self.target_layers, 24)
        golden = json.loads(
            Path(__file__)
            .with_name("deepseek_v41_sm90_pd_24layer_golden.json")
            .read_text()
        )
        self.assertEqual(golden["layers"], 24)
        self.assertEqual(golden["prompt"], self.short_prompt)
        self.assertEqual(golden["max_new_tokens"], self.max_new_tokens)
        first = self.request("reduced_uncached_0", self.short_prompt, False)
        for field in ("input_ids", "output_ids", "response"):
            with self.subTest(stage="golden", field=field):
                self.assertEqual(first[field], golden[field], field)
        for repeat in range(1, 3):
            actual = self.request(
                f"reduced_uncached_{repeat}", self.short_prompt, False
            )
            with self.subTest(stage="sequential", repeat=repeat):
                self.assert_same(first, actual)
        # Repeated concurrent requests exercise changing verification batches.
        # Keep the existing golden; concurrency must not redefine correctness.
        for repeat in range(2):
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
                futures = [
                    pool.submit(
                        self.request,
                        f"reduced_concurrent_{repeat}_{index}",
                        self.short_prompt,
                        False,
                    )
                    for index in range(16)
                ]
                for index, future in enumerate(futures):
                    actual = future.result()
                    with self.subTest(stage="concurrent", repeat=repeat, index=index):
                        self.assert_same(first, actual)
        if self.speculative:
            self.assertTrue(
                any(r["speculative_draft_rounds"] > 1 for r in self.observed)
            )
        if self.decode_cuda_graph:
            with Path(self.graph_server.log_file_path).open("rb") as reader:
                reader.seek(self.log_offset)
                runtime = reader.read().decode(errors="replace")
            flag = int(self.speculative)
            self.assertRegex(
                runtime,
                rf"using CUDA graph forward, is_target_verify={flag}, is_prefill={flag}, graph_bs=[24]\b",
                "Concurrent requests did not exercise a multi-request graph",
            )
            contract.DeepSeekV41ContractTest.assert_decode_graph_replayed_and_record_coverage(
                self
            )
        else:
            (self.output_dir / "v41_eager_coverage.json").write_text(
                json.dumps({"mode": self.mode, "requests": self.observed}, indent=2)
            )

    # specUpdate does not publish generation logits or cumulative scores.
    prefill_concurrency_limit = 16
    compare_generation_logits = False
    extra_server_env = {
        **pd.DeepSeekV41Sm90PDContractTest.extra_server_env,
        "DSV41_SM90_DECODE_GRAPH_MAX_ROWS": "16",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
