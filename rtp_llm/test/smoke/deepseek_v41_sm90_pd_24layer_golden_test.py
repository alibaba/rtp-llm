"""Reduced-layer EP4 P/D determinism regression, not full-model accuracy.

CED compaction requires 40 layers and is intentionally not claimed here.
"""

import json
import logging
import unittest
from pathlib import Path

from rtp_llm.test.smoke import deepseek_v41_contract_test as contract
from rtp_llm.test.smoke import deepseek_v41_sm90_pd_contract_test as pd_contract


class DeepSeekV41Sm90PD24LayerGoldenTest(pd_contract.DeepSeekV41Sm90PDContractTest):
    extra_server_env = {
        **pd_contract.DeepSeekV41Sm90PDContractTest.extra_server_env,
        "HACK_LAYER_NUM": "24",
    }

    def test_graph_engram_and_prefix_reuse(self):
        # Override the full-model test: this fixture covers repeated uncached
        # P/D generation, not cross-path full-prefill or CED/prefix equivalence.
        golden = json.loads(
            Path(__file__)
            .with_name("deepseek_v41_sm90_pd_24layer_golden.json")
            .read_text()
        )
        self.assertEqual(golden["layers"], 24)
        self.assertEqual(golden["prompt"], self.short_prompt)
        self.assertEqual(golden["max_new_tokens"], self.max_new_tokens)
        for attempt in range(3):
            result = self.request(f"golden_repeat_{attempt}", self.short_prompt, False)
            self.assertEqual(result["aux_info"]["prefill_total_reuse_len"], 0)
            for field in ("input_ids", "output_ids", "response"):
                self.assertEqual(
                    result[field], golden[field], f"repeat {attempt}: {field}"
                )
        contract.DeepSeekV41ContractTest.assert_decode_graph_replayed_and_record_coverage(
            self
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
