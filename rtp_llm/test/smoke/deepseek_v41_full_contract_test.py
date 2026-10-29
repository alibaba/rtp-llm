"""GB200 CP2/DP2 P/D smoke with CED, DSpark, Engram and verify graph reuse."""

import json
import logging
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.smoke import deepseek_v41_dspark_contract_test as dspark
from rtp_llm.test.smoke import deepseek_v41_pd_contract_test as pd


class DeepSeekV41FullContractTest(
    dspark.DeepSeekV41DSparkContractTest, pd.DeepSeekV41PDContractTest
):
    prefill_cp_size = 2
    decode_dp_size = 2
    jit_cache_root = "/tmp/rtp-llm/.jit_cache_deepseek_v41_flash_ced_dspark_pd_cp2_dp2_sm100_arm"
    short_prompt = (
        "Write the numbers from one to twenty, separated by commas: one, two,"
    )
    # Main's speculative output reports acceptance, not generation scores.
    # Keep the PD token/text repeatability and CED checks, and the DSpark
    # acceptance and verify-graph checks inherited from the two runners.
    compare_generation_logits = False

    def start_servers(self):
        with patch.dict(os.environ):
            os.environ.pop("DG_JIT_CACHE_DIR", None)
            super().start_servers()

    def test_graph_engram_and_prefix_reuse(self):
        super().test_graph_engram_and_prefix_reuse()
        actual = {}
        for record in self.observed:
            label = record["label"]
            result = json.loads((self.output_dir / f"{label}.json").read_text())
            actual[label] = {
                "query": result["query"],
                "output_ids": result["actual"]["output_ids"],
                "response": result["actual"]["response"],
                "input_len": record["input_len"],
                "prefill_reuse_len": record["measured_reuse_len"],
            }
        (self.output_dir / "full_feature_golden_actual.json").write_text(
            json.dumps(actual, ensure_ascii=False, indent=2) + "\n"
        )
        if os.environ.get("SAVE_RESPONSE", "False") != "True":
            golden = (
                Path(__file__).parent
                / "data/model/deepseek_v41/full_feature_cp2_dp2.json"
            )
            self.assertEqual(actual, json.loads(golden.read_text()))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
