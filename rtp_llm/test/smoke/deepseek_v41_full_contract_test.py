"""GB200 CP2/DP2 P/D smoke with CED, DSpark, Engram and verify graph reuse."""

import json
import logging
import os
import unittest
from pathlib import Path

from rtp_llm.test.smoke import deepseek_v41_dspark_contract_test as dspark
from rtp_llm.test.smoke import deepseek_v41_pd_contract_test as pd
from rtp_llm.test.smoke.v41_startup_memory import record_v41_startup_events


class DeepSeekV41FullContractTest(
    dspark.DeepSeekV41DSparkContractTest, pd.DeepSeekV41PDContractTest
):
    prefill_cp_size = 2
    decode_dp_size = 2
    short_prompt = (
        "Write the numbers from one to twenty, separated by commas: one, two,"
    )
    # Main's speculative output reports acceptance, not generation scores.
    # Keep the PD token/text repeatability and CED checks, and the DSpark
    # acceptance and verify-graph checks inherited from the two runners.
    compare_generation_logits = False

    def start_servers(self):
        with record_v41_startup_events(self.output_dir):
            super().start_servers()

    def _record_long_prefill(self, expected, generated):
        # Single-token requests stay on the prefill engine (PrefillRpcServer).
        # Reuse the exact long prompt and compare the two competing IDs BEFORE
        # P/D transfer or draft verification. Only run after all cache checks.
        old_id, new_id = (row["output_ids"][0][0] for row in (expected, generated))
        if old_id == new_id:
            return
        query = generated["query"].copy()
        query["generate_config"] = dict(
            query["generate_config"],
            max_new_tokens=1,
            can_use_pd_separation=False,
            reuse_cache=False,
            return_logits=True,
            select_tokens_id=[old_id, new_id],
        )
        response = self.session.post(
            f"http://127.0.0.1:{self.server.port}/", json=query, timeout=120
        )
        response.raise_for_status()
        actual = response.json()
        info = actual["aux_info"]
        summary = {
            "candidate_ids_golden_then_pd": [old_id, new_id],
            "prefill_output_ids": actual["output_ids"],
            "prefill_candidate_logits": actual.get("logits"),
            "input_len": info["input_len"],
            "pd_sep": info["pd_sep"],
            "same_input_ids": actual["input_ids"]
            == json.loads((self.output_dir / "long_uncached.json").read_text())[
                "actual"
            ]["input_ids"],
        }
        (self.output_dir / "long_first_token_prefill.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )

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
            expected = json.loads(golden.read_text())
            if expected["long_uncached"] != actual["long_uncached"]:
                try:
                    self._record_long_prefill(
                        expected["long_uncached"], actual["long_uncached"]
                    )
                except Exception:
                    logging.exception("Could not collect the long prefill comparison")
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
