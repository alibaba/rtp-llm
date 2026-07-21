"""GB200 CP2/DP2 P/D smoke with CED, DSpark, Engram and verify graph reuse."""

import json
import logging
import math
import os
import unittest
from pathlib import Path

from rtp_llm.test.smoke import deepseek_v41_dspark_contract_test as dspark
from rtp_llm.test.smoke import deepseek_v41_pd_contract_test as pd


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
        self.assert_openai_logprobs()

    def assert_openai_logprobs(self):
        """Exercise token-aligned target logprobs through CP2/DP2 and DSpARK."""
        query = {
            "messages": [
                {"role": "user", "content": "Count from one to ten in English."}
            ],
            "max_tokens": 24,
            "temperature": 0,
            "enable_thinking": False,
            "logprobs": True,
            "top_logprobs": 3,
            "extra_configs": {
                "top_k": 1,
                "random_seed": 42,
                "reuse_cache": False,
                "can_use_pd_separation": True,
            },
        }
        for streaming in (False, True):
            query["stream"] = streaming
            response = self.session.post(
                f"http://127.0.0.1:{self.server.port}/v1/chat/completions",
                json=query,
                stream=streaming,
                timeout=600,
            )
            self.assertEqual(
                response.status_code,
                200,
                response.text if not streaming else response.reason,
            )
            if streaming:
                chunks = []
                for line in response.iter_lines():
                    if not line.startswith(b"data: "):
                        continue
                    payload = line[6:]
                    if payload == b"[DONE]":
                        break
                    chunks.append(json.loads(payload))
                choices = [
                    chunk["choices"][0] for chunk in chunks if chunk.get("choices")
                ]
                content = "".join(
                    choice.get("delta", {}).get("content") or "" for choice in choices
                )
                actual = chunks
            else:
                actual = response.json()
                choices = actual["choices"]
                self.assertEqual(len(choices), 1)
                self.assertTrue(choices[0]["finish_reason"])
                content = choices[0]["message"]["content"]
            records = [
                record
                for choice in choices
                for record in (choice.get("logprobs") or {}).get("content", [])
            ]
            (self.output_dir / f"openai_logprobs_stream_{streaming}.json").write_text(
                json.dumps(
                    {"query": query, "actual": actual}, ensure_ascii=False, indent=2
                )
                + "\n"
            )
            # The frontend terminates OpenAI streams with finish_reason and EOF.
            self.assertTrue(choices)
            self.assertTrue(choices[-1].get("finish_reason"))
            self.assertTrue(content.strip())
            self.assertGreater(
                len(records), 1, "must cover prefill and speculative decode"
            )
            self.assertEqual(
                b"".join(bytes(record["bytes"]) for record in records).decode("utf-8"),
                content,
                "every emitted content token must have exactly one logprob",
            )
            self.assertTrue(
                any(record["logprob"] < -1e-6 for record in records),
                "logprobs must describe the raw model distribution rather than greedy point masses",
            )
            for record in records:
                self.assertTrue(math.isfinite(record["logprob"]))
                self.assertLessEqual(record["logprob"], 1e-5)
                top = record["top_logprobs"]
                self.assertEqual(len(top), 3)
                scores = [item["logprob"] for item in top]
                self.assertTrue(
                    all(math.isfinite(value) and value <= 1e-5 for value in scores)
                )
                self.assertEqual(scores, sorted(scores, reverse=True))
                self.assertLessEqual(sum(math.exp(value) for value in scores), 1.00001)
                # Equal-score tokens may use different top-k tie ordering.
                self.assertAlmostEqual(record["logprob"], scores[0], places=4)
                self.assertLessEqual(record["logprob"], scores[0] + 1e-5)
                for candidate in top:
                    if candidate["bytes"] == record["bytes"]:
                        self.assertAlmostEqual(
                            candidate["logprob"], record["logprob"], places=4
                        )
            response.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
