"""V4.1 target/DSpark verify graph, prefix reuse and accepted-token history."""

import json
import logging
import math
import shlex
import unittest
from pathlib import Path

from rtp_llm.test.smoke import deepseek_v41_contract_test as contract


class DeepSeekV41DSparkContractTest(contract.DeepSeekV41ContractTest):
    graph_target_verify = 1
    # Main represents multi-token verification with prefix/input lengths.
    graph_is_prefill = 1
    # Main GenerateStream.specUpdate publishes tokens/acceptance statistics,
    # but deliberately leaves logits and cumulative sampling scores undefined.
    require_generation_scores = False

    def server_args(self):
        # Both target and draft weights belong to the release checkpoint.
        return (
            super().server_args()
            + " --sp_checkpoint_path "
            + shlex.quote(self.model_path)
        )

    def request(self, label, prompt, reuse):
        actual = super().request(label, prompt, reuse)
        aux = actual["aux_info"]
        rounds = aux["speculative_draft_rounds"]
        accepted = aux["speculative_accepted_tokens_per_pos"]
        self.assertGreater(rounds, 0, "No DSpark draft/verify round")
        self.assertTrue(accepted, "Missing per-position acceptance counters")
        self.assertTrue(all(0 <= count <= rounds for count in accepted))
        self.assertGreater(sum(accepted), 0, "No accepted speculative tokens")
        self.observed[-1].update(
            speculative_draft_rounds=rounds,
            speculative_accepted_tokens_per_pos=accepted,
        )
        return actual

    def test_graph_engram_and_prefix_reuse(self):
        super().test_graph_engram_and_prefix_reuse()
        self.assertTrue(
            any(item["speculative_draft_rounds"] > 1 for item in self.observed),
            "No request exercised repeated DSpark draft/verify rounds",
        )
        self.assert_sampling_logprobs()
        with Path(self.graph_server.log_file_path).open("rb") as reader:
            reader.seek(self.log_offset)
            runtime = reader.read().decode(errors="replace")
        self.assertRegex(
            runtime,
            r"\[PyWrappedModel\] using CUDA graph forward, is_target_verify=1, is_prefill=1, graph_bs=1",
            "Target verification did not replay its CUDA graph",
        )

    def assert_sampling_logprobs(self):
        query = {
            "messages": [
                {
                    "role": "user",
                    "content": "Invent a peculiar name for a blue dragon and describe its unusual hobbies.",
                }
            ],
            "max_tokens": 32,
            "enable_thinking": False,
            "logprobs": True,
            "top_logprobs": 1,
            "extra_configs": {
                "random_seed": 42,
                "reuse_cache": False,
                "can_use_pd_separation": False,
            },
        }
        # A predictable list can become a point mass at every step after top-p.
        # Exercise both truncated and full distributions with an open-ended prompt.
        for label, stochastic, temperature, top_p in (
            ("greedy", False, 0.0, 1.0),
            ("top_p", True, 1.5, 0.9),
            ("full", True, 0.8, 1.0),
        ):
            query.update(temperature=temperature, top_p=top_p)
            query["extra_configs"].update(
                top_k=0 if stochastic else 1, do_sample=stochastic
            )
            reference = None
            for repeat in range(2):
                response = self.session.post(
                    f"http://127.0.0.1:{self.server.port}/v1/chat/completions",
                    json=query,
                    timeout=600,
                )
                response.raise_for_status()
                actual = response.json()
                response.close()
                (self.output_dir / f"sampling_{label}_{repeat}.json").write_text(
                    json.dumps(
                        {"query": query, "actual": actual}, ensure_ascii=False, indent=2
                    )
                )
                choice = actual["choices"][0]
                content = choice["message"]["content"]
                records = choice["logprobs"]["content"]
                self.assertTrue(choice["finish_reason"])
                self.assertGreater(len(records), 1, "must exercise speculative decode")
                self.assertGreater(actual["aux_info"]["speculative_draft_rounds"], 0)
                self.assertEqual(
                    b"".join(bytes(record["bytes"]) for record in records).decode(
                        "utf-8"
                    ),
                    content,
                )
                for record in records:
                    score = record["logprob"]
                    self.assertTrue(math.isfinite(score))
                    self.assertLessEqual(score, 1e-5)
                    top = record["top_logprobs"]
                    self.assertEqual(len(top), 1)
                    self.assertTrue(math.isfinite(top[0]["logprob"]))
                    self.assertLessEqual(score, top[0]["logprob"] + 1e-5)
                    if not stochastic:
                        self.assertAlmostEqual(score, 0.0, places=5)
                    if top[0]["bytes"] == record["bytes"]:
                        self.assertAlmostEqual(score, top[0]["logprob"], places=5)
                if stochastic:
                    self.assertTrue(
                        any(record["logprob"] < -1e-5 for record in records)
                    )
                # Draft proposals use the global RNG, so a request seed alone
                # does not reproduce stochastic DSpARK output.
                if not stochastic and reference is not None:
                    self.assertEqual(content, reference["message"]["content"])
                    self.assertEqual(records, reference["logprobs"]["content"])
                reference = choice


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
