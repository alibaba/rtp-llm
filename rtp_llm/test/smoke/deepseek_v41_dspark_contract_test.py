"""V4.1 target/DSpark verify graph, prefix reuse and accepted-token history."""

import logging
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
        with Path(self.graph_server.log_file_path).open("rb") as reader:
            reader.seek(self.log_offset)
            runtime = reader.read().decode(errors="replace")
        self.assertRegex(
            runtime,
            r"\[PyWrappedModel\] using CUDA graph forward, is_target_verify=1, is_prefill=1, graph_bs=1",
            "Target verification did not replay its CUDA graph",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
