"""V4.1 full contract plus completed target migration and a serving probe."""

import json
import logging
import os
import unittest
from pathlib import Path

from rtp_llm.test.smoke import deepseek_v41_full_contract_test as full_contract
from rtp_llm.test.smoke.eplb_smoke_checks import check_eplb_applied


class DeepSeekV41EplbContractTest(full_contract.DeepSeekV41FullContractTest):
    # Recorded with the same MegaSE configuration before gate mapping fusion.
    golden_filename = "full_feature_cp2_dp2_mega_se.json"
    # Cover the native decode MegaMHC -> MegaGate -> MegaMoE-SE route.
    decode_env = {
        "DSV4_MOE_STRATEGY": "mega_se",
        "DSV4_USE_MEGA_MOE_SE": "1",
        "DSV41_MEGA_SHARED_EXPERT": "1",
        "DSV4_MEGA_GATE": "1",
    }

    def test_graph_engram_and_prefix_reuse(self):
        # Keep the original verdict while checking migration before unittest
        # runs the parent's registered server cleanups.
        with self.subTest(stage="original_contract"):
            super().test_graph_engram_and_prefix_reuse()

        if os.environ.get("SMOKE_ASSERT_EPLB_APPLIED", "1") != "1":
            return

        evidence = {"post_rebalance_request_passed": False}
        try:
            for role, manager in (
                ("prefill", self.server),
                ("decode", self.graph_server),
            ):
                evidence[role] = check_eplb_applied(
                    Path(manager.log_file_path).parent, self.prefill_cp_size
                )
            reference = json.loads(
                (self.output_dir / "short_uncached.json").read_text()
            )["actual"]
            actual = self.request("post_eplb_probe", self.short_prompt, False)
            self.assert_same(reference, actual)
            evidence["post_rebalance_request_passed"] = True
        finally:
            (self.output_dir / "eplb_migration_evidence.json").write_text(
                json.dumps(evidence, indent=2) + "\n"
            )


if __name__ == "__main__":
    os.environ.setdefault(
        "CHECKPOINT_PATH", "/mnt/hf3fs/3fs/models/DeepSeek-V4.1-Flash"
    )
    logging.basicConfig(level=logging.INFO)
    unittest.main()
