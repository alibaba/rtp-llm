import logging
import os
import unittest
from pathlib import Path

from rtp_llm.test.smoke import deepseek_v41_full_contract_test as full_contract
from rtp_llm.test.smoke.local_checkpoint import resolve_local_checkpoint


class DeepSeekV41MaxPanderContractTest(full_contract.DeepSeekV41FullContractTest):
    def setUp(self):
        os.environ["CHECKPOINT_PATH"] = resolve_local_checkpoint(
            "DeepSeek-V4.1-Flash", os.environ.get("CHECKPOINT_PATH")
        )
        super().setUp()
        prefill_log = Path(self.server.log_file_path).read_text(errors="replace")
        decode_log = Path(self.graph_server.log_file_path).read_text(errors="replace")
        self.assertIn(
            "MaxPander startup placement: logical=384 physical=480 EP=2 seed=0",
            prefill_log,
        )
        self.assertNotIn("MaxPander startup placement:", decode_log)

    def role_args(self, role):
        if role == "prefill":
            return (
                "--maxpander_enable 1 --maxpander_seed 0 --maxpander_wf_passes 8"
                " --redundant_expert 96 --moe_strategy mega_moe"
            )
        if role == "decode":
            return "--maxpander_enable 0 --redundant_expert 0 --moe_strategy mega_moe"
        raise ValueError(f"Unexpected role: {role}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
