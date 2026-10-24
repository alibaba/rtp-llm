import logging
import os
import unittest
from pathlib import Path

from rtp_llm.test.smoke import deepseek_v41_full_contract_test as full_contract
from rtp_llm.test.smoke.local_checkpoint import resolve_local_checkpoint


class DeepSeekV41MaxPanderOffContractTest(full_contract.DeepSeekV41FullContractTest):
    def setUp(self):
        os.environ["CHECKPOINT_PATH"] = resolve_local_checkpoint(
            "DeepSeek-V4.1-Flash", os.environ.get("CHECKPOINT_PATH")
        )
        super().setUp()
        for manager in (self.server, self.graph_server):
            startup = Path(manager.log_file_path).read_text(errors="replace")
            self.assertNotIn("MaxPander startup placement:", startup)

    def role_args(self, role):
        if role not in ("prefill", "decode"):
            raise ValueError(f"Unexpected role: {role}")
        return "--maxpander_enable 0 --redundant_expert 0 --moe_strategy mega_moe"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
