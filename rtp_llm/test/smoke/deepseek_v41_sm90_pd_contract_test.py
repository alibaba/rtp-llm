"""Hopper CP4-prefill / DP4-decode bounded-CED contract with UVA and Graph.

Reuse SM100's managed P/D routing, cache transfer and bounded replay assertions.
This does not assert bounded replay equals ordinary full-prefill semantics.
"""

import json
import logging
import os
import re
import unittest
from pathlib import Path

from rtp_llm.test.smoke import deepseek_v41_pd_contract_test as pd_contract


class DeepSeekV41Sm90PDContractTest(pd_contract.DeepSeekV41PDContractTest):
    prefill_cp_size = 4
    decode_dp_size = 4
    decode_cuda_graph = True
    require_graph_capture = True
    # CP4 canonical cache keys cover 256 * 4 tokens. Exercise two keys;
    # retain every original cache-hit and output/logit assertion.
    prefix_reuse_min_tokens = 2048
    moe_strategy = "sm90_humming"
    extra_server_env = {
        "MOE_STRATEGY": "auto",
        "DSV41_SM90_NATIVE_A8": "1",
        "DSV41_SM90_A8_PROMOTION_GROUP": "32",
        "DSV41_SM90_DENSE_G32_SELECTED": "0",
        "DSV41_SM90_INDEXED_MOE": "0",
        "DSV41_SM90_MULTIQUERY_INDEXER": "1",
        "DSV41_ENGRAM_HOST_LOOKUP": "uva",
        "DSV41_ENGRAM_PREFETCH": "1",
        "DSV41_SM90_DECODE_GRAPH": "1",
        "DSV41_SM90_DECODE_GRAPH_MAX_ROWS": "4",
        "DSV4_MOE_CHUNK_TOKENS": "4096",
    }

    def start_servers(self):
        import torch

        if not torch.version.cuda or int(torch.version.cuda.split(".")[0]) != 13:
            raise RuntimeError("SM90 P/D contract requires CUDA13")
        if torch.cuda.device_count() != 8 or any(
            torch.cuda.get_device_capability(rank) != (9, 0) for rank in range(8)
        ):
            raise RuntimeError("SM90 P4/D4 contract requires eight locked SM90 GPUs")
        super().start_servers()
        for manager in (self.server, self.graph_server):
            startup = Path(manager.log_file_path).read_text(errors="replace")
            self.assertIn("native FP8 WGMMA", startup)
            self.assertNotIn("E8M0 decoded for BF16 MMA", startup)
            self.assertIn("SM90 Engram UVA prefetch enabled", startup)
            self.assertNotIn("Engram shared CPU lookup enabled", startup)

    def assert_decode_graph_replayed_and_record_coverage(self):
        super().assert_decode_graph_replayed_and_record_coverage()
        prefill = Path(self.server.log_file_path).read_text(errors="replace")
        compactions = re.findall(
            r"V4.1 CED compacted: bounded=True cp=4 rows=(\d+)->(\d+)",
            prefill,
        )
        (self.output_dir / "v41_sm90_pd_coverage.json").write_text(
            json.dumps(
                {
                    "requests": self.observed,
                    "prefill_cp": 4,
                    "decode_tp": self.prefill_cp_size // self.decode_dp_size,
                    "decode_dp": self.decode_dp_size,
                    "decode_attention_cp": 1,
                    "decode_moe_chunking": False,
                    "ep_per_role": 4,
                    "execution": "decode_cuda_graph",
                    "engram": "uva_prefetch",
                    "ced_bounded_compactions": compactions,
                    "native_a8": True,
                    "promotion_k": 32,
                    "dspark_vit_separation": "NOT_RUN",
                    "ordinary_path_equivalence": "NOT_CLAIMED",
                    "smoke_args": os.environ["SMOKE_ARGS"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
