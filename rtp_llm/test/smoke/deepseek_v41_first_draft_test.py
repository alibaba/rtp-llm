"""Matched proposal treatment; does not alter or replace the ordinary golden tests."""

import hashlib
import json
import os
import re
import unittest
from pathlib import Path

from rtp_llm.test.smoke import deepseek_v41_dspark_contract_test as dspark
from rtp_llm.test.smoke import deepseek_v41_pd_contract_test as pd
from rtp_llm.test.smoke.scripts.compare_v41_cache_rounds import DIGEST, hash_ids


class FirstDraft(dspark.DeepSeekV41DSparkContractTest, pd.DeepSeekV41PDContractTest):
    prefill_cp_size = 2
    decode_dp_size = 2

    def test_graph_engram_and_prefix_reuse(self):
        role = os.environ["RTP_V41_DRAFT_ROLE"]
        nonce = os.environ["RTP_ARM_PAIR_NONCE"]
        self.assertEqual(
            os.environ["TEST_TARGET"],
            "//rtp_llm/test/smoke:deepseek_v41_first_draft_sm100_arm",
        )
        self.assertEqual(os.environ["RTP_V41_EXPECTED_LOADER"], role)
        self.assertRegex(nonce, r"^[a-f0-9]{32}$")
        root = self.output_dir
        golden = json.loads(
            (
                Path(__file__).parent
                / "data/model/deepseek_v41/full_feature_cp2_dp2.json"
            ).read_text()
        )
        markers = {
            "oldfst": "finally choose load method: fastsafetensors",
            "newfst": "NewLoader load method: requested=fastsafetensors, selected=fastsafetensors",
        }
        for mgr in (self.server, self.graph_server):
            log = Path(mgr.log_file_path).read_text(errors="replace")
            self.assertTrue(markers[role] in log, "required FST marker missing")
            self.assertFalse(
                any(x in log for k, x in markers.items() if k != role),
                "ambiguous loader",
            )
            self.assertFalse(
                bool(
                    re.search(
                        r"finally choose load method: scratch|selected=scratch", log
                    )
                ),
                "scratch fallback",
            )
        marker = root / "v41_cache_trace_stage"
        for label, hit in (("long_cache_fill", False), ("long_cache_hit", True)):
            if hit:
                with marker.open("x") as f:
                    f.write(nonce + " hit\n")
            try:
                query = golden[label]["query"]
                self.assertEqual(query["generate_config"]["top_k"], 1)
                result = self.session.post(
                    f"http://127.0.0.1:{self.server.port}/", json=query, timeout=600
                )
                result.raise_for_status()
                data = result.json()
            finally:
                if hit:
                    marker.unlink(missing_ok=True)
            aux = data["aux_info"]
            ids = data["input_ids"][0]
            out = data["output_ids"][0]
            self.assertTrue(
                data["finished"] and aux["pd_sep"] and len(ids) == 783 and out
            )
            self.assertEqual(
                hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest(),
                DIGEST,
            )
            self.assertEqual(aux[self.reuse_field], 512 if hit else 0)
        p = [json.loads((root / f"v41draft_p{i}.json").read_text()) for i in range(2)]
        self.assertEqual(
            sorted(p[0]["positions"] + p[1]["positions"]), list(range(512, 783))
        )
        dfile = root / "v41draft_d0.json"
        d = json.loads(dfile.read_text())
        self.assertIn("native_head", d)
        scores = json.loads((root / "v41draft_scores.json").read_text())
        self.assertEqual(scores["anchor"], d["query_ids"][0])
        self.assertEqual(d["natural_ids"][0], out[0])
        rows = [
            json.loads(s)
            for s in (root / f"v41_cache_rounds_{nonce}_hit.jsonl")
            .read_text()
            .splitlines()
        ]
        first = [r for r in rows if r["offset"] == 1]
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["history_sha256_le_i32"], hash_ids(ids + [out[0]]))
        self.assertEqual(first[0]["draft"][0], scores["mapped"])
        d.update(
            prompt_digest=DIGEST,
            natural_history_sha=first[0]["history_sha256_le_i32"],
            used_history_sha=hash_ids(ids + [int(scores["anchor"])]),
            actual_hit=512,
            ordinary_golden_observed=out == golden["long_cache_hit"]["output_ids"][0],
            verify={
                "draft": first[0]["draft"],
                "target": first[0]["target"],
                "accept": first[0]["accept_len"],
            },
            ep_weights={
                str(i): json.loads((root / f"v41draft_ep{i}.json").read_text())
                for i in range(2)
            },
        )
        dfile.write_text(json.dumps(d, separators=(",", ":")))


if __name__ == "__main__":
    unittest.main()
