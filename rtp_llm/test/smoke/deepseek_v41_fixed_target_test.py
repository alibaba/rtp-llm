"""Controlled first target verification. Never substitute this for ordinary golden."""

import hashlib
import json
import os
import unittest
import uuid
from pathlib import Path

from rtp_llm.test.smoke import deepseek_v41_dspark_contract_test as dspark
from rtp_llm.test.smoke import deepseek_v41_pd_contract_test as pd
from rtp_llm.test.smoke.scripts.compare_v41_cache_rounds import DIGEST, hash_ids
from rtp_llm.test.smoke.scripts.compare_v41_fixed_target import validate


class FixedVerify(dspark.DeepSeekV41DSparkContractTest, pd.DeepSeekV41PDContractTest):
    prefill_cp_size = 2
    decode_dp_size = 2

    def test_graph_engram_and_prefix_reuse(self):
        role = os.environ["RTP_V41_EXPECTED_LOADER"]
        self.assertEqual(
            os.environ["TEST_TARGET"],
            "//rtp_llm/test/smoke:deepseek_v41_fixed_target_sm100_arm",
        )
        markers = {
            "oldfst": "finally choose load method: fastsafetensors",
            "newfst": "NewLoader load method: requested=fastsafetensors, selected=fastsafetensors",
        }
        for manager in (self.server, self.graph_server):
            log = Path(manager.log_file_path).read_text(errors="replace")
            self.assertTrue(
                markers[role] in log
                and not any(x in log for r, x in markers.items() if r != role)
            )
            self.assertTrue(
                "selected=scratch" not in log
                and "finally choose load method: scratch" not in log
            )
        gold = json.loads(
            (
                Path(__file__).parent
                / "data/model/deepseek_v41/full_feature_cp2_dp2.json"
            ).read_text()
        )
        root = self.output_dir
        run = uuid.uuid4().hex
        stage = root / "v41_cache_trace_stage"
        for phase in ("fill", "hit"):
            query = gold["long_cache_" + phase]["query"]
            self.assertEqual(query["generate_config"]["top_k"], 1)
            self.assertIs(query["generate_config"]["reuse_cache"], True)
            if phase == "hit":
                with stage.open("x") as out:
                    out.write(run + " hit\n")
            try:
                resp = self.session.post(
                    f"http://127.0.0.1:{self.server.port}/", json=query, timeout=600
                )
                resp.raise_for_status()
                actual = resp.json()
            finally:
                if phase == "hit":
                    stage.unlink()
            ids = actual["input_ids"][0]
            self.assertEqual(len(ids), 783)
            self.assertEqual(
                hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest(), DIGEST
            )
            self.assertTrue(actual["finished"] and actual["aux_info"]["pd_sep"])
            self.assertEqual(
                actual["aux_info"][self.reuse_field], 512 if phase == "hit" else 0
            )
        rows = [
            json.loads(x)
            for x in (root / f"v41_cache_rounds_{run}_hit.jsonl")
            .read_text()
            .splitlines()
        ]
        (first,) = [r for r in rows if r["offset"] == 1]
        self.assertEqual(first["run_id"], run)
        self.assertEqual(first["history_sha256_le_i32"], hash_ids(ids + [1606]))
        self.assertEqual(first["result"][0], actual["output_ids"][0][1])
        doc = dict(
            role=role,
            nonce=os.environ["RTP_ARM_PAIR_NONCE"],
            history=first["history_sha256_le_i32"],
            prompt=DIGEST,
            prefix=512,
            anchor=actual["output_ids"][0][0],
            candidates=first["draft"],
            scores=first["fixed_scores"],
            target=first["target"][0],
            first_committed=first["result"][0],
        )
        validate(doc, role)
        (root / "v41_fixed_target.json").write_text(json.dumps(doc, allow_nan=False))


if __name__ == "__main__":
    unittest.main()
