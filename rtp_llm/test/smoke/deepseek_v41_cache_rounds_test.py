"""Run the fixed V4.1 CED/DSpARK cold/fill/hit sequence; preserve the golden."""

import hashlib
import json
import logging
import os
import unittest
import uuid
from pathlib import Path

from rtp_llm.test.smoke import deepseek_v41_dspark_contract_test as dspark
from rtp_llm.test.smoke import deepseek_v41_pd_contract_test as pd
from rtp_llm.test.smoke.scripts.compare_v41_cache_rounds import (
    DIGEST,
    hash_ids,
    validate,
)


class CacheRoundFixture(
    dspark.DeepSeekV41DSparkContractTest, pd.DeepSeekV41PDContractTest
):
    prefill_cp_size = 2
    decode_dp_size = 2

    def test_graph_engram_and_prefix_reuse(self):
        from rtp_llm.models_py.modules.v41_cache_model_trace import finalize

        self.assertTrue(
            os.getenv("RTP_V41_CACHE_TRACE") == "1"
            and os.getenv("TEST_TARGET")
            == "//rtp_llm/test/smoke:deepseek_v41_cache_rounds_sm100_arm",
            "incorrect fixture scope",
        )
        loader = os.environ["RTP_V41_EXPECTED_LOADER"]
        markers = {
            "oldfst": "finally choose load method: fastsafetensors",
            "newfst": "NewLoader load method: requested=fastsafetensors, selected=fastsafetensors",
        }
        for manager in (self.server, self.graph_server):
            startup = Path(manager.log_file_path).read_text(errors="replace")
            self.assertTrue(
                markers[loader] in startup, "required FST path was not selected"
            )
            self.assertFalse(
                any(v in startup for k, v in markers.items() if k != loader),
                "ambiguous loader",
            )
            self.assertFalse(
                any(
                    s in startup
                    for s in ("finally choose load method: scratch", "selected=scratch")
                ),
                "unexpected scratch selection",
            )
        golden = json.loads(
            (
                Path(__file__).parent
                / "data/model/deepseek_v41/full_feature_cp2_dp2.json"
            ).read_text()
        )
        run = uuid.uuid4().hex
        root = self.output_dir
        marker = root / "v41_cache_trace_stage"
        self.assertFalse(
            marker.exists(), "another fixture has armed this output directory"
        )
        stage_tmp = root / ("v41_cache_trace_stage." + run)
        phases = {}
        doc = {
            "schema": 1,
            "run_id": run,
            "loader": loader,
            "reader": "fastsafetensors",
            "input_ids_sha256": DIGEST,
            "input_len": 783,
            "eos_id": self.tokenizer.eos_token_id,
            "vocab_size": self.vocab_size,
            "phases": phases,
        }
        exact = True
        for phase, label, reuse in (
            ("cold", "long_uncached", False),
            ("fill", "long_cache_fill", True),
            ("hit", "long_cache_hit", True),
        ):
            expected = golden[label]
            query = expected["query"]
            cfg = query["generate_config"]
            self.assertTrue(
                cfg["max_new_tokens"] == 24
                and cfg["top_k"] == 1
                and cfg["reuse_cache"] is reuse
                and cfg["can_use_pd_separation"] is True,
                "invalid fixture configuration",
            )
            stage_tmp.write_text(f"{run} {phase}\n")
            stage_tmp.replace(marker)
            try:
                response = self.session.post(
                    f"http://127.0.0.1:{self.server.port}/", json=query, timeout=600
                )
                response.raise_for_status()
                actual = response.json()
            finally:
                marker.unlink(missing_ok=True)
            ids = actual["input_ids"][0]
            digest = hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()
            self.assertTrue(
                len(ids) == 783 and digest == DIGEST, "unexpected request identity"
            )
            aux = actual["aux_info"]
            self.assertTrue(
                actual.get("finished") is True and len(actual["output_ids"]) == 1,
                "invalid HTTP output shape",
            )
            self.assertTrue(
                aux["pd_sep"] is True and aux["input_len"] == 783, "invalid P/D route"
            )
            measured = aux[self.reuse_field]
            self.assertEqual(
                measured, 512 if phase == "hit" else 0, "wrong P-side cache coverage"
            )
            out = actual["output_ids"][0]
            raw = root / f"v41_cache_rounds_{run}_{phase}.jsonl"
            self.assertTrue(
                raw.is_file(),
                "no native DSpARK rejection rounds for fixture phase " + phase,
            )
            rounds = [json.loads(l) for l in raw.read_text().splitlines()]
            for row in rounds:
                n = row["accept_len"]
                off = row["offset"]
                self.assertTrue(
                    row["history_sha256_le_i32"] == hash_ids(ids + out[:off]),
                    "verify history differs from real HTTP prefix",
                )
                row["accepted_draft_ids"] = row["draft"][: n - 1]
            matches = out == expected["output_ids"][0]
            text_matches = actual["response"] == expected["response"]
            exact &= (
                matches
                and text_matches
                and measured == expected["prefill_reuse_len"]
                and aux["input_len"] == expected["input_len"]
            )
            phases[phase] = {
                "pd_sep": True,
                "prefill_reuse_len": measured,
                "first_token": out[0],
                "output_ids": out,
                "expected_ids": expected["output_ids"][0],
                "matches_golden": matches,
                "response_matches_golden": text_matches,
                "draft_rounds": aux["speculative_draft_rounds"],
                "rounds": rounds,
            }
        validate(doc)
        (root / "v41_cache_rounds.json").write_text(
            json.dumps(doc, indent=2, allow_nan=False) + "\n"
        )
        finalize(root, run)
        if os.environ.get("RTP_V41_CAUSAL_ROLE"):
            from rtp_llm.models_py.modules.v41_causal import CELLS

            self.assertEqual(loader, os.environ["RTP_V41_CAUSAL_ROLE"])
            replay = root / "v41_causal_replay_id"
            for mode in ("natural", *CELLS):
                if mode == "natural":
                    repeat, data = run, actual
                else:
                    repeat = uuid.uuid4().hex
                    replay.write_text(f"{repeat} {mode}")
                    try:
                        stage_tmp.write_text(f"{repeat} hit\n")
                        stage_tmp.replace(marker)
                        result = self.session.post(
                            f"http://127.0.0.1:{self.server.port}/",
                            json=golden["long_cache_hit"]["query"],
                            timeout=600,
                        )
                        result.raise_for_status()
                        data = result.json()
                    finally:
                        marker.unlink(missing_ok=True)
                        replay.unlink(missing_ok=True)
                self.assertTrue(data["finished"] and data["aux_info"]["pd_sep"])
                self.assertEqual(data["aux_info"][self.reuse_field], 512)
                self.assertEqual(data["input_ids"], actual["input_ids"])
                self.assertEqual(len(data["output_ids"]), 1)
                emitted = data["output_ids"][0]
                self.assertTrue(2 <= len(emitted) <= 24)
                native = [
                    json.loads(line)
                    for line in (root / f"v41_cache_rounds_{repeat}_hit.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(
                    data["aux_info"]["speculative_draft_rounds"], len(native)
                )
                self.assertGreater(len(native), 0)
                offset, accepts = 1, []
                for row in native:
                    self.assertEqual(
                        (row["run_id"], row["phase"], row["offset"]),
                        (repeat, "hit", offset),
                    )
                    self.assertEqual(
                        row["history_sha256_le_i32"], hash_ids(ids + emitted[:offset])
                    )
                    n = row["accept_len"]
                    self.assertTrue(type(n) is int and 1 <= n <= 4)
                    self.assertEqual(len(row["result"]), n)
                    count = min(n, len(emitted) - offset)
                    self.assertGreater(count, 0)
                    self.assertEqual(
                        row["result"][:count], emitted[offset : offset + count]
                    )
                    self.assertEqual(row["draft"][: n - 1], row["result"][: n - 1])
                    accepts.append(min(count, n - 1))
                    offset += count
                self.assertEqual(offset, len(emitted))
                outcome = dict(
                    input_digest=DIGEST,
                    reuse=512,
                    decode_reuse=data["aux_info"]["decode_total_reuse_len"],
                    output_count=len(emitted),
                    output_hash=hash_ids(emitted),
                    accepts=accepts,
                    matches_golden=(
                        emitted == golden["long_cache_hit"]["output_ids"][0]
                        and data["response"] == golden["long_cache_hit"]["response"]
                    ),
                )
                for rank in range(2):
                    output = root / f"v41_causal_{mode}_r{rank}.json"
                    self.assertTrue(
                        output.is_file(), "missing causal evidence: " + output.name
                    )
                    part = json.loads(output.read_text())
                    self.assertEqual(part["request_id"], repeat)
                    part["outcome"] = outcome
                    temp = output.with_suffix(".tmp")
                    temp.write_text(json.dumps(part, separators=(",", ":")))
                    temp.replace(output)
        self.assertTrue(
            exact,
            "fixture output differs from the unchanged golden; see v41_cache_rounds.json",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
