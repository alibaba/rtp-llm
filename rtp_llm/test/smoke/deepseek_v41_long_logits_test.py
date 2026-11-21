"""Paired GB200 diagnostic, separate from the unchanged full-feature golden."""

import hashlib
import json
import logging
import math
import os
import re
import time
import unittest
from pathlib import Path

from rtp_llm.test.smoke import deepseek_v41_dspark_contract_test as dspark
from rtp_llm.test.smoke import deepseek_v41_pd_contract_test as pd


class DeepSeekV41LongLogitsTest(
    dspark.DeepSeekV41DSparkContractTest, pd.DeepSeekV41PDContractTest
):
    prefill_cp_size = 2
    decode_dp_size = 2

    def test_graph_engram_and_prefix_reuse(self):
        # Keep the ordinary golden and its seven requests in their own target.
        # This fixture reuses that exact prompt, but never sends it to remote D.
        fixture = (
            Path(__file__).parent / "data/model/deepseek_v41/full_feature_cp2_dp2.json"
        )
        golden = json.loads(fixture.read_text())["long_uncached"]
        self.assertEqual(golden["output_ids"][0][0], 1606)
        query = golden["query"]
        query["generate_config"].update(
            max_new_tokens=1,
            reuse_cache=False,
            can_use_pd_separation=False,
        )
        self.assertEqual(query["generate_config"]["top_k"], 1)
        log = Path(self.server.log_file_path)
        startup = log.read_text(errors="replace")
        loader = os.environ["RTP_V41_EXPECTED_LOADER"]
        markers = {
            "oldfst": "finally choose load method: fastsafetensors",
            "newfst": "NewLoader load method: requested=fastsafetensors, selected=fastsafetensors",
        }
        # unittest membership assertions print the entire container on
        # failure. Never include the service startup log in an assertion.
        self.assertTrue(
            markers[loader] in startup, "missing expected FST loader marker"
        )
        other = "newfst" if loader == "oldfst" else "oldfst"
        self.assertFalse(markers[other] in startup, "ambiguous FST loader path")
        self.assertFalse(
            "finally choose load method: scratch" in startup
            or "selected=scratch" in startup,
            "unexpected scratch loader path",
        )
        offset = log.stat().st_size
        result = self.session.post(
            f"http://127.0.0.1:{self.server.port}/", json=query, timeout=600
        )
        result.raise_for_status()
        actual = result.json()
        ids = actual["input_ids"][0]
        self.assertEqual(len(ids), 783)
        digest = hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()
        self.assertEqual(
            digest, "d570954fff2202dfde49e873ece0fc1adcb2df1c6a9095cb98fe6601ac4645c4"
        )
        self.assertIs(actual["aux_info"]["pd_sep"], False)
        self.assertEqual(actual["aux_info"]["reuse_len"], 0)
        self.assertEqual(len(actual["output_ids"]), 1)
        self.assertEqual(len(actual["output_ids"][0]), 1)
        # Native logs may flush after HTTP completion. Bound the wait and fail
        # on a missing row instead of silently publishing null scores.
        deadline = time.monotonic() + 10
        while True:
            with log.open("rb") as stream:
                stream.seek(offset)
                runtime = stream.read().decode(errors="replace")
            rows = re.findall(
                r"V41_FIXTURE_PREFILL_V1 hash=83ef06b1e7cedebb id1606=(\S+) id28=(\S+) argmax=(\d+)",
                runtime,
            )
            if rows or time.monotonic() >= deadline:
                break
            time.sleep(0.1)
        self.assertEqual(
            len(rows), 1, "expected exactly one native prefill sampler row"
        )
        a, b, argmax = rows[0]
        a, b, argmax = float(a), float(b), int(argmax)
        self.assertTrue(math.isfinite(a) and math.isfinite(b))
        sampled = actual["output_ids"][0][0]
        self.assertEqual(
            argmax, sampled, "top-k=1 did not return the pre-sampler argmax"
        )
        # Do not mark a diagnostic successful with missing/misaligned CP
        # rows, truncated per-layer traces, or an unobserved runtime head.
        trace = {}
        root = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"])
        from rtp_llm.models_py.modules.v41_fixture_boundaries import (
            validate_fixture_boundaries,
        )

        validate_fixture_boundaries(root)
        for rank in range(2):
            rank_trace = root / f"v41_fixed_trace_rank{rank}.jsonl"
            entries = [json.loads(line) for line in rank_trace.read_text().splitlines()]
            self.assertTrue(entries, "missing V4.1 fixture trace")
            stages = [row["stage"] for row in entries]
            count = entries[0]["layers"]
            self.assertGreater(count, 0)
            expected = (
                ["embedding"]
                + [f"layer_{i}" for i in range(count)]
                + ["hc_reduce", "norm", "lm_head"]
            )
            self.assertEqual(stages, expected)
            self.assertTrue(all(row["rank"] == rank for row in entries))
            self.assertEqual(entries[-1]["world"], 2)
            trace[rank] = {row["stage"]: row for row in entries}
        self.assertEqual(set(trace[0]), set(trace[1]))
        for stage in trace[0]:
            if stage != "lm_head":
                owners = sum(trace[r][stage]["last_sha"] is not None for r in range(2))
                self.assertEqual(
                    owners, 1, f"{stage}: expected one owner of the last real token"
                )
        candidates = [
            key for r in range(2) for key in trace[r]["lm_head"]["candidate_sha"]
        ]
        self.assertCountEqual(candidates, ["1606", "28"])
        (self.output_dir / "v41_long_layer_trace.json").write_text(
            json.dumps(
                {
                    "schema": 2,
                    "loader": loader,
                    "input_ids_sha256": digest,
                    "ranks": trace,
                },
                indent=2,
            )
            + "\n"
        )
        summary = {
            "schema": 1,
            "loader": loader,
            "reader": "fastsafetensors",
            "input_len": 783,
            "input_ids_sha256": digest,
            "pd_sep": False,
            "reuse_len": 0,
            "logit_1606": a,
            "logit_28": b,
            "margin_1606_minus_28": a - b,
            "argmax": argmax,
            "http_first_token": sampled,
        }
        (self.output_dir / "v41_long_target_logits.json").write_text(
            json.dumps(summary, allow_nan=False, indent=2) + "\n"
        )
        if loader == "oldfst":
            self.assertEqual(
                sampled, 1606, "OldFST must reproduce the golden first token"
            )


class LayerBoundaryProtocolTest(unittest.TestCase):
    """CPU protocol: exercise real PyTorch hooks, CP mapping and in-place inputs."""

    def test_both_ranks_and_full_or_compacted_wkv(self):
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch

        import torch

        from rtp_llm.models_py.modules.v41_fixture_boundaries import (
            observe_fixture_layer,
            validate_fixture_boundaries,
        )

        class Engram(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.wkv = torch.nn.Linear(4, 6, bias=False).to(torch.bfloat16)

            def forward(self, x, active, compact):
                rows = x.flatten(1)
                self.wkv(rows[active] if compact else rows)
                return x.add_(1)

        for compact in (False, True):
            with self.subTest(compact=compact), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                layer = SimpleNamespace(
                    engram=Engram(),
                    attn_norm=torch.nn.Identity(),
                    attn=torch.nn.Identity(),
                    ffn=torch.nn.Identity(),
                )
                with patch.dict(os.environ, TEST_UNDECLARED_OUTPUTS_DIR=temp):
                    for rank in range(2):
                        pos = list(range(rank, 783, 2))
                        valid = torch.arange(392) < len(pos)
                        cp = SimpleNamespace(
                            cp_rank=rank,
                            local_is_real=valid,
                            global_positions=torch.tensor(
                                pos + [782] * (392 - len(pos))
                            ),
                        )
                        h = torch.zeros(392, 2, 2, dtype=torch.bfloat16)
                        layer.engram_token_mask = valid
                        layer.engram_hashes = torch.zeros(392, 1, dtype=torch.int64)
                        hc = SimpleNamespace(pre_mix_out=torch.zeros(392, 2))
                        layer.attn_hc = SimpleNamespace(_previous_ref=lambda: hc)
                        layer.ffn_hc = hc

                        def call(x):
                            x = layer.engram(x, valid, compact)
                            layer.ffn(layer.attn(layer.attn_norm(x[:, 0, :])))
                            return x

                        self.assertIs(observe_fixture_layer(call, layer, cp, h), h)
                        self.assertFalse(layer.engram._forward_hooks)
                layout = (2, 2, 1, 4)
                validate_fixture_boundaries(root, layout=layout)
                path = root / "v41_layer1_boundaries_rank1.json"
                row = json.loads(path.read_text())
                self.assertNotEqual(
                    row["stages"]["engram_in"], row["stages"]["engram_out"]
                )
                self.assertEqual(row["stages"]["engram_in"], row["stages"]["incoming"])
                # Still two complete rank-local records, but one missing real
                # position and a duplicate across ranks (the old test passed).
                row["positions"][-1] = 782
                path.write_text(json.dumps(row))
                with self.assertRaisesRegex(ValueError, "exactly once"):
                    validate_fixture_boundaries(root, layout=layout)
                row["positions"][-1] = 781
                row["metadata"]["previous_mix"] = ["torch.bfloat16", [2]]
                path.write_text(json.dumps(row))
                with self.assertRaisesRegex(ValueError, "metadata"):
                    validate_fixture_boundaries(root, layout=layout)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
