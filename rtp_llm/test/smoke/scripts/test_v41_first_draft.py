"""CPU tests for matched feature/window relay, restoration and refusal to compare."""

import copy
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from compare_v41_first_draft import compare

from rtp_llm.models_py.modules import v41_first_draft as fd


class FirstDraftProtocol(unittest.TestCase):
    def test_commit_and_all_three_windows(self):
        nonce = "12" * 16
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "v41_cache_trace_stage").write_text(nonce + " hit\n")
            run = root / "ws/rtp_llm/test/smoke/arm_causal_inputs"
            run.mkdir(parents=True)
            cp = NS(prefix_length=512, cp_size=2, cp_rank=0)
            model = NS(_dspark_aux_feature_dim=5120)
            req = torch.tensor([0, -1, 0])
            pos = torch.tensor([512, -1, 782])
            src = torch.randn(3, 5120).bfloat16()
            base = dict(
                TEST_UNDECLARED_OUTPUTS_DIR=td,
                TEST_SRCDIR=td,
                TEST_WORKSPACE="ws",
                TEST_TARGET=fd.TARGET,
                RTP_ARM_PAIR_NONCE=nonce,
            )
            ddata = dict(
                query_ids=torch.tensor([1606, 9, 9], dtype=torch.int32),
                query_positions=torch.arange(783, 786),
                committed_positions=torch.arange(655, 783),
            )
            ddata.update(
                {
                    f"layer{i}": torch.arange(128 * 528, dtype=torch.int32)
                    .reshape(128, 528)
                    .byte()
                    + i
                    for i in range(3)
                }
            )
            for role in ("oldfst", "newfst"):
                if role == "newfst":
                    for f in (root / "arm_causal_private").glob("*.safetensors"):
                        shutil.copy(f, run)
                    (root / "v41draft_p0.json").rename(root / "old-p.json")
                with patch.dict(
                    os.environ,
                    dict(base, RTP_V41_DRAFT_ROLE=role, RTP_V41_EXPECTED_LOADER=role),
                ):
                    h = fd.fixed_features(
                        model,
                        src if role == "oldfst" else -src,
                        pos,
                        req,
                        torch.tensor([783]),
                        cp,
                    )
                    self.assertTrue(
                        torch.equal(h[::2], src[::2]) and not h[1].any().item()
                    )
                    d = {k: v.clone() for k, v in ddata.items()}
                    if role == "newfst":
                        d["layer2"].zero_()
                    out = fd.transfer(fd.scope(), "v41draft_d0", 0, "d_propose", d)
                    self.assertEqual(fd.sha(out["layer2"]), fd.sha(ddata["layer2"]))
                    if role == "newfst":
                        with self.assertRaisesRegex(ValueError, "provenance"):
                            fd.transfer(
                                (root, role, "ab" * 16),
                                "v41draft_d0",
                                0,
                                "d_propose",
                                d,
                            )
            bad = dict(ddata)
            bad["layer1"] = torch.empty(127, 528, dtype=torch.uint8)
            with self.assertRaisesRegex(ValueError, "cache"):
                fd.validate(bad, "d_propose")

    def test_rebinding_and_restoring_three_disjoint_pools(self):
        ids = torch.tensor([28, 999, 999], dtype=torch.int32)
        refs = {}
        for i in range(3):
            pool = torch.arange(4 * 64 * 528).reshape(4, 64, 528).byte()
            slots = torch.arange(61, 192)
            inds = fd.selected(pool, slots[:128])
            natural = pool[inds]
            pool[inds] = torch.zeros_like(natural)
            refs[i] = pool, slots, inds, natural
        backup = ids[0].clone()
        ids[0] = 1606
        fd.restore(dict(raw=ids, backup=backup, refs=refs))
        self.assertEqual(ids.tolist(), [28, 999, 999])
        for pool, _, inds, orig in refs.values():
            self.assertTrue(torch.equal(pool[inds], orig))

    def test_compare_refuses_different_inputs_but_reports_weights(self):
        sha = "a" * 64
        p = [dict(positions=[512 + i], natural=sha, used=sha) for i in range(2)]
        d = dict(
            nonce="f" * 32,
            used_history_sha=sha,
            natural_history_sha=sha,
            query_ids=[7, 9, 9],
            prompt_digest=sha,
            layers={
                i: {k: sha for k in ("used", "natural", "attn_input", "query_kv")}
                for i in "012"
            },
            graph_head=sha,
            native_head={"kernel": sha},
            verify={"draft": [8, 9, 9], "accept": 1},
            ep_weights={
                i: dict(paths={"x": "native"}, slots={"a": {"sha": sha}}, missing={})
                for i in "01"
            },
        )
        s = dict(
            temperature=1e-6,
            top=[],
            mapped=8,
            **{
                k: sha
                for k in (
                    "base",
                    "bias",
                    "pre_sampler",
                    "probabilities",
                    "w1_used",
                    "w2",
                    "d2t",
                )
            },
        )
        a = p, d, s
        b = copy.deepcopy(a)
        self.assertTrue(compare(a, b)["weight_attribution_qualified"])
        b[1]["ep_weights"]["1"]["slots"]["a"]["sha"] = "b" * 64
        self.assertFalse(compare(a, b)["weight_attribution_qualified"])
        b[1]["layers"]["2"]["used"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "draft cache"):
            compare(a, b)


if __name__ == "__main__":
    unittest.main()
