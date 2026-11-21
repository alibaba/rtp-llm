"""CPU protocol regression. Does not exercise SM100 or collective kernels."""

import copy
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from compare_v4_ops import GLOBAL, ORDER, SHARED, W, compare

from rtp_llm.models_py.modules.dsv4.prefill.v4_l1_ops import (
    fingerprint,
    instrument,
    relay,
)

NONCE = "1234" * 8
SHA = "a" * 64
POSITIONS = (list(range(14)) + list(range(42, 55)), list(range(14, 42)))


def reports():
    docs = []
    for role in ("oldfst", "newfst"):
        for r in range(2):
            ops = {}
            for k in ORDER + SHARED:
                pos = list(range(55)) if k in GLOBAL else POSITIONS[r]
                ops[k] = {
                    "pos": pos,
                    "rows": [SHA] * len(pos),
                    "shape": [1],
                    "dtype": "torch.bfloat16",
                }
            docs.append(
                dict(
                    role=role,
                    rank=r,
                    nonce=NONCE,
                    family="v4",
                    relay_sha=SHA,
                    natural_rows_equal_relay=True,
                    natural_valid_rows=copy.deepcopy(ops["input"]),
                    natural_full_input_sha=SHA,
                    consumed_full_input_sha=SHA,
                    physical_input_rows=28,
                    valid_input_rows=len(POSITIONS[r]),
                    moe_executor="routed_plus_shared",
                    ops=ops,
                    weights={
                        k: {
                            "sha": SHA,
                            "shape": [1],
                            "stride": [1],
                            "dtype": "torch.bfloat16",
                        }
                        for k in W
                    },
                    ab={
                        k: dict(
                            live=SHA,
                            independent=SHA,
                            shared=SHA,
                            activation_equal=True,
                            scales_equal=True,
                        )
                        for k in ("wq_a", "wkv")
                    },
                    repeat_moe=copy.deepcopy(
                        {
                            k: v
                            for k, v in ops.items()
                            if k.startswith("ffn.") and not k.startswith("ffn.hc")
                        }
                    ),
                )
            )
    return docs


class PairedInputs(unittest.TestCase):
    def test_private_roundtrip_per_rank(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "wrk/rtp_llm/test/smoke/arm_causal_inputs"
            data.mkdir(parents=True)
            with patch.dict(os.environ, {"TEST_SRCDIR": temp, "TEST_WORKSPACE": "wrk"}):
                h = (
                    torch.arange(28 * 4 * 4096, dtype=torch.float32)
                    .reshape(28, 4, 4096)
                    .bfloat16()
                )
                for rank, pos in enumerate(POSITIONS):
                    t = SimpleNamespace(
                        root=root, rank=rank, pos=pos, rows=torch.arange(len(pos))
                    )
                    old, s0, same = relay(t, h, "oldfst", NONCE)
                    self.assertTrue(same)
                    self.assertTrue(torch.equal(old[: len(pos)], h[: len(pos)]))
                    self.assertFalse(old[len(pos) :].any().item())
                    f = root / "arm_causal_private" / ("v4_r%d.safetensors" % rank)
                    shutil.copy(f, data)
                    with self.assertRaisesRegex(ValueError, "duplicate relay"):
                        relay(t, h, "oldfst", NONCE)
                    new, s1, same = relay(t, h, "newfst", NONCE)
                    self.assertTrue(same and s0 == s1 and torch.equal(old, new))
                    drift = h.clone()
                    drift[0, 0, 0] += 1
                    self.assertFalse(relay(t, drift, "newfst", NONCE)[2])
                    with self.assertRaisesRegex(ValueError, "identity"):
                        relay(t, h, "newfst", "0" * 32)
                    swapped = SimpleNamespace(**{**vars(t), "pos": list(reversed(pos))})
                    with self.assertRaisesRegex(ValueError, "row layout"):
                        relay(swapped, h, "newfst", NONCE)

    def test_non_diag_path_is_noop(self):
        with patch.dict(os.environ, {}, clear=True):
            call = lambda x: x
            self.assertIs(instrument(None, None, call), call)
        with patch.dict(
            os.environ, {"RTP_V4_CAUSAL_ROLE": "newfst", "TEST_TARGET": "irrelevant"}
        ):
            with self.assertRaisesRegex(ValueError, "scope"):
                instrument(None, None, None)

    def test_logical_and_physical_fingerprint(self):
        x = torch.arange(12, dtype=torch.int32).reshape(4, 3).t()
        a, b = fingerprint(x), fingerprint(x.contiguous())
        self.assertEqual(a["sha"], b["sha"])
        self.assertNotEqual(a["stride"], b["stride"])

    def test_first_difference_and_repeat(self):
        d = reports()
        self.assertIsNone(compare(d)["first_different_observed_boundary"])
        stage = "ffn.shared.gate_up"
        d[3]["ops"][stage]["rows"][POSITIONS[1].index(40)] = "b" * 64
        r = compare(d)
        self.assertEqual(
            r["first_different_observed_boundary"], {"stage": stage, "positions": [40]}
        )
        self.assertEqual(r["first_different_boundary_at_position40"], stage)
        self.assertEqual(
            r["same_input_moe_repeat_differences"]["newfst1"], {stage: [40]}
        )
        d[2]["weights"]["_se_l1_sf"]["sha"] = "b" * 64
        self.assertEqual(
            compare(d)["runtime_weight_differences"], [{"rank": 0, "name": "_se_l1_sf"}]
        )
        d[3]["natural_valid_rows"]["rows"][POSITIONS[1].index(40)] = "c" * 64
        d[3].update(natural_rows_equal_relay=False, natural_full_input_sha="d" * 64)
        self.assertEqual(
            compare(d)["natural_input_differing_positions_vs_donor"]["newfst1"], [40]
        )

    def test_bad_or_partial_evidence_rejected(self):
        for mutate in (
            lambda d: d[3].update(nonce="0" * 32),
            lambda d: d[3].update(natural_rows_equal_relay=False),
            lambda d: d[2]["ab"].pop("wq_a"),
            lambda d: d[1]["ops"].pop("ffn.shared.out"),
            lambda d: d[0]["repeat_moe"].pop("ffn.pack.x"),
            lambda d: d[0]["ops"]["attn.swa_read"]["rows"].pop(),
            lambda d: d[0]["ops"]["attn.swa_read"]["pos"].__setitem__(40, 39),
        ):
            d = reports()
            mutate(d)
            with self.assertRaises(ValueError):
                compare(d)

    def test_changed_cache_is_evidence_not_a_missing_fixture(self):
        d = reports()
        self.assertFalse(compare(d)["invalid_same_cache_for_downstream"])
        d[3]["ops"]["attn.swa_read"]["rows"][40] = "b" * 64
        r = compare(d)
        self.assertTrue(r["valid"] and r["invalid_same_cache_for_downstream"])
        self.assertFalse(r["fresh_kv_only"])
        self.assertEqual(r["first_different_boundary_at_position40"], "attn.swa_read")
        chain = r["fresh_kv_boundary_checks"]["newfst1"]
        pair = chain["attn.swa_write->attn.swa_read"]
        self.assertEqual(pair["different_positions"], [40])
        self.assertNotEqual(pair["from_digest"], pair["to_digest"])
        self.assertEqual(
            chain["attn.swa_read->attn.mla.kv"]["different_positions"], [40]
        )
        self.assertEqual(r["mla_kv_old_to_new_by_rank"]["1"]["different_positions"], [])
        # Even if every role's chain is internally consistent, different
        # Old/New actual MLA inputs still rule out downstream attribution.
        for stage in GLOBAL:
            d[3]["ops"][stage]["rows"][40] = "b" * 64
        r = compare(d)
        self.assertTrue(r["fresh_kv_only"] and r["invalid_same_cache_for_downstream"])
        self.assertEqual(
            r["mla_kv_old_to_new_by_rank"]["1"]["different_positions"], [40]
        )


if __name__ == "__main__":
    unittest.main()
