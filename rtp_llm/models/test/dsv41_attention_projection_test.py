"""Checkpoint-ordered query/index/output equations and loader name binding."""

import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.config.dsv41_config import V41LayerSources
from rtp_llm.models.test.dsv41_cache_codec_test import load_codec_source
from rtp_llm.models.test.dsv41_mxfp8_test import (
    grouped_reference,
    linear,
    quantization_reference,
)
from rtp_llm.models.test.dsv41_projection_test import norm, projection, rope, rotate

with patch.dict(
    sys.modules,
    {
        "rtp_llm.models_py.modules.dsv4.v41_mxfp8": linear,
        "rtp_llm.models_py.modules.dsv4.v41_projection": projection,
    },
):
    queries = load_codec_source(
        "_dsv41_attention_projection_test", "v41_attention_projection.py"
    )


def weights_fixture(layer=20):
    torch.manual_seed(422)
    weights = {}
    hidden, rank, output_rank = 128, 128, 32
    for name, n, k in (
        ("wq_a", rank, hidden),
        ("wq_b", 32768, rank),
        ("wkv", 512, hidden),
        ("wo_a", output_rank * 8, 4096),
        ("wo_b", hidden, output_rank * 8),
    ):
        weights[f"v4.attn.{name}.weight"] = (torch.randn(n, k) * 2).to(
            torch.float8_e4m3fn
        )
        weights[f"v4.attn.{name}.scale"] = torch.randint(
            120, 123, (n // 32, k // 32), dtype=torch.uint8
        )
    weights["v4.attn.q_norm.weight"] = torch.randn(rank).bfloat16()
    weights["v4.attn.kv_norm.weight"] = torch.randn(512).bfloat16()
    weights["v4.attn.attn_sink"] = torch.randn(64)
    owns_index = layer in (2, 20, 24)
    owns_kv = layer in (2, 20)
    sources = V41LayerSources(
        layer,
        0 if layer == 0 else 2 if layer == 2 else 1,
        None if layer == 0 else 2 if layer == 2 else 20,
        layer if owns_index else None if layer == 0 else 20,
        20 if layer >= 20 else None,
    )
    if owns_index:
        weights["v4.indexer.wq_b.weight"] = (torch.randn(4096, rank) * 2).to(
            torch.float8_e4m3fn
        )
        weights["v4.indexer.wq_b.scale"] = torch.full(
            (128, rank // 32), 121, dtype=torch.uint8
        )
        weights["v4.indexer.weights_proj.weight"] = (
            torch.randn(32, hidden) / hidden**0.5
        ).bfloat16()
    if owns_kv:
        weights["v4.compressor.wkv.weight"] = (
            torch.randn(512, hidden) / hidden**0.5
        ).bfloat16()
        weights["v4.compressor.norm.weight"] = torch.randn(512).bfloat16()
        weights["v41.indexer.wk.weight"] = (torch.randn(128, 512) / 512**0.5).bfloat16()
        weights["v41.indexer.k_norm.weight"] = torch.randn(128).bfloat16()
        if layer == 2:
            weights["v4.compressor.wgate.weight"] = (
                torch.randn(512, hidden) / hidden**0.5
            ).bfloat16()
    frequencies = rope.precompute_freqs_cis(
        64, 512, 0 if layer == 0 else 65536, 10000 if layer == 0 else 160000, 16, 32, 1
    )
    return sources, weights, frequencies


def dense_reference(op, x):
    raw, scales = quantization_reference(x)
    return grouped_reference(
        raw.view(torch.float8_e4m3fn), scales, op.weight, op.scales
    )


class V41AttentionProjectionTest(unittest.TestCase):
    def test_query_index_and_grouped_output_follow_checkpoint_equations(self):
        args = weights_fixture()
        op, _ = queries.build_v41_projections(*args, backend="eager")
        x = torch.randn(3, 128).bfloat16()
        positions = torch.tensor([257, 5, 0])
        valid = torch.ones(3, dtype=torch.bool)
        frame = op.prepare(x, positions, valid)
        qr = norm(dense_reference(op.wq_a, x), op.q_norm)
        q = rotate(
            dense_reference(op.wq_b, qr).reshape(3, 64, 512),
            op.frequencies[positions, None],
        )
        iq = rotate(
            dense_reference(op.index_wq_b, qr).reshape(3, 32, 128),
            op.frequencies[positions, None],
        )
        torch.testing.assert_close(frame.query, q, atol=0, rtol=0)
        torch.testing.assert_close(frame.index_query, iq, atol=0, rtol=0)
        torch.testing.assert_close(
            frame.index_weights, F.linear(x, op.index_weights), atol=0, rtol=0
        )
        attention = torch.randn(3, 64, 512).bfloat16()
        before = attention.clone()
        unrotated = rotate(attention, op.frequencies[positions, None].conj()).reshape(
            3, 8, 4096
        )
        grouped = torch.stack(
            [F.linear(unrotated[:, g], op.wo_a[g]) for g in range(8)], dim=1
        ).flatten(1)
        torch.testing.assert_close(
            op.output(attention, frame),
            dense_reference(op.wo_b, grouped),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(attention, before, atol=0, rtol=0)
        wrong = torch.stack(
            [
                F.linear(attention.reshape(3, 8, 4096)[:, g], op.wo_a[g])
                for g in range(8)
            ],
            dim=1,
        ).flatten(1)
        self.assertFalse(torch.equal(grouped, wrong))

    def test_loader_preserves_raw_weights_and_converts_group32_wo_a_once(self):
        for layer in (0, 2, 21, 24):
            sources, weights, frequencies = weights_fixture(layer)
            original = dict(weights)
            op, kv = queries.build_v41_projections(
                sources, weights, frequencies, backend="eager"
            )
            self.assertEqual(weights.keys(), original.keys())
            for key in weights:
                self.assertIs(weights[key], original[key])
            self.assertIs(op.wq_a.weight, weights["v4.attn.wq_a.weight"])
            self.assertIs(kv.wkv.weight, weights["v4.attn.wkv.weight"])
            scale = (
                torch.exp2(weights["v4.attn.wo_a.scale"].float() - 127)
                .repeat_interleave(32, 0)
                .repeat_interleave(32, 1)
            )
            expected = (weights["v4.attn.wo_a.weight"].float() * scale).bfloat16()
            torch.testing.assert_close(op.wo_a.flatten(0, 1), expected, atol=0, rtol=0)
            self.assertEqual(op.index_wq_b is not None, sources.owns_index)
            self.assertEqual(kv.compressor_weight is not None, sources.owns_kv)
            if layer == 2:
                self.assertEqual(kv.compressor_weight.dtype, torch.float32)
                torch.testing.assert_close(
                    kv.compressor_weight,
                    weights["v4.compressor.wkv.weight"].float(),
                    atol=0,
                    rtol=0,
                )

    def test_query_frame_snapshots_positions_and_masks_and_handles_dead_rows(self):
        op, _ = queries.build_v41_projections(*weights_fixture(), backend="eager")
        x = torch.randn(3, 128).bfloat16()
        x[-1] = float("nan")
        positions = torch.tensor([8, 99, -999])
        valid = torch.tensor([True, True, False])
        frame = op.prepare(x, positions, valid)
        before = frame.frequencies.clone()
        positions.fill_(-123)
        valid.fill_(False)
        attention = torch.randn(3, 64, 512).bfloat16()
        attention[-1] = float("nan")
        output = op.output(attention, frame)
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertTrue(bool((output[-1] == 0).all()))
        self.assertTrue(bool((frame.query[-1] == 0).all()))
        self.assertTrue(bool((frame.index_query[-1] == 0).all()))
        torch.testing.assert_close(frame.frequencies, before, atol=0, rtol=0)
        self.assertEqual(frame.valid_rows.tolist(), [True, True, False])
        empty = op.prepare(x[:0], positions[:0], valid[:0])
        self.assertEqual(op.output(attention[:0], empty).shape, (0, 128))

    def test_invalid_loader_layout_rejected_without_consuming_weights(self):
        sources, weights, frequencies = weights_fixture()
        weights["v4.attn.wo_a.scale"] = weights["v4.attn.wo_a.scale"].view(torch.int32)
        before = dict(weights)
        with self.assertRaises(ValueError):
            queries.build_v41_projections(
                sources, weights, frequencies, backend="eager"
            )
        self.assertEqual(weights.keys(), before.keys())
        for name in before:
            self.assertIs(weights[name], before[name])
        sources, weights, frequencies = weights_fixture()
        weights.pop("v41.indexer.wk.weight")
        with self.assertRaisesRegex(ValueError, "index key weight"):
            queries.build_v41_projections(
                sources, weights, frequencies, backend="eager"
            )


if __name__ == "__main__":
    unittest.main()
