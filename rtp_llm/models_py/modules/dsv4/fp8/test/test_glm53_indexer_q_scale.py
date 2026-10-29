"""GLM53 Q quantization follows the official 128-element UE8M0 contract."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8._indexer_hadamard_quant_triton import (
    indexer_hadamard_quant_fold,
)
from rtp_llm.models_py.modules.dsv4.fp8._indexer_q_quant_triton import (
    indexer_q_fp8_quant_fold,
)
from rtp_llm.models_py.modules.dsv4.fp8._kv_cache_utils import PoolBackedModule
from rtp_llm.models_py.modules.dsv4.fp8.indexer import IndexerFP8
from rtp_llm.models_py.triton_kernels.sparse_mla.fused_prefill_rope_hadamard import (
    hadamard_transform_128,
)


def rotate_reference(q):
    # Independent FP32 butterfly with the explicit BF16 materialization in
    # vLLM GLM5Next fwht128_quant_fp8; no production Hadamard helper is used.
    x = q.float()
    for stride in (1, 2, 4, 8, 16, 32, 64):
        pairs = x.reshape(*q.shape[:-1], -1, 2, stride)
        a, b = pairs[..., 0, :], pairs[..., 1, :]
        x = torch.stack((a + b, a - b), dim=-2).reshape(q.shape)
    return (x * (128**-0.5)).bfloat16()


def quant_reference(q, gates, round_scale=True):
    x = q.float()
    absmax = x.abs().amax(dim=-1)
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(absmax.clamp_min(1e-4) * (1 / 448))))
    else:
        scale = (absmax / 448).clamp_min(1e-12)
    return (x / scale[..., None]).clamp(-448, 448).to(
        torch.float8_e4m3fn
    ), gates.float() * scale


def model(kpool=True):
    indexer = IndexerFP8.__new__(IndexerFP8)
    PoolBackedModule.__init__(indexer)
    indexer.compressor = SimpleNamespace(kpool_mode=kpool)
    indexer.rotate_q = True
    return indexer


def inputs(seq):
    generator = torch.Generator(device="cuda").manual_seed(53 + seq)
    q = torch.randn(2, seq, 32, 128, generator=generator, device="cuda").bfloat16()
    q.reshape(-1, 128)[0].zero_()
    q.reshape(-1, 128)[1].mul_(1e-6)
    gates = torch.randn(2, seq, 32, generator=generator, device="cuda")
    gates.reshape(-1)[:4] = torch.tensor(
        [1.001953125, -1.0029296875, 0, -0.125], device="cuda"
    )
    return q, gates


def assert_quant_equal(actual, expected):
    torch.testing.assert_close(
        actual[0].view(torch.uint8), expected[0].view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    assert actual[1].dtype == torch.float32


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Glm53IndexerQScaleTest(unittest.TestCase):
    def test_fused_masked_rows_match_the_ue8m0_reference(self):
        # Neither row count fills the fused quantizer's 16-row launch tile.
        for rows in (17, 65):
            generator = torch.Generator(device="cuda").manual_seed(5300 + rows)
            q = torch.randn(
                1, rows, 1, 128, generator=generator, device="cuda"
            ).bfloat16()
            gates = torch.randn(1, rows, 1, generator=generator, device="cuda")
            q[:, -1].zero_()
            q[:, -2].mul_(1e-6)
            for dtype in (torch.float32, torch.bfloat16):
                weights = gates.to(dtype)
                with self.subTest(rows=rows, dtype=dtype):
                    expected = quant_reference(rotate_reference(q), weights)
                    assert_quant_equal(
                        indexer_hadamard_quant_fold(q, weights, round_scale=True),
                        expected,
                    )
                    assert_quant_equal(
                        indexer_q_fp8_quant_fold(
                            rotate_reference(q), weights, round_scale=True
                        ),
                        expected,
                    )

    def test_glm53_fused_and_unfused_match_the_ue8m0_reference(self):
        for seq in (1, 4, 17):
            q, gates = inputs(seq)
            rotated = rotate_reference(q)
            torch.testing.assert_close(
                hadamard_transform_128(q), rotated, rtol=0, atol=0
            )
            for dtype in (torch.float32, torch.bfloat16):
                weights = gates.to(dtype)
                expected = quant_reference(rotated, weights)
                for fused in ("0", "1"):
                    with self.subTest(seq=seq, dtype=dtype, fused=fused), patch.dict(
                        os.environ, {"GLM53_INDEXER_FUSED_Q_QUANT": fused}
                    ):
                        actual = model()._quantize_q(
                            q if fused == "1" else rotated, weights
                        )
                        assert_quant_equal(actual, expected)

    def test_graph_replay_observes_changed_q_and_signed_gates(self):
        q, gates = inputs(4)
        original_q, original_gates = q.clone(), gates.clone()
        for fused in ("0", "1"):
            with self.subTest(fused=fused), patch.dict(
                os.environ, {"GLM53_INDEXER_FUSED_Q_QUANT": fused}
            ):
                indexer = model()

                def run():
                    rotated = q if fused == "1" else hadamard_transform_128(q)
                    return indexer._quantize_q(rotated, gates)

                for _ in range(3):
                    run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = run()
                for amplitude in (1.0, 0.0, 1e-6, 2.0):
                    q.copy_(original_q * amplitude)
                    gates.copy_(original_gates * (-1 if amplitude == 0 else 1))
                    graph.replay()
                    assert_quant_equal(
                        actual, quant_reference(rotate_reference(q), gates)
                    )

    def test_dsv4_default_retains_unrounded_scales(self):
        q, gates = inputs(4)
        expected = quant_reference(q, gates, round_scale=False)
        assert_quant_equal(indexer_q_fp8_quant_fold(q, gates), expected)
        assert_quant_equal(
            indexer_hadamard_quant_fold(q, gates, rotate=False), expected
        )
        with patch.dict(os.environ, {"GLM53_INDEXER_FUSED_Q_QUANT": "1"}):
            assert_quant_equal(model(kpool=False)._quantize_q(q, gates), expected)

    def test_empty_q_preserves_output_shapes_and_dtypes(self):
        q = torch.empty(2, 0, 32, 128, device="cuda", dtype=torch.bfloat16)
        gates = torch.empty(2, 0, 32, device="cuda")
        for fused in ("0", "1"):
            with patch.dict(os.environ, {"GLM53_INDEXER_FUSED_Q_QUANT": fused}):
                actual = model()._quantize_q(q, gates)
                self.assertEqual(actual[0].shape, q.shape)
                self.assertEqual(actual[0].dtype, torch.float8_e4m3fn)
                self.assertEqual(actual[1].shape, gates.shape)
                self.assertEqual(actual[1].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
