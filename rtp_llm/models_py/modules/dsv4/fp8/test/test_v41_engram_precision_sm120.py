"""Qualify scoped Engram gating with scalar, alias and changing-graph oracles."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rtp_llm.models_py.modules.dsv4 import engram
from rtp_llm.models_py.modules.dsv4.fp8.test import v41_engram_precision as precision
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_engram_precision import (
    engram_gate_reference,
    reference_engram_gate,
)
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import V41MXFP8Linear


class EngramPrecisionTest(unittest.TestCase):
    def test_projection_selects_only_engram_and_restores_nested_exception(self):
        original_rows = engram.Engram._forward_rows
        original_dense = V41MXFP8Linear.forward_quantized
        selected, other = object(), object()
        owner = SimpleNamespace(wkv=SimpleNamespace(_sm120_linear=selected))

        def rows(owner, *, fail=False):
            chosen = V41MXFP8Linear.forward_quantized(selected, None, None)
            untouched = V41MXFP8Linear.forward_quantized(other, None, None)
            if fail:
                raise RuntimeError("projection scope failure")
            return chosen, untouched

        def unchanged(provider, *args, **kwargs):
            return "original", provider

        with patch.object(engram.Engram, "_forward_rows", rows), patch.object(
            V41MXFP8Linear, "forward_quantized", unchanged
        ), patch.object(precision, "_dense_down", return_value="released"):
            with precision.engram_projection_reference(False):
                self.assertEqual(
                    engram.Engram._forward_rows(owner)[0], ("original", selected)
                )
            with precision.engram_projection_reference(True, kernel=object()):
                with precision.engram_projection_reference(True, kernel=object()):
                    self.assertEqual(
                        engram.Engram._forward_rows(owner),
                        ("released", ("original", other)),
                    )
                with self.assertRaisesRegex(RuntimeError, "projection scope failure"):
                    engram.Engram._forward_rows(owner, fail=True)
                self.assertEqual(
                    V41MXFP8Linear.forward_quantized(selected, None, None),
                    ("original", selected),
                )
                self.assertEqual(engram.Engram._forward_rows(owner)[0], "released")
        self.assertIs(engram.Engram._forward_rows, original_rows)
        self.assertIs(V41MXFP8Linear.forward_quantized, original_dense)
        with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
            with self.assertRaisesRegex(ValueError, "SM120"):
                with precision.engram_projection_reference(True, kernel=object()):
                    self.fail("Unsupported device entered projection diagnostic")

    @torch.inference_mode()
    def test_scalar_signed_clamp_mask_writeback_and_sources(self):
        dim = 32
        hidden = torch.full((2, 4, dim), 2.0, device="cuda", dtype=torch.bfloat16)
        kv = torch.ones(2, 5 * dim, device="cuda", dtype=torch.bfloat16)
        kv[:, 4 * dim :] = 3
        scales = (1.0, -1.0, 0.0, 2.0)
        q = torch.tensor(scales, device="cuda", dtype=torch.bfloat16)[:, None]
        q = q.expand(4, dim).contiguous()
        k = torch.ones_like(q)
        mask = torch.tensor([True, False], device="cuda")
        sources = [value.clone() for value in (kv, q, k, mask)]
        oracle = []
        for scale in scales:
            dot = math.sqrt(dim) * scale
            signed_root = math.copysign(math.sqrt(max(abs(dot), 1e-6)), dot)
            oracle.append(2.0 + 3.0 / (1.0 + math.exp(-signed_root)))
        expected = torch.tensor(oracle, dtype=torch.bfloat16)[:, None].expand(4, dim)
        with engram_gate_reference(True):
            result = engram.gated_engram_residual(hidden, kv, q, k, 0.0, mask)
            aliased = engram.gated_engram_residual(
                hidden, kv, q, k, 0.0, mask, out=hidden
            )
        self.assertEqual(aliased.data_ptr(), hidden.data_ptr())
        self.assertTrue(torch.equal(result, aliased))
        self.assertTrue(torch.equal(result[0].cpu(), expected))
        self.assertTrue(torch.equal(result[1], torch.full_like(result[1], 2)))
        for before, after in zip(sources, (kv, q, k, mask)):
            self.assertTrue(torch.equal(before, after))
        storage = torch.zeros(3, 4, dim, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "hidden alias"):
            reference_engram_gate(storage[:2], kv, q, k, 1e-6, out=storage[1:])
        aliased_kv = torch.zeros(2, 5 * dim, device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "another input"):
            reference_engram_gate(
                hidden,
                aliased_kv,
                q,
                k,
                1e-6,
                out=aliased_kv.reshape(-1)[: hidden.numel()].reshape_as(hidden),
            )

    def test_disabled_nested_exception_and_capability_restoration(self):
        original = engram.gated_engram_residual
        with engram_gate_reference(False):
            self.assertIs(engram.gated_engram_residual, original)
        with self.assertRaisesRegex(RuntimeError, "scope failure"):
            with engram_gate_reference(True):
                self.assertIs(engram.gated_engram_residual, reference_engram_gate)
                with engram_gate_reference(True), engram_gate_reference(False):
                    self.assertIs(engram.gated_engram_residual, reference_engram_gate)
                raise RuntimeError("scope failure")
        self.assertIs(engram.gated_engram_residual, original)
        with patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)):
            with self.assertRaisesRegex(ValueError, "SM120"):
                with engram_gate_reference(True):
                    self.fail("Unsupported device entered Engram diagnostic")
        self.assertIs(engram.gated_engram_residual, original)

    @torch.inference_mode()
    def test_changed_inputs_and_masks_graph_writeback(self):
        torch.manual_seed(411)
        for tokens in (1, 5):
            with self.subTest(tokens=tokens), engram_gate_reference(True):

                def values():
                    return (
                        torch.randn(
                            tokens, 4, 5120, device="cuda", dtype=torch.bfloat16
                        ),
                        torch.randn(
                            tokens, 5 * 5120, device="cuda", dtype=torch.bfloat16
                        ),
                        torch.randn(4, 5120, device="cuda", dtype=torch.bfloat16),
                        torch.randn(4, 5120, device="cuda", dtype=torch.bfloat16),
                        torch.ones(tokens, device="cuda", dtype=torch.bool),
                    )

                static = values()

                def forward():
                    return engram.gated_engram_residual(
                        *static[:4], 1e-6, static[4], out=static[0]
                    )

                forward()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = forward()
                self.assertEqual(captured.data_ptr(), static[0].data_ptr())
                for iteration in range(3):
                    changed = values()
                    changed[4][iteration % tokens] = False
                    eager = engram.gated_engram_residual(*changed[:4], 1e-6, changed[4])
                    for destination, source in zip(static, changed):
                        destination.copy_(source)
                    graph.replay()
                    self.assertTrue(torch.equal(captured, eager))
                    self.assertTrue(
                        torch.equal(captured[~changed[4]], changed[0][~changed[4]])
                    )


if __name__ == "__main__":
    unittest.main()
