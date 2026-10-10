"""WKV provider selection, buffer ownership and restoration on SM120."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8.test import v41_kv_projection_reference as scope
from rtp_llm.models_py.modules.dsv4.fp8.test import v41_query_precision
from rtp_llm.models_py.modules.dsv4.fp8.test.test_v41_attention_owner_sm120 import (
    make_layer,
)
from rtp_llm.models_py.modules.dsv4.fp8.test.v41_expert_precision import (
    load_reference_kernel,
)
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import V41MXFP8Linear


@unittest.skipUnless(torch.cuda.is_available(), "Requires SM120")
class KVProjectionReferenceTest(unittest.TestCase):
    def setUp(self):
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        torch.manual_seed(41243)
        self.owner = make_layer(gen=0)
        self.other = make_layer(gen=0)
        self.x = torch.randn(5, 512, device="cuda", dtype=torch.bfloat16)

    def exact(self, left, right):
        self.assertEqual(left.shape, right.shape)
        self.assertEqual(left.dtype, right.dtype)
        self.assertTrue(
            torch.equal(
                left.contiguous().view(torch.uint8),
                right.contiguous().view(torch.uint8),
            )
        )

    @torch.inference_mode()
    def test_both_widths_entry_paths_and_unselected_providers(self):
        original = V41MXFP8Linear.forward_quantized
        for width in (512, 5120):
            owner = self.owner if width == 512 else make_layer(gen=0, dim=width)
            value = torch.randn(5, width, device="cuda", dtype=torch.bfloat16)
            baseline = owner._lin(owner.wkv, value).clone()
            query = owner._lin(owner.wq_a, value).clone()
            default = self.other._lin(self.other.wkv, self.x).clone()
            quant = owner.wkv._quantize_input(value)
            calls = []

            def selected(provider, values, scales, kernel, *, out=None):
                calls.append(provider)
                return original(provider, values, scales, out=out)

            with patch.object(scope, "_dense_down", side_effect=selected):
                with scope.kv_projection_reference(
                    True, attentions=[owner], kernel=object()
                ):
                    self.exact(baseline, owner._lin(owner.wkv, value))
                    self.exact(
                        baseline,
                        owner._lin_from_shared_quant(owner.wkv, quant, value.shape),
                    )
                    self.exact(query, owner._lin(owner.wq_a, value))
                    self.exact(default, self.other._lin(self.other.wkv, self.x))
            self.assertEqual(calls, [owner.wkv._sm120_linear] * 2)
            self.assertIs(V41MXFP8Linear.forward_quantized, original)

    def test_disabled_nested_and_exception_restoration(self):
        original = V41MXFP8Linear.forward_quantized
        with scope.kv_projection_reference(False):
            self.assertIs(V41MXFP8Linear.forward_quantized, original)
        with scope.kv_projection_reference(
            True, attentions=[self.owner], kernel=object()
        ):
            outer = V41MXFP8Linear.forward_quantized
            with scope.kv_projection_reference(False):
                self.assertIs(V41MXFP8Linear.forward_quantized, outer)
            with scope.kv_projection_reference(
                True, attentions=[self.other], kernel=object()
            ):
                self.assertIsNot(V41MXFP8Linear.forward_quantized, outer)
            self.assertIs(V41MXFP8Linear.forward_quantized, outer)
            with self.assertRaisesRegex(RuntimeError, "scope-control"):
                with scope.kv_projection_reference(
                    True, attentions=[self.other], kernel=object()
                ):
                    raise RuntimeError("scope-control")
            self.assertIs(V41MXFP8Linear.forward_quantized, outer)
        self.assertIs(V41MXFP8Linear.forward_quantized, original)

    def test_rejected_selectors_and_aliases(self):
        original = V41MXFP8Linear.forward_quantized
        for owners in ((), (self.owner, self.owner), (object(),)):
            with self.subTest(owners=owners), self.assertRaisesRegex(
                ValueError, "owners"
            ):
                with scope.kv_projection_reference(
                    True, attentions=owners, kernel=object()
                ):
                    pass
        with self.assertRaisesRegex(ValueError, "bool"):
            with scope.kv_projection_reference(
                1, attentions=[self.owner], kernel=object()
            ):
                pass
        for target, name, value in (
            (self.owner, "dim", 1024),
            (self.owner, "head_dim", 256),
            (self.owner.wkv._sm120_linear, "backend", "deepgemm"),
        ):
            with patch.object(target, name, value), self.assertRaisesRegex(
                ValueError, "providers"
            ):
                with scope.kv_projection_reference(
                    True, attentions=[self.owner], kernel=object()
                ):
                    pass
        with patch.object(self.other, "wkv", self.owner.wkv), self.assertRaisesRegex(
            ValueError, "distinct WKV"
        ):
            with scope.kv_projection_reference(
                True, attentions=[self.owner, self.other], kernel=object()
            ):
                pass
        self.assertIs(V41MXFP8Linear.forward_quantized, original)

    @torch.inference_mode()
    def test_query_scope_composition_both_orders(self):
        original = V41MXFP8Linear.forward_quantized
        expected_query = self.owner._lin(self.owner.wq_a, self.x).clone()
        expected_kv = self.owner._lin(self.owner.wkv, self.x).clone()
        for kv_first in (False, True):
            calls = []

            def selected(name):
                def forward(provider, values, scales, kernel, *, out=None):
                    calls.append((name, provider))
                    return original(provider, values, scales, out=out)

                return forward

            with patch.object(
                scope, "_dense_down", side_effect=selected("kv")
            ), patch.object(
                v41_query_precision, "_dense_down", side_effect=selected("query")
            ):
                kv = scope.kv_projection_reference(
                    True, attentions=[self.owner], kernel=object()
                )
                query = v41_query_precision.query_projection_reference(
                    True, kernel=object()
                )
                first, second = (kv, query) if kv_first else (query, kv)
                with first, second:
                    self.exact(expected_query, self.owner._lin(self.owner.wq_a, self.x))
                    self.exact(expected_kv, self.owner._lin(self.owner.wkv, self.x))
            self.assertEqual(
                calls,
                [
                    ("query", self.owner.wq_a._sm120_linear),
                    ("kv", self.owner.wkv._sm120_linear),
                ],
            )
            self.assertIs(V41MXFP8Linear.forward_quantized, original)

    @torch.inference_mode()
    def test_output_buffers_empty_rows_and_weight_alias_rejection(self):
        kernel = load_reference_kernel()
        provider = self.owner.wkv._sm120_linear
        protected = provider.weight.clone()
        value = self.x[:1].contiguous()
        alias = (
            provider.weight.view(torch.uint8)
            .view(torch.bfloat16)
            .reshape(-1)[:512]
            .reshape(1, 512)
        )
        with scope.kv_projection_reference(
            True, attentions=[self.owner], kernel=kernel
        ):
            expected = self.owner._lin(self.owner.wkv, value).clone()
            out = torch.empty_like(expected)
            self.assertIs(self.owner._lin(self.owner.wkv, value, out=out), out)
            self.exact(expected, out)
            for bad in (
                torch.empty_like(out, dtype=torch.float32),
                torch.empty(2, 512, device="cuda", dtype=torch.bfloat16),
                alias,
            ):
                with self.subTest(shape=bad.shape), self.assertRaisesRegex(
                    ValueError, "out"
                ):
                    self.owner._lin(self.owner.wkv, value, out=bad)
            empty = torch.empty(0, 512, device="cuda", dtype=torch.bfloat16)
            empty_quant = torch.empty(0, 512, device="cuda", dtype=torch.float8_e4m3fn)
            empty_scales = torch.empty(0, 16, device="cuda", dtype=torch.uint8)
            self.assertIs(
                provider.forward_quantized(empty_quant, empty_scales, out=empty), empty
            )
        self.exact(protected, provider.weight)


if __name__ == "__main__":
    unittest.main()
