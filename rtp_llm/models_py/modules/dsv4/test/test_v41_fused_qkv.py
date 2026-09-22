"""Compare the merged Q/KV seam with the two production MXFP8 linears."""

import gc
import json
import os
import unittest
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import torch

from rtp_llm.models_py.modules.dsv4._fused_rmsnorm_rope_triton import fused_rmsnorm_rope
from rtp_llm.models_py.modules.dsv4._v41_fused_qkv import (
    is_supported,
    strided_q_rmsnorm,
    try_project_qr_kv,
)
from rtp_llm.models_py.modules.dsv4.utils import V41MXFP8Linear, merge_v41_qkv_weights
from rtp_llm.ops.compute_ops import rtp_llm_ops

_DIM, _Q, _KV, _RD = 5120, 1280, 512, 64
_EPS = 1e-6
_KEYS = ("qw", "qs", "kw", "ks")


def load_checkpoint_weights(checkpoint, layer):
    """Only materialize this layer's two projections and norm weights."""
    from safetensors import safe_open

    checkpoint = Path(checkpoint)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    suffixes = {
        "qw": "wq_a.weight",
        "qs": "wq_a.scale",
        "kw": "wkv.weight",
        "ks": "wkv.scale",
        "qn": "q_norm.weight",
        "kn": "kv_norm.weight",
    }
    weights = {}
    for key, suffix in suffixes.items():
        name = f"layers.{layer}.attn.{suffix}"
        with safe_open(
            str(checkpoint / index[name]), framework="pt", device="cpu"
        ) as f:
            weights[key] = f.get_tensor(name).cuda()
    weights["qn"] = weights["qn"].bfloat16()
    weights["kn"] = weights["kn"].bfloat16()
    return weights


def synthetic_weights():
    weights = {}
    for prefix, n in (("q", _Q), ("k", _KV)):
        weights[prefix + "w"] = (torch.randn(n, _DIM, device="cuda") * 0.04).to(
            torch.float8_e4m3fn
        )
        weights[prefix + "s"] = torch.ones(n // 32, _DIM // 32, device="cuda").to(
            torch.float8_e8m0fnu
        )
        weights[prefix + "n"] = (1 + torch.randn(n, device="cuda") * 0.1).bfloat16()
    return weights


def make_case(leading, weights=None):
    weights = synthetic_weights() if weights is None else weights.copy()
    merged = merge_v41_qkv_weights(weights, *_KEYS)
    if merged is None:
        raise AssertionError("expected merged V4.1 projections")
    x = torch.randn(*leading, _DIM, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(x.numel() // _DIM, _RD // 2, device="cuda")
    freqs = torch.polar(torch.ones_like(angles), angles)
    return dict(
        x=x,
        freqs=freqs,
        weights=weights,
        merged=merged,
        q=V41MXFP8Linear(weights["qw"], weights["qs"]),
        kv=V41MXFP8Linear(weights["kw"], weights["ks"]),
    )


def reference_norm(x, weight):
    flat = x.reshape(-1, x.shape[-1])
    output = torch.empty_like(flat)
    rtp_llm_ops.rmsnorm(
        output, flat, weight, _EPS, torch.cuda.current_stream().cuda_stream
    )
    return output.view(x.shape)


def unfused(case):
    qr = reference_norm(case["q"](case["x"]), case["weights"]["qn"])
    kv = fused_rmsnorm_rope(
        case["kv"](case["x"]), case["weights"]["kn"], case["freqs"], _RD, eps=_EPS
    )
    return qr, kv


def fused(case):
    result = try_project_qr_kv(
        case["merged"], case["x"], case["weights"]["qn"], _Q, _EPS
    )
    if result is None:
        raise AssertionError("unexpected Q/KV fallback")
    qr, kv_raw = result
    kv = fused_rmsnorm_rope(kv_raw, case["weights"]["kn"], case["freqs"], _RD, eps=_EPS)
    return qr, kv


def assert_outputs_close(actual, expected):
    for got, ref in zip(actual, expected):
        assert got.is_contiguous()
        assert torch.isfinite(got).all().item()
        torch.testing.assert_close(got, ref, rtol=0.008, atol=1e-5)


def errors(actual, expected):
    result = {}
    for name, got, ref in zip(("qr", "kv"), actual, expected):
        diff = got.float() - ref.float()
        result[name] = dict(
            max_abs=diff.abs().max().item(),
            rms=diff.square().mean().sqrt().item(),
            reference_rms=ref.float().square().mean().sqrt().item(),
        )
    return result


def capture(case, call, warmup=5):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            call(case)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outputs = call(case)
    graph.replay()
    torch.cuda.synchronize()
    return graph, outputs


class V41FusedQKVCPUTest(unittest.TestCase):
    def test_merge_allocations_use_feature_weights_region(self):
        from rtp_llm.models_py.modules.dsv4 import utils

        weights = {}
        for prefix, n in (("q", 64), ("k", 32)):
            weights[prefix + "w"] = torch.ones(n, 32, dtype=torch.uint8).view(
                torch.float8_e4m3fn
            )
            weights[prefix + "s"] = torch.full(
                (n // 32, 1), 127, dtype=torch.uint8
            ).view(torch.float8_e8m0fnu)
        active = False
        allocations = []
        original_cat = torch.cat

        @contextmanager
        def region():
            nonlocal active
            self.assertFalse(active)
            active = True
            try:
                yield
            finally:
                active = False

        def concatenate(*args, **kwargs):
            self.assertTrue(active)
            allocations.append("cat")
            return original_cat(*args, **kwargs)

        def build_linear(weight, scales):
            self.assertTrue(active)
            allocations.append("linear_and_packed_scales")
            return SimpleNamespace(weight=weight)

        # All tensors and operations are CPU; only bypass the device gate.
        with (
            patch.object(
                torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True
            ),
            patch.object(utils, "feature_weights_region", side_effect=region),
            patch.object(utils, "V41MXFP8Linear", side_effect=build_linear),
            patch.object(torch, "cat", side_effect=concatenate),
        ):
            merged = merge_v41_qkv_weights(weights, *_KEYS)
        self.assertFalse(active)
        self.assertEqual(allocations, ["cat", "cat", "linear_and_packed_scales"])
        self.assertEqual(merged.weight.device.type, "cpu")

    def test_unsupported_merge_does_not_mutate_weights(self):
        weights = {key: torch.ones(32, 32) for key in _KEYS}
        original = weights.copy()
        self.assertIsNone(merge_v41_qkv_weights(weights, *_KEYS))
        for key in _KEYS:
            self.assertIs(weights[key], original[key])

    def test_unsupported_linear_falls_back_without_backend(self):
        self.assertFalse(is_supported(None, None, None, _Q))
        self.assertIsNone(try_project_qr_kv(None, None, None, _Q, _EPS))

    def test_attention_fixture_without_merged_field_falls_back(self):
        from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8

        attention = AttentionFP8.__new__(AttentionFP8)
        self.assertIsNone(attention._try_fused_qr_kv(None))

    def test_native_scale_reload_slices_in_block32_units(self):
        from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8

        raw_w = torch.arange(128 * 256, dtype=torch.float32).view(128, 256)
        raw_s = torch.arange(4 * 8, dtype=torch.float32).view(4, 8)
        for rows, cols in (
            (slice(32, 96), None),
            (None, slice(64, 192)),
            (slice(32, 96), slice(64, 192)),
        ):
            linear = V41MXFP8Linear.__new__(V41MXFP8Linear)
            torch.nn.Module.__init__(linear)
            expected_w = raw_w[rows or slice(None), cols or slice(None)]
            sr = slice(rows.start // 32, rows.stop // 32) if rows else slice(None)
            sc = slice(cols.start // 32, cols.stop // 32) if cols else slice(None)
            expected_s = raw_s[sr, sc].clone()
            linear.weight = torch.empty_like(expected_w)
            linear.weight_scales = torch.empty_like(expected_s)
            linear._sleep_raw_weight_source = raw_w
            linear._sleep_raw_scale_source = raw_s
            linear._sleep_row_slice, linear._sleep_col_slice = rows, cols
            with patch.object(
                linear, "pack_scales", side_effect=lambda s: s.clone()
            ) as pack:
                AttentionFP8._reload_linear_scale(linear)
            torch.testing.assert_close(pack.call_args.args[0], expected_s)
            torch.testing.assert_close(linear.weight, expected_w)
            torch.testing.assert_close(linear.weight_scales, expected_s)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class V41FusedQKVCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("V4.1 MXFP8 GEMM requires Blackwell")

    def setUp(self):
        torch.manual_seed(4106)

    @torch.no_grad()
    def test_checkpoint_views_release_original_weight_storage(self):
        weights = synthetic_weights()
        originals = [weakref.ref(weights[key]) for key in _KEYS]
        expected = {key: weights[key].view(torch.uint8).clone() for key in _KEYS}
        merged = merge_v41_qkv_weights(weights, *_KEYS)
        gc.collect()
        self.assertTrue(all(ref() is None for ref in originals))
        for key in _KEYS:
            torch.testing.assert_close(weights[key].view(torch.uint8), expected[key])
        allocation = merged.weight.untyped_storage().data_ptr()
        for key in ("qw", "kw"):
            self.assertEqual(allocation, weights[key].untyped_storage().data_ptr())
            self.assertTrue(weights[key].is_contiguous())
        self.assertEqual(merged.weight.untyped_storage().nbytes(), (_Q + _KV) * _DIM)

    @torch.no_grad()
    def test_norm_strided_input_matches_framework(self):
        for rows in (1, 6, 24, 31, 128, 8192):
            with self.subTest(rows=rows):
                source = torch.randn(
                    rows, _Q + _KV, device="cuda", dtype=torch.bfloat16
                )
                weight = torch.randn(_Q, device="cuda", dtype=torch.bfloat16)
                x = source[:, :_Q]
                expected = reference_norm(x.contiguous(), weight)
                actual = strided_q_rmsnorm(x, weight, _EPS)
                assert_outputs_close((actual,), (expected,))

    @torch.no_grad()
    def test_shapes_and_fallback(self):
        weights = synthetic_weights()
        for leading in ((1, 1), (1, 6), (2, 6), (4, 6), (31,), (128,), (8192,)):
            with self.subTest(leading=leading):
                case = make_case(leading, weights)
                expected = unfused(case)
                assert_outputs_close(fused(case), expected)
                # Unsupported dtype still uses the independent projections.
                self.assertIsNone(
                    try_project_qr_kv(
                        case["merged"],
                        case["x"].float(),
                        case["weights"]["qn"],
                        _Q,
                        _EPS,
                    )
                )
                # The original projection modules still use shared weights.
                assert_outputs_close(unfused(case), expected)

    @torch.no_grad()
    def test_graph_replay_reads_changed_input_and_frequencies(self):
        case = make_case((4, 6))
        graph, output = capture(case, fused)
        for scale in (0.0, 1e-3, 1.0, 100.0):
            case["x"].normal_().mul_(scale)
            phase = torch.randn_like(case["freqs"].real)
            case["freqs"].copy_(torch.polar(torch.ones_like(phase), phase))
            expected = unfused(case)
            graph.replay()
            torch.cuda.synchronize()
            assert_outputs_close(output, expected)

    @torch.no_grad()
    def test_merged_scale_reload_preserves_graph_addresses(self):
        from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8

        case = make_case((4, 6))
        expected = tuple(t.clone() for t in fused(case))
        linear = case["merged"]
        pointers = linear.weight.data_ptr(), linear.weight_scales.data_ptr()
        linear.weight_scales.zero_()
        AttentionFP8._reload_linear_scale(linear)
        self.assertEqual(
            pointers, (linear.weight.data_ptr(), linear.weight_scales.data_ptr())
        )
        assert_outputs_close(fused(case), expected)

    @torch.no_grad()
    def test_native_scale_reload_with_tp_row_and_column_slices(self):
        from rtp_llm.models_py.modules.dsv4.fp8.attention import AttentionFP8

        raw_w = torch.randn(1024, 512, device="cuda").to(torch.float8_e4m3fn)
        raw_s = torch.exp2(torch.randint(-5, 3, (32, 16), device="cuda").float()).to(
            torch.float8_e8m0fnu
        )
        for rows, cols in (
            (slice(256, 512), None),
            (None, slice(128, 384)),
            (slice(256, 512), slice(128, 384)),
        ):
            sr = slice(rows.start // 32, rows.stop // 32) if rows else slice(None)
            sc = slice(cols.start // 32, cols.stop // 32) if cols else slice(None)
            expected_w = raw_w[rows or slice(None), cols or slice(None)].contiguous()
            linear = V41MXFP8Linear(expected_w.clone(), raw_s[sr, sc].contiguous())
            expected_s = linear.weight_scales.clone()
            linear._sleep_raw_weight_source = raw_w
            linear._sleep_raw_scale_source = raw_s
            linear._sleep_row_slice, linear._sleep_col_slice = rows, cols
            pointers = linear.weight.data_ptr(), linear.weight_scales.data_ptr()
            linear.weight.view(torch.uint8).zero_()
            linear.weight_scales.zero_()
            AttentionFP8._reload_linear_scale(linear)
            torch.testing.assert_close(
                linear.weight.view(torch.uint8), expected_w.view(torch.uint8)
            )
            torch.testing.assert_close(linear.weight_scales, expected_s)
            self.assertEqual(
                pointers, (linear.weight.data_ptr(), linear.weight_scales.data_ptr())
            )

    @unittest.skipUnless(
        os.environ.get("DSV41_TEST_CHECKPOINT"), "real checkpoint opt-in"
    )
    @torch.no_grad()
    def test_real_checkpoint_projection_and_graph_replay(self):
        for layer in (0, 17, 39):
            weights = load_checkpoint_weights(
                os.environ["DSV41_TEST_CHECKPOINT"], layer
            )
            for leading in ((1, 6), (2, 6), (4, 6), (8192,)):
                with self.subTest(layer=layer, leading=leading):
                    case = make_case(leading, weights)
                    expected = unfused(case)
                    assert_outputs_close(fused(case), expected)
                    graph, outputs = capture(case, fused)
                    case["x"].normal_()
                    expected = unfused(case)
                    graph.replay()
                    torch.cuda.synchronize()
                    assert_outputs_close(outputs, expected)


if __name__ == "__main__":
    unittest.main()
