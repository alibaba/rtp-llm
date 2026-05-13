"""Contracts for DSV4 static weight/scale conversions."""

from __future__ import annotations

import inspect
import re
import unittest

import torch

import rtp_llm.models_py.kernels.cuda.deepgemm_wrapper as deepgemm_wrapper
from rtp_llm.models.deepseek_v4 import DeepSeekV4Weight
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    fp8_fp4_gemm_nt,
    has_deep_gemm,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.dsv4.compressor import Compressor
from rtp_llm.models_py.modules.dsv4.moe.strategies.grouped_fp4 import (
    GroupedFP4Strategy,
)
from rtp_llm.models_py.modules.dsv4.moe.strategies.local_loop import (
    LocalLoopStrategy,
    _select_mn_major_scale_for_index,
)
from rtp_llm.models_py.modules.dsv4.moe.strategies.mega import MegaMoEStrategy
from rtp_llm.models_py.modules.dsv4.quant_layouts import (
    FP4_BLOCK,
    FP8_BLOCK,
    prepare_fp4_weight_scale_for_deepgemm,
)
from rtp_llm.models_py.modules.dsv4.qlinear import QuantizedLinear
from rtp_llm.models_py.modules.dsv4.transformer import V4Transformer


def _requires_sm100_deepgemm(testcase: unittest.TestCase) -> None:
    if not torch.cuda.is_available():
        testcase.skipTest("CUDA is not available")
    if torch.cuda.get_device_capability()[0] < 10:
        testcase.skipTest("SM100 is required")
    if not has_deep_gemm():
        testcase.skipTest("DeepGEMM is not available")


def _make_fp4_tensors(out_dim: int, in_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.randint(
        -128,
        127,
        (out_dim, in_dim // 2),
        dtype=torch.int8,
        device="cuda",
    )
    scale_bytes = torch.full(
        (out_dim, in_dim // FP4_BLOCK),
        127,
        dtype=torch.uint8,
        device="cuda",
    )
    return weight, scale_bytes.view(torch.float8_e8m0fnu)


class DSV4WeightConversionContractTest(unittest.TestCase):
    def test_static_weight_conversions_are_not_in_hot_paths(self) -> None:
        banned = [
            "self.scale.float()",
            "w1_s.float()",
            "w2_s.float()",
            "w3_s.float()",
            "_s13.float()",
            "_s2.float()",
            'compressor_weights["wkv"].to(torch.bfloat16)',
            'compressor_weights["wgate"].to(torch.bfloat16)',
            "gw[W.lm_head].float()",
        ]
        hot_sources = [
            inspect.getsource(QuantizedLinear._fp4_forward_deepgemm),
            inspect.getsource(LocalLoopStrategy._forward_topk_bs1),
            inspect.getsource(LocalLoopStrategy._forward_topk_bsN),
            inspect.getsource(GroupedFP4Strategy.forward),
            inspect.getsource(MegaMoEStrategy.setup_weights),
            inspect.getsource(Compressor.__init__),
            inspect.getsource(V4Transformer.__init__),
            inspect.getsource(deepgemm_wrapper.fp8_fp4_gemm_nt),
            inspect.getsource(deepgemm_wrapper.m_grouped_fp8_fp4_gemm_nt_contiguous),
            inspect.getsource(deepgemm_wrapper.m_grouped_fp8_fp4_gemm_nt_masked),
        ]
        joined = "\n".join(hot_sources)
        for needle in banned:
            self.assertNotIn(needle, joined)
        self.assertIn("_require_sm100_packed_scale_for_fp8_fp4", joined)

    def test_descriptor_converts_static_weights_at_load_time(self) -> None:
        compressor_src = inspect.getsource(DeepSeekV4Weight._build_compressor)
        self.assertRegex(
            compressor_src,
            re.compile(r"wkv_name,.*?data_type=torch\.bfloat16", re.S),
        )
        self.assertRegex(
            compressor_src,
            re.compile(r"wgate_name,.*?data_type=torch\.bfloat16", re.S),
        )

        all_src = inspect.getsource(DeepSeekV4Weight)
        lm_head_pos = all_src.index("W.lm_head")
        lm_head_src = all_src[lm_head_pos : lm_head_pos + 500]
        self.assertIn("data_type=torch.float32", lm_head_src)
        self.assertNotIn("data_type=torch.bfloat16", lm_head_src)

    def test_fp4_weight_scale_is_prepacked_before_gemm(self) -> None:
        _requires_sm100_deepgemm(self)
        torch.manual_seed(20260507)
        m, in_dim, out_dim = 16, 128, 128
        x = torch.randn(m, in_dim, dtype=torch.bfloat16, device="cuda").contiguous()
        x_fp8, x_scale = sgl_per_token_group_quant_fp8(
            x,
            group_size=FP8_BLOCK,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        weight, raw_scale = _make_fp4_tensors(out_dim, in_dim)
        packed_scale = prepare_fp4_weight_scale_for_deepgemm(
            raw_scale, out_dim, in_dim
        )
        self.assertEqual(packed_scale.dtype, torch.int32)

        out = torch.empty(m, out_dim, dtype=torch.bfloat16, device="cuda")
        fp8_fp4_gemm_nt(
            (x_fp8, x_scale),
            (weight, packed_scale),
            out,
            recipe_a=(1, FP8_BLOCK),
            recipe_b=(1, FP4_BLOCK),
        )
        torch.cuda.synchronize()

        with self.assertRaisesRegex(RuntimeError, "prepacked int32 scales"):
            fp8_fp4_gemm_nt(
                (x_fp8, x_scale),
                (weight, raw_scale.float()),
                out,
                recipe_a=(1, FP8_BLOCK),
                recipe_b=(1, FP4_BLOCK),
            )

    def test_topk_scale_select_preserves_mn_major_stride(self) -> None:
        grouped = torch.empty_strided(
            (4, 16, 8),
            (16 * 8, 1, 16),
            dtype=torch.int32,
        )
        expert_idx = torch.tensor([0], dtype=torch.long)

        bad = torch.index_select(grouped, 0, expert_idx).squeeze(0)
        self.assertNotEqual(bad.stride(-2), 1)

        selected = _select_mn_major_scale_for_index(
            grouped.transpose(-1, -2),
            expert_idx,
        )
        self.assertEqual(selected.shape, (16, 8))
        self.assertEqual(selected.stride(-2), 1)
        self.assertEqual(selected.stride(-1), 16)

    def test_quantized_linear_forward_uses_packed_scale_only(self) -> None:
        _requires_sm100_deepgemm(self)
        torch.manual_seed(20260507)
        m, in_dim, out_dim = 16, 128, 128
        weight, raw_scale = _make_fp4_tensors(out_dim, in_dim)
        layer = QuantizedLinear(in_dim, out_dim, storage="fp4")
        layer.bind_fp4_weight(weight, raw_scale)
        self.assertEqual(layer.scale_gemm.dtype, torch.int32)

        layer.scale = None
        x = torch.randn(m, in_dim, dtype=torch.bfloat16, device="cuda")
        out = layer(x)
        self.assertEqual(out.shape, (m, out_dim))
        self.assertEqual(out.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
