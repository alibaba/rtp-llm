"""CPU contracts for per-layer grouped output weight preparation."""

import sys
import types
import unittest
from unittest.mock import Mock, patch

import torch

from rtp_llm.models_py.modules.dsv4.fp8.attention_backend import SparseAttentionBackend


class GroupedOutputWeightsTest(unittest.TestCase):
    def test_default_preserves_deepgemm_expansion_and_packed_scale_contract(self):
        groups, rank, width = 2, 256, 512
        weight = torch.ones(groups * rank, width, dtype=torch.float8_e4m3fn)
        raw = (
            torch.arange(groups * (rank // 128) * (width // 128))
            .reshape(groups * rank // 128, width // 128)
            .float()
            .exp2()
        )
        packed = torch.zeros(groups, rank, width // 512, dtype=torch.int32)
        pack = Mock(return_value=packed)
        layout = types.SimpleNamespace(
            get_mn_major_tma_aligned_packed_ue8m0_tensor=pack
        )
        backend = SparseAttentionBackend()
        with patch.dict(sys.modules, {"deep_gemm.utils.layout": layout}):
            prepared_w, prepared_s = backend.prepare_grouped_output(
                weight, raw, groups, rank, width
            )
        expected = raw.view(groups, rank // 128, width // 128).repeat_interleave(
            128, dim=1
        )
        torch.testing.assert_close(pack.call_args.args[0], expected, rtol=0, atol=0)
        self.assertEqual(pack.call_count, 1)
        self.assertIs(prepared_s, packed)
        self.assertEqual(prepared_w.shape, (groups, rank, width))
        self.assertEqual(prepared_w.data_ptr(), weight.data_ptr())
        self.assertTrue(prepared_w.is_contiguous())

        kernel = Mock(side_effect=lambda equation, a, b, out, **kw: out.fill_(7))
        activation = torch.ones(3, groups, width, dtype=torch.float8_e4m3fn)
        scales = object()
        with patch.dict(
            sys.modules, {"deep_gemm": types.SimpleNamespace(fp8_einsum=kernel)}
        ):
            for _ in range(2):
                result = backend.grouped_output(
                    activation, scales, prepared_w, prepared_s, rank
                )
                self.assertIs(kernel.call_args.args[2][1], packed)
                self.assertEqual(kernel.call_args.kwargs["recipe"], (1, 1, 128))
                self.assertTrue(torch.all(result == 7))
        self.assertEqual(pack.call_count, 1)


class Sm120GroupedOutputWeightsTest(unittest.TestCase):
    def test_compact_scale_preparation_needs_no_deepgemm_and_keeps_tp_slice(self):
        from rtp_llm.models_py.modules.dsv4.fp8.sm120_attention_backend import (
            SM120_ATTENTION,
        )

        groups, rank, width = 2, 128, 512
        weight = torch.ones(2 * groups * rank, width, dtype=torch.float8_e4m3fn)
        raw = torch.arange(4 * width // 128).reshape(4, width // 128).float().exp2()
        for dtype in (torch.float32, torch.float8_e8m0fnu):
            with self.subTest(dtype=dtype), patch.dict(
                sys.modules, {"deep_gemm": None, "deep_gemm.utils.layout": None}
            ):
                shard_w = weight[groups * rank :]
                shard_s = raw.to(dtype)[groups:]
                prepared_w, prepared_s = SM120_ATTENTION.prepare_grouped_output(
                    shard_w, shard_s, groups, rank, width
                )
                self.assertEqual(prepared_w.data_ptr(), shard_w.data_ptr())
                self.assertEqual(prepared_s.shape, (groups, rank // 128, width // 128))
                self.assertEqual(prepared_s.dtype, torch.float32)
                self.assertTrue(prepared_s.is_contiguous())
                torch.testing.assert_close(
                    prepared_s, shard_s.float().view_as(prepared_s), rtol=0, atol=0
                )

    def test_repeated_grouped_output_reuses_prepared_scale_storage(self):
        from rtp_llm.models_py.modules.dsv4.fp8.sm120_attention_backend import (
            SM120_ATTENTION,
        )

        groups, rank, width, rows = 2, 128, 512, 3
        weight = torch.ones(groups * rank, width, dtype=torch.float8_e4m3fn)
        raw = (
            torch.arange(groups * width // 128)
            .reshape(groups, width // 128)
            .float()
            .exp2()
            .to(torch.float8_e8m0fnu)
        )
        prepared_w, prepared_s = SM120_ATTENTION.prepare_grouped_output(
            weight, raw, groups, rank, width
        )
        activation = torch.ones(rows, groups, width, dtype=torch.float8_e4m3fn)
        scales = torch.full((rows, groups, 1), 0x7F7F7F7F, dtype=torch.int32)
        calls = []

        def gemm(a, b, a_scale, b_scale, **kwargs):
            group = len(calls) % groups
            calls.append(b_scale.data_ptr())
            self.assertEqual(b_scale.data_ptr(), prepared_s[group].data_ptr())
            torch.testing.assert_close(
                b_scale, raw[group].float().view_as(b_scale), rtol=0, atol=0
            )
            self.assertEqual(a.shape, (4, width))
            self.assertEqual(kwargs["scale_granularity_mnk"], (1, 128, 128))
            self.assertEqual(kwargs["scale_major_mode"], "K")
            return torch.full((4, rank), 7, dtype=kwargs["out_dtype"])

        with patch.dict(
            sys.modules,
            {
                "flashinfer.gemm": types.SimpleNamespace(gemm_fp8_nt_groupwise=gemm),
                "deep_gemm": None,
            },
        ):
            for _ in range(2):
                result = SM120_ATTENTION.grouped_output(
                    activation, scales, prepared_w, prepared_s, rank
                )
                self.assertEqual(result.shape, (rows, groups, rank))
                self.assertTrue(torch.all(result == 7))
        self.assertEqual(calls[:groups], calls[groups:])

    def test_grouped_output_rejects_scales_prepared_for_another_backend(self):
        from rtp_llm.models_py.modules.dsv4.fp8.sm120_attention_backend import (
            SM120_ATTENTION,
        )

        groups, rank, width, rows = 2, 128, 512, 3
        weight = torch.ones(groups, rank, width, dtype=torch.float8_e4m3fn)
        activation = torch.ones(rows, groups, width, dtype=torch.float8_e4m3fn)
        scales = torch.full((rows, groups, 1), 0x7F7F7F7F, dtype=torch.int32)
        kernel = Mock()
        wrong = {
            "deepgemm_packed": torch.zeros(
                groups, rank, width // 512, dtype=torch.int32
            ),
            "fp32_wrong_shape": torch.ones(groups, rank, width // 128),
        }
        with patch.dict(
            sys.modules,
            {"flashinfer.gemm": types.SimpleNamespace(gemm_fp8_nt_groupwise=kernel)},
        ):
            for name, weight_scales in wrong.items():
                with self.subTest(layout=name), self.assertRaisesRegex(
                    ValueError, "compact FP32 weight scales"
                ):
                    SM120_ATTENTION.grouped_output(
                        activation, scales, weight, weight_scales, rank
                    )
        kernel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
