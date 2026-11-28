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
        packed = torch.zeros(
            groups, width // 512, rank, dtype=torch.int32
        ).permute(0, 2, 1)
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
    def test_sm120_backend_prepares_compact_and_prepacked_int32_for_deepgemm(self):
        from rtp_llm.models_py.modules.dsv4.fp8.sm120_attention_backend import (
            SM120_ATTENTION,
        )

        self.assertTrue(SM120_ATTENTION.uses_deepgemm_einsum)
        self.assertTrue(SM120_ATTENTION.supports_shared_qkv_quant)

        groups, rank = 2, 128
        for width in (512, 1024):
            packed_k = width // 512
            with self.subTest(width=width, mode="compact"):
                weight = torch.ones(2 * groups * rank, width, dtype=torch.float8_e4m3fn)
                raw = (
                    torch.arange(4 * (width // 128))
                    .reshape(4, width // 128)
                    .float()
                    .exp2()
                )
                packed = torch.zeros(
                    groups, packed_k, rank, dtype=torch.int32
                ).permute(0, 2, 1)
                pack = Mock(return_value=packed)
                layout = types.SimpleNamespace(
                    get_mn_major_tma_aligned_packed_ue8m0_tensor=pack
                )
                shard_w = weight[groups * rank :]
                shard_s = raw[groups:]
                with patch.dict(sys.modules, {"deep_gemm.utils.layout": layout}):
                    prepared_w, prepared_s = SM120_ATTENTION.prepare_grouped_output(
                        shard_w, shard_s, groups, rank, width
                    )
                self.assertEqual(prepared_w.shape, (groups, rank, width))
                self.assertEqual(prepared_w.data_ptr(), shard_w.data_ptr())
                self.assertIs(prepared_s, packed)
                expected = shard_s.view(groups, 1, width // 128).repeat_interleave(
                    128, dim=1
                )
                torch.testing.assert_close(
                    pack.call_args.args[0], expected, rtol=0, atol=0
                )

            with self.subTest(width=width, mode="prepacked_int32"):
                weight = torch.ones(groups * rank, width, dtype=torch.float8_e4m3fn)
                int32_scale = torch.arange(
                    groups * rank * packed_k, dtype=torch.int32
                ).reshape(groups * rank, packed_k)
                layout = types.SimpleNamespace(
                    get_mn_major_tma_aligned_packed_ue8m0_tensor=Mock()
                )
                with patch.dict(sys.modules, {"deep_gemm.utils.layout": layout}):
                    prepared_w, prepared_s = SM120_ATTENTION.prepare_grouped_output(
                        weight, int32_scale, groups, rank, width
                    )
                self.assertEqual(prepared_w.shape, (groups, rank, width))
                self.assertEqual(tuple(prepared_s.shape), (groups, rank, packed_k))
                self.assertEqual(prepared_s.dtype, torch.int32)
                self.assertEqual(prepared_s.stride(1), 1)
                self.assertGreaterEqual(prepared_s.stride(2), rank)
                self.assertEqual((prepared_s.stride(2) * 4) % 16, 0)
                self.assertGreaterEqual(
                    prepared_s.stride(0), packed_k * prepared_s.stride(2)
                )
                torch.testing.assert_close(
                    prepared_s, int32_scale.view(groups, rank, packed_k)
                )

    def test_repeated_grouped_output_calls_deepgemm_fp8_einsum_with_reused_scales(self):
        from rtp_llm.models_py.modules.dsv4.fp8.sm120_attention_backend import (
            SM120_ATTENTION,
        )

        groups, rank, rows = 2, 128, 3
        for width in (512, 1024):
            packed_k = width // 512
            with self.subTest(width=width):
                weight = torch.ones(groups * rank, width, dtype=torch.float8_e4m3fn)
                raw_int32 = torch.full(
                    (groups * rank, packed_k), 0x7F7F7F7F, dtype=torch.int32
                )
                layout = types.SimpleNamespace(
                    get_mn_major_tma_aligned_packed_ue8m0_tensor=Mock()
                )
                with patch.dict(sys.modules, {"deep_gemm.utils.layout": layout}):
                    prepared_w, prepared_s = SM120_ATTENTION.prepare_grouped_output(
                        weight, raw_int32, groups, rank, width
                    )
                activation = torch.ones(rows, groups, width, dtype=torch.float8_e4m3fn)
                scales = torch.full(
                    (rows, groups, packed_k), 0x7F7F7F7F, dtype=torch.int32
                )
                ptrs = []

                def fake_einsum(eq, a_pair, b_pair, out, *, recipe):
                    self.assertEqual(eq, "bhr,hdr->bhd")
                    self.assertEqual(recipe, (1, 1, 128))
                    ptrs.append(b_pair[1].data_ptr())
                    out.fill_(7)

                with patch.dict(
                    sys.modules,
                    {"deep_gemm": types.SimpleNamespace(fp8_einsum=fake_einsum)},
                ):
                    for _ in range(2):
                        result = SM120_ATTENTION.grouped_output(
                            activation, scales, prepared_w, prepared_s, rank
                        )
                        self.assertEqual(result.shape, (rows, groups, rank))
                        self.assertTrue(torch.all(result == 7))
                self.assertEqual(len(ptrs), 2)
                self.assertEqual(ptrs[0], ptrs[1])

    def test_grouped_output_rejects_invalid_or_aliasing_weight_scales(self):
        from rtp_llm.models_py.modules.dsv4.fp8.sm120_attention_backend import (
            SM120_ATTENTION,
        )

        groups, rank, width, rows = 2, 128, 1024, 3
        packed_k = width // 512
        weight = torch.ones(groups, rank, width, dtype=torch.float8_e4m3fn)
        activation = torch.ones(rows, groups, width, dtype=torch.float8_e4m3fn)
        scales = torch.full((rows, groups, packed_k), 0x7F7F7F7F, dtype=torch.int32)
        kernel = Mock()
        bad_cases = {
            "non_int32": torch.ones(groups, rank, width // 128, dtype=torch.float32),
            "wrong_shape": torch.zeros(groups, rank, 1, dtype=torch.int32),
            "k_major": torch.zeros(groups, rank, packed_k, dtype=torch.int32),
            "unaligned_inner": torch.empty(
                groups * packed_k * 129, dtype=torch.int32
            ).as_strided((groups, rank, packed_k), (packed_k * 129, 1, 129)),
            "aliasing_cross_group": torch.empty(
                packed_k * 128, dtype=torch.int32
            ).as_strided((groups, rank, packed_k), (0, 1, 128)),
        }
        unaligned_1col = torch.empty(
            groups * 129, dtype=torch.int32
        ).as_strided((groups, rank, 1), (129, 1, 129))
        unaligned_2col_cross = torch.empty(
            groups * 257, dtype=torch.int32
        ).as_strided((groups, rank, packed_k), (257, 1, 128))
        with patch.dict(
            sys.modules, {"deep_gemm": types.SimpleNamespace(fp8_einsum=kernel)}
        ):
            for name, bad_scales in bad_cases.items():
                with self.subTest(case=name), self.assertRaises(ValueError):
                    SM120_ATTENTION.grouped_output(
                        activation, scales, weight, bad_scales, rank
                    )
            with self.subTest(case="unaligned_1col_pitch"), self.assertRaises(ValueError):
                SM120_ATTENTION.grouped_output(
                    activation[:, :, :512], scales[:, :, :1], weight[:, :, :512], unaligned_1col, rank
                )
            with self.subTest(case="unaligned_2col_cross"), self.assertRaises(ValueError):
                SM120_ATTENTION.grouped_output(
                    activation, scales, weight, unaligned_2col_cross, rank
                )
        kernel.assert_not_called()

        # Portable fallback runs cleanly on arch-neutral backends when deep_gemm.fp8_einsum
        # is absent (CUDA 12.9), while SM120 continues rejecting non-int32 scales.
        from rtp_llm.models_py.modules.dsv4.fp8.attention_backend import (
            SparseAttentionBackend,
        )

        with patch.dict(sys.modules, {"deep_gemm": types.SimpleNamespace()}):
            out = SparseAttentionBackend().grouped_output(
                activation[:, :, :512],
                torch.ones(rows, groups, 4, dtype=torch.float32),
                weight[:, :, :512],
                torch.ones(groups, rank, 4, dtype=torch.float32),
                rank,
            )
            self.assertEqual(out.shape, (rows, groups, rank))

    def test_portable_fallback_unpacks_all_four_ue8m0_exponents_from_int32(self):
        from rtp_llm.models_py.modules.dsv4.fp8.attention_backend import (
            SparseAttentionBackend,
        )

        groups, rank, rows = 2, 128, 3
        for width in (512, 1024):
            k_groups = width // 128
            packed_k = width // 512
            with self.subTest(width=width):
                torch.manual_seed(width)
                act_q = (
                    torch.randint(1, 5, (rows, groups, width), dtype=torch.int8)
                    .float()
                    .to(torch.float8_e4m3fn)
                )
                w_q = (
                    torch.randint(1, 5, (groups, rank, width), dtype=torch.int8)
                    .float()
                    .to(torch.float8_e4m3fn)
                )
                # Include exponents both below and >= 128 to exercise all 4 byte lanes.
                act_exps = (125 + torch.arange(k_groups) * 2) % 134
                act_exps = torch.clamp(act_exps, min=124, max=133).expand(
                    rows, groups, k_groups
                )
                w_exps = (124 + (torch.arange(groups * k_groups) % 9)).reshape(
                    groups, 1, k_groups
                )
                w_exps = w_exps.expand(groups, rank, k_groups).contiguous()

                def pack_u8_to_int32(exp_u8: torch.Tensor) -> torch.Tensor:
                    words = torch.zeros(
                        (*exp_u8.shape[:-1], packed_k), dtype=torch.int32
                    )
                    for i in range(k_groups):
                        words[..., i // 4] |= exp_u8[..., i].to(torch.int32) << (
                            8 * (i % 4)
                        )
                    return words

                act_s_int32 = pack_u8_to_int32(act_exps)
                w_s_buf = torch.empty(groups, packed_k, rank, dtype=torch.int32)
                w_s_buf.copy_(pack_u8_to_int32(w_exps).permute(0, 2, 1))
                w_s_int32 = w_s_buf.permute(0, 2, 1)

                act_f_ref = (
                    act_q.float().view(rows, groups, k_groups, 128)
                    * torch.exp2((act_exps - 127).float()).unsqueeze(-1)
                ).view(rows, groups, width)
                w_f_ref = (
                    w_q.float().view(groups, rank, k_groups, 128)
                    * torch.exp2((w_exps - 127).float()).unsqueeze(-1)
                ).view(groups, rank, width)
                expected = torch.einsum("bhr,hdr->bhd", act_f_ref, w_f_ref).to(
                    torch.bfloat16
                )

                with patch.dict(sys.modules, {"deep_gemm": types.SimpleNamespace()}):
                    actual = SparseAttentionBackend().grouped_output(
                        act_q, act_s_int32, w_q, w_s_int32, rank
                    )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
