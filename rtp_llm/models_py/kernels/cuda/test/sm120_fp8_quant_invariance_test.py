import itertools
import unittest

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.utils.arch import is_sm120
from rtp_llm.ops.compute_ops import (
    per_token_group_quant_fp8,
    per_token_group_quant_fp8_v2,
)


class Sm120Fp8QuantInvarianceTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or not is_sm120():
            self.skipTest("requires SM120")

    def test_explicit_v2_across_batch_threshold(self):
        generator = torch.Generator(device="cuda").manual_seed(42)
        layouts = (
            (False, False, False),
            (True, False, False),
            (True, True, False),
            (True, True, True),
        )
        for width, group, layout, dtype in itertools.product(
            (1024, 4096),
            (16, 32, 64, 128),
            layouts,
            (torch.float16, torch.bfloat16),
        ):
            values = torch.randn(
                (11, width), generator=generator, device="cuda", dtype=dtype
            )
            values[0].zero_()
            options = dict(
                group_size=group,
                eps=1e-4,
                column_major_scales=layout[0],
                scale_tma_aligned=layout[1],
                scale_ue8m0=layout[2],
                enable_v2=True,
            )
            threshold = 4 * 1024 * 1024 // width
            expected_q, expected_s = sgl_per_token_group_quant_fp8(values, **options)
            for rows in (1, 17, threshold - 1, threshold, threshold + 1):
                with self.subTest(
                    width=width, group=group, layout=layout, rows=rows, dtype=dtype
                ):
                    order = torch.arange(rows, device="cuda") % len(values)
                    x = values[order].contiguous()
                    actual_q, actual_s = sgl_per_token_group_quant_fp8(x, **options)
                    self.assertTrue(
                        torch.equal(
                            actual_q.view(torch.uint8),
                            expected_q.view(torch.uint8)[order],
                        )
                    )
                    self.assertTrue(torch.equal(actual_s, expected_s[order]))

    def test_explicit_v2_graph_replay(self):
        generator = torch.Generator(device="cuda").manual_seed(43)
        values = torch.randn(
            (17, 4096), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        options = dict(
            group_size=128, column_major_scales=True, eps=1e-4, enable_v2=True
        )
        expected_q, expected_s = sgl_per_token_group_quant_fp8(values, **options)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual_q, actual_s = sgl_per_token_group_quant_fp8(values, **options)
        for _ in range(20):
            graph.replay()
            self.assertTrue(
                torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
            )
            self.assertTrue(torch.equal(actual_s, expected_s))

    def test_partial_packed_ue8m0_words_zero_padded(self):
        # group_size=16 with width=80 gives 5 groups/token -> remainder 1 in the
        # final int32 word, leaving 3 padding bytes that must be zeroed.
        x = torch.randn((7, 80), device="cuda", dtype=torch.bfloat16)
        _, first_s = sgl_per_token_group_quant_fp8(
            x,
            group_size=16,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
            enable_v2=True,
        )
        _, second_s = sgl_per_token_group_quant_fp8(
            x,
            group_size=16,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
            enable_v2=True,
        )
        self.assertTrue(torch.equal(first_s, second_s))
        tail_bytes = (
            first_s[:, -1].contiguous().view(torch.uint8).view(x.shape[0], 4)[:, 1:]
        )
        self.assertTrue(torch.equal(tail_bytes, torch.zeros_like(tail_bytes)))

    def test_v1_rejects_mismatched_output_q_and_fused_silu_matches_reference(self):
        x = torch.randn((4, 1024), device="cuda", dtype=torch.bfloat16)
        half_q = torch.empty((4, 512), device="cuda", dtype=torch.float8_e4m3fn)
        scales = torch.empty((4, 8), device="cuda", dtype=torch.float32)
        with self.assertRaisesRegex(
            RuntimeError, "output_q numel must match input numel"
        ):
            per_token_group_quant_fp8(
                x, half_q, scales, 128, 1e-4, -448.0, 448.0, False
            )

        q_actual, s_actual = sgl_per_token_group_quant_fp8(
            x,
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
            fuse_silu_and_mul=True,
        )
        gate, up = x.float().chunk(2, dim=-1)
        ref = torch.nn.functional.silu(gate) * up
        from rtp_llm.models_py.modules.base.cuda.indexer_op import _unpack_ue8m0_scale

        # Check reconstructed scales on group 0 against quantized reference.
        s0 = _unpack_ue8m0_scale(s_actual[:, :1])
        deq0 = q_actual[:, :128].float() * s0
        rel_err = (deq0 - ref[:, :128]).norm() / ref[:, :128].norm().clamp_min(1e-6)
        self.assertLess(float(rel_err), 0.12)

    def test_masked_m_contract_and_device_bounds_clamp(self):
        x = torch.randn((2, 2, 4096), device="cuda", dtype=torch.bfloat16)
        kwargs = dict(
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
            fuse_silu_and_mul=True,
        )
        ref_q, ref_s = sgl_per_token_group_quant_fp8(
            x.view(4, 4096).contiguous(), **kwargs
        )
        ref_q = ref_q.view(2, 2, 2048)
        ref_s = ref_s.view(2, 2, 4)

        valid_m = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
        q_valid, s_valid = sgl_per_token_group_quant_fp8(x, masked_m=valid_m, **kwargs)
        self.assertTrue(
            torch.equal(
                q_valid[0, :1].view(torch.uint8), ref_q[0, :1].view(torch.uint8)
            )
        )
        self.assertTrue(torch.equal(s_valid[0, :1], ref_s[0, :1]))
        self.assertTrue(
            torch.equal(q_valid[1].view(torch.uint8), ref_q[1].view(torch.uint8))
        )
        self.assertTrue(torch.equal(s_valid[1], ref_s[1]))

        wrong_dtype_m = torch.tensor([1, 1], device="cuda", dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "contiguous int32"):
            sgl_per_token_group_quant_fp8(x, masked_m=wrong_dtype_m, **kwargs)
        with self.assertRaisesRegex(RuntimeError, "contiguous int32"):
            per_token_group_quant_fp8_v2(
                x,
                q_valid,
                s_valid,
                128,
                1e-4,
                -448.0,
                448.0,
                True,
                True,
                wrong_dtype_m,
                False,
            )

        noncontig_m = torch.tensor([1, 9, 2, 9], device="cuda", dtype=torch.int32)[::2]
        self.assertFalse(noncontig_m.is_contiguous())
        with self.assertRaisesRegex(ValueError, "contiguous int32"):
            sgl_per_token_group_quant_fp8(x, masked_m=noncontig_m, **kwargs)
        with self.assertRaisesRegex(RuntimeError, "contiguous int32"):
            per_token_group_quant_fp8_v2(
                x,
                q_valid,
                s_valid,
                128,
                1e-4,
                -448.0,
                448.0,
                True,
                True,
                noncontig_m,
                False,
            )

        # Counts exceeding per-expert capacity (5 > 2) or negative (-3 < 0)
        # must clamp safely without out-of-bounds memory access.
        oob_m = torch.tensor([5, -3], device="cuda", dtype=torch.int32)
        q_oob, s_oob = sgl_per_token_group_quant_fp8(x, masked_m=oob_m, **kwargs)
        torch.cuda.synchronize()
        self.assertTrue(
            torch.equal(q_oob[0].view(torch.uint8), ref_q[0].view(torch.uint8))
        )
        self.assertTrue(torch.equal(s_oob[0], ref_s[0]))


if __name__ == "__main__":
    unittest.main()
