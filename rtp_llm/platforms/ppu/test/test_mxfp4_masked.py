"""PPU GPU regression for full expert capacity, graph counts and LL arenas.

The synthetic buffer tests cover adapter ownership, not real EP communication.
SGLang numerical/performance qualification uses separately frozen model-shape
fixtures; this small arithmetic case runs without SGLang or a checkpoint.
"""

import unittest

import torch

from rtp_llm.platforms.ppu.kernels.ppu_mxfp4_masked import mxfp4_experts_masked
from rtp_llm.platforms.ppu.modules.fused_moe.mxfp4_low_latency import (
    low_latency_mxfp4_moe,
)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "ZW-M890P",
    "requires a PPU M890P",
)
class Mxfp4MaskedTest(unittest.TestCase):
    def setUp(self):
        self.e, self.m, self.d, self.inter = 4, 256, 128, 256

        def pack(rows, columns):
            # Each nibble is exactly 1 and each E8M0 scale is 2**-6.
            data = torch.full(
                (self.e, rows, columns // 2),
                0x22,
                dtype=torch.uint8,
                device="cuda",
            )
            scale = (
                torch.full(
                    (self.e, columns // 64, rows),
                    0x7979,
                    dtype=torch.int16,
                    device="cuda",
                )
                .view(torch.uint16)
                .transpose(-1, -2)
            )
            return data, scale

        self.x = pack(self.m, self.d)
        self.w13 = pack(2 * self.inter, self.d)
        self.w2 = pack(self.d, self.inter)
        self.counts = torch.tensor([0, 1, 129, 256], dtype=torch.int32, device="cuda")
        self.out = torch.empty(
            (self.e, self.m, self.d), dtype=torch.bfloat16, device="cuda"
        )

    def assert_output(self, out):
        valid = torch.arange(self.m, device="cuda")[None, :] < self.counts[:, None]
        # w13 = 128 * 2**-12 = 2**-5. SwiGLU then quantizes to
        # 4 * 2**-13 = 2**-11; w2 sums 256 products with 2**-6.
        self.assertTrue(torch.equal(out[valid], torch.full_like(out[valid], 2**-9)))
        self.assertTrue(torch.equal(out[~valid], torch.full_like(out[~valid], -77)))

    def test_skew_and_dynamic_graph_keep_every_valid_row(self):
        def run():
            return mxfp4_experts_masked(
                self.x,
                self.w13,
                self.w2,
                self.counts,
                expected_m=1,
                out=self.out,
            )

        self.out.fill_(-77)
        run()
        self.assert_output(self.out)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        for counts in ([256, 129, 1, 0], [0, 0, 0, 0], [1, 128, 255, 256]):
            with self.subTest(counts=counts):
                self.counts.copy_(
                    torch.tensor(counts, dtype=torch.int32, device="cuda")
                )
                self.out.fill_(-77)
                graph.replay()
                self.assert_output(self.out)
                self.assertEqual(self.counts.tolist(), counts)

    def test_ll_combine_uses_full_slot_with_direct_or_aliased_arena(self):
        case = self

        class SyntheticBuffer:
            def __init__(self, alias):
                self.output = torch.empty_like(case.out)
                if alias:
                    self.payload = (
                        self.output.view(torch.uint8)
                        .flatten()[: case.x[0].numel()]
                        .view_as(case.x[0])
                    )
                else:
                    self.payload = torch.empty_like(case.x[0])
                self.handle = object()

            def low_latency_dispatch(self, **kwargs):
                self.dispatch = kwargs
                self.payload.copy_(case.x[0])
                return (self.payload, case.x[1]), case.counts, self.handle, None, None

            def get_next_low_latency_combine_buffer(self, handle):
                case.assertIs(handle, self.handle)
                return self.output

            def low_latency_combine(self, **kwargs):
                self.combine = kwargs
                self.combined = kwargs["x"][2, :2].clone()
                return self.combined, None, None

        for alias in (False, True):
            with self.subTest(alias=alias):
                buffer = SyntheticBuffer(alias)
                x = torch.ones((2, self.d), dtype=torch.bfloat16, device="cuda")
                weights = torch.ones((2, 6), dtype=torch.float32, device="cuda")
                indices = torch.arange(6, device="cuda").expand(2, -1).contiguous()

                def run(output_dtype=torch.float32):
                    return low_latency_mxfp4_moe(
                        buffer,
                        x,
                        weights,
                        indices,
                        self.w13,
                        self.w2,
                        num_experts=8,
                        max_dispatch_tokens=256,
                        expected_m=1,
                        output_dtype=output_dtype,
                    )

                result = run()
                self.assertEqual(result.dtype, torch.float32)
                self.assertTrue(torch.equal(result, torch.full_like(result, 2**-9)))
                native = run(torch.bfloat16)
                self.assertIs(native, buffer.combined)
                self.assertEqual(native.dtype, torch.bfloat16)
                self.assertTrue(torch.equal(native.float(), result))
                valid = (
                    torch.arange(self.m, device="cuda")[None, :] < self.counts[:, None]
                )
                self.assertTrue(
                    torch.equal(
                        buffer.output[valid],
                        torch.full_like(buffer.output[valid], 2**-9),
                    )
                )
                self.assertEqual(buffer.dispatch["topk_idx"].shape, (2, 8))
                self.assertTrue(bool((buffer.dispatch["topk_idx"][:, 6:] == -1).all()))
                self.assertTrue(
                    bool((buffer.combine["topk_weights"][:, 6:] == 0).all())
                )
                self.assertTrue(buffer.combine["zero_copy"])
                self.assertEqual(
                    buffer.combine["x"].data_ptr(), buffer.output.data_ptr()
                )
                self.assertEqual(self.counts.tolist(), [0, 1, 129, 256])

    def test_shape_and_scale_layout_rejections_precede_kernel_execution(self):
        with self.assertRaises(ValueError):
            mxfp4_experts_masked(
                self.x,
                self.w13,
                self.w2,
                self.counts,
                expected_m=1,
                out=self.out[:, :128],
            )
        with self.assertRaises(ValueError):
            mxfp4_experts_masked(
                (self.x[0], self.x[1].contiguous()),
                self.w13,
                self.w2,
                self.counts,
                expected_m=1,
            )

    def test_dsv4_weight_setup_preserves_ep_slice_and_prepares_scales(self):
        from rtp_llm.platforms.ppu.models.dsv4.ppu_moe_config import (
            PpuMoeConfig as MoeCfg,
        )
        from rtp_llm.platforms.ppu.models.dsv4.ppu_deepep_fp4 import (
            PpuDeepEPFP4Strategy,
        )
        from rtp_llm.utils.model_weight import W

        cfg = MoeCfg(
            layer_id=0,
            dim=self.d,
            moe_inter_dim=self.inter,
            n_routed_experts=8,
            n_activated_experts=6,
            swiglu_limit=0.0,
            ep_size=2,
            ep_rank=1,
            n_local_experts=self.e,
            local_expert_start=4,
            local_expert_end=8,
            max_tokens_per_rank=128,
        )
        strategy = PpuDeepEPFP4Strategy(cfg)

        def raw_scale(rows, k):
            return torch.full(
                (self.e, rows, k // 32), 121, dtype=torch.uint8, device="cuda"
            ).view(torch.float8_e8m0fnu)

        weights = {
            W.v4_routed_w1_w: self.w13[0][:, : self.inter].view(torch.int8),
            W.v4_routed_w3_w: self.w13[0][:, self.inter :],
            W.v4_routed_w2_w: self.w2[0],
            W.v4_routed_w1_s: raw_scale(self.inter, self.d),
            W.v4_routed_w3_s: raw_scale(self.inter, self.d),
            W.v4_routed_w2_s: raw_scale(self.d, self.inter),
            "unrelated_weight": None,
        }
        strategy.setup_weights(weights)
        self.assertEqual(weights, {"unrelated_weight": None})
        self.assertTrue(torch.equal(strategy._w13, self.w13[0]))
        self.assertTrue(torch.equal(strategy._w2, self.w2[0]))
        self.assertTrue(torch.equal(strategy._s13, self.w13[1]))
        self.assertTrue(torch.equal(strategy._s2, self.w2[1]))
        self.assertTrue(strategy._s13.transpose(-1, -2).is_contiguous())
        self.assertEqual(strategy._expected_m, 192)
        self.assertIsNone(strategy._wrapper)


if __name__ == "__main__":
    unittest.main()
