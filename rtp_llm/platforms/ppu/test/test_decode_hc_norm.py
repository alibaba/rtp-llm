"""HC-normalization boundary, rounding semantics and dynamic Graph storage."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.hc.base import HCUnitBase
from rtp_llm.platforms.ppu.models.dsv4.ppu_decode_provider import PpuDecodeProvider
from rtp_llm.platforms.ppu.models.dsv4.manifest import DECODE_EXECUTION_OPTIONS


class HCNormDefaultTest(unittest.TestCase):
    def test_default_preserves_pre_then_tp_normalization(self):
        unit = HCUnitBase(
            None,
            None,
            None,
            dim=4,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            norm_eps=1e-6,
            hc_eps=1e-6,
        )
        x = torch.empty((3, 4, 4))
        readout = torch.empty((3, 1, 4))
        normalized = torch.randn((3, 4))
        post, comb = object(), object()
        norm = object()
        with patch.object(
            unit, "pre", return_value=(readout, post, comb)
        ) as pre, patch(
            "rtp_llm.models_py.modules.dsv4.tp_norm.tp_rms_norm",
            return_value=normalized,
        ) as apply_norm:
            actual = unit.pre_norm(x, norm, tp_size=8, tp_rank=3, dbg_tag="check")
        pre.assert_called_once_with(x, dbg_tag="check")
        self.assertEqual(apply_norm.call_count, 1)
        self.assertIs(apply_norm.call_args.args[0], norm)
        self.assertEqual(apply_norm.call_args.args[1].shape, (3, 4))
        self.assertEqual(apply_norm.call_args.args[1].data_ptr(), readout.data_ptr())
        self.assertEqual(apply_norm.call_args.kwargs, {"tp_size": 8, "tp_rank": 3})
        self.assertEqual(actual[0].shape, readout.shape)
        self.assertEqual(actual[0].data_ptr(), normalized.data_ptr())
        self.assertIs(actual[1], post)
        self.assertIs(actual[2], comb)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "ZW-M890P",
    "requires a PPU M890P",
)
class HCNormGraphTest(unittest.TestCase):
    def unit(self, *, reduction="fused", norm="fused", zero=False):
        torch.manual_seed(890409)
        fn = torch.randn((24, 16384), device="cuda", dtype=torch.float32) / 128
        if zero:
            fn.zero_()
        from rtp_llm.platforms.ppu.models.dsv4.ppu_hc import PpuHCUnit

        factory = (
            PpuDecodeProvider(DECODE_EXECUTION_OPTIONS).build_hc_unit
            if reduction == "fused" and norm == "fused"
            else lambda *args, **kwargs: PpuHCUnit(
                *args,
                options=DECODE_EXECUTION_OPTIONS,
                allow_graph=True,
                fuse_prenorm=reduction == "fused",
                fuse_norm=norm == "fused",
                **kwargs
            )
        )
        return factory(
            fn,
            torch.zeros(24, device="cuda"),
            torch.ones(3, device="cuda"),
            dim=4096,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            norm_eps=1e-6,
            hc_eps=0.0 if zero else 1e-6,
            tp_size=1,
            tp_rank=0,
        )

    def norm(self):
        from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm

        return RMSNorm(
            torch.linspace(0.25, 1.25, 4096, device="cuda", dtype=torch.bfloat16),
            1e-6,
        )

    @torch.inference_mode()
    def test_analytic_rounding_and_empty(self):
        unit, norm = self.unit(zero=True), self.norm()
        # Zero HC projections make every PRE coefficient exactly 1/2. Use an
        # independent FP64 readout to expose the denominator's rounding point.
        x = torch.randint(-31, 32, (3, 4, 4096), device="cuda").to(torch.bfloat16)
        x /= 8
        readout = x.double().sum(-2) * 0.5
        expected = (
            readout.to(torch.bfloat16).double()
            * torch.rsqrt(readout.square().mean(-1, keepdim=True) + 1e-6)
            * norm.weight.double()
        ).to(torch.bfloat16)
        y, post, comb = unit.pre_norm(x, norm, tp_size=1, tp_rank=0)
        # BF16 output admits at most one representable rounding step.
        torch.testing.assert_close(y, expected, rtol=1 / 128, atol=0)
        self.assertTrue(torch.equal(post, torch.ones_like(post)))
        self.assertTrue(torch.equal(comb, torch.full_like(comb, 0.25)))
        for shape in ((0, 4, 4096), (0, 1, 4, 4096)):
            empty = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
            out = unit.pre_norm(empty, norm, tp_size=1, tp_rank=0)
            self.assertEqual(out[0].shape, (*shape[:-2], 4096))
        with self.assertRaisesRegex(ValueError, "BF16 TP1"):
            unit.pre_norm(
                x.transpose(0, 1).contiguous().transpose(0, 1),
                norm,
                tp_size=1,
                tp_rank=0,
            )
        with self.assertRaisesRegex(ValueError, "BF16 TP1"):
            unit.pre_norm(x, norm, tp_size=2, tp_rank=0)

    @torch.inference_mode()
    def test_decode_block_uses_fused_boundary(self):
        from types import SimpleNamespace

        from rtp_llm.models_py.modules.dsv4.block import Block

        block = Block.__new__(Block)
        torch.nn.Module.__init__(block)
        block.layer_id, block.tp_size, block.tp_rank = 0, 1, 0
        block.attn_hc, block.ffn_hc = self.unit(), self.unit()
        block.attn_norm, block.ffn_norm = self.norm(), self.norm()
        block.attn = SimpleNamespace(forward_decode=lambda x, *args, **kwargs: x)
        block.ffn = lambda x, *args, **kwargs: x
        inputs = torch.randn((3, 1, 4, 4096), device="cuda", dtype=torch.bfloat16)
        residual = torch.empty_like(inputs)
        ids = torch.zeros((3, 1), device="cuda", dtype=torch.int64)

        def run():
            residual.copy_(inputs)
            return block.forward_decode(residual, None, ids)

        # Calling the old separate norm means the actual Decode entry point
        # bypassed the platform's fused boundary.
        with patch(
            "rtp_llm.models_py.modules.dsv4._record_tensor.should_record_layer",
            return_value=False,
        ), patch.object(
            block.attn_norm, "forward", side_effect=AssertionError("separate norm")
        ), patch.object(
            block.ffn_norm, "forward", side_effect=AssertionError("separate norm")
        ):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = run()
            torch.cuda.current_stream().wait_stream(stream)
            for _ in range(5):
                inputs.normal_()
                expected = run().clone()
                output.fill_(float("nan"))
                graph.replay()
                self.assertTrue(torch.equal(output, expected))

    @torch.inference_mode()
    def test_dynamic_graph_preserves_mixers_and_inplace_post(self):
        for reduction in ("torch", "fused"):
            unit, norm = self.unit(reduction=reduction), self.norm()
            for batch in (1, 3, 8, 32, 128):
                with self.subTest(reduction=reduction, batch=batch):
                    inputs = torch.randn(
                        (batch, 1, 4, 4096), device="cuda", dtype=torch.bfloat16
                    )
                    residual = torch.empty_like(inputs)

                    def run():
                        residual.copy_(inputs)
                        y, post, comb = unit.pre_norm(
                            residual, norm, tp_size=1, tp_rank=0
                        )
                        out = unit.post(y, residual, post, comb)
                        self.assertEqual(out.data_ptr(), residual.data_ptr())
                        return y, post, comb, out

                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            run()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        outputs = run()
                    torch.cuda.current_stream().wait_stream(stream)
                    for _ in range(3):
                        inputs.normal_()
                        expected = tuple(t.clone() for t in run())
                        _, plain_post, plain_comb = unit.pre(inputs)
                        torch.testing.assert_close(
                            expected[1], plain_post, rtol=0, atol=0
                        )
                        torch.testing.assert_close(
                            expected[2], plain_comb, rtol=0, atol=0
                        )
                        for tensor in outputs:
                            tensor.fill_(float("nan"))
                        graph.replay()
                        for actual, reference in zip(outputs, expected):
                            self.assertTrue(bool(torch.isfinite(actual).all()))
                            torch.testing.assert_close(
                                actual, reference, rtol=0, atol=0
                            )


if __name__ == "__main__":
    unittest.main()
