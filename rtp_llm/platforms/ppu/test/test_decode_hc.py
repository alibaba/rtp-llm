"""Exercise the selected HC model interface and its in-place Graph lifecycle."""

import unittest

import torch
from rtp_llm.platforms.ppu.models.dsv4.ppu_decode_provider import PpuDecodeProvider


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "ZW-M890P",
    "requires a PPU M890P",
)
class DecodeHCTest(unittest.TestCase):
    def unit(self, *, zero=False):
        torch.manual_seed(890409)
        fn = torch.randn((24, 16384), device="cuda", dtype=torch.float32) / 128
        if zero:
            fn.zero_()
        return PpuDecodeProvider({}).build_hc_unit(
            fn,
            torch.zeros(24, device="cuda"),
            torch.ones(3, device="cuda"),
            dim=4096,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            norm_eps=1e-6,
            hc_eps=1e-6,
            tp_size=1,
            tp_rank=0,
        )

    def test_analytic_pre_and_inplace_post(self):
        unit = self.unit(zero=True)
        for shape in ((3, 4, 4096), (3, 1, 4, 4096)):
            residual = torch.ones(shape, device="cuda", dtype=torch.bfloat16)
            y, post, comb = unit.pre(residual)
            self.assertTrue(torch.equal(y, torch.full_like(y, 2)))
            self.assertTrue(torch.equal(post, torch.ones_like(post)))
            torch.testing.assert_close(
                comb, torch.full_like(comb, 0.25), rtol=0, atol=1e-6
            )
            result = unit.post(y, residual, post, comb)
            self.assertEqual(result.data_ptr(), residual.data_ptr())
            self.assertTrue(torch.equal(result, torch.full_like(result, 3)))

    @torch.inference_mode()
    def test_graph_replay_matches_eager_for_changing_inputs(self):
        unit = self.unit()
        for batch in (1, 3, 8, 32, 64, 128):
            with self.subTest(batch=batch):
                inputs = torch.randn(
                    (batch, 1, 4, 4096), device="cuda", dtype=torch.bfloat16
                )
                residual = torch.empty_like(inputs)

                def run():
                    residual.copy_(inputs)
                    y, post, comb = unit.pre(residual)
                    out = unit.post(y, residual, post, comb)
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
                for _ in range(5):
                    inputs.normal_()
                    expected = tuple(t.clone() for t in run())
                    # Poison call-owned output storage before replay. Captured
                    # scratch/reductions must fully overwrite previous values.
                    for tensor in outputs:
                        tensor.fill_(float("nan"))
                    graph.replay()
                    for actual, reference in zip(outputs, expected):
                        self.assertTrue(bool(torch.isfinite(actual).all()))
                        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_empty_batch_and_instance_capture_policy(self):
        unit = self.unit()
        residual = torch.empty((0, 1, 4, 4096), device="cuda", dtype=torch.bfloat16)
        y, post, comb = unit.pre(residual)
        self.assertEqual(y.shape, (0, 1, 4096))
        self.assertEqual(post.shape, (0, 1, 4, 1))
        self.assertEqual(comb.shape, (0, 1, 4, 4))
        self.assertEqual(unit.post(y, residual, post, comb).shape, residual.shape)
        from rtp_llm.platforms.ppu.models.dsv4.ppu_hc import PpuHCUnit

        with self.assertRaisesRegex(ValueError, "deterministic prenorm"):
            PpuHCUnit(
                unit.fn,
                unit.base,
                unit.scale,
                dim=4096,
                hc_mult=4,
                hc_sinkhorn_iters=20,
                norm_eps=1e-6,
                hc_eps=1e-6,
                options={"DSV4_MHC_PRE_GEMM_BACKEND": "deepgemm"},
                allow_graph=True,
            )


if __name__ == "__main__":
    unittest.main()
