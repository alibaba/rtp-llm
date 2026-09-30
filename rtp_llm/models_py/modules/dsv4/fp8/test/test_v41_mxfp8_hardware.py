"""Real SM120 MXFP8 provider/rounding/graph gates; missing hardware fails."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import V41MXFP8Linear, quantize_group32


class V41MXFP8HardwareTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "MXFP8 hardware gate requires CUDA")
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", previous)

    def assert_quantization(self, x):
        expected, scales = quantize_group32(x.cpu(), backend="eager")
        actual, actual_scales = quantize_group32(x)
        torch.testing.assert_close(
            actual.view(torch.uint8).cpu(), expected.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(actual_scales.cpu(), scales, rtol=0, atol=0)
        return actual, actual_scales

    def assert_projection(self, x, weight, scales, output):
        values, a_scales = self.assert_quantization(x)
        m, k = values.shape
        expected = torch.zeros(m, weight.shape[0], device=x.device, dtype=torch.float32)
        for group in range(k // 32):
            a = values[:, group * 32 : (group + 1) * 32].float()
            b = weight[:, group * 32 : (group + 1) * 32].float()
            partial = a @ b.T
            partial *= torch.exp2(a_scales[:, group].float() - 127)[:, None]
            partial *= torch.exp2(scales[:, group].float() - 127).repeat_interleave(32)[
                None, :
            ]
            expected += partial
        self.assertTrue(bool(torch.isfinite(output).all()))
        # BF16 rounding and different reduction order near cancellation.
        torch.testing.assert_close(
            output.float(), expected.to(torch.bfloat16).float(), rtol=8e-3, atol=2e-3
        )

    @staticmethod
    def inputs(m, n, k):
        generator = torch.Generator(device="cuda").manual_seed(m + n + k)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=generator)
        weight = torch.randn(
            n, k, device="cuda", dtype=torch.bfloat16, generator=generator
        ).to(torch.float8_e4m3fn)
        scales = torch.randint(
            116,
            121,
            (n // 32, k // 32),
            device="cuda",
            dtype=torch.uint8,
            generator=generator,
        )
        return x, weight, scales

    def test_quantization_floor_midpoints_scale_boundaries_and_tail(self):
        levels = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        midpoints = (levels[:-1] + levels[1:]) / 2
        x = torch.zeros(2 * midpoints.numel() + 9, 32)
        x[:252, -1] = 448
        x[:126, 0] = midpoints
        x[126:252, 0] = -midpoints
        x[252:257, 0] = torch.tensor([-0.0, 1e-5, -1e-4, 448 - 2**-15, 448 + 2**-15])
        x[257:, 0] = torch.tensor([0, 448, 2**-126, 448 * 2.0**100])
        self.assert_quantization(x.cuda())
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            values = torch.randn(
                17, 160, generator=torch.Generator().manual_seed(32)
            ).to(dtype)
            self.assert_quantization(values.cuda())

    def test_decode_prefill_and_non_aligned_scale_padding(self):
        for m in (1, 2, 4, 8, 17, 129):
            x, weight, scales = self.inputs(m, 160, 160)
            op = V41MXFP8Linear(weight, scales)
            self.assert_projection(x, weight, scales, op(x))

    def test_checkpoint_dense_and_engram_wkv_geometry(self):
        for n, k in ((1280, 5120), (25600, 6144)):
            for m in (1, 8, 33):
                x, weight, scales = self.inputs(m, n, k)
                op = V41MXFP8Linear(weight, scales)
                self.assertEqual(op.weight.data_ptr(), weight.data_ptr())
                self.assert_projection(x, weight, scales, op(x))

    def test_graph_replay_requantizes_changed_values_and_scales(self):
        x, weight, scales = self.inputs(17, 160, 160)
        op = V41MXFP8Linear(weight, scales)
        output = torch.empty(17, 160, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                op(x, out=output)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(x, out=output)
        initial = x.clone()
        for factor in (0.00001, -8, 2):
            x.copy_(initial * factor)
            graph.replay()
            self.assert_projection(x, weight, scales, output)


if __name__ == "__main__":
    unittest.main()
