"""SM120 shared-expert geometry, routed SwiGLU order and graph replay gates."""

import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4.v41_expert import (
    V41SharedExpert,
    pack_group32_scales,
    quantize_swiglu,
)
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import quantize_group32


class V41ExpertHardwareTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available(), "Expert hardware gate requires CUDA")
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))

    @staticmethod
    def weights():
        generator = torch.Generator(device="cuda").manual_seed(413)
        return {
            "w13_w": torch.randn(
                4608, 5120, device="cuda", dtype=torch.bfloat16, generator=generator
            ).to(torch.float8_e4m3fn),
            "w13_s": torch.randint(
                117,
                121,
                (144, 160),
                device="cuda",
                dtype=torch.uint8,
                generator=generator,
            ),
            "w2_w": torch.randn(
                5120, 2304, device="cuda", dtype=torch.bfloat16, generator=generator
            ).to(torch.float8_e4m3fn),
            "w2_s": torch.randint(
                117,
                121,
                (160, 72),
                device="cuda",
                dtype=torch.uint8,
                generator=generator,
            ),
        }

    def assert_shared(self, reference, x, valid, route, actual):
        expected = reference(
            x.cpu(), valid.cpu(), route_weights=None if route is None else route.cpu()
        )
        actual = actual.cpu()
        self.assertTrue(bool(torch.isfinite(actual).all()))
        # Two quantized GEMMs can straddle intermediate BF16/FP8 rounding boundaries.
        torch.testing.assert_close(actual, expected, rtol=2.5e-2, atol=1.5e-2)
        active = valid.cpu() if route is None else valid.cpu() & (route.cpu() != 0)
        self.assertTrue(bool((actual[~active].view(torch.uint8) == 0).all()))

    def test_routed_swiglu_rounding_bytes_and_packed_scale_layout(self):
        gu = (torch.randn(17, 4608, device="cuda") * 12).bfloat16()
        gate, up = gu.chunk(2, -1)
        valid = torch.arange(17, device="cuda") % 3 != 0
        route = torch.linspace(0, 1.5, 17, device="cuda", dtype=torch.float32)
        gu[~valid] = float("nan")
        g = torch.where(valid[:, None], gate.float(), 0).clamp(max=10)
        u = torch.where(valid[:, None], up.float(), 0).clamp(-10, 10)
        intermediate = (route[:, None] * (F.silu(g) * u)).bfloat16()
        expected, scale = quantize_group32(intermediate.cpu(), backend="eager")
        active = valid & (route != 0)
        scale[~active.cpu()] = 0
        values, actual_scale = quantize_swiglu(gate, up, valid, route_weights=route)
        torch.testing.assert_close(
            values.view(torch.uint8).cpu(), expected.view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(actual_scale.cpu(), scale, rtol=0, atol=0)
        for column_major in (False, True):
            packed = pack_group32_scales(actual_scale, column_major=column_major)
            restored = packed.contiguous().view(torch.uint8)
            torch.testing.assert_close(restored.cpu(), scale, rtol=0, atol=0)

    def test_shared_geometry_decode_prefill_and_graph_replay(self):
        weights = self.weights()
        op = V41SharedExpert(weights)
        reference = V41SharedExpert(
            {name: value.cpu() for name, value in weights.items()}, backend="eager"
        )
        for rows in (1, 8, 33):
            x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
            valid = torch.ones(rows, device="cuda", dtype=torch.bool)
            self.assert_shared(reference, x, valid, None, op(x, valid))
        x = torch.randn(17, 5120, device="cuda", dtype=torch.bfloat16)
        valid = torch.ones(17, device="cuda", dtype=torch.bool)
        route = torch.full((17,), 0.37, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                op(x, valid, route_weights=route, out=out)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            op(x, valid, route_weights=route, out=out)
        for factor in (0, 0.37, 1.5):
            route.fill_(factor)
            valid.copy_(torch.arange(17, device="cuda") % 3 != 0)
            x[~valid] = float("nan")
            graph.replay()
            self.assert_shared(reference, x, valid, route, out)


if __name__ == "__main__":
    unittest.main()
