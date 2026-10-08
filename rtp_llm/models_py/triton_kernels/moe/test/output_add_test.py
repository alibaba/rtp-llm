"""Manual GPU correctness for shared MoE output addition."""

import unittest
from functools import partial

import torch

from rtp_llm.models_py.triton_kernels.moe import output_add

assert_exact = partial(torch.testing.assert_close, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class OutputAddTest(unittest.TestCase):
    def test_residual_two_bf16_roundings_preserves_input_and_replays(self):
        tokens, width = 33, 7168
        routed = torch.randn((tokens, width), device="cuda", dtype=torch.bfloat16) * 64
        shared = torch.randn_like(routed)
        residual = -routed
        routed[0, :4] = torch.tensor(
            [256.0, -256.0, 1e30, -1e30],
            device="cuda",
            dtype=torch.bfloat16,
        )
        residual[0, :4] = -routed[0, :4]
        shared[0, :2] = 1
        residual_before = residual.clone()

        # A default MoE residual add must preserve both BF16 roundings in one
        # GPU launch; silently using two torch additions regresses decode.
        output_add.add_moe_output(routed, shared, residual)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            acc_events=True,
        ) as profile:
            output = output_add.add_moe_output(routed, shared, residual)
            torch.cuda.synchronize()
        kernels = [
            event
            for event in profile.events()
            if event.device_type == torch.autograd.DeviceType.CUDA
        ]
        self.assertEqual(len(kernels), 1)
        assert_exact(output, (routed + shared) + residual)
        self.assertEqual(output[0, :2].tolist(), [0.0, 1.0])
        assert_exact(residual, residual_before)
        assert_exact(output_add.add_moe_output(routed, shared), routed + shared)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = output_add.add_moe_output(routed, shared, residual)
        routed.add_(2)
        shared.sub_(1)
        residual.add_(0.5)
        residual_before = residual.clone()
        graph.replay()
        assert_exact(output, (routed + shared) + residual)
        assert_exact(residual, residual_before)


if __name__ == "__main__":
    unittest.main()
