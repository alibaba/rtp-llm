"""Compare the production shared-expert gate against native RTP semantics."""

import unittest

import torch

from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_inputs import (
    _SHARED_GATE_CACHE,
    ensure_shared_gate_capacity,
    shared_expert_sigmoid,
    stage_shared_scales,
)
from rtp_llm.models_py.triton_kernels.common.moe_gating import (
    sigmoid_gate_scale_add_triton,
)


class SharedInputsTest(unittest.TestCase):
    def test_native_sigmoid_exact(self):
        for tokens in (0, 1, 129, 24601, 49202):
            with self.subTest(tokens=tokens):
                logits = torch.linspace(
                    -20, 20, tokens, device="cuda", dtype=torch.bfloat16
                ).view(-1, 1)
                actual = shared_expert_sigmoid(logits)
                if tokens:
                    reference = torch.zeros(
                        (tokens, 1), device="cuda", dtype=torch.float32
                    )
                    sigmoid_gate_scale_add_triton(
                        logits, torch.ones_like(reference), reference
                    )
                    self.assertTrue(torch.equal(actual, reference[:, 0]))
                self.assertEqual(actual.dtype, torch.float32)

    def test_layout_and_reuse(self):
        for block_m in (64, 128, 240):
            for tokens in (1, 129, 24601, 49202):
                with self.subTest(block_m=block_m, tokens=tokens):
                    pad = (block_m + 127) // 128 * 128
                    rows = (65536 + block_m - 1) // block_m * pad
                    # Production destination is MN-major; source may be strided.
                    target = torch.empty_strided(
                        (rows, 32), (1, rows), device="cuda", dtype=torch.int32
                    )
                    target.fill_(-1)
                    source = torch.randint(
                        0, 2**30, (tokens, 64), device="cuda", dtype=torch.int32
                    )[:, ::2]
                    for count in (tokens, min(tokens, 37)):
                        stage_shared_scales(target, source[:count], block_m)
                        reference = torch.zeros_like(target)
                        row = torch.arange(count, device="cuda")
                        local = row % block_m
                        indices = (
                            row // block_m * pad
                            + local // 128 * 128
                            + local % 32 * 4
                            + local % 128 // 32
                        )
                        reference[indices] = source[:count]
                        self.assertTrue(torch.equal(target, reference))

    def test_cuda_graph_requires_preallocated_buffer(self):
        _SHARED_GATE_CACHE.clear()
        logits = torch.randn(8, 1, device="cuda", dtype=torch.bfloat16)
        graph = torch.cuda.CUDAGraph()
        with self.assertRaisesRegex(RuntimeError, "static gate buffer"):
            with torch.cuda.graph(graph):
                shared_expert_sigmoid(logits)

    def test_cuda_graph_reuses_preallocated_buffer(self):
        ensure_shared_gate_capacity("cuda", 256)
        static_logits = torch.randn(32, 1, device="cuda", dtype=torch.bfloat16)
        shared_expert_sigmoid(static_logits)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = shared_expert_sigmoid(static_logits)
        static_logits.copy_(
            torch.linspace(-8, 8, 32, device="cuda", dtype=torch.bfloat16).view(-1, 1)
        )
        graph.replay()
        reference = torch.zeros((32, 1), device="cuda", dtype=torch.float32)
        sigmoid_gate_scale_add_triton(
            static_logits, torch.ones_like(reference), reference
        )
        self.assertTrue(torch.equal(actual, reference[:, 0]))
        self.assertIs(
            actual.untyped_storage(),
            _SHARED_GATE_CACHE[str(static_logits.device)].untyped_storage(),
        )


if __name__ == "__main__":
    unittest.main()
