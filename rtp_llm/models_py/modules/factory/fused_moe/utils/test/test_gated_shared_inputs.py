"""Gate rounding, graph refresh, and BF16 shared-weight layout contracts."""

import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mega_moe_se import (
    MegaMoeSEExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.shared_inputs import (
    ensure_shared_gate_capacity,
    shared_expert_sigmoid,
)
from rtp_llm.utils.model_weight import W


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GatedSharedInputsTest(unittest.TestCase):
    def test_graph_preserves_bf16_gate_rounding_and_refresh(self):
        logits = torch.linspace(
            -10, 10, 129, device="cuda", dtype=torch.bfloat16
        ).unsqueeze(-1)
        ensure_shared_gate_capacity(logits.device, 129)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            shared_expert_sigmoid(logits)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = shared_expert_sigmoid(logits)
        address = output.data_ptr()
        for value in (-3.0, 0.0, 2.0):
            logits.copy_(torch.randn_like(logits) + value)
            graph.replay()
            self.assertEqual(address, output.data_ptr())
            torch.testing.assert_close(
                output, torch.sigmoid(logits[:, 0].float()), rtol=1e-6, atol=1e-7
            )

    def test_bf16_weight_orientation_and_k32_scales(self):
        # Non-square tensors detect transposition and gate/up reorder errors.
        hidden, inter = 256, 128
        torch.manual_seed(10)
        gate = torch.randn(hidden, inter, dtype=torch.bfloat16, device="cuda")
        up = torch.randn_like(gate) * 2
        down = torch.randn(inter, hidden, dtype=torch.bfloat16, device="cuda")
        weights = {W.ffn_w1: gate, W.ffn_w3: up, W.ffn_w2: down}
        captured = []

        def capture(l1, l2):
            captured.extend((l1, l2))
            return l1, l2

        executor = SimpleNamespace(cfg=SimpleNamespace(n_shared_experts=1))
        MegaMoeSEExecutor._setup_bf16_shared_weights(
            executor,
            weights,
            SimpleNamespace(transform_weights_for_mega_moe=capture),
            hidden,
            inter,
        )
        self.assertEqual(weights, {})
        for (data, packed), reference in zip(
            captured, (torch.cat((gate, up), -1).T, down.T)
        ):
            self.assertEqual(packed.stride(0), 1)
            groups = reference.float().reshape(reference.size(0), -1, 32)
            scale = torch.pow(
                2.0,
                torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448.0)),
            )
            expected_data = (
                (groups / scale.unsqueeze(-1))
                .to(torch.float8_e4m3fn)
                .reshape_as(reference)
            )
            torch.testing.assert_close(
                data.view(torch.uint8), expected_data.view(torch.uint8), rtol=0, atol=0
            )
            # Packing stores four exponent bytes per int32 in K order.
            expected_bytes = (scale.view(torch.int32) >> 23).to(torch.uint8)
            torch.testing.assert_close(
                packed.contiguous().view(torch.uint8), expected_bytes, rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
