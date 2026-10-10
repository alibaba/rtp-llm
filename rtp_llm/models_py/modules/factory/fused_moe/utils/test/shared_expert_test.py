import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.shared_expert import (
    FusedSharedExpertFastPath,
    combine_routed_and_shared,
)


class SharedExpertCombineTest(unittest.TestCase):
    def setUp(self):
        self.routed = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        self.shared = torch.tensor([[0.5, -0.5]], dtype=torch.float32)
        self.out = torch.empty((1, 2), dtype=torch.bfloat16)

    def test_bf16_add_writes_supplied_output(self):
        with patch.dict(
            os.environ,
            {"MOE_SHARED_EXPERT_BF16_ADD": "1", "MOE_STRICT_FUSED": "0"},
        ):
            result = combine_routed_and_shared(
                self.routed, self.shared, torch.bfloat16, out=self.out
            )

        self.assertIs(result, self.out)
        torch.testing.assert_close(
            result, (self.routed + self.shared).to(torch.bfloat16)
        )

    def test_fallback_writes_supplied_output(self):
        triton_module = "rtp_llm.models_py.triton_kernels.moe.shared_expert"
        fake_module = SimpleNamespace(
            fused_moe_epilogue=lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("unavailable")
            )
        )
        with (
            patch.dict(
                os.environ,
                {"MOE_SHARED_EXPERT_BF16_ADD": "0", "MOE_STRICT_FUSED": "0"},
            ),
            patch.dict(sys.modules, {triton_module: fake_module}),
        ):
            result = combine_routed_and_shared(
                self.routed, self.shared, torch.bfloat16, out=self.out
            )

        self.assertIs(result, self.out)
        torch.testing.assert_close(
            result, (self.routed + self.shared).to(torch.bfloat16)
        )


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "requires Blackwell DeepGEMM",
)
class SharedExpertDeepGemmCompatibilityTest(unittest.TestCase):
    def test_group128_shared_expert_preserves_quantization_and_graph_replay(self):
        from deep_gemm.utils.layout import get_mn_major_tma_aligned_packed_ue8m0_tensor

        generator = torch.Generator(device="cuda").manual_seed(471)
        dim, inter = 256, 256
        expert = torch.nn.Module()
        dense_weights = []
        for name, rows, columns in (("w13", 2 * inter, dim), ("w2", dim, inter)):
            linear = torch.nn.Module()
            weight = torch.randn(rows, columns, device="cuda", generator=generator).to(
                torch.float8_e4m3fn
            )
            scale = torch.full((rows, columns // 128), 1 / 32, device="cuda")
            linear.register_buffer("weight", weight)
            linear.register_buffer(
                "weight_scales",
                get_mn_major_tma_aligned_packed_ue8m0_tensor(scale),
            )
            setattr(expert, name, linear)
            dense_weights.append(weight.float() / 32)

        def quantized_reference(values, floor):
            groups = values.float().reshape(values.shape[0], -1, 128)
            scale = torch.exp2(
                torch.ceil(torch.log2((groups.abs().amax(-1) / 448).clamp(min=floor)))
            )
            quant = (groups / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
            return (quant.float() * scale.unsqueeze(-1)).reshape(values.shape)

        def reference(values):
            gate_up = (
                quantized_reference(values, 1.0e-4 / 448) @ dense_weights[0].T
            ).to(torch.bfloat16)
            gate, up = gate_up.float().chunk(2, dim=-1)
            hidden = (
                torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
            ).to(torch.bfloat16)
            return (quantized_reference(hidden, 1.0e-10) @ dense_weights[1].T).to(
                torch.bfloat16
            )

        runner = FusedSharedExpertFastPath(
            max_tokens_per_rank=17, dim=dim, inter_dim=inter, swiglu_limit=10
        )
        x = torch.randn(17, dim, device="cuda", generator=generator).bfloat16()
        runner.prepare(expert)
        for _ in range(3):
            actual = runner._run_prepared(expert, x)
        torch.testing.assert_close(actual, reference(x), rtol=0.02, atol=0.002)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = runner._run_prepared(expert, x)
        x.mul_(0.75)
        graph.replay()
        torch.testing.assert_close(captured, reference(x), rtol=0.02, atol=0.002)


if __name__ == "__main__":
    unittest.main()
