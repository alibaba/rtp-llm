"""Shared-expert phase ordering, instance options and changing Graph inputs."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rtp_llm.platforms.ppu.models.dsv4.manifest import DECODE_EXECUTION_OPTIONS
from rtp_llm.platforms.ppu.models.dsv4.ppu_decode_provider import PpuDecodeProvider
from rtp_llm.platforms.ppu.models.dsv4.ppu_mega_moe import PpuMegaMoEFP4Strategy
from rtp_llm.platforms.ppu.models.dsv4.ppu_shared_expert import PpuSharedExpertExecutor


class StagedSharedContractTest(unittest.TestCase):
    def test_snapshot_and_backend_requirement(self):
        options = {**DECODE_EXECUTION_OPTIONS, "DSV4_PPU_DECODE_MOE_BACKEND": "megamoe"}
        default = PpuDecodeProvider(options)
        options["DSV4_PPU_MEGAMOE_SHARED_STAGED"] = "1"
        staged = PpuDecodeProvider(options)
        options["DSV4_PPU_MEGAMOE_SHARED_STAGED"] = "0"
        self.assertFalse(default.build_shared_expert_executor().staged)
        self.assertTrue(staged.build_shared_expert_executor().staged)
        with self.assertRaisesRegex(ValueError, "MegaMoE backend"):
            PpuDecodeProvider(
                {
                    **options,
                    "DSV4_PPU_DECODE_MOE_BACKEND": "deepep",
                    "DSV4_PPU_MEGAMOE_SHARED_STAGED": "1",
                }
            )
        with self.assertRaisesRegex(ValueError, "early overlap"):
            PpuSharedExpertExecutor(staged=True)

    def test_missing_native_phase_api_rejected_before_forward(self):
        strategy = object.__new__(PpuMegaMoEFP4Strategy)
        torch.nn.Module.__init__(strategy)
        strategy._mega = SimpleNamespace(fp4_fp4_mega_moe=lambda *args: None)
        with self.assertRaisesRegex(RuntimeError, "after_gemm1"):
            strategy.require_staged_shared()
        strategy._mega = SimpleNamespace(
            fp4_fp4_mega_moe=lambda *args, after_gemm1=None: None
        )
        strategy.require_staged_shared()


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "ZW-M890P",
    "requires M890P",
)
class StagedSharedGpuTest(unittest.TestCase):
    @torch.inference_mode()
    def test_graph_replay_and_abort_preserve_bf16_shared_result(self):
        from rtp_llm.platforms.ppu.modules.linear.fp8_linear import (
            quantize_ppu_fp8_activation,
        )

        torch.manual_seed(606890)
        provider = PpuDecodeProvider(
            {
                **DECODE_EXECUTION_OPTIONS,
                "DSV4_PPU_DECODE_MOE_BACKEND": "megamoe",
                "DSV4_PPU_DECODE_FUSIONS": "1",
                "DSV4_PPU_MEGAMOE_SHARED_STAGED": "1",
            }
        )

        def weight(shape):
            return torch.randn(shape, device="cuda").to(torch.float8_e4m3fn)

        def scale(shape):
            return torch.ones(shape, device="cuda").to(torch.float8_e8m0fnu)

        shared = provider.build_shared_expert(
            4096,
            2048,
            {
                "w13_w": weight((4096, 4096)),
                "w13_s": scale((32, 32)),
                "w2_w": weight((4096, 2048)),
                "w2_s": scale((32, 16)),
            },
            swiglu_limit=10.0,
            sglang_moe=True,
        )
        staged = provider.build_shared_expert_executor()
        staged.prepare(shared)
        baseline = PpuSharedExpertExecutor(
            provider.stream_pool, start_before_routing=True
        )
        baseline.prepare(shared)
        for rows in (1, 8, 32):
            x = torch.randn((rows, 4096), device="cuda", dtype=torch.bfloat16)
            for prequantized in (False, True):

                def run(executor):
                    pair = (
                        quantize_ppu_fp8_activation(x, quantization="v2_column")
                        if prequantized
                        else None
                    )
                    executor.start(shared, x, input_quantized=pair)
                    if executor.staged:
                        executor.after_routed_gemm1()
                    return executor.finish()

                for _ in range(3):
                    run(staged)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = run(staged)
                for magnitude in (0.0, 0.001, 1.0, 10.0):
                    x.normal_().mul_(magnitude)
                    expected = run(baseline)
                    graph.replay()
                    self.assertTrue(
                        torch.equal(
                            actual.view(torch.uint8), expected.view(torch.uint8)
                        )
                    )
                del graph, actual
            staged.start(shared, x)
            with self.assertRaisesRegex(RuntimeError, "pending output"):
                staged.start(shared, x)
            with self.assertRaisesRegex(RuntimeError, "did not complete"):
                staged.finish()
            staged.abort()
            staged.start(shared, x)
            with patch.object(
                shared, "finish_projection", side_effect=ValueError("phase failed")
            ):
                with self.assertRaisesRegex(ValueError, "phase failed"):
                    staged.after_routed_gemm1()
            staged.abort()
            staged.start(shared, x)
            staged.after_routed_gemm1()
            actual = staged.finish()
            self.assertTrue(
                torch.equal(actual.view(torch.uint8), shared(x).view(torch.uint8))
            )
            with self.assertRaisesRegex(RuntimeError, "no pending projection"):
                staged.after_routed_gemm1()


if __name__ == "__main__":
    unittest.main()
