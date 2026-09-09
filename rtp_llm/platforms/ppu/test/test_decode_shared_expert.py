"""Validate PPU shared-expert selection and real-weight Graph execution."""

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors import safe_open

from rtp_llm.models_py.modules.dsv4.moe.shared_expert import (
    FusedSharedExpertFastPath,
    SequentialSharedExpertExecutor,
)
from rtp_llm.platforms.ppu.models.dsv4.ppu_decode_provider import PpuDecodeProvider
from rtp_llm.platforms.ppu.models.dsv4.manifest import DECODE_EXECUTION_OPTIONS


@unittest.skipUnless(
    os.environ.get("RTP_PPU_DSV4_CHECKPOINT")
    and torch.cuda.is_available()
    and torch.cuda.get_device_name() == "ZW-M890P",
    "requires a read-only Flash checkpoint and a PPU M890P",
)
class DecodeSharedExpertTest(unittest.TestCase):
    def setUp(self):
        checkpoint = Path(os.environ["RTP_PPU_DSV4_CHECKPOINT"])
        config = json.loads((checkpoint / "config.json").read_text())
        index = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]

        def read(name):
            key = "layers.2.ffn.shared_experts." + name
            with safe_open(checkpoint / index[key], framework="pt", device="cpu") as f:
                return f.get_tensor(key).cuda()

        self.provider = PpuDecodeProvider(DECODE_EXECUTION_OPTIONS)
        self.shared = self.provider.build_shared_expert(
            4096,
            2048,
            expert_weights={
                "w13_w": torch.cat([read("w1.weight"), read("w3.weight")]),
                "w13_s": torch.cat([read("w1.scale"), read("w3.scale")]),
                "w2_w": read("w2.weight"),
                "w2_s": read("w2.scale"),
            },
            swiglu_limit=config.get("swiglu_limit", 0.0),
        )

    @torch.inference_mode()
    def test_graph_with_strict_fused_enabled(self):
        executor = self.provider.build_shared_expert_executor()
        executor.prepare(self.shared)
        torch.manual_seed(890413)
        with patch.dict(os.environ, {"DSV4_MOE_STRICT_FUSED": "1"}), patch.object(
            FusedSharedExpertFastPath,
            "run",
            side_effect=AssertionError("CUDA scale ABI"),
        ):
            for batch in (1, 3, 8, 32, 128):
                with self.subTest(batch=batch):
                    x = torch.randn((batch, 4096), device="cuda", dtype=torch.bfloat16)

                    def run():
                        executor.start(self.shared, x)
                        return executor.finish()

                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            run()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        actual = run()
                    torch.cuda.current_stream().wait_stream(stream)
                    for _ in range(5):
                        x.normal_()
                        expected = self.shared(x)
                        actual.fill_(float("nan"))
                        graph.replay()
                        self.assertEqual(actual.dtype, torch.bfloat16)
                        self.assertTrue(bool(actual.isfinite().all()))
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rejects_unprepared_or_foreign_module(self):
        executor = self.provider.build_shared_expert_executor()
        x = torch.ones((1, 4096), device="cuda", dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "prepared module"):
            executor.start(self.shared, x)
        with self.assertRaisesRegex(RuntimeError, "no pending output"):
            executor.finish()
        with self.assertRaisesRegex(TypeError, "PPU FP8"):
            executor.prepare(torch.nn.Identity())
        executor.prepare(self.shared)
        with self.assertRaisesRegex(RuntimeError, "prepared module"):
            executor.start(torch.nn.Identity(), x)
        with patch.dict(os.environ, {"DSV4_MOE_STRICT_FUSED": "1"}):
            with self.assertRaisesRegex(RuntimeError, "forbids generic"):
                SequentialSharedExpertExecutor().start(self.shared, x)

    @torch.inference_mode()
    def test_overlap_graph_joins_producer_and_consumer(self):
        from rtp_llm.models_py.modules.dsv4.moe.moe_layer import MoE

        provider = PpuDecodeProvider(DECODE_EXECUTION_OPTIONS)
        executor = provider.build_shared_expert_executor()
        executor.prepare(self.shared)
        second = provider.build_shared_expert_executor()
        second.prepare(self.shared)
        self.assertIs(executor._stream, second._stream)
        other_provider = PpuDecodeProvider(DECODE_EXECUTION_OPTIONS)
        other = other_provider.build_shared_expert_executor()
        other.prepare(self.shared)
        self.assertIsNot(executor._stream, other._stream)
        self.assertEqual(executor.name, "ppu_overlap")

        original = self.shared.forward
        calls = []

        def observed(x):
            calls.append(
                (
                    torch.cuda.current_stream().cuda_stream,
                    torch.cuda.is_current_stream_capturing(),
                )
            )
            return original(x)

        torch.manual_seed(890414)
        with patch.object(self.shared, "forward", side_effect=observed), patch.dict(
            os.environ, {"DSV4_MOE_STRICT_FUSED": "1"}
        ):
            for batch in (1, 3, 8, 32, 128):
                with self.subTest(batch=batch):
                    x = torch.randn((batch, 4096), device="cuda", dtype=torch.bfloat16)

                    def run():
                        # Producer work must precede the auxiliary stream's read.
                        x.add_(1)
                        context = SimpleNamespace(
                            _routed_includes_shared=False,
                            _shared_executor=executor,
                            shared_experts=self.shared,
                            _route=lambda values, ids: (values * 0.5, ids),
                        )
                        routed, _ = MoE._route_and_start_shared(context, x, None)
                        shared = executor.finish()
                        # Consumer work must wait for the shared result.
                        return shared.float() + routed.float()

                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            run()
                    torch.cuda.current_stream().wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        actual = run()
                    self.assertEqual(calls[-1], (executor._stream.cuda_stream, True))
                    torch.cuda.current_stream().wait_stream(stream)
                    for _ in range(5):
                        x.normal_()
                        updated = x + 1
                        expected = original(updated).float() + (updated * 0.5).float()
                        actual.fill_(float("nan"))
                        graph.replay()
                        self.assertTrue(bool(actual.isfinite().all()))
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        x = torch.ones((1, 4096), device="cuda", dtype=torch.bfloat16)
        executor.start(self.shared, x)
        with self.assertRaisesRegex(RuntimeError, "pending output"):
            executor.start(self.shared, x)
        with self.assertRaisesRegex(RuntimeError, "pending output"):
            executor.prepare(self.shared)
        executor.finish()


if __name__ == "__main__":
    unittest.main()
