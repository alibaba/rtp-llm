"""Validate PPU shared-expert selection and real-weight Graph execution."""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors import safe_open

from rtp_llm.models_py.modules.dsv4.moe.shared_expert import (
    FusedSharedExpertFastPath,
    SequentialSharedExpertExecutor,
)
from rtp_llm.platforms.ppu.models.dsv4.ppu_decode_provider import PpuDecodeProvider


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

        self.provider = PpuDecodeProvider({"DSV4_PPU_SGLANG_MOE": "1"})
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


if __name__ == "__main__":
    unittest.main()
