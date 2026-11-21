import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeVisionConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeVisionModel

from rtp_llm.models_py.new_models.qwen_service_vision import load_service_vision


class ServiceVisionCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.config = Qwen3VLVisionConfig(
            hidden_size=16,
            intermediate_size=32,
            out_hidden_size=16,
            depth=2,
            num_heads=2,
            in_channels=3,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            num_position_embeddings=16,
            deepstack_visual_indexes=[0],
        )
        self.config._attn_implementation = "eager"
        self.reference = Qwen3VLVisionModel(self.config).eval()

    def _save(self, root, prefix):
        weights = {
            prefix + k: v.detach().clone()
            for k, v in self.reference.state_dict().items()
        }
        weights["model.language_model.embed_tokens.weight"] = torch.zeros(2, 3)
        save_file(weights, str(root / "model.safetensors"))
        with torch.no_grad():
            for p in self.reference.parameters():
                p.copy_(p.half().float())

    def test_real_tower_forward_after_checkpoint(self):
        grid = torch.tensor([[1, 2, 2], [1, 2, 4]])
        pixels = torch.randn(12, 12)
        with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
            root = Path(tmp)
            self._save(root, "model.visual.")
            expected = self.reference(pixels, grid_thw=grid)
            for method in ("scratch", "fastsafetensors"):
                with self.subTest(method=method):
                    tower = Qwen3VLVisionModel(self.config).eval()
                    result = load_service_vision(
                        tower, tmp, torch.float32, "cpu", method, prefix="model.visual."
                    )
                    self.assertIs(result, tower)
                    for a, b in zip(result.parameters(), self.reference.parameters()):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                    actual = result(pixels, grid_thw=grid)
                    # Transformers versions expose tensor tuples or a ModelOutput.
                    for left, right in zip(
                        actual.to_tuple() if hasattr(actual, "to_tuple") else actual,
                        (
                            expected.to_tuple()
                            if hasattr(expected, "to_tuple")
                            else expected
                        ),
                    ):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_missing_weight_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_file(
                {"visual.pos_embed.weight": self.reference.pos_embed.weight.detach()},
                str(Path(tmp) / "model.safetensors"),
            )
            with self.assertRaisesRegex(RuntimeError, "missing required checkpoint"):
                load_service_vision(
                    self.reference, tmp, torch.float32, "cpu", "scratch", prefix=None
                )

    def test_qwen35_exporter_prefix_bf16_real_tower(self):
        config = Qwen3_5MoeVisionConfig(
            depth=1,
            hidden_size=16,
            intermediate_size=32,
            out_hidden_size=16,
            num_heads=2,
            in_channels=3,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            num_position_embeddings=16,
        )
        config._attn_implementation = "eager"
        self.reference = Qwen3_5MoeVisionModel(config).eval()
        with tempfile.TemporaryDirectory() as tmp, torch.no_grad():
            self._save(Path(tmp), "export.model.visual.")
            for p in self.reference.parameters():
                p.data = p.to(torch.bfloat16)
            pixels = torch.randn(4, 12, dtype=torch.bfloat16)
            grid = torch.tensor([[1, 2, 2]])
            expected = self.reference(pixels, grid_thw=grid)
            for method in ("scratch", "fastsafetensors"):
                result = load_service_vision(
                    Qwen3_5MoeVisionModel(config).eval(),
                    tmp,
                    torch.bfloat16,
                    "cpu",
                    method,
                    prefix=None,
                )
                actual = result(pixels, grid_thw=grid)
                torch.testing.assert_close(
                    actual.pooler_output, expected.pooler_output, rtol=0, atol=0
                )

    def test_split_gate_up_two_shards_and_partial_rejection(self):
        nn = torch.nn
        tower = nn.Module()
        tower.mlp = nn.Module()
        tower.mlp.up_gate_proj = nn.Linear(4, 6, bias=False)
        gate, up = torch.randn(3, 4), torch.randn(3, 4)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_file({"visual.mlp.up_proj.weight": up}, str(root / "00.safetensors"))
            with self.assertRaisesRegex(RuntimeError, "missing required checkpoint"):
                load_service_vision(
                    tower, tmp, torch.bfloat16, "cpu", "scratch", prefix="visual."
                )
            save_file(
                {"visual.mlp.gate_proj.weight": gate}, str(root / "01.safetensors")
            )
            for method in ("scratch", "fastsafetensors"):
                result = load_service_vision(
                    tower, tmp, torch.bfloat16, "cpu", method, prefix="visual."
                )
                torch.testing.assert_close(
                    result.mlp.up_gate_proj.weight,
                    torch.cat((gate, up), dim=0).half().bfloat16(),
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
