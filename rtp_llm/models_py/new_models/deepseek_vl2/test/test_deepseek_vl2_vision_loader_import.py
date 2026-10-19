import sys
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file


class _FakeVision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 1152)

    def forward_features(self, images):
        features = self.proj(images.mean(dim=(-1, -2)))
        return features[:, None, :].repeat(1, 4, 1)


class DeepSeekVLV2VisionLoaderImportTest(unittest.TestCase):
    def test_vision_loader_import_is_independent(self):
        language_module = "rtp_llm.models_py.new_models.deepseek_vl2.language"
        self.assertNotIn(language_module, sys.modules)
        from rtp_llm.models_py.new_models.deepseek_vl2.vision import (
            DeepSeekVLV2ForVisionEmbedding,
            DeepSeekVLV2VisionModel,
            load_deepseek_vl2_vision,
        )

        self.assertNotIn(language_module, sys.modules)
        self.assertTrue(
            issubclass(DeepSeekVLV2ForVisionEmbedding, DeepSeekVLV2VisionModel)
        )
        config = {
            "vision_config": {},
            "projector_config": {
                "projector_type": "downsample_mlp_gelu",
                "input_dim": 1152,
                "n_embed": 4,
                "depth": 2,
                "downsample_ratio": 2,
            },
            "candidate_resolutions": [[384, 384]],
        }
        patch_target = (
            "rtp_llm.models_py.new_models.deepseek_vl2.vision.timm.create_model"
        )
        with mock.patch(
            patch_target, side_effect=lambda *args, **kwargs: _FakeVision()
        ):
            expected = DeepSeekVLV2VisionModel(config, torch.float32)
        expected_state = {
            name: tensor.detach().clone()
            for name, tensor in expected.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as model_path:
            save_file(expected_state, f"{model_path}/model.safetensors")
            with mock.patch(
                patch_target,
                side_effect=lambda *args, **kwargs: _FakeVision(),
            ):
                loaded = load_deepseek_vl2_vision(
                    vision_config=config,
                    model_path=model_path,
                    compute_dtype=torch.float32,
                    device="cpu",
                )
        for name, tensor in expected_state.items():
            torch.testing.assert_close(loaded.state_dict()[name], tensor)
        self.assertNotIn(language_module, sys.modules)


if __name__ == "__main__":
    unittest.main()
