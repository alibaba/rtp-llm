import tempfile
import types
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.model_loader.load_config import LoadMethod
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.qwen2_vl.vision import Qwen2VLForVisionEmbedding
from rtp_llm.multimodal.multimodal_mixins.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VisionTransformerPretrainedModel,
    VisionSdpaAttention,
)
from rtp_llm.multimodal.multimodal_mixins.qwen2_vl.qwen2_vl_mixin import (
    Qwen2_VLImageEmbedding,
    Qwen2_VLMixin,
)
from rtp_llm.utils.new_loader import is_new_loader_enabled


class Qwen2VLNewLoaderRoutingTest(unittest.TestCase):
    def test_newloader_switch_matches_language_loader_semantics(self):
        model_config = types.SimpleNamespace(use_new_loader=False)
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(is_new_loader_enabled(model_config))
            model_config.use_new_loader = True
            self.assertTrue(is_new_loader_enabled(model_config))
        model_config.use_new_loader = False
        with mock.patch.dict("os.environ", {"USE_NEW_LOADER": "1"}, clear=True):
            self.assertTrue(is_new_loader_enabled(model_config))
        del model_config.use_new_loader
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(is_new_loader_enabled(model_config))
        model_config.use_new_loader = "1"
        with self.assertRaisesRegex(TypeError, "use_new_loader must be a bool"):
            is_new_loader_enabled(model_config)

    def test_qwen2_vl_mixin_uses_standalone_newloader_vision(self):
        vision_config = {
            "depth": 1,
            "embed_dim": 8,
            "hidden_size": 6,
            "hidden_act": "quick_gelu",
            "mlp_ratio": 2.0,
            "num_heads": 2,
            "in_channels": 1,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
        }
        source = Qwen2VLForVisionEmbedding(
            {"model_type": "qwen2_vl_vision", "vision_config": vision_config},
            NewLoaderConfig(compute_dtype=torch.float32, device="cpu"),
        )
        with tempfile.TemporaryDirectory() as model_path:
            save_file(
                {
                    name: tensor.detach().clone()
                    for name, tensor in source.state_dict().items()
                },
                f"{model_path}/model.safetensors",
            )
            mm_related_params = types.SimpleNamespace(
                config={**vision_config, "ckpt_path": model_path},
                vit_weights=object(),
            )
            with mock.patch(
                "rtp_llm.multimodal.multimodal_mixins.qwen2_vl.qwen2_vl_mixin."
                "Qwen2VLImageProcessor.from_pretrained",
                return_value=object(),
            ):
                mixin = Qwen2_VLMixin(
                    torch.float32,
                    "cpu",
                    mm_related_params,
                    LoadMethod.SCRATCH,
                    VitConfig(),
                    model_path,
                    use_new_loader=True,
                )

        self.assertIsInstance(mixin.mm_part, Qwen2_VLImageEmbedding)
        self.assertIsNone(mm_related_params.vit_weights)
        self.assertNotIn("mm_mixin_loader", vars(mixin))
        for name, expected in source.visual.state_dict().items():
            torch.testing.assert_close(
                mixin.mm_part.visual.state_dict()[name], expected
            )

    def test_default_route_preserves_legacy_qwen2_vl_loader(self):
        vision_config = {
            "depth": 1,
            "embed_dim": 8,
            "hidden_size": 6,
            "hidden_act": "quick_gelu",
            "mlp_ratio": 2.0,
            "num_heads": 2,
            "in_channels": 1,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
        }
        source = Qwen2VisionTransformerPretrainedModel(vision_config).eval()
        with tempfile.TemporaryDirectory() as model_path:
            save_file(
                {
                    f"visual.{name}": tensor.detach().clone()
                    for name, tensor in source.state_dict().items()
                },
                f"{model_path}/model.safetensors",
            )
            mm_related_params = types.SimpleNamespace(
                config={**vision_config, "ckpt_path": model_path},
                vit_weights=None,
            )
            with mock.patch(
                "rtp_llm.multimodal.multimodal_mixins.qwen2_vl.qwen2_vl_mixin."
                "Qwen2VLImageProcessor.from_pretrained",
                return_value=object(),
            ):
                mixin = Qwen2_VLMixin(
                    torch.float32,
                    "cpu",
                    mm_related_params,
                    LoadMethod.SCRATCH,
                    VitConfig(),
                    model_path,
                )

        self.assertIn("mm_mixin_loader", vars(mixin))
        self.assertIsNotNone(mm_related_params.vit_weights)
        for name, expected in source.state_dict().items():
            torch.testing.assert_close(
                mixin.mm_part.visual.state_dict()[name],
                expected,
                atol=2e-4,
                rtol=1e-3,
            )

    def test_new_vision_matches_legacy_video_and_hf_checkpoint_layout(self):
        vision_config = {
            "depth": 1,
            "embed_dim": 8,
            "hidden_size": 6,
            "hidden_act": "quick_gelu",
            "mlp_ratio": 2.0,
            "num_heads": 2,
            "in_channels": 1,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        }
        torch.manual_seed(23)
        legacy = Qwen2VisionTransformerPretrainedModel(vision_config).eval()
        for block in legacy.blocks:
            sdpa = VisionSdpaAttention(vision_config["embed_dim"], num_heads=2)
            sdpa.load_state_dict(block.attn.state_dict())
            block.attn = sdpa

        checkpoint = {
            f"visual.{name}": tensor.detach().clone()
            for name, tensor in legacy.state_dict().items()
        }
        with tempfile.TemporaryDirectory() as model_path:
            save_file(checkpoint, f"{model_path}/model.safetensors")
            loaded = NewModelLoader(
                model_config={
                    "model_type": "qwen2_vl_vision",
                    "vision_config": vision_config,
                },
                load_config=NewLoaderConfig(
                    compute_dtype=torch.float32,
                    device="cpu",
                ),
                model_path=model_path,
            ).load()

        for name, expected in legacy.state_dict().items():
            torch.testing.assert_close(loaded.visual.state_dict()[name], expected)

        pixel_values = torch.linspace(-1.0, 1.0, 128).reshape(16, 8)
        grid_thw = torch.tensor([[2, 2, 2], [1, 4, 2]], dtype=torch.int64)
        with torch.inference_mode():
            expected = legacy(pixel_values, grid_thw)
            actual = loaded(pixel_values, grid_thw)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
