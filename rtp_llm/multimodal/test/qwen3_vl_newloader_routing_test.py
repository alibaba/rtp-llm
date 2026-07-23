import tempfile
import types
import unittest
from unittest import mock

import torch
from PIL import Image
from safetensors.torch import save_file
from transformers import Qwen2VLImageProcessor, Qwen3VLVisionModel
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.model_loader.load_config import LoadMethod
from rtp_llm.models.qwen3_vl import QWen3_VL
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.modules.base.common.multimodal_embedding import (
    reshape_extra_input_to_deepstack,
)
from rtp_llm.models_py.new_models.qwen3_vl.vision import Qwen3VLForVisionEmbedding
from rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin import (
    Qwen3_VLImageEmbedding,
    Qwen3_VLMixin,
    Qwen3_VLMoeLegacyMixin,
)
from rtp_llm.utils.base_model_datatypes import MMUrlType


def _vision_config():
    return {
        "depth": 1,
        "hidden_size": 8,
        "hidden_act": "gelu_pytorch_tanh",
        "intermediate_size": 16,
        "num_heads": 2,
        "in_channels": 1,
        "patch_size": 2,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "out_hidden_size": 6,
        "num_position_embeddings": 16,
        "deepstack_visual_indexes": [0],
    }


def _model_config_json():
    return {
        "vision_start_token_id": 1,
        "vision_end_token_id": 2,
        "vision_config": {
            "hidden_size": 8,
            "spatial_merge_size": 2,
        },
        "text_config": {
            "intermediate_size": 512,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "hidden_size": 256,
            "num_hidden_layers": 2,
            "vocab_size": 64,
            "rope_scaling": {"mrope_section": [16, 24, 24]},
            "dtype": "bfloat16",
            "torch_dtype": "float16",
        },
    }


class Qwen3VLNewLoaderRoutingTest(unittest.TestCase):
    def test_qwen3_vl_mixin_uses_standalone_newloader_vision(self):
        vision_config = _vision_config()
        source = Qwen3VLForVisionEmbedding(
            {"model_type": "qwen3_vl_vision", "vision_config": vision_config},
            NewLoaderConfig(compute_dtype=torch.float32, device="cpu"),
        ).eval()
        with tempfile.TemporaryDirectory() as model_path:
            save_file(
                {
                    f"model.{name}": tensor.detach().to(torch.float16)
                    for name, tensor in source.state_dict().items()
                },
                f"{model_path}/model.safetensors",
            )
            mm_related_params = types.SimpleNamespace(
                config={**vision_config, "ckpt_path": model_path},
                vit_weights=object(),
            )
            processor = types.SimpleNamespace(image_processor=None)
            with mock.patch(
                "rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin."
                "AutoProcessor.from_pretrained",
                return_value=processor,
            ):
                mixin = Qwen3_VLMixin(
                    torch.float32,
                    "cpu",
                    mm_related_params,
                    LoadMethod.SCRATCH,
                    VitConfig(),
                    model_path,
                    use_new_loader=True,
                )

        self.assertIsInstance(mixin.mm_part, Qwen3_VLImageEmbedding)
        self.assertTrue(mixin.mm_part._uses_new_loader_vision)
        self.assertIsNone(mixin.mm_part.mm_processor.image_processor)
        self.assertIsNone(mm_related_params.vit_weights)
        self.assertNotIn("mm_mixin_loader", vars(mixin))
        for name, expected in source.visual.state_dict().items():
            torch.testing.assert_close(
                mixin.mm_part.visual.state_dict()[name],
                expected.to(torch.float16).to(torch.float32),
            )

    def test_default_route_keeps_legacy_vision_loader(self):
        visual = torch.nn.Identity()
        visual.spatial_merge_size = 2
        visual.patch_size = 16
        config = types.SimpleNamespace(
            vision_config=types.SimpleNamespace(_attn_implementation=None)
        )
        mm_related_params = types.SimpleNamespace(
            config={"ckpt_path": "/tmp/model"},
            vit_weights=None,
        )
        processor = types.SimpleNamespace(image_processor=None)
        with mock.patch(
            "rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin."
            "AutoProcessor.from_pretrained",
            return_value=processor,
        ), mock.patch(
            "rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin."
            "Qwen3VLConfig.from_pretrained",
            return_value=config,
        ), mock.patch(
            "rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin."
            "Qwen3VLVisionModel._from_config",
            return_value=visual,
        ):
            mixin = Qwen3_VLMixin.__new__(Qwen3_VLMixin)
            mixin.compute_dtype = torch.float32
            mixin.device = "cpu"
            mixin.mm_related_params = mm_related_params
            mixin.vit_config = VitConfig()
            mixin.load_method = LoadMethod.SCRATCH
            mixin.ckpt_path = "/tmp/model"
            mixin.use_new_loader = False
            mixin._init_multimodal()

        self.assertFalse(mixin.mm_part._uses_new_loader_vision)
        self.assertIsNone(mixin.mm_part.mm_processor.image_processor)
        self.assertIsNotNone(mm_related_params.vit_weights)

    def test_both_routes_preserve_the_native_auto_processor(self):
        def image_processor():
            return Qwen2VLImageProcessor(
                min_pixels=16,
                max_pixels=64,
                patch_size=2,
                temporal_patch_size=2,
                merge_size=2,
            )

        newloader_processor = types.SimpleNamespace(image_processor=image_processor())
        legacy_processor = types.SimpleNamespace(image_processor=image_processor())
        visual = torch.nn.Identity()
        visual.spatial_merge_size = 2
        visual.patch_size = 16
        mm_related_params = types.SimpleNamespace(
            config={"ckpt_path": "/tmp/model"},
            vit_weights=None,
        )
        config = types.SimpleNamespace(
            vision_config=types.SimpleNamespace(_attn_implementation=None)
        )

        with mock.patch(
            "rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin."
            "AutoProcessor.from_pretrained",
            return_value=newloader_processor,
        ):
            newloader_embedding = Qwen3_VLImageEmbedding(
                mm_related_params, visual=visual
            )
        with mock.patch(
            "rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin."
            "AutoProcessor.from_pretrained",
            return_value=legacy_processor,
        ), mock.patch(
            "rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin."
            "Qwen3VLConfig.from_pretrained",
            return_value=config,
        ), mock.patch(
            "rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin."
            "Qwen3VLVisionModel._from_config",
            return_value=visual,
        ):
            legacy_embedding = Qwen3_VLImageEmbedding(mm_related_params)

        self.assertIs(newloader_embedding.mm_processor, newloader_processor)
        self.assertIs(legacy_embedding.mm_processor, legacy_processor)
        self.assertIs(
            newloader_embedding.mm_processor.image_processor,
            newloader_processor.image_processor,
        )
        self.assertIs(
            legacy_embedding.mm_processor.image_processor,
            legacy_processor.image_processor,
        )
        image = Image.new("RGB", (8, 8), color=(32, 64, 128))
        newloader_output = newloader_embedding.mm_processor.image_processor(
            image, return_tensors="pt", do_resize=True
        )
        legacy_output = legacy_embedding.mm_processor.image_processor(
            image, return_tensors="pt", do_resize=True
        )
        torch.testing.assert_close(
            newloader_output["pixel_values"], legacy_output["pixel_values"]
        )
        torch.testing.assert_close(
            newloader_output["image_grid_thw"], legacy_output["image_grid_thw"]
        )

    def test_qwen3_vl_moe_is_explicitly_legacy_only(self):
        with self.assertRaisesRegex(
            ValueError, "qwen3_vl_moe is not supported by the new loader"
        ):
            Qwen3_VLMoeLegacyMixin(
                torch.float32,
                "cpu",
                types.SimpleNamespace(config={}, vit_weights=None),
                LoadMethod.SCRATCH,
                VitConfig(),
                "/tmp/model",
                use_new_loader=True,
            )

        mm_related_params = types.SimpleNamespace(config={}, vit_weights=None)
        with mock.patch.object(Qwen3_VLMixin, "_init_multimodal") as legacy_init:
            Qwen3_VLMoeLegacyMixin(
                torch.float32,
                "cpu",
                mm_related_params,
                LoadMethod.SCRATCH,
                VitConfig(),
                "/tmp/model",
                use_new_loader=False,
            )
        legacy_init.assert_called_once_with()

    def test_batched_multi_image_deepstack_roundtrip_matches_single_image(self):
        class FakeVisual:
            dtype = torch.float32
            device = torch.device("cpu")
            patch_size = 1
            spatial_merge_size = 1

            def __call__(self, pixel_values, grid_thw):
                outputs = []
                offset = 0
                for grid in grid_thw:
                    token_count = int(grid.prod().item())
                    values = pixel_values[offset : offset + token_count, :1]
                    outputs.append(torch.cat((values, values + 1, values + 2), dim=1))
                    offset += token_count
                embeddings = torch.cat(outputs)
                return embeddings, [embeddings + 10, embeddings + 20]

        embedding = Qwen3_VLImageEmbedding.__new__(Qwen3_VLImageEmbedding)
        embedding.visual = FakeVisual()
        embedding.spatial_merge_size = 1
        data_list = [
            (
                torch.tensor([[1.0], [2.0]]),
                torch.tensor([[1, 1, 2]], dtype=torch.int64),
            ),
            (
                torch.tensor([[3.0], [4.0], [5.0]]),
                torch.tensor([[1, 1, 3]], dtype=torch.int64),
            ),
        ]

        batched = embedding.batched_embedding(
            data_list, [MMUrlType.IMAGE, MMUrlType.IMAGE]
        )
        singles = [embedding.embedding(data) for data in data_list]
        for actual, expected in zip(batched, singles):
            torch.testing.assert_close(actual[0], expected[0])
            torch.testing.assert_close(actual[1], expected[1])
            torch.testing.assert_close(actual[2], expected[2])

        restored = reshape_extra_input_to_deepstack(
            [item[2] for item in batched],
            [item[0] for item in batched],
        )
        for expected, deepstack in zip(singles, restored):
            torch.testing.assert_close(
                deepstack,
                expected[2].reshape(2, expected[0].size(0), expected[0].size(1)),
            )

        self.assertEqual(embedding.batched_embedding([], []), [])
        with self.assertRaisesRegex(ValueError, "data_list has 2 entries"):
            embedding.batched_embedding(data_list, [MMUrlType.IMAGE])

    def test_deepstack_transport_rejects_malformed_inputs(self):
        feature = torch.zeros(2, 3)
        with self.assertRaisesRegex(ValueError, "extra_input has 0 entries"):
            reshape_extra_input_to_deepstack([], [feature])
        with self.assertRaisesRegex(ValueError, "flat 1-D"):
            reshape_extra_input_to_deepstack([torch.zeros(2, 3)], [feature])
        with self.assertRaisesRegex(ValueError, "non-empty shape"):
            reshape_extra_input_to_deepstack([torch.zeros(6)], [torch.empty(0, 3)])
        with self.assertRaisesRegex(ValueError, "cannot be reshaped"):
            reshape_extra_input_to_deepstack([torch.zeros(5)], [feature])
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            Qwen3_VLImageEmbedding._unpack_vision_output((torch.zeros(1, 3), []))

    def test_config_json_requires_typed_vision_and_text_sections(self):
        for missing, message in (
            ("vision_config", "vision_config"),
            ("text_config", "text_config"),
        ):
            with self.subTest(missing=missing):
                config_json = _model_config_json()
                del config_json[missing]
                with self.assertRaisesRegex(ValueError, message):
                    QWen3_VL._from_config_json(ModelConfig(), config_json)

    def test_config_json_propagates_vision_and_dtype(self):
        config = ModelConfig()
        config.ckpt_path = "/tmp/qwen3-vl"
        config_json = _model_config_json()
        QWen3_VL._from_config_json(config, config_json)

        self.assertEqual(config.config_dtype, "bfloat16")
        self.assertEqual(config.mm_related_params.config["hidden_size"], 8)
        self.assertEqual(config.mm_related_params.config["ckpt_path"], "/tmp/qwen3-vl")
        self.assertEqual(config.attn_config.rope_config.index_factor, 3)

        fallback_config = ModelConfig()
        fallback_json = _model_config_json()
        fallback_json["text_config"]["dtype"] = None
        QWen3_VL._from_config_json(fallback_config, fallback_json)
        self.assertEqual(fallback_config.config_dtype, "float16")

    def test_config_json_rejects_missing_or_invalid_mrope_sections(self):
        config_json = _model_config_json()
        del config_json["text_config"]["rope_scaling"]
        with self.assertRaisesRegex(ValueError, "rope_scaling"):
            QWen3_VL._from_config_json(ModelConfig(), config_json)

        config_json = _model_config_json()
        config_json["text_config"]["rope_scaling"]["mrope_section"] = [16, 24]
        with self.assertRaisesRegex(ValueError, "three positive integers"):
            QWen3_VL._from_config_json(ModelConfig(), config_json)

        config_json = _model_config_json()
        config_json["text_config"]["rope_scaling"]["mrope_section"] = [16, 24, 23]
        with self.assertRaisesRegex(ValueError, "half of head_dim"):
            QWen3_VL._from_config_json(ModelConfig(), config_json)

    def test_newloader_vision_matches_transformers_reference(self):
        vision_config = _vision_config()
        reference_config = Qwen3VLVisionConfig(**vision_config)
        reference_config._attn_implementation = "sdpa"
        torch.manual_seed(23)
        reference = Qwen3VLVisionModel(reference_config).eval()
        checkpoint_state = {
            name: tensor.detach().to(torch.float16)
            for name, tensor in reference.state_dict().items()
        }
        reference.load_state_dict(checkpoint_state)

        with tempfile.TemporaryDirectory() as model_path:
            save_file(
                {
                    f"model.visual.{name}": tensor
                    for name, tensor in checkpoint_state.items()
                },
                f"{model_path}/model.safetensors",
            )
            actual = NewModelLoader(
                model_config={
                    "model_type": "qwen3_vl_vision",
                    "vision_config": vision_config,
                },
                load_config=NewLoaderConfig(compute_dtype=torch.float32, device="cpu"),
                model_path=model_path,
            ).load()

        pixel_values = torch.linspace(-1.0, 1.0, 128).reshape(16, 8)
        grid_thw = torch.tensor([[2, 2, 2], [1, 4, 2]], dtype=torch.int64)
        with torch.inference_mode():
            expected_output, expected_deepstack = (
                Qwen3_VLImageEmbedding._unpack_vision_output(
                    reference(pixel_values, grid_thw)
                )
            )
            actual_output, actual_deepstack = (
                Qwen3_VLImageEmbedding._unpack_vision_output(
                    actual(pixel_values, grid_thw)
                )
            )

        torch.testing.assert_close(actual_output, expected_output)
        self.assertEqual(len(actual_deepstack), len(expected_deepstack))
        for actual_feature, expected_feature in zip(
            actual_deepstack, expected_deepstack
        ):
            torch.testing.assert_close(actual_feature, expected_feature)


if __name__ == "__main__":
    unittest.main()
