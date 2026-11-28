import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.multimodal.multimodal_mixin_factory import MultimodalMixinFactory
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_mixin import (
    DeepSeekV41Mixin,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_processor import (
    V41ImageInput,
    image_token_types,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_vision import (
    DeepSeekV41VisionEmbedding,
    OwnedDeepSeekV41VisionEmbedding,
    load_v41_service_vision,
)


def tiny_config():
    return V41Config.from_dict(
        {
            "text_config": {
                "hidden_size": 8,
                "vocab_size": 129280,
                "max_position_embeddings": 128,
            },
            "vision_config": {
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "patch_size": 14,
                "downsample_ratio": 3,
                "rope_theta": 10000.0,
                "max_image_tokens": 1024,
                "min_pixels": 295936,
            },
            "quantization_config": {},
            "dtype": "bfloat16",
            "bos_token_id": 0,
            "eos_token_id": 1,
            "pad_token_id": 2,
            "image_token_id": 129264,
        }
    )


class V41VisionNewLoaderTest(unittest.TestCase):
    def setUp(self):
        self.config = tiny_config()
        torch.manual_seed(41)
        self.reference = DeepSeekV41VisionEmbedding(self.config, device="cpu").eval()
        with torch.no_grad():
            for name, value in self.reference.named_parameters():
                value.copy_(torch.randn_like(value) * 0.05)
                if value.dtype == torch.float32:
                    value.add_(0.1234567)
        self.state = {
            k: v.detach().contiguous().clone()
            for k, v in self.reference.state_dict().items()
        }

    def write_checkpoint(self, root, state):
        keys = list(state)
        shards = {
            "vision-a.safetensors": {k: state[k] for k in keys[::2]},
            "vision-b.safetensors": {k: state[k] for k in keys[1::2]},
            "text.safetensors": {
                "layers.0.engram.embed.weight": torch.ones(3),
                "mtp.0.main_proj.weight": torch.ones(3),
                "embed.weight": torch.ones(3),
            },
        }
        index = {}
        for filename, tensors in shards.items():
            save_file(tensors, str(Path(root) / filename))
            index.update({k: filename for k in tensors})
        (Path(root) / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": index})
        )

    def methods(self):
        return (
            ("scratch", "fastsafetensors")
            if os.getenv("NEWLOADER_TEST_NATIVE_FST") == "1"
            else ("scratch",)
        )

    def test_direct_real_checkpoint_and_existing_cpu_image_math(self):
        for method in self.methods():
            with self.subTest(method=method), tempfile.TemporaryDirectory() as root:
                self.write_checkpoint(root, self.state)
                with self.assertLogs("rtp_llm.models_py.model_loader", level="INFO") as loaded:
                    owner, loader = load_v41_service_vision(
                        self.config, root, torch.bfloat16, "cpu", method
                    )
                self.assertIn(
                    f"[checkpoint_owner=deepseek_v41_vision] Streamed checkpoint tensors with {method}",
                    "\n".join(loaded.output),
                )
                self.assertIsInstance(owner, OwnedDeepSeekV41VisionEmbedding)
                self.assertEqual(loader.loaded_method, method)
                self.assertTrue(owner.fastsafetensors_load_isolated)
                self.assertFalse(owner.training)
                for name, value in owner.named_parameters():
                    self.assertEqual(value.dtype, self.state[name].dtype)
                    self.assertFalse(value.requires_grad)
                    torch.testing.assert_close(value, self.state[name], rtol=0, atol=0)
                for h, w in ((3, 6), (3, 3)):
                    types = image_token_types(h // 3, w // 3)
                    image = V41ImageInput(
                        start=0,
                        n_vit_h=h,
                        n_vit_w=w,
                        patches=torch.randn(h * w, 3, 14, 14).bfloat16(),
                        types=types,
                        content_sha256="",
                        processor_identity=owner.processor_config.identity,
                    )
                    torch.testing.assert_close(
                        owner.encode_image(image),
                        self.reference.encode_image(image),
                        rtol=0,
                        atol=0,
                    )
                self.assertEqual(owner.encode_prepared_images([]), [])

    def service_mixin(self, model_type, method, root):
        params = SimpleNamespace(config={"v41_config": self.config}, vit_weights=None)
        model = SimpleNamespace(
            model_type=model_type,
            compute_dtype=torch.float16,
            ckpt_path=root,
            mm_model_config=SimpleNamespace(is_multimodal=True),
            mm_related_params=params,
        )
        engine = SimpleNamespace(load_config=SimpleNamespace(load_method=method))
        return MultimodalMixinFactory._create_multimodal_mixin(model, engine, None, "cpu")

    def test_serving_mixin_publishes_direct_owner_without_legacy_weight_dict(self):
        for method in self.methods():
            with tempfile.TemporaryDirectory() as root:
                self.write_checkpoint(root, self.state)
                for kind in ("deepseek_v41", "deepseek_v41_dspark"):
                    with self.subTest(method=method, kind=kind):
                        mixin = self.service_mixin(kind, method, root)
                        self.assertIsInstance(mixin, DeepSeekV41Mixin)
                        self.assertIs(mixin.mm_part, mixin.checkpoint_owner)
                        self.assertEqual(mixin.model_weights_loader.loaded_method, method)
                        self.assertIsNone(mixin.mm_related_params.vit_weights)
                        self.assertFalse(hasattr(mixin, "weights"))
                        self.assertFalse(hasattr(mixin, "mm_mixin_loader"))
                        self.assertEqual(mixin.mm_part.image_start.dtype, torch.bfloat16)
                        torch.testing.assert_close(
                            mixin.mm_part.vision.norm.weight,
                            self.state["vision.norm.weight"],
                            rtol=0,
                            atol=0,
                        )

    def test_missing_required_and_unknown_vision_tensor_fail(self):
        for method in self.methods():
            for absent in ("image_start", "vision.norm.weight", "aligner.w1.weight"):
                with self.subTest(
                    method=method, absent=absent
                ), tempfile.TemporaryDirectory() as root:
                    state = dict(self.state)
                    del state[absent]
                    self.write_checkpoint(root, state)
                    with self.assertRaises((ValueError, RuntimeError)):
                        self.service_mixin("deepseek_v41", method, root)
            with self.subTest(
                method=method, unknown=True
            ), tempfile.TemporaryDirectory() as root:
                self.write_checkpoint(
                    root, dict(self.state, **{"vision.unowned.weight": torch.ones(1)})
                )
                with self.assertRaises((ValueError, RuntimeError, KeyError)):
                    self.service_mixin("deepseek_v41", method, root)


if __name__ == "__main__":
    unittest.main()
