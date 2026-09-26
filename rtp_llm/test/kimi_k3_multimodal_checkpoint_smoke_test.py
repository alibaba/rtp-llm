"""Checkpoint-backed K3 image prompt assembly without text-model execution."""

import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image
from safetensors import safe_open

from rtp_llm.config.model_args import ModelArgs
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import ProfilingDebugLoggingConfig, VitConfig
from rtp_llm.model_factory import ModelFactory
from rtp_llm.model_loader.load_config import LoadMethod
from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3
from rtp_llm.multimodal.multimodal_mixin_factory import MultimodalMixinFactory
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_config import (
    configure_kimi_k3_multimodal,
)
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_image_processor import (
    KimiK3VisionProcessor,
)
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_vit import (
    KimiK3ImageEmbedding,
)
from rtp_llm.utils.base_model_datatypes import MMUrlType


class KimiK3MultimodalCheckpointSmokeTest(unittest.TestCase):
    def test_factory_weights_and_cpp_embedding_entry(self) -> None:
        """Exercise the production mixin loader and C++-facing image entry."""
        checkpoint = Path(os.environ["K3_CKPT_PATH"])
        top_config = json.loads((checkpoint / "config.json").read_text())
        model_config = ModelConfig()
        model_config.model_type = "kimi_k3"
        model_config.ckpt_path = str(checkpoint)
        model_config.data_type = "bf16"
        configure_kimi_k3_multimodal(model_config, top_config)

        vit_config = VitConfig()
        vit_config.use_local_preprocess = True
        vit_config.disable_access_log = True
        vit_config.mm_cache_item_num = 0
        engine_config = SimpleNamespace(
            load_config=SimpleNamespace(load_method=LoadMethod.AUTO),
            profiling_debug_logging_config=ProfilingDebugLoggingConfig(),
        )
        engine = MultimodalMixinFactory.create_multimodal_process_engine(
            model_config, engine_config, vit_config, device="cuda:0"
        )
        self.addCleanup(engine.stop)

        # Compare the weights used by the factory with the checkpoint itself.
        # A BF16 checkpoint must not be rounded through an FP16 staging dtype.
        weight_map = json.loads(
            (checkpoint / "model.safetensors.index.json").read_text()
        )["weight_map"]
        for name, actual in (
            (
                "vision_tower.patch_embed.proj.weight",
                engine.mm_part.vision_tower.patch_embed.proj.weight,
            ),
            (
                "mm_projector.proj.0.weight",
                engine.mm_part.mm_projector.proj[0].weight,
            ),
        ):
            with self.subTest(weight=name):
                with safe_open(
                    checkpoint / weight_map[name], framework="pt", device="cpu"
                ) as shard:
                    expected = shard.get_tensor(name)
                torch.testing.assert_close(
                    actual.detach().cpu(), expected, rtol=0, atol=0
                )

        images = [
            Image.new("RGB", (28, 28), (128, 64, 32)),
            Image.new("RGB", (56, 28), (32, 64, 128)),
            Image.new("RGB", (448, 224), (64, 128, 32)),
        ]
        image_bytes = []
        for image in images:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_bytes.append(torch.tensor(list(buffer.getvalue()), dtype=torch.uint8))
        result = engine.mm_embedding_cpp(
            [""] * len(images),
            [MMUrlType.IMAGE] * len(images),
            image_bytes,
            [[-1, -1, -1, -1, -1, -1, -1, [], 30000] for _ in images],
        )
        self.assertEqual(len(result.embeddings), len(images))
        self.assertEqual(result.position_ids, [])
        self.assertEqual(result.extra_input, [])

        # With 14px patches and 2x2 merging, these image sizes produce
        # 1, 2 and 128 vision tokens. Each replaces one media pad inside the
        # checkpoint's image prompt; the surrounding rows are text embeddings.
        for image, feature, vision_tokens in zip(
            images, result.embeddings, (1, 2, 128)
        ):
            prompt_ids = engine.mm_part._tokenizer.encode(
                engine.mm_part.image_processor.make_image_prompt(*image.size)
            )
            pad_id = engine.mm_part._tokenizer.encode("<|media_pad|>")[0]
            pad_index = prompt_ids.index(pad_id)
            self.assertEqual(prompt_ids.count(pad_id), 1)
            self.assertEqual(
                tuple(feature.shape), (len(prompt_ids) - 1 + vision_tokens, 7168)
            )
            self.assertTrue(torch.isfinite(feature).all().item())
            word_embeddings = engine.mm_part._word_embedding_weight
            torch.testing.assert_close(
                feature[:pad_index].cpu(),
                word_embeddings[prompt_ids[:pad_index]].to(feature.dtype),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                feature[pad_index + vision_tokens :].cpu(),
                word_embeddings[prompt_ids[pad_index + 1 :]].to(feature.dtype),
                rtol=0,
                atol=0,
            )

        # Feed the same checkpoint-backed tensors into the production C++
        # placeholder expansion. This crosses the Python/C++ feature boundary
        # without requiring the separately owned K3 text-model forward.
        binary = (
            Path(os.environ["TEST_SRCDIR"])
            / os.environ["TEST_WORKSPACE"]
            / "rtp_llm/cpp/multimodal_processor/test/kimi_k3_feature_expand_check"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            media_pad_id = model_config.mm_model_config.mm_sep_tokens[0][0]
            token_ids = torch.tensor(
                [17, media_pad_id, 18, media_pad_id, 19, media_pad_id, 20],
                dtype=torch.int32,
            )
            token_path = Path(temp_dir) / "token_ids.raw"
            token_path.write_bytes(token_ids.numpy().tobytes())
            paths = [
                Path(temp_dir) / f"image_{index}.raw" for index in range(len(images))
            ]
            for path, feature in zip(paths, result.embeddings):
                raw = feature.detach().cpu().contiguous().view(torch.uint8)
                path.write_bytes(raw.numpy().tobytes())
            command = [
                str(binary),
                str(media_pad_id),
                str(token_path),
                str(token_ids.numel()),
            ]
            for index, (path, feature) in enumerate(zip(paths, result.embeddings)):
                command.extend(
                    [
                        f"image-{index}",
                        str(int(MMUrlType.IMAGE)),
                        str(path),
                        str(feature.shape[0]),
                    ]
                )
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_model_factory_binds_k3_vision_from_checkpoint(self) -> None:
        checkpoint = Path(os.environ["K3_CKPT_PATH"])
        top_config = json.loads((checkpoint / "config.json").read_text())

        # Keep the real model-owned checkpoint parser; isolate only the engine
        # configuration stage, which is outside this image-only smoke.
        class K3ConfigOnlyModel(KimiK3):
            @staticmethod
            def _apply_kv_cache_config(config, kv_cache_config):
                pass

            @staticmethod
            def _post_build_model_config(config):
                pass

        def build_text_config(**kwargs):
            config = kwargs["model_config"]
            args = kwargs["model_args"]
            config.ckpt_path = args.ckpt_path
            config.model_type = args.model_type

        args = ModelArgs()
        args.ckpt_path = str(checkpoint)
        args.model_type = "kimi_k3"
        with patch.object(
            ModelFactory, "get_model_cls", return_value=K3ConfigOnlyModel
        ):
            with patch(
                "rtp_llm.model_factory.build_model_config",
                side_effect=build_text_config,
            ):
                config = ModelFactory.create_model_config(
                    args,
                    lora_config=SimpleNamespace(lora_info=""),
                    kv_cache_config=object(),
                    profiling_debug_logging_config=object(),
                )

        self.assertTrue(config.mm_model_config.is_multimodal)
        self.assertEqual(
            config.mm_model_config.mm_sep_tokens,
            [[top_config["media_placeholder_token_id"]]],
        )
        self.assertEqual(
            config.mm_related_params.config["media_proc_cfg"],
            json.loads((checkpoint / "preprocessor_config.json").read_text())[
                "media_proc_cfg"
            ],
        )
        self.assertEqual(
            config.mm_related_params.special_tokens["image_placeholder"],
            top_config["image_placeholder"],
        )

    def test_real_moonvit_and_projector_batch_match_serial(self) -> None:
        checkpoint = Path(os.environ["K3_CKPT_PATH"])
        top_config = json.loads((checkpoint / "config.json").read_text())
        model_config = ModelConfig()
        model_config.ckpt_path = str(checkpoint)
        configure_kimi_k3_multimodal(model_config, top_config)
        embedding = KimiK3ImageEmbedding(model_config.mm_related_params)

        weight_map = json.loads(
            (checkpoint / "model.safetensors.index.json").read_text()
        )["weight_map"]
        vision_state = {}
        projector_state = {}
        vision_shards = {
            name
            for key, name in weight_map.items()
            if key.startswith(("vision_tower.", "mm_projector."))
        }
        for shard_name in sorted(vision_shards):
            with safe_open(
                checkpoint / shard_name, framework="pt", device="cpu"
            ) as shard:
                for key in shard.keys():
                    if key.startswith("vision_tower."):
                        vision_state[key.removeprefix("vision_tower.")] = (
                            shard.get_tensor(key)
                        )
                    elif key.startswith("mm_projector."):
                        projector_state[key.removeprefix("mm_projector.")] = (
                            shard.get_tensor(key)
                        )
        embedding.vision_tower.load_state_dict(vision_state, strict=True)
        embedding.mm_projector.load_state_dict(projector_state, strict=True)
        self.assertEqual(len(vision_state), 165)
        self.assertEqual(len(projector_state), 3)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        embedding.vision_tower.to(device=device, dtype=torch.bfloat16).eval()
        embedding.mm_projector.to(device=device, dtype=torch.bfloat16).eval()
        image_a = Image.new("RGB", (28, 28), (128, 64, 32))
        image_b = Image.new("RGB", (56, 28), (32, 64, 128))
        batched = embedding.image_embedding([image_a, image_b])
        separate = [
            embedding.image_embedding([image])[0] for image in (image_a, image_b)
        ]
        self.assertEqual(
            [tuple(value.shape) for value in batched], [(1, 7168), (2, 7168)]
        )
        for actual, expected in zip(batched, separate):
            self.assertTrue(torch.isfinite(actual).all().item())
            # BF16 matmul accumulates in a different order for batched and
            # single-image shapes; constrain both typical and worst-case error.
            error = (actual.float() - expected.float()).abs()
            self.assertLess(error.mean().item(), 0.005)
            self.assertLess(error.max().item(), 0.06)

    def test_native_image_prompt_and_text_embedding_shard(self) -> None:
        checkpoint = Path(os.environ["K3_CKPT_PATH"])
        top_config = json.loads((checkpoint / "config.json").read_text())
        model_config = ModelConfig()
        model_config.ckpt_path = str(checkpoint)
        configure_kimi_k3_multimodal(model_config, top_config)
        self.assertTrue(model_config.mm_model_config.is_multimodal)

        embedding = object.__new__(KimiK3ImageEmbedding)
        embedding.vision_config = SimpleNamespace(
            text_hidden_size=top_config["vision_config"]["text_hidden_size"]
        )
        embedding.image_processor = KimiK3VisionProcessor(
            model_config.mm_related_params.config["media_proc_cfg"]
        )
        embedding._ckpt_path = str(checkpoint)
        embedding._tokenizer = None
        embedding._word_embedding_weight = None
        embedding._ensure_text_embeddings()

        image = Image.new("RGB", (28, 28))
        expected_visual_tokens = embedding.image_processor.media_tokens_calculator(
            {"image": image}
        )
        self.assertEqual(expected_visual_tokens, 1)
        hidden_size = embedding.vision_config.text_hidden_size
        features = torch.full(
            (expected_visual_tokens, hidden_size),
            0.125,
            dtype=embedding._word_embedding_weight.dtype,
        )
        prompt = embedding.image_processor.make_image_prompt(*image.size)
        prompt_ids = embedding._tokenizer.encode(prompt)
        pad_ids = embedding._tokenizer.encode("<|media_pad|>")
        self.assertEqual(len(pad_ids), 1)
        self.assertEqual(prompt_ids.count(pad_ids[0]), 1)

        assembled = embedding._assemble_image(image, features)
        pad_index = prompt_ids.index(pad_ids[0])
        self.assertEqual(tuple(assembled.shape), (len(prompt_ids), hidden_size))
        torch.testing.assert_close(assembled[pad_index : pad_index + 1], features)
        torch.testing.assert_close(
            assembled[:pad_index], embedding._word_embedding_weight[prompt_ids[:pad_index]]
        )
        torch.testing.assert_close(
            assembled[pad_index + 1 :],
            embedding._word_embedding_weight[prompt_ids[pad_index + 1 :]],
        )


if __name__ == "__main__":
    unittest.main()
