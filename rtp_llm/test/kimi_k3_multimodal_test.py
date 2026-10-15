"""K3 image-token and MoonViT contracts without a text-model forward."""

import json
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import ProfilingDebugLoggingConfig, VitConfig
from rtp_llm.model_loader.load_config import LoadMethod
from rtp_llm.model_loader.multimodal_mixin_loader import MultimodalMixinLoader
from rtp_llm.multimodal.mm_process_engine import MMProcessEngine
from rtp_llm.multimodal.mm_scheduler import MMSchedulerExecutionError
from rtp_llm.multimodal.multimodal_mixin_register import get_multimodal_mixin_cls
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseMultiModalDeployWeightInfo,
    BaseMultiModalMixin,
    BaseVitWeights,
)
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_config import (
    configure_kimi_k3_multimodal,
)
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_image_processor import (
    KimiK3VisionProcessor,
)
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_mixin import KimiK3Mixin
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_vit import (
    KimiK3ImageEmbedding,
)
from rtp_llm.multimodal.multimodal_mixins.multimodal_common import MMWorkEstimate
from rtp_llm.utils.base_model_datatypes import MMUrlType

_CONFIG_PATH = Path(__file__).parent / "testdata/kimi_k3/preprocessor_config.json"


def _media_config():
    return json.loads(_CONFIG_PATH.read_text())["media_proc_cfg"]


def _tiny_image_embedding(media_config=None):
    return KimiK3ImageEmbedding(
        SimpleNamespace(
            config={
                "vision_config": {
                    "vt_hidden_size": 8,
                    "qkv_hidden_size": 8,
                    "vt_intermediate_size": 16,
                    "vt_num_attention_heads": 1,
                    "vt_num_hidden_layers": 1,
                    "text_hidden_size": 16,
                    "init_pos_emb_height": 2,
                    "init_pos_emb_width": 2,
                },
                "media_proc_cfg": (
                    _media_config() if media_config is None else media_config
                ),
            }
        )
    )


class _ProbeDatabase:
    source = torch.tensor([1e-8], dtype=torch.bfloat16)

    def load_tensor(self, name, data_type=torch.float16):
        if name != "model.weight":
            raise AssertionError(name)
        return [self.source.to(data_type)]


class _ProbeMixinBehavior:
    def _init_multimodal(self):
        self.mm_part = torch.nn.Linear(1, 1, bias=False)
        self.mm_related_params.vit_weights = BaseVitWeights({"probe": self.mm_part})

    def create_mm_mixin_loader(self):
        weight_info = BaseMultiModalDeployWeightInfo(
            self.vit_config, self.mm_related_params.vit_weights
        )
        return MultimodalMixinLoader(weight_info, _ProbeDatabase())


class _LegacyPrecisionProbe(_ProbeMixinBehavior, BaseMultiModalMixin):
    pass


class _K3PrecisionProbe(_ProbeMixinBehavior, KimiK3Mixin):
    pass


class KimiK3MultimodalTest(unittest.TestCase):
    def test_k3_preserves_bf16_checkpoint_without_changing_other_mixins(self) -> None:
        def load(mixin_class):
            mixin = mixin_class(
                torch.bfloat16,
                "cpu",
                SimpleNamespace(vit_weights=None),
                LoadMethod.AUTO,
                VitConfig(),
                "unused",
            )
            return mixin.mm_part.weight.detach().flatten()

        # BF16 represents this value, while the legacy FP16 staging dtype
        # rounds it to zero. Only the K3 loader should preserve it.
        torch.testing.assert_close(load(_K3PrecisionProbe), _ProbeDatabase.source)
        torch.testing.assert_close(
            load(_LegacyPrecisionProbe),
            torch.zeros(1, dtype=torch.bfloat16),
        )

    def test_checkpoint_geometry_and_pixel_values(self) -> None:
        processor = KimiK3VisionProcessor(_media_config())
        image = Image.new("RGB", (28, 28), (128, 64, 32))
        processed = processor.preprocess({"image": image}, return_tensors="pt")

        self.assertEqual(tuple(processed.pixel_values.shape), (4, 3, 14, 14))
        self.assertEqual(processed.grid_thws.tolist(), [[1, 2, 2]])
        self.assertEqual(processor.media_tokens_calculator({"image": image}), 1)
        expected = torch.tensor([2 * value / 255 - 1 for value in (128, 64, 32)])
        torch.testing.assert_close(processed.pixel_values[0, :, 0, 0], expected)
        self.assertEqual(
            processor.make_image_prompt(28, 28),
            "<|media_begin|>image 28x28<|media_content|><|media_pad|><|media_end|>",
        )

    def test_transparent_pixels_use_configured_background(self) -> None:
        media_config = _media_config()
        media_config["transparent_bg_config"] = {"pattern": "black"}
        media_config["transparent_bg_fill_stage"] = "before_resize"
        image = Image.new("RGBA", (28, 28), (255, 0, 0, 0))
        processed = KimiK3VisionProcessor(media_config).preprocess({"image": image})
        torch.testing.assert_close(
            processed.pixel_values,
            torch.full((4, 3, 14, 14), -1.0),
        )

    def test_config_binds_single_placeholder_and_registered_mixin(self) -> None:
        self.assertIs(get_multimodal_mixin_cls("kimi_k3"), KimiK3Mixin)
        config = ModelConfig()
        config.ckpt_path = str(_CONFIG_PATH.parent)
        configure_kimi_k3_multimodal(
            config,
            {
                "vision_config": {"vt_hidden_size": 8, "_name_or_path": "unused"},
                "media_placeholder_token_id": 163605,
                "image_placeholder": "<|kimi_image_placeholder|>",
            },
        )
        self.assertTrue(config.mm_model_config.is_multimodal)
        self.assertEqual(config.mm_model_config.mm_sep_tokens, [[163605]])
        self.assertEqual(
            config.mm_related_params.special_token_ids["image_token_index"], 163605
        )
        self.assertNotIn("_name_or_path", config.mm_related_params.config["vision_config"])
        self.assertEqual(config.mm_related_params.config["media_proc_cfg"]["patch_size"], 14)

    def test_image_prompt_embeds_text_on_both_sides_of_vision(self) -> None:
        embedding = object.__new__(KimiK3ImageEmbedding)
        embedding.image_processor = KimiK3VisionProcessor(_media_config())
        embedding._word_embedding_weight = torch.arange(80, dtype=torch.float32).reshape(20, 4)
        prompt = embedding.image_processor.make_image_prompt(28, 28)
        embedding._tokenizer = SimpleNamespace(
            encode=lambda text: [3, 4, 7, 5] if text == prompt else [7]
        )
        features = torch.tensor([[100.0] * 4, [200.0] * 4])

        result = embedding._assemble_image(Image.new("RGB", (28, 28)), features)
        expected = torch.cat(
            [embedding._word_embedding_weight[[3, 4]], features, embedding._word_embedding_weight[[5]]]
        )
        torch.testing.assert_close(result, expected)

    def test_direct_image_bytes_honor_size_cap_and_media_type(self) -> None:
        output = BytesIO()
        Image.new("RGB", (28, 28)).save(output, format="PNG")
        image_bytes = output.getvalue()
        mm_input = SimpleNamespace(
            mm_type=MMUrlType.IMAGE,
            tensor=torch.tensor(list(image_bytes), dtype=torch.uint8),
            url="",
        )
        vit_config = VitConfig()
        vit_config.mm_image_max_file_size_kb = 1
        image = KimiK3ImageEmbedding.preprocess_input([mm_input], vit_config)
        self.assertEqual(image.size, (28, 28))
        mm_input.tensor = torch.tensor(
            list(image_bytes + b"0" * 1024), dtype=torch.uint8
        )
        with self.assertRaisesRegex(ValueError, "per-image limit"):
            KimiK3ImageEmbedding.preprocess_input([mm_input], vit_config)
        mm_input.mm_type = MMUrlType.VIDEO
        with self.assertRaisesRegex(ValueError, "only supports image"):
            KimiK3ImageEmbedding.preprocess_input([mm_input], vit_config)

    def test_tiny_moonvit_projects_one_merged_patch(self) -> None:
        torch.manual_seed(1)
        embedding = KimiK3ImageEmbedding(
            SimpleNamespace(
                config={
                    "vision_config": {
                        "vt_hidden_size": 8,
                        "qkv_hidden_size": 8,
                        "vt_intermediate_size": 16,
                        "vt_num_attention_heads": 1,
                        "vt_num_hidden_layers": 1,
                        "text_hidden_size": 16,
                        "init_pos_emb_height": 2,
                        "init_pos_emb_width": 2,
                    },
                    "media_proc_cfg": _media_config(),
                }
            )
        )
        embedding.vision_tower.eval()
        embedding.mm_projector.eval()
        image = Image.new("RGB", (28, 28))
        features = embedding.image_embedding([image, image])
        serial = embedding.image_embedding([image])[0]
        self.assertEqual(len(features), 2)
        for one_image in features:
            self.assertEqual(tuple(one_image.shape), (1, 16))
            self.assertTrue(torch.isfinite(one_image).all().item())
            torch.testing.assert_close(one_image, serial)


class KimiK3VitWorkEstimateTest(unittest.TestCase):
    def setUp(self):
        self.embedding = _tiny_image_embedding()

    def test_work_matches_padded_patch_geometry(self):
        for size, patches, tokens, workspace in (
            ((28, 28), 4, 1, 11328),
            ((56, 28), 8, 2, 22656),
            ((448, 224), 512, 128, 1449984),
            ((31, 17), 8, 2, 22656),
        ):
            with self.subTest(size=size):
                work = self.embedding.estimate_work(
                    Image.new("RGB", size), MMUrlType.IMAGE
                )
                self.assertEqual(
                    work,
                    MMWorkEstimate(
                        input_patches=patches,
                        output_tokens=tokens,
                        estimated_workspace_bytes=workspace,
                        max_attention_segment=patches,
                        attention_work=patches**2,
                    ),
                )

    def test_budget_scales_additive_costs_but_not_attention_segment(self):
        self.embedding.image_processor.media_proc_cfg["in_patch_limit"] = 16
        self.assertEqual(
            self.embedding.get_batch_work_budget(3),
            MMWorkEstimate(
                input_patches=48,
                output_tokens=12,
                estimated_workspace_bytes=135936,
                max_attention_segment=16,
                attention_work=768,
            ),
        )

    def test_workspace_uses_actual_compute_dtype(self):
        self.embedding.vision_tower.to(dtype=torch.bfloat16)
        work = self.embedding.estimate_work(Image.new("RGB", (28, 28)))
        self.assertIsNotNone(work)
        self.assertEqual(work.estimated_workspace_bytes, 5664)

    def test_workspace_matches_default_qkv_width(self):
        # MoonViT treats an unset QKV width as the vision hidden size.
        self.embedding.vision_config.qkv_hidden_size = None
        work = self.embedding.estimate_work(Image.new("RGB", (28, 28)))
        self.assertEqual(work.estimated_workspace_bytes, 11328)

    def test_serial_budget_remains_unbounded(self):
        self.assertIsNone(self.embedding.get_batch_work_budget(1 << 30))

    def test_invalid_patch_budget_is_rejected(self):
        for limit in (0, -1):
            with self.subTest(limit=limit):
                self.embedding.image_processor.media_proc_cfg["in_patch_limit"] = limit
                with self.assertRaisesRegex(ValueError, "in_patch_limit"):
                    self.embedding.get_batch_work_budget(2)

    def test_work_rejects_non_image_inputs(self):
        with self.assertRaises(TypeError):
            self.embedding.estimate_work(torch.zeros(1))
        for mm_type in (MMUrlType.VIDEO, MMUrlType.AUDIO):
            with self.subTest(mm_type=mm_type):
                with self.assertRaisesRegex(ValueError, "only supports image"):
                    self.embedding.estimate_work(Image.new("RGB", (28, 28)), mm_type)


class KimiK3VitChunkSchedulingTest(unittest.TestCase):
    def _engine(self, *, media_config=None, max_batch_images=2):
        embedding = _tiny_image_embedding(media_config)
        embedding.vision_tower.eval()
        embedding.mm_projector.eval()
        # Substitute only checkpoint I/O; use real preprocessing, MoonViT,
        # projector, prompt assembly, hashing, engine and scheduler.
        embedding._word_embedding_weight = torch.arange(320).reshape(20, 16).float()
        embedding._tokenizer = SimpleNamespace(
            encode=lambda text: [7] if text == "<|media_pad|>" else [3, 4, 7, 5]
        )
        model_config = ModelConfig()
        model_config.mm_related_params.support_batch = True
        vit_config = VitConfig()
        vit_config.use_local_preprocess = True
        vit_config.disable_access_log = True
        vit_config.mm_cache_item_num = 0
        vit_config.mm_cache_gpu_max_bytes = 0
        vit_config.mm_cache_cpu_max_bytes = 0
        vit_config.mm_hash_key_cache_max_bytes = 0
        vit_config.gpu_batch_wait_ms = 0
        vit_config.gpu_max_batch_size = 2
        vit_config.gpu_max_batch_images = max_batch_images
        engine = MMProcessEngine(
            embedding,
            model_config,
            vit_config,
            ProfilingDebugLoggingConfig(),
            device="cpu",
        )
        self.addCleanup(engine.stop)
        return engine

    @staticmethod
    def _run_images(engine, images):
        tensors = []
        for image in images:
            output = BytesIO()
            image.save(output, format="PNG")
            tensors.append(torch.tensor(list(output.getvalue()), dtype=torch.uint8))
        return engine.mm_embedding_cpp(
            [""] * len(images),
            [MMUrlType.IMAGE] * len(images),
            tensors,
            [[-1, -1, -1, -1, -1, -1, -1, [], 5000] for _ in images],
        )

    def test_multi_image_request_chunks_and_preserves_full_feature_order(self):
        engine = self._engine()
        images = [
            Image.new("RGB", (width, 28), (20 * index, 64, 128))
            for index, width in enumerate((28, 56, 84))
        ]
        forward_grids = []
        hook = engine.mm_part.vision_tower.register_forward_pre_hook(
            lambda module, args: forward_grids.append(args[1].tolist())
        )
        try:
            result = self._run_images(engine, images)
        finally:
            hook.remove()
        self.assertEqual(forward_grids, [[[1, 2, 2], [1, 2, 4]], [[1, 2, 6]]])
        self.assertEqual([feature.shape[0] for feature in result.embeddings], [4, 5, 6])
        expected = [engine.mm_part.embedding(image)[0] for image in images]
        for actual, reference in zip(result.embeddings, expected):
            torch.testing.assert_close(actual, reference)

    def test_padded_single_image_over_budget_runs_alone(self):
        media_config = _media_config()
        media_config["in_patch_limit"] = 16
        engine = self._engine(media_config=media_config)
        # 113x28 has 16 unpadded patches but 20 after merge-aligned padding.
        # It exceeds the segment budget, so even a two-image batch must split.
        images = [Image.new("RGB", (113, 28)), Image.new("RGB", (28, 28))]
        forward_grids = []
        hook = engine.mm_part.vision_tower.register_forward_pre_hook(
            lambda module, args: forward_grids.append(args[1].tolist())
        )
        try:
            result = self._run_images(engine, images)
        finally:
            hook.remove()
        self.assertEqual(forward_grids, [[[1, 2, 10]], [[1, 2, 2]]])
        self.assertEqual([feature.shape[0] for feature in result.embeddings], [8, 4])

    def test_failed_chunk_does_not_execute_remaining_images(self):
        engine = self._engine()
        images = [Image.new("RGB", (28 * index, 28)) for index in range(1, 6)]
        forward_grids = []

        def fail_second_forward(module, args):
            forward_grids.append(args[1].tolist())
            if len(forward_grids) == 2:
                raise RuntimeError("K3 test forward failed")

        hook = engine.mm_part.vision_tower.register_forward_pre_hook(
            fail_second_forward
        )
        try:
            with self.assertRaisesRegex(
                MMSchedulerExecutionError, "K3 test forward failed"
            ):
                self._run_images(engine, images)
        finally:
            hook.remove()
        self.assertEqual(
            forward_grids,
            [[[1, 2, 2], [1, 2, 4]], [[1, 2, 6], [1, 2, 8]]],
        )
        # A failed request must not poison later requests or retain its continuation.
        result = self._run_images(engine, [images[0]])
        self.assertEqual(len(result.embeddings), 1)


if __name__ == "__main__":
    unittest.main()
