"""DeepSeek-V4.1 vision tests using only backend-private runtime images."""

import hashlib
import importlib.util
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

import torch
from safetensors import safe_open
from torch.nn.attention import SDPBackend, sdpa_kernel

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_vision import (
    RMSNorm,
    _vision_cos_sin,
    apply_rotary,
)
from rtp_llm.utils.base_model_datatypes import MMUrlType

VISION_SHA256 = "5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c"
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41 import (
    deepseek_v41_vision as vision_module,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_processor import (
    num_image_tokens,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_vision import (
    DeepSeekV41VisionEmbedding,
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


def runtime_image(value=0.25):
    return vision_module._RuntimeImage(
        patches=torch.full((9, 3, 14, 14), value, dtype=torch.bfloat16),
        n_vit_h=3,
        n_vit_w=3,
    )


class DeepSeekV41VisionBackendTest(unittest.TestCase):
    def make_adapter(self):
        adapter = DeepSeekV41VisionEmbedding(tiny_config(), device="cpu").eval()
        with torch.no_grad():
            for parameter in adapter.parameters():
                parameter.fill_(0.125)
        return adapter

    def test_real_cpu_forward_returns_complete_canonical_span(self):
        adapter = self.make_adapter()
        image = runtime_image()

        result = adapter.encode_image(image)

        self.assertEqual(tuple(result.shape), (num_image_tokens(1, 1), 8))
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.equal(result[0], adapter.image_start))
        self.assertTrue(torch.equal(result[-2], adapter.image_newline))
        self.assertTrue(torch.equal(result[-1], adapter.image_end))


class V41VisionEmbeddingTest(TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("V4.1 real-weight vision embedding tests require CUDA")
        model = os.environ.get("DSV41_MODEL_PATH")
        if not model:
            raise unittest.SkipTest(
                "DSV41_MODEL_PATH not set; real-weight vision tests run where "
                "the V4.1 checkpoint is available (GB200 lanes / local runs)"
            )
        model = Path(model)
        source = model / "inference/vision.py"
        if hashlib.sha256(source.read_bytes()).hexdigest() != VISION_SHA256:
            raise ValueError(
                "official vision source differs from the pinned HF revision"
            )
        spec = importlib.util.spec_from_file_location(
            "dsv41_official_vision_embedding", source
        )
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        config = V41Config.from_path(model)
        args = SimpleNamespace(
            **config.vision_parameters(), dim=config.text["hidden_size"]
        )
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("cuda"):
                cls.adapter = DeepSeekV41VisionEmbedding(config).eval()
                cls.reference_vision = reference.ViT(args).eval()
                cls.reference_aligner = reference.Aligner(args).eval()
        finally:
            torch.set_default_dtype(previous)
        loaded = cls.adapter.load_checkpoint(model)
        mapping = json.loads((model / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        names = [
            name
            for name in mapping
            if name.startswith(("vision.", "aligner.", "image_"))
        ]
        state = {}
        for shard in sorted({mapping[name] for name in names}):
            with safe_open(model / shard, framework="pt", device="cpu") as reader:
                for name in names:
                    if mapping[name] == shard:
                        state[name] = reader.get_tensor(name)
        cls.reference_vision.load_state_dict(
            {
                name.removeprefix("vision."): tensor
                for name, tensor in state.items()
                if name.startswith("vision.")
            },
            strict=True,
        )
        cls.reference_aligner.load_state_dict(
            {
                name.removeprefix("aligner."): tensor
                for name, tensor in state.items()
                if name.startswith("aligner.")
            },
            strict=True,
        )
        cls.delimiters = {
            name: state[name].to("cuda")
            for name in ("image_start", "image_newline", "image_end")
        }
        print(
            json.dumps(
                {
                    **loaded,
                    "gpu": torch.cuda.get_device_name(),
                    "torch": torch.__version__,
                    "reference_sha256": VISION_SHA256,
                }
            ),
            flush=True,
        )

    @torch.inference_mode()
    def test_framework_binding_preserves_storage_and_fp32_norms(self):
        installed = {
            "v41." + name: tensor for name, tensor in self.adapter.state_dict().items()
        }
        config = V41Config.from_path(Path(os.environ["DSV41_MODEL_PATH"]))
        bound = DeepSeekV41VisionEmbedding.from_model_weights(config, installed)
        for name, value in bound.state_dict().items():
            self.assertEqual(value.data_ptr(), installed["v41." + name].data_ptr())
        for module in bound.vision.modules():
            if isinstance(module, RMSNorm):
                self.assertEqual(module.weight.dtype, torch.float32)
        with sdpa_kernel(SDPBackend.MATH):
            image = vision_module._RuntimeImage(
                torch.zeros(18, 3, 14, 14, dtype=torch.bfloat16), 3, 6
            )
            torch.testing.assert_close(
                bound.encode_image(image),
                self.adapter.encode_image(image),
                rtol=0,
                atol=0,
            )

    @torch.inference_mode()
    def test_full_image_and_delimiters_match_official_math(self):
        with sdpa_kernel(SDPBackend.MATH), torch.device("cuda"):
            for height, width in ((3, 6), (4, 7), (39, 39)):
                with self.subTest(grid=(height, width)):
                    generator = torch.Generator(device="cuda").manual_seed(101 + height)
                    patches = torch.randn(
                        height * width,
                        3,
                        14,
                        14,
                        generator=generator,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    llm_h, llm_w = (height + 2) // 3, (width + 2) // 3
                    image = vision_module._RuntimeImage(patches, height, width)
                    calls = []
                    hook = self.adapter.vision.register_forward_hook(
                        lambda *args: calls.append(1)
                    )
                    try:
                        actual = self.adapter.encode_image(image)
                    finally:
                        hook.remove()
                    self.assertEqual(len(calls), 1)
                    vision = self.reference_vision(patches, height, width)
                    aligned = self.reference_aligner(vision, height, width)
                    parts = [self.delimiters["image_start"].unsqueeze(0)]
                    for row in aligned.reshape(llm_h, llm_w, -1):
                        parts.extend(
                            (row, self.delimiters["image_newline"].unsqueeze(0))
                        )
                    parts.append(self.delimiters["image_end"].unsqueeze(0))
                    expected = torch.cat(parts)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    self.assertTrue(torch.isfinite(actual).all().item())
                    # The normal mixin returns the complete ordered span; token
                    # placement is covered by CanonicalMultimodalProcessorTest.
                    outputs = self.adapter.batched_embedding([image], [MMUrlType.IMAGE])
                    self.assertEqual(len(outputs), 1)
                    torch.testing.assert_close(outputs[0][0], expected, rtol=0, atol=0)

    def test_inference_weights_bind_outside_inference_mode_without_copy(self):
        with torch.inference_mode():
            installed = {
                "v41." + name: value.clone()
                for name, value in self.adapter.state_dict().items()
            }
        self.assertFalse(torch.is_inference_mode_enabled())
        self.assertTrue(all(torch.is_inference(value) for value in installed.values()))
        config = V41Config.from_path(Path(os.environ["DSV41_MODEL_PATH"]))
        bound = DeepSeekV41VisionEmbedding.from_model_weights(config, installed)
        self.assertTrue(all(not value.requires_grad for value in bound.parameters()))
        for name, value in bound.state_dict().items():
            self.assertEqual(value.data_ptr(), installed["v41." + name].data_ptr())
            self.assertEqual(value.dtype, installed["v41." + name].dtype)
        image = vision_module._RuntimeImage(
            torch.zeros(18, 3, 14, 14, dtype=torch.bfloat16), 3, 6
        )
        with sdpa_kernel(SDPBackend.MATH):
            torch.testing.assert_close(
                bound.encode_image(image),
                self.adapter.encode_image(image),
                rtol=0,
                atol=0,
            )

    def test_norm_dtype_and_three_delimiters(self):
        names = set(self.adapter.state_dict())
        self.assertEqual(
            {name for name in names if name.startswith("image_")},
            {"image_start", "image_end", "image_newline"},
        )
        for module in self.adapter.vision.modules():
            if isinstance(module, RMSNorm):
                self.assertEqual(module.weight.dtype, torch.float32)
        self.assertEqual(self.adapter.image_start.dtype, torch.bfloat16)

    @torch.inference_mode()
    def test_rotary_fused_exact_graph_and_fallback(self):
        from rtp_llm.models_py.modules.dsv41._vision_rope_triton import (
            apply_vision_qk_rope,
        )

        for rows in (1, 7, 19, 127, 1521):
            with self.subTest(rows=rows):
                qkv = torch.randn(rows, 3, 16, 64, device="cuda", dtype=torch.bfloat16)
                q, k, v = qkv.unbind(1)
                before = qkv.clone()
                cos, sin = _vision_cos_sin(1, rows, 32, 10000.0, "cuda:0")
                actual = apply_vision_qk_rope(q, k, cos, sin)
                if actual is None:
                    # SDPA math fallback remains the reference path on grids
                    # the fused kernel does not cover.
                    continue
                self.assertIsNotNone(actual)
                for value, source in zip(actual, (q, k)):
                    torch.testing.assert_close(
                        value, apply_rotary(source, cos, sin), rtol=0, atol=0
                    )
                    self.assertNotEqual(
                        value.untyped_storage().data_ptr(),
                        qkv.untyped_storage().data_ptr(),
                    )
                torch.testing.assert_close(qkv, before, rtol=0, atol=0)
                self.assertIsNone(apply_vision_qk_rope(q.clone(), k.clone(), cos, sin))
                self.assertIsNone(apply_vision_qk_rope(q.float(), k.float(), cos, sin))

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = apply_vision_qk_rope(q, k, cos, sin)
        if captured is not None:
            for _ in range(3):
                qkv.normal_()
                graph.replay()
                for value, source in zip(captured, (q, k)):
                    torch.testing.assert_close(
                        value, apply_rotary(source, cos, sin), rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
