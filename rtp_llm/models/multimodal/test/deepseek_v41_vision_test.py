import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest import TestCase, main
from unittest.mock import patch

import torch
from safetensors import safe_open
from torch.nn.attention import SDPBackend, sdpa_kernel

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.models.multimodal.deepseek_v41_processor import (
    TEXT,
    V41ImageInput,
    V41PreparedInputs,
    image_token_types,
)
from rtp_llm.models.multimodal.deepseek_v41_vision import DeepSeekV41VisionEmbedding
from rtp_llm.models.multimodal.deepseek_vision import (
    RMSNorm,
    _vision_cos_sin,
    apply_rotary,
)

VISION_SHA256 = "5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c"


class V41VisionEmbeddingTest(TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("V4.1 real-weight vision embedding tests require CUDA")
        model = Path(os.environ["DSV41_MODEL_PATH"])
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
            image = V41ImageInput(
                0,
                torch.zeros(18, 3, 14, 14, dtype=torch.bfloat16),
                3,
                6,
                image_token_types(1, 2),
                "",
                bound.processor_config.identity,
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
                    image = V41ImageInput(
                        5,
                        patches,
                        height,
                        width,
                        image_token_types(llm_h, llm_w),
                        "",
                        self.adapter.processor_config.identity,
                    )
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
                    prepared = V41PreparedInputs(
                        "",
                        (0,) * 5 + (129264,) * image.length + (1,) * 3,
                        (TEXT,) * 5 + tuple(image.types.tolist()) + (TEXT,) * 3,
                        (image,),
                    )
                    embeddings = torch.zeros(
                        len(prepared.token_ids),
                        actual.shape[1],
                        dtype=torch.bfloat16,
                        device="cuda",
                    )
                    calls = []
                    hook = self.adapter.vision.register_forward_hook(
                        lambda *args: calls.append(1)
                    )
                    try:
                        result = self.adapter.inject_embeddings(embeddings, prepared)
                    finally:
                        hook.remove()
                    self.assertEqual(len(calls), 1)
                    torch.testing.assert_close(
                        result[5 : 5 + image.length], expected, rtol=0, atol=0
                    )
                    self.assertEqual(torch.count_nonzero(result[:5]).item(), 0)
                    self.assertEqual(torch.count_nonzero(result[-3:]).item(), 0)

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
        image = V41ImageInput(
            0,
            torch.zeros(18, 3, 14, 14, dtype=torch.bfloat16),
            3,
            6,
            image_token_types(1, 2),
            "",
            bound.processor_config.identity,
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
    def test_fa4_shape_and_unavailable_fallback(self):
        from rtp_llm.models_py.modules.dsv41 import _vision_fa4 as fa4

        with patch.dict(os.environ, {"DSV41_VISION_FA4": "1"}):
            for rows in (0, 1, 1521, 4070, 8648, 8650):
                qk = torch.empty(rows, 32, 64, device="cuda", dtype=torch.bfloat16)
                qkv = torch.empty(rows, 3, 16, 64, device="cuda", dtype=torch.bfloat16)
                with patch.object(fa4, "_load_fa4", side_effect=AssertionError):
                    self.assertIsNone(
                        fa4.vision_attention_fa4(qk[:, :16], qk[:, 16:], qkv[:, 2])
                    )

            qk = torch.empty(8649, 32, 64, device="cuda", dtype=torch.bfloat16)
            qkv = torch.empty(8649, 3, 16, 64, device="cuda", dtype=torch.bfloat16)
            q, k, v = qk[:, :16], qk[:, 16:], qkv[:, 2]
            self.assertTrue(fa4._enabled_or_supported(q, k, v))
            self.assertFalse(fa4._enabled_or_supported(q.clone(), k, v))
            self.assertFalse(fa4._enabled_or_supported(q.float(), k, v))
            with torch.enable_grad():
                self.assertIsNone(fa4.vision_attention_fa4(q, k, v))
            with patch.dict(os.environ, {"DSV41_VISION_FA4": "0"}):
                self.assertIsNone(fa4.vision_attention_fa4(q, k, v))

            fa4._load_fa4.cache_clear()
            try:
                with patch.dict(sys.modules, {"flash_attn.cute.interface": None}):
                    with self.assertLogs(fa4.__name__, level="WARNING") as records:
                        self.assertIsNone(fa4.vision_attention_fa4(q, k, v))
                        status = fa4.vision_fa4_status()
                    self.assertFalse(status["available"])
                    self.assertIn("ModuleNotFoundError", status["unavailable_reason"])
                    self.assertEqual(len(records.output), 1)
            finally:
                fa4._load_fa4.cache_clear()

    @torch.inference_mode()
    def test_fa4_math_and_dynamic_graph(self):
        from rtp_llm.models_py.modules.dsv41 import _vision_fa4 as fa4

        with patch.dict(os.environ, {"DSV41_VISION_FA4": "1"}):
            status = fa4.vision_fa4_status()
            if not status["available"]:
                self.skipTest(status["unavailable_reason"])
            generator = torch.Generator(device="cuda").manual_seed(941)
            qk = torch.randn(
                8649, 32, 64, device="cuda", dtype=torch.bfloat16, generator=generator
            )
            qkv = torch.randn(
                8649,
                3,
                16,
                64,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            q, k, v = qk[:, :16], qk[:, 16:], qkv[:, 2]
            before = qkv.clone()
            actual = fa4.vision_attention_fa4(q, k, v)
            with sdpa_kernel(SDPBackend.MATH):
                expected = (
                    torch.nn.functional.scaled_dot_product_attention(
                        q.float().transpose(0, 1).unsqueeze(0),
                        k.float().transpose(0, 1).unsqueeze(0),
                        v.float().transpose(0, 1).unsqueeze(0),
                    )
                    .squeeze(0)
                    .transpose(0, 1)
                )
            torch.testing.assert_close(actual.float(), expected, rtol=1e-2, atol=2e-3)
            torch.testing.assert_close(qkv, before, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = fa4.vision_attention_fa4(q, k, v)
            for _ in range(2):
                qk.normal_(generator=generator)
                qkv.normal_(generator=generator)
                graph.replay()
                torch.testing.assert_close(
                    captured, fa4.vision_attention_fa4(q, k, v), rtol=0, atol=0
                )

    @torch.inference_mode()
    def test_rotary_fused_exact_graph_and_fallback(self):
        from rtp_llm.models_py.modules.dsv41._vision_rope_triton import (
            apply_vision_qk_rope,
        )

        with patch.dict(os.environ, {"DSV41_VISION_ROPE": "1"}):
            for rows in (1, 7, 19, 127, 1521):
                with self.subTest(rows=rows):
                    qkv = torch.randn(
                        rows, 3, 16, 64, device="cuda", dtype=torch.bfloat16
                    )
                    q, k, v = qkv.unbind(1)
                    before = qkv.clone()
                    cos, sin = _vision_cos_sin(1, rows, 32, 10000.0, "cuda:0")
                    actual = apply_vision_qk_rope(q, k, cos, sin)
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
                    self.assertIsNone(
                        apply_vision_qk_rope(q.clone(), k.clone(), cos, sin)
                    )
                    self.assertIsNone(
                        apply_vision_qk_rope(q.float(), k.float(), cos, sin)
                    )

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = apply_vision_qk_rope(q, k, cos, sin)
            for _ in range(3):
                qkv.normal_()
                graph.replay()
                for value, source in zip(captured, (q, k)):
                    torch.testing.assert_close(
                        value, apply_rotary(source, cos, sin), rtol=0, atol=0
                    )
            with patch.dict(os.environ, {"DSV41_VISION_ROPE": "0"}):
                self.assertIsNone(apply_vision_qk_rope(q, k, cos, sin))


if __name__ == "__main__":
    main()
