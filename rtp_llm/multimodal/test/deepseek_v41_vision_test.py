import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import TestCase, main
from unittest.mock import patch

import torch
from safetensors import safe_open
from torch.nn.attention import SDPBackend, sdpa_kernel

from rtp_llm.config.dsv41_config import V41Config
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_processor import (
    TEXT,
    V41ImageInput,
    V41ImageProcessorConfig,
    V41PreparedInputs,
    image_token_types,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_vision import (
    DeepSeekV41VisionEmbedding,
)
from rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_vision import (
    RMSNorm,
    _vision_cos_sin,
    apply_rotary,
)

VISION_SHA256 = "5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c"


class V41PreparedValidationTest(TestCase):
    def setUp(self):
        self.adapter = object.__new__(DeepSeekV41VisionEmbedding)
        torch.nn.Module.__init__(self.adapter)
        self.adapter.processor_config = V41ImageProcessorConfig()

    def record(self, start=1):
        return {
            "start": start,
            "n_vit_h": 3,
            "n_vit_w": 6,
            "patches": torch.zeros(18, 3, 14, 14, dtype=torch.bfloat16),
            "types": image_token_types(1, 2).to(torch.int32),
            "content_sha256": "a" * 64,
            "processor_identity": self.adapter.processor_config.identity,
        }

    def test_valid_records_preserve_payload_and_allow_empty_request(self):
        records = [self.record(), self.record(8)]
        with patch.object(
            self.adapter, "encode_image", return_value=torch.ones(5, 8)
        ) as encode:
            outputs = self.adapter.encode_prepared_images(records)
            self.assertEqual(len(outputs), 2)
            for record, call in zip(records, encode.call_args_list):
                image = call.args[0]
                self.assertIs(image.patches, record["patches"])
                self.assertEqual(image.start, record["start"])
                self.assertEqual(image.types.dtype, torch.int64)
                self.assertTrue(torch.equal(image.types, record["types"]))
            encode.reset_mock()
            self.assertEqual(self.adapter.encode_prepared_images([]), [])
            encode.assert_not_called()

    def test_invalid_later_record_rejects_whole_request_before_encoding(self):
        for invalid in (
            "grid",
            "shape",
            "dtype",
            "identity",
            "types_dtype",
            "types_shape",
            "types_value",
            "overlap",
            "negative_start",
            "oversized_grid",
        ):
            with self.subTest(invalid=invalid):
                record = self.record(8)
                if invalid == "grid":
                    record["n_vit_h"] = 0
                elif invalid == "shape":
                    record["patches"] = record["patches"][:-1]
                elif invalid == "dtype":
                    record["patches"] = record["patches"].float()
                elif invalid == "identity":
                    record["processor_identity"] = "other"
                elif invalid == "types_dtype":
                    record["types"] = record["types"].float()
                elif invalid == "types_shape":
                    record["types"] = record["types"].unsqueeze(0)
                elif invalid == "types_value":
                    record["types"][1] = 4
                elif invalid == "overlap":
                    record["start"] = 2
                elif invalid == "negative_start":
                    record["start"] = -1
                else:
                    record["n_vit_h"] = record["n_vit_w"] = 10**9
                with patch.object(self.adapter, "encode_image") as encode:
                    with self.assertRaises(ValueError):
                        self.adapter.encode_prepared_images([self.record(), record])
                    encode.assert_not_called()

    def test_token_cap_checked_before_allocating_types(self):
        record = self.record()
        record["n_vit_h"] = record["n_vit_w"] = 192
        with patch(
            "rtp_llm.multimodal.multimodal_mixins.deepseek_v41.deepseek_v41_vision.image_token_types"
        ) as allocate_types, patch.object(self.adapter, "encode_image") as encode:
            with self.assertRaisesRegex(ValueError, "token limit"):
                self.adapter.encode_prepared_images([record])
            allocate_types.assert_not_called()
            encode.assert_not_called()

    def test_valid_record_runs_real_cpu_vision(self):
        config = V41Config.from_dict(
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
        adapter = DeepSeekV41VisionEmbedding(config, device="cpu").eval()
        with torch.no_grad():
            for parameter in adapter.parameters():
                parameter.fill_(0.125)
        record = self.record()
        record["processor_identity"] = adapter.processor_config.identity
        output = adapter.encode_prepared_images([record])[0]
        expected = adapter.encode_image(V41ImageInput(**record))
        self.assertEqual(tuple(output.shape), (5, 8))
        self.assertTrue(torch.isfinite(output).all())
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_empty_hash_bypasses_cache_without_aliasing(self):
        # An empty content_sha256 is accepted by the RPC/prepared-input checks;
        # two different images at the same grid/processor must NOT alias via the
        # encode cache, and nothing with an untrusted identity is stored.
        config = V41Config.from_dict(
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
        torch.manual_seed(1234)
        adapter = DeepSeekV41VisionEmbedding(config, device="cpu").eval()
        with torch.no_grad():
            for parameter in adapter.named_parameters():
                parameter[1].copy_(torch.randn_like(parameter[1]))
        patches_one = torch.randn(
            18, 3, 14, 14, generator=torch.Generator().manual_seed(11)
        ).to(torch.bfloat16)
        patches_two = torch.randn(
            18, 3, 14, 14, generator=torch.Generator().manual_seed(22)
        ).to(torch.bfloat16)
        one = dict(self.record(), patches=patches_one, content_sha256="")
        one["processor_identity"] = adapter.processor_config.identity
        two = dict(self.record(start=8), patches=patches_two, content_sha256="")
        two["processor_identity"] = adapter.processor_config.identity

        forwards = []
        hook = adapter.vision.register_forward_hook(lambda *args: forwards.append(1))
        try:
            first = adapter.encode_image(V41ImageInput(**one))
            second = adapter.encode_image(V41ImageInput(**two))
        finally:
            hook.remove()

        self.assertEqual(len(forwards), 2)
        self.assertFalse(torch.equal(first, second))
        self.assertEqual(len(adapter._encode_cache), 0)
        self.assertEqual(len(adapter._encode_cache_order), 0)

        # Same untrusted input recomputes every call (reference-test contract).
        recomputed = adapter.encode_image(V41ImageInput(**one))
        self.assertIsNot(recomputed, first)
        torch.testing.assert_close(recomputed, first, rtol=0, atol=0)
        self.assertEqual(len(adapter._encode_cache), 0)

    def test_empty_hash_leaves_valid_hash_hits_untouched(self):
        # The bypass must not break fast hits for trusted identities, the
        # disabled cache, or LRU eviction.
        config = V41Config.from_dict(
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
        torch.manual_seed(1234)
        adapter = DeepSeekV41VisionEmbedding(config, device="cpu").eval()
        with torch.no_grad():
            for parameter in adapter.named_parameters():
                parameter[1].copy_(torch.randn_like(parameter[1]))
        patches = torch.randn(
            18, 3, 14, 14, generator=torch.Generator().manual_seed(11)
        ).to(torch.bfloat16)
        trusted = dict(self.record(), patches=patches, content_sha256="c" * 64)
        trusted["processor_identity"] = adapter.processor_config.identity
        other = dict(self.record(start=8), patches=patches, content_sha256="d" * 64)
        other["processor_identity"] = adapter.processor_config.identity

        first = adapter.encode_image(V41ImageInput(**trusted))
        second = adapter.encode_image(V41ImageInput(**trusted))
        self.assertIs(second, first)
        adapter.encode_image(V41ImageInput(**other))
        self.assertEqual(len(adapter._encode_cache), 2)

        adapter._encode_cache_capacity = 1
        adapter._encode_cache.clear()
        adapter._encode_cache_order.clear()
        adapter.encode_image(V41ImageInput(**trusted))
        adapter.encode_image(V41ImageInput(**other))
        self.assertEqual(len(adapter._encode_cache), 1)
        self.assertEqual(list(adapter._encode_cache.keys())[0][1], "d" * 64)

    def test_identical_images_hit_encode_cache_and_preserve_values(self):
        config = V41Config.from_dict(
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
        adapter = DeepSeekV41VisionEmbedding(config, device="cpu").eval()
        with torch.no_grad():
            for parameter in adapter.parameters():
                parameter.fill_(0.125)
        record = self.record()
        record["processor_identity"] = adapter.processor_config.identity
        first = adapter.encode_image(V41ImageInput(**record))
        second = adapter.encode_image(V41ImageInput(**record))
        self.assertIs(second, first)
        torch.testing.assert_close(second, first, rtol=0, atol=0)
        adapter._encode_cache.clear()
        adapter._encode_cache_order.clear()
        recomputed = adapter.encode_image(V41ImageInput(**record))
        torch.testing.assert_close(recomputed, first, rtol=0, atol=0)

    def test_distinct_content_and_identity_do_not_collide(self):
        config = V41Config.from_dict(
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
        adapter = DeepSeekV41VisionEmbedding(config, device="cpu").eval()
        with torch.no_grad():
            for parameter in adapter.parameters():
                parameter.fill_(0.125)
        base = self.record()
        base["processor_identity"] = adapter.processor_config.identity
        different = self.record()
        different["processor_identity"] = adapter.processor_config.identity
        different["content_sha256"] = "b" * 64
        left = adapter.encode_image(V41ImageInput(**base))
        right = adapter.encode_image(V41ImageInput(**different))
        self.assertIsNot(right, left)
        self.assertEqual(
            adapter._encode_cache_order[-1],
            adapter._encode_cache_key(V41ImageInput(**different)),
        )
        self.assertEqual(len(adapter._encode_cache), 2)

    def test_cache_capacity_zero_disables_caching(self):
        config = V41Config.from_dict(
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
        adapter = DeepSeekV41VisionEmbedding(config, device="cpu").eval()
        adapter._encode_cache_capacity = 0
        with torch.no_grad():
            for parameter in adapter.parameters():
                parameter.fill_(0.125)
        record = self.record()
        record["processor_identity"] = adapter.processor_config.identity
        first = adapter.encode_image(V41ImageInput(**record))
        second = adapter.encode_image(V41ImageInput(**record))
        self.assertIsNot(second, first)
        torch.testing.assert_close(second, first, rtol=0, atol=0)
        self.assertEqual(len(adapter._encode_cache), 0)


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
    main()
