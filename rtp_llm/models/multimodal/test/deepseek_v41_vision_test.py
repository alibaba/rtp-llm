import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main

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
from rtp_llm.models.multimodal.deepseek_vision import RMSNorm

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


if __name__ == "__main__":
    main()
