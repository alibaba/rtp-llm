import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, main

import torch
from safetensors import safe_open
from torch.nn.attention import SDPBackend, sdpa_kernel

from rtp_llm.models.multimodal.deepseek_vision import Aligner, RMSNorm, ViT


class DeepSeekVisionTest(TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("The vision foundation test requires a CUDA device")
        model_path = Path(os.environ["DSV41_MODEL_PATH"])
        reference_path = model_path / "inference/vision.py"
        spec = importlib.util.spec_from_file_location(
            "dsv41_official_vision", reference_path
        )
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        raw = json.loads((model_path / "config.json").read_text())
        vision = raw["vision_config"]
        cls.config = dict(
            vision_dim=vision["hidden_size"],
            vision_n_heads=vision["num_attention_heads"],
            vision_n_layers=vision["num_hidden_layers"],
            vision_inter_dim=vision["intermediate_size"],
            vision_patch_size=vision["patch_size"],
            vision_downsample_ratio=vision["downsample_ratio"],
            vision_rope_theta=vision["rope_theta"],
            hidden_size=raw["text_config"]["hidden_size"],
        )
        args = SimpleNamespace(**cls.config, dim=raw["text_config"]["hidden_size"])
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            with torch.device("cuda"):
                cls.vision = ViT(cls.config).eval()
                cls.aligner = Aligner(cls.config).eval()
                cls.reference_vision = reference.ViT(args).eval()
                cls.reference_aligner = reference.Aligner(args).eval()
        finally:
            torch.set_default_dtype(previous_dtype)
        weight_map = json.loads(
            (model_path / "model.safetensors.index.json").read_text()
        )["weight_map"]
        for prefix, modules in (
            ("vision", (cls.vision, cls.reference_vision)),
            ("aligner", (cls.aligner, cls.reference_aligner)),
        ):
            state = {}
            names = [name for name in weight_map if name.startswith(prefix + ".")]
            for shard in sorted({weight_map[name] for name in names}):
                with safe_open(
                    model_path / shard, framework="pt", device="cpu"
                ) as weights:
                    for name in names:
                        if weight_map[name] == shard:
                            state[name[len(prefix) + 1 :]] = weights.get_tensor(name)
            for module in modules:
                module.load_state_dict(state, strict=True)
        print(
            json.dumps(
                {
                    "gpu": torch.cuda.get_device_name(),
                    "capability": torch.cuda.get_device_capability(),
                    "torch": torch.__version__,
                    "vision_layers": cls.config["vision_n_layers"],
                    "weight_tensors": sum(
                        name.startswith(("vision.", "aligner.")) for name in weight_map
                    ),
                    "reference": str(reference_path),
                }
            ),
            flush=True,
        )

    @torch.inference_mode()
    def test_initialized_encoder_and_aligner_match_official_math(self):
        # Keep the reference attention backend fixed for the foundation comparison.
        # Production kernel qualification is a separate W03/W08 measurement.
        with sdpa_kernel(SDPBackend.MATH), torch.device("cuda"):
            for height, width in ((3, 6), (4, 7), (16, 16), (39, 39)):
                with self.subTest(grid=(height, width)):
                    generator = torch.Generator(device="cuda").manual_seed(314 + height)
                    patches = torch.randn(
                        height * width,
                        3,
                        14,
                        14,
                        generator=generator,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    expected = self.reference_vision(patches, height, width)
                    actual = self.vision(patches, height, width)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    aligned = self.aligner(actual, height, width)
                    reference_aligned = self.reference_aligner(expected, height, width)
                    torch.testing.assert_close(
                        aligned, reference_aligned, rtol=0, atol=0
                    )
                    self.assertTrue(torch.isfinite(aligned).all().item())

    def test_real_checkpoint_norms_remain_fp32(self):
        norms = [
            module for module in self.vision.modules() if isinstance(module, RMSNorm)
        ]
        self.assertEqual(len(norms), 2 * self.config["vision_n_layers"] + 1)
        for module in norms:
            self.assertEqual(module.weight.dtype, torch.float32)
        self.assertEqual(self.vision.patch_embed.proj.weight.dtype, torch.bfloat16)


if __name__ == "__main__":
    main()
