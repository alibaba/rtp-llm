import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig
from rtp_llm.models_py.new_models.qwen2_vl.vision import (
    Qwen2VLForVisionEmbedding,
    _resolve_flash_attn_varlen,
    load_qwen2_vl_vision,
)


def _vision_config():
    return {
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


class Qwen2VLVisionGpuTest(unittest.TestCase):
    def test_real_newloader_gpu_forward_matches_cpu_reference(self):
        if not torch.cuda.is_available():
            self.skipTest("A CUDA or ROCm accelerator is required")
        if torch.version.hip is None:
            self.assertIsNotNone(
                _resolve_flash_attn_varlen(),
                (
                    f"flash-attn unavailable on {torch.cuda.get_device_name(0)} "
                    f"with capability {torch.cuda.get_device_capability(0)}"
                ),
            )

        torch.manual_seed(11)
        config = _vision_config()
        source = Qwen2VLForVisionEmbedding(
            {"model_type": "qwen2_vl_vision", "vision_config": config},
            NewLoaderConfig(compute_dtype=torch.float32, device="cpu"),
        ).eval()
        pixel_values = torch.linspace(-1.0, 1.0, 128).reshape(16, 8)
        grid_thw = torch.tensor([[2, 2, 2], [1, 4, 2]], dtype=torch.int64)
        with torch.inference_mode():
            expected = source(pixel_values, grid_thw)

        with tempfile.TemporaryDirectory() as model_path:
            save_file(
                {
                    name: tensor.detach().clone()
                    for name, tensor in source.state_dict().items()
                },
                f"{model_path}/model.safetensors",
            )
            visual = load_qwen2_vl_vision(
                vision_config=config,
                model_path=model_path,
                compute_dtype=torch.float16,
                device="cuda:0",
            )

        self.assertFalse(visual.training)
        self.assertEqual(visual.device, torch.device("cuda:0"))
        with torch.inference_mode():
            actual = visual(pixel_values.cuda(), grid_thw.cuda()).float().cpu()
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    def test_fp32_accelerator_forward_uses_sdpa_fallback(self):
        if not torch.cuda.is_available():
            self.skipTest("A CUDA or ROCm accelerator is required")

        config = _vision_config()
        model = Qwen2VLForVisionEmbedding(
            {"model_type": "qwen2_vl_vision", "vision_config": config},
            NewLoaderConfig(compute_dtype=torch.float32, device="cuda:0"),
        ).cuda()
        pixel_values = torch.linspace(-1.0, 1.0, 128, device="cuda").reshape(16, 8)
        grid_thw = torch.tensor(
            [[2, 2, 2], [1, 4, 2]], dtype=torch.int64, device="cuda"
        )

        with mock.patch(
            "rtp_llm.models_py.new_models.qwen2_vl.vision."
            "_resolve_flash_attn_varlen",
            side_effect=AssertionError("FP32 must not resolve flash attention"),
        ):
            with torch.inference_mode():
                output = model(pixel_values, grid_thw)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
