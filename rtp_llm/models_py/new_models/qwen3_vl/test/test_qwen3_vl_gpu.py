import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig
from rtp_llm.models_py.new_models.qwen3_vl.vision import (
    Qwen3VLForVisionEmbedding,
    _resolve_flash_attn_varlen,
    load_qwen3_vl_vision,
)


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


def _reference_interpolated_position_embeddings(visual, grid_values):
    index_lists = [[], [], [], []]
    weight_lists = [[], [], [], []]
    for _, height, width in grid_values:
        height_indexes = torch.linspace(0, visual.num_grid_per_side - 1, height)
        width_indexes = torch.linspace(0, visual.num_grid_per_side - 1, width)
        height_floor = height_indexes.int()
        width_floor = width_indexes.int()
        height_ceil = (height_floor + 1).clip(max=visual.num_grid_per_side - 1)
        width_ceil = (width_floor + 1).clip(max=visual.num_grid_per_side - 1)
        delta_height = height_indexes - height_floor
        delta_width = width_indexes - width_floor
        base_height = height_floor * visual.num_grid_per_side
        base_height_ceil = height_ceil * visual.num_grid_per_side

        indexes = (
            (base_height[:, None] + width_floor[None]).flatten(),
            (base_height[:, None] + width_ceil[None]).flatten(),
            (base_height_ceil[:, None] + width_floor[None]).flatten(),
            (base_height_ceil[:, None] + width_ceil[None]).flatten(),
        )
        weights = (
            ((1 - delta_height)[:, None] * (1 - delta_width)[None]).flatten(),
            ((1 - delta_height)[:, None] * delta_width[None]).flatten(),
            (delta_height[:, None] * (1 - delta_width)[None]).flatten(),
            (delta_height[:, None] * delta_width[None]).flatten(),
        )
        for index in range(4):
            index_lists[index].extend(indexes[index].tolist())
            weight_lists[index].extend(weights[index].tolist())

    index_tensor = torch.tensor(index_lists, dtype=torch.long, device=visual.device)
    weight_tensor = torch.tensor(weight_lists, dtype=visual.dtype, device=visual.device)
    weighted = visual.pos_embed(index_tensor) * weight_tensor[:, :, None]
    interpolated = weighted[0] + weighted[1] + weighted[2] + weighted[3]
    reduction_order = weighted.sum(dim=0)

    def reorder(position_embeddings):
        outputs = []
        offset = 0
        merge_size = visual.spatial_merge_size
        for frames, height, width in grid_values:
            spatial_size = height * width
            item = position_embeddings[offset : offset + spatial_size]
            offset += spatial_size
            outputs.append(
                item.repeat(frames, 1)
                .reshape(
                    frames,
                    height // merge_size,
                    merge_size,
                    width // merge_size,
                    merge_size,
                    -1,
                )
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
        return torch.cat(outputs)

    return reorder(interpolated), reorder(reduction_order)


class Qwen3VLVisionGpuTest(unittest.TestCase):
    def test_bf16_position_interpolation_matches_reference_addition_order(self):
        if not torch.cuda.is_available():
            self.skipTest("A CUDA or ROCm accelerator is required")

        torch.manual_seed(29)
        visual = (
            Qwen3VLForVisionEmbedding(
                {
                    "model_type": "qwen3_vl_vision",
                    "vision_config": _vision_config(),
                },
                NewLoaderConfig(compute_dtype=torch.bfloat16, device="cuda:0"),
            )
            .cuda()
            .eval()
            .visual
        )
        grid_values = [(1, 6, 10)]
        with torch.inference_mode():
            actual = visual._interpolated_position_embeddings(grid_values)
            expected, reduction_order = _reference_interpolated_position_embeddings(
                visual, grid_values
            )

        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(torch.equal(reduction_order, expected))

    def test_real_newloader_gpu_forward_matches_cpu_reference(self):
        if not torch.cuda.is_available():
            self.skipTest("A CUDA or ROCm accelerator is required")
        if torch.version.hip is None:
            self.assertIsNotNone(_resolve_flash_attn_varlen())

        torch.manual_seed(11)
        config = _vision_config()
        source = Qwen3VLForVisionEmbedding(
            {"model_type": "qwen3_vl_vision", "vision_config": config},
            NewLoaderConfig(compute_dtype=torch.float32, device="cpu"),
        ).eval()
        pixel_values = torch.linspace(-1.0, 1.0, 64).reshape(8, 8)
        grid_thw = torch.tensor([[2, 2, 2]], dtype=torch.int64)
        with torch.inference_mode():
            expected_output, expected_deepstack = source(pixel_values, grid_thw)

        with tempfile.TemporaryDirectory() as model_path:
            save_file(
                {
                    f"model.{name}": tensor.detach().clone()
                    for name, tensor in source.state_dict().items()
                },
                f"{model_path}/model.safetensors",
            )
            visual = load_qwen3_vl_vision(
                vision_config=config,
                model_path=model_path,
                compute_dtype=torch.float16,
                device="cuda:0",
            )

        self.assertFalse(visual.training)
        self.assertEqual(visual.device, torch.device("cuda:0"))
        with torch.inference_mode():
            actual_output, actual_deepstack = visual(
                pixel_values.cuda(), grid_thw.cuda()
            )
        torch.testing.assert_close(
            actual_output.float().cpu(),
            expected_output,
            atol=2e-2,
            rtol=2e-2,
        )
        self.assertEqual(len(actual_deepstack), len(expected_deepstack))
        torch.testing.assert_close(
            actual_deepstack[0].float().cpu(),
            expected_deepstack[0],
            atol=2e-2,
            rtol=2e-2,
        )

    def test_fp32_accelerator_forward_uses_sdpa_fallback(self):
        if not torch.cuda.is_available():
            self.skipTest("A CUDA or ROCm accelerator is required")

        config = _vision_config()
        model = (
            Qwen3VLForVisionEmbedding(
                {"model_type": "qwen3_vl_vision", "vision_config": config},
                NewLoaderConfig(compute_dtype=torch.float32, device="cuda:0"),
            )
            .cuda()
            .eval()
        )
        pixel_values = torch.linspace(-1.0, 1.0, 64, device="cuda").reshape(8, 8)
        grid_thw = torch.tensor([[2, 2, 2]], dtype=torch.int64, device="cuda")

        with patch(
            "rtp_llm.models_py.new_models.qwen3_vl.vision._resolve_flash_attn_varlen",
            side_effect=AssertionError("FP32 must not resolve flash-attention"),
        ):
            with torch.inference_mode():
                output, deepstack = model(pixel_values, grid_thw)

        self.assertEqual(output.device.type, "cuda")
        self.assertEqual(len(deepstack), 1)


if __name__ == "__main__":
    unittest.main()
