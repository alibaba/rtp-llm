"""H20 checkpoint regressions for Qwen3-VL and Qwen3.5's 72-wide vision heads."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rtp_llm.models_py.new_models.qwen3_vl.service_test import checkpoint
from rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin import (
    Qwen3_VLImageEmbedding,
    Qwen3_VLMixin,
)


class VisionDeviceRotaryTest(unittest.TestCase):
    @torch.inference_mode()
    def test_qwen35_fst_matches_native_vision_tower(self):
        from safetensors.torch import save_file

        from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe import (
            qwen3_5_moe_mixin as impl,
        )

        self.assertTrue(torch.cuda.is_available(), "requires the GPU-tagged worker")
        target = torch.device("cuda", torch.cuda.current_device())
        config = impl.Qwen3_5MoeVisionConfig(
            depth=1,
            hidden_size=144,
            intermediate_size=192,
            num_heads=2,
            patch_size=2,
            temporal_patch_size=1,
            out_hidden_size=32,
            num_position_embeddings=64,
        )
        dtype = torch.bfloat16
        with tempfile.TemporaryDirectory() as directory, patch.object(
            impl, "default_attn_impl", "sdpa"
        ), patch.object(impl.AutoProcessor, "from_pretrained"), patch.object(
            impl.Qwen2VLImageProcessor, "from_pretrained"
        ):
            config.save_pretrained(directory)
            params = SimpleNamespace(config={"ckpt_path": directory})
            previous = torch.get_default_dtype()
            try:
                torch.set_default_dtype(dtype)
                with torch.device(target):
                    reference = impl.Qwen3_5MoeImageEmbedding(params).visual.eval()
            finally:
                torch.set_default_dtype(previous)
            state = {
                "model.visual." + name: value.to(dtype=dtype, device="cpu")
                for name, value in reference.state_dict().items()
            }
            save_file(state, str(Path(directory) / "model.safetensors"))
            reference.load_state_dict(
                {
                    name.removeprefix("model.visual."): value
                    for name, value in state.items()
                }
            )
            loaded = impl.Qwen3_5MoeMixin.from_new_loader(
                dtype, str(target), params, "fastsafetensors", None, directory
            ).mm_part
            rotary = loaded.visual.rotary_pos_emb.inv_freq
            self.assertEqual(rotary.dtype, torch.float32)
            self.assertNotIn("rotary_pos_emb.inv_freq", loaded.visual.state_dict())
            torch.testing.assert_close(
                rotary, reference.rotary_pos_emb.inv_freq, rtol=0, atol=0
            )
            loaded.prepare_newloader_runtime_buffers("cuda")
            self.assertIs(rotary, loaded.visual.rotary_pos_emb.inv_freq)
            grid = torch.tensor([[1, 2, 128]], device=target)
            pixels = (
                torch.arange(256 * 12, device=target)
                .float()
                .sin()
                .to(dtype)
                .reshape(256, 12)
            )
            expected = reference(pixels, grid_thw=grid).pooler_output
            actual = loaded.visual.eval()(pixels, grid_thw=grid).pooler_output
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @torch.inference_mode()
    def test_real_fst_matches_target_native_tower(self):
        self.assertTrue(torch.cuda.is_available(), "requires the GPU-tagged worker")
        target = torch.device("cuda", torch.cuda.current_device())
        with tempfile.TemporaryDirectory() as directory, patch(
            "rtp_llm.multimodal.multimodal_mixins.qwen3_vl_mixin.default_attn_impl",
            "eager",
        ):
            # Two heads of width 72: the production model has 16 of this width.
            _, _, tensors = checkpoint(Path(directory), True, vision_hidden_size=144)
            params = SimpleNamespace(config={"ckpt_path": directory})
            dtype = torch.bfloat16
            previous = torch.get_default_dtype()
            try:
                torch.set_default_dtype(dtype)
                with torch.device(target):
                    reference = Qwen3_VLImageEmbedding(params).visual.eval()
            finally:
                torch.set_default_dtype(previous)
            reference.load_state_dict(
                {
                    name.removeprefix("model.visual."): value.half().to(
                        device=target, dtype=dtype
                    )
                    for name, value in tensors.items()
                    if name.startswith("model.visual.")
                }
            )
            loaded = Qwen3_VLMixin.from_new_loader(
                dtype, str(target), params, "fastsafetensors", None, directory
            ).mm_part
            frequencies = loaded.visual.rotary_pos_emb.inv_freq
            self.assertEqual(frequencies.dtype, torch.float32)
            self.assertEqual(frequencies.device, target)
            torch.testing.assert_close(
                frequencies, reference.rotary_pos_emb.inv_freq, rtol=0, atol=0
            )
            loaded.prepare_newloader_runtime_buffers("cuda")
            self.assertIs(frequencies, loaded.visual.rotary_pos_emb.inv_freq)
            self.assertNotIn("rotary_pos_emb.inv_freq", loaded.visual.state_dict())
            grid = torch.tensor([[1, 2, 128]], device=target)
            pixels = (
                torch.arange(256 * 12, device=target)
                .float()
                .sin()
                .to(dtype)
                .reshape(256, 12)
            )
            expected = reference(pixels, grid_thw=grid)
            actual = loaded.visual.eval()(pixels, grid_thw=grid)
            torch.testing.assert_close(
                actual.pooler_output, expected.pooler_output, rtol=0, atol=0
            )
            self.assertEqual(len(actual.deepstack_features), 1)
            torch.testing.assert_close(
                actual.deepstack_features[0],
                expected.deepstack_features[0],
                rtol=0,
                atol=0,
            )


if __name__ == "__main__":
    unittest.main()
