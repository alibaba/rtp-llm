import json
import os
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.new_models.kimi_k25.vision import KimiK25ForVisionEmbedding
from rtp_llm.models_py.new_models.kimi_k25.vision_runtime import (
    KimiK25MultiModalProjector,
    KimiK25VisionConfig,
    MoonViT3dPretrainedModel,
    mm_projector_forward,
)


def config():
    return dict(
        vt_hidden_size=16,
        vt_intermediate_size=24,
        vt_num_hidden_layers=2,
        vt_num_attention_heads=2,
        patch_size=2,
        num_channels=3,
        merge_kernel_size=[2, 2],
        init_pos_emb_height=2,
        init_pos_emb_width=2,
        init_pos_emb_time=4,
        text_hidden_size=6,
        max_pos_emb_height=8,
        max_pos_emb_width=8,
    )


def source_modules():
    cfg = KimiK25VisionConfig(**config())
    tower, projector = MoonViT3dPretrainedModel(cfg), KimiK25MultiModalProjector(cfg)
    torch.manual_seed(31)
    with torch.no_grad():
        for module in (tower, projector):
            for parameter in module.parameters():
                parameter.uniform_(-0.1, 0.1)
    return tower, projector


def source_weights(tower, projector):
    return {
        **{
            "vision_tower." + name: value.clone()
            for name, value in tower.state_dict().items()
        },
        **{
            "mm_projector." + name: value.clone()
            for name, value in projector.state_dict().items()
        },
    }


def check_loaded_forward(test, device, load_method=NewLoaderLoadMethod.SCRATCH):
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    tower, projector = source_modules()
    weights = source_weights(tower, projector)
    weights["language_model.model.embed_tokens.weight"] = torch.ones(2, 2)
    with tempfile.TemporaryDirectory() as directory:
        names, index = list(weights), {}
        for shard_id in range(2):
            filename = f"model-{shard_id + 1:05d}-of-00002.safetensors"
            shard = {name: weights[name] for name in names[shard_id::2]}
            save_file(shard, os.path.join(directory, filename))
            index.update({name: filename for name in shard})
        with open(
            os.path.join(directory, "model.safetensors.index.json"), "w"
        ) as stream:
            json.dump({"weight_map": index}, stream)
        loader = NewModelLoader(
            model_config={"model_type": "kimi_k25_vision", "vision_config": config()},
            load_config=NewLoaderConfig(
                compute_dtype=dtype, device=device, load_method=load_method
            ),
            model_path=directory,
        )
        loaded = loader.load()
        test.assertEqual(loader._selected_load_method, load_method)
    for module in (tower, projector):
        for parameter in module.parameters():
            parameter.data = parameter.data.half().to(dtype)
        module.to(device).eval()
    test.assertEqual(
        loaded.vision_tower.patch_embed.pos_emb.time_weight.dtype, torch.float32
    )
    test.assertFalse(
        any(name.startswith("language_model") for name, _ in loaded.named_parameters())
    )
    for grid_values in ([[1, 2, 2]], [[1, 2, 2], [2, 4, 2]]):
        grid = torch.tensor(grid_values, device=device)
        pixels = torch.randn(
            int(grid.prod(dim=1).sum()), 3, 2, 2, dtype=dtype, device=device
        )
        with torch.inference_mode():
            actual, expected = loaded(pixels, grid), mm_projector_forward(
                projector, tower(pixels, grid)
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print(
            "KIMI_VISION",
            device,
            grid_values,
            "max_abs=0",
            "shapes=",
            [tuple(x.shape) for x in actual],
            flush=True,
        )
    return loaded


class KimiVisionTest(unittest.TestCase):
    @unittest.skipIf(
        os.environ.get("NEWLOADER_TEST_NATIVE_FST") == "0",
        "this platform lock does not ship the native FST reader",
    )
    def test_native_fst_keeps_vision_weights_after_checkpoint_close(self):
        check_loaded_forward(self, "cpu", NewLoaderLoadMethod.FASTSAFETENSORS)

    def test_real_sharded_load_image_video_matches_shared_runtime(self):
        check_loaded_forward(self, "cpu")

    def test_missing_projector_tensor_fails_load(self):
        tower, projector = source_modules()
        weights = source_weights(tower, projector)
        del weights["mm_projector.proj.2.weight"]
        with tempfile.TemporaryDirectory() as directory:
            save_file(weights, os.path.join(directory, "model.safetensors"))
            with self.assertRaisesRegex(RuntimeError, "proj.2.weight"):
                NewModelLoader(
                    model_config={
                        "model_type": "kimi_k25_vision",
                        "vision_config": config(),
                    },
                    load_config=NewLoaderConfig(
                        compute_dtype=torch.float32, device="cpu"
                    ),
                    model_path=directory,
                ).load()

    def test_vision_is_not_partitioned_with_language_tp(self):
        with self.assertRaisesRegex(ValueError, "replicated"):
            KimiK25ForVisionEmbedding(
                {"vision_config": config()}, NewLoaderConfig(tp_size=2, tp_rank=1)
            )


if __name__ == "__main__":
    unittest.main()
