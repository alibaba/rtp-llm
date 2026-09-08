import tempfile
import unittest

import torch

from rtp_llm.models_py.model_loader import NewLoaderConfig
from rtp_llm.models_py.new_models.qwen3_vl.vision import (
    Qwen3VLForVisionEmbedding,
    load_qwen3_vl_vision,
)


class Qwen3VLVisionLoaderImportTest(unittest.TestCase):
    def test_public_vision_loader_target_loads_checkpoint_on_cpu(self):
        vision_config = {
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
        source = Qwen3VLForVisionEmbedding(
            {"model_type": "qwen3_vl_vision", "vision_config": vision_config},
            NewLoaderConfig(compute_dtype=torch.float32, device="cpu"),
        )
        with tempfile.TemporaryDirectory() as model_path:
            torch.save(
                {
                    f"model.{name}": tensor.detach().clone()
                    for name, tensor in source.state_dict().items()
                },
                f"{model_path}/pytorch_model.bin",
            )
            loaded = load_qwen3_vl_vision(
                vision_config=vision_config,
                model_path=model_path,
                compute_dtype=torch.float32,
                device="cpu",
            )

        self.assertEqual(loaded.device, torch.device("cpu"))
        for name, expected in source.visual.state_dict().items():
            torch.testing.assert_close(
                loaded.state_dict()[name],
                expected.to(torch.float16).to(torch.float32),
            )


if __name__ == "__main__":
    unittest.main()
