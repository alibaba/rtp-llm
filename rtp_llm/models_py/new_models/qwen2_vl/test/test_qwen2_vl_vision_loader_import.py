import tempfile
import unittest

import torch

from rtp_llm.models_py.model_loader import NewLoaderConfig
from rtp_llm.models_py.new_models.qwen2_vl.vision import (
    Qwen2VLForVisionEmbedding,
    load_qwen2_vl_vision,
)


class Qwen2VLVisionLoaderImportTest(unittest.TestCase):
    def test_public_vision_loader_target_loads_checkpoint_on_cpu(self):
        vision_config = {
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
        source = Qwen2VLForVisionEmbedding(
            {"model_type": "qwen2_vl_vision", "vision_config": vision_config},
            NewLoaderConfig(compute_dtype=torch.float32, device="cpu"),
        )
        with tempfile.TemporaryDirectory() as model_path:
            torch.save(source.state_dict(), f"{model_path}/pytorch_model.bin")
            loaded = load_qwen2_vl_vision(
                vision_config=vision_config,
                model_path=model_path,
                compute_dtype=torch.float32,
                device="cpu",
            )

        self.assertEqual(loaded.device, torch.device("cpu"))
        for name, expected in source.visual.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], expected)


if __name__ == "__main__":
    unittest.main()
