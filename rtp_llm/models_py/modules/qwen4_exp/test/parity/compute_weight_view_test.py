import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.qwen4_exp.weight_views import prepare_compute_weights
from rtp_llm.utils.fp8_constants import FP8_STORAGE_SCALE_SUFFIX as SUFFIX


class ComputeWeightViewTest(unittest.TestCase):
    def test_norm_view_dequantizes_vectors_and_preserves_packed_moe(self):
        dense = torch.arange(16).reshape(4, 4).to(torch.float8_e4m3fn)
        norm = torch.arange(4).to(torch.float8_e4m3fn)
        scale = torch.tensor(0.25)
        packed = torch.arange(8, dtype=torch.uint8).view(2, 4)
        owner = SimpleNamespace(
            weights=[
                {
                    "dense": dense,
                    "dense" + SUFFIX: scale,
                    "moe": packed,
                    "norm": norm,
                    "norm" + SUFFIX: scale,
                }
            ],
            global_weights={"head": dense, "head" + SUFFIX: scale},
        )
        view = prepare_compute_weights(owner)
        self.assertIs(view._fp8_storage_owner, owner)
        self.assertIs(view.weights[0]["moe"], packed)
        self.assertIs(view.weights[0]["dense"], dense)
        self.assertIs(view.global_weights["head"], dense)
        self.assertIn("head" + SUFFIX, view.global_weights)
        torch.testing.assert_close(
            view.weights[0]["norm"], (norm.float() * scale).bfloat16(), rtol=0, atol=0
        )
        self.assertIs(owner.weights[0]["norm"], norm)
        self.assertTrue(view.weights[0]["norm"]._rtp_requires_restart_for_update)


if __name__ == "__main__":
    unittest.main()
