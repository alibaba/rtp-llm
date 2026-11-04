import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.qwen4_exp.weight_views import prepare_compute_weights
from rtp_llm.utils.fp8_constants import FP8_STORAGE_SCALE_SUFFIX as SUFFIX


class ComputeWeightViewTest(unittest.TestCase):
    def test_draft_bf16_view_keeps_storage_owner_and_packed_moe(self):
        dense = torch.arange(16).reshape(4, 4).to(torch.float8_e4m3fn)
        scale = torch.tensor(0.25)
        packed = torch.arange(8, dtype=torch.uint8).view(2, 4)
        owner = SimpleNamespace(
            weights=[{"dense": dense, "dense" + SUFFIX: scale, "moe": packed}],
            global_weights={"head": dense, "head" + SUFFIX: scale},
        )
        view = prepare_compute_weights(owner, dense_bf16=True)
        self.assertIs(view._fp8_storage_owner, owner)
        self.assertIs(view.weights[0]["moe"], packed)
        self.assertIs(owner.weights[0]["dense"], dense)
        self.assertIn("dense" + SUFFIX, owner.weights[0])
        self.assertNotIn("dense" + SUFFIX, view.weights[0])
        self.assertNotIn("head" + SUFFIX, view.global_weights)
        torch.testing.assert_close(
            view.weights[0]["dense"], (dense.float() * scale).bfloat16(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            view.global_weights["head"],
            (dense.float() * scale).bfloat16(),
            rtol=0,
            atol=0,
        )
        self.assertTrue(view.global_weights["head"]._rtp_requires_restart_for_update)

    def test_default_view_preserves_fp8_dense_dispatch(self):
        dense = torch.ones(4, 4).to(torch.float8_e4m3fn)
        owner = SimpleNamespace(
            weights=[{"w": dense, "w" + SUFFIX: torch.tensor(0.5)}], global_weights={}
        )
        view = prepare_compute_weights(owner)
        self.assertIs(view.weights[0]["w"], dense)
        self.assertIn("w" + SUFFIX, view.weights[0])


if __name__ == "__main__":
    unittest.main()
