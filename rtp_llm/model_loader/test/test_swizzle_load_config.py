import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.device.device_impl import RocmImpl
from rtp_llm.model_loader.load_config import LoadConfig
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.utils.model_weight import W


class FakeRocmDevice:
    maybe_rewrite_weight_by_key = RocmImpl.maybe_rewrite_weight_by_key

    def __init__(self, global_use_swizzle_a: bool):
        self.py_env_configs = SimpleNamespace(
            py_hw_kernel_config=SimpleNamespace(
                use_swizzleA=global_use_swizzle_a
            )
        )

    def _is_gfx950(self) -> bool:
        return False


class SwizzleLoadConfigTest(unittest.TestCase):
    def _postprocess(self, global_use_swizzle_a: bool, load_use_swizzle_a: bool):
        device = FakeRocmDevice(global_use_swizzle_a)
        load_config = LoadConfig.model_construct(
            exported_device=device,
            use_swizzleA=load_use_swizzle_a,
        )
        weight = AtomicWeight(W.attn_qkv_w, [])
        tensor = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        return weight._postprocess(tensor, "cpu", load_config), tensor

    def test_per_model_false_keeps_draft_weight_raw(self):
        with patch("rtp_llm.device.device_impl.swizzle_tensor") as swizzle:
            result, tensor = self._postprocess(
                global_use_swizzle_a=True,
                load_use_swizzle_a=False,
            )

        swizzle.assert_not_called()
        self.assertIs(result[W.attn_qkv_w], tensor)

    def test_per_model_true_swizzles_target_weight(self):
        with patch(
            "rtp_llm.device.device_impl.swizzle_tensor",
            side_effect=lambda tensor, _: tensor,
        ) as swizzle:
            result, tensor = self._postprocess(
                global_use_swizzle_a=False,
                load_use_swizzle_a=True,
            )

        swizzle.assert_called_once()
        torch.testing.assert_close(result[W.attn_qkv_w], tensor)


if __name__ == "__main__":
    unittest.main()
