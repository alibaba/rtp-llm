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

    def __init__(self, global_use_swizzle_a: bool, global_force_legacy: bool = False):
        self.py_env_configs = SimpleNamespace(
            py_hw_kernel_config=SimpleNamespace(
                use_swizzleA=global_use_swizzle_a,
                force_legacy_fp8_ptpc=global_force_legacy,
            )
        )

    def _is_gfx950(self) -> bool:
        return False


class SwizzleLoadConfigTest(unittest.TestCase):
    def _postprocess(
        self,
        global_use_swizzle_a: bool,
        load_use_swizzle_a: bool,
        global_force_legacy: bool = False,
        load_force_legacy: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        device = FakeRocmDevice(global_use_swizzle_a, global_force_legacy)
        load_config = LoadConfig.model_construct(
            exported_device=device,
            use_swizzleA=load_use_swizzle_a,
            force_legacy_fp8_ptpc=load_force_legacy,
        )
        weight = AtomicWeight(W.attn_qkv_w, [])
        tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(dtype)
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

        # Models sharing the same device can require different FP8 layouts.
        # Exercise both orders so a preceding model cannot choose this one's
        # layout, including TBStars' explicit raw-weight CKTile path.
        for global_force in (False, True):
            for model_force in (False, True):
                with self.subTest(global_force=global_force, model_force=model_force):
                    with patch(
                        "rtp_llm.device.device_impl.swizzle_tensor",
                        side_effect=lambda tensor, _: tensor,
                    ) as swizzle:
                        result, tensor = self._postprocess(
                            global_use_swizzle_a=False,
                            load_use_swizzle_a=True,
                            global_force_legacy=global_force,
                            load_force_legacy=model_force,
                            dtype=torch.float8_e4m3fn,
                        )

                    if model_force:
                        swizzle.assert_not_called()
                    else:
                        swizzle.assert_called_once()
                    self.assertIs(result[W.attn_qkv_w], tensor)


if __name__ == "__main__":
    unittest.main()
