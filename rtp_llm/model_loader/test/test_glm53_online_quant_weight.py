import sys
import types
import unittest
from unittest.mock import patch

import torch

from rtp_llm.model_loader.online_modelopt_fp4_quant_weight import (
    convert_fp8_moe_to_fp4_ue8m0,
)


class Glm53OnlineQuantWeightTest(unittest.TestCase):
    @staticmethod
    def _fake_deep_gemm(captured):
        package = types.ModuleType("deep_gemm")
        utils = types.ModuleType("deep_gemm.utils")

        def per_token_cast_to_fp4(weight, *, use_ue8m0, gran_k):
            captured.append(weight.clone())
            assert use_ue8m0
            return (
                torch.zeros(weight.shape[0], weight.shape[1] // 2, dtype=torch.int8),
                torch.ones(weight.shape[0], weight.shape[1] // gran_k),
            )

        utils.per_token_cast_to_fp4 = per_token_cast_to_fp4
        package.utils = utils
        return package, utils

    def test_fp8_block_scale_uses_ceil_for_non_aligned_output_rows(self):
        captured = []
        package, utils = self._fake_deep_gemm(captured)
        weight = torch.ones(1, 129, 128)
        scale = torch.tensor([[[2.0], [3.0]]])

        with patch.dict(
            sys.modules,
            {"deep_gemm": package, "deep_gemm.utils": utils},
        ):
            packed, output_scale = convert_fp8_moe_to_fp4_ue8m0(weight, scale)

        self.assertEqual(tuple(packed.shape), (1, 129, 64))
        self.assertEqual(tuple(output_scale.shape), (1, 129, 4))
        torch.testing.assert_close(
            captured[0][:128],
            torch.full((128, 128), 2.0, dtype=torch.bfloat16),
        )
        torch.testing.assert_close(
            captured[0][128], torch.full((128,), 3.0, dtype=torch.bfloat16)
        )

    def test_fp8_block_scale_rejects_floor_sized_non_aligned_layout(self):
        package, utils = self._fake_deep_gemm([])
        weight = torch.ones(1, 129, 128)
        scale = torch.ones(1, 1, 1)

        with patch.dict(
            sys.modules,
            {"deep_gemm": package, "deep_gemm.utils": utils},
        ):
            with self.assertRaisesRegex(ValueError, "Cannot interpret scale shape"):
                convert_fp8_moe_to_fp4_ue8m0(weight, scale)

    def test_fp8_source_rejects_k_not_divisible_by_fp8_block(self):
        package, utils = self._fake_deep_gemm([])
        weight = torch.ones(1, 2, 96)
        scale = torch.ones(1, 2, 1)

        with patch.dict(
            sys.modules,
            {"deep_gemm": package, "deep_gemm.utils": utils},
        ):
            with self.assertRaisesRegex(ValueError, "must be divisible by 128"):
                convert_fp8_moe_to_fp4_ue8m0(weight, scale)


if __name__ == "__main__":
    unittest.main()
