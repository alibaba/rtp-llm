"""PPU linears receive checkpoint scales before CUDA-only repacking."""
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4 import platform_provider, utils
from rtp_llm.models_py.modules.dsv4.fp8 import attention


class PpuDeepGemmLayoutCompatTest(unittest.TestCase):
    def _check_raw_checkpoint_scale_dispatch(self, module):
        weight = torch.zeros((256, 256), dtype=torch.float8_e4m3fn)
        scale = torch.full((2, 2), 127, dtype=torch.uint8).view(torch.float8_e8m0fnu)
        weights = {"weight": weight, "scale": scale}
        constructed = object()
        with patch.object(module, "_repack_v4_fp8_scale_to_int32",
                          side_effect=AssertionError("premature CUDA layout conversion")):
            with patch.object(platform_provider, "build_dsv4_fp8_linear",
                              return_value=constructed) as dispatch:
                result = module._v4_fp8_linear_from_dict(weights, "weight", "scale")
        self.assertIs(result, constructed)
        self.assertIs(dispatch.call_args.args[1], weight)
        self.assertIs(dispatch.call_args.args[2], scale)
        self.assertIs(weights["scale"], scale)

    def test_shared_utility_preserves_checkpoint_scale(self):
        self._check_raw_checkpoint_scale_dispatch(utils)

    def test_attention_preserves_checkpoint_scale(self):
        self._check_raw_checkpoint_scale_dispatch(attention)


if __name__ == "__main__":
    unittest.main()
