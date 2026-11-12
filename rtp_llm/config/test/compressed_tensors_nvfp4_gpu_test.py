"""Native numerical check using an attributed tile from a real HF checkpoint."""

import base64
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.config.nvfp4_utils import reciprocal_global_scale
from rtp_llm.config.quant_config import CompressedTensorsNvfp4Config


def unpack_fp4(packed, scales, divisor):
    # E2M1: low nibble is the first K element, bit3 is sign.
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=packed.device)
    values = lut[codes & 7] * torch.where((codes & 8) != 0, -1, 1)
    return (
        values
        * scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
        / divisor
    )


class CompressedTensorsNvfp4GpuTest(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10,
        "Requires a Blackwell GPU and the built RTP-LLM CUDA environment",
    )
    def test_real_tile_matches_dequantized_reference(self):
        from flashinfer import fp4_quantize

        from rtp_llm.device.device_impl import CudaImpl
        from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp4_linear import (
            CudaFp4GEMMLinear,
        )

        tile = json.loads(
            (Path(__file__).parent / "fixtures/nvfp4/q_proj_tile.json").read_text()
        )

        def load(key, shape):
            return torch.tensor(
                list(base64.b64decode(tile[key])), dtype=torch.uint8, device="cuda"
            ).reshape(shape)

        weight = load("weight_packed", (128, 64))
        scales = load("weight_scale", (128, 8)).view(torch.float8_e4m3fn)
        # Asymmetric exact E2M1 values break the simultaneous-permutation
        # ambiguity of decoding both operands with the same helper.
        pattern = torch.tensor(
            [0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6, 0, 2],
            device="cuda",
        )
        x = pattern.repeat(32, 8).bfloat16()
        # Each group has amax=6, so global divisor=448 yields local scale=448
        # and exactly representable activations, independent of weight globals.
        ig = torch.tensor(448.0, device="cuda")
        wg = torch.tensor(tile["weight_global_scale"], device="cuda")
        xq, xs = fp4_quantize(x, ig, is_sf_swizzled_layout=False)
        torch.testing.assert_close(unpack_fp4(xq, xs, ig), x.float(), rtol=0, atol=0)
        reference = x.float() @ unpack_fp4(weight, scales, wg).T
        with patch.dict("os.environ", {"RTP_LLM_FP4_GEMM_BACKEND": "cutlass"}):
            packed, swizzled = CudaImpl.convert_fp4_gemm_weight_params(weight, scales)
            # Raw scalar ranks cover stored scalars and the loader's unsqueeze.
            for shape in ((), (1,), (1, 1)):
                with self.subTest(raw_global_shape=shape):
                    linear = CudaFp4GEMMLinear(
                        weight=packed,
                        weight_scales=swizzled,
                        input_scales=reciprocal_global_scale([ig.reshape(shape)]),
                        weight_scale_2=reciprocal_global_scale([wg.reshape(shape)]),
                        quant_config=CompressedTensorsNvfp4Config(),
                    )
                    self.assertEqual(linear.alpha.shape, torch.Size([]))
                    torch.testing.assert_close(
                        linear.forward(x).float(), reference, rtol=0.02, atol=2e-5
                    )
            # Mutate only weight packing; keep known input/reference fixed.
            wrong = (weight >> 4) | ((weight & 15) << 4)
            wrong_packed, wrong_scales = CudaImpl.convert_fp4_gemm_weight_params(
                wrong, scales
            )
            wrong_linear = CudaFp4GEMMLinear(
                weight=wrong_packed,
                weight_scales=wrong_scales,
                input_scales=reciprocal_global_scale([ig]),
                weight_scale_2=reciprocal_global_scale([wg]),
                quant_config=CompressedTensorsNvfp4Config(),
            )
            wrong_output = wrong_linear.forward(x).float()
            self.assertGreater(
                ((wrong_output - reference).norm() / reference.norm()).item(), 0.05
            )


if __name__ == "__main__":
    unittest.main()
