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
        ig = torch.tensor([tile["input_global_scale"]], device="cuda")
        wg = torch.tensor([tile["weight_global_scale"]], device="cuda")
        torch.manual_seed(42)
        x = (torch.randn(32, 128, device="cuda") * 0.1).bfloat16()
        # Compare against the same quantized activations; this isolates adapter
        # and layout correctness from activation quantization error.
        xq, xs = fp4_quantize(x, ig, is_sf_swizzled_layout=False)
        reference = unpack_fp4(xq, xs, ig) @ unpack_fp4(weight, scales, wg).T
        with patch.dict("os.environ", {"RTP_LLM_FP4_GEMM_BACKEND": "cutlass"}):
            packed, swizzled = CudaImpl.convert_fp4_gemm_weight_params(weight, scales)
            linear = CudaFp4GEMMLinear(
                weight=packed,
                weight_scales=swizzled,
                input_scales=reciprocal_global_scale([ig]),
                weight_scale_2=reciprocal_global_scale([wg]),
                quant_config=CompressedTensorsNvfp4Config(),
            )
            self.assertTrue(
                linear.can_handle(
                    CompressedTensorsNvfp4Config(),
                    packed,
                    swizzled,
                    weight_scale_2=linear.weight_scale_2,
                    input_scale=linear.input_scale,
                )
            )
            actual = linear.forward(x)
        torch.testing.assert_close(actual.float(), reference, rtol=0.02, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
