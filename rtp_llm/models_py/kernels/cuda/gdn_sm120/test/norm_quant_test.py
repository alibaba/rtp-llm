import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import quant_weight_ue8m0_packed
from rtp_llm.models_py.kernels.cuda.gdn_sm120.norm_quant_gluon import (
    _gdn_norm_quant,
    norm_quant,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
    CudaFp8DeepGEMMLinear,
)
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class GdnNormQuantTest(unittest.TestCase):
    def build(self, width):
        torch.manual_seed(580 + width)
        weight, scale = quant_weight_ue8m0_packed(
            torch.randn(2560, width, device="cuda") * 0.02
        )
        projection = CudaFp8DeepGEMMLinear(weight, weight_scales=scale)
        native = RmsNormGated(
            torch.randn(128, device="cuda", dtype=torch.bfloat16),
            group_size=128,
            activation="sigmoid",
        )
        return native, projection

    def test_runtime_rows_and_strides_reuse_compilation(self):
        for width in (1536, 3072):
            norm, projection = self.build(width)
            x = torch.randn(2048, width, device="cuda", dtype=torch.bfloat16)
            norm_quant(x, x, norm.weight, norm.eps)
            with patch.object(
                _gdn_norm_quant,
                "compile",
                side_effect=AssertionError("runtime sizes triggered recompilation"),
            ):
                for rows, padding in ((2053, 16), (4096, 32), (8191, 64)):
                    with self.subTest(width=width, rows=rows, padding=padding):
                        x = torch.randn(
                            rows, width, device="cuda", dtype=torch.bfloat16
                        )
                        gate = torch.randn(
                            rows, width + padding, device="cuda", dtype=torch.bfloat16
                        )[:, :width]
                        q, s = norm_quant(x, gate, norm.weight, norm.eps)
                        refq, refs = projection.quantize_input(norm(x, gate))
                        self.assertTrue(
                            torch.equal(q.view(torch.uint8), refq.view(torch.uint8))
                        )
                        self.assertTrue(torch.equal(s, refs))


if __name__ == "__main__":
    unittest.main()
