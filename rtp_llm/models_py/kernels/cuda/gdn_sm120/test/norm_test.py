import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.gdn_sm120.norm_gluon import (
    _gdn_flat_head_norm,
    flat_head_norm,
)
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated


class GdnPrefillNormTest(unittest.TestCase):
    def test_runtime_rows_and_gate_stride_reuse_compilation(self):
        for width in (1536, 3072):
            weight = torch.randn(128, device="cuda", dtype=torch.float32)
            native = RmsNormGated(weight, group_size=128, activation="sigmoid")
            x = torch.randn(2048, width, device="cuda", dtype=torch.bfloat16)
            flat_head_norm(x, x, weight, native.eps)
            with patch.object(
                _gdn_flat_head_norm,
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
                        actual = flat_head_norm(x, gate, weight, native.eps)
                        self.assertTrue(
                            torch.equal(
                                actual.view(torch.int16),
                                native(x, gate).view(torch.int16),
                            )
                        )

    def test_flat_heads_actual_stride_prime_rows_and_fp32_weight(self):
        for width in (1536, 3072):
            for rows in (2053, 8191):
                weight = torch.randn(128, device="cuda", dtype=torch.float32)
                native = RmsNormGated(weight, group_size=128, activation="sigmoid")
                model = lambda x, gate: flat_head_norm(x, gate, weight, native.eps)
                x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
                storage = torch.randn(
                    rows, width * 8 // 3, device="cuda", dtype=torch.bfloat16
                )
                gate = storage[:, -width:]
                for scale in (0.0, 1e-10, 100.0):
                    x.normal_().mul_(scale)
                    actual = model(x, gate)
                    expected = native(x, gate)
                    self.assertTrue(
                        torch.equal(
                            actual.view(torch.int16), expected.view(torch.int16)
                        )
                    )


if __name__ == "__main__":
    unittest.main()
