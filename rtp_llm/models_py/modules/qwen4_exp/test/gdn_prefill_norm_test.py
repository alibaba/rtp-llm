import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
from rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_norm_triton import (
    Qwen4PrefillGatedNorm,
)
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated

MODULE = "rtp_llm.models_py.modules.qwen4_exp.gdn_prefill_norm_triton"


class GdnPrefillNormTest(unittest.TestCase):
    def test_flat_heads_actual_stride_prime_rows_and_fp32_weight(self):
        for width in (1536, 3072):
            for rows in (2053, 8191):
                weight = torch.randn(128, device="cuda", dtype=torch.float32)
                native = RmsNormGated(weight, group_size=128, activation="sigmoid")
                model = Qwen4PrefillGatedNorm(native, role="PREFILL")
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

    def test_exact_complete_norm_and_projection(self):
        for width in (768, 1536, 3072):
            for rows in (2048, 4093, 4096, 8192, 12288):
                torch.manual_seed(30000 + width + rows)
                weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
                native = RmsNormGated(weight, group_size=128, activation="sigmoid")
                model = Qwen4PrefillGatedNorm(native, role="PREFILL")
                x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
                z = torch.randn(rows, width * 2, device="cuda", dtype=torch.bfloat16)[
                    :, width:
                ]
                expected = native(x, z)
                actual = model(x, z)
                self.assertTrue(
                    torch.equal(actual.view(torch.int16), expected.view(torch.int16))
                )
                projection = Fp8Linear(
                    (torch.randn(2560, width, device="cuda") * 0.1).to(
                        torch.float8_e4m3fn
                    ),
                    torch.tensor([0.019713], device="cuda"),
                    role="PREFILL",
                )
                self.assertTrue(torch.equal(projection(actual), projection(expected)))

    def test_graph_reads_new_values_strided_gate_and_weight(self):
        for width in (768, 1536, 3072):
            weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
            native = RmsNormGated(weight, group_size=128, activation="sigmoid")
            model = Qwen4PrefillGatedNorm(native, role="PDFUSION")
            x = torch.randn(4096, width, device="cuda", dtype=torch.bfloat16)
            gate = torch.randn(4096, width * 2, device="cuda", dtype=torch.bfloat16)[
                :, width:
            ]
            model(x, gate)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = model(x, gate)
            for scale in (0.01, 1.0, 100.0):
                x.normal_().mul_(scale)
                gate.normal_()
                weight.normal_()
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(actual, native(x, gate)))

    def test_decode_small_and_unmeasured_large_shapes_fallback(self):
        for rows, role, width in (
            (4, "PREFILL", 3072),
            (4096, "DECODE", 3072),
            (12288, "PREFILL", 3072),
            (4096, "PREFILL", 1024),
        ):
            native = RmsNormGated(
                torch.ones(128, device="cuda", dtype=torch.bfloat16),
                group_size=128,
                activation="sigmoid",
            )
            model = Qwen4PrefillGatedNorm(native, role=role)
            x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
            z = torch.randn_like(x)
            with patch(
                MODULE + "._prefill_gated_norm",
                side_effect=AssertionError("unexpected fast path"),
            ):
                self.assertTrue(torch.equal(model(x, z), native(x, z)))


if __name__ == "__main__":
    unittest.main()
