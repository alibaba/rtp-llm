import unittest
import importlib.util
from pathlib import Path

import torch


_source = Path(__file__).resolve().parents[1] / "fp8_kernel" / "fused_activation.py"
_spec = importlib.util.spec_from_file_location("fused_activation_under_test", _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
_norm_source = (
    Path(__file__).resolve().parents[3]
    / "modules" / "kimi_k3" / "native_gated_norm.py"
)
_norm_spec = importlib.util.spec_from_file_location("native_gated_norm_under_test", _norm_source)
_norm_module = importlib.util.module_from_spec(_norm_spec)
_norm_spec.loader.exec_module(_norm_module)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class FusedRmsnormGateQuantTest(unittest.TestCase):
    def test_sigmoid_gate_matches_bf16_eager_then_group128_quant(self):
        for rows, width in ((17, 896), (8192, 1536)):
            with self.subTest(rows=rows, width=width):
                torch.manual_seed(208)
                x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
                gate_storage = torch.randn(
                    rows, 2, width, device="cuda", dtype=torch.bfloat16
                )
                gate = gate_storage[:, 0]
                values, scales = _module.sigmoid_mul_per_token_group_quant_fp8(
                    x, gate
                )

                # vLLM/feat evaluates BF16 sigmoid, then BF16 multiply, before
                # the ordinary grouped E4M3 activation quantizer.
                eager = x * gate.sigmoid()
                groups = eager.float().reshape(rows, width // 128, 128)
                expected_scale = torch.exp2(
                    torch.ceil(
                        torch.log2(
                            torch.clamp(groups.abs().amax(dim=-1), min=1e-4)
                            / 448.0
                        )
                    )
                )
                packed = scales.to(torch.int64)
                exponents = torch.stack(
                    [
                        (packed[:, group // 4] >> (8 * (group % 4))) & 255
                        for group in range(width // 128)
                    ],
                    dim=1,
                )
                actual_scale = torch.exp2(exponents.float() - 127)
                torch.testing.assert_close(
                    actual_scale, expected_scale, rtol=0, atol=0
                )
                expected_values = torch.clamp(
                    groups / actual_scale[..., None], -448, 448
                ).to(torch.float8_e4m3fn)
                torch.testing.assert_close(
                    values.float().reshape_as(groups), expected_values.float(),
                    rtol=0, atol=0,
                )

                if width == 896:
                    expected_padding = 0x3F if rows * width < 4 * 1024 * 1024 else 0
                    torch.testing.assert_close(
                        (packed[:, 1] >> 24) & 255,
                        torch.full(
                            (rows,), expected_padding,
                            device="cuda", dtype=torch.int64,
                        ),
                        rtol=0, atol=0,
                    )

    def test_matches_native_norm_and_group128_scales_with_strided_gate(self):
        for rows in (17, 8192):
            with self.subTest(rows=rows):
                torch.manual_seed(117)
                heads = 7
                x = torch.randn(rows, heads, 128, device="cuda", dtype=torch.bfloat16)
                gate_storage = torch.randn(
                    rows, heads * 2, 128, device="cuda", dtype=torch.bfloat16
                )
                gate = gate_storage[:, ::2, :]
                weight = torch.randn(128, device="cuda", dtype=torch.bfloat16)
                eps = 1e-5

                values, scales = _module.rmsnorm_sigmoid_gate_per_token_group_quant_fp8(
                    x, gate, weight, eps
                )
                self.assertEqual(tuple(values.shape), (rows, heads * 128))
                self.assertEqual(values.dtype, torch.float8_e4m3fn)
                self.assertEqual(tuple(scales.shape), (rows, 2))
                self.assertEqual(scales.dtype, torch.int32)
                self.assertEqual(scales.stride(0), 1)

                native = _norm_module.KimiK3GatedNorm(weight, eps)(
                    x.reshape(-1, 128), gate.reshape(-1, 128)
                ).reshape(rows, heads, 128)
                expected_scale = torch.exp2(torch.ceil(torch.log2(
                    torch.clamp(native.float().abs().amax(dim=-1), min=1e-4) / 448
                )))
                packed = scales.to(torch.int64)
                exponents = torch.stack(
                    [((packed[:, h // 4] >> (8 * (h % 4))) & 255) for h in range(heads)],
                    dim=1,
                )
                actual_scale = torch.exp2(exponents.float() - 127)
                torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=0)
                expected_values = torch.clamp(
                    native.float() / actual_scale[..., None], -448, 448
                ).to(torch.float8_e4m3fn)
                torch.testing.assert_close(
                    values.float().reshape(rows, heads, 128), expected_values.float(),
                    rtol=0, atol=0,
                )
                expected_padding = 0x3F if rows * heads * 128 < 4 * 1024 * 1024 else 0
                torch.testing.assert_close(
                    (packed[:, 1] >> 24) & 255,
                    torch.full((rows,), expected_padding, device="cuda", dtype=torch.int64),
                    rtol=0, atol=0,
                )


if __name__ == "__main__":
    unittest.main()
