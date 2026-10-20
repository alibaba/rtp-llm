"""Grouped FP8 addressing and scale padding at real large element offsets."""
import unittest

import torch

from rtp_llm.models_py.triton_kernels.common.activation import (
    _sigmoid_mul_group128_kernel,
    rmsnorm_sigmoid_gate_per_token_group_quant_fp8,
    sigmoid_mul_per_token_group_quant_fp8,
)


class FusedActivationIndexTest(unittest.TestCase):
    def test_scale_padding_switch(self):
        for rows in (3, 32767, 32768):
            with self.subTest(rows=rows):
                x = torch.ones((rows, 128), device="cuda", dtype=torch.bfloat16)
                values, scales = sigmoid_mul_per_token_group_quant_fp8(x, torch.zeros_like(x))
                expected = 118 if rows * 128 >= 4 * 1024 * 1024 else 0x3F3F3F76
                self.assertTrue((values.view(torch.uint8) == 120).all().item())
                self.assertTrue((scales == expected).all().item())

    def test_scale_padding_extent_exceeds_int32(self):
        # Launch only the first three rows to isolate the total-extent selector.
        # The scales retain the real extent; this is not a truncated full forward.
        rows = (1 << 24) + 1
        x = torch.ones((3, 128), device="cuda", dtype=torch.bfloat16)
        values = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scales = torch.empty((1, (rows + 3) // 4 * 4), device="cuda", dtype=torch.int32)
        _sigmoid_mul_group128_kernel[(3,)](
            x, torch.zeros_like(x), values, scales, rows, 128, 128, 128, 512
        )
        torch.testing.assert_close(scales[0, :3], torch.full_like(scales[0, :3], 118), rtol=0, atol=0)

    def test_strided_inputs_exceed_int32(self):
        stride = (1 << 30) + 1024
        storage = torch.empty_strided((3, 1024), (stride, 1), device="cuda", dtype=torch.bfloat16)
        x, gate = storage[:, :512], storage[:, 512:]
        x.fill_(1)
        gate.zero_()
        self.assertGreater(x.stride(0) * 2, 1 << 31)
        outputs = [sigmoid_mul_per_token_group_quant_fp8(x, gate)]
        outputs.append(rmsnorm_sigmoid_gate_per_token_group_quant_fp8(
            x.view(3, 4, 128), gate.view(3, 4, 128),
            torch.ones(128, device="cuda", dtype=torch.bfloat16), 0.0,
        ))
        for values, scales in outputs:
            self.assertTrue((values.view(torch.uint8) == 120).all().item())
            self.assertTrue((scales == 0x76767676).all().item())


if __name__ == "__main__":
    unittest.main()
