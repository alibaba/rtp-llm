import unittest

import torch
import torch.nn.functional as F
from rtp_llm.models_py.triton_kernels.common.activation import gelu_tanh_and_mul


class GeluTanhAndMulTest(unittest.TestCase):
    def test_matches_torch(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        for dtype, tolerance in (
            (torch.float32, 2e-5),
            (torch.bfloat16, 2e-2),
        ):
            for intermediate in (176, 704):
                with self.subTest(dtype=dtype, intermediate=intermediate):
                    generator = torch.Generator(device="cuda").manual_seed(1701)
                    source = torch.randn(
                        17,
                        2 * intermediate,
                        dtype=dtype,
                        device="cuda",
                        generator=generator,
                    )
                    output = torch.empty(17, intermediate, dtype=dtype, device="cuda")
                    gelu_tanh_and_mul(output, source)
                    gate, up = source.chunk(2, dim=-1)
                    expected = F.gelu(gate, approximate="tanh") * up
                    torch.testing.assert_close(
                        output, expected, rtol=tolerance, atol=tolerance
                    )


if __name__ == "__main__":
    unittest.main()
