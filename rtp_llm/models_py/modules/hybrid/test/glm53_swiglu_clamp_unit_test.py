import unittest

import torch
import torch.nn.functional as F

from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul


class Glm53SwiGLUClampTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")

    def test_asymmetric_clamp_matches_reference(self) -> None:
        limit = 10.0
        up = torch.tensor(
            [[-20.0, -2.0, 3.0, 20.0], [11.0, -11.0, 0.5, -0.5]],
            dtype=torch.bfloat16,
            device="cuda",
        )
        gate = torch.tensor(
            [[-20.0, -2.0, 3.0, 20.0], [11.0, -11.0, 0.5, -0.5]],
            dtype=torch.bfloat16,
            device="cuda",
        )
        merged = torch.cat((up, gate), dim=-1)
        actual = torch.empty_like(up)

        silu_and_mul(actual, merged, clamp_limit=limit)

        expected = F.silu(torch.clamp(gate.float(), max=limit)) * torch.clamp(
            up.float(), min=-limit, max=limit
        )
        torch.testing.assert_close(actual.float(), expected, rtol=1e-2, atol=2e-2)

    def test_default_keeps_existing_unclamped_semantics(self) -> None:
        up = torch.tensor(
            [[-20.0, 20.0, 1.0, -1.0]], dtype=torch.bfloat16, device="cuda"
        )
        gate = torch.tensor(
            [[20.0, -20.0, 2.0, -2.0]], dtype=torch.bfloat16, device="cuda"
        )
        merged = torch.cat((up, gate), dim=-1)
        actual = torch.empty_like(up)

        silu_and_mul(actual, merged)

        expected = F.silu(gate.float()) * up.float()
        torch.testing.assert_close(actual.float(), expected, rtol=1e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
