import importlib.util
import unittest
from pathlib import Path

import torch


def load_fixed_m_linear():
    path = Path(__file__).resolve().parents[1] / "fixed_m_linear.py"
    spec = importlib.util.spec_from_file_location("fixed_m_linear_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fixed_m_linear


class FixedMLinearTest(unittest.TestCase):
    def test_router_is_invariant_to_neighbors_and_tail_size(self):
        fixed_m_linear = load_fixed_m_linear()

        def shape_sensitive_router(x):
            # Model a GEMM dispatch boundary that can change the winning expert.
            drift = 0.001 if x.shape[0] == 8 else -0.001
            return torch.stack((x[:, 0], x[:, 0] + drift), dim=1)

        query = torch.tensor([[1.0, 2.0]], device="cpu")
        neighbors = torch.arange(26, dtype=torch.float32).reshape(13, 2)
        self.assertNotEqual(
            shape_sensitive_router(query).argmax(-1).item(),
            shape_sensitive_router(torch.cat((query, neighbors[:7])))[0]
            .argmax()
            .item(),
        )
        alone = fixed_m_linear(shape_sensitive_router, query, 8)
        mixed = fixed_m_linear(shape_sensitive_router, torch.cat((query, neighbors)), 8)
        torch.testing.assert_close(alone[0], mixed[0], rtol=0, atol=0)
        self.assertEqual(mixed.shape, (14, 2))

    def test_padding_does_not_change_valid_rows_or_bias(self):
        fixed_m_linear = load_fixed_m_linear()
        x = torch.arange(51, dtype=torch.float32, device="cpu").reshape(17, 3)
        actual = fixed_m_linear(lambda chunk: chunk * 3 + 2, x, 7)
        torch.testing.assert_close(actual, x * 3 + 2, rtol=0, atol=0)

    def test_rejects_invalid_chunk_and_rank(self):
        fixed_m_linear = load_fixed_m_linear()
        with self.assertRaises(ValueError):
            fixed_m_linear(lambda x: x, torch.zeros(2, 3), 0)
        with self.assertRaises(ValueError):
            fixed_m_linear(lambda x: x, torch.zeros(3), 8)


if __name__ == "__main__":
    unittest.main()
