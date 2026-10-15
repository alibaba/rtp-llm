import os
from unittest import TestCase, main
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.linear.fixed_m_linear import fixed_m_linear


class FixedMLinearTest(TestCase):
    def test_every_chunk_keeps_fixed_geometry(self) -> None:
        for enabled in ("0", "1"):
            for rows in (1, 7, 8, 9, 17):
                source = torch.arange(rows * 5, dtype=torch.float32).reshape(rows, 5)
                seen = []

                def linear(value: torch.Tensor) -> torch.Tensor:
                    seen.append(value.clone())
                    return value * 2 + 1

                with patch.dict(os.environ, {"RTP_LLM_FUSED_ROUTER_PAD": enabled}):
                    actual = fixed_m_linear(linear, source, 8)

                torch.testing.assert_close(actual, source * 2 + 1, atol=0, rtol=0)
                self.assertEqual(
                    [tuple(value.shape) for value in seen], [(8, 5)] * ((rows + 7) // 8)
                )
                if rows % 8:
                    self.assertEqual(
                        torch.count_nonzero(seen[-1][rows % 8 :]).item(), 0
                    )

    def test_noncontiguous_input_matches_reference(self) -> None:
        source = torch.arange(70, dtype=torch.float32).reshape(7, 10)[:, ::2]
        with patch.dict(os.environ, {"RTP_LLM_FUSED_ROUTER_PAD": "1"}):
            actual = fixed_m_linear(lambda value: value.square(), source, 8)
        torch.testing.assert_close(actual, source.square(), atol=0, rtol=0)

    def test_autograd_fallback_preserves_gradients(self) -> None:
        source = torch.ones(3, 5, requires_grad=True)
        with patch.dict(os.environ, {"RTP_LLM_FUSED_ROUTER_PAD": "1"}):
            actual = fixed_m_linear(lambda value: value.square(), source, 8)
            actual.sum().backward()
        torch.testing.assert_close(
            source.grad, torch.full_like(source, 2), atol=0, rtol=0
        )

    def test_empty_input_uses_empty_linear(self) -> None:
        source = torch.empty(0, 5)
        actual = fixed_m_linear(lambda value: value + 1, source, 8)
        self.assertEqual(actual.shape, source.shape)


if __name__ == "__main__":
    main()
