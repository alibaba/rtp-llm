import unittest
from types import SimpleNamespace

import torch
from torch import nn

from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpModel
from rtp_llm.models_py.model_desc.qwen4_exp_mtp import (
    Qwen4ExpMTPInputProjection,
    Qwen4ExpMTPModel,
)


class _Matmul(nn.Module):
    def __init__(self, runtime_weight: torch.Tensor):
        super().__init__()
        self.runtime_weight = runtime_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.runtime_weight


def _raw_gamma_rms_norm(
    value: torch.Tensor, gamma: torch.Tensor, eps: float
) -> torch.Tensor:
    normalized = value.float() * torch.rsqrt(
        value.float().pow(2).mean(-1, keepdim=True) + eps
    )
    return (normalized * (1.0 + gamma.float())).type_as(value)


class Qwen4ExpMTPInputProjectionTest(unittest.TestCase):
    def test_two_projection_forward_matches_small_shape_reference(self):
        torch.manual_seed(11)
        tokens, hc_mult, hidden_size = 3, 2, 4
        eps = 1e-6
        embedding = torch.randn(tokens, hidden_size, dtype=torch.float32)
        target_hidden = torch.randn(tokens, hc_mult * hidden_size, dtype=torch.float32)
        embedding_gamma = torch.tensor(
            [0.003, -0.007, 0.011, -0.013], dtype=torch.bfloat16
        )
        hidden_gamma = torch.tensor(
            [-0.005, 0.009, -0.015, 0.017, 0.019, -0.021, 0.023, -0.025],
            dtype=torch.bfloat16,
        )
        # Runtime linear weights have already transposed checkpoint [out, in]
        # tensors and are therefore [in, out].
        embedding_fc = torch.randn(hidden_size, hidden_size)
        hidden_fc = torch.randn(hidden_size, hidden_size)
        component = Qwen4ExpMTPInputProjection(
            embedding_gamma,
            hidden_gamma,
            _Matmul(embedding_fc),
            _Matmul(hidden_fc),
            hidden_size=hidden_size,
            norm_eps=eps,
        )

        actual = component(embedding, target_hidden)
        expected_embedding = (
            _raw_gamma_rms_norm(embedding, embedding_gamma, eps) @ embedding_fc
        ).unsqueeze(1)
        expected_hidden = (
            _raw_gamma_rms_norm(
                target_hidden.reshape(tokens, hc_mult, hidden_size),
                hidden_gamma.reshape(hc_mult, hidden_size),
                eps,
            ).reshape(-1, hidden_size)
            @ hidden_fc
        ).reshape(tokens, hc_mult, hidden_size)
        expected = (expected_embedding + expected_hidden).flatten(-2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        # Pooling the branches before fc_hidden is a tempting shape workaround,
        # but it erases the released head's hyper-connection semantics.
        pooled_hidden = (
            _raw_gamma_rms_norm(
                target_hidden.reshape(tokens, hc_mult, hidden_size),
                hidden_gamma.reshape(hc_mult, hidden_size),
                eps,
            ).mean(1)
            @ hidden_fc
        )
        wrongly_pooled = (
            (expected_embedding + pooled_hidden.unsqueeze(1))
            .expand(-1, hc_mult, -1)
            .flatten(-2)
        )
        self.assertGreater(float((actual - wrongly_pooled).abs().max()), 1e-5)

        # Folding +1 into BF16 is observably different from upstream's fp32
        # addition and would make this parity anchor fail.
        folded_embedding = embedding_gamma + torch.ones_like(embedding_gamma)
        folded_hidden = hidden_gamma + torch.ones_like(hidden_gamma)
        normalized_embedding = embedding * torch.rsqrt(
            embedding.pow(2).mean(-1, keepdim=True) + eps
        )
        hidden_groups = target_hidden.reshape(tokens, hc_mult, hidden_size)
        normalized_hidden = hidden_groups * torch.rsqrt(
            hidden_groups.pow(2).mean(-1, keepdim=True) + eps
        )
        folded_embedding_out = (
            normalized_embedding * folded_embedding.float()
        ) @ embedding_fc
        folded_hidden_out = (
            (
                normalized_hidden * folded_hidden.reshape(hc_mult, hidden_size).float()
            ).reshape(-1, hidden_size)
            @ hidden_fc
        ).reshape(tokens, hc_mult, hidden_size)
        folded = (folded_embedding_out.unsqueeze(1) + folded_hidden_out).flatten(-2)
        self.assertGreater(float((actual - folded).abs().max()), 1e-5)

    def test_rejects_non_packed_or_mismatched_target_hidden_input(self):
        component = Qwen4ExpMTPInputProjection(
            torch.zeros(4),
            torch.zeros(8),
            nn.Identity(),
            nn.Identity(),
            hidden_size=4,
            norm_eps=1e-6,
        )
        with self.assertRaisesRegex(RuntimeError, "packed 2-D"):
            component(torch.zeros(1, 1, 4), torch.zeros(1, 1, 8))
        with self.assertRaisesRegex(RuntimeError, "row counts differ"):
            component(torch.zeros(2, 4), torch.zeros(1, 8))
        with self.assertRaisesRegex(RuntimeError, "embedding width mismatch"):
            component(torch.zeros(2, 3), torch.zeros(2, 8))
        with self.assertRaisesRegex(RuntimeError, "target hidden width mismatch"):
            component(torch.zeros(2, 4), torch.zeros(2, 4))

    def test_target_tiles_once_and_mtp_projection_is_not_retiled(self):
        target = SimpleNamespace(
            config=SimpleNamespace(hidden_size=4),
            hc_mult=2,
        )
        embedding = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        tiled = Qwen4ExpModel._initial_hyper_states(target, embedding)
        torch.testing.assert_close(tiled, embedding.repeat(1, 2))

        draft = SimpleNamespace(config=SimpleNamespace(hidden_size=4, hc_mult=2))
        projected = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        actual = Qwen4ExpMTPModel._initial_hyper_states(draft, projected)
        self.assertEqual(actual.data_ptr(), projected.data_ptr())

        with self.assertRaisesRegex(RuntimeError, "projected residual width mismatch"):
            Qwen4ExpMTPModel._initial_hyper_states(draft, embedding)

    def test_target_hidden_accessor_exposes_precollapse_rows(self):
        wide = torch.arange(24, dtype=torch.float32).reshape(3, 8)
        model = SimpleNamespace(_mtp_target_hidden_states=wide)

        actual = Qwen4ExpModel.get_mtp_target_hidden_states(model, 2)
        torch.testing.assert_close(actual, wide[:2])
        torch.testing.assert_close(
            Qwen4ExpModel.get_mtp_target_hidden_states(model, -1), wide
        )
        with self.assertRaisesRegex(RuntimeError, "exceed the last forward"):
            Qwen4ExpModel.get_mtp_target_hidden_states(model, 4)


if __name__ == "__main__":
    unittest.main()
