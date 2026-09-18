"""GPU tests for the native 4-head QSA indexer scoring kernel.

Three layers of checking:

* the kernel matches an fp64 reference computed on the same quantized inputs
  (products are exact for fp8-in-bf16, so this is a tight bound, not a model);
* window semantics: exactly ``-inf`` outside ``[ks, ke)``, finite inside;
* the kernel agrees with the *reference implementation* the model ships
  (``Qwen4ExpQSAIndexer.block_scores``) when both are fed the same
  quantize-then-dequantize tensors -- this is the "same math" claim, made
  concrete against the module the sparse path is validated against.
"""

import math
import unittest

import torch

from rtp_llm.models_py.modules.qwen4_exp.indexer import Qwen4ExpQSAIndexer
from rtp_llm.models_py.modules.qwen4_exp.indexer_score import (
    qsa_indexer_score,
    quantize_per_head,
)

_H = 4
_D = 128
_HIDDEN = 32
_RATIO = 4
_TOKEN_BUDGET = 32
_EPS = 1e-6
_DEVICE = "cuda"


def _fp8_randn(shape, scale=0.3, device=_DEVICE):
    return (torch.randn(*shape, device=device) * scale).to(torch.float8_e4m3fn)


def _dequant_k(k_fp8, k_scale):
    return k_fp8.float() * k_scale[:, None]


def _reference_logits(q_fp8, weight, k_fp8, k_scale, ks, ke):
    """fp64 transcription of the kernel formula, on the quantized inputs."""
    q = q_fp8.double()
    k = _dequant_k(k_fp8, k_scale).double()
    per_head = torch.relu(q @ k.T)  # [M, H, N]
    out = (per_head * weight.double()[:, :, None]).sum(dim=1)
    columns = torch.arange(k_fp8.shape[0], device=q_fp8.device)
    valid = (columns[None, :] >= ks[:, None]) & (columns[None, :] < ke[:, None])
    return torch.where(valid, out, torch.full_like(out, float("-inf")))


class QsaIndexerScoreTest(unittest.TestCase):
    def test_matches_an_fp64_reference(self):
        torch.manual_seed(0)
        M, N = 70, 133
        q = _fp8_randn((M, _H, _D))
        k = _fp8_randn((N, _D))
        k_scale = torch.rand(N, device=_DEVICE) * 2 + 0.1
        weight = torch.full((M, _H), 1.0 / math.sqrt(_D), device=_DEVICE)
        ks = torch.zeros(M, dtype=torch.int32, device=_DEVICE)
        ke = torch.full((M,), N, dtype=torch.int32, device=_DEVICE)

        logits = qsa_indexer_score(q, weight, k, k_scale, ks, ke)

        expected = _reference_logits(q, weight, k, k_scale, ks, ke)
        torch.testing.assert_close(logits.double(), expected, atol=1e-4, rtol=1e-4)

    def test_windows_are_filled_with_neg_inf(self):
        torch.manual_seed(1)
        M, N = 33, 71
        q = _fp8_randn((M, _H, _D))
        k = _fp8_randn((N, _D))
        k_scale = torch.rand(N, device=_DEVICE) + 0.5
        weight = torch.full((M, _H), 1.0 / math.sqrt(_D), device=_DEVICE)
        ks = torch.randint(0, N, (M,), dtype=torch.int32, device=_DEVICE)
        ke = torch.minimum(
            ks + torch.randint(0, N, (M,), dtype=torch.int32, device=_DEVICE),
            torch.full((M,), N, dtype=torch.int32, device=_DEVICE),
        )
        # Exercise a fully empty window and a zero-length window at the edges.
        ks[0], ke[0] = 5, 5
        ks[1], ke[1] = 0, 0
        ks[2], ke[2] = N, N

        logits = qsa_indexer_score(q, weight, k, k_scale, ks, ke)

        columns = torch.arange(N, device=_DEVICE)
        valid = (columns[None, :] >= ks[:, None]) & (columns[None, :] < ke[:, None])
        self.assertTrue(torch.isinf(logits[~valid]).all())
        self.assertTrue((logits[~valid] < 0).all())
        self.assertTrue(torch.isfinite(logits[valid]).all())

    def test_matches_the_reference_implementation_in_the_quantized_domain(self):
        torch.manual_seed(2)
        seq_len = 16
        indexer = Qwen4ExpQSAIndexer(
            qk_proj=torch.randn((_H + 1) * _D, _HIDDEN, device=_DEVICE) * 0.05,
            q_norm_gamma=torch.rand(_D, device=_DEVICE),
            k_norm_gamma=torch.rand(_D, device=_DEVICE),
            n_heads=_H,
            kv_heads=1,
            head_dim=_D,
            token_budget=_TOKEN_BUDGET,
            compress_ratio=_RATIO,
            norm_eps=_EPS,
        ).to(_DEVICE)

        hidden = torch.randn(1, seq_len, _HIDDEN, device=_DEVICE)
        angles = torch.rand(1, seq_len, _D // 2, device=_DEVICE) * 6.28
        cos = angles.cos().repeat_interleave(2, dim=-1)
        sin = angles.sin().repeat_interleave(2, dim=-1)

        q, raw_keys = indexer.project(hidden)
        block_keys = indexer.pooled_block_keys_all(raw_keys, cos, sin)
        num_blocks = block_keys.shape[1]

        q_fp8, q_scale = quantize_per_head(q[0])
        k_fp8, k_scale = quantize_per_head(block_keys[0])
        weight = q_scale * (1.0 / math.sqrt(_D))
        ks = torch.zeros(seq_len, dtype=torch.int32, device=_DEVICE)
        ke = torch.full((seq_len,), num_blocks, dtype=torch.int32, device=_DEVICE)

        logits = qsa_indexer_score(q_fp8, weight, k_fp8, k_scale, ks, ke)

        k_dequant = _dequant_k(k_fp8, k_scale)
        per_row = [
            indexer.block_scores(q_fp8[i].float() * q_scale[i][:, None], k_dequant)
            for i in range(seq_len)
        ]
        expected = torch.stack(per_row)
        torch.testing.assert_close(logits, expected, atol=1e-4, rtol=1e-4)

    def test_ragged_shapes(self):
        torch.manual_seed(3)
        for M, N in ((1, 1), (5, 3), (65, 129), (64, 64)):
            q = _fp8_randn((M, _H, _D))
            k = _fp8_randn((N, _D))
            k_scale = torch.rand(N, device=_DEVICE) + 0.5
            weight = torch.full((M, _H), 1.0 / math.sqrt(_D), device=_DEVICE)
            ks = torch.zeros(M, dtype=torch.int32, device=_DEVICE)
            ke = torch.full((M,), N, dtype=torch.int32, device=_DEVICE)

            logits = qsa_indexer_score(q, weight, k, k_scale, ks, ke)
            expected = _reference_logits(q, weight, k, k_scale, ks, ke)
            torch.testing.assert_close(logits.double(), expected, atol=1e-4, rtol=1e-4)

    def test_quantize_per_head_round_trips(self):
        torch.manual_seed(4)
        x = torch.randn(7, _H, _D, device=_DEVICE) * 3.0
        x_fp8, scale = quantize_per_head(x)
        restored = x_fp8.float() * scale[:, :, None]
        torch.testing.assert_close(restored, x, rtol=0.1, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
