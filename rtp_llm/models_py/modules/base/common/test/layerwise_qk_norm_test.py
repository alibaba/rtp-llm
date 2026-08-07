import itertools
from unittest import SkipTest, TestCase, main

import torch
from torch import dtype as _dtype


def _hf_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference: variance over the last dim, multiply by weight."""
    x_f = x.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    return (x_f * torch.rsqrt(var + eps)).to(x.dtype) * weight


class LayerwiseQKRMSNormTest(TestCase):
    DTYPES = [torch.bfloat16]
    NUM_TOKENS = [1, 4, 16, 1024]
    SHAPES = [
        # (head_num, kv_head_num, head_dim)
        (48, 8, 128),
        (24, 4, 128),
        (12, 2, 128),
        (8, 2, 64),
    ]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA not available")
        torch.set_default_device("cuda")

    def _run(
        self, m: int, head_num: int, kv_head_num: int, head_dim: int, dtype: _dtype
    ):
        from rtp_llm.models_py.modules.base.common.norm import LayerwiseQKRMSNorm

        eps = 1e-6
        torch.manual_seed(0)

        q_size = head_num * head_dim
        kv_size = kv_head_num * head_dim
        n = q_size + 2 * kv_size

        qkv = torch.randn(m, n, dtype=dtype)
        q_gamma = torch.randn(q_size, dtype=dtype)
        k_gamma = torch.randn(kv_size, dtype=dtype)

        mod = LayerwiseQKRMSNorm(
            q_gamma,
            k_gamma,
            head_num,
            kv_head_num,
            head_dim,
            eps,
            tp_size=1,
            tp_rank=0,
        )
        out = mod(qkv.clone())

        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        ref = torch.cat(
            [_hf_rms_norm(q, q_gamma, eps), _hf_rms_norm(k, k_gamma, eps), v], dim=-1
        )

        self.assertTrue(
            torch.allclose(ref, out, atol=5e-3, rtol=1e-2),
            f"max diff {(out.float() - ref.float()).abs().max().item():.6f}",
        )

    def test_layerwise_qk_rms_norm(self):
        for params in itertools.product(self.NUM_TOKENS, self.SHAPES, self.DTYPES):
            m, (h, kv, d), dtype = params
            with self.subTest(m=m, head_num=h, kv_head_num=kv, head_dim=d, dtype=dtype):
                self._run(m, h, kv, d, dtype)


class LayerwiseQKRMSNormTPTest(TestCase):
    """Verify tp_size>1 path: each rank's slice, with mocked all_reduce on the
    combined [m, 2] sum_sq tensor, reassembles into the full reference."""

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA not available")
        torch.set_default_device("cuda")

    def _run_tp(self, tp_size, head_num_total, kv_head_num_total, head_dim, m=4):
        from unittest.mock import patch

        from rtp_llm.models_py.modules.base.common.norm import LayerwiseQKRMSNorm

        eps = 1e-6
        head_num_per_rank = head_num_total // tp_size
        kv_per_rank = kv_head_num_total // tp_size
        q_size_total = head_num_total * head_dim
        kv_size_total = kv_head_num_total * head_dim
        q_per_rank = head_num_per_rank * head_dim
        kv_per_rank_sz = kv_per_rank * head_dim

        torch.manual_seed(42)
        q_gamma = torch.randn(q_size_total, dtype=torch.bfloat16)
        k_gamma = torch.randn(kv_size_total, dtype=torch.bfloat16)
        qkv_full = torch.randn(
            m, q_size_total + 2 * kv_size_total, dtype=torch.bfloat16
        )

        q_full, k_full, v_full = qkv_full.split(
            [q_size_total, kv_size_total, kv_size_total], dim=-1
        )
        ref = torch.cat(
            [
                _hf_rms_norm(q_full, q_gamma, eps),
                _hf_rms_norm(k_full, k_gamma, eps),
                v_full,
            ],
            dim=-1,
        )

        # Triton path issues a single all_reduce on the combined [m, 2] sum_sq.
        q_global_sum_sq = q_full.float().pow(2).sum(dim=-1, keepdim=True)
        k_global_sum_sq = k_full.float().pow(2).sum(dim=-1, keepdim=True)
        combined = torch.cat([q_global_sum_sq, k_global_sum_sq], dim=-1)

        def _mock_all_reduce(tensor, **kwargs):
            assert (
                tensor.dim() == 2 and tensor.shape[-1] == 2
            ), f"expected combined [m, 2] AR, got {tensor.shape}"
            return combined

        rank_outs = []
        for rank in range(tp_size):
            q_slice = q_full[:, rank * q_per_rank : (rank + 1) * q_per_rank]
            k_slice = k_full[:, rank * kv_per_rank_sz : (rank + 1) * kv_per_rank_sz]
            v_slice = v_full[:, rank * kv_per_rank_sz : (rank + 1) * kv_per_rank_sz]
            qkv_rank = torch.cat([q_slice, k_slice, v_slice], dim=-1)

            with patch(
                "rtp_llm.models_py.distributed.collective_torch.all_reduce",
                side_effect=_mock_all_reduce,
            ):
                mod = LayerwiseQKRMSNorm(
                    q_gamma.clone(),
                    k_gamma.clone(),
                    head_num_per_rank,
                    kv_per_rank,
                    head_dim,
                    eps,
                    tp_size=tp_size,
                    tp_rank=rank,
                )
                rank_outs.append(mod(qkv_rank.clone()))

        q_outs = torch.cat([r[:, :q_per_rank] for r in rank_outs], dim=-1)
        k_outs = torch.cat(
            [r[:, q_per_rank : q_per_rank + kv_per_rank_sz] for r in rank_outs], dim=-1
        )
        v_outs = torch.cat(
            [r[:, q_per_rank + kv_per_rank_sz :] for r in rank_outs], dim=-1
        )
        assembled = torch.cat([q_outs, k_outs, v_outs], dim=-1)
        max_diff = (assembled.float() - ref.float()).abs().max().item()
        self.assertTrue(
            torch.allclose(ref, assembled, atol=5e-3, rtol=1e-2),
            f"TP={tp_size} reassembled max diff {max_diff:.6f}",
        )

    def test_tp2_small(self):
        self._run_tp(2, 8, 2, 64)

    def test_tp4_m2_shapes(self):
        # MiniMax-M2 attention: 48 heads, 8 kv-heads, head_dim=128.
        self._run_tp(4, 48, 8, 128, m=8)

    def test_tp8_m2_shapes(self):
        self._run_tp(8, 48, 8, 128, m=2)


if __name__ == "__main__":
    main()
