import unittest

import torch

from rtp_llm.models_py.triton_kernels.common.scatter_qkv import scatter_qkv


def _split_view_baseline(
    mixed_qkv: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
):
    """Reference implementation: torch.split + view (the path scatter_qkv replaces).

    This is the exact code that lived in Qwen3NextGatedDeltaNetPrefill._fla
    before the scatter_qkv optimization landed.
    """
    M = mixed_qkv.shape[0]
    k_dim = num_k_heads * head_k_dim
    v_dim = num_v_heads * head_v_dim
    q, k, v = torch.split(mixed_qkv, [k_dim, k_dim, v_dim], dim=-1)
    q = q.view(1, M, num_k_heads, head_k_dim).contiguous()
    k = k.view(1, M, num_k_heads, head_k_dim).contiguous()
    v = v.view(1, M, num_v_heads, head_v_dim).contiguous()
    return q, k, v


class TestScatterQKV(unittest.TestCase):
    """Equivalence tests for scatter_qkv vs torch.split + view baseline.

    scatter_qkv is a pure data-movement kernel (no math), so equality must be
    bit-exact (torch.equal), not numerically close.
    """

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        self.device = torch.device("cuda:0")

    def _assert_equivalent(
        self,
        M: int,
        num_k: int,
        num_v: int,
        head_k: int,
        head_v: int,
        dtype: torch.dtype,
    ) -> None:
        torch.manual_seed(M * 1000 + num_k * 100 + num_v * 10 + head_k)
        total = 2 * num_k * head_k + num_v * head_v
        x = torch.randn(M, total, dtype=dtype, device=self.device)

        q_ref, k_ref, v_ref = _split_view_baseline(x, num_k, num_v, head_k, head_v)
        q_new, k_new, v_new = scatter_qkv(x, num_k, num_v, head_k, head_v)

        for name, new, ref in (
            ("q", q_new, q_ref),
            ("k", k_new, k_ref),
            ("v", v_new, v_ref),
        ):
            self.assertEqual(new.shape, ref.shape, f"{name} shape mismatch")
            self.assertEqual(new.dtype, ref.dtype, f"{name} dtype mismatch")
            self.assertTrue(new.is_contiguous(), f"{name} must be contiguous")
            self.assertTrue(
                torch.equal(new, ref),
                f"{name} value mismatch (max abs diff = {(new - ref).abs().max().item()})",
            )

    def test_qwen3_next_tp2_prefill_15k(self) -> None:
        """Production case: Qwen3.5-9B TP=2 GDN, 15K-token prefill (the trace baseline)."""
        self._assert_equivalent(15384, 8, 16, 128, 128, torch.bfloat16)

    def test_threshold_boundary_M(self) -> None:
        """qwen3_next.py uses M>=2048 to gate scatter_qkv; the kernel itself must
        produce correct output at the boundary regardless."""
        for M in (2047, 2048, 2049):
            with self.subTest(M=M):
                self._assert_equivalent(M, 8, 16, 128, 128, torch.bfloat16)

    def test_dtypes(self) -> None:
        """bf16 is the production dtype; fp16/fp32 should also work."""
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                self._assert_equivalent(4096, 8, 16, 128, 128, dtype)

    def test_head_configurations(self) -> None:
        """Cover Qwen3-Next TP=1/2/4 head splits, plus a few atypical layouts."""
        configs = [
            # (num_k_heads, num_v_heads, head_k_dim, head_v_dim)
            (16, 32, 128, 128),  # Qwen3-Next TP=1
            (8, 16, 128, 128),  # Qwen3-Next TP=2
            (4, 8, 128, 128),  # Qwen3-Next TP=4
            (2, 4, 128, 128),  # small MQA-like
            (4, 4, 64, 64),  # 1:1 head ratio, smaller head dim
        ]
        for num_k, num_v, head_k, head_v in configs:
            with self.subTest(num_k=num_k, num_v=num_v, head_k=head_k, head_v=head_v):
                self._assert_equivalent(
                    4096, num_k, num_v, head_k, head_v, torch.bfloat16
                )

    def test_small_M(self) -> None:
        """Even though scatter_qkv is gated to M>=2048 in production, the kernel
        itself must still be correct at small M (chunked-prefill / unit tests)."""
        for M in (1, 16, 256, 1024):
            with self.subTest(M=M):
                self._assert_equivalent(M, 8, 16, 128, 128, torch.bfloat16)

    def test_qwen3_next_call_path(self) -> None:
        """Mirror the exact call site in Qwen3NextGatedDeltaNetPrefill._fla:
        mixed_qkv shape (M, k_dim*2 + v_dim) contiguous from causal_conv1d.transpose
        -> scatter_qkv -> (q, k, v) shaped (1, M, n_heads, head_dim) contig
        """
        M = 4096
        num_k, num_v, head_dim = 8, 16, 128
        total = 2 * num_k * head_dim + num_v * head_dim
        mixed_qkv = torch.randn(M, total, dtype=torch.bfloat16, device=self.device)

        q, k, v = scatter_qkv(mixed_qkv, num_k, num_v, head_dim, head_dim)
        self.assertEqual(tuple(q.shape), (1, M, num_k, head_dim))
        self.assertEqual(tuple(k.shape), (1, M, num_k, head_dim))
        self.assertEqual(tuple(v.shape), (1, M, num_v, head_dim))
        self.assertTrue(q.is_contiguous() and k.is_contiguous() and v.is_contiguous())

        q_ref, k_ref, v_ref = _split_view_baseline(
            mixed_qkv, num_k, num_v, head_dim, head_dim
        )
        self.assertTrue(torch.equal(q, q_ref))
        self.assertTrue(torch.equal(k, k_ref))
        self.assertTrue(torch.equal(v, v_ref))

    def test_input_assertions(self) -> None:
        """scatter_qkv must reject malformed inputs."""
        # 3D input
        with self.assertRaises(ValueError):
            scatter_qkv(
                torch.randn(2, 4096, 4096, dtype=torch.bfloat16, device=self.device),
                8,
                16,
                128,
                128,
            )
        # Non-contiguous input
        x = torch.randn(15384, 8192, dtype=torch.bfloat16, device=self.device)
        with self.assertRaises(ValueError):
            scatter_qkv(x[:, :4096], 8, 16, 128, 128)  # slice -> non-contig
        # Wrong last-dim
        with self.assertRaises(ValueError):
            scatter_qkv(
                torch.randn(4096, 4097, dtype=torch.bfloat16, device=self.device),
                8,
                16,
                128,
                128,
            )


if __name__ == "__main__":
    unittest.main()
