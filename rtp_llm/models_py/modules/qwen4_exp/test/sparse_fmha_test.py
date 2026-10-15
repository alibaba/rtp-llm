"""GPU tests: triton sparse FMHA vs the torch baseline.

Verifies that the triton kernel produces the same output as
``gather_and_attend`` (the torch reference) on random data with
a small production-like geometry.
"""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_fmha import sparse_prefill_attn
from rtp_llm.models_py.modules.qwen4_exp.test.reference.sparse_fmha import (
    sparse_prefill_attn_torch_reference,
)

_H_Q, _H_KV, _D = 24, 2, 256
_DEV = "cuda"


class SparseFmhaTest(unittest.TestCase):
    def test_automatic_grouping_matches_fp64_with_repeated_keys_and_empty_tiles(self):
        torch.manual_seed(88)
        for gqa in (6, 12):
            with self.subTest(gqa=gqa):
                q = torch.randn(2, 2 * gqa, 131, _D, device=_DEV).bfloat16()
                k = torch.randn(2, 2, 257, _D, device=_DEV).bfloat16()
                v = torch.randn_like(k)
                selected = torch.randint(
                    0, 257, (2, 131, 257), device=_DEV, dtype=torch.int32
                )
                selected[..., 32:224] = -1
                selected[:, 0] = -1
                selected[:, 1, :2] = 31
                selected[:, 2] = -1
                selected[:, 2, -1] = 256
                baseline = sparse_prefill_attn_torch_reference(
                    q.double(), k.double(), v.double(), selected
                ).to(q.dtype)
                expected = sparse_prefill_attn_torch_reference(
                    q.double(), k.double(), v.double(), selected
                ).to(q.dtype)
                candidate = sparse_prefill_attn(q, k, v, selected)
                torch.testing.assert_close(
                    candidate, expected, atol=0.0078125, rtol=0.008
                )
                torch.testing.assert_close(
                    candidate, baseline, atol=0.0078125, rtol=0.008
                )
                self.assertTrue((candidate[:, :, 0] == 0).all())

    def test_sparse_grouping_preserves_sparse_tails_duplicates_and_empty_rows(self):
        torch.manual_seed(73)
        for gqa in (6, 12):
            with self.subTest(gqa=gqa):
                q = torch.randn(2, 2 * gqa, 131, _D, device=_DEV).bfloat16()
                k = torch.randn(2, 2, 257, _D, device=_DEV).bfloat16()
                v = torch.randn_like(k)
                selected = torch.randint(
                    0, 257, (2, 131, 257), device=_DEV, dtype=torch.int32
                )
                selected[..., 32:224] = -1
                selected[:, 0] = -1
                selected[:, 1] = -1
                selected[:, 1, -1] = 256
                selected[:, 2, :2] = 31  # Multiplicity must be retained.
                selected[:, 3, :32] = -7  # All negative IDs are padding.
                baseline = sparse_prefill_attn_torch_reference(
                    q.double(), k.double(), v.double(), selected
                ).to(q.dtype)
                candidate = sparse_prefill_attn(q, k, v, selected)
                old_tf32 = torch.backends.cuda.matmul.allow_tf32
                try:
                    torch.backends.cuda.matmul.allow_tf32 = False
                    expected = sparse_prefill_attn_torch_reference(
                        q.float(), k.float(), v.float(), selected
                    ).to(q.dtype)
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = old_tf32
                torch.testing.assert_close(
                    candidate, expected, atol=0.0078125, rtol=0.008
                )
                torch.testing.assert_close(
                    candidate, baseline, atol=0.0078125, rtol=0.008
                )
                self.assertTrue((candidate[:, :, 0] == 0).all())

    def test_grouped_tensor_core_matches_reference_with_head_padding(self):
        torch.manual_seed(18)
        # TP8 uses three Q heads per local KV head. Also cover two local KV
        # heads, sparse tails, duplicate selections, and an entirely empty row.
        for heads, kv_heads in ((3, 1), (6, 2), (24, 2)):
            for width in (1, 33, 65, 257):
                with self.subTest(heads=heads, kv_heads=kv_heads, width=width):
                    q = torch.randn(2, heads, 5, _D, device=_DEV).to(torch.bfloat16)
                    k = torch.randn(2, kv_heads, 512, _D, device=_DEV).to(
                        torch.bfloat16
                    )
                    v = torch.randn_like(k)
                    selected = torch.randint(
                        0, 512, (2, 5, width), device=_DEV, dtype=torch.int32
                    )
                    selected[:, 0] = -1
                    selected[:, 1, width // 2 :] = -1
                    actual = sparse_prefill_attn(q, k, v, selected)
                    expected = sparse_prefill_attn_torch_reference(q, k, v, selected)
                    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
                    self.assertTrue((actual[:, :, 0] == 0).all())

    def test_grouped_tensor_core_matches_online_on_long_sparse_rows(self):
        torch.manual_seed(19)
        q = torch.randn(1, 3, 7, _D, device=_DEV).to(torch.bfloat16)
        k = torch.randn(1, 1, 2048, _D, device=_DEV).to(torch.bfloat16)
        v = torch.randn_like(k)
        selected = torch.randint(0, 2048, (1, 7, 1027), device=_DEV, dtype=torch.int32)
        selected[:, 0] = -1
        selected[:, 1, 17:] = -1
        expected = sparse_prefill_attn_torch_reference(
            q.double(), k.double(), v.double(), selected
        ).to(q.dtype)
        actual = sparse_prefill_attn(q, k, v, selected)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    def test_auto_grouping_matches_reference_on_long_tp8_prefill(self):
        torch.manual_seed(20)
        q = torch.randn(1, 3, 131, _D, device=_DEV).to(torch.bfloat16)
        k = torch.randn(1, 1, 257, _D, device=_DEV).to(torch.bfloat16)
        v = torch.randn_like(k)
        selected = torch.randint(0, 257, (1, 131, 257), device=_DEV, dtype=torch.int32)
        selected[:, 0] = -1
        actual = sparse_prefill_attn(q, k, v, selected)
        expected = sparse_prefill_attn_torch_reference(q, k, v, selected)
        torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    def test_grouping_falls_back_without_nvidia_bf16_tensor_cores(self):
        q = torch.randn(1, 3, 2, _D, device=_DEV).to(torch.bfloat16)
        k = torch.randn(1, 1, 16, _D, device=_DEV).to(torch.bfloat16)
        v = torch.randn_like(k)
        selected = torch.arange(16, device=_DEV, dtype=torch.int32)[
            None, None, :
        ].expand(1, 2, 16)
        for backend in ("old_cuda", "hip"):
            with self.subTest(backend=backend):
                with (
                    patch("torch.cuda.get_device_capability", return_value=(7, 0)),
                    patch("torch.version.hip", "hip" if backend == "hip" else None),
                    patch(
                        "rtp_llm.models_py.modules.qwen4_exp.qsa.sparse_fmha._sparse_prefill_grouped_kernel",
                        new=object(),
                    ),
                ):
                    actual = sparse_prefill_attn(q, k, v, selected)
                expected = sparse_prefill_attn_torch_reference(q, k, v, selected)
                torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)

    def test_matches_torch_reference(self):
        torch.manual_seed(0)
        B, S, T, K = 1, 8, 64, 16
        q = torch.randn(B, _H_Q, S, _D, device=_DEV).to(torch.bfloat16)
        k = torch.randn(B, _H_KV, T, _D, device=_DEV).to(torch.bfloat16)
        v = (torch.randn(B, _H_KV, T, _D, device=_DEV) * 0.3).to(torch.bfloat16)

        # causal selection
        sel = -torch.ones(B, S, K, dtype=torch.int32, device=_DEV)
        for b in range(B):
            for s in range(S):
                n = min(K - 1, s + 1)
                sel[b, s, :n] = torch.arange(n, device=_DEV, dtype=torch.int32)

        triton_out = sparse_prefill_attn(q, k, v, sel)
        torch_out = sparse_prefill_attn_torch_reference(q, k, v, sel)

        torch.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=1e-2)

    def test_all_masked_produces_zero(self):
        B, S, K = 1, 4, 8
        q = torch.randn(B, _H_Q, S, _D, device=_DEV).to(torch.bfloat16)
        k = torch.randn(B, _H_KV, S, _D, device=_DEV).to(torch.bfloat16)
        v = torch.randn(B, _H_KV, S, _D, device=_DEV).to(torch.bfloat16)
        sel = -torch.ones(B, S, K, dtype=torch.int32, device=_DEV)

        triton_out = sparse_prefill_attn(q, k, v, sel)
        self.assertTrue((triton_out == 0).all())

    def test_out_of_range_selected_index_is_rejected(self):
        B, S, K = 1, 4, 8
        q = torch.randn(B, _H_Q, S, _D, device=_DEV).to(torch.bfloat16)
        k = torch.randn(B, _H_KV, S, _D, device=_DEV).to(torch.bfloat16)
        v = torch.randn(B, _H_KV, S, _D, device=_DEV).to(torch.bfloat16)
        selected = -torch.ones(B, S, K, dtype=torch.int32, device=_DEV)
        selected[0, 0, 0] = S

        with self.assertRaisesRegex(ValueError, "outside KV sequence length"):
            sparse_prefill_attn(q, k, v, selected)

    def test_matches_torch_on_larger_shape(self):
        torch.manual_seed(1)
        B, S, T, K = 2, 16, 128, 32
        q = (torch.randn(B, _H_Q, S, _D, device=_DEV) * 0.3).to(torch.bfloat16)
        k = (torch.randn(B, _H_KV, T, _D, device=_DEV) * 0.3).to(torch.bfloat16)
        v = (torch.randn(B, _H_KV, T, _D, device=_DEV) * 0.3).to(torch.bfloat16)
        sel = torch.randint(0, T, (B, S, K), dtype=torch.int32, device=_DEV)
        sel[:, :, : K // 2] = -1  # mix in some padding

        triton_out = sparse_prefill_attn(q, k, v, sel)
        torch_out = sparse_prefill_attn_torch_reference(q, k, v, sel)
        torch.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=1e-2)

    def test_online_matches_two_pass_across_sparse_widths(self):
        torch.manual_seed(11)
        for width in (1, 63, 64, 65, 257, 1027):
            with self.subTest(width=width):
                q = torch.randn(1, _H_Q, 3, _D, device=_DEV).to(torch.bfloat16)
                k = torch.randn(1, _H_KV, 2048, _D, device=_DEV).to(torch.bfloat16)
                v = torch.randn_like(k)
                selected = torch.randint(
                    0, 2048, (1, 3, width), device=_DEV, dtype=torch.int32
                )
                selected[:, 0] = -1
                selected[:, 1, width // 2 :] = -1
                baseline = sparse_prefill_attn_torch_reference(
                    q.double(), k.double(), v.double(), selected
                ).to(q.dtype)
                candidate = sparse_prefill_attn(q, k, v, selected)
                torch.testing.assert_close(candidate, baseline, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
