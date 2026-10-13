from __future__ import annotations

import importlib.util
import os
import unittest

import torch


def _load_rope_only():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.abspath(os.path.join(here, "..", "_rope_only_triton.py"))
    spec = importlib.util.spec_from_file_location("_rope_only_triton", src)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.rope_only_inplace


def _make_freqs(rows: int, rd: int) -> torch.Tensor:
    angle = torch.rand(rows, rd // 2, device="cuda") * 6.28
    return torch.polar(torch.ones_like(angle), angle).to(torch.complex64).contiguous()


def _eager_apply_rope_inplace(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    y = x
    xc = torch.view_as_complex(x.float().unflatten(-1, (x.size(-1) // 2, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    freqs_flat = freqs_cis.reshape(-1, freqs_cis.shape[-1])
    if xc.ndim == 4:
        if freqs_flat.shape[0] == xc.size(0):
            freqs = freqs_flat.view(xc.size(0), 1, 1, xc.size(-1))
        else:
            freqs = freqs_flat.view(xc.size(0), xc.size(1), 1, xc.size(-1))
    elif freqs_flat.shape[0] == xc.size(0):
        freqs = freqs_flat.view(xc.size(0), 1, xc.size(-1))
    else:
        freqs = freqs_flat.view(xc.size(0), xc.size(1), xc.size(-1))
    out = torch.view_as_real(xc * freqs).flatten(-2)
    y.copy_(out)
    return y


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class RopeOnlyTest(unittest.TestCase):
    def _check(self, *, shape, rd: int, inverse: bool, group_heads: int | None = None):
        torch.manual_seed(123 + len(shape) + rd + int(inverse))
        base = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
        ref = base.clone()
        cand = base.clone()
        freqs = _make_freqs(shape[1], rd)

        _eager_apply_rope_inplace(ref[..., -rd:], freqs, inverse=inverse)
        rope_only = _load_rope_only()
        rope_only(cand[..., -rd:], freqs, inverse=inverse, group_heads=group_heads)
        torch.cuda.synchronize()

        torch.testing.assert_close(cand, ref, rtol=0, atol=3e-2)

    def test_q_prefill_tail_view_grouped(self):
        self._check(shape=(1, 128, 64, 128), rd=64, inverse=False, group_heads=8)

    def test_q_prefill_tail_view_scalar(self):
        self._check(shape=(1, 96, 7, 128), rd=64, inverse=False, group_heads=1)

    def test_inverse(self):
        self._check(shape=(1, 64, 16, 128), rd=64, inverse=True, group_heads=4)

    def test_decode_speculative_batched_shape(self):
        torch.manual_seed(125)
        bsz, q_len, heads, head_dim, rd = 3, 2, 8, 128, 64
        base = torch.randn(
            bsz, q_len, heads, head_dim, dtype=torch.bfloat16, device="cuda"
        )
        ref = base.clone()
        cand = base.clone()
        freqs = _make_freqs(bsz * q_len, rd)

        _eager_apply_rope_inplace(ref[..., -rd:], freqs)
        rope_only = _load_rope_only()
        rope_only(cand[..., -rd:], freqs, group_heads=4)
        torch.cuda.synchronize()

        torch.testing.assert_close(cand, ref, rtol=0, atol=3e-2)

    def test_decode_flat_indexer_shape(self):
        torch.manual_seed(124)
        tokens, heads, head_dim, rd = 6, 8, 128, 64
        base = torch.randn(tokens, heads, head_dim, dtype=torch.bfloat16, device="cuda")
        ref = base.clone()
        cand = base.clone()
        freqs = _make_freqs(tokens, rd)

        _eager_apply_rope_inplace(ref[..., -rd:], freqs)
        rope_only = _load_rope_only()
        rope_only(cand[..., -rd:], freqs, group_heads=4)
        torch.cuda.synchronize()

        torch.testing.assert_close(cand, ref, rtol=0, atol=3e-2)

    def test_empty_noop(self):
        x = torch.empty(0, 64, dtype=torch.bfloat16, device="cuda")
        freqs = torch.empty(0, 32, dtype=torch.complex64, device="cuda")
        rope_only = _load_rope_only()
        out = rope_only(x, freqs)
        self.assertIs(out, x)


if __name__ == "__main__":
    unittest.main()
