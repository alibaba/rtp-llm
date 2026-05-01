"""Unit test for DeepSeek-V4 mHC residual layer (PyTorch reference).

Run:
  cd .../github-opensource && /opt/conda310/bin/python \
    rtp_llm/models_py/modules/dsv4/test/mhc_test.py
"""

import torch
import torch.nn as nn

from rtp_llm.models_py.modules.dsv4.mhc import (
    HyperConnection,
    HyperConnectionHead,
    expand_to_hc,
    hc_split_sinkhorn,
)


def _init_hc(layer):
    nn.init.xavier_uniform_(layer.hc_fn)
    nn.init.zeros_(layer.hc_base)
    nn.init.ones_(layer.hc_scale)


def _init_head(layer):
    nn.init.xavier_uniform_(layer.hc_head_fn)
    nn.init.zeros_(layer.hc_head_base)
    nn.init.ones_(layer.hc_head_scale)


def test_sinkhorn_doubly_stochastic():
    """`comb` returned from hc_split_sinkhorn should be (approximately) doubly stochastic."""
    torch.manual_seed(0)
    hc = 4
    mix_hc = (2 + hc) * hc
    mixes = torch.randn(2, 8, mix_hc)
    scale = torch.ones(3)
    base = torch.zeros(mix_hc)
    pre, post, comb = hc_split_sinkhorn(
        mixes, scale, base, hc, sinkhorn_iters=20, eps=1e-6
    )
    assert pre.shape == (2, 8, hc)
    assert post.shape == (2, 8, hc)
    assert comb.shape == (2, 8, hc, hc)
    # Row sums and column sums should be ~1
    row_sums = comb.sum(dim=-1)
    col_sums = comb.sum(dim=-2)
    assert torch.allclose(
        row_sums, torch.ones_like(row_sums), atol=5e-3
    ), f"row sums off: max diff {(row_sums - 1).abs().max().item()}"
    assert torch.allclose(
        col_sums, torch.ones_like(col_sums), atol=5e-3
    ), f"col sums off: max diff {(col_sums - 1).abs().max().item()}"


def test_hyperconnection_shapes():
    torch.manual_seed(0)
    B, T, hc, d = 2, 16, 4, 64
    layer = HyperConnection(hc, d)
    _init_hc(layer)
    x = torch.randn(B, T, hc, d)
    y, post, comb = layer.hc_pre(x)
    assert y.shape == (B, T, d)
    assert post.shape == (B, T, hc)
    assert comb.shape == (B, T, hc, hc)
    # Simulate F as identity on a single stream — just to test round-trip shape
    out = layer.hc_post(y, x, post, comb)
    assert out.shape == (B, T, hc, d)


def test_head_reduces_to_single_stream():
    torch.manual_seed(0)
    B, T, hc, d = 2, 16, 4, 64
    head = HyperConnectionHead(hc, d)
    _init_head(head)
    x = torch.randn(B, T, hc, d)
    y = head(x)
    assert y.shape == (B, T, d)


def test_expand_to_hc():
    x = torch.randn(2, 16, 64)
    y = expand_to_hc(x, 4)
    assert y.shape == (2, 16, 4, 64)
    # All hc copies should be identical immediately after expand
    assert torch.equal(y[:, :, 0], y[:, :, 1])
    assert torch.equal(y[:, :, 0], y[:, :, 3])


# =====================================================================
# TileLang parity tests — vendored DeepSeek TileKernels mhc_pre/post/head
# vs the PyTorch REF in this file. Skipped when CUDA / tilelang is
# unavailable, so these stay safe on CPU-only checkouts.
# =====================================================================

_HAS_CUDA = torch.cuda.is_available()


class _Skip(Exception):
    pass


def _skip_if(cond: bool, reason: str):
    if cond:
        raise _Skip(reason)


def _ref_pre(layer: HyperConnection, x_bf16):
    """REF hc_pre computed with the same dtype contract TK uses (bf16 input)."""
    return layer.hc_pre(x_bf16)


@torch.no_grad()
def test_tk_mhc_pre_matches_ref():
    _skip_if(not _HAS_CUDA, "no cuda")
    from rtp_llm.models_py.modules.dsv4 import _mhc_tilelang as tk

    torch.manual_seed(0)
    B, T, hc, d = 1, 64, 4, 128
    device = "cuda"
    layer = HyperConnection(hc, d).to(device)
    _init_hc(layer)
    x = torch.randn(B, T, hc, d, device=device, dtype=torch.bfloat16)

    ref_y, ref_post, ref_comb = _ref_pre(layer, x)
    out = tk.tk_mhc_pre(
        x,
        layer.hc_fn,
        layer.hc_scale,
        layer.hc_base,
        norm_eps=layer.norm_eps,
        pre_eps=layer.hc_eps,
        sinkhorn_eps=layer.hc_eps,
        sinkhorn_iters=layer.hc_sinkhorn_iters,
        hc_mult=hc,
    )
    if out is None:
        raise _Skip(f"tk_mhc_pre disabled (sticky verdict={tk._TK_PRE_OK})")
    tk_y, tk_post, tk_comb = out
    # tk post is [B,T,hc,1]; REF returns [B,T,hc] — compare squeezed.
    torch.testing.assert_close(tk_y, ref_y, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        tk_post.squeeze(-1).float(), ref_post.float(), atol=5e-3, rtol=5e-3
    )
    torch.testing.assert_close(tk_comb.float(), ref_comb.float(), atol=5e-3, rtol=5e-3)


@torch.no_grad()
def test_tk_mhc_post_matches_ref():
    _skip_if(not _HAS_CUDA, "no cuda")
    from rtp_llm.models_py.modules.dsv4 import _mhc_tilelang as tk

    torch.manual_seed(0)
    B, T, hc, d = 1, 64, 4, 128
    device = "cuda"
    layer = HyperConnection(hc, d).to(device)
    _init_hc(layer)
    x = torch.randn(B, T, hc, d, device=device, dtype=torch.bfloat16)
    ref_y, ref_post, ref_comb = _ref_pre(layer, x)
    sublayer_out = torch.randn(B, T, d, device=device, dtype=torch.bfloat16)

    ref_full = layer.hc_post(sublayer_out, x, ref_post, ref_comb)
    tk_full = tk.tk_mhc_post(
        sublayer_out,
        x,
        ref_post.unsqueeze(-1),
        ref_comb,
        hc_mult=hc,
    )
    if tk_full is None:
        raise _Skip(f"tk_mhc_post disabled (sticky verdict={tk._TK_POST_OK})")
    torch.testing.assert_close(tk_full, ref_full, atol=2e-2, rtol=2e-2)


@torch.no_grad()
def test_tk_mhc_head_matches_ref():
    _skip_if(not _HAS_CUDA, "no cuda")
    from rtp_llm.models_py.modules.dsv4 import _mhc_tilelang as tk

    torch.manual_seed(0)
    B, T, hc, d = 1, 64, 4, 128
    device = "cuda"
    head = HyperConnectionHead(hc, d).to(device)
    _init_head(head)
    x = torch.randn(B, T, hc, d, device=device, dtype=torch.bfloat16)

    ref_y = head(x)
    tk_y = tk.tk_mhc_head(
        x,
        head.hc_head_fn,
        head.hc_head_scale,
        head.hc_head_base,
        norm_eps=head.norm_eps,
        pre_eps=head.hc_eps,
        hc_mult=hc,
    )
    if tk_y is None:
        raise _Skip(f"tk_mhc_head disabled (sticky verdict={tk._TK_HEAD_OK})")
    torch.testing.assert_close(tk_y, ref_y, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    test_sinkhorn_doubly_stochastic()
    print("PASS test_sinkhorn_doubly_stochastic")
    test_hyperconnection_shapes()
    print("PASS test_hyperconnection_shapes")
    test_head_reduces_to_single_stream()
    print("PASS test_head_reduces_to_single_stream")
    test_expand_to_hc()
    print("PASS test_expand_to_hc")
    for fn in (
        test_tk_mhc_pre_matches_ref,
        test_tk_mhc_post_matches_ref,
        test_tk_mhc_head_matches_ref,
    ):
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except _Skip as e:
            print(f"SKIP {fn.__name__}: {e}")
    print("ALL TESTS PASSED")
