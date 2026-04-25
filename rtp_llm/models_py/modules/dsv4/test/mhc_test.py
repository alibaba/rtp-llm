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
    pre, post, comb = hc_split_sinkhorn(mixes, scale, base, hc, sinkhorn_iters=20, eps=1e-6)
    assert pre.shape == (2, 8, hc)
    assert post.shape == (2, 8, hc)
    assert comb.shape == (2, 8, hc, hc)
    # Row sums and column sums should be ~1
    row_sums = comb.sum(dim=-1)
    col_sums = comb.sum(dim=-2)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=5e-3), \
        f"row sums off: max diff {(row_sums - 1).abs().max().item()}"
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=5e-3), \
        f"col sums off: max diff {(col_sums - 1).abs().max().item()}"


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


if __name__ == "__main__":
    test_sinkhorn_doubly_stochastic()
    print("PASS test_sinkhorn_doubly_stochastic")
    test_hyperconnection_shapes()
    print("PASS test_hyperconnection_shapes")
    test_head_reduces_to_single_stream()
    print("PASS test_head_reduces_to_single_stream")
    test_expand_to_hc()
    print("PASS test_expand_to_hc")
    print("ALL TESTS PASSED")
