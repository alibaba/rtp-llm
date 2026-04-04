"""Unit tests for Dao-AILab Flash Attention 3 prefill correctness on H20."""

import pytest
import torch

_HAS_FA3 = False
try:
    from flash_attn_interface import flash_attn_varlen_func

    _HAS_FA3 = True
except ImportError:
    pass


def _sdpa_reference(q, k, v, cu_seqlens, num_heads, kv_heads):
    """Unfused SDPA reference per batch."""
    batch_size = len(cu_seqlens) - 1
    outputs = []
    for i in range(batch_size):
        s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        qi = q[s:e].transpose(0, 1).unsqueeze(0)
        ki = k[s:e].transpose(0, 1).unsqueeze(0)
        vi = v[s:e].transpose(0, 1).unsqueeze(0)
        if kv_heads != num_heads:
            ki = ki.repeat_interleave(num_heads // kv_heads, dim=1)
            vi = vi.repeat_interleave(num_heads // kv_heads, dim=1)
        oi = torch.nn.functional.scaled_dot_product_attention(
            qi, ki, vi, is_causal=True
        )
        outputs.append(oi.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0)


def _run_fa3(q, k, v, cu_seqlens, max_seqlen):
    out = flash_attn_varlen_func(
        q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True
    )
    return out[0] if isinstance(out, tuple) else out


@pytest.mark.gpu(type="H20")
@pytest.mark.skipif(not _HAS_FA3, reason="flash_attn_interface not installed")
class TestFlashAttention3Prefill:

    @pytest.mark.parametrize(
        "dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
    )
    def test_fa3_causal_dtype(self, dtype):
        bs, sl, nh, kvh, hd = 2, 1024, 32, 8, 128
        total = bs * sl
        q = torch.randn(total, nh, hd, dtype=dtype, device="cuda")
        k = torch.randn(total, kvh, hd, dtype=dtype, device="cuda")
        v = torch.randn(total, kvh, hd, dtype=dtype, device="cuda")
        cu = torch.arange(0, (bs + 1) * sl, sl, dtype=torch.int32, device="cuda")

        out = _run_fa3(q, k, v, cu, sl)
        ref = _sdpa_reference(q, k, v, cu, nh, kvh)
        max_diff = (out - ref).abs().max().item()
        assert max_diff < 1e-2, f"FA3 max_diff={max_diff:.5f} > 1e-2"

    @pytest.mark.parametrize(
        "nh,kvh", [(32, 32), (32, 8), (64, 8)], ids=["MHA", "GQA-4", "GQA-8"]
    )
    def test_fa3_gqa(self, nh, kvh):
        bs, sl, hd = 2, 512, 128
        total = bs * sl
        q = torch.randn(total, nh, hd, dtype=torch.float16, device="cuda")
        k = torch.randn(total, kvh, hd, dtype=torch.float16, device="cuda")
        v = torch.randn(total, kvh, hd, dtype=torch.float16, device="cuda")
        cu = torch.arange(0, (bs + 1) * sl, sl, dtype=torch.int32, device="cuda")

        out = _run_fa3(q, k, v, cu, sl)
        ref = _sdpa_reference(q, k, v, cu, nh, kvh)
        max_diff = (out - ref).abs().max().item()
        assert max_diff < 1e-2, f"FA3 GQA max_diff={max_diff:.5f} > 1e-2"

    @pytest.mark.parametrize(
        "sl", [128, 512, 1024, 4096], ids=["s128", "s512", "s1024", "s4096"]
    )
    def test_fa3_various_seqlen(self, sl):
        bs, nh, kvh, hd = 2, 32, 8, 128
        total = bs * sl
        q = torch.randn(total, nh, hd, dtype=torch.float16, device="cuda")
        k = torch.randn(total, kvh, hd, dtype=torch.float16, device="cuda")
        v = torch.randn(total, kvh, hd, dtype=torch.float16, device="cuda")
        cu = torch.arange(0, (bs + 1) * sl, sl, dtype=torch.int32, device="cuda")

        out = _run_fa3(q, k, v, cu, sl)
        ref = _sdpa_reference(q, k, v, cu, nh, kvh)
        max_diff = (out - ref).abs().max().item()
        assert max_diff < 1e-2, f"FA3 seqlen={sl} max_diff={max_diff:.5f} > 1e-2"

    @pytest.mark.parametrize("hd", [64, 128, 192], ids=["d64", "d128", "d192"])
    def test_fa3_various_head_dim(self, hd):
        bs, sl, nh, kvh = 2, 512, 16, 4
        total = bs * sl
        q = torch.randn(total, nh, hd, dtype=torch.float16, device="cuda")
        k = torch.randn(total, kvh, hd, dtype=torch.float16, device="cuda")
        v = torch.randn(total, kvh, hd, dtype=torch.float16, device="cuda")
        cu = torch.arange(0, (bs + 1) * sl, sl, dtype=torch.int32, device="cuda")

        out = _run_fa3(q, k, v, cu, sl)
        ref = _sdpa_reference(q, k, v, cu, nh, kvh)
        max_diff = (out - ref).abs().max().item()
        assert max_diff < 1e-2, f"FA3 head_dim={hd} max_diff={max_diff:.5f} > 1e-2"
