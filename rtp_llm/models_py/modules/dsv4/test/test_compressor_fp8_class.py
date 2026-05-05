"""End-to-end class-level UT for ``CompressorFP8``.

Covers the prefill path: build a layer with random weights, set up the
state pool + KV pool, run ``forward``, then dequant the FP8 KV pool via
the production reader and compare against a pure-PyTorch reference that
does pool→RMSNorm→RoPE→bf16 directly.

This is the "the new class actually works" gate — the lower-level kernel
correctness is already locked by ``test_compressor_fp8_writer.py`` /
``test_compressor_fp8_reader.py``.
"""

from __future__ import annotations

import math

import torch

try:
    import pytest

    HAVE_PYTEST = True
except ImportError:
    HAVE_PYTEST = False

    class _NoOpMark:
        def parametrize(self, *args, **kwargs):
            def deco(fn):
                return fn

            return deco

    class _NoOpPytest:
        mark = _NoOpMark()

        @staticmethod
        def skip(msg):
            raise SystemExit(f"SKIP: {msg}")

    pytest = _NoOpPytest()

from rtp_llm.models_py.modules.dsv4._compressor_fused_triton import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
)
from rtp_llm.models_py.modules.dsv4._compressor_kv_fused_triton import (
    KV_ENTRY_BYTES,
    KV_HEAD_DIM,
)
from rtp_llm.models_py.modules.dsv4._kv_fp8_pool_io import dequantize_and_gather_k_cache
from rtp_llm.models_py.modules.dsv4.compressor_fp8 import CompressorFP8


def _build_freqs_cis(max_pos: int, rope_dim: int, device, base: float = 10000.0):
    """Standard half-pair complex freqs_cis the layer's RoPE expects."""
    half = rope_dim // 2
    freqs = 1.0 / (
        base ** (torch.arange(0, half, device=device, dtype=torch.float32) / half)
    )
    pos = torch.arange(max_pos, device=device, dtype=torch.float32)
    angles = torch.outer(pos, freqs)
    return torch.polar(torch.ones_like(angles), angles).to(torch.complex64)


def _ref_pool_rmsnorm_rope_bf16(
    kv_state, score_state, norm_weight, freqs_cis_per_b, *, norm_eps, nope_dim, rope_dim
):
    """Pure-PyTorch reference matching the kernel's compute order."""
    score_sm = torch.softmax(score_state, dim=1)
    pooled = (kv_state * score_sm).sum(dim=1)
    w_fp32 = norm_weight.to(torch.float32)
    var = (pooled * pooled).mean(dim=-1, keepdim=True)
    rrms = torch.rsqrt(var + norm_eps)
    normed = pooled * rrms * w_fp32

    out = normed.clone()
    rope = normed[:, nope_dim : nope_dim + rope_dim]
    pair = rope.view(rope.shape[0], rope_dim // 2, 2)
    even, odd = pair[..., 0], pair[..., 1]
    cos = freqs_cis_per_b.real.to(torch.float32)
    sin = freqs_cis_per_b.imag.to(torch.float32)
    new_even = even * cos - odd * sin
    new_odd = odd * cos + even * sin
    rotated = torch.stack([new_even, new_odd], dim=-1).view(rope.shape[0], rope_dim)
    out[:, nope_dim : nope_dim + rope_dim] = rotated
    return out.to(torch.bfloat16)


@pytest.mark.parametrize(
    "head_dim,compress_ratio",
    [
        (KV_HEAD_DIM, 128),  # HCA  584B
        (KV_HEAD_DIM, 4),  # CSA  584B
        (INDEXER_HEAD_DIM, 4),  # indexer 132B (CSA-style overlap)
        (INDEXER_HEAD_DIM, 128),  # indexer 132B (HCA-style)
    ],
)
def test_compressor_fp8_prefill_round_trips(head_dim, compress_ratio):
    """CompressorFP8 prefill writes a pool that the matching reader
    recovers within fp8 quant noise vs python reference. Single class
    handles both head_dim=512 (584B) and head_dim=128 (132B) layouts."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(0xCAFE)
    device = torch.device("cuda")

    dim = 1024
    rope_dim = 64
    nope_dim = head_dim - rope_dim
    norm_eps = 1e-6
    overlap = compress_ratio == 4
    is_584 = head_dim == KV_HEAD_DIM
    entry_bytes = KV_ENTRY_BYTES if is_584 else INDEXER_ENTRY_BYTES

    # Single request, S = 2 * compress_ratio so we get 2 compressed tokens.
    bsz = 1
    S = 2 * compress_ratio
    block_size = 64
    # Need at least ceil(S/ratio / bs) blocks for the compressor pool.
    # Here we only emit S//ratio = 2 compressed tokens, so 1 block suffices.
    num_blocks = 4

    layer = CompressorFP8(
        dim=dim,
        head_dim=head_dim,
        rope_head_dim=rope_dim,
        compress_ratio=compress_ratio,
        max_batch_size=bsz,
        norm_eps=norm_eps,
        weights=None,
    ).to(device)

    # Random weight init (factory_mode=False starts from torch defaults; bump
    # the norm weight closer to 1 so RMSNorm doesn't blow up).
    with torch.no_grad():
        layer.norm.weight.copy_(
            (torch.randn(head_dim, device=device) * 0.05 + 1.0).to(torch.bfloat16)
        )
        layer.ape.copy_(
            torch.randn(compress_ratio, layer.ape.shape[1], device=device) * 0.1
        )

    # Build freqs_cis up to S+compress_ratio (the layer indexes
    # ``sp + p*ratio + (1 - ratio)``, which can be ≤ S).
    layer.freqs_cis = _build_freqs_cis(S + compress_ratio, rope_dim, device)

    # ── Bind pools ──
    # State pool: needs slot capacity ≥ state_rows. PoolBackedModule reads
    # state via ``_compute_pool_slots`` with eb=state_eb.
    state_pool_view = torch.zeros(
        num_blocks * block_size,
        layer._state_dim * 2,
        dtype=torch.float32,
        device=device,
    )
    state_block_table = torch.arange(
        1, num_blocks + 1, dtype=torch.int32, device=device
    ).view(bsz, num_blocks)

    # KV pool: ``entry_bytes`` per slot (584 for CSA/HCA, 132 for indexer),
    # contiguous flat. PoolBackedModule's ``_kv_pool_view`` is the flat
    # pool; ``_kv_eb`` is the per-block entry count (block_size).
    kv_pool_view = torch.zeros(
        num_blocks * block_size, entry_bytes, dtype=torch.uint8, device=device
    )
    kv_block_table = torch.arange(
        1, num_blocks + 1, dtype=torch.int32, device=device
    ).view(bsz, num_blocks)

    layer.set_pool_context(
        kv_pool_view=kv_pool_view,
        kv_block_table=kv_block_table,
        kv_eb=block_size,
        state_pool_view=state_pool_view,
        state_block_table=state_block_table,
        state_eb=block_size,
    )

    # ── Forward (fresh prefill, sp=0) ──
    x = torch.randn(bsz, S, dim, dtype=torch.bfloat16, device=device) * 0.1
    layer(x, start_pos=0)

    # ── Build expected K via pure PyTorch on the same intermediates ──
    # Re-derive the kv/score the layer fed to the kernel:
    x32 = x.float()
    kv = torch.nn.functional.linear(x32, layer.wkv.weight)
    score = torch.nn.functional.linear(x32, layer.wgate.weight)
    # ape add (post-unflatten in the layer; we reproduce the [B, NB, ratio, D]
    # shape after the same processing).
    cutoff = S  # remainder == 0 (S = 2 * ratio)
    kv = kv.unflatten(1, (-1, compress_ratio))
    score = score.unflatten(1, (-1, compress_ratio)) + layer.ape

    if overlap:
        # _overlap_transform; first ratio rows get value=0/-inf because sp=0
        # (no prior).
        b, nb, _, _ = kv.shape
        d = head_dim
        kv_o = kv.new_full((b, nb, 2 * compress_ratio, d), 0.0)
        score_o = score.new_full((b, nb, 2 * compress_ratio, d), float("-inf"))
        kv_o[:, :, compress_ratio:] = kv[:, :, :, d:]
        score_o[:, :, compress_ratio:] = score[:, :, :, d:]
        kv_o[:, 1:, :compress_ratio] = kv[:, :-1, :, :d]
        score_o[:, 1:, :compress_ratio] = score[:, :-1, :, :d]
        kv = kv_o
        score = score_o

    # Per-token freq idx the layer used.
    NB = kv.shape[1]
    pos_local = torch.arange(NB, device=device, dtype=torch.long)
    freq_idx = torch.clamp(0 + pos_local * compress_ratio + (1 - compress_ratio), min=0)
    K_ref = []
    for n in range(NB):
        kv_n = kv[:, n].squeeze(0)  # [G, d]
        sc_n = score[:, n].squeeze(0)
        freqs_n = layer.freqs_cis[freq_idx[n].item()].unsqueeze(0)
        ref_b = _ref_pool_rmsnorm_rope_bf16(
            kv_n.unsqueeze(0),
            sc_n.unsqueeze(0),
            layer.norm.weight,
            freqs_n,
            norm_eps=norm_eps,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
        )
        K_ref.append(ref_b)
    K_ref = torch.cat(K_ref, dim=0)  # [NB, head_dim]

    # ── Dequant the pool that the layer wrote ──
    # The compressor wrote the first NB compressed tokens at logical
    # positions [0, NB). Pool slot p is blk=block_table[0, p // bs],
    # off=p % bs.
    pool_3d = kv_pool_view.view(num_blocks, block_size, entry_bytes)
    if is_584:
        out = torch.zeros(bsz, NB, KV_HEAD_DIM, dtype=torch.bfloat16, device=device)
        seq_lens = torch.tensor([NB], dtype=torch.int32, device=device)
        dequantize_and_gather_k_cache(
            out,
            pool_3d,
            seq_lens,
            None,
            kv_block_table,
            block_size,
            offset=0,
        )
        K_got = out[0]  # [NB, head_dim]
    else:
        # 132B layout: per block, [bs * 128 K | bs * 4 fp32 scales].
        # Inline pure-python dequant — same convention as DeepGEMM consumes.
        # Decode for the same NB compressed tokens as above.
        first_blk = int(kv_block_table[0, 0].item())
        flat = pool_3d.reshape(num_blocks, block_size * entry_bytes)
        K_got = torch.empty(NB, head_dim, dtype=torch.bfloat16, device=device)
        for n in range(NB):
            slot = n  # logical compressed position
            blk = first_blk + slot // block_size
            off = slot % block_size
            k_bytes = flat[blk, off * head_dim : (off + 1) * head_dim]
            scale_off = block_size * head_dim + off * 4
            scale_bytes = flat[blk, scale_off : scale_off + 4]
            scale = scale_bytes.view(torch.float32)
            fp8 = k_bytes.view(torch.float8_e4m3fn).to(torch.float32)
            K_got[n] = (fp8 * scale).to(torch.bfloat16)

    # ── Compare ──
    diff = (K_got.to(torch.float32) - K_ref.to(torch.float32)).abs()
    max_abs = diff.max().item()
    # NoPE goes through fp8 quant; bound by scale/127 ≈ absmax/(448*127)
    # per element. Post-RMSNorm absmax can hit ~1-3, so per-element error
    # ~5e-3 to 5e-2; UE8M0 power-of-2 rounding can push that to ~0.5
    # worst-case. RoPE bytes are bf16-direct except for fp32-reduction
    # 1-ULP noise. Use the same 0.5 tolerance as test_compressor_fp8_reader.
    assert max_abs < 0.5, (
        f"compressor_fp8 → dequant vs reference: max_abs={max_abs:.4f} "
        f"too large (overlap={overlap})"
    )


if __name__ == "__main__":
    test_compressor_fp8_prefill_round_trips(KV_HEAD_DIM, 128)
    test_compressor_fp8_prefill_round_trips(KV_HEAD_DIM, 4)
    test_compressor_fp8_prefill_round_trips(INDEXER_HEAD_DIM, 4)
    test_compressor_fp8_prefill_round_trips(INDEXER_HEAD_DIM, 128)
    print("OK")
