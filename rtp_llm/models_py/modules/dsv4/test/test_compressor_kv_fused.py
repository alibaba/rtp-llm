"""UT for ``_compressor_kv_fused_triton.v4_compressor_kv_fused``.

Validates the fused {pool → RMSNorm → RoPE → FP8 quant → cache scatter}
kernel for the **CSA / HCA KV pool** (head_dim=512, 584 B/slot canonical
fp8_model1_mla layout, see ``DSV4CacheConfig.h:78-91``):

  bytes [  0: 448]  448 x fp8_e4m3 NoPE
  bytes [448: 576]  64  x bf16     RoPE
  bytes [576: 583]  7   x UE8M0    NoPE scales (one per QUANT_BLOCK=64)
  bytes [583: 584]  pad

Two angles:

  1. **Byte-layout** vs a pure-torch reference packer — locks each region
     of the slot independently (NoPE fp8 bytes, RoPE bf16, UE8M0 scales).
  2. **Round-trip dequant** vs the bf16 pool/rmsnorm/rope chain — same
     tolerance pattern as ``test_v4_compressor_fused`` (FP8 e4m3 noise).

Run:
  cd .../github-opensource && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \\
    /opt/conda310/bin/python3 \\
    rtp_llm/models_py/modules/dsv4/test/test_compressor_kv_fused.py
"""

from __future__ import annotations

import math

import torch

from rtp_llm.models_py.modules.dsv4._compressor_fused_triton import freqs_cis_to_cos_sin
from rtp_llm.models_py.modules.dsv4._compressor_kv_fused_triton import (
    KV_ENTRY_BYTES,
    KV_HEAD_DIM,
    KV_N_NOPE_BLOCKS,
    KV_NOPE_DIM,
    KV_NOPE_OFFSET,
    KV_QUANT_BLOCK,
    KV_ROPE_DIM,
    KV_ROPE_OFFSET,
    KV_SCALE_OFFSET,
    v4_compressor_kv_fused,
)
from rtp_llm.models_py.modules.dsv4._compressor_triton import v4_compressor_pool
from rtp_llm.models_py.modules.dsv4.rope import apply_rotary_emb_batched

FP8_MAX = 448.0


# ---------------------------------------------------------------------------
# Reference packer — matches the kernel's per-slot layout byte-for-byte.
# ---------------------------------------------------------------------------
def _ref_pool_rmsnorm_rope_fp32(
    kv_state,
    score_state,
    norm_weight_bf16,
    freqs_cis_per_b,
    *,
    overlap,
    head_dim,
    rope_head_dim,
    eps,
):
    """Returns fp32 ``[B, head_dim]`` post-pool/RMSNorm/RoPE — kernel runs
    fp32 throughout, so we compare in fp32 (no bf16 round-trip on the
    rotated output yet)."""
    kv4 = kv_state.unsqueeze(1)
    sc4 = score_state.unsqueeze(1)
    pooled = v4_compressor_pool(
        kv4, sc4, overlap=overlap, out_d=head_dim if overlap else None
    ).squeeze(
        1
    )  # [B, head_dim] fp32
    w_f32 = norm_weight_bf16.float()
    rms = (pooled.pow(2).mean(-1, keepdim=True) + eps).rsqrt()
    normed = pooled * rms * w_f32  # [B, head_dim] fp32
    # GPT-J RoPE on the trailing rope_head_dim, in-place on a [B,1,...] view.
    rope_view = normed[..., -rope_head_dim:].unsqueeze(1).contiguous()
    apply_rotary_emb_batched(rope_view, freqs_cis_per_b)
    normed[..., -rope_head_dim:] = rope_view.squeeze(1)
    return normed


def _ref_pack_slot(rotated_fp32: torch.Tensor) -> torch.Tensor:
    """Pack a single ``[head_dim]`` fp32 row into the canonical 584 B slot.

    Quant matches the kernel:
      bf16 round-trip on input
      per-64 block: scale = 2^ceil(log2(absmax/448)); inv_scale = 2^(-exp)
      fp8 = clamp(x*inv_scale, ±448).to(float8_e4m3fn)
      UE8M0 = clamp(exp + 127, 0, 255).to(uint8)
    """
    out = torch.zeros(KV_ENTRY_BYTES, dtype=torch.uint8, device=rotated_fp32.device)
    # bf16 round-trip on the entire normed vector (kernel does this once).
    bf = rotated_fp32.to(torch.bfloat16).to(torch.float32)

    # NoPE quant
    nope = bf[:KV_NOPE_DIM].view(KV_N_NOPE_BLOCKS, KV_QUANT_BLOCK)
    absmax = nope.abs().amax(dim=1).clamp_min(1e-4)
    raw = absmax / FP8_MAX
    exponents = torch.ceil(torch.log2(raw))
    inv_scale = torch.pow(2.0, -exponents).unsqueeze(-1)
    q = (nope * inv_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    out[KV_NOPE_OFFSET : KV_NOPE_OFFSET + KV_NOPE_DIM] = q.reshape(-1).view(torch.uint8)

    # UE8M0 scales
    ue8m0 = (exponents + 127.0).clamp(0.0, 255.0).to(torch.uint8)
    out[KV_SCALE_OFFSET : KV_SCALE_OFFSET + KV_N_NOPE_BLOCKS] = ue8m0

    # RoPE bf16 store (post-RoPE values from rotated_fp32)
    rope_bf = bf[KV_NOPE_DIM:].to(torch.bfloat16)
    out[KV_ROPE_OFFSET : KV_ROPE_OFFSET + KV_ROPE_DIM * 2] = rope_bf.view(torch.uint8)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_state(B, G, D, *, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    kv = torch.randn(B, G, D, dtype=torch.float32, device="cuda", generator=g)
    sc = torch.randn(B, G, D, dtype=torch.float32, device="cuda", generator=g)
    return kv.contiguous(), sc.contiguous()


def _make_freqs(B, rope_head_dim, *, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    angles = torch.rand(B, rope_head_dim // 2, generator=g, device="cuda") * (
        2 * math.pi
    )
    return torch.polar(torch.ones_like(angles), angles).to(torch.complex64)


def _make_cache(num_blocks: int, block_size: int) -> torch.Tensor:
    """Allocate a 584 B/slot cache. Natural stride (block_size * 584); the
    TMA-padded variant is exercised separately via ``_make_cache_tma_padded``."""
    return torch.zeros(
        num_blocks, block_size, KV_ENTRY_BYTES, dtype=torch.uint8, device="cuda"
    )


def _make_cache_tma_padded(num_blocks: int, block_size: int) -> torch.Tensor:
    """Allocate with TMA-padded per-block stride (576-aligned, see
    ``DSV4PoolSpec::padded_block_size_bytes``). Returns a view whose
    ``.stride(0) > block_size * 584`` so the kernel must respect the
    explicit stride argument."""
    align = 576
    natural = block_size * KV_ENTRY_BYTES
    padded = ((natural + align - 1) // align) * align
    flat = torch.zeros(num_blocks, padded, dtype=torch.uint8, device="cuda")
    # Slice to [num_blocks, block_size, 584] — slice on dim-1 then reshape
    # would lose the padded stride; instead use as_strided so dim-0 stride
    # stays at ``padded`` bytes.
    return torch.as_strided(
        flat,
        size=(num_blocks, block_size, KV_ENTRY_BYTES),
        stride=(padded, KV_ENTRY_BYTES, 1),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def _run_and_decode(B, G, D_in, *, overlap, seed_off):
    head_dim = KV_HEAD_DIM
    rope_head_dim = KV_ROPE_DIM
    eps = 1e-6

    kv, sc = _make_state(B, G, D_in, seed=42 + seed_off)
    weight = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda") * 0.1 + 1.0
    freqs = _make_freqs(B, rope_head_dim, seed=99 + seed_off)
    cos, sin = freqs_cis_to_cos_sin(freqs)

    block_size = 8
    num_blocks = (B + block_size - 1) // block_size + 2
    cache = _make_cache(num_blocks, block_size)
    # First slot left as a sentinel hole; assign B unique slots starting at 1.
    slot_mapping = (torch.arange(B, dtype=torch.int64, device="cuda") + 1).contiguous()

    v4_compressor_kv_fused(
        kv,
        sc,
        slot_mapping,
        weight,
        cos,
        sin,
        cache,
        cache_block_stride_bytes=int(cache.stride(0)),
        overlap=overlap,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        norm_eps=eps,
    )

    rotated = _ref_pool_rmsnorm_rope_fp32(
        kv,
        sc,
        weight,
        freqs,
        overlap=overlap,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        eps=eps,
    )
    return cache, slot_mapping, rotated, weight, freqs


def _assert_layout(cache, slot_mapping, rotated, *, tag):
    """Per-slot byte-for-byte comparison of NoPE fp8 / RoPE bf16 / UE8M0."""
    block_size = cache.shape[1]
    for i in range(slot_mapping.numel()):
        slot = int(slot_mapping[i].item())
        block = slot // block_size
        off = slot % block_size
        actual = cache[block, off].clone()  # [584] uint8
        ref = _ref_pack_slot(rotated[i])
        # NoPE: fp8 compares as identical bytes when quant matches.
        assert torch.equal(
            actual[:KV_NOPE_DIM], ref[:KV_NOPE_DIM]
        ), f"[{tag}] slot {slot} NoPE bytes differ"
        # RoPE: bf16 bytes; allow exact equality (kernel uses .to(bf16) on the
        # same fp32 input as the ref).
        assert torch.equal(
            actual[KV_ROPE_OFFSET : KV_ROPE_OFFSET + 2 * KV_ROPE_DIM],
            ref[KV_ROPE_OFFSET : KV_ROPE_OFFSET + 2 * KV_ROPE_DIM],
        ), f"[{tag}] slot {slot} RoPE bf16 bytes differ"
        # UE8M0 scales: 7 bytes.
        assert torch.equal(
            actual[KV_SCALE_OFFSET : KV_SCALE_OFFSET + KV_N_NOPE_BLOCKS],
            ref[KV_SCALE_OFFSET : KV_SCALE_OFFSET + KV_N_NOPE_BLOCKS],
        ), f"[{tag}] slot {slot} UE8M0 scales differ"
        # Pad byte stays zero.
        assert (
            actual[KV_ENTRY_BYTES - 1].item() == 0
        ), f"[{tag}] slot {slot} pad byte not zero"


def test_csa_layout():
    """CSA: overlap=True, ratio=4 ⇒ G=8, D_in=2*512=1024."""
    cache, slots, rotated, _, _ = _run_and_decode(
        B=12, G=8, D_in=2 * KV_HEAD_DIM, overlap=True, seed_off=0
    )
    _assert_layout(cache, slots, rotated, tag="csa")
    print("[csa] byte-layout OK")


def test_hca_layout():
    """HCA: overlap=False, ratio=128 ⇒ G=128, D_in=512."""
    cache, slots, rotated, _, _ = _run_and_decode(
        B=4, G=128, D_in=KV_HEAD_DIM, overlap=False, seed_off=1
    )
    _assert_layout(cache, slots, rotated, tag="hca")
    print("[hca] byte-layout OK")


def test_skip_negative_slot():
    """Slot < 0 ⇒ kernel must early-exit, leaving the would-be slot zero."""
    head_dim = KV_HEAD_DIM
    B, G, D_in = 6, 8, 2 * head_dim
    kv, sc = _make_state(B, G, D_in, seed=7)
    weight = torch.ones(head_dim, dtype=torch.bfloat16, device="cuda")
    freqs = _make_freqs(B, KV_ROPE_DIM, seed=7)
    cos, sin = freqs_cis_to_cos_sin(freqs)

    block_size = 4
    cache = _make_cache(8, block_size)
    slot_mapping = torch.tensor([1, -1, 3, -1, 5, 7], dtype=torch.int64, device="cuda")
    v4_compressor_kv_fused(
        kv,
        sc,
        slot_mapping,
        weight,
        cos,
        sin,
        cache,
        cache_block_stride_bytes=int(cache.stride(0)),
        overlap=True,
    )
    # Skipped slots must remain all-zero.
    for skipped in (0, 2, 4, 6):
        block, off = skipped // block_size, skipped % block_size
        assert cache[block, off].sum().item() == 0, f"slot {skipped} not zero"
    print("[skip] negative-slot early-exit OK")


def test_tma_padded_block_stride():
    """Verify the kernel honors a TMA-padded per-block stride that exceeds
    ``block_size * 584``."""
    head_dim = KV_HEAD_DIM
    B, G, D_in = 9, 8, 2 * head_dim
    kv, sc = _make_state(B, G, D_in, seed=33)
    weight = torch.randn(head_dim, dtype=torch.bfloat16, device="cuda") * 0.1 + 1.0
    freqs = _make_freqs(B, KV_ROPE_DIM, seed=33)
    cos, sin = freqs_cis_to_cos_sin(freqs)

    block_size = 8
    cache = _make_cache_tma_padded(num_blocks=4, block_size=block_size)
    natural = block_size * KV_ENTRY_BYTES
    assert (
        int(cache.stride(0)) > natural
    ), f"test setup: stride {cache.stride(0)} expected > {natural}"

    slot_mapping = (torch.arange(B, dtype=torch.int64, device="cuda") + 1).contiguous()
    v4_compressor_kv_fused(
        kv,
        sc,
        slot_mapping,
        weight,
        cos,
        sin,
        cache,
        cache_block_stride_bytes=int(cache.stride(0)),
        overlap=True,
    )
    rotated = _ref_pool_rmsnorm_rope_fp32(
        kv,
        sc,
        weight,
        freqs,
        overlap=True,
        head_dim=head_dim,
        rope_head_dim=KV_ROPE_DIM,
        eps=1e-6,
    )
    _assert_layout(cache, slot_mapping, rotated, tag="tma_padded")
    print("[tma_padded] byte-layout OK with padded block stride")


def main():
    assert torch.cuda.is_available(), "test requires CUDA"
    test_csa_layout()
    test_hca_layout()
    test_skip_negative_slot()
    test_tma_padded_block_stride()
    print("\nALL OK")


if __name__ == "__main__":
    main()
