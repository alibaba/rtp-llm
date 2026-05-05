"""UT for ``_compressor_fused_triton.v4_compressor_fused``.

Validates the fused {pool → RMSNorm → RoPE → FP8 quant → cache scatter}
kernel against the existing 4-launch chain that it replaces:

    ref(kv, score, slot_mapping, weight, freqs_cis, cache):
        pooled  = v4_compressor_pool(kv, score, overlap=...)
        normed  = rmsnorm(pooled.to(bf16), weight, eps)
        rotated = apply_rotary_emb_inplace(normed[..., -rd:], freqs_cis)
        quantize_indexer_k(normed.view(-1, 128), slot_mapping, cache)

Key cases (all use head_dim=128 = indexer compressor head dim):

  * CSA decode (overlap=True, ratio=4)        — G=8,  D_in=2*128
  * HCA decode (overlap=False, ratio=128)     — G=128, D_in=128
  * CSA prefill chunk (B=64, mixed -1 slots)
  * HCA prefill chunk (B=128)
  * Empty (no valid slots) — must be a no-op
  * Round-trip via ``dequantize_indexer_k`` — locks the cache layout

Bench: 4-launch vs fused.

Run:
  cd .../github-opensource && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \\
    /opt/conda310/bin/python3 \\
    rtp_llm/models_py/modules/dsv4/test/test_v4_compressor_fused.py
"""

from __future__ import annotations

import math

import torch

from rtp_llm.models_py.modules.dsv4._compressor_fused_triton import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
    freqs_cis_to_cos_sin,
    v4_compressor_fused,
)
from rtp_llm.models_py.modules.dsv4._compressor_triton import v4_compressor_pool
from rtp_llm.models_py.modules.dsv4._indexer_fp8_quant_triton import (
    dequantize_indexer_k,
    quantize_indexer_k,
)
from rtp_llm.models_py.modules.dsv4.rope import (
    apply_rotary_emb,
    apply_rotary_emb_batched,
)


# ---------------------------------------------------------------------------
# Reference: the 4-launch chain we're replacing.
# ---------------------------------------------------------------------------
def _ref_rmsnorm_bf16(
    x_bf: torch.Tensor, weight_bf: torch.Tensor, eps: float
) -> torch.Tensor:
    """Pure-torch RMSNorm in fp32 with bf16 input and bf16 weight; bf16 out."""
    x32 = x_bf.float()
    rms = (x32.pow(2).mean(-1, keepdim=True) + eps).rsqrt()
    return (x32 * rms * weight_bf.float()).to(torch.bfloat16)


def _ref_pool_rmsnorm_rope(
    kv_state: torch.Tensor,  # [B, G, D_in] fp32
    score_state: torch.Tensor,
    norm_weight: torch.Tensor,  # [head_dim] bf16
    freqs_cis_per_b: torch.Tensor,  # [B, rope_head_dim/2] complex64
    *,
    overlap: bool,
    head_dim: int,
    rope_head_dim: int,
    eps: float,
) -> torch.Tensor:
    """Returns bf16 [B, head_dim] post-pool, post-RMSNorm, post-RoPE."""
    kv4 = kv_state.unsqueeze(1)
    sc4 = score_state.unsqueeze(1)
    pooled = v4_compressor_pool(
        kv4,
        sc4,
        overlap=overlap,
        out_d=head_dim if overlap else None,
    )  # [B, 1, head_dim] fp32
    normed = _ref_rmsnorm_bf16(pooled.to(torch.bfloat16), norm_weight, eps)
    # apply_rotary_emb_batched is in-place on a [B, S=1, last] tensor.
    apply_rotary_emb_batched(normed[..., -rope_head_dim:], freqs_cis_per_b)
    return normed.squeeze(1)  # [B, head_dim] bf16


def _ref_full_chain(
    kv_state,
    score_state,
    slot_mapping,
    norm_weight,
    freqs_cis_per_b,
    cache,
    *,
    overlap,
    head_dim,
    rope_head_dim,
    eps,
) -> None:
    normed = _ref_pool_rmsnorm_rope(
        kv_state,
        score_state,
        norm_weight,
        freqs_cis_per_b,
        overlap=overlap,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        eps=eps,
    )
    quantize_indexer_k(normed.contiguous(), slot_mapping, cache)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_state(B: int, G: int, D: int, *, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    kv = torch.randn(B, G, D, dtype=torch.float32, device="cuda", generator=g)
    sc = torch.randn(B, G, D, dtype=torch.float32, device="cuda", generator=g)
    return kv.contiguous(), sc.contiguous()


def _make_freqs(B: int, rope_head_dim: int, *, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    angles = torch.rand(B, rope_head_dim // 2, generator=g, device="cuda") * (
        2 * math.pi
    )
    return torch.polar(torch.ones_like(angles), angles).to(torch.complex64)


def _make_cache(num_blocks: int, block_size: int) -> torch.Tensor:
    return torch.zeros(
        num_blocks,
        block_size,
        INDEXER_ENTRY_BYTES,
        dtype=torch.uint8,
        device="cuda",
    )


def _bench(fn, *a, warmup: int = 25, iters: int = 200) -> float:
    for _ in range(warmup):
        fn(*a)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn(*a)
    e.record()
    e.synchronize()
    return s.elapsed_time(e) / iters  # ms


def _assert_cache_close(
    cand: torch.Tensor,
    ref: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    tag: str,
    rel_tol: float = 0.05,
) -> None:
    """Compare two FP8 caches by dequantizing valid slots.  FP8 e4m3 has
    ~6% per-element noise; both fused and ref go through the same fp8
    quant so the *difference* between them should be much smaller than
    the magnitude — bound mean_abs at 5% of ref max_abs."""
    valid_slots = slot_mapping[slot_mapping >= 0]
    if valid_slots.numel() == 0:
        assert torch.equal(cand, ref), f"{tag}: empty path wrote to cache"
        print(f"  [{tag}] empty no-op OK")
        return
    cand_d = dequantize_indexer_k(cand, valid_slots, out_dtype=torch.float32)
    ref_d = dequantize_indexer_k(ref, valid_slots, out_dtype=torch.float32)
    diff = (cand_d - ref_d).abs()
    ref_max = ref_d.abs().max().item()
    me = diff.mean().item()
    mx = diff.max().item()
    print(
        f"  [{tag}] valid={valid_slots.numel():3d}  "
        f"max_abs={mx:.4e}  mean_abs={me:.4e}  ref_max={ref_max:.4e}"
    )
    assert (
        me < rel_tol * ref_max
    ), f"{tag}: mean_abs {me:.4e} > {rel_tol*100:.0f}% of ref_max {ref_max:.4e}"


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
HEAD_DIM = INDEXER_HEAD_DIM  # 128
ROPE_HEAD_DIM = 64
EPS = 1e-6


def test_csa_decode():
    """CSA decode shape: B=4, overlap=True, ratio=4, G=8, D_in=2*128."""
    ratio = 4
    B = 4
    G = 2 * ratio
    D_in = 2 * HEAD_DIM
    kv, sc = _make_state(B, G, D_in, seed=10)
    weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.5 + 1.0
    freqs = _make_freqs(B, ROPE_HEAD_DIM, seed=11)
    block_size = 64
    num_blocks = max(1, (B + block_size - 1) // block_size)
    slots = torch.arange(B, dtype=torch.int64, device="cuda")
    cos, sin = freqs_cis_to_cos_sin(freqs)

    cache_fused = _make_cache(num_blocks, block_size)
    cache_ref = _make_cache(num_blocks, block_size)
    v4_compressor_fused(
        kv,
        sc,
        slots,
        weight,
        cos,
        sin,
        cache_fused,
        overlap=True,
        head_dim=HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        norm_eps=EPS,
    )
    _ref_full_chain(
        kv,
        sc,
        slots,
        weight,
        freqs,
        cache_ref,
        overlap=True,
        head_dim=HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        eps=EPS,
    )
    _assert_cache_close(cache_fused, cache_ref, slots, tag="CSA decode")


def test_hca_decode():
    """HCA decode shape: B=2, overlap=False, ratio=128, G=128, D_in=128."""
    ratio = 128
    B = 2
    G = ratio
    D_in = HEAD_DIM
    kv, sc = _make_state(B, G, D_in, seed=20)
    weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.3 + 1.0
    freqs = _make_freqs(B, ROPE_HEAD_DIM, seed=21)
    block_size = 32
    slots = torch.arange(B, dtype=torch.int64, device="cuda")
    cos, sin = freqs_cis_to_cos_sin(freqs)

    cache_fused = _make_cache(1, block_size)
    cache_ref = _make_cache(1, block_size)
    v4_compressor_fused(
        kv,
        sc,
        slots,
        weight,
        cos,
        sin,
        cache_fused,
        overlap=False,
        head_dim=HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        norm_eps=EPS,
    )
    _ref_full_chain(
        kv,
        sc,
        slots,
        weight,
        freqs,
        cache_ref,
        overlap=False,
        head_dim=HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        eps=EPS,
    )
    _assert_cache_close(cache_fused, cache_ref, slots, tag="HCA decode")


def test_csa_prefill_chunk():
    """CSA prefill chunk: B=64 tokens, ratio=4, ~3/4 are non-boundary (slot=-1)."""
    ratio = 4
    B = 64
    G = 2 * ratio
    D_in = 2 * HEAD_DIM
    kv, sc = _make_state(B, G, D_in, seed=30)
    weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.5 + 1.0
    freqs = _make_freqs(B, ROPE_HEAD_DIM, seed=31)
    block_size = 64
    # Mark only every-4th token as boundary, with realistic compressed-slot ids.
    slots = torch.full((B,), -1, dtype=torch.int64, device="cuda")
    valid_idx = torch.arange(0, B, ratio, device="cuda")
    slots[valid_idx] = (valid_idx // ratio).to(torch.int64)
    num_blocks = max(1, (slots.max().item() // block_size) + 1)
    cos, sin = freqs_cis_to_cos_sin(freqs)

    cache_fused = _make_cache(num_blocks, block_size)
    cache_ref = _make_cache(num_blocks, block_size)
    v4_compressor_fused(
        kv,
        sc,
        slots,
        weight,
        cos,
        sin,
        cache_fused,
        overlap=True,
        head_dim=HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        norm_eps=EPS,
    )
    _ref_full_chain(
        kv,
        sc,
        slots,
        weight,
        freqs,
        cache_ref,
        overlap=True,
        head_dim=HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        eps=EPS,
    )
    _assert_cache_close(cache_fused, cache_ref, slots, tag="CSA prefill chunk")
    # Locked-down: -1 slots leave their cache regions untouched.
    assert torch.equal(
        cache_fused.view(-1)[: int(slots.max().item() + 1) * INDEXER_HEAD_DIM] != 0,
        cache_ref.view(-1)[: int(slots.max().item() + 1) * INDEXER_HEAD_DIM] != 0,
    ), "CSA prefill: -1 slot mask diverges from ref"


def test_hca_prefill_chunk():
    ratio = 128
    B = 128
    G = ratio
    D_in = HEAD_DIM
    kv, sc = _make_state(B, G, D_in, seed=40)
    weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.3 + 1.0
    freqs = _make_freqs(B, ROPE_HEAD_DIM, seed=41)
    block_size = 32
    # In an HCA prefill chunk, only the last token of each ratio-window writes.
    slots = torch.full((B,), -1, dtype=torch.int64, device="cuda")
    boundary = torch.arange(ratio - 1, B, ratio, device="cuda")
    slots[boundary] = torch.arange(boundary.numel(), device="cuda", dtype=torch.int64)
    num_blocks = max(1, (boundary.numel() + block_size - 1) // block_size)
    cos, sin = freqs_cis_to_cos_sin(freqs)

    cache_fused = _make_cache(num_blocks, block_size)
    cache_ref = _make_cache(num_blocks, block_size)
    v4_compressor_fused(
        kv,
        sc,
        slots,
        weight,
        cos,
        sin,
        cache_fused,
        overlap=False,
        head_dim=HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        norm_eps=EPS,
    )
    _ref_full_chain(
        kv,
        sc,
        slots,
        weight,
        freqs,
        cache_ref,
        overlap=False,
        head_dim=HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        eps=EPS,
    )
    _assert_cache_close(cache_fused, cache_ref, slots, tag="HCA prefill chunk")


def test_empty_no_op():
    """All -1 slots — cache must stay zero/unchanged."""
    ratio = 4
    B = 16
    G = 2 * ratio
    D_in = 2 * HEAD_DIM
    kv, sc = _make_state(B, G, D_in, seed=50)
    weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    freqs = _make_freqs(B, ROPE_HEAD_DIM, seed=51)
    slots = torch.full((B,), -1, dtype=torch.int64, device="cuda")
    cos, sin = freqs_cis_to_cos_sin(freqs)
    cache = _make_cache(1, 32)
    cache_pre = cache.clone()
    v4_compressor_fused(
        kv,
        sc,
        slots,
        weight,
        cos,
        sin,
        cache,
        overlap=True,
        head_dim=HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        norm_eps=EPS,
    )
    assert torch.equal(cache, cache_pre), "empty path mutated cache"
    print("  [empty no-op] OK")


def test_round_trip_via_dequant():
    """Fused write → dequantize_indexer_k recovers ~ ref bf16 vector."""
    ratio = 4
    B = 8
    G = 2 * ratio
    D_in = 2 * HEAD_DIM
    kv, sc = _make_state(B, G, D_in, seed=60)
    weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.4 + 1.0
    freqs = _make_freqs(B, ROPE_HEAD_DIM, seed=61)
    slots = torch.arange(B, dtype=torch.int64, device="cuda")
    cos, sin = freqs_cis_to_cos_sin(freqs)
    cache = _make_cache(1, 16)
    v4_compressor_fused(
        kv,
        sc,
        slots,
        weight,
        cos,
        sin,
        cache,
        overlap=True,
        head_dim=HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        norm_eps=EPS,
    )
    ref_vec = _ref_pool_rmsnorm_rope(
        kv,
        sc,
        weight,
        freqs,
        overlap=True,
        head_dim=HEAD_DIM,
        rope_head_dim=ROPE_HEAD_DIM,
        eps=EPS,
    )  # [B, 128] bf16
    recon = dequantize_indexer_k(cache, slots, out_dtype=torch.bfloat16)
    diff = (recon.float() - ref_vec.float()).abs()
    ref_max = ref_vec.float().abs().amax().item()
    # Use magnitude-relative — per-element rel explodes on near-zero ref
    # entries (FP8 quant noise has fixed abs scale ~ ref_max/127).
    rel_mean = diff.mean().item() / (ref_max + 1e-6)
    rel_max = diff.amax().item() / (ref_max + 1e-6)
    print(
        f"  [round-trip vs ref] rel_max={rel_max:.4f}  rel_mean={rel_mean:.4f}  "
        f"ref_max={ref_max:.4f}"
    )
    assert (
        rel_max < 0.15
    ), f"fused FP8 round-trip vs ref bf16 exceeds fp8 noise: rel_max={rel_max:.4f}"


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------
def bench_decode():
    print("\n  decode-shape bench (4-launch chain vs fused)")
    print(
        "    {:<28}  {:>10}  {:>10}  {:>10}".format(
            "case", "4-launch", "fused", "speedup"
        )
    )
    cases = [
        ("CSA decode B=1", 1, True, 4),
        ("CSA decode B=4", 4, True, 4),
        ("CSA decode B=16", 16, True, 4),
        ("HCA decode B=1", 1, False, 128),
        ("HCA decode B=4", 4, False, 128),
    ]
    for name, B, overlap, ratio in cases:
        G = 2 * ratio if overlap else ratio
        D_in = 2 * HEAD_DIM if overlap else HEAD_DIM
        kv, sc = _make_state(B, G, D_in, seed=100)
        weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        freqs = _make_freqs(B, ROPE_HEAD_DIM, seed=101)
        slots = torch.arange(B, dtype=torch.int64, device="cuda")
        cos, sin = freqs_cis_to_cos_sin(freqs)
        block_size = 64
        num_blocks = max(1, (B + block_size - 1) // block_size)
        cache = _make_cache(num_blocks, block_size)

        def run_ref():
            _ref_full_chain(
                kv,
                sc,
                slots,
                weight,
                freqs,
                cache,
                overlap=overlap,
                head_dim=HEAD_DIM,
                rope_head_dim=ROPE_HEAD_DIM,
                eps=EPS,
            )

        def run_fused():
            v4_compressor_fused(
                kv,
                sc,
                slots,
                weight,
                cos,
                sin,
                cache,
                overlap=overlap,
                head_dim=HEAD_DIM,
                rope_head_dim=ROPE_HEAD_DIM,
                norm_eps=EPS,
            )

        t_r = _bench(run_ref)
        t_f = _bench(run_fused)
        marker = "" if t_f < t_r else " (REGRESS)"
        print(
            f"    {name:<28}  {t_r*1e3:8.2f}us  {t_f*1e3:8.2f}us  "
            f"{t_r/t_f:8.2f}x{marker}"
        )


if __name__ == "__main__":
    print("== Correctness ==")
    test_csa_decode()
    test_hca_decode()
    test_csa_prefill_chunk()
    test_hca_prefill_chunk()
    test_empty_no_op()
    test_round_trip_via_dequant()
    print("\n== Benchmark ==")
    bench_decode()
    print("\nOK")
