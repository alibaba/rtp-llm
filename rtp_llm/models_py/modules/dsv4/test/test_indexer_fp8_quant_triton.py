"""UT for ``_indexer_fp8_quant_triton`` (quantize + dequantize).

Locks down the per-slot 132-byte indexer FP8 cache layout matching
``_pool_spec[INDEXER_KV] = (uint8, 132)`` introduced by upstream commit
``eb64793f5 Wire DSV4 Python KV cache dtype``.

References (must match these byte-for-byte):
  * ``TestIndexerFp8LayoutRoundTrip`` in
    ``rtp_llm/models_py/modules/dsv4/decode/test/fp8_kv_decode_path_test.py``
    — pure-torch reference quantize/dequant, same algorithm.
  * vLLM's ``indexer_k_quant_and_cache`` kernel
    (``vllm/csrc/cache_kernels.cu:515``) — same per-vector absmax + ue8m0
    or fp32 scale, but writes scales in a separate per-block region rather
    than appended per-slot, so the within-block layout differs.

Run:
  cd .../github-opensource && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \\
    /opt/conda310/bin/python3 \\
    rtp_llm/models_py/modules/dsv4/test/test_indexer_fp8_quant_triton.py
"""

from __future__ import annotations

import torch

from rtp_llm.models_py.modules.dsv4.fp8._indexer_quant_triton import (
    FP8_E4M3_MAX,
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
    dequantize_indexer_k,
    quantize_indexer_k,
)


# ---------------------------------------------------------------------------
# Reference: vLLM/DeepGEMM per-block layout
#   block bytes [0 : block_size * 128)        = K bytes (token-major)
#   block bytes [block_size * 128 : *132)     = scale bytes (one fp32/token)
# ---------------------------------------------------------------------------
def ref_quantize_into_pool(
    k_bf16: torch.Tensor,  # [T, 128] bf16
    slot_mapping: torch.Tensor,  # [T] int64
    pool: torch.Tensor,  # [num_blocks, block_size, 132] uint8 (zeroed)
) -> None:
    T, D = k_bf16.shape
    assert D == INDEXER_HEAD_DIM
    block_size = pool.shape[1]
    absmax = k_bf16.abs().float().amax(dim=-1).clamp(min=1e-12)
    scale = (absmax / FP8_E4M3_MAX).contiguous()
    q_fp8 = (
        (k_bf16.float() / scale.unsqueeze(-1))
        .clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    # vLLM-layout view: [num_blocks, K-region(block_size*128) | scale-region(block_size*4)]
    flat = pool.view(pool.shape[0], block_size * INDEXER_ENTRY_BYTES)
    for t in range(T):
        s = int(slot_mapping[t].item())
        if s < 0:
            continue
        b, off = s // block_size, s % block_size
        # K region: offset [off*128 : (off+1)*128)
        flat[b, off * D : (off + 1) * D] = q_fp8[t]
        # Scale region: offset [block_size*128 + off*4 : +4)
        sb = scale[t : t + 1].contiguous().view(torch.uint8)  # [4] uint8
        scale_off = block_size * D + off * 4
        flat[b, scale_off : scale_off + 4] = sb


def ref_dequantize_from_pool(
    pool: torch.Tensor,
    slot_mapping: torch.Tensor,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    T = slot_mapping.shape[0]
    block_size = pool.shape[1]
    flat = pool.view(pool.shape[0], block_size * INDEXER_ENTRY_BYTES)
    out = torch.zeros(T, INDEXER_HEAD_DIM, dtype=out_dtype, device=pool.device)
    for t in range(T):
        s = int(slot_mapping[t].item())
        if s < 0:
            continue
        b, off = s // block_size, s % block_size
        D = INDEXER_HEAD_DIM
        k_bytes = flat[b, off * D : (off + 1) * D]
        q_f32 = k_bytes.view(torch.float8_e4m3fn).to(torch.float32)
        scale_off = block_size * D + off * 4
        scale = flat[b, scale_off : scale_off + 4].view(torch.float32).item()
        out[t] = (q_f32 * scale).to(out_dtype)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_cache(num_blocks: int, block_size: int) -> torch.Tensor:
    return torch.zeros(
        num_blocks,
        block_size,
        INDEXER_ENTRY_BYTES,
        dtype=torch.uint8,
        device="cuda",
    )


def _bench(fn, *a, warmup: int = 50, iters: int = 500) -> float:
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


# ---------------------------------------------------------------------------
# Quantize correctness
# ---------------------------------------------------------------------------
def test_quantize_single_block():
    """T=8 tokens fit in one block (block_size=64).  Pool bytes must
    match the per-block grouped reference (vLLM layout)."""
    torch.manual_seed(0)
    T, block_size = 8, 64
    k = torch.randn(T, INDEXER_HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.5
    slot_mapping = torch.arange(T, dtype=torch.int64, device="cuda")
    cache = _make_cache(num_blocks=1, block_size=block_size)
    quantize_indexer_k(k, slot_mapping, cache)

    cache_ref = _make_cache(num_blocks=1, block_size=block_size)
    ref_quantize_into_pool(k, slot_mapping, cache_ref)
    assert torch.equal(cache, cache_ref), "single-block layout diverges from vLLM ref"
    print(f"  [single block] T={T} block_size={block_size} OK")


def test_quantize_multi_block():
    torch.manual_seed(1)
    T, block_size = 200, 64
    num_blocks = (T + block_size - 1) // block_size
    k = torch.randn(T, INDEXER_HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.5
    slot_mapping = torch.arange(T, dtype=torch.int64, device="cuda")
    cache = _make_cache(num_blocks=num_blocks, block_size=block_size)
    quantize_indexer_k(k, slot_mapping, cache)

    cache_ref = _make_cache(num_blocks=num_blocks, block_size=block_size)
    ref_quantize_into_pool(k, slot_mapping, cache_ref)
    assert torch.equal(cache, cache_ref), "multi-block round-trip failed"
    print(f"  [multi block] T={T} blocks={num_blocks} OK")


def test_quantize_skip_negative_slot():
    torch.manual_seed(2)
    T, block_size = 4, 8
    k = torch.randn(T, INDEXER_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    slot_mapping = torch.tensor([0, -1, 2, -1], dtype=torch.int64, device="cuda")
    cache = _make_cache(num_blocks=1, block_size=block_size)
    quantize_indexer_k(k, slot_mapping, cache)

    cache_ref = _make_cache(num_blocks=1, block_size=block_size)
    ref_quantize_into_pool(k, slot_mapping, cache_ref)
    assert torch.equal(cache, cache_ref)
    # And the dead slot regions stay zero.
    flat = cache.view(1, block_size * INDEXER_ENTRY_BYTES)
    D = INDEXER_HEAD_DIM
    assert (flat[0, 1 * D : 2 * D] == 0).all(), "slot 1 K bytes leaked"
    assert (flat[0, 3 * D : 4 * D] == 0).all(), "slot 3 K bytes leaked"
    print("  [skip -1 slot] OK")


def test_quantize_slot_remap_non_identity():
    torch.manual_seed(3)
    T, block_size = 6, 4
    k = torch.randn(T, INDEXER_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    slots = torch.tensor([3, 7, 1, 5, 2, 6], dtype=torch.int64, device="cuda")
    cache = _make_cache(num_blocks=2, block_size=block_size)
    quantize_indexer_k(k, slots, cache)
    cache_ref = _make_cache(num_blocks=2, block_size=block_size)
    ref_quantize_into_pool(k, slots, cache_ref)
    assert torch.equal(cache, cache_ref), "non-monotone slot remap diverges"
    print("  [non-monotone slot map] OK")


def test_empty_T():
    cache = _make_cache(1, 8)
    cache_pre = cache.clone()
    k = torch.empty(0, INDEXER_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    slots = torch.empty(0, dtype=torch.int64, device="cuda")
    quantize_indexer_k(k, slots, cache)
    assert torch.equal(cache, cache_pre), "T=0 should be a no-op"
    print("  [T=0 no-op] OK")


# ---------------------------------------------------------------------------
# Dequantize correctness
# ---------------------------------------------------------------------------
def test_round_trip_fp32():
    """quant → dequant recovers within fp8_e4m3 precision."""
    torch.manual_seed(10)
    T, block_size = 16, 32
    k = torch.randn(T, INDEXER_HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.5
    slots = torch.arange(T, dtype=torch.int64, device="cuda")
    cache = _make_cache(1, block_size)
    quantize_indexer_k(k, slots, cache)
    recon = dequantize_indexer_k(cache, slots, out_dtype=torch.float32)
    rel = (recon - k.float()).abs() / (k.float().abs() + 1e-6)
    print(f"  [round-trip fp32] max_rel={rel.amax().item():.4f}")
    assert rel.amax().item() < 0.15, f"fp8 precision exceeded: {rel.amax().item():.4f}"


def test_round_trip_bf16_out():
    torch.manual_seed(11)
    T, block_size = 8, 16
    k = torch.randn(T, INDEXER_HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.3
    slots = torch.arange(T, dtype=torch.int64, device="cuda")
    cache = _make_cache(1, block_size)
    quantize_indexer_k(k, slots, cache)
    recon = dequantize_indexer_k(cache, slots, out_dtype=torch.bfloat16)
    assert recon.dtype == torch.bfloat16
    rel = (recon.float() - k.float()).abs() / (k.float().abs() + 1e-6)
    print(f"  [round-trip bf16] max_rel={rel.amax().item():.4f}")
    assert rel.amax().item() < 0.2


def test_dequant_padded_yields_zero():
    """slot==-1 in dequant gives zeros."""
    torch.manual_seed(12)
    T, block_size = 4, 8
    k = torch.randn(T, INDEXER_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    slots_w = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="cuda")
    cache = _make_cache(1, block_size)
    quantize_indexer_k(k, slots_w, cache)
    slots_r = torch.tensor([0, -1, 2, -1], dtype=torch.int64, device="cuda")
    recon = dequantize_indexer_k(cache, slots_r, out_dtype=torch.float32)
    assert (recon[1] == 0).all() and (recon[3] == 0).all()
    print("  [dequant -1 → zero] OK")


def test_dequant_matches_reference():
    """Dequant of cache populated by kernel == ref_dequantize_per_slot."""
    torch.manual_seed(13)
    T, block_size = 12, 32
    k = torch.randn(T, INDEXER_HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 0.4
    slots = torch.arange(T, dtype=torch.int64, device="cuda")
    cache = _make_cache(1, block_size)
    quantize_indexer_k(k, slots, cache)

    cand = dequantize_indexer_k(cache, slots, out_dtype=torch.float32)
    ref = ref_dequantize_from_pool(cache, slots, out_dtype=torch.float32)
    assert torch.equal(cand, ref), "kernel dequant diverges from reference"
    print("  [dequant vs ref] OK")


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------
def bench_quantize_decode():
    """Decode-shape bench: T=B*1 tokens / step (B=1..16)."""
    print("\n  quantize_indexer_k — kernel vs torch reference")
    print("    {:>3}  {:>10}  {:>10}  {:>10}".format("B", "torch", "kernel", "speedup"))
    fail = []
    for B in (1, 4, 16, 64):
        T = B
        block_size = 64
        num_blocks = max(1, (T + block_size - 1) // block_size)
        k = torch.randn(T, INDEXER_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        slots = torch.arange(T, dtype=torch.int64, device="cuda")
        cache = _make_cache(num_blocks, block_size)

        def run_torch():
            ref_quantize_into_pool(k, slots, cache)

        def run_kernel():
            quantize_indexer_k(k, slots, cache)

        t_t = _bench(run_torch, warmup=10, iters=50)
        t_k = _bench(run_kernel, warmup=20, iters=200)
        marker = "" if t_k < t_t else " (REGRESS)"
        print(
            f"    {B:3d}  {t_t*1e3:8.2f}us  {t_k*1e3:8.2f}us  {t_t/t_k:8.2f}x{marker}"
        )
        if not (t_k < t_t):
            fail.append(B)
    assert not fail, f"quantize_indexer_k slower than torch ref at B={fail}"


# ---------------------------------------------------------------------------
# Integration: FP8 cache → dequant → v4_indexer_score
# Validates the round-trip pipeline that an indexer running on FP8 cache
# would actually take: write bf16 K to FP8 cache, gather + dequant to bf16,
# feed to the score kernel, compare against direct bf16 score over the
# same logical K.
# ---------------------------------------------------------------------------
def test_fp8_cache_dequant_then_score():
    """Score(dequant(quant(K))) should approximate Score(K) within fp8 noise."""
    from rtp_llm.models_py.modules.dsv4._indexer_score_triton import v4_indexer_score

    torch.manual_seed(20)
    B, S, T, H, D = 1, 1, 64, 64, INDEXER_HEAD_DIM
    block_size = 16
    num_blocks = (T + block_size - 1) // block_size

    q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
    weights = torch.randn(B, S, H, dtype=torch.bfloat16, device="cuda")
    k_bf16 = torch.randn(T, D, dtype=torch.bfloat16, device="cuda") * 0.5
    slots = torch.arange(T, dtype=torch.int64, device="cuda")

    # FP8 path: quant → dequant
    cache = _make_cache(num_blocks, block_size)
    quantize_indexer_k(k_bf16, slots, cache)
    k_recon = (
        dequantize_indexer_k(cache, slots, out_dtype=torch.bfloat16)
        .view(B, T, D)
        .contiguous()
    )

    # Direct bf16 path
    k_direct = k_bf16.view(B, T, D).contiguous()

    score_fp8 = v4_indexer_score(
        q.contiguous(), k_recon, weights.contiguous(), q_pos=None, compress_ratio=4
    )
    score_bf16 = v4_indexer_score(
        q.contiguous(), k_direct, weights.contiguous(), q_pos=None, compress_ratio=4
    )

    # Per-element rel-diff is dominated by near-zero score positions; the
    # MEAN is the meaningful aggregate.  Top-K overlap is what the indexer
    # actually cares about — locks down the high-magnitude positions.
    diff = (score_fp8 - score_bf16).abs()
    score_max = score_bf16.abs().amax().item()
    print(
        f"  [fp8→dequant→score] mean_abs={diff.mean().item():.4f}  "
        f"max_abs={diff.amax().item():.4f}  bf16_max={score_max:.4f}"
    )
    # Mean abs error should be small relative to the score magnitude.
    assert (
        diff.mean().item() < 0.05 * score_max
    ), f"fp8 mean error {diff.mean().item():.4f} > 5% of score range {score_max:.4f}"

    # Top-K agreement is what actually matters for the indexer.
    topk_fp8 = score_fp8.topk(8, dim=-1)[1].squeeze().tolist()
    topk_bf = score_bf16.topk(8, dim=-1)[1].squeeze().tolist()
    overlap = len(set(topk_fp8) & set(topk_bf))
    print(f"  [top-8 overlap] {overlap}/8")
    assert overlap >= 6, f"top-8 overlap only {overlap}/8 — fp8 noise broke selection"


if __name__ == "__main__":
    print("== Quantize correctness ==")
    test_quantize_single_block()
    test_quantize_multi_block()
    test_quantize_skip_negative_slot()
    test_quantize_slot_remap_non_identity()
    test_empty_T()
    print("\n== Dequantize correctness ==")
    test_round_trip_fp32()
    test_round_trip_bf16_out()
    test_dequant_padded_yields_zero()
    test_dequant_matches_reference()
    print("\n== Integration ==")
    test_fp8_cache_dequant_then_score()
    print("\n== Benchmark ==")
    bench_quantize_decode()
    print("\nOK")
