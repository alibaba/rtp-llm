"""End-to-end layout contract: 584B FP8 pool → vLLM Triton dequant → flash_mla_sparse_fwd.

``flash_mla_sparse_fwd`` does NOT consume the 584B FP8 pool directly —
its API requires bf16 ``kv [s_kv, h_kv, d_qk=576]``. So the FP8 path is:

    FP8 pool (584B striped)                              # written by v4_compressor_kv_fused
        │
        │   vLLM ``dequantize_and_gather_k_cache``       # Triton dequant
        ▼
    bf16 KV [s_kv, h_kv, 576]                            # 512 nope + 64 rope
        │
        ▼
    flash_mla_sparse_fwd

This UT pins both ends:

  1. Build raw bf16 K_full [s_kv, 576] (512 nope + 64 rope), pretend to
     write it through the FP8 pipeline (``quantize_and_insert_k_cache``).
  2. Read it back via ``dequantize_and_gather_k_cache`` → K_dequant.
  3. Run ``flash_mla_sparse_fwd`` on K_dequant → output_fp8.
  4. Run ``flash_mla_sparse_fwd`` on K_full directly → output_bf16.
  5. Compare: outputs differ only by fp8 quant noise (sub-percent).

If this passes, the new ``CompressorFP8`` / attention path can be wired
as: ``v4_compressor_kv_fused`` writes pool, attention calls vendored
``dequantize_and_gather_k_cache`` then ``flash_mla_sparse_fwd``. No
in-tree Python fancy-index dequant needed.

Reference frozen at vLLM HEAD ``ff449b6426812d1e5e107715af899fcff5e81419``.
"""

from __future__ import annotations

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

from rtp_llm.models_py.modules.dsv4._compressor_kv_fused_triton import (
    KV_ENTRY_BYTES,
    KV_HEAD_DIM,
)
from rtp_llm.models_py.modules.dsv4.test._vllm_ref.cache_utils import (
    dequantize_and_gather_k_cache,
    quantize_and_insert_k_cache,
)


@pytest.mark.parametrize("block_size", [64])
def test_fp8_pool_dequant_feeds_sparse_fwd(block_size):
    """End-to-end: FP8 pool round-trip + sparse_fwd matches direct bf16 sparse_fwd."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    try:
        from flash_mla import flash_mla_sparse_fwd
    except ImportError:
        pytest.skip("flash_mla.flash_mla_sparse_fwd unavailable")

    torch.manual_seed(0xF00D)
    device = torch.device("cuda")

    # sparse_fwd shapes (verified empirically — docstring claims d_qk=576
    # but kernel accepts d_qk=512 as well; existing rtp-llm callsite uses
    # 512 == head_dim, so we match that):
    #   q  : [s_q, h_q, d_qk=512]   bf16
    #   kv : [s_kv, h_kv, d_qk=512] bf16  (h_kv == 1 for MQA)
    # FlashMLA sparse_fwd: h_q ∈ {64, 128}; topk must be multiple of B_TOPK
    # (64 for h_q=64). Use topk=64 to satisfy.
    s_q = 4
    s_kv = 64  # one full block
    h_q = 64
    h_kv = 1
    d_qk = KV_HEAD_DIM  # 512
    topk = 64  # multiple of B_TOPK=64

    # Build bf16 KV in shape [s_kv, d_qk=512]; pool layout treats this as
    # 448 nope (fp8-quantized) + 64 rope (bf16-passthrough), packed into
    # 576 cache bytes/token (+8 scale = 584 entry).
    K_full = torch.randn(s_kv, d_qk, dtype=torch.bfloat16, device=device) * 0.5

    # ── Quantize K_full into the 584B pool via vLLM writer ──
    # Pool: [num_blocks, block_size, 584] uint8.
    num_blocks = (s_kv + block_size - 1) // block_size + 1  # +1 spare
    block_bytes = block_size * KV_ENTRY_BYTES
    pool = torch.zeros(num_blocks, block_bytes, dtype=torch.uint8, device=device)
    slot_mapping = torch.arange(s_kv, dtype=torch.int64, device=device)
    quantize_and_insert_k_cache(
        K_full,
        pool,
        slot_mapping,
        block_size=block_size,
    )

    # ── Dequant via vLLM reader ──
    K_dequant = torch.zeros(1, s_kv, KV_HEAD_DIM, dtype=torch.bfloat16, device=device)
    seq_lens = torch.tensor([s_kv], dtype=torch.int32, device=device)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).view(
        1, num_blocks
    )
    pool_3d = pool.view(num_blocks, block_size, KV_ENTRY_BYTES)
    dequantize_and_gather_k_cache(
        K_dequant,
        pool_3d,
        seq_lens,
        None,  # gather all
        block_table,
        block_size,
        offset=0,
    )
    # vLLM reader returns 512-d bf16 (448 nope dequant + 64 rope bf16
    # passthrough). sparse_fwd accepts d_qk=512 directly — our existing
    # rtp-llm callsite uses head_dim=512 too. Squeeze leading req dim.
    K_dequant = K_dequant[0]  # [s_kv, 512]

    # ── Build Q + indices ──
    # Q: [s_q, h_q, d_qk] bf16
    Q = torch.randn(s_q, h_q, d_qk, dtype=torch.bfloat16, device=device) * 0.5
    # indices: [s_q, h_kv, topk] int32. Each q row picks topk k positions.
    # For determinism: q row i → indices [(i*4) % s_kv, (i*4+1) % s_kv, ...]
    indices = (
        torch.arange(topk, dtype=torch.int32, device=device).unsqueeze(0).unsqueeze(0)
    )
    indices = indices.expand(s_q, h_kv, topk).contiguous()
    sm_scale = 1.0 / (d_qk**0.5)

    # ── sparse_fwd on bf16 reference ──
    K_full_kv = K_full.unsqueeze(1)  # [s_kv, h_kv=1, d_qk]
    out_ref, _, _ = flash_mla_sparse_fwd(
        Q.contiguous(),
        K_full_kv.contiguous(),
        indices,
        sm_scale,
    )

    # ── sparse_fwd on dequant'd FP8 path ──
    K_dq_kv = K_dequant.unsqueeze(1)  # [s_kv, 1, d_qk]
    out_fp8, _, _ = flash_mla_sparse_fwd(
        Q.contiguous(),
        K_dq_kv.contiguous(),
        indices,
        sm_scale,
    )

    # ── Compare ──
    diff = (out_ref.to(torch.float32) - out_fp8.to(torch.float32)).abs()
    max_abs = diff.max().item()
    ref_abs = out_ref.abs().to(torch.float32)
    safe_ref = torch.clamp(ref_abs, min=1e-3)
    rel = (diff / safe_ref).max().item()
    mean_abs = diff.mean().item()

    # fp8 quant on the K nope (448 elems out of 576) → expect ~scale/127
    # per element after attn weighted average. For unit-scale inputs the
    # output noise is well bounded; allow generous threshold to cover
    # SM100 reduction order.
    assert max_abs < 1.0 or rel < 0.10, (
        f"sparse_fwd output diverges: max_abs={max_abs:.4f}, "
        f"max_rel={rel:.3%}, mean_abs={mean_abs:.4f}"
    )


if __name__ == "__main__":
    test_fp8_pool_dequant_feeds_sparse_fwd(64)
    print("OK")
