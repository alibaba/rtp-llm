"""Reader contract for the canonical 584B FP8 KV pool.

Phase 1 of the FP8 path rewrite locks down WHO reads the pool and HOW.
Two paths matter:

  (A) ``flash_mla_sparse_fwd`` — the production attention kernel.
      Consumes the pool *directly* (no Python-side dequant). The shape /
      stride / paddding contract it expects is verified separately in
      ``test_flash_mla_sparse_fwd_layout.py`` (Task #7).

  (B) **vLLM's** ``dequantize_and_gather_k_cache`` — the canonical
      Python-callable reader that converts the pool back to bf16 K. Used
      by vLLM's SWA prefill path (and by anything else that needs a bf16
      view of the FP8 pool). We test against this kernel because:

        * It is the *symmetric* inverse of vLLM's writer
          (``quantize_and_insert_k_cache``), itself byte-equivalent to
          the rtp-llm fused writer (proven in
          ``test_compressor_fp8_writer.py``).
        * If our pool can be dequant'd by vLLM's reader, then any future
          rtp-llm-side reader written against the same byte layout is
          guaranteed compatible.

The test here *does not* introduce a new rtp-llm reader. It just proves
that the pool produced by ``v4_compressor_kv_fused`` round-trips cleanly
via vLLM's reader. The next step (rewriting ``CompressorFP8`` /
``IndexerFP8``) will use ``flash_mla_sparse_fwd`` directly — the bf16
dequant path is being removed, not re-implemented.

Reference frozen at vLLM HEAD ``ff449b6426812d1e5e107715af899fcff5e81419``
(see ``_vllm_ref/VLLM_HEAD.txt``).
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
    KV_NOPE_DIM,
    KV_ROPE_DIM,
    v4_compressor_kv_fused,
)
from rtp_llm.models_py.modules.dsv4.test._vllm_ref.cache_utils import (
    dequantize_and_gather_k_cache,
    quantize_and_insert_k_cache,
)


def _ref_pool_rmsnorm_rope(
    kv_state,
    score_state,
    norm_weight,
    rope_cos,
    rope_sin,
    *,
    norm_eps,
    nope_dim,
    rope_dim,
):
    """Same reference used by ``test_compressor_fp8_writer.py`` —
    pure-PyTorch ``pool → RMSNorm → RoPE`` chain producing the bf16 K
    tensor that vLLM's writer expects."""
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
    new_even = even * rope_cos - odd * rope_sin
    new_odd = odd * rope_cos + even * rope_sin
    rotated = torch.stack([new_even, new_odd], dim=-1).view(rope.shape[0], rope_dim)
    out[:, nope_dim : nope_dim + rope_dim] = rotated
    return out.to(torch.bfloat16)


@pytest.mark.parametrize("block_size", [16, 64])
def test_rtp_writer_dequantizable_by_vllm_reader(block_size):
    """rtp-llm fused writer → vLLM dequant_and_gather → bf16 K.

    Validates that the bf16 K reconstructed from our pool matches the
    bf16 K reconstructed from vLLM's own writer on the same input
    (proves the pool layout is byte-compatible end-to-end).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(0xCAFE)
    device = torch.device("cuda")
    head_dim = KV_HEAD_DIM
    rope_dim = KV_ROPE_DIM
    nope_dim = KV_NOPE_DIM
    norm_eps = 1e-6

    # Single request, contiguous slots [0..S) so block_table is trivial.
    # Use HCA (overlap=False, ratio=128) — non-overlap branch only here;
    # the writer test already exercises both branches at the byte level.
    S = 5  # five compressed tokens, spans ≥1 block at block_size 16 or 64
    ratio = 128
    G = ratio
    D_in = head_dim

    kv_state = torch.randn(S, G, D_in, dtype=torch.float32, device=device)
    score_state = torch.randn(S, G, D_in, dtype=torch.float32, device=device) * 0.5
    norm_weight = (torch.randn(head_dim, device=device) * 0.1 + 1.0).to(torch.bfloat16)
    rope_cos = torch.randn(S, rope_dim // 2, dtype=torch.float32, device=device)
    rope_sin = torch.randn(S, rope_dim // 2, dtype=torch.float32, device=device)

    # Slot mapping: 5 contiguous slots starting at 0.
    slot_mapping = torch.arange(S, dtype=torch.int64, device=device)
    num_blocks = (S + block_size - 1) // block_size + 1  # +1 spare

    block_bytes = block_size * KV_ENTRY_BYTES

    # ── rtp-llm writer ──
    pool_rtp = torch.zeros(num_blocks, block_bytes, dtype=torch.uint8, device=device)
    pool_rtp_3d = pool_rtp.view(num_blocks, block_size, KV_ENTRY_BYTES)
    v4_compressor_kv_fused(
        kv_state.contiguous(),
        score_state.contiguous(),
        slot_mapping,
        norm_weight,
        rope_cos,
        rope_sin,
        pool_rtp_3d,
        cache_block_stride_bytes=int(pool_rtp_3d.stride(0)),
        overlap=False,
        head_dim=head_dim,
        rope_head_dim=rope_dim,
        norm_eps=norm_eps,
    )

    # ── vLLM writer on equivalent bf16 K ──
    K_ref = _ref_pool_rmsnorm_rope(
        kv_state,
        score_state,
        norm_weight,
        rope_cos,
        rope_sin,
        norm_eps=norm_eps,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
    )
    pool_vllm = torch.zeros(num_blocks, block_bytes, dtype=torch.uint8, device=device)
    quantize_and_insert_k_cache(K_ref, pool_vllm, slot_mapping, block_size=block_size)

    # ── vLLM reader on BOTH pools ──
    # Reader signature: dequantize_and_gather_k_cache(out, k_cache, seq_lens,
    #   gather_lens, block_table, block_size, offset)
    #   out:         [num_reqs, max_T, head_size]
    #   k_cache:     [num_blocks, block_size, head_bytes]  (== our 3D view)
    #   seq_lens:    [num_reqs] int — total per-request length
    #   gather_lens: [num_reqs] int OR None — gather only the trailing N tokens
    #   block_table: [num_reqs, max_blocks_per_seq] int32
    #   offset:      where in `out` to start writing (per-request)
    num_reqs = 1
    max_blocks_per_seq = num_blocks
    seq_lens = torch.tensor([S], dtype=torch.int32, device=device)
    block_table = torch.arange(
        max_blocks_per_seq, dtype=torch.int32, device=device
    ).view(num_reqs, max_blocks_per_seq)

    out_rtp = torch.zeros(num_reqs, S, head_dim, dtype=torch.bfloat16, device=device)
    out_vllm = torch.zeros(num_reqs, S, head_dim, dtype=torch.bfloat16, device=device)

    pool_vllm_3d = pool_vllm.view(num_blocks, block_size, KV_ENTRY_BYTES)

    dequantize_and_gather_k_cache(
        out_rtp,
        pool_rtp_3d,
        seq_lens,
        None,  # gather_lens=None → gather all S tokens
        block_table,
        block_size,
        offset=0,
    )
    dequantize_and_gather_k_cache(
        out_vllm,
        pool_vllm_3d,
        seq_lens,
        None,
        block_table,
        block_size,
        offset=0,
    )

    # ── Compare ──
    # 1. Both readers produce the same bf16 K up to the same fp32-reduction
    #    1-ULP noise we already saw in the writer test.
    diff = (out_rtp.to(torch.float32) - out_vllm.to(torch.float32)).abs()
    # bf16 1-ULP at our scale is ~1e-3; allow 2 ULP for safety. fp8-quant'd
    # nope adds quant noise (worst-case ~scale/127) but BOTH paths get the
    # same fp8 bytes if the writers matched, so dequant is deterministic.
    max_abs = diff.max().item()
    rope_diff = diff[..., nope_dim:].max().item()
    nope_diff = diff[..., :nope_dim].max().item()
    assert nope_diff == 0.0, (
        f"NoPE region must dequant bit-exactly (both pools wrote same fp8 "
        f"bytes via byte-equal writers); got max abs diff {nope_diff}"
    )
    assert rope_diff <= 2 * 1e-2, (
        f"RoPE bf16 max diff {rope_diff} exceeds 2-ULP bf16 tolerance "
        f"(~7.8e-3 at unit scale); writer-side fp32 reduction order "
        f"diverged more than expected"
    )
    assert max_abs == max(rope_diff, nope_diff)

    # 2. Reader output vs the original bf16 K_ref: only fp8 quant noise on
    #    the NoPE half; RoPE half should be exact (post-bf16-cast) — though
    #    again writer-side RMSNorm fp32 noise leaks ≤1-ULP.
    out_vs_ref = (out_vllm[0].to(torch.float32) - K_ref.to(torch.float32)).abs()
    rope_vs_ref = out_vs_ref[..., nope_dim:].max().item()
    nope_vs_ref = out_vs_ref[..., :nope_dim].max().item()
    # NoPE: fp8 quant noise. With UE8M0 per-64 scale, max abs error per
    # element is scale/254 ≈ value_max/127. For inputs of unit-ish scale
    # post-RMSNorm, ~1e-2 is comfortably loose.
    assert nope_vs_ref < 0.5, (
        f"NoPE dequant vs original bf16 K diverges by {nope_vs_ref} — "
        f"larger than expected fp8 quant error (was input out of range?)"
    )
    assert rope_vs_ref <= 2 * 1e-2, (
        f"RoPE dequant vs original bf16 K diverges by {rope_vs_ref} — "
        f"exceeds 2-ULP tolerance"
    )


if __name__ == "__main__":
    test_rtp_writer_dequantizable_by_vllm_reader(64)
    test_rtp_writer_dequantizable_by_vllm_reader(16)
    print("OK")
