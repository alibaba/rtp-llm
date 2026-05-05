"""Byte-level diff: rtp-llm fused 584B compressor writer vs vLLM reference.

The rtp-llm kernel ``v4_compressor_kv_fused`` fuses pool→RMSNorm→RoPE→FP8
quant→striped scatter into the canonical fp8_model1_mla layout consumed by
``flash_mla_sparse_fwd``. The vLLM reference ``quantize_and_insert_k_cache``
covers only the FP8 quant + striped scatter half (its caller already did
pool/RMSNorm/RoPE).

To compare apples-to-apples we:

  1. Sample raw ``(kv_state, score_state)`` fp32 inputs the rtp-llm kernel
     consumes.
  2. Run a pure-PyTorch reference for ``pool → RMSNorm → RoPE`` to derive
     the bf16 K tensor that vLLM's writer expects.
  3. Run the rtp-llm fused kernel on raw inputs.
  4. Run vLLM's writer on the reference bf16 K.
  5. Byte-diff the two pool buffers.

Bit-equality requirements are tight: both kernels do the bf16 round-trip
on the post-RMSNorm value before FP8 quant, both encode UE8M0 the same
way (ceil(log2(absmax/fp8_max)) → exponent+127), and both store rope as
bf16 directly. Any divergence is a real bug.

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
    KV_QUANT_BLOCK,
    KV_ROPE_DIM,
    KV_TOKEN_DATA_SIZE,
    v4_compressor_kv_fused,
)
from rtp_llm.models_py.modules.dsv4.test._vllm_ref.cache_utils import (
    quantize_and_insert_k_cache,
)


# ---------------------------------------------------------------------------
# Pure-PyTorch reference: pool → RMSNorm → RoPE → bf16 K (vLLM writer input)
# ---------------------------------------------------------------------------
def _ref_pool_rmsnorm_rope(
    kv_state: torch.Tensor,  # [B, G, D] fp32
    score_state: torch.Tensor,  # [B, G, D] fp32
    norm_weight: torch.Tensor,  # [D] bf16
    rope_cos: torch.Tensor,  # [B, rope_dim/2] fp32
    rope_sin: torch.Tensor,  # [B, rope_dim/2] fp32
    *,
    norm_eps: float,
    nope_dim: int,
    rope_dim: int,
) -> torch.Tensor:
    """Returns ``[B, head_dim] bf16`` matching what vLLM's writer expects.

    Mirrors the rtp-llm kernel's compute order so post-RoPE bf16 bytes
    are bit-equal to what the kernel writes.
    """
    # ── softmax over G, weighted-sum (a.k.a. "pool") ──
    score_sm = torch.softmax(score_state, dim=1)  # fp32
    pooled = (kv_state * score_sm).sum(dim=1)  # [B, D] fp32

    # ── RMSNorm in fp32 with bf16 weight cast to fp32 ──
    w_fp32 = norm_weight.to(torch.float32)
    var = (pooled * pooled).mean(dim=-1, keepdim=True)
    rrms = torch.rsqrt(var + norm_eps)
    normed = pooled * rrms * w_fp32  # [B, D] fp32

    # ── RoPE on the trailing rope_dim, GPT-J pair-interleaved ──
    out = normed.clone()  # NoPE half kept as fp32 for now
    rope = normed[:, nope_dim : nope_dim + rope_dim]  # [B, rope_dim] fp32
    pair = rope.view(rope.shape[0], rope_dim // 2, 2)
    even = pair[..., 0]  # [B, rope_dim/2]
    odd = pair[..., 1]
    cos = rope_cos  # [B, rope_dim/2]
    sin = rope_sin
    new_even = even * cos - odd * sin
    new_odd = odd * cos + even * sin
    rotated = torch.stack([new_even, new_odd], dim=-1).view(rope.shape[0], rope_dim)
    out[:, nope_dim : nope_dim + rope_dim] = rotated
    return out.to(torch.bfloat16)


def _allocate_pool(num_blocks: int, block_size: int, *, device):
    """Flat ``[num_blocks, block_size * KV_ENTRY_BYTES]`` uint8 — matches
    the per-block byte stride both kernels write to. The rtp-llm kernel
    needs a 3D view ``[num_blocks, block_size, 584]`` with stride(0)
    equal to the block byte stride; we slice/view as needed."""
    block_bytes = block_size * KV_ENTRY_BYTES
    pool_flat = torch.zeros(num_blocks, block_bytes, dtype=torch.uint8, device=device)
    return pool_flat


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overlap,ratio",
    [
        (False, 128),  # HCA
        (True, 4),  # CSA
    ],
)
@pytest.mark.parametrize("block_size", [16, 64])
def test_v4_compressor_kv_fused_matches_vllm_layout(overlap, ratio, block_size):
    """rtp-llm fused writer vs vLLM ``quantize_and_insert_k_cache``."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(0xC0DE)
    device = torch.device("cuda")
    head_dim = KV_HEAD_DIM
    rope_dim = KV_ROPE_DIM
    nope_dim = KV_NOPE_DIM
    norm_eps = 1e-6

    # 8 tokens, with one masked-out slot (-1) to exercise the skip branch.
    num_tokens = 8
    G = 2 * ratio if overlap else ratio
    D_in = 2 * head_dim if overlap else head_dim

    # Inputs the rtp-llm kernel consumes directly.
    kv_state = torch.randn(num_tokens, G, D_in, dtype=torch.float32, device=device)
    score_state = (
        torch.randn(num_tokens, G, D_in, dtype=torch.float32, device=device) * 0.5
    )
    norm_weight = (torch.randn(head_dim, device=device) * 0.1 + 1.0).to(torch.bfloat16)
    rope_cos = torch.randn(
        num_tokens, rope_dim // 2, dtype=torch.float32, device=device
    )
    rope_sin = torch.randn(
        num_tokens, rope_dim // 2, dtype=torch.float32, device=device
    )
    norm_weight = norm_weight.contiguous()

    # Slot mapping: spread tokens across multiple blocks; mask out token 3.
    # Use slot ids that span ≥2 blocks for both block_size choices.
    slot_mapping = torch.tensor(
        [0, 1, block_size, -1, block_size + 1, 2 * block_size, 5, block_size + 7],
        dtype=torch.int64,
        device=device,
    )
    assert slot_mapping.shape[0] == num_tokens
    num_blocks = int(slot_mapping.max().item()) // block_size + 2

    # ── rtp-llm path ──
    pool_rtp_flat = _allocate_pool(num_blocks, block_size, device=device)
    pool_rtp_3d = pool_rtp_flat.view(num_blocks, block_size, KV_ENTRY_BYTES)
    # Sanity: the 3D view must NOT alias outside the per-block byte stride.
    assert pool_rtp_3d.stride(0) == block_size * KV_ENTRY_BYTES
    v4_compressor_kv_fused(
        kv_state.contiguous(),
        score_state.contiguous(),
        slot_mapping,
        norm_weight,
        rope_cos,
        rope_sin,
        pool_rtp_3d,
        cache_block_stride_bytes=int(pool_rtp_3d.stride(0)),
        overlap=overlap,
        head_dim=head_dim,
        rope_head_dim=rope_dim,
        norm_eps=norm_eps,
    )

    # ── reference path: emulate the rtp-llm pool/RMSNorm/RoPE in pure
    #     PyTorch, then feed the resulting bf16 K to vLLM's writer ──
    if overlap:
        # OVERLAP: kv/score laid out as [B, 2r, 2d]; first ratio rows take
        # the lower-half of the d axis, the rest take the upper half.
        # Build the post-cat view explicitly.
        lower = kv_state[:, :ratio, :head_dim]  # [B, ratio, D]
        upper = kv_state[:, ratio:, head_dim:]  # [B, ratio, D]
        kv_post = torch.cat([lower, upper], dim=1)  # [B, 2r, D]
        s_lower = score_state[:, :ratio, :head_dim]
        s_upper = score_state[:, ratio:, head_dim:]
        score_post = torch.cat([s_lower, s_upper], dim=1)
    else:
        kv_post = kv_state  # [B, ratio, D]
        score_post = score_state

    K_ref = _ref_pool_rmsnorm_rope(
        kv_post,
        score_post,
        norm_weight,
        rope_cos,
        rope_sin,
        norm_eps=norm_eps,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
    )  # [B, head_dim] bf16

    pool_vllm_flat = _allocate_pool(num_blocks, block_size, device=device)
    quantize_and_insert_k_cache(
        K_ref,
        pool_vllm_flat,
        slot_mapping,
        block_size=block_size,
    )

    # ── byte-level diff ──
    # Two zones:
    #   (1) FP8 NoPE bytes + UE8M0 scales — must be byte-exact. Both
    #       kernels do the bf16 round-trip on post-RMSNorm fp32 then
    #       UE8M0-quant from identical inputs, so any diff is a real bug.
    #   (2) BF16 RoPE bytes — fp32-reduction order in RMSNorm differs
    #       (Triton tree-reduce vs torch.mean), giving up to 1-ULP bf16
    #       diffs (lower byte ±1). Tolerate ≤1 byte magnitude diff and
    #       only in the lower byte of a bf16 pair.
    diff_mask = pool_rtp_flat != pool_vllm_flat
    if not bool(diff_mask.any()):
        return  # exact match — best case

    diffs = diff_mask.nonzero(as_tuple=False)
    rtp_bytes = pool_rtp_flat[diff_mask].to(torch.int16)
    vllm_bytes = pool_vllm_flat[diff_mask].to(torch.int16)
    abs_byte_diff = (rtp_bytes - vllm_bytes).abs()

    bad = []
    for i in range(diffs.shape[0]):
        blk, off = int(diffs[i, 0]), int(diffs[i, 1])
        in_data = off < block_size * KV_TOKEN_DATA_SIZE
        if in_data:
            in_tok = off % KV_TOKEN_DATA_SIZE
            in_rope = in_tok >= KV_NOPE_DIM
            # bf16 element index within the rope segment; lower byte == even
            rope_byte = in_tok - KV_NOPE_DIM if in_rope else 0
            is_lower_byte = (rope_byte % 2) == 0
        else:
            in_rope = False
            is_lower_byte = False
        ok = in_rope and is_lower_byte and int(abs_byte_diff[i].item()) <= 1
        if not ok:
            tok = (off % (block_size * KV_TOKEN_DATA_SIZE)) // KV_TOKEN_DATA_SIZE
            in_tok = off % KV_TOKEN_DATA_SIZE
            region = (
                "scale"
                if not in_data
                else ("fp8_nope" if in_tok < KV_NOPE_DIM else "bf16_rope")
            )
            bad.append(
                (
                    blk,
                    off,
                    region,
                    tok,
                    in_tok,
                    int(rtp_bytes[i].item()),
                    int(vllm_bytes[i].item()),
                )
            )

    if bad:
        msg_lines = [f"intolerable byte diffs ({len(bad)} of {diffs.shape[0]} total):"]
        for blk, off, region, tok, in_tok, r, v in bad[:8]:
            msg_lines.append(
                f"  blk={blk} off={off} region={region} tok={tok} in_tok={in_tok} "
                f"rtp=0x{r:02x} vllm=0x{v:02x}"
            )
        raise AssertionError("\n".join(msg_lines))

    # All diffs are tolerable bf16 RoPE 1-ULP — also assert they're sparse.
    rope_bytes_total = num_tokens * (KV_ROPE_DIM)  # one byte per bf16 elem
    tol_ratio = diffs.shape[0] / max(1, rope_bytes_total)
    assert tol_ratio < 0.05, (
        f"too many tolerable diffs: {diffs.shape[0]} bytes "
        f"out of {rope_bytes_total} rope-low bytes ({tol_ratio:.3%})"
    )


if __name__ == "__main__":
    # Allow direct invocation for ad-hoc diagnosis.
    test_v4_compressor_kv_fused_matches_vllm_layout(False, 128, 64)
    test_v4_compressor_kv_fused_matches_vllm_layout(True, 4, 64)
    test_v4_compressor_kv_fused_matches_vllm_layout(False, 128, 16)
    test_v4_compressor_kv_fused_matches_vllm_layout(True, 4, 16)
    print("OK")
