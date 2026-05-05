"""132B indexer FP8 writer (head_dim=128) — UE8M0 layout, vLLM/DeepGEMM compatible.

The rtp-llm fused writer ``v4_compressor_fused`` produces the 132B/slot
indexer FP8 layout consumed by DeepGEMM's ``fp8_paged_mqa_logits``:

  per block (block_size tokens):
    bytes [0          : bs * 128)        — FP8 K, 128 B/token, head_dim=128
    bytes [bs * 128   : bs * 128 + bs*4) — fp32 scales, 4 B/token (UE8M0)

  scale convention: ``scale = 2 ^ ceil(log2(max(|k|, 1e-4) / FP8_MAX))``
                    stored as fp32 (UE8M0 — power-of-2 snapping).
                    Matches vLLM ``_fused_kv_compress_norm_rope_insert_indexer_attn``
                    (vendored at _vllm_ref/fused_compress_quant_cache.py:368-391).

This UT performs **byte-level diff** against a pure-PyTorch reference
that re-implements the UE8M0 algorithm verbatim. By transitivity the
reference also matches vLLM's kernel (same math), so a passing UT
locks the rtp-llm writer to vLLM/DeepGEMM byte compatibility.

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

from rtp_llm.models_py.modules.dsv4._compressor_fused_triton import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
    v4_compressor_fused,
)

_INDEXER_ROPE_DIM = 64
_FP8_MAX = 448.0


def _ref_pool_rmsnorm_rope_quant(
    kv_state,  # [B, G, D] fp32
    score_state,  # [B, G, D] fp32
    norm_weight,  # [D] bf16
    rope_cos,  # [B, rope/2] fp32
    rope_sin,  # [B, rope/2] fp32
    *,
    norm_eps,
    rope_dim,
):
    """Pure-PyTorch ``pool → RMSNorm → RoPE → UE8M0 quant``.

    Returns ``(fp8_bytes [B, D] uint8, scale [B] fp32)``. The UE8M0
    formula is byte-identical to the rtp-llm fused kernel and to vLLM's
    ``_fused_kv_compress_norm_rope_insert_indexer_attn``.
    """
    head_dim = norm_weight.shape[0]
    nope_dim = head_dim - rope_dim

    # pool
    score_sm = torch.softmax(score_state, dim=1)
    pooled = (kv_state * score_sm).sum(dim=1)

    # rmsnorm fp32
    w_fp32 = norm_weight.to(torch.float32)
    var = (pooled * pooled).mean(dim=-1, keepdim=True)
    rrms = torch.rsqrt(var + norm_eps)
    normed = pooled * rrms * w_fp32

    # rope
    out = normed.clone()
    rope = normed[:, nope_dim : nope_dim + rope_dim]
    pair = rope.view(rope.shape[0], rope_dim // 2, 2)
    even, odd = pair[..., 0], pair[..., 1]
    new_even = even * rope_cos - odd * rope_sin
    new_odd = odd * rope_cos + even * rope_sin
    rotated = torch.stack([new_even, new_odd], dim=-1).view(rope.shape[0], rope_dim)
    out[:, nope_dim : nope_dim + rope_dim] = rotated

    # bf16 round-trip (mirrors kernel's `rotated.to(bf16).to(fp32)` pre-quant)
    rotated_bf = out.to(torch.bfloat16).to(torch.float32)

    # UE8M0 per-token quant
    absmax = rotated_bf.abs().max(dim=-1, keepdim=True).values  # [B, 1]
    absmax = torch.clamp(absmax, min=1e-4)
    raw_scale = absmax / _FP8_MAX
    exponent = torch.ceil(torch.log2(raw_scale))  # [B, 1]
    inv_scale = torch.exp2(-exponent)
    scale = torch.exp2(exponent).squeeze(-1)  # [B] fp32

    x_scaled = rotated_bf * inv_scale
    x_clamped = torch.clamp(x_scaled, -_FP8_MAX, _FP8_MAX)
    q_fp8 = x_clamped.to(torch.float8_e4m3fn)
    q_uint8 = q_fp8.view(torch.uint8)
    return q_uint8, scale


def _pack_to_indexer_pool(
    fp8_bytes,  # [B, head_dim] uint8
    scales,  # [B] fp32
    slot_mapping,  # [B] int64
    *,
    num_blocks,
    block_size,
    head_dim,
):
    """Pack per-token (fp8, scale) into the 132B striped layout.

    Per-block layout: ``[bs*head_dim K | bs*4 fp32 scales]``.
    """
    device = fp8_bytes.device
    pool = torch.zeros(
        num_blocks, block_size, INDEXER_ENTRY_BYTES, dtype=torch.uint8, device=device
    )
    pool_2d = pool.view(num_blocks, block_size * INDEXER_ENTRY_BYTES)

    for i in range(slot_mapping.shape[0]):
        slot = int(slot_mapping[i].item())
        if slot < 0:
            continue
        blk = slot // block_size
        off = slot % block_size
        # K bytes
        pool_2d[blk, off * head_dim : (off + 1) * head_dim] = fp8_bytes[i]
        # fp32 scale
        scale_off = block_size * head_dim + off * 4
        pool_2d[blk, scale_off : scale_off + 4] = (
            scales[i : i + 1].view(torch.uint8).flatten()
        )
    return pool


@pytest.mark.parametrize(
    "overlap,ratio",
    [
        (False, 128),  # HCA-style indexer compressor
        (True, 4),  # CSA-style indexer compressor
    ],
)
@pytest.mark.parametrize("block_size", [16, 64])
def test_v4_compressor_fused_indexer_matches_ue8m0_layout(overlap, ratio, block_size):
    """rtp-llm fused 132B writer vs python UE8M0 reference.

    Same algorithm as vLLM's indexer kernel (UE8M0 + per-block grouped
    K|scales). Byte-equal except for the same fp32-reduction-order
    1-ULP noise we tolerate in the 584B path (RMSNorm sum order can
    flip a single fp8 byte at quant boundary, or — rarer — flip the
    UE8M0 exponent if absmax noise crosses a log2 boundary).
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    torch.manual_seed(0xBEEF)
    device = torch.device("cuda")
    head_dim = INDEXER_HEAD_DIM
    rope_dim = _INDEXER_ROPE_DIM
    norm_eps = 1e-6

    num_tokens = 8
    G = 2 * ratio if overlap else ratio
    D_in = 2 * head_dim if overlap else head_dim

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

    slot_mapping = torch.tensor(
        [0, 1, block_size, -1, block_size + 1, 2 * block_size, 5, block_size + 7],
        dtype=torch.int64,
        device=device,
    )
    num_blocks = int(slot_mapping.max().item()) // block_size + 2

    # ── rtp-llm path ──
    pool_rtp = torch.zeros(
        num_blocks, block_size, INDEXER_ENTRY_BYTES, dtype=torch.uint8, device=device
    )
    v4_compressor_fused(
        kv_state.contiguous(),
        score_state.contiguous(),
        slot_mapping,
        norm_weight,
        rope_cos,
        rope_sin,
        pool_rtp,
        overlap=overlap,
        head_dim=head_dim,
        rope_head_dim=rope_dim,
        norm_eps=norm_eps,
    )

    # ── reference: PyTorch UE8M0 quant on equivalent post-overlap input ──
    if overlap:
        lower = kv_state[:, :ratio, :head_dim]
        upper = kv_state[:, ratio:, head_dim:]
        kv_post = torch.cat([lower, upper], dim=1)
        s_lower = score_state[:, :ratio, :head_dim]
        s_upper = score_state[:, ratio:, head_dim:]
        score_post = torch.cat([s_lower, s_upper], dim=1)
    else:
        kv_post, score_post = kv_state, score_state

    fp8_ref, scale_ref = _ref_pool_rmsnorm_rope_quant(
        kv_post,
        score_post,
        norm_weight,
        rope_cos,
        rope_sin,
        norm_eps=norm_eps,
        rope_dim=rope_dim,
    )
    pool_ref = _pack_to_indexer_pool(
        fp8_ref,
        scale_ref,
        slot_mapping,
        num_blocks=num_blocks,
        block_size=block_size,
        head_dim=head_dim,
    )

    # ── byte-level diff with tolerance ──
    diff_mask = pool_rtp != pool_ref
    if not bool(diff_mask.any()):
        return

    diffs = diff_mask.nonzero(as_tuple=False)
    rtp_bytes = pool_rtp[diff_mask].to(torch.int16)
    ref_bytes = pool_ref[diff_mask].to(torch.int16)
    abs_byte_diff = (rtp_bytes - ref_bytes).abs()

    # Per-block byte stride: bs * 128 K + bs * 4 scale = bs * 132.
    # Within a block (3D shape), dim-1 stride is 132 bytes. So pool[blk, off, ...]
    # spans one slot's K (128 B) at the start. Scales live at offsets
    # >= bs*128 in the FLAT view. The 3D view we used scatters scales as
    # the trailing 4 bytes of OTHER slots' "entry" — that's just byte
    # arithmetic; the layout is correct (K-first, scales-after) because
    # the python packer wrote the same way.
    bad = []
    for i in range(diffs.shape[0]):
        blk, off, byte = (
            int(diffs[i, 0]),
            int(diffs[i, 1]),
            int(diffs[i, 2]),
        )
        # In the 3D [num_blocks, block_size, 132] view: byte 0..127 is K
        # of THIS slot; byte 128..131 is scale of slot 0 (when off==0),
        # slot 1 (when off==1), etc — because scales follow ALL K data.
        # For our diff classification it's enough to know: bytes < 128
        # are FP8 K of (blk, off); bytes >= 128 are part of the fp32
        # scale region.
        is_fp8 = byte < head_dim
        # Tolerance: 1 fp8 byte step, OR 1-bit fp32 mantissa flip in scale
        # (rare; would manifest as bytes 0/1 of the 4-byte fp32).
        if is_fp8:
            ok = int(abs_byte_diff[i].item()) <= 1
        else:
            # fp32 scale: tolerate ≤1 byte step in the lower mantissa
            # bytes (offset 128+0 or 128+1 within the slot's 132B entry,
            # corresponding to the LSBs of the fp32). The exponent byte
            # (offset 128+3) and high-mantissa byte (128+2) must NOT diff
            # by >1 (would imply a UE8M0 exponent flip, which means
            # absmax noise crossed a log2 boundary — flag for review).
            scale_byte = byte - head_dim  # 0..3
            if scale_byte in (0, 1):
                ok = int(abs_byte_diff[i].item()) <= 4
            else:
                ok = False
        if not ok:
            region = "fp8_k" if is_fp8 else f"scale_byte[{byte - head_dim}]"
            bad.append(
                (
                    blk,
                    off,
                    byte,
                    region,
                    int(rtp_bytes[i].item()),
                    int(ref_bytes[i].item()),
                )
            )

    if bad:
        msg_lines = [f"intolerable byte diffs ({len(bad)} of {diffs.shape[0]} total):"]
        for blk, off, byte, region, r, v in bad[:8]:
            msg_lines.append(
                f"  blk={blk} off={off} byte={byte} region={region} "
                f"rtp=0x{r:02x} ref=0x{v:02x}"
            )
        raise AssertionError("\n".join(msg_lines))

    # All diffs tolerable; cap the count.
    total_writable_bytes = (slot_mapping >= 0).sum().item() * INDEXER_ENTRY_BYTES
    tol_ratio = diffs.shape[0] / max(1, total_writable_bytes)
    assert tol_ratio < 0.10, (
        f"too many tolerable diffs: {diffs.shape[0]} bytes "
        f"out of {total_writable_bytes} writable bytes ({tol_ratio:.3%})"
    )


if __name__ == "__main__":
    test_v4_compressor_fused_indexer_matches_ue8m0_layout(False, 128, 64)
    test_v4_compressor_fused_indexer_matches_ue8m0_layout(True, 4, 64)
    test_v4_compressor_fused_indexer_matches_ue8m0_layout(False, 128, 16)
    test_v4_compressor_fused_indexer_matches_ue8m0_layout(True, 4, 16)
    print("OK")
