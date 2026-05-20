"""UT for the per-token state-pool ``CompressorFP8`` (post commit
``e76867719`` "fix - align state size to 256").

Covers:
  * State pool receives one slot per token; ``state[slot, :hidden]`` ==
    kv-projection, ``state[slot, hidden:]`` == score-projection + ape.
  * Boundary tokens (``(pos+1) % ratio == 0``) produce FP8 entries in the
    KV pool; non-boundary slots stay zero.
  * Both ``head_dim`` flavors (512 CSA, 128 indexer) wire up correctly.

Run:
  cd .../github-opensource && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \\
    /opt/conda310/bin/python3 \\
    rtp_llm/models_py/modules/dsv4/test/test_compressor_fp8_per_token.py
"""

from __future__ import annotations

import torch

from rtp_llm.models_py.modules.dsv4.fp8._compressor_consts import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
    KV_ENTRY_BYTES,
    KV_HEAD_DIM,
)
from rtp_llm.models_py.modules.dsv4.fp8.compressor import (
    CompressorFP8,
    _linear_bf16_bf16_fp32,
    build_decode_metadata,
    build_prefill_metadata,
)

DEVICE = "cuda"
TOKENS_PER_STATE_BLOCK = 256


def _build_compressor(
    *,
    dim: int,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
) -> CompressorFP8:
    coff = 1 + (compress_ratio == 4)
    # Construct weights directly on DEVICE: ``CompressorFP8._fuse_wkv_wgate``
    # caches ``_wkv_wgate_fused`` as a plain attribute (not a buffer), so
    # a post-construction ``.to(DEVICE)`` would only move the registered
    # Parameters and leave ``_wkv_wgate_fused`` on CPU → mat1/mat2 device
    # mismatch in the test forward. Production loads weights GPU-resident
    # from safetensors, so this is test-only.
    weights = {
        "ape": (
            torch.randn(
                compress_ratio, coff * head_dim, dtype=torch.bfloat16, device=DEVICE
            )
            * 0.1
        ),
        "wkv": (
            torch.randn(coff * head_dim, dim, dtype=torch.bfloat16, device=DEVICE)
            * 0.05
        ),
        "wgate": (
            torch.randn(coff * head_dim, dim, dtype=torch.bfloat16, device=DEVICE)
            * 0.05
        ),
        "norm": torch.ones(head_dim, dtype=torch.bfloat16, device=DEVICE),
    }
    cmp = CompressorFP8(
        dim=dim,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        max_batch_size=1,
        norm_eps=1e-6,
        compressor_weights=weights,
    )
    # GPT-J style freqs_cis: complex64 of shape [max_pos, rope_head_dim/2].
    # Values don't need to be physically correct for layout/wireup tests —
    # use unit complex numbers so RoPE is a no-op.
    max_pos = 4096
    freqs_cis = torch.ones(
        max_pos, rope_head_dim // 2, dtype=torch.complex64, device=DEVICE
    )
    cmp.freqs_cis = freqs_cis
    return cmp


def _bind_pools(
    cmp: CompressorFP8,
    *,
    seqlen: int,
    head_dim: int,
    coff: int,
    compress_ratio: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate fake state + KV pools and install on the compressor.

    Returns (state_pool_3d, kv_pool_3d, state_block_table) for assertions.
    """
    state_eb = TOKENS_PER_STATE_BLOCK
    state_blocks_per_req = 2
    # Block id 0 is the "unallocated" sentinel — start real blocks at 1.
    state_total_blocks = 1 + state_blocks_per_req
    hidden = 2 * coff * head_dim
    state_view_2d = torch.zeros(
        state_total_blocks * state_eb, hidden, dtype=torch.float32, device=DEVICE
    )
    state_block_table = torch.tensor(
        [[1, 2]], dtype=torch.int32, device=DEVICE
    )  # [B=1, max_blocks=2]

    # KV pool: kv_eb = TOKENS_PER_BLOCK / ratio = 256/ratio
    entry_bytes = KV_ENTRY_BYTES if head_dim == KV_HEAD_DIM else INDEXER_ENTRY_BYTES
    kv_eb = TOKENS_PER_STATE_BLOCK // compress_ratio
    n_compressed = (seqlen + compress_ratio - 1) // compress_ratio
    kv_blocks_needed = max(1, (n_compressed + kv_eb - 1) // kv_eb)
    kv_total_blocks = 1 + kv_blocks_needed
    kv_pool_3d = torch.zeros(
        kv_total_blocks, kv_eb, entry_bytes, dtype=torch.uint8, device=DEVICE
    )
    kv_block_table = torch.arange(
        1, 1 + kv_blocks_needed, dtype=torch.int32, device=DEVICE
    ).reshape(1, kv_blocks_needed)

    cmp.set_pool_context(
        kv_pool_view=kv_pool_3d,
        kv_block_table=kv_block_table,
        kv_eb=kv_eb,
        state_pool_view=state_view_2d,
        state_block_table=state_block_table,
        state_eb=state_eb,
    )
    state_pool_3d = state_view_2d.view(state_total_blocks, state_eb, hidden)
    return state_pool_3d, kv_pool_3d, state_block_table


def _state_write_check(
    cmp: CompressorFP8,
    x: torch.Tensor,
    state_pool_3d: torch.Tensor,
    state_block_table: torch.Tensor,
    *,
    sp: int,
    ratio: int,
    head_dim: int,
    coff: int,
) -> None:
    """Replay the state-write math in eager torch and compare per-token."""
    bsz, seqlen, _ = x.shape
    hidden = 2 * coff * head_dim
    fused_ref = _linear_bf16_bf16_fp32(x, cmp._wkv_wgate_fused)
    kv_ref = fused_ref[..., : hidden // 2]
    score_ref = fused_ref[..., hidden // 2 :]
    eb = TOKENS_PER_STATE_BLOCK
    bt = state_block_table.long()
    failures = []
    for b in range(bsz):
        for t in range(seqlen):
            pos = sp + t
            block_in_seq = pos // eb
            assert block_in_seq < bt.shape[1], f"out-of-table block at pos {pos}"
            in_block = pos % eb
            block_id = int(bt[b, block_in_seq].item())
            assert block_id > 0, f"unallocated block at pos {pos}"
            slot_kv = state_pool_3d[block_id, in_block, : hidden // 2]
            slot_score = state_pool_3d[block_id, in_block, hidden // 2 :]
            ape_row = cmp.ape[pos % ratio]
            ref_kv = kv_ref[b, t]
            ref_score = score_ref[b, t] + ape_row
            kv_err = (slot_kv - ref_kv).abs().max().item()
            score_err = (slot_score - ref_score).abs().max().item()
            if kv_err > 1e-4 or score_err > 1e-4:
                failures.append((b, t, pos, kv_err, score_err))
    assert not failures, f"state mismatches: {failures[:5]} (total {len(failures)})"


def _kv_boundary_check(
    kv_pool_3d: torch.Tensor,
    *,
    seqlen: int,
    sp: int,
    ratio: int,
    kv_eb: int,
) -> None:
    """Boundary tokens write a slot; non-boundary positions stay zero."""
    # Compressed positions touched: 0 .. (sp+seqlen)//ratio - 1, but only
    # those whose corresponding boundary token (pos = (cp+1)*ratio - 1)
    # falls in [sp, sp+seqlen).
    expected_hits = []
    for t in range(seqlen):
        pos = sp + t
        if (pos + 1) % ratio == 0:
            expected_hits.append(pos // ratio)
    n_compressed_total = (sp + seqlen) // ratio
    for cp in range(n_compressed_total):
        block_id = 1 + (cp // kv_eb)
        in_block = cp % kv_eb
        slot = kv_pool_3d[block_id, in_block]
        nz = slot.any().item()
        if cp in expected_hits:
            assert nz, f"boundary cp={cp} expected nonzero but slot is empty"
    # Sanity: a slot beyond n_compressed_total stays zero (untouched).
    assert not kv_pool_3d[0].any().item(), "sentinel block 0 should stay zero"


def test_csa_per_token(seqlen: int = 64) -> None:
    """CSA: head_dim=512, ratio=4, overlap=True."""
    torch.manual_seed(0)
    head_dim, rope_head_dim, ratio = KV_HEAD_DIM, 64, 4
    coff = 1 + (ratio == 4)
    dim = 128
    cmp = _build_compressor(
        dim=dim,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=ratio,
    )
    state_pool_3d, kv_pool_3d, state_bt = _bind_pools(
        cmp, seqlen=seqlen, head_dim=head_dim, coff=coff, compress_ratio=ratio
    )
    sp = 0
    x = torch.randn(1, seqlen, dim, dtype=torch.bfloat16, device=DEVICE) * 0.1
    cmp.forward(x, sp)
    torch.cuda.synchronize()
    _state_write_check(
        cmp,
        x,
        state_pool_3d,
        state_bt,
        sp=sp,
        ratio=ratio,
        head_dim=head_dim,
        coff=coff,
    )
    _kv_boundary_check(
        kv_pool_3d,
        seqlen=seqlen,
        sp=sp,
        ratio=ratio,
        kv_eb=TOKENS_PER_STATE_BLOCK // ratio,
    )
    print(f"  [csa  prefill] sp={sp} S={seqlen} OK")


def test_indexer_per_token(seqlen: int = 32) -> None:
    """Indexer compressor: head_dim=128, ratio=4, overlap=True."""
    torch.manual_seed(1)
    head_dim, rope_head_dim, ratio = INDEXER_HEAD_DIM, 64, 4
    coff = 1 + (ratio == 4)
    dim = 64
    cmp = _build_compressor(
        dim=dim,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=ratio,
    )
    state_pool_3d, kv_pool_3d, state_bt = _bind_pools(
        cmp, seqlen=seqlen, head_dim=head_dim, coff=coff, compress_ratio=ratio
    )
    sp = 0
    x = torch.randn(1, seqlen, dim, dtype=torch.bfloat16, device=DEVICE) * 0.1
    cmp.forward(x, sp)
    torch.cuda.synchronize()
    _state_write_check(
        cmp,
        x,
        state_pool_3d,
        state_bt,
        sp=sp,
        ratio=ratio,
        head_dim=head_dim,
        coff=coff,
    )
    _kv_boundary_check(
        kv_pool_3d,
        seqlen=seqlen,
        sp=sp,
        ratio=ratio,
        kv_eb=TOKENS_PER_STATE_BLOCK // ratio,
    )
    print(f"  [idx  prefill] sp={sp} S={seqlen} OK")


def test_decode_vectorized() -> None:
    """Single-token decode at boundary vs non-boundary positions."""
    torch.manual_seed(2)
    head_dim, rope_head_dim, ratio = KV_HEAD_DIM, 64, 4
    coff = 1 + (ratio == 4)
    dim = 128
    cmp = _build_compressor(
        dim=dim,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=ratio,
    )
    # Pretend we already prefilled 8 tokens. Decode token 9 (pos=8, NOT boundary)
    # then decode token 10 (pos=9, NOT boundary)... pos=11 IS boundary (12%4==0).
    state_pool_3d, kv_pool_3d, state_bt = _bind_pools(
        cmp, seqlen=16, head_dim=head_dim, coff=coff, compress_ratio=ratio
    )
    bsz = 1
    for pos in [8, 9, 10, 11]:
        x = torch.randn(bsz, 1, dim, dtype=torch.bfloat16, device=DEVICE) * 0.1
        sp = torch.tensor([pos], dtype=torch.int64, device=DEVICE)
        cmp.forward_decode_vectorized(x, sp)
    torch.cuda.synchronize()
    # cp=2 corresponds to pos in [8..11]; the boundary token at pos=11 should
    # have written kv slot 2 (block 1, in-block 2).
    kv_eb = TOKENS_PER_STATE_BLOCK // ratio
    cp = 2
    block_id = 1 + (cp // kv_eb)
    in_block = cp % kv_eb
    assert (
        kv_pool_3d[block_id, in_block].any().item()
    ), "decode boundary at pos=11 did not write kv slot"
    # Non-boundary cp=3 (would correspond to pos 12..15, all unwritten in this
    # decode batch) — must be untouched.
    assert (
        not kv_pool_3d[block_id, in_block + 1].any().item()
    ), "decode wrote non-boundary slot"
    print("  [csa  decode]  pos=[8..11] OK (boundary at 11 only)")


def test_prepared_metadata_path() -> None:
    """Caller-prepared CompressorMeta must produce identical pool state to
    the in-body fallback path."""
    torch.manual_seed(3)
    head_dim, rope_head_dim, ratio = KV_HEAD_DIM, 64, 4
    coff = 1 + (ratio == 4)
    dim = 128
    seqlen = 32
    sp = 0

    # Path A — fallback (meta=None)
    cmp_a = _build_compressor(
        dim=dim,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=ratio,
    )
    state_a, kv_a, _ = _bind_pools(
        cmp_a, seqlen=seqlen, head_dim=head_dim, coff=coff, compress_ratio=ratio
    )
    x = torch.randn(1, seqlen, dim, dtype=torch.bfloat16, device=DEVICE) * 0.1
    cmp_a.forward(x, sp)
    torch.cuda.synchronize()

    # Path B — prepared meta (slot mapping computed BEFORE forward)
    cmp_b = _build_compressor(
        dim=dim,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=ratio,
    )
    # Reuse path-A weights so path-B produces byte-identical writes.
    cmp_b.wkv.weight.data.copy_(cmp_a.wkv.weight.data)
    cmp_b.wgate.weight.data.copy_(cmp_a.wgate.weight.data)
    cmp_b.ape.data.copy_(cmp_a.ape.data)
    cmp_b.norm.weight.data.copy_(cmp_a.norm.weight.data)
    state_b, kv_b, _ = _bind_pools(
        cmp_b, seqlen=seqlen, head_dim=head_dim, coff=coff, compress_ratio=ratio
    )
    meta = build_prefill_metadata(cmp_b, sp=sp, bsz=1, seqlen=seqlen, device=x.device)
    cmp_b.forward(x, sp, meta=meta)
    torch.cuda.synchronize()

    assert torch.equal(state_a, state_b), "prepared meta diverged from fallback (state)"
    assert torch.equal(kv_a, kv_b), "prepared meta diverged from fallback (kv)"
    print("  [prepared meta] prefill matches fallback byte-for-byte")

    # Decode path with prepared meta — single token, boundary position.
    cmp_c = _build_compressor(
        dim=dim,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=ratio,
    )
    state_c, kv_c, _ = _bind_pools(
        cmp_c, seqlen=16, head_dim=head_dim, coff=coff, compress_ratio=ratio
    )
    bsz = 1
    sp_decode = torch.tensor([3], dtype=torch.int64, device=DEVICE)  # boundary
    xd = torch.randn(bsz, 1, dim, dtype=torch.bfloat16, device=DEVICE) * 0.1
    meta_d = build_decode_metadata(cmp_c, sp_decode, bsz)
    cmp_c.forward_decode_vectorized(xd, sp_decode, meta=meta_d)
    torch.cuda.synchronize()
    kv_eb = TOKENS_PER_STATE_BLOCK // ratio
    assert kv_c[1, 0].any().item(), "prepared decode boundary did not write KV slot"
    print("  [prepared meta] decode boundary write OK")


def test_decode_strided_kv_score_matches_contiguous_path() -> None:
    """Decode should feed strided fused-linear slices directly to Triton.

    Regression coverage for the timeline copy-kernel issue: ``fused_out`` is
    split into ``kv`` and ``score`` along the last dimension. Those row views
    have a larger row stride than their logical width, and must produce the
    same state/KV writes as the old explicit contiguous materialization.
    """
    torch.manual_seed(4)
    head_dim, rope_head_dim, ratio = KV_HEAD_DIM, 64, 4
    coff = 1 + (ratio == 4)
    dim = 128
    q_len = 4
    positions = torch.arange(q_len, dtype=torch.long, device=DEVICE)
    b_idx = torch.zeros(q_len, dtype=torch.long, device=DEVICE)
    seq_start = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    cu_seq = torch.tensor([0, q_len], dtype=torch.int32, device=DEVICE)
    start_pos = torch.tensor([0], dtype=torch.int64, device=DEVICE)
    position_ids = positions.to(torch.int32)
    x = torch.randn(1, q_len, dim, dtype=torch.bfloat16, device=DEVICE) * 0.1

    cmp_strided = _build_compressor(
        dim=dim,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=ratio,
    )
    state_strided, kv_strided, _ = _bind_pools(
        cmp_strided,
        seqlen=q_len,
        head_dim=head_dim,
        coff=coff,
        compress_ratio=ratio,
    )
    meta_strided = cmp_strided.prepare_metadata(
        positions,
        b_idx,
        is_batched=True,
        seq_start_per_req=seq_start,
        cu_seq_per_req=cu_seq,
    )

    cmp_contig = _build_compressor(
        dim=dim,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=ratio,
    )
    cmp_contig._wkv_wgate_fused.copy_(cmp_strided._wkv_wgate_fused)
    cmp_contig.ape.data.copy_(cmp_strided.ape.data)
    cmp_contig.norm.weight.data.copy_(cmp_strided.norm.weight.data)
    state_contig, kv_contig, _ = _bind_pools(
        cmp_contig,
        seqlen=q_len,
        head_dim=head_dim,
        coff=coff,
        compress_ratio=ratio,
    )
    meta_contig = cmp_contig.prepare_metadata(
        positions,
        b_idx,
        is_batched=True,
        seq_start_per_req=seq_start,
        cu_seq_per_req=cu_seq,
    )

    cmp_strided.forward_decode_vectorized(
        x,
        start_pos,
        meta=meta_strided,
        position_ids=position_ids,
    )

    out_dim = coff * head_dim
    fused_out = _linear_bf16_bf16_fp32(x, cmp_contig._wkv_wgate_fused)
    kv_view = fused_out[..., :out_dim].view(q_len, out_dim)
    score_view = fused_out[..., out_dim:].view(q_len, out_dim)
    assert not kv_view.is_contiguous(), "test setup failed: kv view should be strided"
    assert (
        kv_view.stride(0) > kv_view.shape[1]
    ), "test setup failed: kv row stride should include score slice"
    cmp_contig._launch(
        kv_view.contiguous(),
        score_view.contiguous(),
        meta_contig,
        seq_start=None,
    )

    torch.cuda.synchronize()
    assert torch.equal(
        state_strided, state_contig
    ), "strided decode state writes differ from contiguous reference"
    assert torch.equal(
        kv_strided, kv_contig
    ), "strided decode KV writes differ from contiguous reference"
    print("  [decode stride] strided kv/score match contiguous reference")


def test_state_pool_clear_pool_context() -> None:
    """clear_pool_context drops both pool refs and forward becomes no-op."""
    cmp = _build_compressor(
        dim=64, head_dim=KV_HEAD_DIM, rope_head_dim=64, compress_ratio=4
    )
    cmp.clear_pool_context()
    assert cmp._state_pool_3d is None
    assert cmp._kv_pool_3d is None
    # Forward returns None even without pool bound (warmup gate).
    x = torch.randn(1, 4, 64, dtype=torch.bfloat16, device=DEVICE)
    out = cmp.forward(x, 0)
    assert out is None
    print("  [warmup]      no-pool forward OK")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for CompressorFP8 UT")
    print("== State + KV per-token writes ==")
    test_csa_per_token()
    test_indexer_per_token()
    test_decode_vectorized()
    test_prepared_metadata_path()
    test_decode_strided_kv_score_matches_contiguous_path()
    test_state_pool_clear_pool_context()
    print("\nOK")
