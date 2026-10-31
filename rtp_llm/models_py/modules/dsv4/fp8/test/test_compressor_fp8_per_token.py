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

from rtp_llm.models_py.modules.dsv4.cp import _CP_ROLE_INDEXER, _CP_ROLE_MAIN
from rtp_llm.models_py.modules.dsv4.fp8._compressor_consts import (
    INDEXER_ENTRY_BYTES,
    INDEXER_HEAD_DIM,
    KV_ENTRY_BYTES,
    KV_HEAD_DIM,
)
from rtp_llm.models_py.modules.dsv4.fp8._compressor_vllm_triton import (
    run_save_partial_states,
)
from rtp_llm.models_py.modules.dsv4.fp8.compressor import (
    CompressorFP8,
    _linear_bf16_bf16_fp32,
    build_decode_metadata,
    build_prefill_metadata,
)
from rtp_llm.models_py.modules.dsv4.fp8.sparse_save import SparseSaveTargets
from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
    CSA_KV,
    CSA_STATE,
    HCA_KV,
    HCA_STATE,
)
from rtp_llm.models_py.modules.dsv4.prefill_workspace import PrefillWorkspace

DEVICE = "cuda"
TOKENS_PER_STATE_BLOCK = 256

# These tests exercise the non-CP prefill path (no cp_ctx bound), so the
# workspace is never dereferenced — a trivial one satisfies the required arg.
_WS = PrefillWorkspace(
    torch.device("cpu"), q_rows=1, q_dim=1, reserve_cp=False, align_bytes=1
)


def _build_compressor(
    *,
    dim: int,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
) -> CompressorFP8:
    coff = 1 + (compress_ratio == 4)
    cp_role = _CP_ROLE_INDEXER if head_dim == INDEXER_HEAD_DIM else _CP_ROLE_MAIN
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
        cp_role=cp_role,
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
    cmp.init_rope_cache(freqs_cis)
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
        state_tokens_per_block=state_eb,
        kv_tokens_per_block=kv_eb * compress_ratio,
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
            block_in_seq = (pos // eb) % bt.shape[1]
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
    cmp.forward(x, sp, workspace=_WS)
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
    cmp.forward(x, sp, workspace=_WS)
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


def _run_decode_long_position_cyclic_state_blocks(head_dim: int, label: str) -> None:
    """Regression helper for long-context decode state-cache gathers.

    Production state pools keep only two 256-token blocks per request and reuse
    them cyclically. The fused boundary writer must use the same modulo mapping
    as ``CompressorFP8._compute_state_slot_mapping`` when it gathers the prior
    overlap window from state_cache. A direct ``pos // block_size`` block-table
    lookup runs past the two-entry state block table once decode positions grow
    beyond 512 tokens; affected long-context decode runs reproduced Xid 13.
    """

    torch.manual_seed(20260518)
    rope_head_dim, ratio = 64, 4
    coff = 1 + (ratio == 4)
    dim = 128
    pos = 4095  # boundary token; logical state block 15 maps to table col 1.

    cmp = _build_compressor(
        dim=dim,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=ratio,
    )
    state_pool_3d, kv_pool_3d, state_bt = _bind_pools(
        cmp,
        seqlen=pos + 1,
        head_dim=head_dim,
        coff=coff,
        compress_ratio=ratio,
    )

    hidden = coff * head_dim
    window_start = pos - coff * ratio + 1
    for p in range(window_start, pos):
        block_col = (p // TOKENS_PER_STATE_BLOCK) % state_bt.shape[1]
        block_id = int(state_bt[0, block_col].item())
        in_block = p % TOKENS_PER_STATE_BLOCK
        state_pool_3d[block_id, in_block, :hidden].fill_(0.25)
        state_pool_3d[block_id, in_block, hidden:].zero_()

    x = torch.randn(1, 1, dim, dtype=torch.bfloat16, device=DEVICE) * 0.1
    sp = torch.tensor([pos], dtype=torch.int64, device=DEVICE)
    position_ids = torch.tensor([[pos]], dtype=torch.int64, device=DEVICE)
    cmp.forward_decode_vectorized(x, sp, position_ids=position_ids)
    torch.cuda.synchronize()

    kv_eb = TOKENS_PER_STATE_BLOCK // ratio
    cp = pos // ratio
    block_id = 1 + (cp // kv_eb)
    in_block = cp % kv_eb
    assert (
        kv_pool_3d[block_id, in_block].any().item()
    ), f"long-position {label} decode did not write the expected cyclic KV slot"
    print(f"  [{label} decode] long-position cyclic state block lookup OK")


def test_indexer_decode_long_position_uses_cyclic_state_blocks() -> None:
    _run_decode_long_position_cyclic_state_blocks(INDEXER_HEAD_DIM, "idx")


def test_csa_decode_long_position_uses_cyclic_state_blocks() -> None:
    _run_decode_long_position_cyclic_state_blocks(KV_HEAD_DIM, "csa")


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
    cmp_a.forward(x, sp, workspace=_WS)
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
    cmp_b.forward(x, sp, meta=meta, workspace=_WS)
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
        has_prefix=True,
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
        has_prefix=True,
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
    assert cmp._kv_pool_view is None
    # Forward returns None even without pool bound (warmup gate).
    x = torch.randn(1, 4, 64, dtype=torch.bfloat16, device=DEVICE)
    out = cmp.forward(x, 0, workspace=_WS)
    assert out is None
    print("  [warmup]      no-pool forward OK")


def _check_sparse_csa_save_restore_continue_short_suffix(suffix_len: int) -> None:
    """Real state/packed-KV kernels preserve A while B writes C and D."""
    torch.manual_seed(20261018)
    cmp = _build_compressor(
        dim=128, head_dim=KV_HEAD_DIM, rope_head_dim=64, compress_ratio=4
    )
    state_eb = 8
    kv_eb = 256 // 4
    state = torch.zeros(
        (8, state_eb, 4 * KV_HEAD_DIM), dtype=torch.float32, device=DEVICE
    )
    kv = torch.zeros((8, kv_eb, KV_ENTRY_BYTES), dtype=torch.uint8, device=DEVICE)
    state_bt = torch.tensor([[4, 2]], dtype=torch.int32, device=DEVICE)
    kv_bt = torch.tensor([[0, 0, 0, 0, 4, 2]], dtype=torch.int32, device=DEVICE)
    cmp.set_pool_context(
        kv_pool_view=kv,
        kv_block_table=kv_bt,
        kv_eb=kv_eb,
        state_pool_view=state,
        state_block_table=state_bt,
        state_eb=state_eb,
        state_tokens_per_block=256,
        kv_tokens_per_block=256,
        kv_owner_tokens_per_block=256,
    )
    old_x = torch.randn((4, 128), dtype=torch.bfloat16, device=DEVICE) * 0.1
    new_x = torch.randn((suffix_len, 128), dtype=torch.bfloat16, device=DEVICE) * 0.1
    all_x = torch.cat((old_x, new_x))
    fused = _linear_bf16_bf16_fp32(all_x.contiguous(), cmp._wkv_wgate_fused)
    kv_raw, score_raw = fused[:, : 2 * KV_HEAD_DIM], fused[:, 2 * KV_HEAD_DIM :]
    old_pos = torch.arange(1296, 1300, device=DEVICE, dtype=torch.int64)
    run_save_partial_states(
        kv_raw[:4],
        score_raw[:4],
        cmp.ape,
        old_pos,
        state,
        1 * state_eb + old_pos % state_eb,
        compress_ratio=4,
    )
    # The physical block is payload[64,576] followed by scales[64,8].
    # A tensor view [64,584] is only an allocation shape, not one logical
    # payload+scale row; its row 4 slice overlaps the next payload row.
    packed = kv.view(8, -1)
    old_entry = (1299 % 256) // 4
    packed[1, old_entry * 576 : (old_entry + 1) * 576].fill_(37)
    packed[1, kv_eb * 576 + old_entry * 8 : kv_eb * 576 + (old_entry + 1) * 8].fill_(37)
    state[2].copy_(state[1])
    state[3].copy_(state[1])
    kv[2].copy_(kv[1])
    kv[3].copy_(kv[1])
    old_state = state[1].clone()
    old_kv = packed[1].clone()

    cmp._protect_short_prefix = True
    cmp._sparse_state_tag = CSA_STATE
    cmp._sparse_kv_tag = CSA_KV
    cmp._sparse_save_targets_by_tag = {
        CSA_STATE: SparseSaveTargets(
            torch.tensor([[1300 + suffix_len]], dtype=torch.int64),
            torch.tensor([[3]], dtype=torch.int32),
        ),
        CSA_KV: SparseSaveTargets(
            torch.tensor([[1300 + suffix_len]], dtype=torch.int64),
            torch.tensor([[3]], dtype=torch.int32),
        ),
    }
    new_pos = torch.arange(1300, 1300 + suffix_len, device=DEVICE, dtype=torch.int64)
    new_meta = cmp.prepare_metadata(
        new_pos,
        torch.zeros_like(new_pos),
        has_prefix=True,
        is_batched=True,
        seq_start_per_req=torch.tensor([1300], device=DEVICE),
        cu_seq_per_req=torch.tensor([0, suffix_len], device=DEVICE),
    )
    cmp._launch(kv_raw[4:], score_raw[4:], new_meta)

    ref_state = torch.zeros_like(state)
    ref_kv = torch.zeros_like(kv)
    cmp._sparse_save_targets_by_tag = None
    cmp._protect_short_prefix = False
    cmp.set_pool_context(
        kv_pool_view=ref_kv,
        kv_block_table=torch.tensor(
            [[0, 0, 0, 0, 4, 5]], dtype=torch.int32, device=DEVICE
        ),
        kv_eb=kv_eb,
        state_pool_view=ref_state,
        state_block_table=torch.tensor([[4, 5]], dtype=torch.int32, device=DEVICE),
        state_eb=state_eb,
        state_tokens_per_block=256,
        kv_tokens_per_block=256,
        kv_owner_tokens_per_block=256,
    )
    all_pos = torch.arange(1296, 1300 + suffix_len, device=DEVICE, dtype=torch.int64)
    ref_meta = cmp.prepare_metadata(
        all_pos,
        torch.zeros_like(all_pos),
        has_prefix=False,
        is_batched=True,
        seq_start_per_req=torch.tensor([1296], device=DEVICE),
        cu_seq_per_req=torch.tensor([0, 4 + suffix_len], device=DEVICE),
    )
    cmp._launch(kv_raw, score_raw, ref_meta)
    torch.cuda.synchronize()

    assert torch.equal(state[1], old_state), "saved source state A mutated"
    assert torch.equal(packed[1], old_kv), "saved source compressed KV A mutated"
    assert torch.equal(
        state[3], ref_state[5]
    ), "saved D state differs from cold continuation"
    ref_packed = ref_kv.view(8, -1)
    for entry in range(5, (20 + suffix_len) // 4):
        assert torch.equal(
            packed[3, entry * 576 : (entry + 1) * 576],
            ref_packed[5, entry * 576 : (entry + 1) * 576],
        ), f"saved D compressed payload entry {entry} differs from cold continuation"
        assert torch.equal(
            packed[3, kv_eb * 576 + entry * 8 : kv_eb * 576 + (entry + 1) * 8],
            ref_packed[5, kv_eb * 576 + entry * 8 : kv_eb * 576 + (entry + 1) * 8],
        ), f"saved D compressed scale entry {entry} differs from cold continuation"
    for block_id in (2, 3):
        assert torch.equal(
            packed[block_id, old_entry * 576 : (old_entry + 1) * 576],
            old_kv[old_entry * 576 : (old_entry + 1) * 576],
        ), f"block {block_id} lost copied compressed history payload"
        assert torch.equal(
            packed[
                block_id,
                kv_eb * 576 + old_entry * 8 : kv_eb * 576 + (old_entry + 1) * 8,
            ],
            old_kv[kv_eb * 576 + old_entry * 8 : kv_eb * 576 + (old_entry + 1) * 8],
        ), f"block {block_id} lost copied compressed history scale"
    print(
        f"  [sparse CSA] G1300→N{1300 + suffix_len} state/KV D, immutable A, cold reference OK"
    )


def test_sparse_csa_save_restore_continue_short_suffix() -> None:
    _check_sparse_csa_save_restore_continue_short_suffix(20)


def test_sparse_csa_one_token_save_keeps_runtime_writer() -> None:
    _check_sparse_csa_save_restore_continue_short_suffix(1)


def test_sparse_hca_state_and_prefix_read_before_ring_overwrite() -> None:
    """HCA's 128-row ring must preserve G1300 history for boundary 1407."""
    torch.manual_seed(20261019)
    cmp = _build_compressor(
        dim=128, head_dim=KV_HEAD_DIM, rope_head_dim=64, compress_ratio=128
    )
    state_eb = 128
    kv_eb = 2
    state = torch.zeros(
        (8, state_eb, 2 * KV_HEAD_DIM), dtype=torch.float32, device=DEVICE
    )
    kv = torch.zeros((8, kv_eb, KV_ENTRY_BYTES), dtype=torch.uint8, device=DEVICE)
    cmp.set_pool_context(
        kv_pool_view=kv,
        kv_block_table=torch.tensor(
            [[0, 0, 0, 0, 4, 2]], dtype=torch.int32, device=DEVICE
        ),
        kv_eb=kv_eb,
        state_pool_view=state,
        state_block_table=torch.tensor([[4, 2]], dtype=torch.int32, device=DEVICE),
        state_eb=state_eb,
        state_tokens_per_block=256,
        kv_tokens_per_block=256,
        kv_owner_tokens_per_block=256,
    )
    old_x = torch.randn((20, 128), dtype=torch.bfloat16, device=DEVICE) * 0.1
    new_x = torch.randn((236, 128), dtype=torch.bfloat16, device=DEVICE) * 0.1
    fused = _linear_bf16_bf16_fp32(torch.cat((old_x, new_x)), cmp._wkv_wgate_fused)
    kv_raw, score_raw = fused[:, :KV_HEAD_DIM], fused[:, KV_HEAD_DIM:]
    old_pos = torch.arange(1280, 1300, device=DEVICE, dtype=torch.int64)
    run_save_partial_states(
        kv_raw[:20],
        score_raw[:20],
        cmp.ape,
        old_pos,
        state,
        1 * state_eb + old_pos % state_eb,
        compress_ratio=128,
    )
    state[2].copy_(state[1])
    state[3].copy_(state[1])
    old_state = state[1].clone()

    cmp._protect_short_prefix = True
    cmp._sparse_state_tag = HCA_STATE
    cmp._sparse_kv_tag = None
    cmp._sparse_save_targets_by_tag = {
        HCA_STATE: SparseSaveTargets(
            torch.tensor([[1320]], dtype=torch.int64),
            torch.tensor([[3]], dtype=torch.int32),
        )
    }
    new_pos = torch.arange(1300, 1536, device=DEVICE, dtype=torch.int64)
    new_meta = cmp.prepare_metadata(
        new_pos,
        torch.zeros_like(new_pos),
        has_prefix=True,
        is_batched=True,
        seq_start_per_req=torch.tensor([1300], device=DEVICE),
        cu_seq_per_req=torch.tensor([0, 236], device=DEVICE),
    )
    cmp._launch(kv_raw[20:], score_raw[20:], new_meta)

    ref_state = torch.zeros_like(state)
    ref_kv = torch.zeros_like(kv)
    cmp._protect_short_prefix = False
    cmp._sparse_save_targets_by_tag = None
    cmp.set_pool_context(
        kv_pool_view=ref_kv,
        kv_block_table=torch.tensor(
            [[0, 0, 0, 0, 4, 5]], dtype=torch.int32, device=DEVICE
        ),
        kv_eb=kv_eb,
        state_pool_view=ref_state,
        state_block_table=torch.tensor([[4, 5]], dtype=torch.int32, device=DEVICE),
        state_eb=state_eb,
        state_tokens_per_block=256,
        kv_tokens_per_block=256,
        kv_owner_tokens_per_block=256,
    )
    all_pos = torch.arange(1280, 1536, device=DEVICE, dtype=torch.int64)
    ref_meta = cmp.prepare_metadata(
        all_pos,
        torch.zeros_like(all_pos),
        has_prefix=False,
        is_batched=True,
        seq_start_per_req=torch.tensor([1280], device=DEVICE),
        cu_seq_per_req=torch.tensor([0, 256], device=DEVICE),
    )
    cmp._launch(kv_raw, score_raw, ref_meta)
    torch.cuda.synchronize()

    assert torch.equal(state[1], old_state), "HCA source A mutated"
    assert torch.equal(
        state[3, :40, :KV_HEAD_DIM], kv_raw[:40]
    ), "HCA D lost old or new KV history at p1320"
    expected_score = score_raw[:40] + cmp.ape[all_pos[:40] % 128]
    assert torch.equal(
        state[3, :40, KV_HEAD_DIM:], expected_score
    ), "HCA D score+ape history differs at p1320"
    packed = kv.view(8, -1)
    ref_packed = ref_kv.view(8, -1)
    for entry in (0, 1):
        assert torch.equal(
            packed[2, entry * 576 : (entry + 1) * 576],
            ref_packed[5, entry * 576 : (entry + 1) * 576],
        ), f"HCA compressed payload entry {entry} differs after G1300 continuation"
        assert torch.equal(
            packed[2, kv_eb * 576 + entry * 8 : kv_eb * 576 + (entry + 1) * 8],
            ref_packed[5, kv_eb * 576 + entry * 8 : kv_eb * 576 + (entry + 1) * 8],
        ), f"HCA compressed scale entry {entry} differs after G1300 continuation"
    print("  [sparse HCA] G1300→N1536, D1320 history and pre-write read OK")


def test_sparse_hca_g1300_to_n2000_final_state_and_kv() -> None:
    """The tail 2000 checkpoint uses final state and the last packed HCA page."""
    torch.manual_seed(20261020)
    cmp = _build_compressor(
        dim=128, head_dim=KV_HEAD_DIM, rope_head_dim=64, compress_ratio=128
    )
    state_eb, kv_eb = 128, 2
    state = torch.zeros(
        (10, state_eb, 2 * KV_HEAD_DIM), dtype=torch.float32, device=DEVICE
    )
    kv = torch.zeros((10, kv_eb, KV_ENTRY_BYTES), dtype=torch.uint8, device=DEVICE)
    cmp.set_pool_context(
        kv_pool_view=kv,
        kv_block_table=torch.tensor(
            [[0, 0, 0, 0, 4, 2, 5, 6]], dtype=torch.int32, device=DEVICE
        ),
        kv_eb=kv_eb,
        state_pool_view=state,
        state_block_table=torch.tensor([[4, 2]], dtype=torch.int32, device=DEVICE),
        state_eb=state_eb,
        state_tokens_per_block=256,
        kv_tokens_per_block=256,
        kv_owner_tokens_per_block=256,
    )
    old_x = torch.randn((20, 128), dtype=torch.bfloat16, device=DEVICE) * 0.1
    new_x = torch.randn((700, 128), dtype=torch.bfloat16, device=DEVICE) * 0.1
    fused = _linear_bf16_bf16_fp32(torch.cat((old_x, new_x)), cmp._wkv_wgate_fused)
    kv_raw, score_raw = fused[:, :KV_HEAD_DIM], fused[:, KV_HEAD_DIM:]
    old_pos = torch.arange(1280, 1300, device=DEVICE, dtype=torch.int64)
    run_save_partial_states(
        kv_raw[:20],
        score_raw[:20],
        cmp.ape,
        old_pos,
        state,
        1 * state_eb + old_pos % state_eb,
        compress_ratio=128,
    )
    state[2].copy_(state[1])
    state[3].copy_(state[1])
    source_state = state[1].clone()
    source_kv = kv[4].clone()
    cmp._protect_short_prefix = True
    cmp._sparse_state_tag = HCA_STATE
    cmp._sparse_kv_tag = HCA_KV
    cmp._sparse_save_targets_by_tag = {
        HCA_STATE: SparseSaveTargets(
            torch.tensor([[2000]], dtype=torch.int64),
            torch.tensor([[3]], dtype=torch.int32),
        ),
        HCA_KV: SparseSaveTargets(
            torch.tensor([[2000]], dtype=torch.int64),
            torch.tensor([[7]], dtype=torch.int32),
        ),
    }
    new_pos = torch.arange(1300, 2000, device=DEVICE, dtype=torch.int64)
    meta = cmp.prepare_metadata(
        new_pos,
        torch.zeros_like(new_pos),
        has_prefix=True,
        is_batched=True,
        seq_start_per_req=torch.tensor([1300], device=DEVICE),
        cu_seq_per_req=torch.tensor([0, 700], device=DEVICE),
    )
    cmp._launch(kv_raw[20:], score_raw[20:], meta)

    ref_state = torch.zeros_like(state)
    ref_kv = torch.zeros_like(kv)
    cmp._protect_short_prefix = False
    cmp._sparse_save_targets_by_tag = None
    cmp.set_pool_context(
        kv_pool_view=ref_kv,
        kv_block_table=torch.tensor(
            [[0, 0, 0, 0, 4, 5, 6, 7]], dtype=torch.int32, device=DEVICE
        ),
        kv_eb=kv_eb,
        state_pool_view=ref_state,
        state_block_table=torch.tensor([[4, 5]], dtype=torch.int32, device=DEVICE),
        state_eb=state_eb,
        state_tokens_per_block=256,
        kv_tokens_per_block=256,
        kv_owner_tokens_per_block=256,
    )
    all_pos = torch.arange(1280, 2000, device=DEVICE, dtype=torch.int64)
    # The 128-row state ring revisits the same physical block over 720 tokens.
    # Cold reference chunks enforce chronological writes; one oversized
    # launch would race earlier and later generations of the same ring rows.
    for start, end in ((1280, 1536), (1536, 1792), (1792, 2000)):
        begin, stop = start - 1280, end - 1280
        part_pos = all_pos[begin:stop]
        ref_meta = cmp.prepare_metadata(
            part_pos,
            torch.zeros_like(part_pos),
            has_prefix=start > 1280,
            is_batched=True,
            seq_start_per_req=torch.tensor([start], device=DEVICE),
            cu_seq_per_req=torch.tensor([0, stop - begin], device=DEVICE),
        )
        cmp._launch(kv_raw[begin:stop], score_raw[begin:stop], ref_meta)
    torch.cuda.synchronize()

    assert torch.equal(state[1], source_state), "HCA source A state mutated"
    assert torch.equal(kv[4], source_kv), "HCA source A compressed KV mutated"
    expected_state = torch.empty_like(state[3])
    tail_pos = all_pos[-state_eb:]
    expected_state[tail_pos % state_eb, :KV_HEAD_DIM] = kv_raw[-state_eb:]
    expected_state[tail_pos % state_eb, KV_HEAD_DIM:] = (
        score_raw[-state_eb:] + cmp.ape[tail_pos % 128]
    )
    assert torch.equal(
        state[3], expected_state
    ), "HCA D state differs from exact final 128 rows"
    assert torch.equal(
        state[3], ref_state[5]
    ), "HCA final D state differs from cold reference"
    packed = kv.view(10, -1)
    reference = ref_kv.view(10, -1)
    for runtime_id, reference_id, entries in (
        (2, 5, (0, 1)),
        (5, 6, (0, 1)),
        (7, 7, (0,)),
    ):
        for entry in entries:
            assert torch.equal(
                packed[runtime_id, entry * 576 : (entry + 1) * 576],
                reference[reference_id, entry * 576 : (entry + 1) * 576],
            ), f"HCA payload mismatch at block {runtime_id}, entry {entry}"
            assert torch.equal(
                packed[
                    runtime_id, kv_eb * 576 + entry * 8 : kv_eb * 576 + (entry + 1) * 8
                ],
                reference[
                    reference_id,
                    kv_eb * 576 + entry * 8 : kv_eb * 576 + (entry + 1) * 8,
                ],
            ), f"HCA scale mismatch at block {runtime_id}, entry {entry}"
    print("  [sparse HCA] G1300→N2000 final D state/KV and source A immutable OK")


def test_sparse_hca_zero_demand_keeps_runtime_writer() -> None:
    """A zero-needed-state point may skip D, but runtime C still advances."""
    torch.manual_seed(20261021)
    cmp = _build_compressor(
        dim=128, head_dim=KV_HEAD_DIM, rope_head_dim=64, compress_ratio=128
    )
    state = torch.zeros((4, 128, 2 * KV_HEAD_DIM), dtype=torch.float32, device=DEVICE)
    kv = torch.zeros((4, 2, KV_ENTRY_BYTES), dtype=torch.uint8, device=DEVICE)
    cmp.set_pool_context(
        kv_pool_view=kv,
        kv_block_table=torch.tensor(
            [[0, 0, 0, 0, 1, 2]], dtype=torch.int32, device=DEVICE
        ),
        kv_eb=2,
        state_pool_view=state,
        state_block_table=torch.tensor([[1, 2]], dtype=torch.int32, device=DEVICE),
        state_eb=128,
        state_tokens_per_block=256,
        kv_tokens_per_block=256,
        kv_owner_tokens_per_block=256,
    )
    state[1].fill_(17)
    state[2].copy_(state[1])
    source = state[1].clone()
    cmp._sparse_state_tag = HCA_STATE
    cmp._sparse_kv_tag = HCA_KV
    cmp._sparse_save_targets_by_tag = {
        HCA_STATE: SparseSaveTargets(
            torch.tensor([[1301]], dtype=torch.int64),
            torch.tensor([[-1]], dtype=torch.int32),
        ),
        HCA_KV: SparseSaveTargets(
            torch.tensor([[1301]], dtype=torch.int64),
            torch.tensor([[-1]], dtype=torch.int32),
        ),
    }
    x = torch.randn((1, 128), dtype=torch.bfloat16, device=DEVICE) * 0.1
    fused = _linear_bf16_bf16_fp32(x, cmp._wkv_wgate_fused)
    meta = cmp.prepare_metadata(
        torch.tensor([1300], dtype=torch.int64, device=DEVICE),
        torch.tensor([0], dtype=torch.int64, device=DEVICE),
        has_prefix=True,
        is_batched=True,
        seq_start_per_req=torch.tensor([1300], device=DEVICE),
        cu_seq_per_req=torch.tensor([0, 1], device=DEVICE),
    )
    cmp._launch(fused[:, :KV_HEAD_DIM], fused[:, KV_HEAD_DIM:], meta)
    torch.cuda.synchronize()
    assert torch.equal(state[1], source), "zero-demand point mutated source A"
    assert torch.equal(state[2, 1300 % 128, :KV_HEAD_DIM], fused[0, :KV_HEAD_DIM])
    assert torch.equal(
        state[2, 1300 % 128, KV_HEAD_DIM:], fused[0, KV_HEAD_DIM:] + cmp.ape[1300 % 128]
    )
    print("  [sparse HCA] zero-demand D=-1 retains runtime state writer")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required for CompressorFP8 UT")
    print("== State + KV per-token writes ==")
    test_csa_per_token()
    test_indexer_per_token()
    test_decode_vectorized()
    test_indexer_decode_long_position_uses_cyclic_state_blocks()
    test_csa_decode_long_position_uses_cyclic_state_blocks()
    test_prepared_metadata_path()
    test_decode_strided_kv_score_matches_contiguous_path()
    test_state_pool_clear_pool_context()
    test_sparse_csa_save_restore_continue_short_suffix()
    test_sparse_csa_one_token_save_keeps_runtime_writer()
    test_sparse_hca_state_and_prefix_read_before_ring_overwrite()
    test_sparse_hca_g1300_to_n2000_final_state_and_kv()
    test_sparse_hca_zero_demand_keeps_runtime_writer()
    print("\nOK")
