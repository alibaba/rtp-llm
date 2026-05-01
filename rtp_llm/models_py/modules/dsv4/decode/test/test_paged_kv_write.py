"""Phase 1 sanity test for the paged decode KV-write path.

Verifies that:
  * ``PoolDescriptor`` derives the right entries_per_block / vec_dim /
    bytes_per_entry for each attn_type, given the empirical layout from
    ``POOL_LAYOUT.md``.
  * ``compute_kv_pool_slot_mapping`` produces ``block_id*E + offset``
    correctly, including ``-1`` skip propagation.
  * ``write_kv_to_pool`` (mask_negative=False & True) lands the same
    bytes in the BlockPool that a register_buffer-style write produces
    when read back through ``PoolDescriptor.view()``.

Run as a plain pytest (no bazel needed):
    pytest rtp_llm/models_py/modules/dsv4/decode/test/test_paged_kv_write.py
"""

from __future__ import annotations

import torch

from rtp_llm.models_py.modules.dsv4.decode.kv_write_decode_op import write_kv_to_pool
from rtp_llm.models_py.modules.dsv4.decode.pool_layout import (
    CSA_KV,
    CSA_STATE,
    HCA_KV,
    HCA_STATE,
    INDEXER_KV,
    INDEXER_STATE,
    SWA_KV,
    PoolDescriptor,
    build_pool_descriptor,
)
from rtp_llm.models_py.modules.dsv4.decode.pool_slot_mapping import (
    compute_kv_pool_slot_mapping,
    compute_state_pool_slot_mapping,
)

HEAD_DIM = 512
IDX_HEAD_DIM = 128

# Empirical (entries_per_block, bytes_per_entry, vec_dim, dtype) per attn_type
# Source: POOL_LAYOUT.md (probed at runtime 2026-05-01).
EXPECTED = {
    SWA_KV: (256, 1024, HEAD_DIM, torch.bfloat16),
    CSA_KV: (64, 1024, HEAD_DIM, torch.bfloat16),
    HCA_KV: (2, 1024, HEAD_DIM, torch.bfloat16),
    INDEXER_KV: (64, 256, IDX_HEAD_DIM, torch.bfloat16),
    CSA_STATE: (1, 8192, 2 * 2 * HEAD_DIM, torch.float32),
    HCA_STATE: (2, 4096, 2 * HEAD_DIM, torch.float32),
    INDEXER_STATE: (1, 2048, 2 * 2 * IDX_HEAD_DIM, torch.float32),
}


def _make_pool(attn_type: int, num_blocks: int = 16) -> torch.Tensor:
    eb, bpe, _, _ = EXPECTED[attn_type]
    return torch.zeros(num_blocks, eb * bpe, dtype=torch.uint8)


def test_pool_descriptor_geometry():
    for at, (exp_eb, exp_bpe, exp_dim, exp_dtype) in EXPECTED.items():
        pool = _make_pool(at)
        coff = 2 if at in (CSA_STATE, INDEXER_STATE, CSA_KV, INDEXER_KV) else 1
        desc = build_pool_descriptor(
            pool,
            at,
            head_dim=HEAD_DIM,
            indexer_head_dim=IDX_HEAD_DIM,
            coff=coff,
        )
        assert desc is not None, f"pool {at} unexpectedly None"
        assert desc.entries_per_block == exp_eb, (at, desc.entries_per_block, exp_eb)
        assert desc.bytes_per_entry == exp_bpe, (at, desc.bytes_per_entry, exp_bpe)
        assert desc.vec_dim == exp_dim, (at, desc.vec_dim, exp_dim)
        assert desc.vec_dtype == exp_dtype, (at, desc.vec_dtype, exp_dtype)


def test_pool_descriptor_view_round_trip():
    """Writing through .view() must preserve bytes when read back."""
    for at in (SWA_KV, CSA_KV, HCA_KV, INDEXER_KV, CSA_STATE, HCA_STATE):
        coff = 2 if at in (CSA_STATE, CSA_KV, INDEXER_KV) else 1
        pool = _make_pool(at, num_blocks=8)
        desc = build_pool_descriptor(pool, at, HEAD_DIM, IDX_HEAD_DIM, coff)
        assert desc is not None
        v = desc.view()
        assert v.shape == (desc.num_blocks * desc.entries_per_block, desc.vec_dim)
        # Write through the view, read back through a fresh view.
        scribble = torch.arange(v.numel(), dtype=desc.vec_dtype).view_as(v)
        v.copy_(scribble)
        v2 = desc.view()
        assert torch.equal(v2, scribble), f"round-trip view mismatch for attn_type={at}"


def test_compute_kv_pool_slot_mapping_basic():
    # B=3, max_blocks=4, entries_per_block=64 (CSA-K layout)
    block_table = torch.tensor(
        [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]],
        dtype=torch.int32,
    )
    # CSA writes only on (pos+1)%4==0 boundaries; -1 marks skip.
    abs_pos = torch.tensor([0, 63, 64, -1, 65, 200], dtype=torch.int32)
    # B=3, q_len=2 → req_idx=[0,0,1,1,2,2]
    out = compute_kv_pool_slot_mapping(
        block_table,
        abs_pos,
        entries_per_block=64,
    )
    # req 0 token 0: block_table[0, 0]=10 * 64 + 0 = 640
    # req 0 token 1: block_table[0, 0]=10 * 64 + 63 = 703
    # req 1 token 0: block_table[1, 1]=21 * 64 + 0 = 1344
    # req 1 token 1: skip → -1
    # req 2 token 0: block_table[2, 1]=31 * 64 + 1 = 1985
    # req 2 token 1: block_table[2, 3]=33 * 64 + 8 = 2120
    expected = torch.tensor([640, 703, 1344, -1, 1985, 2120], dtype=torch.long)
    assert torch.equal(out, expected), (out, expected)


def test_compute_state_pool_slot_mapping():
    block_table = torch.tensor(
        [[100, 101, 102, 103, 104, 105, 106, 107]],
        dtype=torch.int32,
    )
    slot = torch.tensor([0, 3, -1, 7], dtype=torch.int32)
    out = compute_state_pool_slot_mapping(block_table, slot)
    expected = torch.tensor([100, 103, -1, 107], dtype=torch.long)
    assert torch.equal(out, expected), (out, expected)


def test_write_kv_to_pool_unconditional_swa():
    """Unconditional SWA write: every slot valid."""
    pool = _make_pool(SWA_KV, num_blocks=4)
    desc = build_pool_descriptor(pool, SWA_KV, HEAD_DIM, IDX_HEAD_DIM, coff=1)
    assert desc is not None

    # 3 tokens with hand-chosen slots in 3 different blocks.
    k = torch.randn(3, HEAD_DIM, dtype=torch.bfloat16)
    slots = torch.tensor([0 * 256 + 5, 1 * 256 + 100, 3 * 256 + 255], dtype=torch.long)
    write_kv_to_pool(k, slots, desc.view(), mask_negative=False)

    v = desc.view()
    for i, s in enumerate(slots.tolist()):
        assert torch.equal(v[s], k[i]), f"slot {s} mismatch"


def test_write_kv_to_pool_skip_negatives_csa():
    """Compressed-K write: -1 entries are no-ops (slot 0 unaffected)."""
    pool = _make_pool(CSA_KV, num_blocks=4)
    desc = build_pool_descriptor(pool, CSA_KV, HEAD_DIM, IDX_HEAD_DIM, coff=2)
    assert desc is not None
    v = desc.view()
    # Pre-fill slot 0 with a known sentinel — it must stay untouched after
    # the safe-redirect of -1 entries to slot 0.
    sentinel = torch.full((HEAD_DIM,), 0.5, dtype=torch.bfloat16)
    v[0] = sentinel

    k = torch.randn(4, HEAD_DIM, dtype=torch.bfloat16)
    # Two valid writes (slots 100, 255), two skips.
    slots = torch.tensor([100, -1, 255, -1], dtype=torch.long)
    write_kv_to_pool(k, slots, desc.view(), mask_negative=True)

    v2 = desc.view()
    assert torch.allclose(v2[100], k[0]), "slot 100 valid write missing"
    assert torch.allclose(v2[255], k[2]), "slot 255 valid write missing"
    assert torch.allclose(
        v2[0], sentinel, atol=1e-3
    ), f"sentinel at slot 0 corrupted by -1 redirect: {v2[0][:8]} vs {sentinel[:8]}"


def test_write_kv_to_pool_state_pool():
    """STATE pool: same write path, fp32 + 1 entry/block layout."""
    # State pool must be sized to cover the highest block_id we'll address.
    pool = _make_pool(CSA_STATE, num_blocks=200)
    desc = build_pool_descriptor(pool, CSA_STATE, HEAD_DIM, IDX_HEAD_DIM, coff=2)
    assert desc is not None
    # 1 entry/block → slot == block_id.
    block_table = torch.arange(8, dtype=torch.int32).view(1, 8) + 100
    # Write to compressor slots [0, 3, 5, 7].
    slot_in_compressor = torch.tensor([0, 3, 5, 7], dtype=torch.int32)
    pool_slots = compute_state_pool_slot_mapping(block_table, slot_in_compressor)
    assert pool_slots.tolist() == [100, 103, 105, 107]

    state = torch.randn(4, desc.vec_dim, dtype=torch.float32)
    write_kv_to_pool(state, pool_slots, desc.view(), mask_negative=False)
    v = desc.view()
    for i, s in enumerate(pool_slots.tolist()):
        assert torch.allclose(v[s], state[i]), f"state slot {s} mismatch"


if __name__ == "__main__":
    test_pool_descriptor_geometry()
    test_pool_descriptor_view_round_trip()
    test_compute_kv_pool_slot_mapping_basic()
    test_compute_state_pool_slot_mapping()
    test_write_kv_to_pool_unconditional_swa()
    test_write_kv_to_pool_skip_negatives_csa()
    test_write_kv_to_pool_state_pool()
    print("OK — all 7 tests passed")
