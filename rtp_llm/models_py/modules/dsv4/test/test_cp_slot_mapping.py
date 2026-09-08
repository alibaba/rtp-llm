"""Stage 5b-2 — CP-aware slot_mapping builder UT.

Covers ``_cp_slot_mapping`` primitives used on the writer (compressor) side
under ``kv_cache_sharded``:

  * ``cp_global_block_to_local`` — global logical block → local physical
    block id; non-owned positions get ``owned_mask=False``.
  * ``cp_kv_slot_mapping`` — boundary tokens of OWNED blocks get a real
    slot; everything else gets ``-1``.
  * ``cp_state_slot_mapping`` — intra-block fixed-pool slice ownership for
    INDEXER_STATE / CSA_STATE / HCA_STATE.

Pure CPU. Hand-rolled reference in Python loops.
"""

import importlib.util
import sys
import types
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[5]


def _import_mod():
    spec = importlib.util.spec_from_file_location(
        "_dsv4_cp_slot_for_test_5b2",
        _REPO_ROOT / "rtp_llm/models_py/modules/dsv4/fp8/_cp_slot_mapping.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _import_mod()


def _make_block_table_local(
    per_req_total_kv, tokens_per_block, cp_size, cp_rank, base_id=1
):
    """Synthesize a per-rank block_table where local block l of req b is
    assigned a unique non-zero physical id (so block_id==0 means
    unallocated), matching the write-side allocator semantics.
    """
    virtual_block_size = tokens_per_block * cp_size
    B = len(per_req_total_kv)
    local_per_req = []
    for T in per_req_total_kv:
        n_vb = (T + virtual_block_size - 1) // virtual_block_size
        local_per_req.append(n_vb)
    local_max = max(local_per_req) if local_per_req else 0
    bt = torch.zeros((B, local_max), dtype=torch.int64)
    pid = base_id
    for b, n in enumerate(local_per_req):
        for l in range(n):
            bt[b, l] = pid
            pid += 1
    return bt


def test_global_block_to_local_owner_formula():
    cp_size, cp_rank = 4, 1
    tokens_per_block = 4
    per_req = [16]
    bt = _make_block_table_local(per_req, tokens_per_block, cp_size, cp_rank)
    positions = torch.arange(16, dtype=torch.int64)
    b_idx = torch.zeros(16, dtype=torch.int64)
    block_id, owned = M.cp_global_block_to_local(
        positions, bt, b_idx, tokens_per_block, cp_size, cp_rank
    )
    # Owner of token t is (t // 4) % 4. cp_rank=1 owns blocks 1,5,9,...
    expected_owned = ((positions // tokens_per_block) % cp_size) == cp_rank
    assert torch.equal(owned, expected_owned)
    # For owned tokens (block 1 → tokens 4..7), local_blk=0, block_id=bt[0,0]
    assert int(block_id[4]) == int(bt[0, 0])


def test_kv_slot_mapping_boundary_and_owned():
    cp_size, cp_rank = 2, 0
    tokens_per_block = 8
    ratio = 4
    kv_eb = tokens_per_block // ratio  # 2
    per_req = [16]
    bt = _make_block_table_local(per_req, tokens_per_block, cp_size, cp_rank)
    positions = torch.arange(16, dtype=torch.int64)
    b_idx = torch.zeros(16, dtype=torch.int64)
    slot = M.cp_kv_slot_mapping(
        positions, bt, b_idx, tokens_per_block, kv_eb, ratio, cp_size, cp_rank
    )
    # Owned blocks: 0 (tokens 0..7). Non-owned: block 1 (tokens 8..15).
    # Boundaries: pos in {3,7,11,15}. Owned boundaries: 3,7. Slots:
    #   pos=3 → block_id=bt[0,0], in_block_compressed=(3%8)//4=0 → bt[0,0]*2+0
    #   pos=7 → bt[0,0]*2+1
    expected = torch.full((16,), -1, dtype=torch.int64)
    expected[3] = int(bt[0, 0]) * kv_eb + 0
    expected[7] = int(bt[0, 0]) * kv_eb + 1
    assert torch.equal(slot, expected), f"got {slot.tolist()}, want {expected.tolist()}"


def test_kv_slot_mapping_other_rank():
    # Same setup as above but cp_rank=1: owned tokens are 8..15.
    cp_size, cp_rank = 2, 1
    tokens_per_block = 8
    ratio = 4
    kv_eb = tokens_per_block // ratio
    per_req = [16]
    bt = _make_block_table_local(per_req, tokens_per_block, cp_size, cp_rank)
    positions = torch.arange(16, dtype=torch.int64)
    b_idx = torch.zeros(16, dtype=torch.int64)
    slot = M.cp_kv_slot_mapping(
        positions, bt, b_idx, tokens_per_block, kv_eb, ratio, cp_size, cp_rank
    )
    expected = torch.full((16,), -1, dtype=torch.int64)
    expected[11] = int(bt[0, 0]) * kv_eb + 0
    expected[15] = int(bt[0, 0]) * kv_eb + 1
    assert torch.equal(slot, expected)


def test_kv_slot_mapping_physical_owner_with_kernel_blocks_rank0():
    cp_size, cp_rank = 2, 0
    tokens_per_block = 128
    owner_tokens_per_block = 256
    ratio = 64
    kv_eb = tokens_per_block // ratio
    bt = torch.tensor([[10, 11]], dtype=torch.int64)
    positions = torch.arange(512, dtype=torch.int64)
    b_idx = torch.zeros(512, dtype=torch.int64)

    slot = M.cp_kv_slot_mapping(
        positions,
        bt,
        b_idx,
        tokens_per_block,
        kv_eb,
        ratio,
        cp_size,
        cp_rank,
        owner_tokens_per_block=owner_tokens_per_block,
    )

    expected = torch.full((512,), -1, dtype=torch.int64)
    # Rank0 owns physical block0, so both kernel rows inside tokens 0..255
    # must be written locally. Physical block1 (tokens 256..511) is rank1.
    expected[63] = 10 * kv_eb + 0
    expected[127] = 10 * kv_eb + 1
    expected[191] = 11 * kv_eb + 0
    expected[255] = 11 * kv_eb + 1
    assert torch.equal(slot, expected), f"got {slot.tolist()}, want {expected.tolist()}"


def test_kv_slot_mapping_physical_owner_with_kernel_blocks_rank1():
    cp_size, cp_rank = 2, 1
    tokens_per_block = 128
    owner_tokens_per_block = 256
    ratio = 64
    kv_eb = tokens_per_block // ratio
    bt = torch.tensor([[20, 21]], dtype=torch.int64)
    positions = torch.arange(512, dtype=torch.int64)
    b_idx = torch.zeros(512, dtype=torch.int64)

    slot = M.cp_kv_slot_mapping(
        positions,
        bt,
        b_idx,
        tokens_per_block,
        kv_eb,
        ratio,
        cp_size,
        cp_rank,
        owner_tokens_per_block=owner_tokens_per_block,
    )

    expected = torch.full((512,), -1, dtype=torch.int64)
    # Rank1 owns physical block1, represented by its two local kernel rows.
    expected[319] = 20 * kv_eb + 0
    expected[383] = 20 * kv_eb + 1
    expected[447] = 21 * kv_eb + 0
    expected[511] = 21 * kv_eb + 1
    assert torch.equal(slot, expected), f"got {slot.tolist()}, want {expected.tolist()}"


def test_state_slot_mapping_intrablock_slice():
    cp_size, cp_rank = 2, 0
    local_eb = 4
    tokens_per_block = 16
    bt = torch.tensor([[5, 6]], dtype=torch.int64)
    positions = torch.arange(16, dtype=torch.int64)
    b_idx = torch.zeros(16, dtype=torch.int64)
    seq_end = torch.tensor([16], dtype=torch.int64)
    slot = M.cp_state_slot_mapping(
        positions,
        bt,
        b_idx,
        local_eb,
        tokens_per_block,
        cp_size,
        cp_rank,
        seq_end,
    )
    expected = torch.full((16,), -1, dtype=torch.int64)
    # full ring = 8. Rank 0 owns logical offsets 0..3 in each ring.
    # Ring mask keeps the last full-ring positions before the block end.
    expected[8:12] = 5 * local_eb + torch.arange(4)
    assert torch.equal(slot, expected), f"got {slot.tolist()}, want {expected.tolist()}"


def test_state_slot_mapping_other_rank():
    cp_size, cp_rank = 2, 1
    local_eb = 4
    tokens_per_block = 16
    bt = torch.tensor([[5, 6]], dtype=torch.int64)
    positions = torch.arange(16, dtype=torch.int64)
    b_idx = torch.zeros(16, dtype=torch.int64)
    seq_end = torch.tensor([16], dtype=torch.int64)
    slot = M.cp_state_slot_mapping(
        positions,
        bt,
        b_idx,
        local_eb,
        tokens_per_block,
        cp_size,
        cp_rank,
        seq_end,
    )
    expected = torch.full((16,), -1, dtype=torch.int64)
    expected[12:16] = 5 * local_eb + torch.arange(4)
    assert torch.equal(slot, expected)


def test_cp_size_one_passthrough_kv():
    cp_size, cp_rank = 1, 0
    tokens_per_block = 8
    ratio = 4
    kv_eb = 2
    per_req = [8]
    bt = _make_block_table_local(per_req, tokens_per_block, cp_size, cp_rank)
    positions = torch.arange(8, dtype=torch.int64)
    b_idx = torch.zeros(8, dtype=torch.int64)
    slot = M.cp_kv_slot_mapping(
        positions, bt, b_idx, tokens_per_block, kv_eb, ratio, cp_size, cp_rank
    )
    # cp_size=1 → everything owned. Boundaries at 3,7.
    expected = torch.full((8,), -1, dtype=torch.int64)
    expected[3] = int(bt[0, 0]) * kv_eb + 0
    expected[7] = int(bt[0, 0]) * kv_eb + 1
    assert torch.equal(slot, expected)


def test_unallocated_block_zero_yields_minus_one():
    cp_size, cp_rank = 2, 0
    tokens_per_block = 8
    ratio = 4
    kv_eb = 2
    # Synthesize block_table with explicit zero (unallocated)
    bt = torch.zeros((1, 1), dtype=torch.int64)  # block_id=0 → unallocated
    positions = torch.arange(8, dtype=torch.int64)
    b_idx = torch.zeros(8, dtype=torch.int64)
    slot = M.cp_kv_slot_mapping(
        positions, bt, b_idx, tokens_per_block, kv_eb, ratio, cp_size, cp_rank
    )
    assert torch.equal(slot, torch.full((8,), -1, dtype=torch.int64))


def test_multi_request_kv():
    cp_size, cp_rank = 2, 0
    tokens_per_block = 4
    ratio = 2
    kv_eb = tokens_per_block // ratio  # 2
    per_req = [8, 4]
    bt = _make_block_table_local(per_req, tokens_per_block, cp_size, cp_rank)
    positions = torch.cat([torch.arange(8), torch.arange(4)]).to(torch.int64)
    b_idx = torch.cat(
        [torch.zeros(8, dtype=torch.int64), torch.ones(4, dtype=torch.int64)]
    )
    slot = M.cp_kv_slot_mapping(
        positions, bt, b_idx, tokens_per_block, kv_eb, ratio, cp_size, cp_rank
    )
    # req0: owned blocks 0 (t=0..3). Boundaries at t=1,3 → real slots.
    # req0 block 1 (t=4..7) non-owned → -1.
    # req1: owned block 0 (t=0..3). Boundaries at t=1,3.
    expected = torch.full((12,), -1, dtype=torch.int64)
    expected[1] = int(bt[0, 0]) * kv_eb + 0
    expected[3] = int(bt[0, 0]) * kv_eb + 1
    expected[8 + 1] = int(bt[1, 0]) * kv_eb + 0
    expected[8 + 3] = int(bt[1, 0]) * kv_eb + 1
    assert torch.equal(slot, expected)


def test_cp_global_block_to_local_out_of_capacity_is_masked():
    cp_size, cp_rank = 1, 0
    tokens_per_block = 4
    bt = torch.zeros((1, 0), dtype=torch.int64)
    positions = torch.arange(6, dtype=torch.int64)
    b_idx = torch.zeros(6, dtype=torch.int64)
    block_id, owned = M.cp_global_block_to_local(
        positions, bt, b_idx, tokens_per_block, cp_size, cp_rank
    )
    assert torch.equal(block_id, torch.zeros_like(positions))
    assert torch.equal(owned, torch.zeros_like(positions, dtype=torch.bool))


def test_kv_slot_mapping_zero_eb_or_ratio():
    positions = torch.arange(4, dtype=torch.int64)
    bt = torch.ones((1, 1), dtype=torch.int64)
    b_idx = torch.zeros(4, dtype=torch.int64)
    out = M.cp_kv_slot_mapping(positions, bt, b_idx, 4, 0, 4, 2, 0)
    assert torch.equal(out, torch.full((4,), -1, dtype=torch.int64))
    out = M.cp_kv_slot_mapping(positions, bt, b_idx, 4, 1, 0, 2, 0)
    assert torch.equal(out, torch.full((4,), -1, dtype=torch.int64))


def test_state_slot_mapping_zero_eb():
    positions = torch.arange(4, dtype=torch.int64)
    bt = torch.ones((1, 1), dtype=torch.int64)
    b_idx = torch.zeros(4, dtype=torch.int64)
    out = M.cp_state_slot_mapping(positions, bt, b_idx, 0, 4, 2, 0)
    assert torch.equal(out, torch.full((4,), -1, dtype=torch.int64))


if __name__ == "__main__":
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok  {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {exc!r}")
    sys.exit(1 if failures else 0)
