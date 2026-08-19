"""Stage 5b — CP-sharded paged-pool block gather UTs.

Two layers of testing:

1. ``cp_interleave_gathered_pool_blocks`` — pure tensor reorder, fully
   tested on CPU here. This is the bit that maps NCCL all_gather's
   rank-major output into logical block order.

2. ``cp_gather_request_pool_blocks`` and its batched counterpart — wrap NCCL
   ``all_gather`` + the
   interleave. We don't spin up a real distributed group here; instead we
   monkey-patch the ``all_gather`` symbol cp.py imported, then verify the
   per-rank packing + interleave + post-trim behavior end-to-end.

The interleave + ownership formula are the parts that, if wrong, silently
read garbage from the wrong physical block. CPU-only coverage is enough
to catch arithmetic bugs; the live NCCL path is exercised by Stage 6 PD
smoke under real CP=2/4 deployment.
"""

import importlib.util
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[5]


def _stub_distributed():
    """Stub ``rtp_llm.models_py.distributed.collective_torch`` so cp.py imports
    cleanly without dragging in real collective bindings.
    """
    mod_name = "rtp_llm.models_py.distributed.collective_torch"
    if mod_name in sys.modules:
        return
    stub = types.ModuleType(mod_name)

    class _Group:
        TP = "TP"
        DP = "DP"

    def _stub_all_gather(local, group=None):
        # Real impl replaced per-test; this default just round-trips so
        # accidental import-time references don't crash.
        return local

    stub.Group = _Group
    stub.all_gather = _stub_all_gather
    sys.modules[mod_name] = stub
    profiler = types.ModuleType("rtp_llm.models_py.modules.dsv4._profiler")
    profiler.record_function_range = lambda _name: nullcontext()
    sys.modules[profiler.__name__] = profiler
    # Also stub the parent packages if not present.
    for p in (
        "rtp_llm",
        "rtp_llm.models_py",
        "rtp_llm.models_py.distributed",
    ):
        if p not in sys.modules:
            sys.modules[p] = types.ModuleType(p)


def _import_cp():
    _stub_distributed()
    spec = importlib.util.spec_from_file_location(
        "_dsv4_cp_for_test",
        _REPO_ROOT / "rtp_llm/models_py/modules/dsv4/cp.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CP = _import_cp()


# ---------------------------------------------------------------------------
# Layer 1 — pure interleave reorder
# ---------------------------------------------------------------------------
def test_interleave_basic_cp2_even():
    # cp_size=2, L=3 local blocks per rank, each block is a 2D slot of [4, 8]
    # rank 0 contributes blocks for logical positions 0, 2, 4
    # rank 1 contributes blocks for logical positions 1, 3, 5
    # Use the logical block id as the block's value (broadcast) so we can
    # verify ownership wiring directly.
    cp_size = 2
    L = 3
    block_shape = (4, 8)
    rank0 = torch.stack(
        [torch.full(block_shape, fill_value=i * 2 + 0) for i in range(L)]
    )
    rank1 = torch.stack(
        [torch.full(block_shape, fill_value=i * 2 + 1) for i in range(L)]
    )
    gathered = torch.cat([rank0, rank1], dim=0)  # [cp_size*L, 4, 8] rank-major

    out = CP.cp_interleave_gathered_pool_blocks(
        gathered, cp_size, total_logical_blocks=cp_size * L
    )
    assert out.shape == (cp_size * L, *block_shape)
    for b in range(cp_size * L):
        assert torch.all(
            out[b] == b
        ), f"block {b} mismatch: got {out[b].flatten()[0].item()}"


def test_interleave_cp4():
    cp_size = 4
    L = 2
    block_shape = (3,)
    # Build per-rank contributions where rank r block i has logical id (r + i*cp_size).
    per_rank = []
    for r in range(cp_size):
        per_rank.append(
            torch.stack(
                [torch.full(block_shape, fill_value=r + i * cp_size) for i in range(L)]
            )
        )
    gathered = torch.cat(per_rank, dim=0)
    out = CP.cp_interleave_gathered_pool_blocks(
        gathered, cp_size, total_logical_blocks=cp_size * L
    )
    expected = torch.stack(
        [torch.full(block_shape, fill_value=b) for b in range(cp_size * L)]
    )
    assert torch.equal(out, expected)


def test_interleave_total_not_multiple_of_cp_size():
    """When total_logical_blocks isn't divisible by cp_size, trailing rows are dropped."""
    cp_size = 4
    L = 2  # each rank padded to L=2; total blocks = 8 padded, but actual = 5
    block_shape = (2,)
    per_rank = []
    for r in range(cp_size):
        per_rank.append(
            torch.stack(
                [torch.full(block_shape, fill_value=r + i * cp_size) for i in range(L)]
            )
        )
    gathered = torch.cat(per_rank, dim=0)
    total = 5  # only logical 0..4 are real
    out = CP.cp_interleave_gathered_pool_blocks(
        gathered, cp_size, total_logical_blocks=total
    )
    assert out.shape == (total, *block_shape)
    # Verify ownership formula: out[b] sourced from rank (b % cp_size) local (b // cp_size)
    for b in range(total):
        assert torch.all(out[b] == b)


def test_interleave_single_rank_passthrough():
    """cp_size=1 should be identity (modulo trim)."""
    cp_size = 1
    L = 5
    block_shape = (7,)
    gathered = torch.stack([torch.full(block_shape, fill_value=i) for i in range(L)])
    out = CP.cp_interleave_gathered_pool_blocks(
        gathered, cp_size, total_logical_blocks=L
    )
    assert torch.equal(out, gathered)


def test_interleave_rejects_size_mismatch():
    gathered = torch.zeros(7, 3)
    try:
        CP.cp_interleave_gathered_pool_blocks(
            gathered, cp_size=2, total_logical_blocks=7
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for size not divisible by cp_size")


def test_interleave_rejects_overshoot():
    gathered = torch.zeros(8, 3)
    try:
        CP.cp_interleave_gathered_pool_blocks(
            gathered, cp_size=2, total_logical_blocks=10
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for total_logical_blocks > gathered rows")


# ---------------------------------------------------------------------------
# Layer 2 — gather wrapper, NCCL stubbed
# ---------------------------------------------------------------------------
def _patch_all_gather(per_rank_outputs):
    """Replace cp.py's ``all_gather`` symbol with a stub that returns the
    pre-built gathered tensor. ``per_rank_outputs`` is a list of per-rank
    [L, *block_shape] tensors; the stub concatenates them rank-major.
    """
    stacked = torch.cat(per_rank_outputs, dim=0)

    def fake_all_gather(local, group=None):
        # The real all_gather returns rank-major concatenation of ``local``
        # across all ranks. We don't see ``local`` here because the test
        # already pre-built the global view — but verify shape consistency.
        L = per_rank_outputs[0].shape[0]
        assert local.shape[0] == L, f"local rows {local.shape[0]} != L {L}"
        return stacked

    CP.all_gather = fake_all_gather
    return stacked


def test_gather_request_pool_blocks_cp2():
    cp_size = 2
    cp_rank = 0  # this rank
    block_size = 4
    slot_bytes = 8
    # Construct a fake local pool with 16 blocks. Mark each block with its
    # physical block id so we can confirm index_select packs the right ones.
    num_pool_blocks = 16
    local_pool = torch.zeros(num_pool_blocks, block_size, slot_bytes, dtype=torch.uint8)
    for blk in range(num_pool_blocks):
        local_pool[blk] = blk

    # Request 1: 4 logical blocks total. Owner rule: logical b -> rank b%2.
    # Rank 0 owns logical {0, 2}, mapped to physical pool blocks {3, 9} (arbitrary).
    # Rank 1 owns logical {1, 3}, mapped to physical pool blocks {5, 12}.
    rank0_local_bt = torch.tensor([3, 9], dtype=torch.int32)
    # Stub all_gather output: the rank0 contribution is local_pool[[3,9]],
    # rank1 contribution we build manually using a "rank1 fake pool" with
    # values offset so we can distinguish.
    rank0_packed = local_pool.index_select(0, rank0_local_bt.long())  # [2, 4, 8]
    rank1_packed = torch.full(
        (2, block_size, slot_bytes), fill_value=200, dtype=torch.uint8
    )
    rank1_packed[0] = 201  # logical block 1
    rank1_packed[1] = 203  # logical block 3
    _patch_all_gather([rank0_packed, rank1_packed])

    out = CP.cp_gather_request_pool_blocks(
        local_pool=local_pool,
        local_block_table_for_req=rank0_local_bt,
        cp_size=cp_size,
        cp_rank=cp_rank,
        total_logical_blocks=4,
    )
    assert out.shape == (4, block_size, slot_bytes)
    # Logical block 0 = rank0's local block 0 = physical 3 = filled with 3.
    assert torch.all(out[0] == 3)
    # Logical block 1 = rank1's local block 0 = filled with 201.
    assert torch.all(out[1] == 201)
    # Logical block 2 = rank0's local block 1 = physical 9 = filled with 9.
    assert torch.all(out[2] == 9)
    # Logical block 3 = rank1's local block 1 = filled with 203.
    assert torch.all(out[3] == 203)


def test_gather_request_pool_blocks_cp4_padded():
    cp_size = 4
    cp_rank = 1
    block_size = 2
    slot_bytes = 4
    num_pool_blocks = 8
    local_pool = torch.zeros(num_pool_blocks, block_size, slot_bytes, dtype=torch.uint8)
    for blk in range(num_pool_blocks):
        local_pool[blk] = blk

    # 5 logical blocks, padded to L=2 per rank (8 - 3 padded slots dropped).
    # Owner of logical 0 = rank 0, 1 = rank 1, 2 = rank 2, 3 = rank 3, 4 = rank 0.
    # Rank 1 owns logical {1}, mapped to physical 7. Pad slot is 4 (any).
    rank_bts = [
        torch.tensor([2, 6], dtype=torch.int32),  # rank 0
        torch.tensor([7, 4], dtype=torch.int32),  # rank 1 (this rank)
        torch.tensor([5, 4], dtype=torch.int32),  # rank 2
        torch.tensor([1, 4], dtype=torch.int32),  # rank 3
    ]
    per_rank_packed = [local_pool.index_select(0, t.long()) for t in rank_bts]
    _patch_all_gather(per_rank_packed)

    out = CP.cp_gather_request_pool_blocks(
        local_pool=local_pool,
        local_block_table_for_req=rank_bts[cp_rank],
        cp_size=cp_size,
        cp_rank=cp_rank,
        total_logical_blocks=5,
    )
    assert out.shape == (5, block_size, slot_bytes)
    # Check ownership wiring across all 5 logical blocks.
    assert torch.all(out[0] == 2)  # rank 0 local 0 -> physical 2
    assert torch.all(out[1] == 7)  # rank 1 local 0 -> physical 7
    assert torch.all(out[2] == 5)  # rank 2 local 0 -> physical 5
    assert torch.all(out[3] == 1)  # rank 3 local 0 -> physical 1
    assert torch.all(out[4] == 6)  # rank 0 local 1 -> physical 6


def test_gather_request_rejects_bad_local_size():
    cp_size = 2
    block_size = 2
    slot_bytes = 2
    local_pool = torch.zeros(4, block_size, slot_bytes)
    # Total 6 logical blocks => expected L = 3 per rank; pass L=2 (wrong).
    bad_bt = torch.tensor([0, 1], dtype=torch.int32)
    try:
        CP.cp_gather_request_pool_blocks(
            local_pool=local_pool,
            local_block_table_for_req=bad_bt,
            cp_size=cp_size,
            cp_rank=0,
            total_logical_blocks=6,
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for L mismatch")


def test_gather_batched_request_pool_blocks_cp4_ragged():
    cp_size = 4
    cp_rank = 2
    block_shape = (2,)
    logical_per_req = (5, 0, 8, 3)
    # ceil(P_b / 4) => [2, 0, 2, 1], packed contribution S=5 per rank.
    rank_tables = [
        torch.tensor([[1, 2], [0, 0], [3, 4], [5, 0]], dtype=torch.int32),
        torch.tensor([[6, 7], [0, 0], [8, 9], [10, 0]], dtype=torch.int32),
        torch.tensor([[11, 12], [0, 0], [13, 14], [15, 0]], dtype=torch.int32),
        torch.tensor([[16, 17], [0, 0], [18, 19], [20, 0]], dtype=torch.int32),
    ]
    local_pool = torch.stack(
        [torch.full(block_shape, i, dtype=torch.int32) for i in range(21)]
    )
    per_rank_packed = []
    for table in rank_tables:
        ids = torch.cat([table[0, :2], table[2, :2], table[3, :1]]).long()
        per_rank_packed.append(local_pool.index_select(0, ids))
    _patch_all_gather(per_rank_packed)

    out = CP.cp_gather_batched_request_pool_blocks(
        local_pool,
        rank_tables[cp_rank],
        logical_per_req,
        cp_size,
        cp_rank,
    )
    # Independent oracle: concatenate each request's logical page values.
    expected_ids = []
    for req, logical_pages in enumerate(logical_per_req):
        for page in range(logical_pages):
            owner = page % cp_size
            col = page // cp_size
            expected_ids.append(rank_tables[owner][req, col].item())
    expected = torch.stack([local_pool[i] for i in expected_ids])
    assert torch.equal(out, expected)


def test_gather_batched_request_pool_blocks_uses_one_collective():
    calls = 0
    local_pool = torch.arange(12, dtype=torch.float32).view(6, 2)
    table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)

    def fake_all_gather(local, group=None):
        nonlocal calls
        calls += 1
        return torch.cat([local, local], dim=0)

    CP.all_gather = fake_all_gather
    plan = CP.build_cp_batched_pool_gather_plan(table, (3, 4), cp_size=2, cp_rank=0)
    CP.cp_gather_batched_request_pool_blocks(
        local_pool,
        table,
        (3, 4),
        cp_size=2,
        cp_rank=0,
        plan=plan,
    )
    assert calls == 1


def test_gather_batched_plan_can_return_rank_major_for_fused_restore():
    local_pool = torch.arange(12, dtype=torch.float32).view(6, 2)
    table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    plan = CP.build_cp_batched_pool_gather_plan(table, (3, 4), cp_size=2, cp_rank=0)
    local_owned = local_pool.index_select(0, plan.packed_block_ids)
    rank_major = torch.cat([local_owned, local_owned + 100], dim=0)
    CP.all_gather = lambda local, group=None: rank_major

    raw = CP.cp_gather_batched_request_pool_blocks(
        local_pool,
        table,
        (3, 4),
        cp_size=2,
        cp_rank=0,
        plan=plan,
        restore_logical_order=False,
    )
    ordered = CP.cp_gather_batched_request_pool_blocks(
        local_pool,
        table,
        (3, 4),
        cp_size=2,
        cp_rank=0,
        plan=plan,
    )
    assert torch.equal(raw, rank_major)
    assert torch.equal(ordered, rank_major.index_select(0, plan.restore_indices))


if __name__ == "__main__":
    # Allow `python test_cp_pool_block_gather.py` for quick local debugging.
    import inspect

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
