"""Phase E prep: paged-read round-trip byte-equal tests.

The Phase B dual-write tests prove the paged WRITE path produces
byte-identical pool state to the deleted ``_scatter_kv_pool`` /
``_scatter_state_pool`` helpers. Phase E will delete the
``register_buffer`` mirrors and route production READS through the
pool via ``PoolDescriptor.view()`` + block_table + slot_mapping.

This test exercises the full write → read-back round trip: populate
a pool via the Phase-B paged writer, then read back via the typed
``pool.view()`` using the same slot_mapping, and confirm the bytes
match the source buffer exactly.

If this passes, the Phase E read-path swap becomes:
  BEFORE: ``kv_cat = self.kv_cache[:bsz]``  (reads register_buffer)
  AFTER : ``kv_cat = gather_from_pool(pool_view, block_table, bsz)``
         (reads same bytes directly from the pool we already wrote to)

The gather shape ``[bsz, T, head_dim]`` is reconstructed from the flat
pool view via an ``index_select`` keyed by the same slot_mapping
the writer used — the operations are strict inverses when
``mask_negative=True`` writes are honored (sentinel slots leave the
destination untouched).

Bypasses ``rtp_llm`` package init (pulls C++ ops linked against a
different torch build) by loading decode submodules directly from
disk (same trick as ``test_phase_b_prefill_dual_write.py``).

Run standalone:
    python rtp_llm/models_py/modules/dsv4/decode/test/test_phase_e_paged_read_roundtrip.py
"""

from __future__ import annotations

import importlib.util
import os
import sys

import torch


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pl = _load(f"{_BASE}/pool_layout.py", "_pl_phaseE")
kw = _load(f"{_BASE}/kv_write_decode_op.py", "_kw_phaseE")


# ---------------------------------------------------------------------------
# Writer: mirrors ``Attention._prefill_paged_write_kv`` exactly.
# Identical to ``_prefill_paged_write_kv_ref`` in test_phase_b — kept here
# for test independence and to document the slot_mapping formula the
# corresponding reader has to invert.
# ---------------------------------------------------------------------------


def _build_slot_mapping(
    bt: torch.Tensor, T: int, eb: int, device: torch.device
) -> torch.Tensor:
    """Build ``[1, T]`` slot_mapping matching the production writer.

    Mirrors ``Attention._prefill_paged_write_kv``:
      ``slot = (block_id > 0) ? block_id * eb + (pos % eb) : -1``
      where ``block_id = bt[0, pos // eb]``.
    """
    assert bt.shape[0] == 1
    max_blocks = bt.shape[1]
    pos = torch.arange(T, device=device, dtype=torch.long)
    block_in_seq = (pos // eb).clamp_(0, max(0, max_blocks - 1))
    in_block = pos % eb
    bt_long = bt.to(torch.long)
    block_id = bt_long[0, block_in_seq]
    valid = block_id > 0
    slot_per = torch.where(
        valid,
        block_id * eb + in_block,
        torch.full_like(in_block, -1),
    )
    return slot_per


def _paged_write(
    desc: "pl.PoolDescriptor",
    bt: torch.Tensor,
    source_buf: torch.Tensor,
) -> None:
    T = int(source_buf.shape[1])
    D = int(source_buf.shape[2])
    if T == 0:
        return
    slot_mapping = _build_slot_mapping(bt, T, desc.entries_per_block, source_buf.device)
    kw.write_kv_to_pool(
        source_buf.reshape(T, D),
        slot_mapping,
        desc.view(),
        mask_negative=True,
    )


# ---------------------------------------------------------------------------
# Reader: the symmetrical op the Phase E read-path swap will call.
#
# For each position ``pos`` in [0, T):
#   slot = block_id[pos] * eb + (pos % eb)   if valid, else -1
#   out[pos] = pool_view[slot]               if valid, else undefined
# ---------------------------------------------------------------------------


def _paged_read(
    desc: "pl.PoolDescriptor",
    bt: torch.Tensor,
    T: int,
    D: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Gather ``[1, T, D]`` from pool via block_table — inverse of _paged_write.

    Entries whose slot is -1 (sentinel block_id <= 0) are filled with zeros —
    the writer leaves the pool unchanged for those positions, so the tensor
    value is whatever the pool had (pool starts zero-initialized in tests).

    NOTE: the production read path (``Attention.forward`` line 1631) reads
    the FULL ``self.kv_cache[:bsz]``, including positions never written.
    Those are zero because the register_buffer is ``torch.zeros(...)``.
    Pool is also zero-initialized in the framework allocator. So 'read -1
    → zero' matches exactly what the current production path sees.
    """
    assert bt.shape[0] == 1
    device = bt.device
    eb = desc.entries_per_block
    slot_mapping = _build_slot_mapping(bt, T, eb, device)  # [T] int64
    pool_view = desc.view()  # [num_blocks * eb, D] typed

    # index_select with -1 is illegal — redirect to slot 0 (same trick as
    # the writer's safe-redirect). After gather, mask back to zeros.
    valid = slot_mapping >= 0
    safe_slot = torch.where(valid, slot_mapping, torch.zeros_like(slot_mapping))  # [T]
    gathered = pool_view.index_select(0, safe_slot)  # [T, D]
    zero_row = torch.zeros(D, dtype=dtype, device=device)
    out_flat = torch.where(valid.unsqueeze(-1), gathered, zero_row)
    return out_flat.view(1, T, D)


# ---------------------------------------------------------------------------
# Round-trip validation
# ---------------------------------------------------------------------------


def _run_kv_roundtrip(
    *,
    attn_type: int,
    head_dim: int,
    entries_per_block: int,
    num_blocks: int,
    T: int,
    block_ids: list[int],
    coff: int,
) -> None:
    device = torch.device("cpu")
    stride_bytes = entries_per_block * head_dim * 2  # bf16

    torch.manual_seed(0xE7E7 + attn_type + T)
    src = torch.randn(1, T, head_dim, dtype=torch.bfloat16, device=device)

    pool = torch.zeros(num_blocks, stride_bytes, dtype=torch.uint8, device=device)
    bt = torch.tensor(block_ids, dtype=torch.int64, device=device).unsqueeze(0)

    desc = pl.build_pool_descriptor(
        pool,
        attn_type=attn_type,
        head_dim=head_dim,
        indexer_head_dim=head_dim,  # unused for non-INDEXER
        coff=coff,
    )
    assert desc is not None
    assert desc.entries_per_block == entries_per_block

    # Write then read back.
    _paged_write(desc, bt, src)
    out = _paged_read(desc, bt, T, head_dim, torch.bfloat16)

    # Build the expected tensor: source at positions with valid block,
    # zero at positions with sentinel block_id.
    slot_mapping = _build_slot_mapping(bt, T, entries_per_block, device)
    valid = (slot_mapping >= 0).view(1, T, 1).expand(1, T, head_dim)
    expect = torch.where(valid, src, torch.zeros_like(src))

    # Byte-level equality.
    diff_bytes = int(
        (out.contiguous().view(torch.uint8) != expect.contiguous().view(torch.uint8))
        .sum()
        .item()
    )
    assert diff_bytes == 0, (
        f"attn_type={attn_type} T={T} block_ids={block_ids}: "
        f"{diff_bytes} byte(s) differ after write→read round trip"
    )


def _run_state_roundtrip(
    *,
    attn_type: int,
    half_dim: int,
    entries_per_block: int,
    state_rows: int,
    num_blocks: int,
    block_ids: list[int],
    coff: int,
) -> None:
    """STATE pool round trip: write kv_state‖score_state, read back both halves."""
    device = torch.device("cpu")
    state_dim = 2 * half_dim
    stride_bytes = entries_per_block * state_dim * 4  # fp32

    torch.manual_seed(0xDEAD + attn_type)
    kv_s = torch.randn(1, state_rows, half_dim, dtype=torch.float32, device=device)
    sc_s = torch.randn(1, state_rows, half_dim, dtype=torch.float32, device=device)
    src = torch.cat([kv_s, sc_s], dim=-1)  # [1, state_rows, state_dim]

    pool = torch.zeros(num_blocks, stride_bytes, dtype=torch.uint8, device=device)
    bt = torch.tensor(block_ids, dtype=torch.int64, device=device).unsqueeze(0)

    desc = pl.build_pool_descriptor(
        pool,
        attn_type=attn_type,
        head_dim=half_dim // coff,
        indexer_head_dim=half_dim // coff,
        coff=coff,
    )
    assert desc is not None
    assert desc.entries_per_block == entries_per_block

    _paged_write(desc, bt, src)
    out = _paged_read(desc, bt, state_rows, state_dim, torch.float32)

    slot_mapping = _build_slot_mapping(bt, state_rows, entries_per_block, device)
    valid = (slot_mapping >= 0).view(1, state_rows, 1).expand(1, state_rows, state_dim)
    expect = torch.where(valid, src, torch.zeros_like(src))

    diff_bytes = int(
        (out.contiguous().view(torch.uint8) != expect.contiguous().view(torch.uint8))
        .sum()
        .item()
    )
    assert diff_bytes == 0, (
        f"STATE attn_type={attn_type} state_rows={state_rows} "
        f"block_ids={block_ids}: {diff_bytes} byte(s) differ after round trip"
    )

    # Also verify the half-split: kv_state is the first half_dim dims,
    # score_state is the last half_dim dims. Phase E compressor read path
    # will pull out each half via slicing.
    kv_out = out[..., :half_dim]
    sc_out = out[..., half_dim:]
    kv_expect = torch.where(valid[..., :half_dim], kv_s, torch.zeros_like(kv_s))
    sc_expect = torch.where(valid[..., half_dim:], sc_s, torch.zeros_like(sc_s))
    assert (
        int((kv_out != kv_expect).sum().item()) == 0
    ), f"kv_state half mismatch for attn_type={attn_type}"
    assert (
        int((sc_out != sc_expect).sum().item()) == 0
    ), f"score_state half mismatch for attn_type={attn_type}"


# ---------------------------------------------------------------------------
# KV round-trip test cases — mirror the Phase-B write test coverage.
# ---------------------------------------------------------------------------


def test_swa_roundtrip_full_block():
    _run_kv_roundtrip(
        attn_type=pl.SWA_KV,
        head_dim=512,
        entries_per_block=256,
        num_blocks=8,
        T=128,
        block_ids=[3, 5],
        coff=1,
    )


def test_swa_roundtrip_sentinel():
    _run_kv_roundtrip(
        attn_type=pl.SWA_KV,
        head_dim=512,
        entries_per_block=256,
        num_blocks=8,
        T=256,
        block_ids=[4, 0],  # 2nd block sentinel → positions in block-1 read zero
        coff=1,
    )


def test_csa_roundtrip():
    _run_kv_roundtrip(
        attn_type=pl.CSA_KV,
        head_dim=512,
        entries_per_block=64,
        num_blocks=16,
        T=64,
        block_ids=[7, 9],
        coff=2,
    )


def test_hca_roundtrip():
    _run_kv_roundtrip(
        attn_type=pl.HCA_KV,
        head_dim=512,
        entries_per_block=2,
        num_blocks=32,
        T=2,
        block_ids=[11, 13],
        coff=1,
    )


def test_indexer_kv_roundtrip():
    _run_kv_roundtrip(
        attn_type=pl.INDEXER_KV,
        head_dim=128,
        entries_per_block=64,
        num_blocks=24,
        T=64,
        block_ids=[17, 0],
        coff=2,
    )


def test_swa_roundtrip_T_less_than_eb():
    _run_kv_roundtrip(
        attn_type=pl.SWA_KV,
        head_dim=512,
        entries_per_block=256,
        num_blocks=4,
        T=100,
        block_ids=[2, 3],
        coff=1,
    )


# ---------------------------------------------------------------------------
# STATE round-trip test cases.
# ---------------------------------------------------------------------------


def test_csa_state_roundtrip():
    _run_state_roundtrip(
        attn_type=pl.CSA_STATE,
        half_dim=1024,
        entries_per_block=4,
        state_rows=8,
        num_blocks=8,
        block_ids=[1, 2],
        coff=2,
    )


def test_hca_state_roundtrip():
    _run_state_roundtrip(
        attn_type=pl.HCA_STATE,
        half_dim=512,
        entries_per_block=8,
        state_rows=16,
        num_blocks=8,
        block_ids=[3, 5],
        coff=1,
    )


def test_indexer_state_roundtrip():
    _run_state_roundtrip(
        attn_type=pl.INDEXER_STATE,
        half_dim=256,
        entries_per_block=4,
        state_rows=8,
        num_blocks=8,
        block_ids=[7, 0],
        coff=2,
    )


if __name__ == "__main__":
    test_swa_roundtrip_full_block()
    test_swa_roundtrip_sentinel()
    test_csa_roundtrip()
    test_hca_roundtrip()
    test_indexer_kv_roundtrip()
    test_swa_roundtrip_T_less_than_eb()
    test_csa_state_roundtrip()
    test_hca_state_roundtrip()
    test_indexer_state_roundtrip()
    print("All Phase E paged-read round-trip tests passed.")
