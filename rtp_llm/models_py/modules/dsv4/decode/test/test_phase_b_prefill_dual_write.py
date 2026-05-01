"""Phase B sanity: prefill SWA/CSA/HCA dual-write byte-equals scatter.

Validates that the slot-mapping + ``write_kv_to_pool`` path used by
``Attention._prefill_paged_write_kv`` writes EXACTLY the same bytes to
the framework BlockPool that ``_scatter_kv_pool``'s byte-level copy
produces. If this test passes, Phase D (drop scatter for the 4 KV
pools) is safe — both halves of the dual-write land the same data.

Bypasses the ``rtp_llm`` package init (which pulls C++ ops linked
against a different torch build) by importing the decode submodules
directly from disk.

Run standalone:
    pytest rtp_llm/models_py/modules/dsv4/decode/test/test_phase_b_prefill_dual_write.py
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
pl = _load(f"{_BASE}/pool_layout.py", "_pl_phaseB")
kw = _load(f"{_BASE}/kv_write_decode_op.py", "_kw_phaseB")


# ---------------------------------------------------------------------------
# Reference implementation: mirrors the inlined loop in
# ``DeepSeekV4Model._scatter_kv_pool`` exactly.
# ---------------------------------------------------------------------------


def _scatter_kv_pool_ref(
    fw_tensor: torch.Tensor,  # [num_blocks, stride_bytes] uint8
    block_ids_2d: torch.Tensor,  # [B, max_blocks] int
    python_buf: torch.Tensor,  # [B, T, head_dim] bf16
    entries_per_block: int,
    head_dim: int,
    B: int,
) -> None:
    bytes_per_entry = head_dim * 2  # bf16
    page_bytes_avail = fw_tensor.size(1)
    capacity = min(entries_per_block, page_bytes_avail // bytes_per_entry)
    T = python_buf.size(1)
    bids_cpu = block_ids_2d[:B].cpu()
    max_blks = min(2, bids_cpu.size(1))
    for b in range(B):
        row = bids_cpu[b]
        for k in range(min(row.size(0), max_blks)):
            bid = int(row[k].item())
            if bid <= 0:
                continue
            entry_start = k * capacity
            entry_end = min(entry_start + capacity, T)
            n = entry_end - entry_start
            if n <= 0:
                break
            data = (
                python_buf[b, entry_start:entry_end]
                .contiguous()
                .view(torch.uint8)
                .reshape(-1)
            )
            fw_tensor[bid, : n * bytes_per_entry] = data


# ---------------------------------------------------------------------------
# Paged path: mirrors ``Attention._prefill_paged_write_kv`` exactly.
# ---------------------------------------------------------------------------


def _prefill_paged_write_kv_ref(
    desc: "pl.PoolDescriptor",
    bt: torch.Tensor,  # [1, max_blocks] int
    source_buf: torch.Tensor,  # [1, T, vec_dim]
) -> None:
    assert source_buf.shape[0] == 1 and bt.shape[0] == 1
    T = int(source_buf.shape[1])
    D = int(source_buf.shape[2])
    if T == 0:
        return
    device = source_buf.device
    eb = desc.entries_per_block
    pos = torch.arange(T, device=device, dtype=torch.long)
    max_blocks = bt.shape[1]
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
    slot_mapping = slot_per.reshape(-1)
    buf_flat = source_buf.reshape(T, D)
    kw.write_kv_to_pool(buf_flat, slot_mapping, desc.view(), mask_negative=True)


# ---------------------------------------------------------------------------
# Byte-equal validation: apply both writers to independent pools, compare.
# ---------------------------------------------------------------------------


def _run_case(
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
    max_blocks_per_req = len(block_ids)

    # Source data (same bytes for both paths).
    torch.manual_seed(0xB00B + attn_type + T)
    src = torch.randn(1, T, head_dim, dtype=torch.bfloat16, device=device)

    # Two independent pools.
    pool_scatter = torch.zeros(
        num_blocks, stride_bytes, dtype=torch.uint8, device=device
    )
    pool_paged = torch.zeros(num_blocks, stride_bytes, dtype=torch.uint8, device=device)

    bt = torch.tensor(block_ids, dtype=torch.int64, device=device).unsqueeze(0)

    # Scatter path.
    _scatter_kv_pool_ref(pool_scatter, bt, src, entries_per_block, head_dim, B=1)

    # Paged path.
    desc = pl.build_pool_descriptor(
        pool_paged,
        attn_type=attn_type,
        head_dim=head_dim,
        indexer_head_dim=head_dim,  # unused for non-INDEXER attn_types
        coff=coff,
    )
    assert desc is not None
    assert (
        desc.entries_per_block == entries_per_block
    ), f"entries_per_block mismatch: desc={desc.entries_per_block} vs test={entries_per_block}"
    _prefill_paged_write_kv_ref(desc, bt, src)

    # Byte-level equality.
    diff_bytes = int((pool_scatter != pool_paged).sum().item())
    assert diff_bytes == 0, (
        f"attn_type={attn_type} eb={entries_per_block} T={T} block_ids={block_ids}: "
        f"{diff_bytes} byte(s) differ between scatter and paged"
    )


def test_swa_exact_win():
    # SWA with head_dim=512, eb=256, T=win=128 → block_in_seq always 0.
    _run_case(
        attn_type=pl.SWA_KV,
        head_dim=512,
        entries_per_block=256,
        num_blocks=8,
        T=128,
        block_ids=[3, 5],  # block 0 valid, block 1 valid
        coff=1,
    )


def test_swa_sentinel_block():
    # Block 1 unallocated (bid=0) — scatter skips it, paged must emit -1 for those slots.
    _run_case(
        attn_type=pl.SWA_KV,
        head_dim=512,
        entries_per_block=256,
        num_blocks=8,
        T=256,  # T = 1 full block
        block_ids=[4, 0],  # block 1 sentinel
        coff=1,
    )


def test_csa_ratio4():
    # CSA: head_dim=512, eb=64 (256//4).
    _run_case(
        attn_type=pl.CSA_KV,
        head_dim=512,
        entries_per_block=64,
        num_blocks=16,
        T=64,  # 1 full block
        block_ids=[7, 9],
        coff=2,  # CSA overlap=True
    )


def test_hca_ratio128():
    # HCA: head_dim=512, eb=2 (256//128).
    _run_case(
        attn_type=pl.HCA_KV,
        head_dim=512,
        entries_per_block=2,
        num_blocks=32,
        T=2,
        block_ids=[11, 13],
        coff=1,  # HCA overlap=False
    )


def test_indexer_kv():
    # INDEXER_KV: head_dim=128 (idx_hd), eb=64.
    _run_case(
        attn_type=pl.INDEXER_KV,
        head_dim=128,
        entries_per_block=64,
        num_blocks=24,
        T=64,
        block_ids=[17, 0],  # second block sentinel
        coff=2,
    )


def test_swa_T_less_than_block():
    # T < entries_per_block; only partial slot range touched.
    _run_case(
        attn_type=pl.SWA_KV,
        head_dim=512,
        entries_per_block=256,
        num_blocks=4,
        T=100,  # less than eb
        block_ids=[2, 3],
        coff=1,
    )


# ---------------------------------------------------------------------------
# Phase B.3: STATE pool byte-equal tests
# ---------------------------------------------------------------------------


def _scatter_state_pool_ref(
    fw_tensor: torch.Tensor,  # [num_blocks, stride_bytes] uint8
    block_ids_2d: torch.Tensor,  # [B, max_blocks] int
    kv_state: torch.Tensor,  # [B, state_rows, half_dim] fp32
    score_state: torch.Tensor,  # [B, state_rows, half_dim] fp32
    entries_per_block: int,
    state_dim: int,
    B: int,
) -> None:
    """Exact mirror of ``DeepSeekV4Model._scatter_state_pool``."""
    half_dim = state_dim // 2
    bids_cpu = block_ids_2d[:B].cpu()
    max_blks = min(2, bids_cpu.size(1))
    state_rows = kv_state.size(1)
    for b in range(B):
        for blk_idx in range(max_blks):
            bid = int(bids_cpu[b, blk_idx].item())
            if bid <= 0:
                continue
            page_fp32 = (
                fw_tensor[bid, : entries_per_block * state_dim * 4]
                .view(torch.float32)
                .view(entries_per_block, state_dim)
            )
            row_start = blk_idx * entries_per_block
            n = min(entries_per_block, state_rows - row_start)
            if n <= 0:
                break
            page_fp32[:n, :half_dim] = kv_state[b, row_start : row_start + n]
            page_fp32[:n, half_dim:] = score_state[b, row_start : row_start + n]


def _prefill_paged_write_state_ref(
    desc: "pl.PoolDescriptor",
    bt: torch.Tensor,  # [1, max_blocks]
    kv_state: torch.Tensor,  # [1, state_rows, half_dim]
    score_state: torch.Tensor,  # [1, state_rows, half_dim]
) -> None:
    """Mirror of Attention._prefill_paged_write_state — cat(kv||score) then
    hand to the generic paged writer."""
    merged = torch.cat([kv_state, score_state], dim=-1)
    _prefill_paged_write_kv_ref(desc, bt, merged)


def _run_state_case(
    *,
    attn_type: int,
    half_dim: int,  # kv_state / score_state inner dim
    entries_per_block: int,  # pool block capacity (state slots)
    state_rows: int,  # rows in compressor.kv_state[:B, :, :]
    num_blocks: int,
    block_ids: list[int],
    coff: int,
) -> None:
    device = torch.device("cpu")
    state_dim = 2 * half_dim
    stride_bytes = entries_per_block * state_dim * 4  # fp32

    torch.manual_seed(0xBEEF + attn_type)
    kv_s = torch.randn(1, state_rows, half_dim, dtype=torch.float32, device=device)
    sc_s = torch.randn(1, state_rows, half_dim, dtype=torch.float32, device=device)

    pool_scatter = torch.zeros(
        num_blocks, stride_bytes, dtype=torch.uint8, device=device
    )
    pool_paged = torch.zeros(num_blocks, stride_bytes, dtype=torch.uint8, device=device)

    bt = torch.tensor(block_ids, dtype=torch.int64, device=device).unsqueeze(0)

    # Scatter path.
    _scatter_state_pool_ref(
        pool_scatter, bt, kv_s, sc_s, entries_per_block, state_dim, B=1
    )

    # Paged path. indexer_head_dim only matters for INDEXER_STATE.
    desc = pl.build_pool_descriptor(
        pool_paged,
        attn_type=attn_type,
        head_dim=half_dim // coff,
        indexer_head_dim=half_dim // coff,
        coff=coff,
    )
    assert desc is not None
    assert desc.entries_per_block == entries_per_block, (
        f"entries_per_block mismatch: desc={desc.entries_per_block} vs "
        f"test={entries_per_block}"
    )
    _prefill_paged_write_state_ref(desc, bt, kv_s, sc_s)

    diff_bytes = int((pool_scatter != pool_paged).sum().item())
    assert diff_bytes == 0, (
        f"STATE attn_type={attn_type} eb={entries_per_block} "
        f"state_rows={state_rows} block_ids={block_ids}: {diff_bytes} byte(s) differ"
    )


def test_csa_state():
    # CSA_STATE: coff=2, head_dim=512 → half_dim=1024, state_dim=2048, eb=4.
    # state_rows = coff*ratio = 8 = eb*2, fills exactly 2 blocks.
    _run_state_case(
        attn_type=pl.CSA_STATE,
        half_dim=1024,
        entries_per_block=4,
        state_rows=8,
        num_blocks=8,
        block_ids=[1, 2],
        coff=2,
    )


def test_hca_state():
    # HCA_STATE: coff=1, head_dim=512 → half_dim=512, state_dim=1024, eb=8.
    # state_rows = ratio = 128? Actually HCA compressor: coff*ratio = 128.
    # 128 rows / eb=8 per block = 16 blocks, but allocator gives only 2 →
    # scatter only writes first 2*eb=16 rows. Test with state_rows=16.
    _run_state_case(
        attn_type=pl.HCA_STATE,
        half_dim=512,
        entries_per_block=8,
        state_rows=16,
        num_blocks=8,
        block_ids=[3, 5],
        coff=1,
    )


def test_indexer_state():
    # INDEXER_STATE: coff=2, idx_hd=128 → half_dim=256, state_dim=512, eb=4.
    _run_state_case(
        attn_type=pl.INDEXER_STATE,
        half_dim=256,
        entries_per_block=4,
        state_rows=8,
        num_blocks=8,
        block_ids=[7, 0],  # 2nd block sentinel
        coff=2,
    )


if __name__ == "__main__":
    test_swa_exact_win()
    test_swa_sentinel_block()
    test_csa_ratio4()
    test_hca_ratio128()
    test_indexer_kv()
    test_swa_T_less_than_block()
    test_csa_state()
    test_hca_state()
    test_indexer_state()
    print("All Phase B dual-write byte-equal tests passed.")
