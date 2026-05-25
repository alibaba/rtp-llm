"""Comprehensive UT + perf test for fused decode metadata Triton kernels.

Covers:
  - fused_update_decode_meta_pure q_len=1: bs=1-256, seqlen=1-1M (189 cases)
  - fused_update_decode_meta_pure q_len=2,4,6:
    bs=1-128, seqlen=4-131072 plus ratio=4/128 boundary starts (693 cases)
    — validates MTP/target-verify path
  - fused_update_decode_meta_pure heterogeneous start_pos per request
    (q_len=1,2,4,6; bs=4,16,64) — exercises per-q HCA causality (12 cases)
  - fused_phase2b_pool_slot_mapping q_len=1,2,4,6:
    bs=1-128, seqlen=4-131072 (448 cases)

Reference: standalone PyTorch reimplementation of the old code.

Run:
  CUDA_VISIBLE_DEVICES=0 python3 rtp_llm/models_py/modules/dsv4/fp8/test/test_fused_decode_meta_comprehensive.py
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

# Direct-load modules to avoid package init chain
_THIS_DIR = Path(__file__).resolve().parent
_DECODE_DIR = _THIS_DIR.parent / "decode"

import importlib.util
import types


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Pre-register attn_type so that fused module's internal import resolves
_attn_type_mod = _load_module(
    "rtp_llm.models_py.modules.dsv4.attn_type",
    _THIS_DIR.parent.parent / "attn_type.py",
)
_fused_mod = _load_module(
    "rtp_llm.models_py.modules.dsv4.fp8.decode._fused_prepare_meta_triton",
    _DECODE_DIR / "_fused_prepare_meta_triton.py",
)

fused_update_decode_meta_pure = _fused_mod.fused_update_decode_meta_pure
fused_phase2b_pool_slot_mapping = _fused_mod.fused_phase2b_pool_slot_mapping

# ---------------------------------------------------------------------------
# Minimal mock metadata for testing (mirrors DSv4DecodeAttnMetadataFP8 fields)
# ---------------------------------------------------------------------------

_WINDOW = 128
_INDEX_TOPK = 512
_HCA_DENSE_WIDTH = 512  # ceil(max_seq_len//128 / 64) * 64, typical


@dataclass
class MockMeta:
    q_len_per_req: int
    window_size: int
    max_seq_len: int
    compressed_buffer_t_dim_per_ratio: Dict[int, int]

    start_pos: torch.Tensor
    slot_mapping_swa: torch.Tensor
    slot_mapping_compressed: Dict[int, torch.Tensor]
    compressed_lens: Dict[int, torch.Tensor]
    compressed_lens_per_token: Dict[int, torch.Tensor]
    topk_window_idxs: torch.Tensor
    topk_buffer_compressed: torch.Tensor
    topk_total_by_ratio: Dict[int, torch.Tensor]
    swa_abs_idx: torch.Tensor

    # Phase 2b fields
    pool_block_tables: Dict[int, torch.Tensor] = field(default_factory=dict)
    pool_write_slot_mappings: Dict[int, torch.Tensor] = field(default_factory=dict)


def _alloc_mock(max_bs: int, q_len: int, max_seq_len: int, device) -> MockMeta:
    T = max_bs * q_len
    stride_4 = max_seq_len // 4
    stride_128 = max_seq_len // 128
    hca_dense = ((stride_128 + 63) // 64) * 64

    return MockMeta(
        q_len_per_req=q_len,
        window_size=_WINDOW,
        max_seq_len=max_seq_len,
        compressed_buffer_t_dim_per_ratio={4: stride_4, 128: stride_128},
        start_pos=torch.full((max_bs,), -1, dtype=torch.int32, device=device),
        slot_mapping_swa=torch.full((T,), -1, dtype=torch.int32, device=device),
        slot_mapping_compressed={
            4: torch.full((T,), -1, dtype=torch.int32, device=device),
            128: torch.full((T,), -1, dtype=torch.int32, device=device),
        },
        compressed_lens={
            4: torch.zeros(max_bs, dtype=torch.int32, device=device),
            128: torch.zeros(max_bs, dtype=torch.int32, device=device),
        },
        compressed_lens_per_token={
            4: torch.zeros(max_bs, q_len, dtype=torch.int32, device=device),
            128: torch.zeros(max_bs, q_len, dtype=torch.int32, device=device),
        },
        topk_window_idxs=torch.full(
            (max_bs, q_len, _WINDOW), -1, dtype=torch.int32, device=device
        ),
        topk_buffer_compressed=torch.full(
            (max_bs, q_len, _INDEX_TOPK), -1, dtype=torch.int32, device=device
        ),
        topk_total_by_ratio={
            4: torch.full(
                (max_bs, q_len, _WINDOW + _INDEX_TOPK),
                -1,
                dtype=torch.int32,
                device=device,
            ),
            128: torch.full(
                (max_bs, q_len, _WINDOW + hca_dense),
                -1,
                dtype=torch.int32,
                device=device,
            ),
        },
        swa_abs_idx=torch.full(
            (max_bs, q_len, _WINDOW), -1, dtype=torch.int32, device=device
        ),
    )


def _alloc_mock_subset(
    max_bs: int,
    q_len: int,
    max_seq_len: int,
    device,
    ratios: Tuple[int, ...] = (),
) -> MockMeta:
    """Mock with an arbitrary subset of compress ratios from {4, 128}.

    ``ratios=()``  → SWA-only (HAS_CMP_4=False, HAS_CMP_128=False),
                     models MTP draft layers (``compress_ratios=[0]``).
    ``ratios=(4,)`` → CSA/INDEXER only (HAS_CMP_4=True, HAS_CMP_128=False).
    ``ratios=(128,)`` → HCA only (HAS_CMP_4=False, HAS_CMP_128=True).

    Exercises the kernel's per-ratio constexpr gate + None-pointer path.
    """
    T = max_bs * q_len
    stride_4 = max_seq_len // 4
    stride_128 = max_seq_len // 128
    hca_dense = ((stride_128 + 63) // 64) * 64

    compressed_buffer_t_dim_per_ratio: Dict[int, int] = {}
    slot_mapping_compressed: Dict[int, torch.Tensor] = {}
    compressed_lens: Dict[int, torch.Tensor] = {}
    compressed_lens_per_token: Dict[int, torch.Tensor] = {}
    topk_total_by_ratio: Dict[int, torch.Tensor] = {}

    if 4 in ratios:
        compressed_buffer_t_dim_per_ratio[4] = stride_4
        slot_mapping_compressed[4] = torch.full(
            (T,), -1, dtype=torch.int32, device=device
        )
        compressed_lens[4] = torch.zeros(max_bs, dtype=torch.int32, device=device)
        compressed_lens_per_token[4] = torch.zeros(
            max_bs, q_len, dtype=torch.int32, device=device
        )
        topk_total_by_ratio[4] = torch.full(
            (max_bs, q_len, _WINDOW + _INDEX_TOPK),
            -1,
            dtype=torch.int32,
            device=device,
        )

    if 128 in ratios:
        compressed_buffer_t_dim_per_ratio[128] = stride_128
        slot_mapping_compressed[128] = torch.full(
            (T,), -1, dtype=torch.int32, device=device
        )
        compressed_lens[128] = torch.zeros(max_bs, dtype=torch.int32, device=device)
        compressed_lens_per_token[128] = torch.zeros(
            max_bs, q_len, dtype=torch.int32, device=device
        )
        topk_total_by_ratio[128] = torch.full(
            (max_bs, q_len, _WINDOW + hca_dense),
            -1,
            dtype=torch.int32,
            device=device,
        )

    return MockMeta(
        q_len_per_req=q_len,
        window_size=_WINDOW,
        max_seq_len=max_seq_len,
        compressed_buffer_t_dim_per_ratio=compressed_buffer_t_dim_per_ratio,
        start_pos=torch.full((max_bs,), -1, dtype=torch.int32, device=device),
        slot_mapping_swa=torch.full((T,), -1, dtype=torch.int32, device=device),
        slot_mapping_compressed=slot_mapping_compressed,
        compressed_lens=compressed_lens,
        compressed_lens_per_token=compressed_lens_per_token,
        topk_window_idxs=torch.full(
            (max_bs, q_len, _WINDOW), -1, dtype=torch.int32, device=device
        ),
        topk_buffer_compressed=torch.full(
            (max_bs, q_len, _INDEX_TOPK), -1, dtype=torch.int32, device=device
        ),
        topk_total_by_ratio=topk_total_by_ratio,
        swa_abs_idx=torch.full(
            (max_bs, q_len, _WINDOW), -1, dtype=torch.int32, device=device
        ),
    )


def _alloc_mock_swa_only(max_bs: int, q_len: int, max_seq_len: int, device) -> MockMeta:
    """Back-compat shim: equivalent to ``_alloc_mock_subset(ratios=())``."""
    return _alloc_mock_subset(max_bs, q_len, max_seq_len, device, ratios=())


def _alloc_phase2b(meta: MockMeta, bs: int, device, seqlen: int = 0) -> None:
    """Add pool_block_tables and pool_write_slot_mappings for phase2b test.

    Block tables are sized dynamically based on seqlen and have 0 (sentinel)
    beyond the blocks actually needed, matching production behavior.
    """
    SWA_KV, CSA_KV, HCA_KV, INDEXER_KV = 7, 1, 2, 3
    entries = {SWA_KV: 256, CSA_KV: 64, INDEXER_KV: 64, HCA_KV: 2}
    T = bs * meta.q_len_per_req

    for at in (SWA_KV, CSA_KV, INDEXER_KV, HCA_KV):
        E = entries[at]
        if at == SWA_KV:
            needed = (seqlen + E - 1) // E + 1
        elif at in (CSA_KV, INDEXER_KV):
            needed = (seqlen // 4 + E - 1) // E + 1
        else:
            needed = (seqlen // 128 + E - 1) // E + 1
        max_blocks = max(needed + 4, 32)
        bt = torch.zeros((bs, max_blocks), dtype=torch.int32, device=device)
        bt[:, :needed] = torch.randint(
            1, 1000, (bs, needed), dtype=torch.int32, device=device
        )
        meta.pool_block_tables[at] = bt
        meta.pool_write_slot_mappings[at] = torch.full(
            (T,), -1, dtype=torch.int64, device=device
        )


# ---------------------------------------------------------------------------
# PyTorch reference (the code we removed from update_decode_metadata_in_place)
# ---------------------------------------------------------------------------


def ref_update_pure(meta: MockMeta, start_pos: torch.Tensor) -> None:
    """Reference PyTorch implementation of what fused_update_decode_meta_pure does."""
    bs = int(start_pos.shape[0])
    q_len = meta.q_len_per_req
    window_size = meta.window_size
    device = start_pos.device

    # position_ids_2d
    offs = torch.arange(q_len, device=device, dtype=torch.int32)
    position_ids_2d = start_pos[:bs].unsqueeze(1) + offs.unsqueeze(0)

    # start_pos
    meta.start_pos[:bs].copy_(start_pos)

    # SWA slot mapping
    in_ring = position_ids_2d % window_size
    swa_base = (
        torch.arange(bs, device=device, dtype=torch.int32).view(bs, 1) * window_size
    )
    meta.slot_mapping_swa[: bs * q_len].copy_((swa_base + in_ring).reshape(-1))

    # Window topk indices
    abs_pos_b = position_ids_2d.unsqueeze(-1)
    sp_ring = (position_ids_2d % window_size).unsqueeze(-1)
    k = torch.arange(window_size, device=device, dtype=torch.int32).view(
        1, 1, window_size
    )
    ring_full = (sp_ring + 1 + k) % window_size
    partial = torch.where(k <= abs_pos_b, k, torch.full_like(k, -1))
    is_full = abs_pos_b >= (window_size - 1)
    window_idxs = torch.where(is_full, ring_full, partial).to(torch.int32)
    meta.topk_window_idxs[:bs].copy_(window_idxs)

    # swa_abs_idx
    win_start = (abs_pos_b - window_size + 1).clamp(min=0)
    candidate = win_start + k
    valid_pos = candidate <= abs_pos_b
    meta.swa_abs_idx[:bs].copy_(
        torch.where(valid_pos, candidate, torch.full_like(candidate, -1))
    )

    # Per-ratio compressed
    for r, slot_t in meta.slot_mapping_compressed.items():
        stride = meta.compressed_buffer_t_dim_per_ratio[r]
        abs_pos_plus_1 = position_ids_2d + 1
        on_boundary = (abs_pos_plus_1 % r) == 0
        in_req = abs_pos_plus_1 // r - 1
        compressed_lens_per_token = (abs_pos_plus_1 // r).to(torch.int32)
        cmp_req_base = (
            torch.arange(bs, device=device, dtype=torch.int32).view(bs, 1) * stride
        )
        flat = (cmp_req_base + in_req).reshape(-1)
        mask = on_boundary.reshape(-1)
        cmp_slots = torch.where(mask, flat, torch.full_like(flat, -1))
        slot_t[: bs * q_len].copy_(cmp_slots)

        meta.compressed_lens_per_token[r][:bs].copy_(compressed_lens_per_token)
        meta.compressed_lens[r][:bs].copy_(compressed_lens_per_token[:, -1])

        total = meta.topk_total_by_ratio[r]
        total[:bs, :, :window_size].copy_(window_idxs)
        if r != 4:
            K_dense = total.shape[-1] - window_size
            dense_idxs = (
                torch.arange(K_dense, device=device, dtype=torch.int32)
                .view(1, 1, K_dense)
                .expand(bs, q_len, K_dense)
            )
            cmp_lens_pt = compressed_lens_per_token.view(bs, q_len, 1)
            valid_h = dense_idxs < cmp_lens_pt
            total[:bs, :, window_size:].copy_(
                torch.where(valid_h, dense_idxs, torch.full_like(dense_idxs, -1))
            )
        else:
            total[:bs, :, window_size:].fill_(-1)


def ref_phase2b(
    meta: MockMeta, start_pos: torch.Tensor, bs: int, entries_per_block: Dict[int, int]
) -> None:
    """Reference for fused_phase2b_pool_slot_mapping."""
    SWA_KV, CSA_KV, HCA_KV, INDEXER_KV = 7, 1, 2, 3
    q_len = meta.q_len_per_req
    device = start_pos.device

    offs = torch.arange(q_len, device=device, dtype=torch.int32)
    position_ids_2d = start_pos[:bs].unsqueeze(1) + offs.unsqueeze(0)
    abs_pos_flat = position_ids_2d.reshape(-1)
    abs_pos_plus_1 = position_ids_2d + 1

    def _slot_mapping(bt, abs_pos_1d, E):
        block_in_seq = abs_pos_1d // E
        in_block = abs_pos_1d % E
        max_blocks = bt.shape[1]
        safe_bis = block_in_seq.clamp(0, max_blocks - 1)
        req_idx = (
            torch.arange(bs, device=device, dtype=torch.long)
            .unsqueeze(1)
            .expand(-1, q_len)
            .reshape(-1)
        )
        bid = bt.to(torch.long)[req_idx, safe_bis.to(torch.long)]
        valid = bid > 0
        slot = bid * E + in_block.to(torch.long)
        return torch.where(valid, slot, torch.full_like(slot, -1))

    # SWA
    E_swa = entries_per_block[SWA_KV]
    meta.pool_write_slot_mappings[SWA_KV][: bs * q_len].copy_(
        _slot_mapping(meta.pool_block_tables[SWA_KV][:bs], abs_pos_flat, E_swa)
    )

    # CSA + INDEXER (ratio=4)
    on_b4 = (abs_pos_plus_1 % 4) == 0
    cmp_idx_4 = abs_pos_plus_1 // 4 - 1
    cmp_4_with_skip = torch.where(
        on_b4, cmp_idx_4, torch.full_like(cmp_idx_4, -1)
    ).reshape(-1)
    for at in (CSA_KV, INDEXER_KV):
        E = entries_per_block[at]
        bt = meta.pool_block_tables[at][:bs]
        is_skip = cmp_4_with_skip < 0
        safe_idx = torch.where(
            is_skip, torch.zeros_like(cmp_4_with_skip), cmp_4_with_skip
        )
        block_in_seq = safe_idx // E
        in_block = safe_idx % E
        max_blocks = bt.shape[1]
        safe_bis = block_in_seq.clamp(0, max_blocks - 1)
        req_idx = (
            torch.arange(bs, device=device, dtype=torch.long)
            .unsqueeze(1)
            .expand(-1, q_len)
            .reshape(-1)
        )
        bid = bt.to(torch.long)[req_idx, safe_bis.to(torch.long)]
        valid = (~is_skip) & (bid > 0)
        slot = bid * E + in_block.to(torch.long)
        meta.pool_write_slot_mappings[at][: bs * q_len].copy_(
            torch.where(valid, slot, torch.full_like(slot, -1))
        )

    # HCA (ratio=128)
    on_b128 = (abs_pos_plus_1 % 128) == 0
    cmp_idx_128 = abs_pos_plus_1 // 128 - 1
    cmp_128_with_skip = torch.where(
        on_b128, cmp_idx_128, torch.full_like(cmp_idx_128, -1)
    ).reshape(-1)
    E_hca = entries_per_block[HCA_KV]
    bt_hca = meta.pool_block_tables[HCA_KV][:bs]
    is_skip = cmp_128_with_skip < 0
    safe_idx = torch.where(
        is_skip, torch.zeros_like(cmp_128_with_skip), cmp_128_with_skip
    )
    block_in_seq = safe_idx // E_hca
    in_block = safe_idx % E_hca
    max_blocks = bt_hca.shape[1]
    safe_bis = block_in_seq.clamp(0, max_blocks - 1)
    req_idx = (
        torch.arange(bs, device=device, dtype=torch.long)
        .unsqueeze(1)
        .expand(-1, q_len)
        .reshape(-1)
    )
    bid = bt_hca.to(torch.long)[req_idx, safe_bis.to(torch.long)]
    valid = (~is_skip) & (bid > 0)
    slot = bid * E_hca + in_block.to(torch.long)
    meta.pool_write_slot_mappings[HCA_KV][: bs * q_len].copy_(
        torch.where(valid, slot, torch.full_like(slot, -1))
    )


# ---------------------------------------------------------------------------
# Compare utilities
# ---------------------------------------------------------------------------


def _compare_pure(
    ref_meta: MockMeta, fused_meta: MockMeta, bs: int, q_len: int
) -> Optional[str]:
    T = bs * q_len
    checks = [
        ("start_pos", ref_meta.start_pos[:bs], fused_meta.start_pos[:bs]),
        (
            "slot_mapping_swa",
            ref_meta.slot_mapping_swa[:T],
            fused_meta.slot_mapping_swa[:T],
        ),
        (
            "topk_window_idxs",
            ref_meta.topk_window_idxs[:bs],
            fused_meta.topk_window_idxs[:bs],
        ),
        ("swa_abs_idx", ref_meta.swa_abs_idx[:bs], fused_meta.swa_abs_idx[:bs]),
    ]
    # Only compare ratios present in BOTH metas — SWA-only path has empty dicts.
    common_ratios = sorted(
        set(ref_meta.slot_mapping_compressed.keys())
        & set(fused_meta.slot_mapping_compressed.keys())
    )
    for r in common_ratios:
        checks.append(
            (
                f"slot_cmp[{r}]",
                ref_meta.slot_mapping_compressed[r][:T],
                fused_meta.slot_mapping_compressed[r][:T],
            )
        )
        checks.append(
            (
                f"cmp_lens[{r}]",
                ref_meta.compressed_lens[r][:bs],
                fused_meta.compressed_lens[r][:bs],
            )
        )
        checks.append(
            (
                f"cmp_lens_pt[{r}]",
                ref_meta.compressed_lens_per_token[r][:bs],
                fused_meta.compressed_lens_per_token[r][:bs],
            )
        )
        checks.append(
            (
                f"topk_total[{r}]",
                ref_meta.topk_total_by_ratio[r][:bs],
                fused_meta.topk_total_by_ratio[r][:bs],
            )
        )

    for name, a, b in checks:
        if not torch.equal(a, b):
            n_diff = int((a != b).sum().item())
            return f"{name}: {n_diff}/{a.numel()} diffs"
    return None


def _compare_phase2b(
    ref_meta: MockMeta, fused_meta: MockMeta, bs: int, q_len: int
) -> Optional[str]:
    T = bs * q_len
    SWA_KV, CSA_KV, HCA_KV, INDEXER_KV = 7, 1, 2, 3
    for at, name in [
        (SWA_KV, "SWA"),
        (CSA_KV, "CSA"),
        (INDEXER_KV, "IDX"),
        (HCA_KV, "HCA"),
    ]:
        a = ref_meta.pool_write_slot_mappings[at][:T]
        b = fused_meta.pool_write_slot_mappings[at][:T]
        if not torch.equal(a, b):
            n_diff = int((a != b).sum().item())
            return f"phase2b[{name}]: {n_diff}/{a.numel()} diffs"
    return None


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------


def run_correctness_tests():
    device = "cuda"
    bs_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    q_len_list = [1]
    seqlen_list = [2**i for i in range(0, 21)]  # 1 to 1048576

    passed = 0
    failed = 0
    total = 0

    print("Testing fused_update_decode_meta_pure (q_len=1)...")
    for bs in bs_list:
        for q_len in q_len_list:
            for seqlen in seqlen_list:
                max_seq_len = max(seqlen + 6 + 1, 256)
                max_seq_len = ((max_seq_len + 127) // 128) * 128
                sp_val = min(seqlen, max_seq_len - q_len - 1)
                start_pos = torch.full((bs,), sp_val, dtype=torch.int32, device=device)

                ref_m = _alloc_mock(bs, q_len, max_seq_len, device)
                fused_m = _alloc_mock(bs, q_len, max_seq_len, device)

                ref_update_pure(ref_m, start_pos)
                fused_update_decode_meta_pure(fused_m, start_pos, max_seq_len)

                err = _compare_pure(ref_m, fused_m, bs, q_len)
                total += 1
                if err:
                    print(f"  FAIL bs={bs} q={q_len} seq={seqlen}: {err}")
                    failed += 1
                    if failed >= 10:
                        break
                else:
                    passed += 1
            if failed >= 10:
                break
        if failed >= 10:
            break

    print(f"  Result: {passed}/{total} passed, {failed} failed")

    # q_len > 1 tests (MTP target-verify / speculative decode path)
    print("\nTesting fused_update_decode_meta_pure (q_len=2,4,6)...")
    passed_mq = 0
    failed_mq = 0
    total_mq = 0
    mq_bs_list = [1, 2, 4, 16, 32, 64, 128]
    mq_qlen_list = [2, 4, 6]
    mq_seqlen_list = sorted(
        set(
            [2**i for i in range(2, 18)]
            # ratio=128 boundary starts
            + [126, 127, 128, 129, 254, 255, 256]
            # ratio=4 boundary starts (small values + a few mid-range)
            + [3, 5, 7, 9, 11, 13, 15, 17, 125, 130, 252, 257]
        )
    )

    for bs in mq_bs_list:
        for q_len in mq_qlen_list:
            for seqlen in mq_seqlen_list:
                max_seq_len = max(seqlen + q_len + 1, 256)
                max_seq_len = ((max_seq_len + 127) // 128) * 128
                sp_val = min(seqlen, max_seq_len - q_len - 1)
                start_pos = torch.full(
                    (bs,), sp_val, dtype=torch.int32, device=device
                )

                ref_m = _alloc_mock(bs, q_len, max_seq_len, device)
                fused_m = _alloc_mock(bs, q_len, max_seq_len, device)

                ref_update_pure(ref_m, start_pos)
                fused_update_decode_meta_pure(fused_m, start_pos, max_seq_len)

                err = _compare_pure(ref_m, fused_m, bs, q_len)
                total_mq += 1
                if err:
                    print(f"  FAIL bs={bs} q={q_len} seq={seqlen}: {err}")
                    failed_mq += 1
                    if failed_mq >= 10:
                        break
                else:
                    passed_mq += 1
            if failed_mq >= 10:
                break
        if failed_mq >= 10:
            break

    print(f"  Result: {passed_mq}/{total_mq} passed, {failed_mq} failed")

    # Heterogeneous start_pos per request — every prior loop uses a
    # uniform sp_val across the batch, so per-q HCA causality and the
    # per-token compressed_lens write are only exercised at one phase
    # offset per case. Sprinkle requests across boundaries (just before,
    # on, just after multiples of 4 and 128) so different (r,q) threads
    # hit different ratio-boundary states in the same kernel launch.
    print("\nTesting fused_update_decode_meta_pure (heterogeneous start_pos)...")
    passed_h = 0
    failed_h = 0
    total_h = 0
    het_offsets = torch.tensor(
        [3, 4, 5, 7, 8, 9, 125, 127, 128, 129, 253, 256, 257, 511, 1024, 8191],
        dtype=torch.int32,
        device=device,
    )
    for bs in (4, 16, 64):
        for q_len in (1, 2, 4, 6):
            # Tile/truncate the offset bank to the batch size.
            reps = (bs + het_offsets.numel() - 1) // het_offsets.numel()
            start_pos = het_offsets.repeat(reps)[:bs].clone()
            max_seq_len = int(start_pos.max().item()) + q_len + 16
            max_seq_len = ((max_seq_len + 127) // 128) * 128
            max_seq_len = max(max_seq_len, 256)

            ref_m = _alloc_mock(bs, q_len, max_seq_len, device)
            fused_m = _alloc_mock(bs, q_len, max_seq_len, device)

            ref_update_pure(ref_m, start_pos)
            fused_update_decode_meta_pure(fused_m, start_pos, max_seq_len)

            err = _compare_pure(ref_m, fused_m, bs, q_len)
            total_h += 1
            if err:
                print(f"  FAIL bs={bs} q={q_len} (het): {err}")
                failed_h += 1
            else:
                passed_h += 1

    print(f"  Result: {passed_h}/{total_h} passed, {failed_h} failed")

    # Ratio-subset coverage: SWA-only (MTP draft), CSA/INDEXER-only ({4}),
    # HCA-only ({128}). Each subset exercises a different HAS_CMP_* gate
    # combination in the kernel and the corresponding None-pointer path.
    # The {128}-only subset at small max_seq_len also hits the HCA dense
    # width == 0 boundary (max_seq_len < 128 → topk_total_by_ratio[128]
    # has shape [..., WINDOW]; the kernel must skip the dense write).
    print(
        "\nTesting fused_update_decode_meta_pure (ratio subsets: "
        "(), (4,), (128,))..."
    )
    passed_s = 0
    failed_s = 0
    total_s = 0
    subset_bs_list = [1, 2, 4, 16, 32, 128]
    subset_qlen_list = [1, 2, 4]
    # Cover small seqlens (1-128) to trip HCA dense width == 0 for {128}
    # plus mid + large for general correctness.
    subset_seqlen_list = [
        1, 2, 4, 8, 16, 64, 100, 127, 128, 129, 1024, 4096, 65535, 131072,
    ]
    for ratios in ((), (4,), (128,)):
        label = f"ratios={ratios or '()'}"
        for bs in subset_bs_list:
            for q_len in subset_qlen_list:
                for seqlen in subset_seqlen_list:
                    max_seq_len = max(seqlen + q_len + 1, 64)
                    # 128-align only for the {128} subset (kernel writes
                    # compressed_buffer with stride max_seq_len // 128).
                    if 128 in ratios:
                        max_seq_len = ((max_seq_len + 63) // 64) * 64
                    sp_val = min(seqlen, max_seq_len - q_len - 1)
                    start_pos = torch.full(
                        (bs,), sp_val, dtype=torch.int32, device=device
                    )

                    ref_m = _alloc_mock_subset(
                        bs, q_len, max_seq_len, device, ratios=ratios
                    )
                    fused_m = _alloc_mock_subset(
                        bs, q_len, max_seq_len, device, ratios=ratios
                    )

                    ref_update_pure(ref_m, start_pos)
                    fused_update_decode_meta_pure(fused_m, start_pos, max_seq_len)

                    err = _compare_pure(ref_m, fused_m, bs, q_len)
                    total_s += 1
                    if err:
                        print(
                            f"  FAIL {label} bs={bs} q={q_len} seq={seqlen}: {err}"
                        )
                        failed_s += 1
                        if failed_s >= 10:
                            break
                    else:
                        passed_s += 1
                if failed_s >= 10:
                    break
            if failed_s >= 10:
                break
        if failed_s >= 10:
            break

    print(f"  Result: {passed_s}/{total_s} passed, {failed_s} failed")

    # Targeted: HCA dense width == 0 boundary (max_seq_len < 128 with
    # ratios=(128,)). Without the kernel's ``if HCA_DENSE_WIDTH > 0`` guard
    # the dense-half store overruns ``topk_total_by_ratio[128]`` (shape
    # [..., WINDOW + 0]). Asserts: (a) no crash/OOB, (b) the window half
    # equals the SWA-window indices, (c) all the compressed-length
    # writes report 0 because abs_pos+1 < 128.
    print("\nTesting fused_update_decode_meta_pure (HCA dense width==0 boundary)...")
    passed_bnd = 0
    failed_bnd = 0
    for bs, q_len, max_seq_len_bnd, sp_val in [
        (1, 1, 64, 5),
        (4, 2, 64, 60),    # abs_pos hits {60..61}, still < 128
        (8, 4, 96, 90),    # abs_pos {90..93}, still < 128
        (32, 1, 64, 0),    # smallest viable max_seq_len
    ]:
        start_pos = torch.full((bs,), sp_val, dtype=torch.int32, device=device)
        m = _alloc_mock_subset(bs, q_len, max_seq_len_bnd, device, ratios=(128,))
        assert m.topk_total_by_ratio[128].shape[-1] == _WINDOW, (
            "Test setup invariant: HCA dense half must be empty for this case"
        )
        fused_update_decode_meta_pure(m, start_pos, max_seq_len_bnd)
        torch.cuda.synchronize()

        # All compressed-len writes are 0 (no full 128-block formed yet).
        ok = (
            int(m.compressed_lens[128].max()) == 0
            and int(m.compressed_lens_per_token[128].max()) == 0
            and int(m.slot_mapping_compressed[128].max()) == -1
        )
        if ok:
            passed_bnd += 1
        else:
            print(
                f"  FAIL bs={bs} q={q_len} max_s={max_seq_len_bnd} sp={sp_val}: "
                f"cmp_lens={m.compressed_lens[128][:bs].tolist()}, "
                f"slot={m.slot_mapping_compressed[128][:bs * q_len].tolist()}"
            )
            failed_bnd += 1

    print(f"  Result: {passed_bnd}/{passed_bnd + failed_bnd} passed, {failed_bnd} failed")

    # Phase 2b tests (q_len=1 and q_len>1)
    passed2 = 0
    failed2 = 0
    total2 = 0
    SWA_KV, CSA_KV, HCA_KV, INDEXER_KV = 7, 1, 2, 3
    entries = {SWA_KV: 256, CSA_KV: 64, INDEXER_KV: 64, HCA_KV: 2}

    all_qlen_list = [1, 2, 4, 6]
    p2b_bs_list = [1, 2, 4, 16, 32, 64, 128]
    p2b_seqlen_list = [2**i for i in range(2, 18)]  # 4 to 131072

    print("\nTesting fused_phase2b_pool_slot_mapping (q_len=1,2,4,6)...")
    for bs in p2b_bs_list:
        for q_len in all_qlen_list:
            for seqlen in p2b_seqlen_list:
                max_seq_len = max(seqlen + q_len + 1, 256)
                max_seq_len = ((max_seq_len + 127) // 128) * 128
                sp_val = min(seqlen, max_seq_len - q_len - 1)
                start_pos = torch.full(
                    (bs,), sp_val, dtype=torch.int32, device=device
                )

                ref_m = _alloc_mock(bs, q_len, max_seq_len, device)
                fused_m = _alloc_mock(bs, q_len, max_seq_len, device)
                _alloc_phase2b(ref_m, bs, device, seqlen=sp_val + q_len)
                for at in (SWA_KV, CSA_KV, HCA_KV, INDEXER_KV):
                    fused_m.pool_block_tables[at] = ref_m.pool_block_tables[
                        at
                    ].clone()
                    fused_m.pool_write_slot_mappings[at] = torch.full_like(
                        ref_m.pool_write_slot_mappings[at], -1
                    )

                ref_phase2b(ref_m, start_pos, bs, entries)
                fused_phase2b_pool_slot_mapping(fused_m, start_pos, bs, entries)

                err = _compare_phase2b(ref_m, fused_m, bs, q_len)
                total2 += 1
                if err:
                    print(f"  FAIL bs={bs} q={q_len} seq={seqlen}: {err}")
                    failed2 += 1
                    if failed2 >= 10:
                        break
                else:
                    passed2 += 1
            if failed2 >= 10:
                break
        if failed2 >= 10:
            break

    print(f"  Result: {passed2}/{total2} passed, {failed2} failed")
    return failed + failed_mq + failed_h + failed_s + failed_bnd + failed2


def _benchmark(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iters


def run_benchmarks():
    device = "cuda"
    print("\n=== Benchmark: fused_update_decode_meta_pure ===")
    print(
        f"{'bs':>6} {'q_len':>6} {'seqlen':>10} | {'PyTorch ms':>12} {'Triton ms':>12} {'Speedup':>8}"
    )
    print("-" * 70)

    configs = [
        (1, 1, 1024),
        (16, 1, 4096),
        (64, 1, 65536),
        (128, 1, 65536),
        (256, 1, 131072),
        (64, 6, 65536),
        (128, 6, 65536),
    ]

    for bs, q_len, seqlen in configs:
        max_seq_len = ((seqlen + 127) // 128) * 128
        sp_val = min(seqlen - 1, max_seq_len - q_len - 1)
        start_pos = torch.full((bs,), sp_val, dtype=torch.int32, device=device)

        ref_m = _alloc_mock(bs, q_len, max_seq_len, device)
        fused_m = _alloc_mock(bs, q_len, max_seq_len, device)

        t_ref = _benchmark(lambda: ref_update_pure(ref_m, start_pos))
        t_fused = _benchmark(
            lambda: fused_update_decode_meta_pure(fused_m, start_pos, max_seq_len)
        )
        speedup = t_ref / t_fused if t_fused > 0 else float("inf")
        print(
            f"{bs:>6} {q_len:>6} {seqlen:>10} | {t_ref:>10.3f}ms {t_fused:>10.3f}ms {speedup:>7.2f}x"
        )

    print("\n=== Benchmark: fused_phase2b_pool_slot_mapping ===")
    print(
        f"{'bs':>6} {'q_len':>6} {'seqlen':>10} | {'PyTorch ms':>12} {'Triton ms':>12} {'Speedup':>8}"
    )
    print("-" * 70)

    SWA_KV, CSA_KV, HCA_KV, INDEXER_KV = 7, 1, 2, 3
    entries = {SWA_KV: 256, CSA_KV: 64, INDEXER_KV: 64, HCA_KV: 2}

    for bs, q_len, seqlen in configs:
        max_seq_len = ((seqlen + 127) // 128) * 128
        sp_val = min(seqlen - 1, max_seq_len - q_len - 1)
        start_pos = torch.full((bs,), sp_val, dtype=torch.int32, device=device)

        ref_m = _alloc_mock(bs, q_len, max_seq_len, device)
        fused_m = _alloc_mock(bs, q_len, max_seq_len, device)
        _alloc_phase2b(ref_m, bs, device, seqlen=sp_val + q_len)
        for at in (SWA_KV, CSA_KV, INDEXER_KV, HCA_KV):
            fused_m.pool_block_tables[at] = ref_m.pool_block_tables[at].clone()
            fused_m.pool_write_slot_mappings[at] = torch.full_like(
                ref_m.pool_write_slot_mappings[at], -1
            )

        t_ref = _benchmark(lambda: ref_phase2b(ref_m, start_pos, bs, entries))
        t_fused = _benchmark(
            lambda: fused_phase2b_pool_slot_mapping(fused_m, start_pos, bs, entries)
        )
        speedup = t_ref / t_fused if t_fused > 0 else float("inf")
        print(
            f"{bs:>6} {q_len:>6} {seqlen:>10} | {t_ref:>10.3f}ms {t_fused:>10.3f}ms {speedup:>7.2f}x"
        )


if __name__ == "__main__":
    total_failures = run_correctness_tests()
    if total_failures == 0:
        print("\nAll correctness tests passed!")
        run_benchmarks()
    else:
        print(f"\n{total_failures} failures — skipping benchmarks")
        sys.exit(1)
