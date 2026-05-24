"""UT: numerical equivalence of fused Triton prepare vs local reference.

Compares every persistent buffer written by
``update_decode_metadata_in_place_fp8`` against a local Python reference.
Production update intentionally has no eager fallback; reference math lives
in this UT.

Dev-box note: ``rtp_llm/models_py/modules/__init__.py`` pulls in the
whole framework (compiled ops + base layers). This UT only needs the
two pure-Python files under ``dsv4/fp8/decode``, so it loads them via
``importlib`` directly from disk — same code, no package init chain.

Run (bypassing package init):
  CUDA_VISIBLE_DEVICES=0 /opt/conda310/bin/python3 \\
    rtp_llm/models_py/modules/dsv4/fp8/test/test_fused_prepare_meta_triton.py
"""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_DECODE_DIR = _THIS_DIR.parent / "decode"


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ``_fused_prepare_meta_triton`` is imported lazily from inside
# ``update_decode_metadata_in_place_fp8`` — pre-register it so that
# import resolves to our direct load instead of traversing the package.
_attn_type_mod = _load_module(
    "rtp_llm.models_py.modules.dsv4.attn_type",
    _THIS_DIR.parent.parent / "attn_type.py",
)
_fused_mod = _load_module(
    "rtp_llm.models_py.modules.dsv4.fp8.decode._fused_prepare_meta_triton",
    _DECODE_DIR / "_fused_prepare_meta_triton.py",
)
# pool_slot_mapping imports attn_type from the package path;
# pre-register it via the loaded attn_type module above so the import
# resolves without triggering the full framework __init__ chain.
_pool_mod = _load_module(
    "rtp_llm.models_py.modules.dsv4.fp8.decode.pool_slot_mapping",
    _DECODE_DIR / "pool_slot_mapping.py",
)
_translator_mod = _load_module(
    "rtp_llm.models_py.modules.dsv4.fp8.decode.paged_topk_translator",
    _DECODE_DIR / "paged_topk_translator.py",
)
_meta_mod = _load_module(
    "rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata",
    _DECODE_DIR / "decode_attn_metadata.py",
)

allocate_decode_metadata_fp8 = _meta_mod.allocate_decode_metadata_fp8
update_decode_metadata_in_place_fp8 = _meta_mod.update_decode_metadata_in_place_fp8


# V4-Flash config: ratios {4, 128}, window=128, index_topk=512.
_V4_COMPRESS_RATIOS = [0, 0] + [4, 128] * 20 + [4, 0]
_SWA_ONLY_COMPRESS_RATIOS = [0]
_WINDOW = 128
_HEAD_DIM = 512
_INDEX_TOPK = 512

SWA_KV = _attn_type_mod.SWA_KV
CSA_KV = _attn_type_mod.CSA_KV
HCA_KV = _attn_type_mod.HCA_KV
INDEXER_KV = _attn_type_mod.INDEXER_KV


def _alloc(
    max_bs: int,
    q_len: int,
    max_seq_len: int,
    device: torch.device,
    compress_ratios=None,
):
    return allocate_decode_metadata_fp8(
        max_batch_size=max_bs,
        q_len=q_len,
        window_size=_WINDOW,
        head_dim=_HEAD_DIM,
        max_seq_len=max_seq_len,
        compress_ratios=compress_ratios or _V4_COMPRESS_RATIOS,
        index_topk=_INDEX_TOPK,
        device=device,
    )


def _ref_position_ids_2d(start_pos, q_len):
    B = int(start_pos.shape[0])
    return start_pos.view(B, 1) + torch.arange(
        q_len, device=start_pos.device, dtype=torch.int32
    ).view(1, q_len)


def _ref_swa_slot_mapping(position_ids_2d, window_size):
    B = int(position_ids_2d.shape[0])
    req_base = (
        torch.arange(B, device=position_ids_2d.device, dtype=torch.int32).view(B, 1)
        * window_size
    )
    return (req_base + (position_ids_2d % window_size)).reshape(-1)


def _ref_window_topk_idxs(position_ids_2d, window_size):
    k_range = torch.arange(
        window_size, device=position_ids_2d.device, dtype=torch.int32
    ).view(1, 1, window_size)
    abs_pos_b = position_ids_2d.unsqueeze(-1)
    sp = (position_ids_2d % window_size).unsqueeze(-1)
    ring_full_idx = (sp + 1 + k_range) % window_size
    partial_idx = torch.where(
        k_range <= abs_pos_b, k_range, torch.full_like(k_range, -1)
    )
    return torch.where(abs_pos_b >= (window_size - 1), ring_full_idx, partial_idx)


def _ref_state_pool_slot_mapping(block_table, positions, req_idx, entries_per_block):
    pos_i64 = positions.to(torch.long)
    req_i64 = req_idx.to(torch.long)
    bt_long = block_table.to(torch.long)
    max_blocks = int(bt_long.shape[1])
    block_in_seq = (pos_i64 // entries_per_block) % max_blocks
    in_block = pos_i64 % entries_per_block
    block_id = bt_long[req_i64, block_in_seq]
    slot = block_id * entries_per_block + in_block
    return torch.where(block_id > 0, slot, torch.full_like(slot, -1))


def _reference_update_decode_metadata_in_place(
    meta,
    start_pos,
    paged_block_tables=None,
    paged_pool_entries_per_block=None,
):
    q_len = meta.q_len_per_req
    window_size = meta.window_size
    bs = int(start_pos.shape[0])
    T = bs * q_len

    if start_pos.device != meta.start_pos.device:
        start_pos = start_pos.to(meta.start_pos.device)
    if start_pos.dtype != torch.int32:
        start_pos = start_pos.to(torch.int32)
    start_pos = torch.clamp(
        start_pos, min=0, max=max(0, int(meta.max_seq_len) - int(q_len))
    )

    position_ids_2d = _ref_position_ids_2d(start_pos, q_len)
    position_ids_flat = position_ids_2d.reshape(-1).contiguous()

    meta.start_pos[:bs].copy_(start_pos)
    meta.position_ids[:T].copy_(position_ids_flat)
    if meta.position_ids_long is not None:
        meta.position_ids_long[:T].copy_(position_ids_flat.to(torch.long))
    if meta.decode_seq_start_per_req is not None:
        meta.decode_seq_start_per_req[:bs].copy_(start_pos)
    if meta.cache_seqlens_i32 is not None:
        meta.cache_seqlens_i32[:bs].copy_(position_ids_2d[:, -1] + 1)
    meta.topk_buffer_compressed[:bs].fill_(-1)

    meta.slot_mapping_swa[:T].copy_(
        _ref_swa_slot_mapping(position_ids_2d, window_size)
    )
    window_idxs = _ref_window_topk_idxs(position_ids_2d, window_size)
    meta.topk_window_idxs[:bs].copy_(window_idxs)

    for r, slot_t in meta.slot_mapping_compressed.items():
        stride = meta.compressed_buffer_t_dim_per_ratio[r]
        abs_pos_plus_1 = position_ids_2d + 1
        on_boundary = (abs_pos_plus_1 % r) == 0
        in_req = abs_pos_plus_1 // r - 1
        cmp_lens_per_token = (abs_pos_plus_1 // r).to(torch.int32)
        cmp_req_base = (
            torch.arange(bs, device=start_pos.device, dtype=torch.int32).view(bs, 1)
            * stride
        )
        flat = (cmp_req_base + in_req).reshape(-1)
        slot_t[:T].copy_(
            torch.where(on_boundary.reshape(-1), flat, torch.full_like(flat, -1))
        )
        meta.compressed_lens_per_token[r][:bs].copy_(cmp_lens_per_token)
        meta.compressed_lens[r][:bs].copy_(cmp_lens_per_token[:, -1])

        total = meta.topk_total_by_ratio[r]
        total[:bs, :, :window_size].copy_(window_idxs)
        if r == 4:
            total[:bs, :, window_size:].fill_(-1)
        else:
            K_dense = total.shape[-1] - window_size
            dense_idxs = (
                torch.arange(K_dense, device=start_pos.device, dtype=torch.int32)
                .view(1, 1, K_dense)
                .expand(bs, q_len, K_dense)
            )
            valid = dense_idxs < cmp_lens_per_token.view(bs, q_len, 1)
            total[:bs, :, window_size:].copy_(
                torch.where(valid, dense_idxs, torch.full_like(dense_idxs, -1))
            )

    if meta.swa_abs_idx is not None:
        win_range = torch.arange(
            window_size, device=start_pos.device, dtype=torch.int32
        ).view(1, 1, window_size)
        win_start = (position_ids_2d.unsqueeze(-1) - window_size + 1).clamp(min=0)
        candidate = win_start + win_range
        valid_pos = candidate <= position_ids_2d.unsqueeze(-1)
        meta.swa_abs_idx[:bs].copy_(
            torch.where(valid_pos, candidate, torch.full_like(candidate, -1))
        )

    if paged_block_tables is not None and paged_pool_entries_per_block is not None:
        for at, src_bt in paged_block_tables.items():
            dst_bt = meta.pool_block_tables.get(at)
            if dst_bt is None:
                continue
            n_rows = min(src_bt.shape[0], dst_bt.shape[0])
            n_cols = min(src_bt.shape[1], dst_bt.shape[1])
            dst_bt[bs:].zero_()
            dst_bt[:n_rows, :n_cols].copy_(src_bt[:n_rows, :n_cols])
            if n_cols < dst_bt.shape[1]:
                dst_bt[:n_rows, n_cols:].zero_()

        if SWA_KV in meta.pool_block_tables:
            E = paged_pool_entries_per_block.get(SWA_KV, window_size)
            mapped = _pool_mod.compute_kv_pool_slot_mapping(
                meta.pool_block_tables[SWA_KV][:bs],
                position_ids_flat,
                E,
            )
            meta.pool_write_slot_mappings[SWA_KV][:T].copy_(mapped)

        for ratio_key, attn_type_writers in (
            (4, [CSA_KV, INDEXER_KV]),
            (128, [HCA_KV]),
        ):
            if ratio_key not in meta.slot_mapping_compressed:
                continue
            abs_pos_plus_1 = position_ids_2d + 1
            on_boundary = (abs_pos_plus_1 % ratio_key) == 0
            cmp_idx = abs_pos_plus_1 // ratio_key - 1
            cmp_idx_with_skip = torch.where(
                on_boundary, cmp_idx, torch.full_like(cmp_idx, -1)
            ).reshape(-1)
            for at in attn_type_writers:
                if at not in meta.pool_block_tables:
                    continue
                E = paged_pool_entries_per_block.get(at, 1)
                mapped = _pool_mod.compute_kv_pool_slot_mapping(
                    meta.pool_block_tables[at][:bs],
                    cmp_idx_with_skip,
                    E,
                )
                meta.pool_write_slot_mappings[at][:T].copy_(mapped)

        if meta.req_id_per_token is not None and meta.swa_global_slots is not None:
            req_id_bs = meta.req_id_per_token[:T]
            if SWA_KV in meta.pool_block_tables and meta.swa_abs_idx is not None:
                swa_eb = paged_pool_entries_per_block.get(SWA_KV, window_size)
                swa_local = meta.swa_abs_idx[:bs].reshape(T, window_size)
                meta.swa_global_slots[:T].copy_(
                    _translator_mod.translate_local_to_global_slots(
                        req_id_bs, meta.pool_block_tables[SWA_KV][:bs], swa_local, swa_eb
                    )
                )
            if HCA_KV in meta.pool_block_tables and 128 in meta.topk_total_by_ratio:
                hca_eb = paged_pool_entries_per_block.get(HCA_KV, 1)
                hca_tt = meta.topk_total_by_ratio[128]
                K_h = hca_tt.shape[-1] - window_size
                hca_local = hca_tt[:bs, :, window_size:].reshape(T, K_h).contiguous()
                meta.hca_cmp_global_slots[:T].copy_(
                    _translator_mod.translate_local_to_global_slots(
                        req_id_bs, meta.pool_block_tables[HCA_KV][:bs], hca_local, hca_eb
                    )
                )

        if meta.compressor_state_slot_mappings:
            positions = meta.position_ids_long[:T]
            req_idx = meta.req_id_per_token_long[:T]
            for at, out in meta.compressor_state_slot_mappings.items():
                if at not in meta.pool_block_tables:
                    continue
                E = paged_pool_entries_per_block.get(at, 1)
                out[:T].copy_(
                    _ref_state_pool_slot_mapping(
                        meta.pool_block_tables[at][:bs], positions, req_idx, E
                    )
                )

    meta.batch_size = bs
    meta.total_tokens = T


def _run_update(meta, start_pos, fused: bool):
    if fused:
        update_decode_metadata_in_place_fp8(meta, start_pos)
    else:
        _reference_update_decode_metadata_in_place(meta, start_pos)


def _assert_equal(tc, name, a, b):
    if torch.equal(a, b):
        return
    diff_mask = a != b
    n_diff = int(diff_mask.sum().item())
    idx = torch.nonzero(diff_mask.reshape(-1), as_tuple=False).reshape(-1)[:8]
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    samples = [
        f"  [{int(i.item())}] py={int(flat_a[i].item())} fused={int(flat_b[i].item())}"
        for i in idx
    ]
    tc.fail(
        f"mismatch on '{name}': {n_diff}/{a.numel()} elements differ.\n"
        + "\n".join(samples)
    )


def _compare(tc, meta_py, meta_fu, bs: int, q_len: int):
    T = bs * q_len
    _assert_equal(tc, "start_pos", meta_py.start_pos[:bs], meta_fu.start_pos[:bs])
    _assert_equal(
        tc,
        "cache_seqlens_i32",
        meta_py.cache_seqlens_i32[:bs],
        meta_fu.cache_seqlens_i32[:bs],
    )
    _assert_equal(
        tc,
        "position_ids",
        meta_py.position_ids[:T],
        meta_fu.position_ids[:T],
    )
    _assert_equal(
        tc,
        "position_ids_long",
        meta_py.position_ids_long[:T],
        meta_fu.position_ids_long[:T],
    )
    _assert_equal(
        tc,
        "decode_seq_start_per_req",
        meta_py.decode_seq_start_per_req[:bs],
        meta_fu.decode_seq_start_per_req[:bs],
    )
    _assert_equal(
        tc,
        "slot_mapping_swa",
        meta_py.slot_mapping_swa[:T],
        meta_fu.slot_mapping_swa[:T],
    )
    for r in sorted(meta_py.slot_mapping_compressed):
        _assert_equal(
            tc,
            f"slot_mapping_compressed[{r}]",
            meta_py.slot_mapping_compressed[r][:T],
            meta_fu.slot_mapping_compressed[r][:T],
        )
        _assert_equal(
            tc,
            f"compressed_lens[{r}]",
            meta_py.compressed_lens[r][:bs],
            meta_fu.compressed_lens[r][:bs],
        )
        _assert_equal(
            tc,
            f"compressed_lens_per_token[{r}]",
            meta_py.compressed_lens_per_token[r][:bs],
            meta_fu.compressed_lens_per_token[r][:bs],
        )
        _assert_equal(
            tc,
            f"topk_total_by_ratio[{r}]",
            meta_py.topk_total_by_ratio[r][:bs],
            meta_fu.topk_total_by_ratio[r][:bs],
        )
    _assert_equal(
        tc,
        "topk_window_idxs",
        meta_py.topk_window_idxs[:bs],
        meta_fu.topk_window_idxs[:bs],
    )
    _assert_equal(
        tc,
        "swa_abs_idx",
        meta_py.swa_abs_idx[:bs],
        meta_fu.swa_abs_idx[:bs],
    )
    _assert_equal(
        tc,
        "topk_buffer_compressed",
        meta_py.topk_buffer_compressed[:bs],
        meta_fu.topk_buffer_compressed[:bs],
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FusedPrepareMetaEquivalenceTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda:0")

    def _run_case(
        self,
        bs: int,
        q_len: int,
        max_seq_len: int,
        start_pos_values,
        compress_ratios=None,
    ):
        meta_py = _alloc(bs, q_len, max_seq_len, self.device, compress_ratios)
        meta_fu = _alloc(bs, q_len, max_seq_len, self.device, compress_ratios)

        start_pos = torch.tensor(
            start_pos_values, dtype=torch.int32, device=self.device
        )
        self.assertEqual(start_pos.shape[0], bs)

        _run_update(meta_py, start_pos, fused=False)
        _run_update(meta_fu, start_pos, fused=True)

        _compare(self, meta_py, meta_fu, bs, q_len)

    def test_prepare_kind_full_v4(self):
        meta = _alloc(1, 1, 4096, self.device)
        self.assertEqual(
            _fused_mod.fused_prepare_kind(meta),
            _fused_mod.FUSED_PREPARE_FULL_V4,
        )

    def test_prepare_kind_swa_only(self):
        meta = _alloc(1, 1, 4096, self.device, _SWA_ONLY_COMPRESS_RATIOS)
        self.assertEqual(
            _fused_mod.fused_prepare_kind(meta),
            _fused_mod.FUSED_PREPARE_SWA_ONLY,
        )

    def test_prepare_kind_rejects_third_shape(self):
        meta = _alloc(1, 1, 4096, self.device, [4])
        with self.assertRaisesRegex(RuntimeError, "SWA-only or ratios \\{4, 128\\}"):
            update_decode_metadata_in_place_fp8(
                meta, torch.tensor([0], dtype=torch.int32, device=self.device)
            )

    def test_early_decode_all_zero(self):
        self._run_case(bs=4, q_len=1, max_seq_len=4096, start_pos_values=[0, 0, 0, 0])

    def test_small_batch_mixed_positions(self):
        self._run_case(
            bs=8,
            q_len=1,
            max_seq_len=4096,
            start_pos_values=[0, 1, 63, 127, 128, 511, 4095, 12287],
        )

    def test_bs1_large_pos(self):
        self._run_case(bs=1, q_len=1, max_seq_len=65536, start_pos_values=[60000])

    def test_bs128_randomised(self):
        torch.manual_seed(0)
        bs = 128
        max_seq_len = 65664
        vals = torch.randint(0, max_seq_len - 1, (bs,), dtype=torch.int32).tolist()
        vals[0] = 0
        vals[1] = 1
        vals[2] = 127
        vals[3] = 128
        vals[4] = 511
        vals[5] = max_seq_len - 2
        self._run_case(bs=bs, q_len=1, max_seq_len=max_seq_len, start_pos_values=vals)

    def test_intermediate_batch_sizes(self):
        max_seq_len = 8192
        for bs in (2, 16, 32, 64):
            vals = ((torch.arange(bs, dtype=torch.int32) * 37) % (max_seq_len - 1)).tolist()
            vals[0] = 0
            vals[min(1, bs - 1)] = 127
            vals[min(2, bs - 1)] = 255
            vals[-1] = max_seq_len - 2
            with self.subTest(bs=bs):
                self._run_case(
                    bs=bs,
                    q_len=1,
                    max_seq_len=max_seq_len,
                    start_pos_values=vals,
                )

    def test_qlen_2(self):
        self._run_case(
            bs=4, q_len=2, max_seq_len=4096, start_pos_values=[0, 127, 511, 2048]
        )

    def test_qlen_4(self):
        self._run_case(
            bs=4, q_len=4, max_seq_len=4096, start_pos_values=[0, 126, 511, 2048]
        )

    def test_swa_only_qlen_1(self):
        self._run_case(
            bs=4,
            q_len=1,
            max_seq_len=4096,
            start_pos_values=[0, 1, 127, 2048],
            compress_ratios=_SWA_ONLY_COMPRESS_RATIOS,
        )

    def test_swa_only_qlen_4_cross_window(self):
        self._run_case(
            bs=4,
            q_len=4,
            max_seq_len=4096,
            start_pos_values=[0, 126, 511, 2048],
            compress_ratios=_SWA_ONLY_COMPRESS_RATIOS,
        )


# ---------------------------------------------------------------------------
# Phase 2b: fused pool_write_slot_mapping equivalence.
# ---------------------------------------------------------------------------
# The base suite above runs ``update_decode_metadata_in_place_fp8`` WITHOUT
# ``paged_block_tables``, so phase2b is skipped.  This suite wires up the
# paged-pool path: it allocates with ``paged_pool_specs`` so that
# ``meta.pool_block_tables`` / ``pool_write_slot_mappings`` exist, seeds
# the block tables with random valid block ids, then compares the fused
# Triton kernel's outputs to the Python ``compute_kv_pool_slot_mapping``
# loop.

# Pool specs mirror V4-Flash runtime geometry enough for numerical test:
#   SWA:     E=256, max_blocks per req fits a single window
#   CSA:     E=64,  ratio=4
#   INDEXER: E=64,  ratio=4 (shares CSA's boundary)
#   HCA:     E=2,   ratio=128
_SWA_E = 256
_CSA_E = 64
_IDX_E = 64
_HCA_E = 2


def _alloc_with_paged_pools(
    max_bs: int,
    q_len: int,
    max_seq_len: int,
    device: torch.device,
    compress_ratios=None,
):
    # max_blocks_per_req per pool — pick with headroom so clamp isn't
    # excercised on valid positions in the tests below.
    ratios = compress_ratios or _V4_COMPRESS_RATIOS
    positive_ratios = {r for r in ratios if r > 1}
    paged_pool_specs = {SWA_KV: (_SWA_E, max(1, (_WINDOW + _SWA_E - 1) // _SWA_E))}
    if positive_ratios == {4, 128}:
        paged_pool_specs.update(
            {
                CSA_KV: (_CSA_E, max(1, (max_seq_len // 4 + _CSA_E - 1) // _CSA_E)),
                INDEXER_KV: (
                    _IDX_E,
                    max(1, (max_seq_len // 4 + _IDX_E - 1) // _IDX_E),
                ),
                HCA_KV: (
                    _HCA_E,
                    max(1, (max_seq_len // 128 + _HCA_E - 1) // _HCA_E),
                ),
            }
        )
    return allocate_decode_metadata_fp8(
        max_batch_size=max_bs,
        q_len=q_len,
        window_size=_WINDOW,
        head_dim=_HEAD_DIM,
        max_seq_len=max_seq_len,
        compress_ratios=ratios,
        index_topk=_INDEX_TOPK,
        device=device,
        paged_pool_specs=paged_pool_specs,
    )


def _seed_block_tables(meta, seed: int = 0):
    """Fill meta.pool_block_tables with deterministic non-zero block ids.
    Returns (paged_block_tables_arg, paged_pool_entries_per_block_arg).
    """
    gen = torch.Generator(device=meta.pool_block_tables[SWA_KV].device).manual_seed(
        seed
    )
    base_by_attn_type = {
        SWA_KV: 1000,
        CSA_KV: 2000,
        INDEXER_KV: 3000,
        HCA_KV: 4000,
    }
    entries_by_attn_type = {
        SWA_KV: _SWA_E,
        CSA_KV: _CSA_E,
        INDEXER_KV: _IDX_E,
        HCA_KV: _HCA_E,
    }
    for at, bt in meta.pool_block_tables.items():
        # Distinct ranges per pool so cross-pool aliasing bugs surface.
        base = base_by_attn_type[at]
        rand = torch.randint(
            0, 10000, bt.shape, generator=gen, dtype=torch.int32, device=bt.device
        )
        bt.copy_(rand + base)

    paged_pool_entries_per_block = {
        at: entries_by_attn_type[at] for at in meta.pool_block_tables
    }
    # paged_block_tables arg is the same tensor reference — caller
    # would pass the framework's live block table; for the test the
    # pre-seeded meta buffers are fine.
    paged_block_tables = dict(meta.pool_block_tables)
    return paged_block_tables, paged_pool_entries_per_block


def _run_update_paged(meta, start_pos, paged_bt, paged_e, fused: bool):
    if fused:
        update_decode_metadata_in_place_fp8(
            meta,
            start_pos,
            paged_block_tables=paged_bt,
            paged_pool_entries_per_block=paged_e,
        )
    else:
        _reference_update_decode_metadata_in_place(
            meta,
            start_pos,
            paged_block_tables=paged_bt,
            paged_pool_entries_per_block=paged_e,
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class FusedPhase2bEquivalenceTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda:0")

    def _run_case(
        self,
        bs: int,
        q_len: int,
        max_seq_len: int,
        start_pos_values,
        compress_ratios=None,
    ):
        meta_py = _alloc_with_paged_pools(
            bs, q_len, max_seq_len, self.device, compress_ratios
        )
        meta_fu = _alloc_with_paged_pools(
            bs, q_len, max_seq_len, self.device, compress_ratios
        )

        paged_bt_py, paged_e = _seed_block_tables(meta_py, seed=42)
        paged_bt_fu, _ = _seed_block_tables(meta_fu, seed=42)

        start_pos = torch.tensor(
            start_pos_values, dtype=torch.int32, device=self.device
        )
        self.assertEqual(start_pos.shape[0], bs)

        _run_update_paged(meta_py, start_pos, paged_bt_py, paged_e, fused=False)
        _run_update_paged(meta_fu, start_pos, paged_bt_fu, paged_e, fused=True)

        T = bs * q_len
        labels = {
            SWA_KV: "SWA",
            CSA_KV: "CSA",
            INDEXER_KV: "INDEXER",
            HCA_KV: "HCA",
        }
        for at, slot_py in meta_py.pool_write_slot_mappings.items():
            _assert_equal(
                self,
                f"pool_write_slot_mappings[{labels[at]}]",
                slot_py[:T],
                meta_fu.pool_write_slot_mappings[at][:T],
            )

        if meta_py.swa_global_slots is not None:
            _assert_equal(
                self,
                "swa_global_slots",
                meta_py.swa_global_slots[:T],
                meta_fu.swa_global_slots[:T],
            )
        if meta_py.hca_cmp_global_slots is not None:
            _assert_equal(
                self,
                "hca_cmp_global_slots",
                meta_py.hca_cmp_global_slots[:T],
                meta_fu.hca_cmp_global_slots[:T],
            )
        for at in meta_py.compressor_state_slot_mappings:
            _assert_equal(
                self,
                f"compressor_state_slot_mappings[{at}]",
                meta_py.compressor_state_slot_mappings[at][:T],
                meta_fu.compressor_state_slot_mappings[at][:T],
            )

    def test_phase2b_small_batch(self):
        self._run_case(
            bs=4, q_len=1, max_seq_len=4096, start_pos_values=[0, 3, 127, 2048]
        )

    def test_phase2b_boundary_cases(self):
        # start_pos values that hit both on-boundary and off-boundary cases
        # for CSA (ratio=4) and HCA (ratio=128).
        self._run_case(
            bs=8,
            q_len=1,
            max_seq_len=4096,
            start_pos_values=[0, 1, 2, 3, 127, 255, 511, 1023],
        )

    def test_phase2b_bs128(self):
        torch.manual_seed(0)
        bs = 128
        max_seq_len = 65664
        vals = torch.randint(0, max_seq_len - 1, (bs,), dtype=torch.int32).tolist()
        vals[0] = 0
        vals[1] = 3  # just before ratio=4 boundary
        vals[2] = 127  # just before ratio=128 boundary
        vals[3] = 255
        self._run_case(bs=bs, q_len=1, max_seq_len=max_seq_len, start_pos_values=vals)

    def test_phase2b_intermediate_batch_sizes(self):
        max_seq_len = 8192
        for bs in (2, 16, 32, 64):
            vals = ((torch.arange(bs, dtype=torch.int32) * 41) % (max_seq_len - 1)).tolist()
            vals[0] = 0
            vals[min(1, bs - 1)] = 3
            vals[min(2, bs - 1)] = 127
            vals[min(3, bs - 1)] = 255
            vals[-1] = max_seq_len - 2
            with self.subTest(bs=bs):
                self._run_case(
                    bs=bs,
                    q_len=1,
                    max_seq_len=max_seq_len,
                    start_pos_values=vals,
                )

    def test_phase2b_qlen_2(self):
        self._run_case(
            bs=4,
            q_len=2,
            max_seq_len=4096,
            start_pos_values=[0, 127, 511, 2048],
        )

    def test_phase2b_qlen_4(self):
        self._run_case(
            bs=4,
            q_len=4,
            max_seq_len=4096,
            start_pos_values=[0, 126, 511, 2048],
        )

    def test_phase2b_swa_only_qlen_1(self):
        self._run_case(
            bs=4,
            q_len=1,
            max_seq_len=4096,
            start_pos_values=[0, 1, 127, 2048],
            compress_ratios=_SWA_ONLY_COMPRESS_RATIOS,
        )

    def test_phase2b_swa_only_qlen_4(self):
        self._run_case(
            bs=4,
            q_len=4,
            max_seq_len=4096,
            start_pos_values=[0, 126, 511, 2048],
            compress_ratios=_SWA_ONLY_COMPRESS_RATIOS,
        )


if __name__ == "__main__":
    unittest.main()
