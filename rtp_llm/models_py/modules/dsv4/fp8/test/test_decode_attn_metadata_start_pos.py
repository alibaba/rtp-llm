"""CUDA tests for DSv4 FP8 decode metadata start-position selection."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.models_py.modules.dsv4.attn_type import (
    CSA_KV,
    CSA_STATE,
    HCA_KV,
    HCA_STATE,
    INDEXER_KV,
    INDEXER_STATE,
    SWA_KV,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.paged_topk_translator import (
    translate_local_to_global_slots,
)
from rtp_llm.models_py.modules.dsv4.fp8.decode.pool_slot_mapping import (
    compute_kv_pool_slot_mapping,
)

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


_meta_mod = _load_module(
    "dsv4_fp8_decode_attn_metadata_under_test",
    _DECODE_DIR / "decode_attn_metadata.py",
)

allocate_decode_metadata_fp8 = _meta_mod.allocate_decode_metadata_fp8
build_decode_metadata_fp8 = _meta_mod.build_decode_metadata_fp8
update_decode_metadata_in_place_fp8 = _meta_mod.update_decode_metadata_in_place_fp8


def _alloc(q_len: int, max_bs: int = 4, max_seq_len: int = 65600):
    return allocate_decode_metadata_fp8(
        max_batch_size=max_bs,
        q_len=q_len,
        window_size=128,
        head_dim=512,
        max_seq_len=max_seq_len,
        compress_ratios=[0, 4, 128],
        index_topk=16,
        device=(
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    )


def _i32(vals):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.tensor(vals, dtype=torch.int32, device=device)


def _ptr_snapshot(meta):
    ptrs = {
        "start_pos": meta.start_pos.data_ptr(),
        "position_ids": meta.position_ids.data_ptr(),
        "position_ids_long": meta.position_ids_long.data_ptr(),
        "slot_mapping_swa": meta.slot_mapping_swa.data_ptr(),
        "topk_window_idxs": meta.topk_window_idxs.data_ptr(),
        "topk_buffer_compressed": meta.topk_buffer_compressed.data_ptr(),
        "decode_seq_start_per_req": meta.decode_seq_start_per_req.data_ptr(),
    }
    for r, t in meta.slot_mapping_compressed.items():
        ptrs[f"slot_mapping_compressed[{r}]"] = t.data_ptr()
    for r, t in meta.compressed_lens.items():
        ptrs[f"compressed_lens[{r}]"] = t.data_ptr()
    for r, t in meta.compressed_lens_per_token.items():
        ptrs[f"compressed_lens_per_token[{r}]"] = t.data_ptr()
    for r, t in meta.topk_total_by_ratio.items():
        ptrs[f"topk_total_by_ratio[{r}]"] = t.data_ptr()
    return ptrs


def _reference_start_pos(attention_inputs, device, max_seq_len, q_len):
    if isinstance(attention_inputs, torch.Tensor):
        start_pos = attention_inputs
    else:
        is_target_verify = bool(getattr(attention_inputs, "is_target_verify", False))
        is_prefill = bool(getattr(attention_inputs, "is_prefill", False))
        if is_target_verify or (is_prefill and q_len > 1):
            start_pos = attention_inputs.prefix_lengths
        else:
            start_pos = attention_inputs.sequence_lengths
    if start_pos.device != device:
        start_pos = start_pos.to(device)
    if start_pos.dtype != torch.int32:
        start_pos = start_pos.to(torch.int32)
    max_start = max(0, int(max_seq_len) - int(q_len))
    return torch.clamp(start_pos, min=0, max=max_start)


def _reference_window_topk(position_ids_2d, window_size):
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


def _reference_update_decode_metadata_in_place(meta, attention_inputs, forbid_realloc=False):
    q_len = meta.q_len_per_req
    window_size = meta.window_size
    device = meta.start_pos.device
    ptrs = _ptr_snapshot(meta) if forbid_realloc else None

    start_pos = _reference_start_pos(attention_inputs, device, meta.max_seq_len, q_len)
    bs = int(start_pos.shape[0])
    T = bs * q_len
    position_ids_2d = start_pos.view(bs, 1) + torch.arange(
        q_len, device=device, dtype=torch.int32
    ).view(1, q_len)
    position_ids_flat = position_ids_2d.reshape(-1).contiguous()

    meta.start_pos[:bs].copy_(start_pos)
    meta.position_ids[:T].copy_(position_ids_flat)
    meta.position_ids_long[:T].copy_(position_ids_flat.to(torch.long))
    meta.decode_seq_start_per_req[:bs].copy_(start_pos)
    meta.topk_buffer_compressed[:bs].fill_(-1)

    req_base = torch.arange(bs, device=device, dtype=torch.int32).view(bs, 1)
    meta.slot_mapping_swa[:T].copy_(
        (req_base * window_size + (position_ids_2d % window_size)).reshape(-1)
    )

    window_idxs = _reference_window_topk(position_ids_2d, window_size)
    meta.topk_window_idxs[:bs].copy_(window_idxs)

    for r, slot_t in meta.slot_mapping_compressed.items():
        stride = meta.compressed_buffer_t_dim_per_ratio[r]
        abs_pos_plus_1 = position_ids_2d + 1
        on_boundary = (abs_pos_plus_1 % r) == 0
        in_req = abs_pos_plus_1 // r - 1
        cmp_lens_per_token = (abs_pos_plus_1 // r).to(torch.int32)
        cmp_req_base = req_base * stride
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
                torch.arange(K_dense, device=device, dtype=torch.int32)
                .view(1, 1, K_dense)
                .expand(bs, q_len, K_dense)
            )
            valid = dense_idxs < cmp_lens_per_token.view(bs, q_len, 1)
            total[:bs, :, window_size:].copy_(
                torch.where(valid, dense_idxs, torch.full_like(dense_idxs, -1))
            )

    if meta.swa_abs_idx is not None:
        win_range = torch.arange(window_size, device=device, dtype=torch.int32).view(
            1, 1, window_size
        )
        win_start = (position_ids_2d.unsqueeze(-1) - window_size + 1).clamp(min=0)
        candidate = win_start + win_range
        valid_pos = candidate <= position_ids_2d.unsqueeze(-1)
        meta.swa_abs_idx[:bs].copy_(
            torch.where(valid_pos, candidate, torch.full_like(candidate, -1))
        )

    meta.batch_size = bs
    meta.total_tokens = T

    if ptrs is not None:
        cur = _ptr_snapshot(meta)
        for name, ptr in ptrs.items():
            assert cur[name] == ptr, name


def _ref_compressed_kv_slots(
    block_table: torch.Tensor,
    positions: torch.Tensor,
    req_idx: torch.Tensor,
    *,
    ratio: int,
    kv_eb: int,
) -> torch.Tensor:
    pos = positions.to(torch.long)
    req = req_idx.to(torch.long)
    bt = block_table.to(torch.long)
    tokens_per_block = kv_eb * ratio
    block_in_seq = pos // tokens_per_block
    in_block = (pos % tokens_per_block) // ratio
    in_capacity = block_in_seq < bt.shape[1]
    safe_block = block_in_seq.clamp(min=0, max=bt.shape[1] - 1)
    block_id = bt[req, safe_block]
    slot = block_id * kv_eb + in_block
    valid = ((pos + 1) % ratio == 0) & in_capacity & (block_id > 0)
    return torch.where(valid, slot, torch.full_like(slot, -1))


def _ref_kv_slots(
    block_table: torch.Tensor,
    abs_pos: torch.Tensor,
    req_idx: torch.Tensor,
    *,
    entries_per_block: int,
) -> torch.Tensor:
    pos = abs_pos.to(torch.long)
    req = req_idx.to(torch.long)
    bt = block_table.to(torch.long)
    block_in_seq = pos // entries_per_block
    in_block = pos % entries_per_block
    in_capacity = block_in_seq < bt.shape[1]
    safe_block = block_in_seq.clamp(min=0, max=bt.shape[1] - 1)
    block_id = bt[req, safe_block]
    slot = block_id * entries_per_block + in_block
    valid = (pos >= 0) & in_capacity & (block_id > 0)
    return torch.where(valid, slot, torch.full_like(slot, -1))


def _ref_state_slots(
    block_table: torch.Tensor,
    positions: torch.Tensor,
    req_idx: torch.Tensor,
    *,
    entries_per_block: int,
) -> torch.Tensor:
    pos = positions.to(torch.long)
    req = req_idx.to(torch.long)
    bt = block_table.to(torch.long)
    block_in_seq = (pos // entries_per_block) % bt.shape[1]
    in_block = pos % entries_per_block
    block_id = bt[req, block_in_seq]
    slot = block_id * entries_per_block + in_block
    return torch.where(block_id > 0, slot, torch.full_like(slot, -1))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for fused decode meta")
class TestDecodeMetadataStartPos(unittest.TestCase):
    def test_in_place_update_writes_cuda_fused_path_without_realloc(self):
        meta = _alloc(q_len=1)
        ptrs = _ptr_snapshot(meta)

        update_decode_metadata_in_place_fp8(meta, _i32([0]), forbid_realloc=True)

        self.assertEqual(meta.start_pos[:1].tolist(), [0])
        self.assertEqual(meta.position_ids[:1].tolist(), [0])
        self.assertEqual(meta.slot_mapping_swa[:1].tolist(), [0])
        self.assertEqual(_ptr_snapshot(meta), ptrs)

    def test_normal_decode_uses_sequence_lengths(self):
        meta = _alloc(q_len=1)
        attn = SimpleNamespace(
            is_prefill=False,
            is_target_verify=False,
            sequence_lengths=_i32([4, 127]),
            prefix_lengths=_i32([1000, 2000]),
        )

        _reference_update_decode_metadata_in_place(meta, attn, forbid_realloc=True)

        self.assertEqual(meta.start_pos[:2].tolist(), [4, 127])
        self.assertEqual(meta.position_ids[:2].tolist(), [4, 127])
        self.assertEqual(meta.position_ids_long[:2].tolist(), [4, 127])
        self.assertEqual(meta.req_id_per_token[:2].tolist(), [0, 1])
        self.assertEqual(meta.req_id_per_token_long[:2].tolist(), [0, 1])
        self.assertEqual(meta.decode_seq_start_per_req[:2].tolist(), [4, 127])
        self.assertEqual(meta.decode_cu_seq_per_req[:3].tolist(), [0, 1, 2])

    def test_target_verify_uses_prefix_lengths(self):
        meta = _alloc(q_len=2)
        attn = SimpleNamespace(
            is_prefill=True,
            is_target_verify=True,
            sequence_lengths=_i32([]),
            prefix_lengths=_i32([10, 20]),
        )

        _reference_update_decode_metadata_in_place(meta, attn, forbid_realloc=True)

        self.assertEqual(meta.start_pos[:2].tolist(), [10, 20])
        self.assertEqual(meta.position_ids[:4].tolist(), [10, 11, 20, 21])
        self.assertEqual(meta.position_ids_long[:4].tolist(), [10, 11, 20, 21])
        self.assertEqual(meta.req_id_per_token[:4].tolist(), [0, 0, 1, 1])
        self.assertEqual(meta.req_id_per_token_long[:4].tolist(), [0, 0, 1, 1])
        self.assertEqual(meta.decode_seq_start_per_req[:2].tolist(), [10, 20])
        self.assertEqual(meta.decode_cu_seq_per_req[:3].tolist(), [0, 2, 4])
        self.assertEqual(meta.compressed_lens[4][:2].tolist(), [3, 5])
        self.assertEqual(
            meta.compressed_lens_per_token[4][:2].tolist(),
            [[2, 3], [5, 5]],
        )
        self.assertEqual(
            meta.compressed_lens_per_token[128][:2].tolist(),
            [[0, 0], [0, 0]],
        )

    def test_draft_prefill_graph_uses_prefix_not_stale_sequence_lengths(self):
        meta = _alloc(q_len=2, max_seq_len=65600)
        attn = SimpleNamespace(
            is_prefill=True,
            is_target_verify=False,
            # In CudaGraphRunner prefill graph replay this tensor is not
            # refreshed; it keeps the capture-time sentinel near max_seq_len.
            sequence_lengths=_i32([65597, 65597]),
            prefix_lengths=_i32([65598, 8]),
        )

        _reference_update_decode_metadata_in_place(meta, attn, forbid_realloc=True)

        self.assertEqual(meta.start_pos[:2].tolist(), [65598, 8])
        self.assertEqual(meta.position_ids[:4].tolist(), [65598, 65599, 8, 9])
        self.assertNotEqual(
            meta.position_ids[:4].tolist(), [65597, 65598, 65597, 65598]
        )

    def test_multi_token_graph_clamps_whole_position_window(self):
        meta = _alloc(q_len=4, max_bs=1, max_seq_len=18)
        attn = SimpleNamespace(
            is_prefill=True,
            is_target_verify=False,
            sequence_lengths=_i32([17]),
            prefix_lengths=_i32([17]),
        )

        _reference_update_decode_metadata_in_place(meta, attn, forbid_realloc=True)

        self.assertEqual(meta.start_pos[:1].tolist(), [14])
        self.assertEqual(meta.position_ids[:4].tolist(), [14, 15, 16, 17])

        ratio4_slots = meta.slot_mapping_compressed[4][:4]
        self.assertEqual(ratio4_slots.tolist(), [-1, 3, -1, -1])
        valid = ratio4_slots[ratio4_slots >= 0]
        self.assertTrue(torch.all(valid < meta.compressed_buffer_t_dim_per_ratio[4]))

    def test_eager_builder_clamps_whole_position_window(self):
        meta = build_decode_metadata_fp8(
            attention_inputs=_i32([17]),
            q_len=4,
            window_size=128,
            head_dim=512,
            max_seq_len=18,
            compress_ratios=[0, 4, 128],
            index_topk=16,
            device=torch.device("cpu"),
        )

        self.assertEqual(meta.start_pos.tolist(), [14])
        self.assertEqual(meta.position_ids.tolist(), [14, 15, 16, 17])
        ratio4_slots = meta.slot_mapping_compressed[4]
        self.assertEqual(ratio4_slots.tolist(), [-1, 3, -1, -1])
        valid = ratio4_slots[ratio4_slots >= 0]
        self.assertTrue(torch.all(valid < meta.compressed_buffer_t_dim_per_ratio[4]))

    def test_eager_builder_attention_inputs_object_normal_decode(self):
        """Direct ``build_decode_metadata_fp8`` with a SimpleNamespace
        attn_inputs (no torch.Tensor shortcut): normal-decode branch must
        pull from ``sequence_lengths`` and ignore stale ``prefix_lengths``.
        """
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        attn = SimpleNamespace(
            is_prefill=False,
            is_target_verify=False,
            sequence_lengths=torch.tensor([4, 127], dtype=torch.int32, device=device),
            prefix_lengths=torch.tensor(
                [1000, 2000], dtype=torch.int32, device=device
            ),
        )
        meta = build_decode_metadata_fp8(
            attention_inputs=attn,
            q_len=1,
            window_size=128,
            head_dim=512,
            max_seq_len=65600,
            compress_ratios=[0, 4, 128],
            index_topk=16,
            device=device,
        )

        self.assertEqual(meta.start_pos.tolist(), [4, 127])
        self.assertEqual(meta.position_ids.tolist(), [4, 127])

    def test_eager_builder_attention_inputs_object_target_verify(self):
        """Direct ``build_decode_metadata_fp8`` with target-verify
        SimpleNamespace: must read ``prefix_lengths`` and produce a
        q_len=2 position window per request.
        """
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        attn = SimpleNamespace(
            is_prefill=True,
            is_target_verify=True,
            sequence_lengths=torch.tensor([], dtype=torch.int32, device=device),
            prefix_lengths=torch.tensor([10, 20], dtype=torch.int32, device=device),
        )
        meta = build_decode_metadata_fp8(
            attention_inputs=attn,
            q_len=2,
            window_size=128,
            head_dim=512,
            max_seq_len=65600,
            compress_ratios=[0, 4, 128],
            index_topk=16,
            device=device,
        )

        self.assertEqual(meta.start_pos.tolist(), [10, 20])
        self.assertEqual(meta.position_ids.tolist(), [10, 11, 20, 21])
        # ratio=4 compressed lens: floor((sp + q_len) / 4) per request
        self.assertEqual(meta.compressed_lens[4].tolist(), [3, 5])

    def test_hca_dense_indices_cover_1m_context(self):
        max_seq_len = 1048576
        device = torch.device("cuda")
        start_pos = torch.tensor([max_seq_len - 1], dtype=torch.int32, device=device)

        eager = build_decode_metadata_fp8(
            attention_inputs=start_pos,
            q_len=1,
            window_size=128,
            head_dim=512,
            max_seq_len=max_seq_len,
            compress_ratios=[0, 4, 128],
            index_topk=512,
            device=device,
        )
        eager_hca = eager.topk_total_by_ratio[128][0, 0, 128:]
        self.assertEqual(eager_hca.numel(), 8192)
        self.assertEqual(int((eager_hca >= 0).sum()), 8192)
        self.assertEqual(int(eager_hca[-1]), 8191)

        graph_meta = allocate_decode_metadata_fp8(
            max_batch_size=1,
            q_len=1,
            window_size=128,
            head_dim=512,
            max_seq_len=max_seq_len,
            compress_ratios=[0, 4, 128],
            index_topk=512,
            device=device,
        )
        _reference_update_decode_metadata_in_place(
            graph_meta, start_pos, forbid_realloc=True
        )
        graph_hca = graph_meta.topk_total_by_ratio[128][0, 0, 128:]
        self.assertEqual(graph_hca.numel(), 8192)
        self.assertEqual(int((graph_hca >= 0).sum()), 8192)
        self.assertEqual(int(graph_hca[-1]), 8191)

        # CSA/indexer width must remain controlled by index_topk.
        self.assertEqual(graph_meta.topk_buffer_compressed.shape[-1], 512)
        self.assertEqual(graph_meta.topk_total_by_ratio[4].shape[-1], 128 + 512)

    def test_hca_dense_indices_align_70k_context_for_flashmla(self):
        max_seq_len = 70000
        device = torch.device("cuda")
        start_pos = torch.tensor([max_seq_len - 1], dtype=torch.int32, device=device)

        graph_meta = allocate_decode_metadata_fp8(
            max_batch_size=1,
            q_len=1,
            window_size=128,
            head_dim=512,
            max_seq_len=max_seq_len,
            compress_ratios=[0, 4, 128],
            index_topk=512,
            device=device,
        )
        _reference_update_decode_metadata_in_place(
            graph_meta, start_pos, forbid_realloc=True
        )
        graph_hca = graph_meta.topk_total_by_ratio[128][0, 0, 128:]

        self.assertEqual(max_seq_len // 128, 546)
        self.assertEqual(graph_hca.numel(), 576)
        self.assertEqual(graph_meta.hca_cmp_global_slots.shape[-1], 576)
        self.assertEqual(graph_hca.numel() % 64, 0)
        self.assertEqual(int((graph_hca >= 0).sum()), 546)
        self.assertEqual(int(graph_hca[545]), 545)
        self.assertTrue(torch.all(graph_hca[546:] == -1))

    def test_paged_compressor_slots_match_compressor_formula_for_bs(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for paged metadata translator")
        device = torch.device("cuda")
        q_len = 3
        paged_specs = {
            SWA_KV: (256, 4),
            CSA_KV: (64, 4),
            INDEXER_KV: (64, 4),
            HCA_KV: (2, 4),
            CSA_STATE: (256, 4),
            INDEXER_STATE: (256, 4),
            HCA_STATE: (256, 4),
        }
        meta = allocate_decode_metadata_fp8(
            max_batch_size=2,
            q_len=q_len,
            window_size=128,
            head_dim=512,
            max_seq_len=1024,
            compress_ratios=[4, 128],
            index_topk=16,
            device=device,
            paged_pool_specs=paged_specs,
        )
        attn = SimpleNamespace(
            is_prefill=True,
            is_target_verify=True,
            sequence_lengths=_i32([]).to(device),
            # Positions: req0=[2,3,4], req1=[254,255,256].
            # Covers ratio-4 boundaries 3/255 and ratio-128 boundary 255.
            prefix_lengths=_i32([2, 254]).to(device),
        )
        paged_block_tables = {
            SWA_KV: _i32([[11, 12, 13, 14], [21, 22, 23, 24]]).to(device),
            CSA_KV: _i32([[101, 102, 103, 104], [201, 202, 203, 204]]).to(device),
            INDEXER_KV: _i32([[301, 302, 303, 304], [401, 402, 403, 404]]).to(device),
            HCA_KV: _i32([[501, 502, 503, 504], [601, 602, 603, 604]]).to(device),
            CSA_STATE: _i32([[701, 702, 703, 704], [801, 802, 803, 804]]).to(device),
            INDEXER_STATE: _i32([[901, 902, 903, 904], [1001, 1002, 1003, 1004]]).to(
                device
            ),
            HCA_STATE: _i32([[1101, 1102, 1103, 1104], [1201, 1202, 1203, 1204]]).to(
                device
            ),
        }
        entries_per_block = {k: v[0] for k, v in paged_specs.items()}

        update_decode_metadata_in_place_fp8(
            meta,
            attn,
            forbid_realloc=True,
            paged_block_tables=paged_block_tables,
            paged_pool_entries_per_block=entries_per_block,
        )

        T = 2 * q_len
        positions = torch.tensor(
            [2, 3, 4, 254, 255, 256], dtype=torch.long, device=device
        )
        req_idx = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long, device=device)

        expected_swa = _ref_kv_slots(
            paged_block_tables[SWA_KV],
            positions,
            req_idx,
            entries_per_block=256,
        )
        actual_swa = meta.pool_write_slot_mappings[SWA_KV][:T]
        self.assertEqual(actual_swa.tolist(), expected_swa.tolist())

        for attn_type, ratio, kv_eb in (
            (CSA_KV, 4, 64),
            (INDEXER_KV, 4, 64),
            (HCA_KV, 128, 2),
        ):
            expected = _ref_compressed_kv_slots(
                paged_block_tables[attn_type],
                positions,
                req_idx,
                ratio=ratio,
                kv_eb=kv_eb,
            )
            actual = meta.pool_write_slot_mappings[attn_type][:T]
            self.assertEqual(actual.tolist(), expected.tolist())

        for attn_type in (CSA_STATE, INDEXER_STATE, HCA_STATE):
            expected = _ref_state_slots(
                paged_block_tables[attn_type],
                positions,
                req_idx,
                entries_per_block=256,
            )
            actual = meta.compressor_state_slot_mappings[attn_type][:T]
            self.assertEqual(actual.tolist(), expected.tolist())

        self.assertIsNotNone(meta.swa_global_slots)
        self.assertIsNotNone(meta.hca_cmp_global_slots)
        req_id = meta.req_id_per_token[:T]
        expected_swa_global = translate_local_to_global_slots(
            req_id,
            paged_block_tables[SWA_KV],
            meta.swa_abs_idx[:2].reshape(T, 128),
            block_size=256,
        )
        self.assertEqual(
            meta.swa_global_slots[:T].tolist(), expected_swa_global.tolist()
        )

        hca_dense = meta.topk_total_by_ratio[128][:2, :, 128:].reshape(T, -1)
        expected_hca_global = translate_local_to_global_slots(
            req_id,
            paged_block_tables[HCA_KV],
            hca_dense,
            block_size=2,
        )
        self.assertEqual(
            meta.hca_cmp_global_slots[:T].tolist(), expected_hca_global.tolist()
        )

    def test_paged_slot_mapping_normalizes_unallocated_blocks(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for paged metadata translator")
        device = torch.device("cuda")
        block_table = _i32([[0, -1, 9]]).to(device)
        abs_pos = torch.tensor([3, 260, 520], dtype=torch.int32, device=device)

        mapped = compute_kv_pool_slot_mapping(
            block_table,
            abs_pos,
            entries_per_block=256,
        )
        self.assertEqual(mapped.tolist(), [-1, -1, 9 * 256 + 8])

        req_id = torch.zeros(1, dtype=torch.int32, device=device)
        local = torch.tensor([[3, 260, 520, -1]], dtype=torch.int32, device=device)
        translated = translate_local_to_global_slots(
            req_id,
            block_table,
            local,
            block_size=256,
        )
        self.assertEqual(translated.tolist(), [[-1, -1, 9 * 256 + 8, -1]])


if __name__ == "__main__":
    unittest.main()
