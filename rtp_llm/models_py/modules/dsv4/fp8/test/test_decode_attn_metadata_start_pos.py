"""CPU tests for DSv4 FP8 decode metadata start-position selection."""

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
        device=torch.device("cpu"),
    )


def _i32(vals):
    return torch.tensor(vals, dtype=torch.int32)


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


class TestDecodeMetadataStartPos(unittest.TestCase):
    def test_normal_decode_uses_sequence_lengths(self):
        meta = _alloc(q_len=1)
        attn = SimpleNamespace(
            is_prefill=False,
            is_target_verify=False,
            sequence_lengths=_i32([4, 127]),
            prefix_lengths=_i32([1000, 2000]),
        )

        update_decode_metadata_in_place_fp8(meta, attn, forbid_realloc=True)

        self.assertEqual(meta.start_pos[:2].tolist(), [4, 127])
        self.assertEqual(meta.position_ids[:2].tolist(), [4, 127])
        self.assertEqual(meta.position_ids_long[:2].tolist(), [4, 127])
        self.assertEqual(meta.req_id_per_token[:2].tolist(), [0, 1])
        self.assertEqual(meta.req_id_per_token_long[:2].tolist(), [0, 1])
        self.assertEqual(meta.decode_seq_start_per_req[:2].tolist(), [4, 127])
        self.assertEqual(meta.decode_cu_seq_per_req[:3].tolist(), [0, 1, 2])
        self.assertEqual(meta.cache_seqlens_i32[:2].tolist(), [5, 128])

    def test_target_verify_uses_prefix_lengths(self):
        meta = _alloc(q_len=2)
        attn = SimpleNamespace(
            is_prefill=True,
            is_target_verify=True,
            sequence_lengths=_i32([]),
            prefix_lengths=_i32([10, 20]),
        )

        update_decode_metadata_in_place_fp8(meta, attn, forbid_realloc=True)

        self.assertEqual(meta.start_pos[:2].tolist(), [10, 20])
        self.assertEqual(meta.position_ids[:4].tolist(), [10, 11, 20, 21])
        self.assertEqual(meta.position_ids_long[:4].tolist(), [10, 11, 20, 21])
        self.assertEqual(meta.req_id_per_token[:4].tolist(), [0, 0, 1, 1])
        self.assertEqual(meta.req_id_per_token_long[:4].tolist(), [0, 0, 1, 1])
        self.assertEqual(meta.decode_seq_start_per_req[:2].tolist(), [10, 20])
        self.assertEqual(meta.decode_cu_seq_per_req[:3].tolist(), [0, 2, 4])
        self.assertEqual(meta.cache_seqlens_i32[:2].tolist(), [12, 22])

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

        update_decode_metadata_in_place_fp8(meta, attn, forbid_realloc=True)

        self.assertEqual(meta.start_pos[:2].tolist(), [65598, 8])
        self.assertEqual(meta.position_ids[:4].tolist(), [65598, 65599, 8, 9])
        self.assertEqual(meta.cache_seqlens_i32[:2].tolist(), [65600, 10])
        self.assertNotEqual(meta.position_ids[:4].tolist(), [65597, 65598, 65597, 65598])

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
            INDEXER_STATE: _i32(
                [[901, 902, 903, 904], [1001, 1002, 1003, 1004]]
            ).to(device),
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


if __name__ == "__main__":
    unittest.main()
