"""CPU tests for DSv4 FP8 decode metadata start-position selection."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

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


if __name__ == "__main__":
    unittest.main()
