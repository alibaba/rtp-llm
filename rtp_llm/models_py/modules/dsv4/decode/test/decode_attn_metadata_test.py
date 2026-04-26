"""Unit tests for DSv4 decode attention metadata builder.

Pure-Python oracle vs vectorized torch builder. CPU-only — no kernels.
"""

import os
import sys
import unittest
from typing import Dict, List

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rtp_llm.models_py.modules.dsv4.decode.decode_attn_metadata import (
    DSv4DecodeAttnMetadata,
    build_decode_metadata,
)

# --- Pure-python oracles -------------------------------------------------


def _oracle_swa_slot(start_pos: List[int], q_len: int, window: int) -> List[int]:
    out = []
    for r, sp in enumerate(start_pos):
        for s in range(q_len):
            ring = (sp + s) % window
            out.append(r * window + ring)
    return out


def _oracle_cmp_slot(
    start_pos: List[int], q_len: int, ratio: int, max_seq_len: int
) -> List[int]:
    stride = max_seq_len // ratio
    out = []
    for r, sp in enumerate(start_pos):
        for s in range(q_len):
            abs_pos_plus_1 = sp + s + 1
            if abs_pos_plus_1 % ratio == 0:
                in_req = abs_pos_plus_1 // ratio - 1
                out.append(r * stride + in_req)
            else:
                out.append(-1)
    return out


def _oracle_window_idxs(
    start_pos: List[int], q_len: int, window: int
) -> List[List[List[int]]]:
    """[B, q_len, window] of request-local ring slots, -1 for OOB."""
    out = []
    for sp in start_pos:
        per_req = []
        for s in range(q_len):
            abs_pos = sp + s
            row = []
            for k in range(window):
                desired = abs_pos - (window - 1) + k
                if desired < 0:
                    row.append(-1)
                else:
                    row.append(desired % window)
            per_req.append(row)
        out.append(per_req)
    return out


def _oracle_compressed_lens(start_pos: List[int], q_len: int, ratio: int) -> List[int]:
    return [(sp + q_len) // ratio for sp in start_pos]


# --- Tests ---------------------------------------------------------------


class TestSwaSlotMapping(unittest.TestCase):
    def test_basic_b4_qlen1(self):
        device = torch.device("cpu")
        start_pos = torch.tensor([0, 5, 130, 1000], dtype=torch.int32)
        meta = build_decode_metadata(
            start_pos=start_pos,
            q_len=1,
            window_size=128,
            head_dim=512,
            max_seq_len=2048,
            compress_ratios=[0, 4, 128],
            index_topk=64,
            device=device,
        )
        oracle = _oracle_swa_slot([0, 5, 130, 1000], 1, 128)
        self.assertEqual(meta.slot_mapping_swa.tolist(), oracle)

    def test_qlen_gt_1_for_spec_decode(self):
        device = torch.device("cpu")
        start_pos = torch.tensor([10, 200], dtype=torch.int32)
        meta = build_decode_metadata(
            start_pos=start_pos,
            q_len=3,
            window_size=128,
            head_dim=512,
            max_seq_len=2048,
            compress_ratios=[0, 4, 128],
            index_topk=64,
            device=device,
        )
        oracle = _oracle_swa_slot([10, 200], 3, 128)
        self.assertEqual(meta.slot_mapping_swa.tolist(), oracle)


class TestCompressedSlotMapping(unittest.TestCase):
    def test_ratio4_boundaries(self):
        device = torch.device("cpu")
        # start_pos=3 with q_len=1 → abs_pos=3, +1=4 → boundary for ratio=4
        # start_pos=4 with q_len=1 → abs_pos=4, +1=5 → not boundary
        # start_pos=11 → +1=12 → boundary
        start_pos = torch.tensor([3, 4, 11], dtype=torch.int32)
        meta = build_decode_metadata(
            start_pos=start_pos,
            q_len=1,
            window_size=128,
            head_dim=512,
            max_seq_len=2048,
            compress_ratios=[4],
            index_topk=64,
            device=device,
        )
        slots = meta.slot_mapping_compressed[4].tolist()
        # request 0: stride=512, in_req=0 → 0*512+0=0
        # request 1: -1
        # request 2: stride=512, in_req=2 → 2*512+2=1026
        self.assertEqual(slots, [0, -1, 1026])

    def test_ratio128_boundaries(self):
        device = torch.device("cpu")
        # abs_pos+1 == 128 → in_req=0; abs_pos+1 == 256 → in_req=1
        start_pos = torch.tensor([127, 128, 255], dtype=torch.int32)
        meta = build_decode_metadata(
            start_pos=start_pos,
            q_len=1,
            window_size=128,
            head_dim=512,
            max_seq_len=4096,
            compress_ratios=[128],
            index_topk=64,
            device=device,
        )
        slots = meta.slot_mapping_compressed[128].tolist()
        # stride = 4096//128 = 32
        self.assertEqual(slots, [0 * 32 + 0, -1, 2 * 32 + 1])

    def test_qlen3_with_boundary_in_middle(self):
        device = torch.device("cpu")
        # start_pos=2, q_len=3 → abs_pos in {2, 3, 4} → +1 in {3, 4, 5}
        # ratio=4: only abs_pos=3 (mid-step) hits boundary at offset 1
        start_pos = torch.tensor([2], dtype=torch.int32)
        meta = build_decode_metadata(
            start_pos=start_pos,
            q_len=3,
            window_size=128,
            head_dim=512,
            max_seq_len=512,
            compress_ratios=[4],
            index_topk=64,
            device=device,
        )
        # stride=128
        # t=0 (abs=2): -1, t=1 (abs=3): in_req=(4//4)-1=0 → slot 0,
        # t=2 (abs=4): -1
        self.assertEqual(meta.slot_mapping_compressed[4].tolist(), [-1, 0, -1])


class TestWindowTopkIdxs(unittest.TestCase):
    def test_full_window(self):
        device = torch.device("cpu")
        start_pos = torch.tensor([1000], dtype=torch.int32)
        meta = build_decode_metadata(
            start_pos=start_pos,
            q_len=1,
            window_size=128,
            head_dim=512,
            max_seq_len=2048,
            compress_ratios=[4],
            index_topk=64,
            device=device,
        )
        oracle = _oracle_window_idxs([1000], 1, 128)
        self.assertEqual(meta.topk_window_idxs.tolist(), oracle)

    def test_early_decode_underwin(self):
        device = torch.device("cpu")
        # abs_pos=5 < window=128 → only positions 0..5 valid; rest -1.
        start_pos = torch.tensor([5], dtype=torch.int32)
        meta = build_decode_metadata(
            start_pos=start_pos,
            q_len=1,
            window_size=128,
            head_dim=512,
            max_seq_len=2048,
            compress_ratios=[4],
            index_topk=64,
            device=device,
        )
        row = meta.topk_window_idxs[0, 0].tolist()
        # Last 6 positions in [win] should be 0..5; first 122 should be -1.
        self.assertEqual(row[:122], [-1] * 122)
        self.assertEqual(row[122:], [0, 1, 2, 3, 4, 5])

    def test_zero_pos(self):
        device = torch.device("cpu")
        start_pos = torch.tensor([0], dtype=torch.int32)
        meta = build_decode_metadata(
            start_pos=start_pos,
            q_len=1,
            window_size=8,
            head_dim=512,
            max_seq_len=64,
            compress_ratios=[4],
            index_topk=8,
            device=device,
        )
        row = meta.topk_window_idxs[0, 0].tolist()
        # abs_pos=0: only position 0 valid, in last slot
        self.assertEqual(row, [-1, -1, -1, -1, -1, -1, -1, 0])


class TestCompressedLens(unittest.TestCase):
    def test_after_step(self):
        device = torch.device("cpu")
        start_pos = torch.tensor([0, 7, 8, 100], dtype=torch.int32)
        meta = build_decode_metadata(
            start_pos=start_pos,
            q_len=1,
            window_size=128,
            head_dim=512,
            max_seq_len=512,
            compress_ratios=[4, 128],
            index_topk=64,
            device=device,
        )
        # Post-step length per request = (start_pos + 1) // ratio.
        self.assertEqual(
            meta.compressed_lens[4].tolist(),
            _oracle_compressed_lens([0, 7, 8, 100], 1, 4),
        )
        self.assertEqual(
            meta.compressed_lens[128].tolist(),
            _oracle_compressed_lens([0, 7, 8, 100], 1, 128),
        )


class TestTopkTotalByRatio(unittest.TestCase):
    def test_csa_compressed_half_starts_neg1(self):
        device = torch.device("cpu")
        start_pos = torch.tensor([100], dtype=torch.int32)
        meta = build_decode_metadata(
            start_pos=start_pos,
            q_len=1,
            window_size=128,
            head_dim=512,
            max_seq_len=2048,
            compress_ratios=[4],
            index_topk=16,
            device=device,
        )
        total = meta.topk_total_by_ratio[4]
        # First [128] should equal the window topk; last [16] is -1 sentinel
        # awaiting indexer fill.
        self.assertEqual(total.shape, (1, 1, 144))
        self.assertEqual(
            total[0, 0, :128].tolist(), meta.topk_window_idxs[0, 0].tolist()
        )
        self.assertEqual(total[0, 0, 128:].tolist(), [-1] * 16)

    def test_hca_compressed_half_filled_dense(self):
        device = torch.device("cpu")
        # ratio=128, max_seq_len=2048 → max compressed entries=16.
        # start_pos=900 + q_len=1 → compressed_lens[128] = 901 // 128 = 7.
        # So first 7 entries of compressed half should be [0..6], rest -1.
        start_pos = torch.tensor([900], dtype=torch.int32)
        meta = build_decode_metadata(
            start_pos=start_pos,
            q_len=1,
            window_size=128,
            head_dim=512,
            max_seq_len=2048,
            compress_ratios=[128],
            index_topk=16,
            device=device,
        )
        cmp_half = meta.topk_total_by_ratio[128][0, 0, 128:].tolist()
        # 901 // 128 = 7
        self.assertEqual(cmp_half[:7], [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(cmp_half[7:], [-1] * 9)


class TestBufferAllocation(unittest.TestCase):
    def test_topk_buffer_shape_and_dtype(self):
        device = torch.device("cpu")
        meta = build_decode_metadata(
            start_pos=torch.tensor([0, 0, 0, 0], dtype=torch.int32),
            q_len=1,
            window_size=128,
            head_dim=512,
            max_seq_len=2048,
            compress_ratios=[4, 128],
            index_topk=512,
            device=device,
        )
        self.assertEqual(meta.topk_buffer_compressed.shape, (4, 1, 512))
        self.assertEqual(meta.topk_buffer_compressed.dtype, torch.int32)
        # Initialized to -1
        self.assertTrue((meta.topk_buffer_compressed == -1).all())

    def test_metadata_geometry_fields(self):
        device = torch.device("cpu")
        meta = build_decode_metadata(
            start_pos=torch.tensor([10, 20, 30], dtype=torch.int32),
            q_len=1,
            window_size=128,
            head_dim=512,
            max_seq_len=512,
            compress_ratios=[0, 4, 128, 4, 128],
            index_topk=64,
            device=device,
        )
        self.assertEqual(meta.batch_size, 3)
        self.assertEqual(meta.q_len_per_req, 1)
        self.assertEqual(meta.total_tokens, 3)
        self.assertEqual(meta.window_size, 128)
        self.assertEqual(meta.head_dim, 512)
        self.assertEqual(meta.compressed_buffer_t_dim_per_ratio, {4: 128, 128: 4})
        self.assertEqual(meta.compressed_offset, 128)


class TestCudaParity(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "no cuda")
    def test_cuda_matches_cpu(self):
        cpu_dev = torch.device("cpu")
        cuda_dev = torch.device("cuda:0")
        sp_cpu = torch.tensor([3, 7, 130, 1500], dtype=torch.int32)
        sp_cuda = sp_cpu.cuda()
        kw = dict(
            q_len=1,
            window_size=128,
            head_dim=512,
            max_seq_len=2048,
            compress_ratios=[0, 4, 128],
            index_topk=64,
        )
        meta_cpu = build_decode_metadata(start_pos=sp_cpu, device=cpu_dev, **kw)
        meta_cuda = build_decode_metadata(start_pos=sp_cuda, device=cuda_dev, **kw)
        self.assertEqual(
            meta_cpu.slot_mapping_swa.tolist(),
            meta_cuda.slot_mapping_swa.cpu().tolist(),
        )
        self.assertEqual(
            meta_cpu.slot_mapping_compressed[4].tolist(),
            meta_cuda.slot_mapping_compressed[4].cpu().tolist(),
        )
        self.assertEqual(
            meta_cpu.topk_window_idxs.tolist(),
            meta_cuda.topk_window_idxs.cpu().tolist(),
        )


if __name__ == "__main__":
    unittest.main()
