#!/usr/bin/env python3
"""Direct UT for fused SWA varlen topk metadata.

The test loads ``_swa_ops_triton.py`` directly so it can run from a source
tree without compiled RTP-LLM shared libraries.  It validates the fused
``topk_idxs + topk_length`` helper against the pre-existing torch formula.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


def _load_swa_ops():
    path = Path(__file__).resolve().parents[1] / "_swa_ops_triton.py"
    spec = importlib.util.spec_from_file_location("dsv4_swa_ops_triton_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_swa_ops = _load_swa_ops()


def _flat_positions(prefix_lengths: list[int], input_lengths: list[int], device):
    positions = torch.cat(
        [
            torch.arange(p, p + length, dtype=torch.int64, device=device)
            for p, length in zip(prefix_lengths, input_lengths)
        ],
        dim=0,
    )
    req_id = torch.cat(
        [
            torch.full((length,), b, dtype=torch.int32, device=device)
            for b, length in enumerate(input_lengths)
        ],
        dim=0,
    )
    cu_seqlens = torch.zeros(len(input_lengths) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(
        torch.tensor(input_lengths, dtype=torch.int32, device=device), dim=0
    )
    return positions, req_id, cu_seqlens


def _reference(window_size, cu_seqlens, position_ids, prefix_lengths, req_id):
    req_id_idx = req_id.reshape(-1).to(dtype=torch.long)
    cu_i32 = cu_seqlens.reshape(-1).to(dtype=torch.int32)
    prefix_i32 = prefix_lengths.reshape(-1).to(dtype=torch.int32)
    pos_i32 = position_ids.reshape(-1).to(dtype=torch.int32)
    prefix_per_token = prefix_i32.gather(0, req_id_idx)
    req_start = cu_i32.gather(0, req_id_idx)
    local_pos = pos_i32 - prefix_per_token
    offsets = torch.arange(window_size, dtype=torch.int32, device=position_ids.device)
    win_start = (local_pos.unsqueeze(1) - window_size + 1).clamp_min(0)
    local_idx = win_start + offsets
    topk = torch.where(
        local_idx > local_pos.unsqueeze(1),
        torch.full_like(local_idx, -1),
        req_start.unsqueeze(1) + local_idx,
    ).contiguous()
    length = torch.clamp(local_pos + 1, max=window_size).contiguous()
    return topk, length


def _slot_ref(position_ids, req_id, prefix_lengths, *, M, window_size, base_offset=0):
    req = req_id.reshape(-1).to(dtype=torch.long)
    pos = position_ids.reshape(-1).to(dtype=torch.long)
    prefix = prefix_lengths.reshape(-1).to(dtype=torch.long)
    p = torch.clamp_max(prefix, window_size - 1)
    return (
        req * M + base_offset + p.gather(0, req) + (pos - prefix.gather(0, req))
    ).contiguous()


def _slot_from_cu_ref(cu_seqlens, prefix_lengths, *, num_tokens, M, window_size, base_offset=0):
    cu = cu_seqlens.reshape(-1).to(dtype=torch.long)
    prefix = prefix_lengths.reshape(-1).to(dtype=torch.long)
    g = torch.arange(num_tokens, dtype=torch.long, device=cu.device)
    req = torch.bucketize(g, cu[1:], right=True).clamp(max=prefix.numel() - 1)
    p = torch.clamp_max(prefix, window_size - 1)
    local_pos = g - cu.gather(0, req)
    return (req * M + base_offset + p.gather(0, req) + local_pos).contiguous()


class SwaTopkMetaTritonTest(unittest.TestCase):
    def _check(self, window_size, prefix_lengths, input_lengths, *, device):
        pos, req_id, cu = _flat_positions(prefix_lengths, input_lengths, device)
        prefix = torch.tensor(prefix_lengths, dtype=torch.int32, device=device)
        got_topk, got_len = _swa_ops.compute_window_topk_and_length_varlen(
            window_size, cu, pos, prefix, req_id
        )
        ref_topk, ref_len = _reference(window_size, cu, pos, prefix, req_id)
        self.assertEqual(got_topk.dtype, torch.int32)
        self.assertEqual(got_len.dtype, torch.int32)
        self.assertTrue(torch.equal(got_topk.cpu(), ref_topk.cpu()))
        self.assertTrue(torch.equal(got_len.cpu(), ref_len.cpu()))

    def test_cpu_fallback_b1_continuation(self):
        self._check(8, [100], [6], device=torch.device("cpu"))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_b1_cold(self):
        self._check(8, [0], [12], device=torch.device("cuda"))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_mixed_batch(self):
        self._check(16, [0, 7, 1000], [5, 9, 33], device=torch.device("cuda"))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_large_window(self):
        self._check(128, [4096], [1024], device=torch.device("cuda"))

    def _check_slot_in_flat(self, prefix_lengths, input_lengths, *, device):
        pos, req_id, _cu = _flat_positions(prefix_lengths, input_lengths, device)
        prefix = torch.tensor(prefix_lengths, dtype=torch.int32, device=device)
        got = _swa_ops.compute_swa_slot_in_flat(
            pos,
            req_id,
            prefix,
            M=257,
            window_size=16,
            base_offset=31,
        )
        ref = _slot_ref(
            pos,
            req_id,
            prefix,
            M=257,
            window_size=16,
            base_offset=31,
        )
        self.assertEqual(got.dtype, torch.long)
        self.assertTrue(torch.equal(got.cpu(), ref.cpu()))

    def test_cpu_slot_in_flat(self):
        self._check_slot_in_flat([0, 7, 1000], [5, 9, 33], device=torch.device("cpu"))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_slot_in_flat(self):
        self._check_slot_in_flat([0, 7, 1000], [5, 9, 33], device=torch.device("cuda"))

    def _check_slot_in_flat_from_cu(self, prefix_lengths, input_lengths, *, device):
        _pos, _req_id, cu = _flat_positions(prefix_lengths, input_lengths, device)
        prefix = torch.tensor(prefix_lengths, dtype=torch.int32, device=device)
        num_tokens = int(sum(input_lengths))
        got = _swa_ops.compute_swa_slot_in_flat_from_cu(
            cu,
            prefix,
            num_tokens=num_tokens,
            M=257,
            window_size=16,
            base_offset=31,
        )
        ref = _slot_from_cu_ref(
            cu,
            prefix,
            num_tokens=num_tokens,
            M=257,
            window_size=16,
            base_offset=31,
        )
        self.assertEqual(got.dtype, torch.long)
        self.assertTrue(torch.equal(got.cpu(), ref.cpu()))

    def test_cpu_slot_in_flat_from_cu(self):
        self._check_slot_in_flat_from_cu([0, 7, 1000], [5, 9, 33], device=torch.device("cpu"))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_slot_in_flat_from_cu_b1(self):
        self._check_slot_in_flat_from_cu([4096], [1024], device=torch.device("cuda"))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_slot_in_flat_from_cu_mixed_batch(self):
        self._check_slot_in_flat_from_cu([0, 7, 1000], [5, 9, 33], device=torch.device("cuda"))


if __name__ == "__main__":
    unittest.main()
