import unittest

import torch

from rtp_llm.models_py.modules.dsv4._metadata_triton import (
    _TRITON_AVAILABLE,
    build_cp_compress_topk_idxs,
    build_cp_window_topk_idxs,
    build_pool_slots,
    build_swa_pool_slot_mapping,
)


def _ref_cp_window(global_positions, bsz, seq_len_total, window_size):
    S = int(global_positions.numel())
    W = min(window_size, max(seq_len_total, 1))
    base = global_positions.unsqueeze(1)
    window_start = (base - W + 1).clamp_min(0)
    offs = torch.arange(W, device=global_positions.device, dtype=torch.long)
    matrix = window_start + offs
    invalid = (matrix > base) | (matrix >= seq_len_total)
    matrix = torch.where(invalid, torch.full_like(matrix, -1), matrix)
    if W < window_size:
        pad = torch.full(
            (S, window_size - W), -1, dtype=torch.long, device=global_positions.device
        )
        matrix = torch.cat([matrix, pad], dim=1)
    return matrix.unsqueeze(0).expand(bsz, -1, -1).contiguous()


def _ref_cp_compress(global_positions, bsz, seq_len_total, ratio, offset):
    S = int(global_positions.numel())
    T_comp = max(seq_len_total // ratio, 0)
    if T_comp == 0:
        return torch.empty((bsz, S, 0), dtype=torch.long, device=global_positions.device)
    cols = torch.arange(T_comp, device=global_positions.device, dtype=torch.long)
    max_allowed = (global_positions + 1) // ratio
    matrix = torch.where(
        cols.unsqueeze(0) < max_allowed.unsqueeze(1),
        cols.unsqueeze(0) + offset,
        -1,
    )
    return matrix.unsqueeze(0).expand(bsz, -1, -1).contiguous()


def _ref_pool_slots(block_table, bsz, T, eb):
    max_blocks = block_table.shape[1]
    pool_capacity = max_blocks * eb
    pos = torch.arange(T, device=block_table.device, dtype=torch.long)
    in_capacity_row = pos < pool_capacity
    safe_pos = torch.where(in_capacity_row, pos, torch.zeros_like(pos))
    block_in_seq = safe_pos // eb
    in_block = safe_pos % eb
    bt_long = block_table.to(torch.long)
    b_idx = torch.arange(bsz, device=block_table.device, dtype=torch.long).unsqueeze(1)
    block_id = bt_long[:bsz][b_idx, block_in_seq.unsqueeze(0)]
    in_capacity = in_capacity_row.unsqueeze(0).expand(bsz, -1)
    valid = (block_id > 0) & in_capacity
    safe_slot = torch.where(
        valid, block_id * eb + in_block.unsqueeze(0), torch.zeros_like(block_id)
    )
    return valid, safe_slot


def _ref_swa_slot_mapping(block_table, bsz, T, eb, sp, row_seqlens):
    device = block_table.device
    if isinstance(sp, torch.Tensor):
        sp_t = sp.to(device=device, dtype=torch.long).reshape(-1)
    else:
        sp_t = torch.full((bsz,), int(sp), device=device, dtype=torch.long)
    if isinstance(row_seqlens, torch.Tensor):
        seq_t = row_seqlens.to(device=device, dtype=torch.long).reshape(-1)
    elif row_seqlens is None:
        seq_t = torch.full((bsz,), T, device=device, dtype=torch.long)
    else:
        seq_t = torch.full((bsz,), int(row_seqlens), device=device, dtype=torch.long)
    j = torch.arange(T, device=device, dtype=torch.long)
    global_pos = sp_t.unsqueeze(1) + j.unsqueeze(0)
    block_in_seq = global_pos // eb
    in_block = global_pos % eb
    in_capacity = block_in_seq < int(block_table.shape[1])
    safe_block = torch.where(in_capacity, block_in_seq, torch.zeros_like(block_in_seq))
    block_id = block_table[:bsz].to(torch.long).gather(1, safe_block)
    valid = (j.unsqueeze(0) < seq_t.unsqueeze(1)) & in_capacity & (block_id > 0)
    slot = torch.where(valid, block_id * eb + in_block, torch.full_like(in_block, -1))
    return slot.reshape(-1)


@unittest.skipUnless(torch.cuda.is_available() and _TRITON_AVAILABLE, "requires CUDA Triton")
class TestMetadataTriton(unittest.TestCase):
    def test_cp_topk_helpers_match_reference(self):
        global_positions = torch.tensor([0, 1, 5, 8, 12, 15], device="cuda", dtype=torch.long)
        got_window = build_cp_window_topk_idxs(
            global_positions, bsz=2, seq_len_total=13, window_size=5
        )
        exp_window = _ref_cp_window(global_positions, 2, 13, 5)
        torch.testing.assert_close(got_window, exp_window, rtol=0, atol=0)

        got_compress = build_cp_compress_topk_idxs(
            global_positions, bsz=2, seq_len_total=13, ratio=4, offset=20
        )
        exp_compress = _ref_cp_compress(global_positions, 2, 13, 4, 20)
        torch.testing.assert_close(got_compress, exp_compress, rtol=0, atol=0)

    def test_pool_slot_helpers_match_reference(self):
        block_table = torch.tensor(
            [[3, 4, -1, 6], [7, 0, 8, 9]], device="cuda", dtype=torch.int32
        )
        valid, safe_slot = build_pool_slots(block_table, bsz=2, T=18, eb=4)
        exp_valid, exp_safe_slot = _ref_pool_slots(block_table, 2, 18, 4)
        torch.testing.assert_close(valid, exp_valid, rtol=0, atol=0)
        torch.testing.assert_close(safe_slot, exp_safe_slot, rtol=0, atol=0)

        sp = torch.tensor([3, 10], device="cuda", dtype=torch.long)
        row_seqlens = torch.tensor([5, 2], device="cuda", dtype=torch.long)
        got = build_swa_pool_slot_mapping(
            block_table, bsz=2, T=6, eb=4, sp=sp, row_seqlens=row_seqlens
        )
        exp = _ref_swa_slot_mapping(block_table, 2, 6, 4, sp, row_seqlens)
        torch.testing.assert_close(got, exp, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
