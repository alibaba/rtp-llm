"""Parity and microbenchmark suite for Track B-1 prefill CP, pool reader, and SWA Triton kernels."""

from __future__ import annotations

import statistics
import time
import unittest
from types import SimpleNamespace
from typing import Callable, Dict, Tuple
from unittest import mock

import torch
from torch.profiler import ProfilerActivity

from rtp_llm.models_py.modules.dsv4.cp import (
    CPContext,
    build_cp_context,
    cp_actual_owned_kv_lens,
    cp_gather_last_by_request,
    cp_padded_local_kv_lens,
)
from rtp_llm.models_py.modules.dsv4.fp8._cp_attention_shard import (
    build_swa_cp_local_indices,
    remap_topk_to_cp_local,
)
from rtp_llm.models_py.modules.dsv4.fp8._indexer_cp_assembler import (
    _pack_indexer_k_payload,
    _restore_indexer_k_packed,
    build_indexer_cp_chunk_plan,
    copy_actual_indexer_k_to_padded,
)
from rtp_llm.models_py.modules.dsv4.fp8._pool_reader import (
    _pack_padded_to_flat,
    _scatter_flat_to_workspace,
    make_compressed_k_pool_reader,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_cp_byte_sliced import (
    CPByteSlicedSlotCompaction,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
    ENTRY_BYTES,
    HEAD_DIM,
    dequantize_packed_k_cache_flat,
    direct_triton_fast_path_supported,
    try_gather_k_cache_packed_to_flat,
    try_restore_dequantize_scatter_packed_k_cache_flat,
)
from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
    quantize_and_insert_k_cache,
    quantize_and_insert_k_cache_cp_byte_sliced,
)


def _ref_remap_topk_to_cp_local(
    topk_indices: torch.Tensor,
    per_req_total_kv_lens: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    block_size: int,
    req_id_per_token: torch.Tensor,
    cu_cmp_offsets: torch.Tensor,
) -> torch.Tensor:
    req = req_id_per_token.to(torch.int64)
    req_base = cu_cmp_offsets.to(torch.int64).index_select(0, req).unsqueeze(1)
    idx = topk_indices.to(torch.int64)
    valid = idx >= 0
    safe_idx = torch.where(valid, idx, torch.zeros_like(idx))
    block = safe_idx // block_size
    token_in_block = safe_idx % block_size
    owner = block % cp_size
    local_block = block // cp_size
    local_pos = local_block * block_size + token_in_block
    req_len = per_req_total_kv_lens.to(torch.int64).index_select(0, req).unsqueeze(1)
    owned = valid & (owner == cp_rank) & (safe_idx < req_len)
    return torch.where(owned, req_base + local_pos, torch.full_like(idx, -1)).to(
        torch.int32
    )


def _ref_build_swa_cp_local_indices(
    global_positions: torch.Tensor,
    *,
    prefix_lengths: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    window_size: int,
    M: int,
    N: int,
    req_id_per_token: torch.Tensor,
    owner_chunk_size: int,
    cu_swa_offsets: torch.Tensor | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = global_positions.device
    T = int(global_positions.numel())
    req = req_id_per_token.to(torch.int64)
    gp = global_positions.to(torch.int64)
    prefix = prefix_lengths.to(torch.int64)
    prefix_per_row = prefix.index_select(0, req)
    p_per_row = torch.clamp(prefix_per_row, max=window_size - 1)
    gather_start_per_row = prefix_per_row - p_per_row
    swa_len_per_row = torch.clamp(gp + 1, max=window_size)
    key_start_per_row = gp - swa_len_per_row + 1
    col_offset = torch.arange(window_size, device=device, dtype=torch.int64).unsqueeze(
        0
    )
    key_pos = key_start_per_row.unsqueeze(1) + col_offset
    valid_in_window = col_offset < swa_len_per_row.unsqueeze(1)
    owned_mask = (
        ((key_pos // owner_chunk_size) % cp_size) == cp_rank
    ) & valid_in_window
    if cu_swa_offsets is not None:
        swa_ws_base = cu_swa_offsets.to(torch.int64).index_select(0, req).unsqueeze(1)
    else:
        swa_ws_base = req.unsqueeze(1) * M + N
    target_value = swa_ws_base + (key_pos - gather_start_per_row.unsqueeze(1))
    rank_in_row = owned_mask.cumsum(dim=1) - 1
    row_idx_grid = (
        torch.arange(T, device=device, dtype=torch.int64)
        .unsqueeze(1)
        .expand(-1, window_size)
    )
    flat_mask = owned_mask.reshape(-1)
    flat_row = row_idx_grid.reshape(-1).masked_select(flat_mask)
    flat_col = rank_in_row.reshape(-1).masked_select(flat_mask)
    flat_val = target_value.reshape(-1).masked_select(flat_mask).to(torch.int32)
    indices = torch.full((T, window_size), -1, dtype=torch.int32, device=device)
    if flat_row.numel() > 0:
        indices.index_put_((flat_row, flat_col), flat_val, accumulate=False)
    lens = owned_mask.sum(dim=1).to(torch.int32)
    return indices, lens


def _make_cp_ctx(
    input_lens: list[int],
    prefix_lens: list[int],
    cp_size: int,
    cp_rank: int,
    device: torch.device,
) -> CPContext:
    B = len(input_lens)
    align = 2 * cp_size
    chunk_lens = [((x + align - 1) // align) * 2 for x in input_lens]
    chunk_length = sum(chunk_lens)
    restore_list: list[int] = []
    # Build padded concat restore in global request order matching cp.py layout.
    # For cp_gather_last_by_request + build_indexer_cp_chunk_plan we only need
    # consistent chunk/global metadata.
    padded_chunks: list[int] = []
    rank_local_gpos: list[int] = []
    rank_local_req: list[int] = []
    rank_local_real: list[bool] = []
    pad_off = 0
    for r, (c_len, real_len, pref) in enumerate(
        zip(chunk_lens, input_lens, prefix_lens)
    ):
        global_req_tokens = c_len * cp_size
        half = c_len // 2
        front_s = cp_rank * half
        back_s = global_req_tokens - (cp_rank + 1) * half
        for local_pos in list(range(front_s, front_s + half)) + list(
            range(back_s, back_s + half)
        ):
            is_real = local_pos < real_len
            rank_local_real.append(is_real)
            rank_local_req.append(r)
            clamped = local_pos if is_real else max(real_len - 1, 0)
            rank_local_gpos.append(pref + clamped)
        for i in range(real_len):
            restore_list.append(pad_off + i)
        pad_off += global_req_tokens
        padded_chunks.append(pad_off)
    del padded_chunks
    seq_len_full = sum(input_lens)
    cp_info = SimpleNamespace(
        prefill_cp_chunk_lengths=torch.tensor(chunk_lens, dtype=torch.int32),
        prefill_actual_input_lengths_cpu=torch.tensor(input_lens, dtype=torch.int32),
    )
    return CPContext(
        cp_size=cp_size,
        cp_rank=cp_rank,
        chunk_length=chunk_length,
        padded_seq_len=cp_size * chunk_length,
        seq_len_full=seq_len_full,
        relative_positions=torch.arange(chunk_length, dtype=torch.int64, device=device),
        prefix_length=prefix_lens[0] if B == 1 else 0,
        global_positions=torch.tensor(
            rank_local_gpos, dtype=torch.int64, device=device
        ),
        local_is_real=torch.tensor(rank_local_real, dtype=torch.bool, device=device),
        unpad_restore=torch.tensor(restore_list, dtype=torch.int64, device=device),
        seq_len_total=sum(p + s for p, s in zip(prefix_lens, input_lens)),
        cp_info=cp_info,
        req_id_per_token=torch.tensor(rank_local_req, dtype=torch.int32, device=device),
        prefix_lengths=torch.tensor(prefix_lens, dtype=torch.int64, device=device),
        input_lengths_global=torch.tensor(input_lens, dtype=torch.int32, device=device),
        cu_seqlens_global=torch.tensor(
            [0] + list(torch.tensor(input_lens).cumsum(0).tolist()),
            dtype=torch.int32,
            device=device,
        ),
        chunk_lengths_per_req=tuple(chunk_lens),
        kv_cache_sharded=True,
        input_lengths_global_host=tuple(input_lens),
        prefix_lengths_host=tuple(prefix_lens),
    )


def _measure_op(
    fn: Callable[[], object], *, warmup: int = 5, iters: int = 25
) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_ev = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_ev = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    e2e_us: list[float] = []
    for i in range(iters):
        t0 = time.perf_counter()
        start_ev[i].record()
        fn()
        end_ev[i].record()
        e2e_us.append((time.perf_counter() - t0) * 1e6)
    torch.cuda.synchronize()
    gpu_us = [start_ev[i].elapsed_time(end_ev[i]) * 1000.0 for i in range(iters)]

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        fn()
        torch.cuda.synchronize()

    events = prof.events()
    cuda_kernels = sum(
        1
        for ev in events
        if getattr(ev, "device_type", None) == torch.autograd.DeviceType.CUDA
    )
    launch_calls = sum(
        1 for ev in events if ev.name in ("cudaLaunchKernel", "cudaLaunchKernelExC")
    )
    memcpy_dtod = sum(1 for ev in events if "Memcpy" in ev.name and "DtoD" in ev.name)
    memset_calls = sum(1 for ev in events if "Memset" in ev.name)
    d2h_syncs = sum(
        1
        for ev in events
        if ev.name in ("cudaStreamSynchronize", "cudaEventSynchronize")
        or ("Memcpy" in ev.name and "DtoH" in ev.name)
    )

    return {
        "gpu_us": statistics.median(gpu_us),
        "e2e_us": statistics.median(e2e_us),
        "kernels": float(cuda_kernels),
        "launches": float(launch_calls),
        "dtod": float(memcpy_dtod),
        "memset": float(memset_calls),
        "d2h_sync": float(d2h_syncs),
    }


class TrackB1PrefillCpPoolSwaTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        self.device = torch.device("cuda:0")
        torch.manual_seed(42)

    def test_parity_all_trackb1_ops(self) -> None:
        dev = self.device

        # 1. remap_topk_to_cp_local: ragged B=4 + invalid -1 + out-of-range
        B, T, K = 4, 256, 256
        per_req = torch.tensor([311, 64, 513, 129], dtype=torch.int32, device=dev)
        req_id = torch.repeat_interleave(
            torch.arange(B, dtype=torch.int32, device=dev), T // B
        )
        cu_cmp = torch.tensor([0, 96, 128, 288, 352], dtype=torch.int32, device=dev)
        topk = torch.randint(-8, 600, (T, K), dtype=torch.int32, device=dev)
        for cp_size in (2, 4):
            for cp_rank in range(cp_size):
                for bs in (64, 256):
                    for cu_arg in (None, cu_cmp):
                        cu_eff = (
                            cu_cmp[:B]
                            if cu_arg is not None
                            else torch.cat(
                                [
                                    torch.zeros(1, dtype=torch.int64, device=dev),
                                    cp_padded_local_kv_lens(
                                        per_req, cp_size, bs
                                    ).cumsum(0)[:-1],
                                ]
                            )
                        )
                        ref = _ref_remap_topk_to_cp_local(
                            topk, per_req, cp_size, cp_rank, bs, req_id, cu_eff
                        )
                        got = remap_topk_to_cp_local(
                            topk,
                            per_req_total_kv_lens=per_req,
                            cp_size=cp_size,
                            cp_rank=cp_rank,
                            block_size=bs,
                            req_id_per_token=req_id,
                            cu_cmp_offsets=cu_arg,
                        )
                        self.assertTrue(torch.equal(got, ref))

        # 2. build_swa_cp_local_indices: ragged B=3, multiple window/chunk sizes
        prefix_lens = torch.tensor([0, 320, 700], dtype=torch.int64, device=dev)
        input_lens = [64, 150, 90]
        gp_list: list[torch.Tensor] = []
        req_list: list[torch.Tensor] = []
        for r, (p, s) in enumerate(zip(prefix_lens.tolist(), input_lens)):
            gp_list.append(torch.arange(p, p + s, dtype=torch.int64, device=dev))
            req_list.append(torch.full((s,), r, dtype=torch.int32, device=dev))
        gp = torch.cat(gp_list)
        req_swa = torch.cat(req_list)
        cu_swa = torch.tensor([0, 128, 800, 1400], dtype=torch.int32, device=dev)
        for win in (128, 512):
            for chunk_sz in (1, 64):
                for cu_arg in (None, cu_swa):
                    for cp_rank in (0, 3):
                        ref_idx, ref_len = _ref_build_swa_cp_local_indices(
                            gp,
                            prefix_lengths=prefix_lens,
                            cp_size=4,
                            cp_rank=cp_rank,
                            window_size=win,
                            M=800,
                            N=200,
                            req_id_per_token=req_swa,
                            owner_chunk_size=chunk_sz,
                            cu_swa_offsets=cu_arg,
                        )
                        got_idx, got_len = build_swa_cp_local_indices(
                            gp,
                            prefix_lengths=prefix_lens,
                            cp_size=4,
                            cp_rank=cp_rank,
                            window_size=win,
                            M=800,
                            N=200,
                            req_id_per_token=req_swa,
                            owner_chunk_size=chunk_sz,
                            cu_swa_offsets=cu_arg,
                        )
                        self.assertTrue(torch.equal(got_len, ref_len))
                        self.assertTrue(torch.equal(got_idx, ref_idx))

        # 3. quantize_and_insert_k_cache + quantize_and_insert_k_cache_cp_byte_sliced parity
        eb = 256
        cp_size = 4
        full_stride = ((eb * ENTRY_BYTES + 575) // 576) * 576
        local_slice = full_stride // cp_size
        k_in = torch.randn((310, 512), dtype=torch.bfloat16, device=dev) * 3.5
        unique_blocks = torch.tensor([2, 0], dtype=torch.int64, device=dev)
        compact_slots = torch.arange(310, dtype=torch.int64, device=dev)
        compact_slots[17] = -1
        compaction = CPByteSlicedSlotCompaction(
            unique_blocks=unique_blocks,
            compact_slots=compact_slots,
            gather_lens_cpu=(),
            contiguous_block_start=-1,
            gather_lens=torch.empty(0, dtype=torch.int32, device=dev),
        )
        full_pool = torch.zeros((4, full_stride), dtype=torch.uint8, device=dev)
        full_slots = compact_slots.clone()
        for idx_tok in range(310):
            if full_slots[idx_tok] >= 0:
                blk = int(full_slots[idx_tok] // eb)
                pos = int(full_slots[idx_tok] % eb)
                full_slots[idx_tok] = int(unique_blocks[blk]) * eb + pos
        quantize_and_insert_k_cache(
            k_in,
            full_pool.as_strided((4, eb, ENTRY_BYTES), (full_stride, ENTRY_BYTES, 1)),
            full_slots,
        )
        for cp_rank in range(cp_size):
            raw_pool = torch.full((4, local_slice), 0x5A, dtype=torch.uint8, device=dev)
            quantize_and_insert_k_cache_cp_byte_sliced(
                k_in,
                raw_pool,
                full_slots,
                full_entries_per_block=eb,
                cp_rank=cp_rank,
                cp_size=cp_size,
                compaction=compaction,
            )
            s0 = cp_rank * local_slice
            s1 = s0 + local_slice
            self.assertTrue(
                torch.equal(
                    raw_pool.index_select(0, unique_blocks),
                    full_pool.index_select(0, unique_blocks)[:, s0:s1],
                )
            )

        # 4. SWA gather packed + dequant + restore_dequantize_scatter parity
        max_blocks = 3
        bt = torch.tensor(
            [[0, 1, 2], [3, 0, 1], [2, 3, 0]], dtype=torch.int32, device=dev
        )
        k_pool = torch.zeros((4, eb, ENTRY_BYTES), dtype=torch.uint8, device=dev)
        src_k = torch.randn((4 * eb, 512), dtype=torch.bfloat16, device=dev) * 2.0
        quantize_and_insert_k_cache(
            src_k, k_pool, torch.arange(4 * eb, dtype=torch.int64, device=dev)
        )
        if direct_triton_fast_path_supported(dev):
            per_req_tot = torch.tensor([600, 310, 512], dtype=torch.int64, device=dev)
            padded_l = cp_padded_local_kv_lens(per_req_tot, 4, eb).to(torch.int32)
            actual_l = cp_actual_owned_kv_lens(per_req_tot, 4, eb, 0).to(torch.int32)
            flat = torch.empty(
                (int(padded_l.sum().item()), ENTRY_BYTES), dtype=torch.uint8, device=dev
            )
            ok = try_gather_k_cache_packed_to_flat(
                flat,
                k_pool,
                bt,
                padded_l,
                actual_l,
                block_size=eb,
                has_actual_tokens=True,
            )
            self.assertTrue(ok)
            deq = torch.empty(
                (flat.shape[0], HEAD_DIM), dtype=torch.bfloat16, device=dev
            )
            dequantize_packed_k_cache_flat(deq, flat)
            # Parity vs single-slot baseline for non-zero rows
            restore_idx = torch.arange(flat.shape[0], dtype=torch.int64, device=dev)
            seq_l = padded_l.clone()
            cu_dst = torch.tensor([0, 300, 700, 1200], dtype=torch.int32, device=dev)
            ws_out = torch.empty((1600, HEAD_DIM), dtype=torch.bfloat16, device=dev)
            ok_r = try_restore_dequantize_scatter_packed_k_cache_flat(
                ws_out,
                flat,
                restore_idx,
                seq_l,
                offset=5,
                seq_lens_total=int(seq_l.sum().item()),
                cu_dst_offsets=cu_dst,
            )
            self.assertTrue(ok_r)
            off = 0
            for r in range(3):
                ln = int(seq_l[r].item())
                self.assertTrue(
                    torch.equal(
                        ws_out[int(cu_dst[r]) + 5 : int(cu_dst[r]) + 5 + ln],
                        deq[off : off + ln],
                    )
                )
                off += ln

        # 5. Indexer CP assembler parity
        cp_ctx = _make_cp_ctx([64, 96], [0, 0], cp_size=4, cp_rank=1, device=dev)
        plan = build_indexer_cp_chunk_plan(
            cp_ctx,
            torch.tensor([310, 140], dtype=torch.int64, device=dev),
            block_size=64,
            device=dev,
        )
        act_q = torch.randint(
            0, 255, (plan.total_actual_local_T, 128), dtype=torch.uint8, device=dev
        )
        act_s = torch.randint(
            0, 255, (plan.total_actual_local_T, 4), dtype=torch.uint8, device=dev
        )
        pad_q = torch.zeros((plan.total_local_T, 128), dtype=torch.uint8, device=dev)
        pad_s = torch.zeros((plan.total_local_T, 4), dtype=torch.uint8, device=dev)
        copy_actual_indexer_k_to_padded(
            plan=plan,
            actual_k_quant=act_q,
            actual_k_scale=act_s,
            padded_k_quant=pad_q,
            padded_k_scale=pad_s,
        )
        packed, qc, sc = _pack_indexer_k_payload(pad_q, pad_s)
        gathered_p = packed.repeat(4, 1)
        out_q = torch.empty(
            (int(plan.restore_indices.numel()), qc), dtype=torch.uint8, device=dev
        )
        out_s = torch.empty(
            (int(plan.restore_indices.numel()), sc), dtype=torch.uint8, device=dev
        )
        _restore_indexer_k_packed(
            plan=plan,
            gathered_packed=gathered_p,
            q_cols=qc,
            s_cols=sc,
            out_k_quant=out_q,
            out_k_scale=out_s,
        )
        self.assertTrue(
            torch.equal(out_q, gathered_p.index_select(0, plan.restore_indices)[:, :qc])
        )
        self.assertTrue(
            torch.equal(
                out_s, gathered_p.index_select(0, plan.restore_indices)[:, qc : qc + sc]
            )
        )

        # 6. _pack_padded_to_flat + _scatter_flat_to_workspace (ragged 2D & 3D) parity
        lens_p = torch.tensor([11, 0, 29, 7], dtype=torch.int32, device=dev)
        tot_p = int(lens_p.sum().item())
        padded_3d = torch.randint(
            0, 255, (4, 32, ENTRY_BYTES), dtype=torch.uint8, device=dev
        )
        flat_got = _pack_padded_to_flat(padded_3d, lens_p, tot_p, ENTRY_BYTES)
        flat_ref = torch.cat([padded_3d[r, : int(lens_p[r])] for r in range(4)], dim=0)
        self.assertTrue(torch.equal(flat_got, flat_ref))

        src_bf16 = torch.randn((tot_p, HEAD_DIM), dtype=torch.bfloat16, device=dev)
        ws_3d_got = torch.zeros((4, 40, HEAD_DIM), dtype=torch.bfloat16, device=dev)
        ws_3d_ref = torch.zeros_like(ws_3d_got)
        _scatter_flat_to_workspace(src_bf16, ws_3d_got, lens_p, 3)
        pos_off = 0
        for r, ln in enumerate(lens_p.tolist()):
            if ln > 0:
                ws_3d_ref[r, 3 : 3 + ln] = src_bf16[pos_off : pos_off + ln]
            pos_off += ln
        self.assertTrue(torch.equal(ws_3d_got, ws_3d_ref))

        cu_dst_p = torch.tensor([0, 20, 25, 60], dtype=torch.int32, device=dev)
        ws_2d_got = torch.zeros((100, HEAD_DIM), dtype=torch.bfloat16, device=dev)
        ws_2d_ref = torch.zeros_like(ws_2d_got)
        _scatter_flat_to_workspace(
            src_bf16, ws_2d_got, lens_p, 2, cu_dst_offsets=cu_dst_p
        )
        pos_off = 0
        for r, ln in enumerate(lens_p.tolist()):
            if ln > 0:
                ws_2d_ref[int(cu_dst_p[r]) + 2 : int(cu_dst_p[r]) + 2 + ln] = src_bf16[
                    pos_off : pos_off + ln
                ]
            pos_off += ln
        self.assertTrue(torch.equal(ws_2d_got, ws_2d_ref))

        # 7. cp_gather_last_by_request parity
        cp_last = _make_cp_ctx(
            [13, 21, 8, 17], [4, 0, 9, 2], cp_size=1, cp_rank=0, device=dev
        )
        loc_last = torch.randn(
            (cp_last.chunk_length, 7168), dtype=torch.bfloat16, device=dev
        )
        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.cp.all_gather",
            side_effect=lambda t, group=None: t,
        ):
            got_last = cp_gather_last_by_request(loc_last, cp_last)
        c0, c1, c2, c3 = cp_last.chunk_lengths_per_req
        self.assertTrue(torch.equal(got_last[0], loc_last[12]))
        self.assertTrue(torch.equal(got_last[1], loc_last[c0 + 20]))
        self.assertTrue(torch.equal(got_last[2], loc_last[c0 + c1 + 7]))
        self.assertTrue(torch.equal(got_last[3], loc_last[c0 + c1 + c2 + 16]))

        # 8. Zero-sync CPU path in make_compressed_k_pool_reader + sync/kernel assertions
        per_req_cpu = torch.tensor([512, 1024, 256, 768], dtype=torch.int64)
        cp4_p = _make_cp_ctx(
            [256, 256, 256, 256],
            [512, 1024, 256, 768],
            cp_size=4,
            cp_rank=0,
            device=dev,
        )
        m_cpu = _measure_op(
            lambda: make_compressed_k_pool_reader(
                cp4_p, True, per_req_cpu, block_size=256
            ),
            warmup=1,
            iters=5,
        )
        self.assertEqual(m_cpu["d2h_sync"], 0.0)
        reader_cpu = make_compressed_k_pool_reader(
            cp4_p, True, per_req_cpu, block_size=256
        )
        self.assertEqual(reader_cpu.cfg.seq_lens_total, 2560)

        # Verify single-kernel & zero-D2H-sync contracts on key optimized ops
        m_remap = _measure_op(
            lambda: remap_topk_to_cp_local(
                topk,
                per_req_total_kv_lens=per_req,
                cp_size=4,
                cp_rank=1,
                block_size=64,
                req_id_per_token=req_id,
                cu_cmp_offsets=cu_cmp,
            ),
            warmup=2,
            iters=5,
        )
        self.assertEqual(m_remap["kernels"], 1.0)
        self.assertEqual(m_remap["d2h_sync"], 0.0)

        m_swa = _measure_op(
            lambda: build_swa_cp_local_indices(
                gp,
                prefix_lengths=prefix_lens,
                cp_size=4,
                cp_rank=1,
                window_size=512,
                req_id_per_token=req_swa,
                owner_chunk_size=1,
                cu_swa_offsets=cu_swa,
            ),
            warmup=2,
            iters=5,
        )
        self.assertEqual(m_swa["kernels"], 1.0)
        self.assertEqual(m_swa["d2h_sync"], 0.0)

        m_idx_cpy = _measure_op(
            lambda: copy_actual_indexer_k_to_padded(
                plan=plan,
                actual_k_quant=act_q,
                actual_k_scale=act_s,
                padded_k_quant=pad_q,
                padded_k_scale=pad_s,
            ),
            warmup=2,
            iters=5,
        )
        self.assertEqual(m_idx_cpy["kernels"], 1.0)

        m_sc_chk = _measure_op(
            lambda: _scatter_flat_to_workspace(
                src_bf16, ws_2d_got, lens_p, 2, cu_dst_offsets=cu_dst_p
            ),
            warmup=2,
            iters=5,
        )
        self.assertEqual(m_sc_chk["kernels"], 1.0)

    def test_benchmark_trackb1_prefill_ops(self) -> None:
        dev = self.device
        print(f"\n[Track B-1 Microbenchmark] device={torch.cuda.get_device_name(dev)}")

        for T in (4096, 16384):
            # A. remap_topk_to_cp_local (B=4, CP=4, K=2048, bs=64)
            B = 4
            K = 2048
            per_req = torch.full((B,), T // B, dtype=torch.int32, device=dev)
            req_id = torch.repeat_interleave(
                torch.arange(B, dtype=torch.int32, device=dev), T // B
            )
            cu_cmp = torch.arange(B + 1, dtype=torch.int32, device=dev) * (
                (T // B) // 4
            )
            topk = torch.randint(-1, T // B, (T, K), dtype=torch.int32, device=dev)
            m = _measure_op(
                lambda: remap_topk_to_cp_local(
                    topk,
                    per_req_total_kv_lens=per_req,
                    cp_size=4,
                    cp_rank=1,
                    block_size=64,
                    req_id_per_token=req_id,
                    cu_cmp_offsets=cu_cmp,
                )
            )
            print(
                f"remap_topk        T={T:5d} B=4 CP=4 K=2048 | "
                f"gpu={m['gpu_us']:7.1f}us e2e={m['e2e_us']:7.1f}us "
                f"kernels={m['kernels']:.0f} launches={m['launches']:.0f} "
                f"memset={m['memset']:.0f} sync={m['d2h_sync']:.0f}"
            )

        for T in (4096, 8192):
            # B. build_swa_cp_local_indices (B=4, CP=4, W=512)
            B = 4
            seq = T // B
            gp = torch.arange(seq, dtype=torch.int64, device=dev).repeat(B)
            prefix = torch.zeros((B,), dtype=torch.int64, device=dev)
            req = torch.repeat_interleave(
                torch.arange(B, dtype=torch.int32, device=dev), seq
            )
            cu_swa = torch.arange(B, dtype=torch.int32, device=dev) * seq
            m = _measure_op(
                lambda: build_swa_cp_local_indices(
                    gp,
                    prefix_lengths=prefix,
                    cp_size=4,
                    cp_rank=1,
                    window_size=512,
                    req_id_per_token=req,
                    owner_chunk_size=1,
                    cu_swa_offsets=cu_swa,
                )
            )
            print(
                f"build_swa_cp_idx  T={T:5d} B=4 CP=4 W=512  | "
                f"gpu={m['gpu_us']:7.1f}us e2e={m['e2e_us']:7.1f}us "
                f"kernels={m['kernels']:.0f} launches={m['launches']:.0f} "
                f"memset={m['memset']:.0f} sync={m['d2h_sync']:.0f}"
            )

        eb = 256
        full_stride = ((eb * ENTRY_BYTES + 575) // 576) * 576
        local_slice = full_stride // 4
        for T in (4096, 16384):
            # C. quantize_and_insert_k_cache & cp_byte_sliced
            num_blk = (T + eb - 1) // eb
            k_in = torch.randn((T, 512), dtype=torch.bfloat16, device=dev)
            slots = torch.arange(T, dtype=torch.int64, device=dev)
            k_pool_3d = torch.empty(
                (num_blk, full_stride), dtype=torch.uint8, device=dev
            ).as_strided((num_blk, eb, ENTRY_BYTES), (full_stride, ENTRY_BYTES, 1))
            m_ins = _measure_op(
                lambda: quantize_and_insert_k_cache(k_in, k_pool_3d, slots)
            )
            print(
                f"swa_kv_insert     T={T:5d} eb=256        | "
                f"gpu={m_ins['gpu_us']:7.1f}us e2e={m_ins['e2e_us']:7.1f}us "
                f"kernels={m_ins['kernels']:.0f} launches={m_ins['launches']:.0f} "
                f"memset={m_ins['memset']:.0f}"
            )

            raw_pool = torch.zeros(
                (num_blk + 4, local_slice), dtype=torch.uint8, device=dev
            )
            comp = CPByteSlicedSlotCompaction(
                unique_blocks=torch.arange(num_blk, dtype=torch.int64, device=dev),
                compact_slots=slots,
                gather_lens_cpu=(),
                contiguous_block_start=-1,
                gather_lens=torch.empty(0, dtype=torch.int32, device=dev),
            )
            m_bs = _measure_op(
                lambda: quantize_and_insert_k_cache_cp_byte_sliced(
                    k_in,
                    raw_pool,
                    slots,
                    full_entries_per_block=eb,
                    cp_rank=1,
                    cp_size=4,
                    compaction=comp,
                )
            )
            print(
                f"swa_insert_bslice T={T:5d} CP=4 eb=256   | "
                f"gpu={m_bs['gpu_us']:7.1f}us e2e={m_bs['e2e_us']:7.1f}us "
                f"kernels={m_bs['kernels']:.0f} launches={m_bs['launches']:.0f} "
                f"memset={m_bs['memset']:.0f}"
            )

        if direct_triton_fast_path_supported(dev):
            for T in (4096, 16384):
                # D. SWA packed gather + dequant + restore_dequant_scatter
                B = 4
                per_req_tot = torch.full((B,), T, dtype=torch.int64, device=dev)
                padded_l = cp_padded_local_kv_lens(per_req_tot, 4, eb).to(torch.int32)
                actual_l = cp_actual_owned_kv_lens(per_req_tot, 4, eb, 1).to(
                    torch.int32
                )
                tot_local = int(padded_l.sum().item())
                nblk_per_req = (T // 4 + eb - 1) // eb
                bt = torch.arange(B * nblk_per_req, dtype=torch.int32, device=dev).view(
                    B, nblk_per_req
                )
                k_pool = torch.zeros(
                    (B * nblk_per_req, eb, ENTRY_BYTES), dtype=torch.uint8, device=dev
                )
                flat = torch.empty(
                    (tot_local, ENTRY_BYTES), dtype=torch.uint8, device=dev
                )
                m_g = _measure_op(
                    lambda: try_gather_k_cache_packed_to_flat(
                        flat,
                        k_pool,
                        bt,
                        padded_l,
                        actual_l,
                        block_size=eb,
                        has_actual_tokens=True,
                    )
                )
                print(
                    f"swa_gather_flat   T={T:5d} B=4 CP=4      | "
                    f"gpu={m_g['gpu_us']:7.1f}us e2e={m_g['e2e_us']:7.1f}us "
                    f"kernels={m_g['kernels']:.0f} launches={m_g['launches']:.0f}"
                )

                gathered = torch.empty((T, ENTRY_BYTES), dtype=torch.uint8, device=dev)
                deq_out = torch.empty((T, HEAD_DIM), dtype=torch.bfloat16, device=dev)
                m_d = _measure_op(
                    lambda: dequantize_packed_k_cache_flat(deq_out, gathered)
                )
                print(
                    f"swa_dequant_flat  N={T:5d}               | "
                    f"gpu={m_d['gpu_us']:7.1f}us e2e={m_d['e2e_us']:7.1f}us "
                    f"kernels={m_d['kernels']:.0f} launches={m_d['launches']:.0f}"
                )

                restore_idx = torch.arange(T, dtype=torch.int64, device=dev)
                seq_l = torch.full((B,), T // B, dtype=torch.int32, device=dev)
                cu_dst = torch.arange(B, dtype=torch.int32, device=dev) * (T // B)
                ws_out = torch.empty((T, HEAD_DIM), dtype=torch.bfloat16, device=dev)
                m_r = _measure_op(
                    lambda: try_restore_dequantize_scatter_packed_k_cache_flat(
                        ws_out,
                        gathered,
                        restore_idx,
                        seq_l,
                        offset=0,
                        seq_lens_total=T,
                        cu_dst_offsets=cu_dst,
                    )
                )
                print(
                    f"swa_restore_deq   N={T:5d} B=4           | "
                    f"gpu={m_r['gpu_us']:7.1f}us e2e={m_r['e2e_us']:7.1f}us "
                    f"kernels={m_r['kernels']:.0f} launches={m_r['launches']:.0f}"
                )

        # E. Indexer CP assembler (copy_actual_to_padded + pack + restore_packed)
        for T in (4096, 16384):
            B = 4
            per_req = torch.full((B,), T, dtype=torch.int64, device=dev)
            cp_ctx = _make_cp_ctx([256] * B, [0] * B, cp_size=4, cp_rank=1, device=dev)
            plan = build_indexer_cp_chunk_plan(
                cp_ctx, per_req, block_size=64, device=dev
            )
            act_q = torch.empty(
                (plan.total_actual_local_T, 128), dtype=torch.uint8, device=dev
            )
            act_s = torch.empty(
                (plan.total_actual_local_T, 4), dtype=torch.uint8, device=dev
            )
            pad_q = torch.empty(
                (plan.total_local_T, 128), dtype=torch.uint8, device=dev
            )
            pad_s = torch.empty((plan.total_local_T, 4), dtype=torch.uint8, device=dev)
            m_cpy = _measure_op(
                lambda: copy_actual_indexer_k_to_padded(
                    plan=plan,
                    actual_k_quant=act_q,
                    actual_k_scale=act_s,
                    padded_k_quant=pad_q,
                    padded_k_scale=pad_s,
                )
            )
            m_pk = _measure_op(lambda: _pack_indexer_k_payload(pad_q, pad_s))
            gathered_p = torch.empty(
                (plan.total_local_T * 4, 132), dtype=torch.uint8, device=dev
            )
            out_q = torch.empty(
                (int(plan.restore_indices.numel()), 128), dtype=torch.uint8, device=dev
            )
            out_s = torch.empty(
                (int(plan.restore_indices.numel()), 4), dtype=torch.uint8, device=dev
            )
            m_rst = _measure_op(
                lambda: _restore_indexer_k_packed(
                    plan=plan,
                    gathered_packed=gathered_p,
                    q_cols=128,
                    s_cols=4,
                    out_k_quant=out_q,
                    out_k_scale=out_s,
                )
            )
            print(
                f"idx_copy_padded   T={T*4:5d}               | "
                f"gpu={m_cpy['gpu_us']:7.1f}us e2e={m_cpy['e2e_us']:7.1f}us "
                f"kernels={m_cpy['kernels']:.0f} launches={m_cpy['launches']:.0f}"
            )
            print(
                f"idx_pack          T={T:5d}               | "
                f"gpu={m_pk['gpu_us']:7.1f}us e2e={m_pk['e2e_us']:7.1f}us "
                f"kernels={m_pk['kernels']:.0f} launches={m_pk['launches']:.0f}"
            )
            print(
                f"idx_restore_pack  T={int(plan.restore_indices.numel()):5d}               | "
                f"gpu={m_rst['gpu_us']:7.1f}us e2e={m_rst['e2e_us']:7.1f}us "
                f"kernels={m_rst['kernels']:.0f} launches={m_rst['launches']:.0f}"
            )

        # F. _scatter_flat_to_workspace + cp_gather_last_by_request + make_compressed_k_pool_reader
        T = 8192
        B = 4
        seq_l = torch.full((B,), T // B, dtype=torch.int32, device=dev)
        cu_dst = torch.arange(B, dtype=torch.int32, device=dev) * (T // B)
        restored = torch.empty((T, HEAD_DIM), dtype=torch.bfloat16, device=dev)
        ws_out = torch.empty((T, HEAD_DIM), dtype=torch.bfloat16, device=dev)
        m_sc = _measure_op(
            lambda: _scatter_flat_to_workspace(
                restored, ws_out, seq_l, 0, cu_dst_offsets=cu_dst
            )
        )
        print(
            f"pool_scatter_ws   T={T:5d} B=4           | "
            f"gpu={m_sc['gpu_us']:7.1f}us e2e={m_sc['e2e_us']:7.1f}us "
            f"kernels={m_sc['kernels']:.0f} launches={m_sc['launches']:.0f}"
        )

        cp1 = _make_cp_ctx(
            [1024, 1024, 1024, 1024], [0, 0, 0, 0], cp_size=1, cp_rank=0, device=dev
        )
        loc_2d = torch.empty((cp1.chunk_length, 7168), dtype=torch.bfloat16, device=dev)
        with mock.patch(
            "rtp_llm.models_py.modules.dsv4.cp.all_gather",
            side_effect=lambda t, group=None: t,
        ):
            m_last = _measure_op(lambda: cp_gather_last_by_request(loc_2d, cp1))
        print(
            f"cp_gather_last    T={cp1.chunk_length:5d} B=4 H=7168    | "
            f"gpu={m_last['gpu_us']:7.1f}us e2e={m_last['e2e_us']:7.1f}us "
            f"kernels={m_last['kernels']:.0f} launches={m_last['launches']:.0f} "
            f"memset={m_last['memset']:.0f}"
        )

        # make_compressed_k_pool_reader D2H sync count check
        cp4 = _make_cp_ctx(
            [256, 256, 256, 256],
            [512, 1024, 256, 768],
            cp_size=4,
            cp_rank=0,
            device=dev,
        )
        per_req_gpu = torch.tensor([512, 1024, 256, 768], dtype=torch.int64, device=dev)
        m_mk = _measure_op(
            lambda: make_compressed_k_pool_reader(
                cp4, True, per_req_gpu, block_size=256, total_kv_len=2560
            ),
            warmup=2,
            iters=10,
        )
        print(
            f"make_pool_reader  B=4 GPU lens          | e2e={m_mk['e2e_us']:7.1f}us sync={m_mk['d2h_sync']:.0f}"
        )


if __name__ == "__main__":
    unittest.main()
