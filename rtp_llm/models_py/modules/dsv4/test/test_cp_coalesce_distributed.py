"""Two-rank distributed NCCL parity & microbenchmark test for W4-1 CP all_gather coalescing."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from datetime import timedelta
from types import SimpleNamespace
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.profiler import ProfilerActivity, profile

from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    acquire_grow_only_buffer,
    all_gather,
    all_gather_into,
    clear_grow_only_buffers,
)
from rtp_llm.models_py.modules.dsv4.cp import (
    CoalescedCPGatherHandle,
    build_cp_context,
    cp_all_gather_coalesced,
    cp_all_gather_full_varlen,
    cp_restore_from_coalesced,
    prepare_cp_all_gather_coalesced,
    start_cp_all_gather_coalesced,
    wait_cp_all_gather_coalesced,
)
from rtp_llm.models_py.utils.memory import dispose_tensor

_WORLD_SIZE = 2


def _install_world_as_tp_group() -> None:
    collective_torch._group_map.clear()
    collective_torch._group_map[Group.WORLD] = dist.group.WORLD
    collective_torch._parallelism_config = SimpleNamespace(
        tp_size=_WORLD_SIZE,
        dp_size=1,
        pp_size=1,
        world_size=_WORLD_SIZE,
    )
    collective_torch._initialized = True


def _clear_collective_torch_state() -> None:
    clear_grow_only_buffers()
    collective_torch._group_map.clear()
    collective_torch._parallelism_config = None
    collective_torch._initialized = False


def _make_cp_context(
    rank: int,
    device: torch.device,
    *,
    per_req_lens: Tuple[int, ...],
    prefix_mode: bool,
) -> object:
    world_size = _WORLD_SIZE
    block = 2 * world_size
    chunk_lengths = [((l + block - 1) // block) * 2 for l in per_req_lens]
    chunk_length = sum(chunk_lengths)
    padded_seq_len = world_size * chunk_length
    seq_len_full = sum(per_req_lens)

    if prefix_mode:
        assert len(per_req_lens) == 1
        # Keep on CPU so build_cp_context hits the unpad_restore_is_prefix=True path.
        padding_mask = torch.zeros((padded_seq_len,), dtype=torch.int32)
        padding_mask[:seq_len_full] = 1
        restore_indices = torch.arange(padded_seq_len, dtype=torch.int32)
        cp_info = SimpleNamespace(
            prefill_qkv_padding_mask=padding_mask,
            prefill_qkv_restore_indice=restore_indices,
            prefill_actual_input_lengths_cpu=torch.tensor(
                per_req_lens, dtype=torch.int32
            ),
            prefill_cp_chunk_lengths=torch.tensor(chunk_lengths, dtype=torch.int32),
            prefill_shuffle_indices=None,
        )
        return build_cp_context(
            cp_info=cp_info,
            cp_size=world_size,
            cp_rank=rank,
            chunk_length=chunk_length,
            device=device,
        )

    # Multi-request non-prefix zigzag metadata matching cp.py build_cp_context L940-956.
    padding_mask = torch.zeros((padded_seq_len,), dtype=torch.int32)
    restore_indices = torch.zeros((padded_seq_len,), dtype=torch.int32)
    cu_chunks = [0]
    for c in chunk_lengths:
        cu_chunks.append(cu_chunks[-1] + c)

    pad_off = 0
    for req_id, (r_len, r_chunk) in enumerate(zip(per_req_lens, chunk_lengths)):
        r_padded = r_chunk * world_size
        pair = r_chunk // 2
        req_chunk_off = cu_chunks[req_id]
        for t in range(r_len):
            seg = t // pair
            in_seg = t % pair
            if seg < world_size:
                owner_rank = seg
                local_in_chunk = in_seg
            else:
                owner_rank = (2 * world_size - 1) - seg
                local_in_chunk = pair + in_seg
            src_slot = owner_rank * chunk_length + req_chunk_off + local_in_chunk
            padding_mask[pad_off + t] = 1
            restore_indices[pad_off + t] = src_slot
        pad_off += r_padded

    cp_info = SimpleNamespace(
        prefill_qkv_padding_mask=padding_mask,
        prefill_qkv_restore_indice=restore_indices,
        prefill_actual_input_lengths_cpu=torch.tensor(per_req_lens, dtype=torch.int32),
        prefill_cp_chunk_lengths=torch.tensor(chunk_lengths, dtype=torch.int32),
        prefill_shuffle_indices=None,
    )
    return build_cp_context(
        cp_info=cp_info,
        cp_size=world_size,
        cp_rank=rank,
        chunk_length=chunk_length,
        device=device,
    )


def _parity_and_bench_worker(rank: int, world_size: int, init_file: str) -> None:
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    try:
        dist.init_process_group(
            backend="nccl",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=120),
        )
        _install_world_as_tp_group()
        torch.manual_seed(1000 + rank)

        # ------------------------------------------------------------------
        # 1. Parity: col2d layout (same row count, 16B-aligned row widths)
        # ------------------------------------------------------------------
        t_local = 16
        a_col = torch.randn((t_local, 512), dtype=torch.float32, device=device)
        b_col = torch.randn((t_local, 2048), dtype=torch.float32, device=device)
        c_col = torch.randn((t_local, 512), dtype=torch.bfloat16, device=device)

        ref_col = tuple(all_gather(t, group=Group.TP) for t in (a_col, b_col, c_col))
        sync_col = cp_all_gather_coalesced(
            (a_col, b_col, c_col),
            group=Group.TP,
            allow_col2d=True,
            slot_key="ut.col2d",
        )
        for idx, (got, ref) in enumerate(zip(sync_col, ref_col)):
            if not torch.equal(got, ref):
                raise AssertionError(f"rank {rank} col2d sync mismatch idx={idx}")

        gather_stream = torch.cuda.Stream(device=device)
        post_stream = torch.cuda.Stream(device=device)
        h_col = start_cp_all_gather_coalesced(
            (a_col, b_col, c_col),
            group=Group.TP,
            async_op=True,
            stream=gather_stream,
            allow_col2d=True,
            slot_key="ut.col2d.async",
        )
        if h_col.layout != "col2d":
            raise AssertionError(f"expected col2d layout, got {h_col.layout}")
        prepare_cp_all_gather_coalesced(h_col, stream=post_stream)
        async_col = wait_cp_all_gather_coalesced(h_col)
        for idx, (got, ref) in enumerate(zip(async_col, ref_col)):
            if not torch.equal(got, ref):
                raise AssertionError(f"rank {rank} col2d async mismatch idx={idx}")

        # ------------------------------------------------------------------
        # 2. Parity: flat1d mixed shapes/dtypes, different row counts,
        #    unaligned row widths, >8 tensors (chunking), and N=0 tensors
        # ------------------------------------------------------------------
        idx_local = torch.randint(0, 250, (17, 132), dtype=torch.uint8, device=device)
        cmp_local = torch.randint(0, 250, (31, 584), dtype=torch.uint8, device=device)
        swa_local = torch.randint(0, 250, (5, 2920), dtype=torch.uint8, device=device)
        unaligned_f32 = torch.randn((11, 17), dtype=torch.float32, device=device)
        odd_u8 = torch.randint(0, 250, (3, 7), dtype=torch.uint8, device=device)
        state_i32 = torch.randint(-100, 100, (3, 19), dtype=torch.int32, device=device)
        zero_u8 = torch.empty((0, 132), dtype=torch.uint8, device=device)
        extra_bf16 = torch.randn((4, 64), dtype=torch.bfloat16, device=device)
        extra_c64 = torch.randn((2, 8), dtype=torch.complex64, device=device)

        mixed_inputs = (
            idx_local,
            cmp_local,
            swa_local,
            unaligned_f32,
            odd_u8,
            state_i32,
            zero_u8,
            extra_bf16,
            extra_c64,
        )
        ref_mixed = tuple(all_gather(t, group=Group.TP) for t in mixed_inputs)

        h_mixed = start_cp_all_gather_coalesced(
            mixed_inputs,
            group=Group.TP,
            async_op=True,
            stream=gather_stream,
            allow_col2d=True,
            slot_key="ut.mixed",
        )
        if h_mixed.layout != "flat1d":
            raise AssertionError(f"expected flat1d layout, got {h_mixed.layout}")
        prepare_cp_all_gather_coalesced(h_mixed, stream=post_stream)
        async_mixed = wait_cp_all_gather_coalesced(h_mixed)
        for idx, (got, ref) in enumerate(zip(async_mixed, ref_mixed)):
            if got.shape != ref.shape or got.dtype != ref.dtype:
                raise AssertionError(
                    f"rank {rank} mixed metadata mismatch idx={idx}: {got.shape}/{got.dtype} vs {ref.shape}/{ref.dtype}"
                )
            if not torch.equal(got, ref):
                raise AssertionError(f"rank {rank} mixed async mismatch idx={idx}")

        sync_mixed = cp_all_gather_coalesced(
            mixed_inputs,
            group=Group.TP,
            allow_col2d=False,
            slot_key="ut.mixed.sync",
        )
        for idx, (got, ref) in enumerate(zip(sync_mixed, ref_mixed)):
            if not torch.equal(got, ref):
                raise AssertionError(f"rank {rank} mixed sync mismatch idx={idx}")

        # All-empty edge case (N=0 across every tensor)
        empty_a = torch.empty((0, 512), dtype=torch.bfloat16, device=device)
        empty_b = torch.empty((0, 132), dtype=torch.uint8, device=device)
        out_ea, out_eb = cp_all_gather_coalesced(
            (empty_a, empty_b), group=Group.TP, slot_key="ut.empty"
        )
        if out_ea.shape != (0, 512) or out_eb.shape != (0, 132):
            raise AssertionError(
                f"unexpected N=0 shapes: {out_ea.shape}, {out_eb.shape}"
            )

        # ------------------------------------------------------------------
        # 3. Varlen cp_restore_from_coalesced parity + own_prefix_view safety
        # ------------------------------------------------------------------
        for prefix_mode, req_lens in ((True, (12,)), (False, (7, 11))):
            cp_ctx = _make_cp_context(
                rank, device, per_req_lens=req_lens, prefix_mode=prefix_mode
            )
            kv_local = torch.randn(
                (cp_ctx.chunk_length, 512), dtype=torch.bfloat16, device=device
            )
            other_local = torch.randn(
                (cp_ctx.chunk_length, 1024), dtype=torch.float32, device=device
            )
            kv_ref = cp_all_gather_full_varlen(kv_local, cp_ctx)
            kv_g, other_g = cp_all_gather_coalesced(
                (kv_local, other_local),
                group=Group.TP,
                allow_col2d=True,
                slot_key="ut.varlen_restore",
            )
            kv_got = cp_restore_from_coalesced(
                kv_g, cp_ctx, (512,), own_prefix_view=True
            )
            if not torch.equal(kv_got, kv_ref):
                raise AssertionError(
                    f"rank {rank} varlen restore mismatch prefix={prefix_mode}"
                )
            other_ref = cp_all_gather_full_varlen(other_local, cp_ctx)
            other_got = cp_restore_from_coalesced(
                other_g, cp_ctx, (1024,), own_prefix_view=False
            )
            if not torch.equal(other_got, other_ref):
                raise AssertionError(
                    f"rank {rank} varlen restore non-owned mismatch prefix={prefix_mode}"
                )
            # Disposing kv_got must not touch the cached grow-only buffer.
            dispose_tensor(kv_got)
            kv_g2, _ = cp_all_gather_coalesced(
                (kv_local, other_local),
                group=Group.TP,
                allow_col2d=True,
                slot_key="ut.varlen_restore",
            )
            kv_got2 = cp_restore_from_coalesced(
                kv_g2, cp_ctx, (512,), own_prefix_view=True
            )
            if not torch.equal(kv_got2, kv_ref):
                raise AssertionError("grow-only buffer corrupted after dispose_tensor")

        # ------------------------------------------------------------------
        # 4. Grow-only buffer test + profiler zero aten::empty verification
        #    and PF_22_47 (21 CSA + 20 HCA + 2 SWA layers x 3 forwards) bench
        # ------------------------------------------------------------------
        t_chunk = 32
        idx_f = torch.randn((t_chunk, 512), dtype=torch.float32, device=device)
        main_f_csa = torch.randn((t_chunk, 2048), dtype=torch.float32, device=device)
        main_f_hca = torch.randn((t_chunk, 1024), dtype=torch.float32, device=device)
        kv_l = torch.randn((t_chunk, 512), dtype=torch.bfloat16, device=device)
        state_idx = torch.randint(0, 250, (2, 19296), dtype=torch.uint8, device=device)
        state_main = torch.randint(0, 250, (2, 19296), dtype=torch.uint8, device=device)
        idx_pool = torch.randint(0, 250, (128, 132), dtype=torch.uint8, device=device)
        cmp_pool = torch.randint(0, 250, (128, 584), dtype=torch.uint8, device=device)
        swa_pool = torch.randint(0, 250, (16, 2920), dtype=torch.uint8, device=device)

        # Warm up grow-only slots so steady state performs zero aten::empty.
        for _ in range(3):
            cp_all_gather_coalesced(
                (idx_f, main_f_csa, kv_l, state_idx, state_main),
                group=Group.TP,
                allow_col2d=False,
                slot_key="bench.csa.g1",
            )
            cp_all_gather_coalesced(
                (idx_pool, cmp_pool, swa_pool),
                group=Group.TP,
                allow_col2d=False,
                slot_key="bench.csa.g2",
            )
            cp_all_gather_coalesced(
                (main_f_hca, kv_l, state_main),
                group=Group.TP,
                allow_col2d=False,
                slot_key="bench.hca.g1",
            )
            cp_all_gather_coalesced(
                (cmp_pool, swa_pool),
                group=Group.TP,
                allow_col2d=False,
                slot_key="bench.hca.g2",
            )
            swa_dst = acquire_grow_only_buffer(
                device,
                "bench.swa.dst",
                (world_size * kv_l.shape[0], kv_l.shape[1]),
                dtype=kv_l.dtype,
            )
            all_gather_into(swa_dst, kv_l, group=Group.TP)
        torch.cuda.synchronize(device)
        dist.barrier()

        def _run_baseline_trace() -> None:
            for _fwd in range(3):
                for _ in range(21):  # 21 CSA layers
                    all_gather(idx_f, group=Group.TP)
                    all_gather(main_f_csa, group=Group.TP)
                    all_gather(kv_l, group=Group.TP)
                    all_gather(state_idx, group=Group.TP)
                    all_gather(state_main, group=Group.TP)
                    all_gather(idx_pool, group=Group.TP)
                    all_gather(cmp_pool, group=Group.TP)
                    all_gather(swa_pool, group=Group.TP)
                for _ in range(20):  # 20 HCA layers
                    all_gather(main_f_hca, group=Group.TP)
                    all_gather(kv_l, group=Group.TP)
                    all_gather(state_main, group=Group.TP)
                    all_gather(cmp_pool, group=Group.TP)
                    all_gather(swa_pool, group=Group.TP)
                for _ in range(2):  # 2 SWA layers
                    all_gather(kv_l, group=Group.TP)

        def _run_coalesced_trace() -> None:
            for _fwd in range(3):
                for _ in range(21):  # 21 CSA layers -> 2 gathers/layer
                    cp_all_gather_coalesced(
                        (idx_f, main_f_csa, kv_l, state_idx, state_main),
                        group=Group.TP,
                        allow_col2d=False,
                        slot_key="bench.csa.g1",
                    )
                    cp_all_gather_coalesced(
                        (idx_pool, cmp_pool, swa_pool),
                        group=Group.TP,
                        allow_col2d=False,
                        slot_key="bench.csa.g2",
                    )
                for _ in range(20):  # 20 HCA layers -> 2 gathers/layer
                    cp_all_gather_coalesced(
                        (main_f_hca, kv_l, state_main),
                        group=Group.TP,
                        allow_col2d=False,
                        slot_key="bench.hca.g1",
                    )
                    cp_all_gather_coalesced(
                        (cmp_pool, swa_pool),
                        group=Group.TP,
                        allow_col2d=False,
                        slot_key="bench.hca.g2",
                    )
                for _ in range(2):  # 2 SWA layers -> 1 gather/layer
                    swa_dst_local = acquire_grow_only_buffer(
                        device,
                        "bench.swa.dst",
                        (world_size * kv_l.shape[0], kv_l.shape[1]),
                        dtype=kv_l.dtype,
                    )
                    all_gather_into(swa_dst_local, kv_l, group=Group.TP)

        # Warm up baseline & Triton coalesced specializations before timing.
        _run_baseline_trace()
        _run_coalesced_trace()
        torch.cuda.synchronize(device)
        dist.barrier()

        # Unprofiled wall-clock timing pass (avoids CPU profiler hook distortion).
        t0 = time.perf_counter()
        _run_baseline_trace()
        torch.cuda.synchronize(device)
        base_ms = (time.perf_counter() - t0) * 1000.0

        dist.barrier()

        t0 = time.perf_counter()
        _run_coalesced_trace()
        torch.cuda.synchronize(device)
        coal_ms = (time.perf_counter() - t0) * 1000.0

        dist.barrier()

        # Separate CPU profiler pass to assert collective & aten::empty counts.
        with profile(activities=[ProfilerActivity.CPU]) as prof_base:
            _run_baseline_trace()
            torch.cuda.synchronize(device)

        dist.barrier()

        with profile(activities=[ProfilerActivity.CPU]) as prof_coal:
            _run_coalesced_trace()
            torch.cuda.synchronize(device)

        def _count_op(prof_obj: profile, substr: str) -> int:
            total = 0
            for evt in prof_obj.key_averages():
                if substr in evt.key:
                    total += int(evt.count)
            return total

        base_gathers = _count_op(prof_base, "_allgather_base_")
        coal_gathers = _count_op(prof_coal, "_allgather_base_")
        base_empties = _count_op(prof_base, "aten::empty")
        coal_empties = _count_op(prof_coal, "aten::empty")

        if base_gathers <= 0 or coal_gathers * 3 > base_gathers:
            raise AssertionError(
                f"rank {rank} insufficient gather reduction: base={base_gathers}, coal={coal_gathers}"
            )
        if coal_empties != 0:
            raise AssertionError(
                f"rank {rank} steady-state coalesced trace executed {coal_empties} aten::empty calls (expected 0)"
            )
        if rank == 0:
            total_layers = 3 * (21 + 20 + 2)
            print(
                f"[W4-1 BENCH] rank0 gathers: {base_gathers} -> {coal_gathers} "
                f"({base_gathers / max(coal_gathers, 1):.2f}x fewer); "
                f"aten::empty: {base_empties} -> {coal_empties}; "
                f"wall_ms: {base_ms:.2f} -> {coal_ms:.2f} "
                f"({(base_ms - coal_ms) * 1000.0 / total_layers:.1f} us/layer saved)"
            )
        dist.barrier()
    finally:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize(device)
            except Exception:
                pass
        if dist.is_initialized():
            dist.destroy_process_group()
        _clear_collective_torch_state()


class CPCoalesceDistributedTest(unittest.TestCase):
    def test_two_rank_coalesced_parity_zero_empty_and_bench(self) -> None:
        if not torch.cuda.is_available() or torch.cuda.device_count() < _WORLD_SIZE:
            self.skipTest("two CUDA devices are required")
        if not dist.is_available() or not dist.is_nccl_available():
            self.skipTest("NCCL torch.distributed backend is required")

        with tempfile.TemporaryDirectory(prefix="dsv4_cp_coalesce_dist_") as temp_dir:
            init_file = os.path.join(temp_dir, "nccl_file_store")
            mp.spawn(
                _parity_and_bench_worker,
                args=(_WORLD_SIZE, init_file),
                nprocs=_WORLD_SIZE,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
