# -*- coding: utf-8 -*-
"""
Unit tests for chunk_gated_delta_rule_fwd_cp_zigzag correctness.

Verifies that CP zigzag output matches single-GPU reference for both
fixed-batch and variable-length (cu_seqlens) cases.

Layout: each rank holds two halves (seg0 + seg1) per sequence:
  - seg0 sits at causal position = rank   (physical [rank*half, (rank+1)*half))
  - seg1 sits at causal position = 2*cp_size-1-rank
         (physical [seq_len-(rank+1)*half, seq_len-rank*half))

"""

import logging
import multiprocessing as mp
import os
import unittest
from typing import List

import torch
import torch.nn.functional as F

from rtp_llm.test.utils.port_util import PortManager

logging.basicConfig(level=logging.INFO, format="%(message)s")


# ---------------------------------------------------------------------------
# Helpers: zigzag layout
# ---------------------------------------------------------------------------


def _zigzag_seg_starts(seq_len: int, cp_size: int, rank: int) -> tuple:
    """Return (seg0_start, seg1_start, half) physical offsets in the full
    sequence for this rank's two halves."""
    half = seq_len // (2 * cp_size)
    seg0_start = rank * half
    seg1_start = seq_len - (rank + 1) * half
    return seg0_start, seg1_start, half


def _build_local_from_full(
    full: torch.Tensor, seq_lengths: List[int], cp_size: int, rank: int
) -> torch.Tensor:
    """Slice each sequence's zigzag halves out of `full` (shape [1, T_total, ...])
    and concat them in [seg0_seq0, seg1_seq0, seg0_seq1, seg1_seq1, ...] order
    used by the CP zigzag kernel input (each rank's seg0/seg1 interleaved per seq)."""
    parts = []
    offset = 0
    for sl in seq_lengths:
        s0, s1, half = _zigzag_seg_starts(sl, cp_size, rank)
        parts.append(full[:, offset + s0 : offset + s0 + half])
        parts.append(full[:, offset + s1 : offset + s1 + half])
        offset += sl
    return torch.cat(parts, dim=1).contiguous()


# ---------------------------------------------------------------------------
# Worker: fixed-batch test
# ---------------------------------------------------------------------------


def _worker_fixed_batch(rank, world_size, nccl_port, B, T, H, K, V):
    import torch.distributed as dist

    from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule_fwd
    from rtp_llm.models_py.triton_kernels.fla.cp.chunk_cp_zigzag import (
        build_segment_cu_seqlens,
        chunk_gated_delta_rule_fwd_cp_zigzag,
        zigzag_causal_order,
    )

    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(nccl_port)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        dtype = torch.bfloat16

        torch.manual_seed(42)
        scale = K**-0.5

        # Single B=1 sequence of length T (zigzag splits per sequence).
        # Stack B sequences of equal length into a varlen layout.
        seq_lengths = [T] * B
        T_total = T * B

        q_full = torch.randn(1, T_total, H, K, dtype=dtype, device=device)
        k_full = F.normalize(
            torch.randn(1, T_total, H, K, dtype=torch.float32, device=device),
            p=2,
            dim=-1,
        ).to(dtype)
        v_full = torch.randn(1, T_total, H, V, dtype=dtype, device=device)
        g_full = F.logsigmoid(torch.rand(1, T_total, H, dtype=dtype, device=device))
        beta_full = torch.rand(1, T_total, H, dtype=dtype, device=device).sigmoid()
        h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=device)

        full_cu = torch.zeros(B + 1, dtype=torch.long, device=device)
        for i, sl in enumerate(seq_lengths):
            full_cu[i + 1] = full_cu[i] + sl

        # Build local zigzag inputs for this rank
        q_l = _build_local_from_full(q_full, seq_lengths, world_size, rank)
        k_l = _build_local_from_full(k_full, seq_lengths, world_size, rank)
        v_l = _build_local_from_full(v_full, seq_lengths, world_size, rank)
        g_l = _build_local_from_full(g_full, seq_lengths, world_size, rank)
        b_l = _build_local_from_full(beta_full, seq_lengths, world_size, rank)

        T_local = T // world_size
        local_cu = torch.zeros(B + 1, dtype=torch.long, device=device)
        for i in range(B):
            local_cu[i + 1] = local_cu[i] + T_local
        seg_cu = build_segment_cu_seqlens(local_cu)
        causal_order = torch.tensor(
            zigzag_causal_order(world_size),
            dtype=torch.long,
            device=device,
        )

        # Reference: single-GPU on full sequence
        _, o_ref, _, fs_ref, _, _, _ = chunk_gated_delta_rule_fwd(
            q=q_full.clone(),
            k=k_full.clone(),
            v=v_full.clone(),
            g=g_full.clone(),
            beta=beta_full.clone(),
            scale=scale,
            initial_state=h0.clone(),
            output_final_state=True,
            cu_seqlens=full_cu,
        )

        # CP zigzag
        o_z, _, fs_z = chunk_gated_delta_rule_fwd_cp_zigzag(
            q=q_l.clone(),
            k=k_l.clone(),
            v=v_l.clone(),
            g=g_l.clone(),
            beta=b_l.clone(),
            scale=scale,
            initial_state=h0.clone(),
            output_final_state=True,
            cp_group=dist.group.WORLD,
            cu_seqlens=local_cu,
            seg_cu=seg_cu,
            causal_order=causal_order,
        )

        # Compare per-sequence per-segment
        o_diff = 0.0
        local_offset = 0
        full_offset = 0
        for sl in seq_lengths:
            s0, s1, half = _zigzag_seg_starts(sl, world_size, rank)
            # seg0
            d0 = (
                (
                    o_z[:, local_offset : local_offset + half].float()
                    - o_ref[:, full_offset + s0 : full_offset + s0 + half].float()
                )
                .abs()
                .max()
                .item()
            )
            # seg1
            d1 = (
                (
                    o_z[:, local_offset + half : local_offset + 2 * half].float()
                    - o_ref[:, full_offset + s1 : full_offset + s1 + half].float()
                )
                .abs()
                .max()
                .item()
            )
            o_diff = max(o_diff, d0, d1)
            local_offset += 2 * half
            full_offset += sl

        # final_state: cp_merge runs to global tail on every rank, so all ranks
        # should hold the same final_state; check it.
        fs_diff = (
            (fs_z.float() - fs_ref.float()).abs().max().item()
            if fs_z is not None
            else 0.0
        )
        passed = max(o_diff, fs_diff) < 1e-2

        dist.barrier()
        logging.info(
            f"  rank {rank}: o={o_diff:.6f} fs={fs_diff:.6f} "
            f"{'PASS' if passed else 'FAIL'}"
        )
        dist.barrier()
        torch.cuda.synchronize()
        dist.destroy_process_group()

        assert passed, f"rank {rank} failed: o={o_diff:.6f} fs={fs_diff:.6f}"

    except Exception as e:
        print(f"Rank {rank} error: {e}")
        import traceback

        traceback.print_exc()
        raise


# ---------------------------------------------------------------------------
# Worker: varlen test
# ---------------------------------------------------------------------------


def _worker_varlen(rank, world_size, nccl_port, seq_lengths, H, K, V):
    import torch.distributed as dist

    from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule_fwd
    from rtp_llm.models_py.triton_kernels.fla.cp.chunk_cp_zigzag import (
        build_segment_cu_seqlens,
        chunk_gated_delta_rule_fwd_cp_zigzag,
        zigzag_causal_order,
    )

    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(nccl_port)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        dtype = torch.bfloat16

        torch.manual_seed(42)
        N = len(seq_lengths)
        T_total = sum(seq_lengths)
        scale = K**-0.5

        q_full = torch.randn(1, T_total, H, K, dtype=dtype, device=device)
        k_full = F.normalize(
            torch.randn(1, T_total, H, K, dtype=torch.float32, device=device),
            p=2,
            dim=-1,
        ).to(dtype)
        v_full = torch.randn(1, T_total, H, V, dtype=dtype, device=device)
        g_full = F.logsigmoid(torch.rand(1, T_total, H, dtype=dtype, device=device))
        beta_full = torch.rand(1, T_total, H, dtype=dtype, device=device).sigmoid()
        h0 = torch.randn(N, H, K, V, dtype=torch.float32, device=device)

        full_cu = torch.zeros(N + 1, dtype=torch.long, device=device)
        for i, sl in enumerate(seq_lengths):
            full_cu[i + 1] = full_cu[i] + sl

        q_l = _build_local_from_full(q_full, seq_lengths, world_size, rank)
        k_l = _build_local_from_full(k_full, seq_lengths, world_size, rank)
        v_l = _build_local_from_full(v_full, seq_lengths, world_size, rank)
        g_l = _build_local_from_full(g_full, seq_lengths, world_size, rank)
        b_l = _build_local_from_full(beta_full, seq_lengths, world_size, rank)

        local_lengths = [sl // world_size for sl in seq_lengths]
        local_cu = torch.zeros(N + 1, dtype=torch.long, device=device)
        for i, ll in enumerate(local_lengths):
            local_cu[i + 1] = local_cu[i] + ll
        seg_cu = build_segment_cu_seqlens(local_cu)
        causal_order = torch.tensor(
            zigzag_causal_order(world_size),
            dtype=torch.long,
            device=device,
        )

        _, o_ref, _, fs_ref, _, _, _ = chunk_gated_delta_rule_fwd(
            q=q_full.clone(),
            k=k_full.clone(),
            v=v_full.clone(),
            g=g_full.clone(),
            beta=beta_full.clone(),
            scale=scale,
            initial_state=h0.clone(),
            output_final_state=True,
            cu_seqlens=full_cu,
        )
        o_z, _, fs_z = chunk_gated_delta_rule_fwd_cp_zigzag(
            q=q_l.clone(),
            k=k_l.clone(),
            v=v_l.clone(),
            g=g_l.clone(),
            beta=b_l.clone(),
            scale=scale,
            initial_state=h0.clone(),
            output_final_state=True,
            cp_group=dist.group.WORLD,
            cu_seqlens=local_cu,
            seg_cu=seg_cu,
            causal_order=causal_order,
        )

        o_diff = 0.0
        local_offset = 0
        full_offset = 0
        for sl in seq_lengths:
            s0, s1, half = _zigzag_seg_starts(sl, world_size, rank)
            d0 = (
                (
                    o_z[:, local_offset : local_offset + half].float()
                    - o_ref[:, full_offset + s0 : full_offset + s0 + half].float()
                )
                .abs()
                .max()
                .item()
            )
            d1 = (
                (
                    o_z[:, local_offset + half : local_offset + 2 * half].float()
                    - o_ref[:, full_offset + s1 : full_offset + s1 + half].float()
                )
                .abs()
                .max()
                .item()
            )
            o_diff = max(o_diff, d0, d1)
            local_offset += 2 * half
            full_offset += sl

        fs_diff = (
            (fs_z.float() - fs_ref.float()).abs().max().item()
            if fs_z is not None
            else 0.0
        )
        passed = max(o_diff, fs_diff) < 1e-2

        dist.barrier()
        logging.info(
            f"  rank {rank}: o={o_diff:.6f} fs={fs_diff:.6f} "
            f"{'PASS' if passed else 'FAIL'}"
        )
        dist.barrier()
        torch.cuda.synchronize()
        dist.destroy_process_group()

        assert passed, f"rank {rank} failed: o={o_diff:.6f} fs={fs_diff:.6f}"

    except Exception as e:
        print(f"Rank {rank} error: {e}")
        import traceback

        traceback.print_exc()
        raise


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

# CP zigzag's exchange_conv_context currently asserts cp_size==2; the kernel
# itself supports general cp_size, but we test cp=2 to match production usage.
GPU_COUNT = int(os.environ.get("GPU_COUNT", "2"))


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.device_count() >= GPU_COUNT,
    f"Requires >= {GPU_COUNT} GPUs",
)
class TestChunkCPZigzag(unittest.TestCase):

    def setUp(self):
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        self.port_manager = PortManager()

    def _run_fixed_batch(self, B, T, H=16, K=128, V=128):
        ports, locks = self.port_manager.get_consecutive_ports(1)
        nccl_port = ports[0]
        try:
            processes = []
            for rank in range(GPU_COUNT):
                p = mp.Process(
                    target=_worker_fixed_batch,
                    args=(rank, GPU_COUNT, nccl_port, B, T, H, K, V),
                    name=f"rank-{rank}",
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join(timeout=120)
                if p.exitcode != 0:
                    raise RuntimeError(
                        f"Process {p.name} exited with code {p.exitcode}"
                    )
        finally:
            for lock in locks:
                lock.__exit__(None, None, None)

    def _run_varlen(self, seq_lengths, H=16, K=128, V=128):
        ports, locks = self.port_manager.get_consecutive_ports(1)
        nccl_port = ports[0]
        try:
            processes = []
            for rank in range(GPU_COUNT):
                p = mp.Process(
                    target=_worker_varlen,
                    args=(rank, GPU_COUNT, nccl_port, seq_lengths, H, K, V),
                    name=f"rank-{rank}",
                )
                p.start()
                processes.append(p)
            for p in processes:
                p.join(timeout=120)
                if p.exitcode != 0:
                    raise RuntimeError(
                        f"Process {p.name} exited with code {p.exitcode}"
                    )
        finally:
            for lock in locks:
                lock.__exit__(None, None, None)

    # --- Fixed batch (single seq per batch) ---

    def test_fixed_batch_256(self):
        self._run_fixed_batch(B=1, T=256)

    def test_fixed_batch_1024(self):
        self._run_fixed_batch(B=1, T=1024)

    def test_fixed_batch_8192(self):
        self._run_fixed_batch(B=1, T=8192)

    def test_fixed_batch_32k(self):
        self._run_fixed_batch(B=1, T=32768)

    def test_fixed_batch_64k(self):
        self._run_fixed_batch(B=1, T=65536)

    # --- Varlen ---

    def test_varlen_two_equal(self):
        self._run_varlen([4096, 4096])

    def test_varlen_two_unequal(self):
        self._run_varlen([8192, 16384])

    def test_varlen_three(self):
        self._run_varlen([4096, 8192, 4096])


if __name__ == "__main__":
    unittest.main()
