# -*- coding: utf-8 -*-
"""
Unit tests for chunk_gated_delta_rule_fwd_cp_scan correctness.

Verifies that CP scan output matches single-GPU reference for both
fixed-batch and variable-length (cu_seqlens) cases.

Run with:
  bazel test //rtp_llm/models_py/triton_kernels/fla/test:test_chunk_cp_scan_ut
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
# Worker: fixed-batch test
# ---------------------------------------------------------------------------


def _worker_fixed_batch(rank, world_size, nccl_port, B, T, H, K, V):
    import torch.distributed as dist

    from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule_fwd
    from rtp_llm.models_py.triton_kernels.fla.chunk_cp_scan import (
        chunk_gated_delta_rule_fwd_cp_scan,
    )

    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(nccl_port)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        dtype = torch.bfloat16

        torch.manual_seed(42)
        T_local = T // world_size
        scale = K**-0.5

        q_full = torch.randn(B, T, H, K, dtype=dtype, device=device)
        k_full = F.normalize(
            torch.randn(B, T, H, K, dtype=torch.float32, device=device), p=2, dim=-1
        ).to(dtype)
        v_full = torch.randn(B, T, H, V, dtype=dtype, device=device)
        g_full = F.logsigmoid(torch.rand(B, T, H, dtype=dtype, device=device))
        beta_full = torch.rand(B, T, H, dtype=dtype, device=device).sigmoid()
        h0 = torch.randn(B, H, K, V, dtype=torch.float32, device=device)

        q_l = q_full[:, rank * T_local : (rank + 1) * T_local].clone()
        k_l = k_full[:, rank * T_local : (rank + 1) * T_local].clone()
        v_l = v_full[:, rank * T_local : (rank + 1) * T_local].clone()
        g_l = g_full[:, rank * T_local : (rank + 1) * T_local].clone()
        b_l = beta_full[:, rank * T_local : (rank + 1) * T_local].clone()

        g_ref, o_ref, _, fs_ref, _, h_ref, _ = chunk_gated_delta_rule_fwd(
            q=q_full.clone(),
            k=k_full.clone(),
            v=v_full.clone(),
            g=g_full.clone(),
            beta=beta_full.clone(),
            scale=scale,
            initial_state=h0.clone(),
            output_final_state=True,
        )
        o_s, h_s, fs_s = chunk_gated_delta_rule_fwd_cp_scan(
            q=q_l.clone(),
            k=k_l.clone(),
            v=v_l.clone(),
            g=g_l.clone(),
            beta=b_l.clone(),
            scale=scale,
            initial_state=h0.clone(),
            output_final_state=True,
            cp_group=dist.group.WORLD,
        )

        NT_l = T_local // 64
        o_diff = (
            (o_s.float() - o_ref[:, rank * T_local : (rank + 1) * T_local].float())
            .abs()
            .max()
            .item()
        )
        h_diff = (
            (h_s.float() - h_ref[:, rank * NT_l : (rank + 1) * NT_l].float())
            .abs()
            .max()
            .item()
        )
        fs_diff = (
            (fs_s.float() - fs_ref.float()).abs().max().item()
            if rank == world_size - 1
            else 0.0
        )
        passed = max(o_diff, h_diff, fs_diff) < 1e-2

        dist.barrier()
        logging.info(
            f"  rank {rank}: o={o_diff:.6f} h={h_diff:.6f} fs={fs_diff:.6f} {'PASS' if passed else 'FAIL'}"
        )
        dist.barrier()
        torch.cuda.synchronize()
        dist.destroy_process_group()

        assert (
            passed
        ), f"rank {rank} failed: o={o_diff:.6f} h={h_diff:.6f} fs={fs_diff:.6f}"

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
    from rtp_llm.models_py.triton_kernels.fla.chunk_cp_scan import (
        chunk_gated_delta_rule_fwd_cp_scan,
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

        local_lengths = [sl // world_size for sl in seq_lengths]
        local_slices = []
        for i in range(N):
            seq_start = full_cu[i].item()
            ll = local_lengths[i]
            local_slices.append(
                slice(seq_start + rank * ll, seq_start + rank * ll + ll)
            )

        q_l = torch.cat([q_full[:, s] for s in local_slices], dim=1).contiguous()
        k_l = torch.cat([k_full[:, s] for s in local_slices], dim=1).contiguous()
        v_l = torch.cat([v_full[:, s] for s in local_slices], dim=1).contiguous()
        g_l = torch.cat([g_full[:, s] for s in local_slices], dim=1).contiguous()
        b_l = torch.cat([beta_full[:, s] for s in local_slices], dim=1).contiguous()

        local_cu = torch.zeros(N + 1, dtype=torch.long, device=device)
        for i in range(N):
            local_cu[i + 1] = local_cu[i] + local_lengths[i]

        g_ref, o_ref, _, fs_ref, _, _, _ = chunk_gated_delta_rule_fwd(
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
        o_s, _, fs_s = chunk_gated_delta_rule_fwd_cp_scan(
            q=q_l,
            k=k_l,
            v=v_l,
            g=g_l,
            beta=b_l,
            scale=scale,
            initial_state=h0.clone(),
            output_final_state=True,
            cp_group=dist.group.WORLD,
            cu_seqlens=local_cu,
        )

        o_diff = 0.0
        local_offset = 0
        for i in range(N):
            ll = local_lengths[i]
            ref_start = full_cu[i].item() + rank * ll
            o_d = (
                (
                    o_s[:, local_offset : local_offset + ll].float()
                    - o_ref[:, ref_start : ref_start + ll].float()
                )
                .abs()
                .max()
                .item()
            )
            o_diff = max(o_diff, o_d)
            local_offset += ll

        fs_diff = 0.0
        if rank == world_size - 1 and fs_s is not None:
            fs_diff = (fs_s.float() - fs_ref.float()).abs().max().item()

        passed = max(o_diff, fs_diff) < 1e-2

        dist.barrier()
        logging.info(
            f"  rank {rank}: o={o_diff:.6f} fs={fs_diff:.6f} {'PASS' if passed else 'FAIL'}"
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

GPU_COUNT = int(os.environ.get("GPU_COUNT", "4"))


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.device_count() >= GPU_COUNT,
    f"Requires >= {GPU_COUNT} GPUs",
)
class TestChunkCPScan(unittest.TestCase):

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

    # --- Fixed batch ---

    def test_fixed_batch_256(self):
        self._run_fixed_batch(B=1, T=256)

    def test_fixed_batch_1024(self):
        self._run_fixed_batch(B=1, T=1024)

    def test_fixed_batch_32k(self):
        self._run_fixed_batch(B=1, T=32768)

    def test_fixed_batch_64k(self):
        self._run_fixed_batch(B=1, T=65536)

    # --- Varlen ---

    def test_varlen_two_equal(self):
        self._run_varlen([8192, 8192])

    def test_varlen_two_unequal(self):
        self._run_varlen([16384, 32768])

    def test_varlen_three(self):
        self._run_varlen([8192, 16384, 8192])


if __name__ == "__main__":
    unittest.main()
