"""Fixed-size PCCL transport for the native grouped W4A8 Decode GEMMs."""

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.dsv41.ppu.moe import (
    COMM_SLOTS,
    ep_gather,
    quantize_per_row_fp8,
    routed_swiglu_fp8,
)


class GroupedW4A8Graph:
    """Model-owned communication buffers, serialized by the forward stream.

    The graph uses the same row capacity on every peer, including a peer with
    no requests. Gather BF16 inputs and routes, evaluate only local experts
    and sum the BF16 rank-local contributions as FP32. This transport uses
    the fixed-size collectives already used by Graph Engram.
    """

    def __init__(self, engine, example, layer_ids):
        import deep_gemm

        self.engine = engine
        self.group = engine.group
        if self.group is None or (self.group.size(), self.group.rank()) != (
            engine.ep_size, engine.ep_rank,
        ):
            raise ValueError("grouped Graph requires the model EP process group")
        if not example.is_cuda or not engine._use_grouped_gemm(example):
            raise ValueError("grouped W4A8 Graph requires device DeepGEMM")
        self.gemm = deep_gemm.m_grouped_gemm_fp8_fp4_bf16_nt_nopad
        self.capacity = example.shape[0]
        self.local_x = torch.empty_like(example)
        self.local_ids = example.new_empty((self.capacity, COMM_SLOTS), dtype=torch.int64)
        self.local_weights = torch.empty_like(self.local_ids, dtype=torch.float32)
        rows = self.capacity * engine.ep_size
        self.recv_x = example.new_empty((rows, engine.hidden_size))
        self.recv_ids = self.local_ids.new_empty((rows, COMM_SLOTS))
        self.recv_weights = torch.empty_like(self.recv_ids, dtype=torch.float32)
        self.assignment_rows = torch.arange(rows, device=example.device).repeat_interleave(COMM_SLOTS)
        self.assignment_ordinals = torch.arange(rows * COMM_SLOTS, device=example.device)
        self.experts = torch.arange(engine.num_local_experts, device=example.device)
        self.received = torch.zeros(
            (max(layer_ids) + 1, engine.num_local_experts), dtype=torch.int32, device=example.device
        )
        self.gateup = example.new_empty((rows * COMM_SLOTS, 2 * engine.inter_dim))
        self.down = example.new_empty((rows * COMM_SLOTS, engine.hidden_size))
        self.local_y = torch.empty_like(self.recv_x)
        self.combine = torch.empty_like(self.recv_x, dtype=torch.float32)
        self.ones = torch.ones_like(self.recv_weights)
        self.weights = {
            layer: engine._processed_weights(layer, example.device) for layer in layer_ids
        }

    def forward(self, x, ids, weights, *, layer_id, expert_load):
        engine = self.engine
        rows = x.shape[0]
        if rows > self.capacity or ids.shape != (rows, COMM_SLOTS):
            raise ValueError("grouped Graph input exceeds fixed EP capacity or route width")
        self.local_x[rows:].zero_()
        self.local_ids[rows:].fill_(-1)
        self.local_weights[rows:].zero_()
        self.local_x[:rows].copy_(x)
        self.local_ids[:rows].copy_(ids)
        self.local_weights[:rows].copy_(weights)
        dist.all_gather_into_tensor(self.recv_x, self.local_x, group=self.group)
        dist.all_gather_into_tensor(self.recv_ids, self.local_ids, group=self.group)
        dist.all_gather_into_tensor(self.recv_weights, self.local_weights, group=self.group)

        local = self.recv_ids - engine.local_expert_start
        real = (local >= 0) & (local < engine.num_local_experts)
        real_ids = torch.where(real, local, -1)
        counts = (real_ids.reshape(-1, 1) == self.experts).sum(dim=0, dtype=torch.int32)
        self.received[layer_id].copy_(counts)
        if expert_load is not None:
            engine._expert_load_counts(expert_load, x.device).add_(counts)

        # Use the last existing expert for zero work slots, preserving the
        # native nopad ABI (sum(m_rows) == physical M even on an empty owner).
        # Stats above count real assignments only. No host sizes or IDs enter.
        last = engine.num_local_experts - 1
        groups = torch.where(real, local, last).flatten()
        order = torch.argsort(groups, stable=True)
        m_indices = groups[order].to(torch.int32)
        m_rows = counts.clone()
        m_rows[-1].add_(groups.numel() - counts.sum(dtype=torch.int32))
        grouped_live = real.flatten()[order]
        grouped_x = self.recv_x[self.assignment_rows[order]]
        grouped_x = torch.where(grouped_live[:, None], grouped_x, 0)
        grouped_weights = torch.where(grouped_live, self.recv_weights.flatten()[order], 0)
        w13, s13, w2, s2, inv13, inv2 = self.weights[layer_id]
        self.gemm(
            quantize_per_row_fp8(grouped_x), (w13, s13), self.gateup,
            m_indices, m_rows, inv_scale=inv13,
        )
        self.gemm(
            routed_swiglu_fp8(self.gateup, grouped_weights, engine.swiglu_limit),
            (w2, s2), self.down, m_indices, m_rows, inv_scale=inv2,
        )

        # Reuse the eager gather's original top-k summation order and BF16
        # rank-local boundary. Invalid IDs skip their output index entirely.
        inverse = torch.empty_like(order).scatter_(0, order, self.assignment_ordinals)
        ep_gather(
            self.down, real_ids.int(), self.ones,
            inverse.reshape_as(real_ids).int(), self.local_y,
        )
        self.combine.copy_(self.local_y)
        dist.all_reduce(self.combine, group=self.group)
        start = engine.ep_rank * self.capacity
        return self.combine[start : start + rows].to(x.dtype)
