"""RDMA collectives around node-local MegaMoE symmetric memory.

Global expert ownership remains contiguous. Peers with the same local rank
exchange padded tokens, execute their node's experts, and sum node contributions
back to the source. No symmetric-memory pointer crosses a host boundary.
"""

import os
import socket

import torch
import torch.distributed as dist
import triton

from rtp_llm.platforms.ppu.models.dsv4.ppu_mega_transport import (
    pack_tokens, unpack_tokens, sum_contributions,
)


class PpuMegaMoEHierarchy:
    LOCAL_SIZE = 8

    def __init__(self, cfg, capacity):
        self.nodes = cfg.ep_size // self.LOCAL_SIZE
        self.node = cfg.ep_rank // self.LOCAL_SIZE
        self.capacity = capacity
        self.experts = cfg.n_routed_experts // self.nodes
        hosts = [None] * cfg.ep_size
        dist.all_gather_object(hosts, socket.gethostname())
        node_hosts = [hosts[i:i + self.LOCAL_SIZE] for i in range(0, cfg.ep_size, self.LOCAL_SIZE)]
        if any(len(set(group)) != 1 for group in node_hosts) or len({group[0] for group in node_hosts}) != self.nodes:
            raise RuntimeError("MegaMoE requires contiguous groups of eight ranks on distinct hosts")

        # All WORLD ranks create every subgroup in the same order.
        for node in range(self.nodes):
            ranks = list(range(node * self.LOCAL_SIZE, (node + 1) * self.LOCAL_SIZE))
            group = dist.new_group(ranks)
            if node == self.node:
                self.local_group = group
        for local_rank in range(self.LOCAL_SIZE):
            ranks = list(range(local_rank, cfg.ep_size, self.LOCAL_SIZE))
            group = dist.new_group(ranks)
            if cfg.ep_rank % self.LOCAL_SIZE == local_rank:
                self.cross_group = group

        device = torch.device("cuda", torch.cuda.current_device())
        self.dim = cfg.dim
        self.words = cfg.dim // 2 + 12
        self.packet = torch.empty((capacity, self.words), dtype=torch.int32, device=device)
        self.packets = torch.empty((capacity * self.nodes, self.words), dtype=torch.int32, device=device)
        self.recv_x = torch.empty((capacity * self.nodes, cfg.dim), dtype=torch.bfloat16, device=device)
        self.recv_ids = torch.empty((capacity * self.nodes, 8), dtype=torch.int64, device=device)
        self.recv_weights = torch.empty((capacity * self.nodes, 8), dtype=torch.float32, device=device)
        self.local_ids = torch.empty_like(self.recv_ids)
        self.local_weights = torch.empty_like(self.recv_weights)
        self.parts = torch.empty_like(self.recv_x)
        self.reduced = torch.empty((capacity, cfg.dim), dtype=torch.bfloat16, device=device)
        transport = os.environ.get("DSV4_PPU_MEGAMOE_TRANSPORT", "collective")
        if transport not in ("collective", "sparse"):
            raise ValueError("MegaMoE transport must be collective or sparse")
        self.sparse = None
        if transport == "sparse":
            if cfg.ep_size != 32 or getattr(cfg, "tp_size", 1) != 1:
                raise ValueError("Sparse MegaMoE transport supports TP1 EP32 only")
            from .ppu_mega_sparse import SparseHierarchy

            self.sparse = SparseHierarchy(self, capacity)

    def dispatch(self, x, weights, indices, active_token_mask):
        count = x.shape[0]
        if count > self.capacity or indices.shape != (count, 8) or weights.shape != indices.shape:
            raise ValueError("MegaMoE hierarchical input exceeds its fixed dispatch layout")
        if self.sparse is not None:
            self.sparse.start(x, weights, indices, active_token_mask)
            self.sparse.dispatch_exchange.finish()
            received, ids, scores = self.sparse.unpack()
            return received, scores, ids
        packet = self.packet
        pack_tokens[(self.capacity,)](
            x, indices, weights, active_token_mask, packet, count,
            self.dim, self.words, active_token_mask is not None,
            triton.next_power_of_2(self.dim), num_warps=4,
        )
        dist.all_gather_into_tensor(self.packets, packet, group=self.cross_group)
        unpack_tokens[(self.capacity * self.nodes,)](
            self.packets, self.recv_x, self.recv_ids, self.recv_weights,
            self.local_ids, self.local_weights, self.dim, self.words,
            self.node * self.experts, self.experts,
            triton.next_power_of_2(self.dim), num_warps=4,
        )
        return self.recv_x, self.local_weights, self.local_ids

    def combine(self, y, count):
        if self.sparse is not None:
            return self.sparse.finish(y, count)
        # Transfer existing BF16 node outputs; sum in FP32 at the destination.
        dist.all_to_all_single(self.parts, y, group=self.cross_group)
        size = self.capacity * self.dim
        sum_contributions[(triton.cdiv(size, 256),)](
            self.parts, self.reduced, size, self.nodes, 256, num_warps=4,
        )
        return self.reduced[:count]
