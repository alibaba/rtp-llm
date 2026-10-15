"""Static physical expert routing, between logical Top-K and MegaMoE packing.

Only startup reads tensors on the host. Each invocation recomputes local token
demand and WF quotas on GPU, including graph replay; there is no CPU fallback.
Placement is immutable for the lifetime of this module. Counts are rank-local,
not a claim of globally optimal balancing across CP/DP ranks.
"""

import os

import torch
import torch.nn as nn


def maxpander_enabled() -> bool:
    return os.environ.get("MAXPANDER_ENABLE", "0") == "1"


class MaxPanderRouter(nn.Module):
    def __init__(self, log2phy, counts, *, num_logical, num_physical, ep_size):
        super().__init__()
        if not (1 < ep_size and 0 < num_logical <= num_physical <= 2048):
            raise ValueError("MaxPander requires EP>1 and 0<E<=P<=2048")
        if num_physical % ep_size or num_physical // ep_size > 512:
            raise ValueError("MegaMoE requires P divisible by EP and P/EP<=512")
        if log2phy is None or counts is None:
            raise ValueError("MaxPander requires loader-created log2phy and counts")
        if log2phy.ndim != 2 or log2phy.shape[0] != num_logical:
            raise ValueError("log2phy must have shape [logical experts, replicas]")
        if counts.shape != (num_logical,):
            raise ValueError("invalid replica count shape")
        # Synchronous checks are deliberately constructor-only, before capture.
        table = log2phy.detach().cpu().tolist()
        replica_counts = counts.detach().cpu().tolist()
        slots = []
        cap = num_physical // ep_size
        candidates = []
        for row, count in zip(table, replica_counts):
            if count not in (1, 2) or count > len(row):
                raise ValueError("MaxPander WF supports one or two copies per expert")
            physical = row[:count]
            if any(p < 0 or p >= num_physical for p in physical):
                raise ValueError("physical expert outside [0,P)")
            if len({p // cap for p in physical}) != count:
                raise ValueError("replicas must reside on distinct EP ranks")
            if any(p != -1 for p in row[count:]):
                raise ValueError("unused log2phy entries must be -1")
            slots.extend(physical)
            candidates.append(physical + [-1] * (2 - count))
        if sorted(slots) != list(range(num_physical)):
            raise ValueError("placement must cover every physical slot exactly once")
        if not log2phy.is_cuda or counts.device != log2phy.device:
            raise ValueError("MaxPander routing tables must be on the same CUDA device")
        self.num_logical = num_logical
        self.ep_size = ep_size
        self.passes = int(os.environ.get("MAXPANDER_WF_PASSES", "8"))
        if not 1 <= self.passes <= 64:
            raise ValueError("MAXPANDER_WF_PASSES must be in [1,64]")
        device = log2phy.device

        def buffer(name, value):
            self.register_buffer(name, value, persistent=False)

        candidate_tensor = torch.tensor(candidates, dtype=torch.int32, device=device)
        buffer("cand_phys", candidate_tensor)
        buffer(
            "cand_gpu", torch.where(candidate_tensor >= 0, candidate_tensor // cap, -1)
        )
        buffer("rep_count", counts.to(dtype=torch.int32).clone())
        buffer("counts", torch.zeros(num_logical, dtype=torch.float32, device=device))
        buffer("quota", torch.empty((num_logical, 2), dtype=torch.int32, device=device))
        buffer(
            "alloc", torch.empty((num_logical, 2), dtype=torch.float32, device=device)
        )
        buffer(
            "rank_counter", torch.zeros(num_logical, dtype=torch.int32, device=device)
        )

    def forward(self, logical_ids):
        import triton

        from rtp_llm.models_py.modules.maxpander_kernels import (
            _wf_count_kernel,
            _wf_quota_kernel,
            _wf_scatter_kernel,
        )

        if logical_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("expert IDs must be int32/int64")
        if logical_ids.device != self.counts.device:
            raise ValueError("expert IDs and placement must share a CUDA device")
        flat = logical_ids.reshape(-1).contiguous()
        n = flat.numel()
        if n == 0:
            return logical_ids  # Caller still executes the EP collective.
        if n > 2**24:
            raise ValueError("rank-local WF counts exceed exact float32 integer range")
        self.counts.zero_()
        self.rank_counter.zero_()
        block = min(1024, triton.next_power_of_2(n))
        grid = (triton.cdiv(n, block),)
        _wf_count_kernel[grid](flat, self.counts, self.num_logical, n, BLOCK_SIZE=block)
        hot_order = torch.argsort(self.counts, descending=True, stable=True)
        _wf_quota_kernel[(1,)](
            self.counts,
            hot_order,
            self.cand_gpu,
            self.rep_count,
            self.quota,
            self.alloc,
            self.num_logical,
            MAX_PASSES=self.passes,
            BLOCK_G=triton.next_power_of_2(self.ep_size),
            TILE_M=0,
        )
        output = torch.empty_like(flat)
        _wf_scatter_kernel[grid](
            flat,
            self.quota,
            self.cand_phys,
            self.rank_counter,
            output,
            self.num_logical,
            n,
            BLOCK_SIZE=block,
        )
        return output.view_as(logical_ids)
