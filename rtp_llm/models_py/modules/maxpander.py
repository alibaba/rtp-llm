"""Static physical expert routing after packing, before MegaMoE dispatch.

Only startup reads tensors on the host. Zero redundancy uses a direct lookup;
redundant placements recompute local token demand and WF quotas on GPU, including
graph replay. There is no CPU fallback.
Placement is immutable for the lifetime of this module. Counts are rank-local,
not a claim of globally optimal balancing across CP/DP ranks.
"""

import inspect
import logging
from functools import lru_cache, partial

import torch
import torch.nn as nn


class PackedMoeExecutorMixin:
    def setup_mega_moe_tile_policy(self, deep_gemm):
        self._maxpander_tile_policy = None
        config = self.cfg.maxpander_config
        if (
            not config.enabled
            or self.cfg.physical_expert_num == self.cfg.n_routed_experts
        ):
            return
        if config.role != "PREFILL":
            raise ValueError("MaxPander tile selection requires a PREFILL worker")
        kernel = deep_gemm.fp8_fp4_mega_moe
        resolver = getattr(deep_gemm, "get_block_m_for_mega_moe", None)
        if _supports_mega_moe_tile_override(kernel, resolver):
            self._maxpander_tile_policy = MaxPanderMegaMoeTilePolicy(
                self.cfg, kernel, resolver
            )

    def _select_mega_moe_kernel(self, deep_gemm, tokens):
        policy = getattr(self, "_maxpander_tile_policy", None)
        if policy is None:
            return deep_gemm.fp8_fp4_mega_moe
        return policy.kernel_for(tokens)

    def setup_routing(self, layer_weights):
        config = self.cfg.maxpander_config
        self.physical_router = None
        if config.enabled:
            from rtp_llm.utils.model_weight import W

            config.validate_placement(
                expert_num=self.cfg.n_routed_experts,
                physical_expert_num=self.cfg.physical_expert_num,
                ep_size=self.cfg.ep_size,
            )
            self.physical_router = MaxPanderRouter(
                layer_weights.get(W.log2phy),
                layer_weights.get(W.logic_expert_cnt),
                num_logical=self.cfg.n_routed_experts,
                num_physical=self.cfg.physical_expert_num,
                ep_size=self.cfg.ep_size,
                wf_passes=config.wf_passes,
            )

    def _execute_packed_inputs(self, tokens, device):
        router = getattr(self, "physical_router", None)
        if router is not None:
            router.remap_inplace(self._mega_buf.topk_idx[:tokens])
        output = self._mega_y[:tokens]
        self._launch(output, tokens, device)
        return output


@lru_cache(maxsize=16)
def _supports_mega_moe_tile_override(kernel, resolver):
    try:
        supported = (
            callable(resolver) and "block_m" in inspect.signature(kernel).parameters
        )
    except (TypeError, ValueError):
        supported = False
    if not supported:
        logging.warning(
            "MaxPander: DeepGEMM lacks the per-call block_m API; "
            "retaining native tile selection without the tile optimization"
        )
    return supported


class MaxPanderMegaMoeTilePolicy:
    """Match the OFF block_m without changing the physical expert layout."""

    def __init__(self, config, kernel, resolver):
        self._kernel = kernel
        self._kernels = {}
        self._resolver = resolver
        self._ranks = int(config.ep_size)
        self._logical = int(config.n_routed_experts)
        self._capacity = int(config.max_tokens_per_rank)
        self._topk = int(config.n_activated_experts)
        self._resolve = lru_cache(maxsize=128)(self._resolve_uncached)

    def _resolve_uncached(self, tokens):
        block_m = int(
            self._resolver(
                self._ranks,
                self._logical,
                self._capacity,
                tokens,
                self._topk,
                "fp8xfp4",
            )
        )
        if block_m not in self._kernels:
            self._kernels[block_m] = partial(self._kernel, block_m=block_m)
        return block_m, self._kernels[block_m]

    def block_m(self, tokens):
        return self._resolve(tokens)[0]

    def kernel_for(self, tokens):
        return self._resolve(tokens)[1]


class MaxPanderRouter(nn.Module):
    def __init__(
        self, log2phy, counts, *, num_logical, num_physical, ep_size, wf_passes=2
    ):
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
        self._direct_mapping = num_physical == num_logical
        self.ep_size = ep_size
        self.passes = int(wf_passes)
        if not 1 <= self.passes <= 64:
            raise ValueError("MAXPANDER_WF_PASSES must be in [1,64]")
        device = log2phy.device

        def buffer(name, value):
            self.register_buffer(name, value, persistent=False)

        candidate_tensor = torch.tensor(candidates, dtype=torch.int32, device=device)
        buffer("cand_phys", candidate_tensor)
        if self._direct_mapping:
            for name in (
                "cand_gpu",
                "rep_count",
                "counts",
                "partial_counts",
                "hot_order",
                "quota",
                "alloc",
                "rank_counter",
                "refine_order",
                "work_counts",
            ):
                buffer(name, None)
            return
        from rtp_llm.models_py.modules.maxpander_kernels import MAX_PARTITIONS

        buffer(
            "cand_gpu", torch.where(candidate_tensor >= 0, candidate_tensor // cap, -1)
        )
        buffer("rep_count", counts.to(dtype=torch.int32).clone())
        buffer("counts", torch.zeros(num_logical, dtype=torch.float32, device=device))
        buffer(
            "partial_counts",
            torch.empty(
                (MAX_PARTITIONS, 1 << (num_logical - 1).bit_length()),
                dtype=torch.int32,
                device=device,
            ),
        )
        buffer("hot_order", torch.empty(num_logical, dtype=torch.int32, device=device))
        buffer("quota", torch.empty((num_logical, 2), dtype=torch.int32, device=device))
        buffer(
            "alloc", torch.empty((num_logical, 2), dtype=torch.float32, device=device)
        )
        buffer(
            "rank_counter", torch.zeros(num_logical, dtype=torch.int32, device=device)
        )
        buffer(
            "refine_order", torch.empty(num_logical, dtype=torch.int32, device=device)
        )
        buffer("work_counts", torch.empty(2, dtype=torch.int32, device=device))
        self._launch_cache = {}

    def _apply(self, fn, recurse=True):
        if not self._direct_mapping:
            self._launch_cache.clear()
        return super()._apply(fn, recurse=recurse)

    def __getstate__(self):
        state = super().__getstate__()
        if not self._direct_mapping:
            state["_launch_cache"] = {}
        return state

    def forward(self, logical_ids):
        return self.remap_inplace(logical_ids.clone())

    def remap_inplace(self, logical_ids):
        import triton

        from rtp_llm.models_py.modules.maxpander_kernels import (
            COUNT_BLOCK_SIZE,
            MAX_PARTITIONS,
            REDUCE_PARTITION_TILE,
            _direct_remap_kernel,
            _wf_count_order_kernel,
            _wf_partial_count_kernel,
            _wf_quota_kernel,
            _wf_reduce_order_kernel,
            _wf_scatter_kernel,
        )

        if logical_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("expert IDs must be int32/int64")
        if logical_ids.device != self.cand_phys.device:
            raise ValueError("expert IDs and placement must share a CUDA device")
        if logical_ids.ndim != 2 or logical_ids.stride(1) != 1:
            raise ValueError(
                "expert IDs must have shape [tokens, topk] with contiguous columns"
            )
        if logical_ids.stride(0) < logical_ids.size(1):
            raise ValueError("expert ID rows must not overlap")
        numel = logical_ids.numel()
        if numel == 0:
            return logical_ids  # Caller still executes the EP collective.
        block = min(1024, triton.next_power_of_2(numel))
        grid = (triton.cdiv(numel, block),)
        if self._direct_mapping:
            _direct_remap_kernel[grid](
                logical_ids,
                self.cand_phys,
                self.num_logical,
                numel,
                logical_ids.stride(0),
                TOPK=logical_ids.size(1),
                BLOCK_SIZE=block,
            )
            return logical_ids
        if numel > 2**24:
            raise ValueError("rank-local WF counts exceed exact float32 integer range")
        if torch.cuda.current_device() != logical_ids.device.index:
            raise ValueError("MaxPander input must match the current CUDA device")
        row_stride = logical_ids.stride(0)
        topk = logical_ids.size(1)
        workspace_signature = tuple(
            (buffer.data_ptr(), buffer.dtype, buffer.device)
            for buffer in self._buffers.values()
            if buffer is not None
        )
        key = (
            logical_ids.data_ptr(),
            logical_ids.dtype,
            logical_ids.device,
            numel,
            row_stride,
            topk,
            self.passes,
            workspace_signature,
        )
        launchers = self._launch_cache.get(key)
        if launchers is not None:
            for launch in launchers:
                launch()
            return logical_ids
        block_experts = triton.next_power_of_2(self.num_logical)
        launch_grid = (grid[0], 1, 1)
        fused_count_order = numel <= 1024 and self.num_logical <= 512
        if fused_count_order:
            preparation_kernels = ((_wf_count_order_kernel, (1, 1, 1), {}),)
            preparation_arguments = (
                (
                    logical_ids,
                    self.counts,
                    self.hot_order,
                    self.rank_counter,
                    self.rep_count,
                    self.refine_order,
                    self.work_counts,
                    self.alloc,
                    self.num_logical,
                    numel,
                    row_stride,
                    topk,
                    block,
                    block_experts,
                ),
            )
        else:
            partitions = min(MAX_PARTITIONS, triton.cdiv(numel, COUNT_BLOCK_SIZE))
            preparation_kernels = (
                (_wf_partial_count_kernel, (partitions, 1, 1), {}),
                (_wf_reduce_order_kernel, (1, 1, 1), {}),
            )
            preparation_arguments = (
                (
                    logical_ids,
                    self.partial_counts,
                    self.num_logical,
                    numel,
                    row_stride,
                    topk,
                    COUNT_BLOCK_SIZE,
                    block_experts,
                ),
                (
                    self.partial_counts,
                    self.counts,
                    self.hot_order,
                    self.rank_counter,
                    self.rep_count,
                    self.refine_order,
                    self.work_counts,
                    self.alloc,
                    self.num_logical,
                    partitions,
                    block_experts,
                    REDUCE_PARTITION_TILE,
                ),
            )
        arguments = preparation_arguments + (
            (
                self.counts,
                self.hot_order,
                self.cand_gpu,
                self.rep_count,
                self.quota,
                self.alloc,
                self.refine_order,
                self.work_counts,
                self.num_logical,
                self.passes,
                triton.next_power_of_2(self.ep_size),
                block_experts,
            ),
            (
                logical_ids,
                self.quota,
                self.cand_phys,
                self.rank_counter,
                logical_ids,
                self.num_logical,
                numel,
                row_stride,
                topk,
                block,
            ),
        )
        kernels = preparation_kernels + (
            (_wf_quota_kernel, (1, 1, 1), {"num_warps": 1}),
            (_wf_scatter_kernel, launch_grid, {}),
        )
        launchers = tuple(
            partial(
                kernel[kernel_grid](*kernel_args, **options)[kernel_grid],
                *(
                    value.data_ptr() if isinstance(value, torch.Tensor) else value
                    for value in kernel_args
                ),
            )
            for (kernel, kernel_grid, options), kernel_args in zip(kernels, arguments)
        )
        if len(self._launch_cache) == 16:
            del self._launch_cache[next(iter(self._launch_cache))]
        self._launch_cache[key] = launchers
        return logical_ids
