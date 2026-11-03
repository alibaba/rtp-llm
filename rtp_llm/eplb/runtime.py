"""DeepInfer EPLB at the target verification boundary, over full WORLD experts.

Adapted from DeepInfer 9bd9cf243fc53ef9afbf8f04d391d989117d6ab9
(Apache-2.0), deepinfer/model_executor/eplb/__init__.py.
"""

import logging
import time
from collections.abc import Sequence

import torch
import torch.distributed as dist

from rtp_llm.eplb.policy import rebalance_experts_global
from rtp_llm.eplb.shuffle import (
    expert_model_parallel_shuffle_inplace,
    expert_model_parallel_shuffle_inplace_intranode,
)


class EplbLayer:
    """Borrow fixed routing tensors and actual executor weights.

    Every weight/scale has local physical experts on dimension zero. Other
    dimensions may be non-contiguous. Neither storage nor tensor addresses are
    replaced after CUDA graph capture. Gate IDs are logical IDs or -1.
    """

    def __init__(
        self,
        layer_id: int,
        logical_to_physical: torch.Tensor,
        logical_count: torch.Tensor,
        weights: Sequence[torch.Tensor],
        ep_rank: int = 0,
        replica_offset: int = 0,
    ) -> None:
        self.layer_id = layer_id
        self.logical_to_physical = logical_to_physical
        self.logical_count = logical_count
        self.weights = tuple(weights)
        self.ep_rank = ep_rank
        # PureTP peers reduce partial outputs and must select the same copy.
        # Independent dispatch streams may rotate by their own rank.
        self.replica_offset = replica_offset
        if (
            layer_id < 0
            or logical_count.ndim != 1
            or logical_count.numel() == 0
            or logical_to_physical.ndim != 2
            or logical_to_physical.size(0) != logical_count.numel()
            or logical_to_physical.dtype not in (torch.int32, torch.int64)
            or logical_count.dtype not in (torch.int32, torch.int64)
            or logical_count.device != logical_to_physical.device
        ):
            raise ValueError(
                "EPLB requires a nonnegative layer ID and integer routing maps"
            )
        counts = logical_count.cpu().long()
        mapping = logical_to_physical.cpu().long()
        if bool(((counts <= 0) | (counts > mapping.size(1))).any()):
            raise ValueError("Every logical expert must have valid physical replicas")
        valid = torch.arange(mapping.size(1), device="cpu").unsqueeze(
            0
        ) < counts.unsqueeze(1)
        physical_slots = mapping[valid]
        self.num_logical_experts = counts.numel()
        self.num_physical_experts = physical_slots.numel()
        if (
            mapping.size(1) != self.num_physical_experts - self.num_logical_experts + 1
            or not torch.equal(
                physical_slots.sort().values,
                torch.arange(self.num_physical_experts, device="cpu"),
            )
            or bool((mapping[~valid] != -1).any())
        ):
            raise ValueError("EPLB maps must cover every physical slot exactly once")
        if not self.weights or self.weights[0].ndim == 0:
            raise ValueError(
                "EPLB requires actual executor weights with an expert dimension"
            )
        self.num_local_experts = self.weights[0].size(0)
        if (
            self.num_local_experts <= 0
            or self.num_physical_experts % self.num_local_experts != 0
            or not 0 <= ep_rank < self.num_physical_experts // self.num_local_experts
            or any(
                w.ndim == 0
                or w.size(0) != self.num_local_experts
                or w.device != logical_to_physical.device
                for w in self.weights
            )
        ):
            raise ValueError(
                "EPLB weights must have matching local experts and routing device"
            )
        self.physical_to_logical = torch.empty(
            self.num_physical_experts, dtype=torch.int64, device="cpu"
        )
        logical_ids = (
            torch.arange(self.num_logical_experts, device="cpu")
            .unsqueeze(1)
            .expand_as(mapping)
        )
        self.physical_to_logical[physical_slots] = logical_ids[valid]
        self.physical_to_logical = self.physical_to_logical.to(
            device=logical_count.device, dtype=torch.int32
        )

    def map_ids(self, ids: torch.Tensor) -> torch.Tensor:
        """Map [..., top_k] IDs without host synchronization; preserve -1."""
        logical = ids.clamp_min(0).long()
        position = torch.arange(ids.numel(), device=ids.device).reshape(ids.shape)
        if ids.ndim > 1:
            position = position // ids.size(-1)
        # DeepGEMM's gate uses uint32(input_rank + token_index * 23333).
        # The top-k column does not participate in replica selection.
        rotation = (self.replica_offset + position * 23333) & 0xFFFFFFFF
        replica = rotation % self.logical_count[logical]
        physical = self.logical_to_physical[logical, replica]
        return torch.where(ids >= 0, physical, -1).to(ids.dtype)


class _ExpertLayoutGroup:
    """One homogeneous DeepInfer layout; preserve executor-owned map storage.

    The physical map and window are owned here. Layer physical maps are views;
    logical maps/counts remain the executor's graph-stable buffers. C++ owns and
    clears step counts after both EPLB and the independent monitor consume them.
    """

    def __init__(self, layers, config, ep_group, model_id, is_normal_engine):
        self.layers = tuple(layers)
        self.ep_group = ep_group
        self.model_id = model_id
        first = layers[0]
        self.device = first.logical_count.device
        self.num_logical = first.num_logical_experts
        self.num_physical = first.num_physical_experts
        self.num_local = first.num_local_experts
        self.layer_ids = torch.tensor(
            [layer.layer_id for layer in layers], dtype=torch.int64, device=self.device
        )
        self.physical_to_logical = torch.stack(
            [layer.physical_to_logical for layer in layers]
        )
        for index, layer in enumerate(layers):
            layer.physical_to_logical = self.physical_to_logical[index]
        # Communication views share the real executor storage. DeepInfer uses
        # the same transpose for column-major scale tensors.
        self.weights = [
            [w.mT if w.ndim >= 3 and w.stride(-1) != 1 else w for w in layer.weights]
            for layer in layers
        ]
        self.window = torch.zeros(
            config.eplb_stats_window_size,
            len(layers),
            self.num_local,
            dtype=torch.int32,
            device=self.device,
        )
        self.after_prefill = is_normal_engine and config.eplb_rearrange_after_prefill
        self.skip_prefill = is_normal_engine and not self.after_prefill
        self.p2p_groups = config.eplb_rearrange_p2p_groups
        self.updates = 0
        self.window_index = 0
        interval = config.eplb_update_time
        self.steps = max(0, interval - max(1, interval // 4))

    def initialize_shuffle(self):
        # Match DeepInfer's physical-domain test, not hostname or P2P access.
        from rtp_llm.utils.deepep_utils import configure_deepep_nccl_root

        configure_deepep_nccl_root()
        import deep_ep

        torch.cuda.empty_cache()
        get_physical_domain_size = getattr(deep_ep, "get_physical_domain_size", None)
        self.intranode_shuffle = False
        if get_physical_domain_size is None:
            # CUDA 12 wheels predate the topology API. Keep the same P2P
            # shuffle used outside NVLink domains instead of guessing topology.
            logging.warning(
                "[EPLB] DeepEP has no physical-domain API; using P2P shuffle"
            )
        else:
            self.intranode_shuffle = (
                get_physical_domain_size(self.ep_group)[1] == self.ep_group.size()
            )
        if self.intranode_shuffle:
            import torch.distributed._symmetric_memory as symm_mem

            self.symmetric_weight_buffers = [
                symm_mem.empty(w.shape, dtype=w.dtype, device=self.device)
                for w in self.weights[0]
            ]
            self.symmetric_memory_handles = [
                symm_mem.rendezvous(w, self.ep_group)
                for w in self.symmetric_weight_buffers
            ]
            self.ipc_weight_buffers = [
                [
                    handle.get_buffer(rank, weight.shape, weight.dtype)
                    for weight, handle in zip(
                        self.symmetric_weight_buffers, self.symmetric_memory_handles
                    )
                ]
                for rank in range(self.ep_group.size())
            ]

    def step(self, counts, is_prompt, mode, interval):
        if interval == 0 or (is_prompt and self.skip_prefill):
            return False
        self.window[self.window_index].copy_(counts.index_select(0, self.layer_ids))
        self.window_index = (self.window_index + 1) % self.window.size(0)
        if is_prompt and self.after_prefill:
            self.steps = interval - 1
            return False
        self.steps += 1
        if self.steps < interval:
            return False
        self.steps = 0
        # STATS collects without changing weights.
        if mode == 1:
            return False
        return self.rearrange()

    def _plan(self):
        """Plan from the physical count window without changing runtime state."""
        rank = self.ep_group.rank()
        local_map = self.physical_to_logical[
            :, rank * self.num_local : (rank + 1) * self.num_local
        ]
        logical_counts = torch.zeros(
            self.window.size(0),
            len(self.layers),
            self.num_logical,
            dtype=torch.int32,
            device=self.device,
        )
        logical_counts.scatter_add_(
            -1, local_map.long().expand_as(self.window), self.window
        )
        dist.all_reduce(logical_counts, group=self.ep_group)
        return rebalance_experts_global(
            logical_counts, self.num_physical, self.num_local
        )

    def rearrange(self):
        rank = self.ep_group.rank()
        should_log = rank == 0
        if should_log:
            torch.cuda.synchronize(self.device)
            started = time.perf_counter()
        physical, inverse, replicas = self._plan()
        # The window remains a physical-slot ring, exactly as in DeepInfer.
        # Do not replace it with a logical history or clear it after a shuffle.
        changed_slots = (physical != self.physical_to_logical).sum(-1).tolist()
        if self.intranode_shuffle:
            expert_model_parallel_shuffle_inplace_intranode(
                self.ipc_weight_buffers,
                self.physical_to_logical,
                self.weights,
                physical,
                self.ep_group,
            )
        else:
            expert_model_parallel_shuffle_inplace(
                self.physical_to_logical,
                self.weights,
                physical,
                self.ep_group,
                p2p_groups=self.p2p_groups,
            )
        self.physical_to_logical.copy_(physical)
        for index, layer in enumerate(self.layers):
            layer.logical_to_physical.copy_(inverse[index])
            layer.logical_count.copy_(replicas[index])
        self.updates += 1
        # RTP's completion records cover every rank. Wait for the actual
        # weights and graph-stable maps, once per group rather than per layer.
        torch.cuda.synchronize(self.device)
        logging.info(
            "[EPLB] applied model_id=%s layers=%s update=%d "
            "changed_slots=%s ep_rank=%d",
            self.model_id,
            [layer.layer_id for layer in self.layers],
            self.updates,
            changed_slots,
            rank,
        )
        if should_log:
            logging.info(
                "[EPLB] rearranged model_id=%s layers=%s update=%d "
                "changed_slots=%s transport=%s time=%.3fs",
                self.model_id,
                [layer.layer_id for layer in self.layers],
                self.updates,
                changed_slots,
                "symm_mem" if self.intranode_shuffle else "p2p",
                time.perf_counter() - started,
            )
        return any(changed_slots)


class RuntimeExpertBalancer:
    """Adapt DeepInfer's layer groups to RTP's target-only counters and control.

    Shapes and placement are fixed before graph capture. Call step only after
    target verification, never during warmup, proposal or speculative commit.
    Step counts are borrowed and must be cleared by their C++ owner afterward.
    """

    def __init__(
        self,
        layers: Sequence[EplbLayer],
        config,
        ep_rank,
        ep_size,
        model_id,
        is_normal_engine=False,
        sync_prompt_phase=False,
    ):
        self.layers = tuple(sorted(layers, key=lambda layer: layer.layer_id))
        if not self.layers:
            raise ValueError("EPLB requires at least one routed expert layer")
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.model_id = model_id
        self.device = self.layers[0].logical_count.device
        if (
            ep_size <= 0
            or not 0 <= ep_rank < ep_size
            or len({layer.layer_id for layer in self.layers}) != len(self.layers)
            or any(
                layer.ep_rank != ep_rank
                or layer.num_physical_experts != layer.num_local_experts * ep_size
                or layer.logical_count.device != self.device
                for layer in self.layers
            )
        ):
            raise ValueError(
                "EPLB layers must share a device and full EP=WORLD experts"
            )
        if ep_size > 1 and (
            self.device.type != "cuda"
            or not dist.is_initialized()
            or dist.get_world_size() != ep_size
            or dist.get_rank() != ep_rank
            or dist.get_backend() != "nccl"
        ):
            raise ValueError("EPLB requires EP=WORLD using CUDA tensors and NCCL")
        self.window_size = config.eplb_stats_window_size
        if self.window_size < 0 or config.eplb_rearrange_p2p_groups <= 0:
            raise ValueError(
                "EPLB requires a nonnegative window and positive P2P groups"
            )
        self.mode = int(config.eplb_mode)
        self.interval = config.eplb_update_time
        if self.mode not in (0, 1, 2, 3) or self.interval < 0:
            raise ValueError("EPLB mode must be 0..3 and update interval nonnegative")
        if self.interval > 0 and not 0 < self.window_size <= self.interval:
            raise ValueError("EPLB enabled interval requires 0 < window <= interval")
        grouped = {}
        for layer in self.layers:
            layout = (
                layer.num_logical_experts,
                layer.num_physical_experts,
                tuple(
                    (
                        tuple((w.mT if w.ndim >= 3 and w.stride(-1) != 1 else w).shape),
                        w.dtype,
                    )
                    for w in layer.weights
                ),
            )
            grouped.setdefault(layout, []).append(layer)
        self.groups = tuple(
            _ExpertLayoutGroup(
                group,
                config,
                dist.group.WORLD if ep_size > 1 else None,
                model_id,
                is_normal_engine,
            )
            for group in grouped.values()
        )
        # RTP may combine prefill on one DP rank with decode on another.
        # DeepInfer's NORMAL engine supplies one global phase. The caller
        # enables coordination only where its scheduler does not guarantee it.
        self.prompt_flag = (
            torch.empty(1, dtype=torch.int32, device=self.device)
            if sync_prompt_phase
            else None
        )
        if ep_size > 1:
            # Prepare communication and planning kernels before serving requests.
            # The initial zero window does not publish maps or advance steps.
            started = time.perf_counter()
            for group in self.groups:
                group.initialize_shuffle()
                if self.window_size > 0:
                    group._plan()
            torch.cuda.synchronize(self.device)
            logging.info(
                "[EPLB] warmed up model_id=%s ep_rank=%d time=%.3fs",
                model_id,
                ep_rank,
                time.perf_counter() - started,
            )

    @torch.inference_mode()
    def step(self, counts: torch.Tensor, is_prompt=False):
        if self.mode == 0 or self.ep_size <= 1 or self.interval == 0:
            return False
        if self.prompt_flag is not None:
            self.prompt_flag.fill_(is_prompt)
            dist.all_reduce(self.prompt_flag, op=dist.ReduceOp.MAX)
            is_prompt = bool(self.prompt_flag.item())
        changed = False
        for group in self.groups:
            # Do not short-circuit: all ranks must visit every layout group.
            changed = group.step(counts, is_prompt, self.mode, self.interval) or changed
        return changed
