"""V4.1 SM120 group32 expert compute with PP1 WORLD collectives.

This eager strategy preserves the V4.1 gate-first expert recipe. Each rank
computes its resident expert shard; FP32 all-reduce combines contributions.
CUDA graphs require a separate fixed-capacity routing implementation.
"""

import logging

import torch
from rtp_llm.models_py.modules.dsv41.v41_moe import V41LocalMoE
from rtp_llm.models_py.modules.dsv41.v41_routed import V41RoutedExperts
from rtp_llm.utils.model_weight import W

from .base import RoutedExpertsStrategy, register_strategy


@register_strategy
class Sm120V41Strategy(RoutedExpertsStrategy):
    name = "sm120_v41"
    requires_synchronized_chunk_schedule = True

    @classmethod
    def can_handle(cls, cfg):
        return (
            cfg.ep_size > 1
            and cfg.n_local_experts > 0
            and cfg.physical_expert_num == cfg.ep_size * cfg.n_local_experts
            and cfg.local_expert_start == cfg.ep_rank * cfg.n_local_experts
            and cfg.local_expert_end == cfg.local_expert_start + cfg.n_local_experts
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability() == (12, 0)
        )

    def __init__(self, cfg):
        super().__init__(cfg)
        self.local = None

    def _world(self):
        from rtp_llm.models_py.distributed import collective_torch

        parallel = collective_torch._parallelism_config
        dist = torch.distributed
        if parallel is None or parallel.pp_size != 1:
            raise RuntimeError("V4.1 SM120 WORLD MoE requires initialized PP1")
        if not dist.is_initialized():
            raise RuntimeError("V4.1 SM120 MoE requires torch.distributed")
        group = dist.group.WORLD
        world, rank = dist.get_world_size(group), dist.get_rank(group)
        if world != self.cfg.ep_size or rank != self.cfg.ep_rank:
            raise RuntimeError(
                "V4.1 SM120 WORLD must match EP exactly: "
                f"world/rank={world}/{rank}, "
                f"ep_size/ep_rank={self.cfg.ep_size}/{self.cfg.ep_rank}"
            )
        return dist, group, world, rank

    def setup_runtime(self):
        _, _, world, rank = self._world()
        logging.info(
            "[DSV41 MoE] SM120 group32 expert path: layer=%d world=%d rank=%d "
            "local_experts=%d chunk_tokens=%d CUDA_graph=False",
            self.cfg.layer_id,
            world,
            rank,
            self.cfg.n_local_experts,
            self.cfg.max_tokens_per_rank,
        )

    def setup_weights(self, layer_weights):
        cfg = self.cfg
        names = (
            W.v4_routed_w1_w,
            W.v4_routed_w1_s,
            W.v4_routed_w3_w,
            W.v4_routed_w3_s,
            W.v4_routed_w2_w,
            W.v4_routed_w2_s,
        )
        w1, s1, w3, s3, w2, s2 = (layer_weights[key] for key in names)
        expected = (
            (cfg.n_local_experts, cfg.moe_inter_dim, cfg.dim // 2),
            (cfg.n_local_experts, cfg.moe_inter_dim, cfg.dim // 32),
            (cfg.n_local_experts, cfg.moe_inter_dim, cfg.dim // 2),
            (cfg.n_local_experts, cfg.moe_inter_dim, cfg.dim // 32),
            (cfg.n_local_experts, cfg.dim, cfg.moe_inter_dim // 2),
            (cfg.n_local_experts, cfg.dim, cfg.moe_inter_dim // 32),
        )
        if any(
            tuple(layer_weights[key].shape) != shape
            for key, shape in zip(names, expected)
        ):
            raise ValueError("V4.1 SM120 expert weights disagree with the EP shard")
        experts = V41RoutedExperts(
            {
                "w13_w": torch.cat((w1, w3), dim=1),
                "w13_s": torch.cat((s1, s3), dim=1),
                "w2_w": w2,
                "w2_s": s2,
            },
            swiglu_limit=cfg.swiglu_limit,
        )
        # Every EP rank receives the other ranks' bounded input chunks.
        max_tokens = cfg.ep_size * cfg.max_tokens_per_rank
        self.local = V41LocalMoE(
            experts,
            expert_start_id=cfg.local_expert_start,
            max_tokens=max_tokens,
            masked_token_limit=min(128, max_tokens),
            max_topk=cfg.n_activated_experts,
        )
        for key in names:
            layer_weights.pop(key)

    def get_eplb_weights(self):
        experts = self.local.experts
        # Both scale views are resident: migration must update raw checkpoint
        # bytes and the provider's packed scales together along expert axis 0.
        return (
            experts.w13,
            experts.s13,
            experts.w2,
            experts.s2,
            experts._packed_s13,
            experts._packed_s2,
        )

    def synchronized_chunk_extent(self, local_tokens, device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("V4.1 SM120 MoE currently requires CUDA graphs disabled")
        dist, group, _, _ = self._world()
        extent = torch.tensor([local_tokens], dtype=torch.int64, device=device)
        dist.all_reduce(extent, op=dist.ReduceOp.MAX, group=group)
        return int(extent.item())

    def forward(self, x, weights, indices, expert_load=None, eplb_layer=None):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("V4.1 SM120 MoE currently requires CUDA graphs disabled")
        dist, group, world, rank = self._world()
        if x.shape[0] > self.cfg.max_tokens_per_rank:
            raise ValueError(
                "V4.1 SM120 MoE input exceeds its synchronized chunk bound"
            )
        self.finalize_routing(indices, weights, eplb_layer)
        count = torch.tensor([x.shape[0]], dtype=torch.int64, device=x.device)
        all_counts = [torch.empty_like(count) for _ in range(world)]
        dist.all_gather(all_counts, count, group=group)
        counts = [int(value.item()) for value in all_counts]
        width = max(max(counts), 1)

        def gather(tensor, fill):
            padded = torch.full(
                (width, *tensor.shape[1:]),
                fill,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            padded[: tensor.shape[0]].copy_(tensor)
            gathered = [torch.empty_like(padded) for _ in range(world)]
            dist.all_gather(gathered, padded, group=group)
            return torch.cat([value[:n] for value, n in zip(gathered, counts)], dim=0)

        all_x = gather(x, 0)
        all_weights = gather(weights, 0)
        all_indices = gather(indices, -1)
        if not sum(counts):
            return x.new_empty((0, x.shape[-1]), dtype=torch.float32)
        partial = self.local(all_x, all_weights, all_indices, layout="contiguous")
        dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=group)
        begin = sum(counts[:rank])
        return partial[begin : begin + counts[rank]].contiguous()
