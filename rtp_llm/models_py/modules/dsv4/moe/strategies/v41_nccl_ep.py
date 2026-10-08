"""V4.1 recipe on the base NCCL row transport, with lossless FP32 returns."""

import hashlib
import json
import os
from dataclasses import asdict, replace

import torch

from rtp_llm.models_py.modules.dsv4.moe._nccl_ep_combine_triton import fp32_peer_sum
from rtp_llm.models_py.modules.dsv4.v41_dispatch import V41DispatchLayout, pack_dispatch
from rtp_llm.models_py.modules.dsv4.v41_expert import V41_EXPERT_RECIPE

from .base import register_strategy
from .nccl_ep_mxfp8 import NcclEpMxfp8Strategy
from .v41_grouped_fp4 import V41GroupedFP4Strategy


@register_strategy
class V41NcclEpStrategy(NcclEpMxfp8Strategy):
    name = "v41_nccl_ep"
    expert_recipe = V41_EXPERT_RECIPE

    @classmethod
    def can_handle(cls, cfg):
        context = cfg.stage_context
        return (
            cfg.expert_recipe == cls.expert_recipe
            and cfg.ep_size in (2, 4, 8)
            and 0 <= cfg.ep_rank < cfg.ep_size
            and cfg.n_routed_experts == cfg.n_local_experts * cfg.ep_size
            and cfg.local_expert_start == cfg.ep_rank * cfg.n_local_experts
            and cfg.local_expert_end == cfg.local_expert_start + cfg.n_local_experts
            and cfg.n_local_experts > 0
            and 0 < cfg.n_activated_experts <= cfg.n_routed_experts
            and cfg.max_tokens_per_rank > 0
            and cfg.tp_size == 1
            and cfg.dim % 128 == cfg.moe_inter_dim % 128 == 0
            and min(cfg.dim, cfg.moe_inter_dim) >= 128
            and (context is None or context.backend == "fork_nccl_mxfp8")
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability() == (12, 0)
        )

    def __init__(self, cfg):
        if not self.can_handle(cfg):
            raise ValueError("V4.1 NCCL requires a compatible SM120 expert shard")
        super().__init__(cfg)
        self.layout = V41DispatchLayout(
            cfg.dim, cfg.n_activated_experts, cfg.n_routed_experts
        )
        self._admitted_identity = None
        self._failed = False

    def _make_local_strategy(self, local_cfg):
        # A receiver computes all peers' rows and keeps global expert IDs.
        return V41GroupedFP4Strategy(
            replace(
                local_cfg,
                local_expert_start=self.cfg.local_expert_start,
                local_expert_end=self.cfg.local_expert_end,
                max_tokens_per_rank=self.cfg.max_tokens_per_rank * self.cfg.ep_size,
            )
        )

    def _admit(self, device):
        if self._failed:
            raise RuntimeError("V4.1 EP failed; replace the worker/group before retry")
        if torch.cuda.is_current_stream_capturing():
            raise NotImplementedError(
                "V4.1 NCCL strategy currently requires eager execution"
            )
        if self._local.local is None or device != self._local.local.experts.device:
            raise RuntimeError(
                "V4.1 EP requires initialized local weights on the input device"
            )
        group, world, rank = self._stage()
        dist = torch.distributed
        if dist.get_world_size(group) != world or dist.get_rank(group) != rank:
            raise RuntimeError("V4.1 EP process group disagrees with its stage roster")
        ranks = tuple(dist.get_process_group_ranks(group))
        stream = int(torch.cuda.current_stream(device).cuda_stream)
        cfg, context = self.cfg, self.cfg.stage_context
        stage = None if context is None else (context.pp_rank, context.generation)
        options = tuple(
            os.environ.get(name, "0") == "1"
            for name in (
                "DSV4_MOE_PREQUANT_INPUT",
                "DSV4_MOE_PREQUANT_INPUT_REQUIRED",
                "DSV4_MOE_LOCAL_REPLAY",
                "DSV4_MOE_LOCAL_REPLAY_REQUIRED",
                "DSV4_NCCL_EP_MXFP8_DISPATCH_PACK",
                "DSV4_MOE_EXTENT_COUNT_FUSE",
            )
        )
        identity = (
            id(group),
            ranks,
            device,
            stream,
            id(self._local.local),
            cfg,
            self.layout,
            stage,
            options,
        )
        if self._admitted_identity is not None:
            if self._admitted_identity != identity:
                raise RuntimeError(
                    "V4.1 EP configuration, group, weights, device or stream changed after admission"
                )
            return group, world, rank
        metadata = {
            "protocol": "v41-nccl-a2a-fp32-v1",
            "dispatch": asdict(self.layout),
            "layer": cfg.layer_id,
            "intermediate": cfg.moe_inter_dim,
            "swiglu_limit": cfg.swiglu_limit,
            "rows_per_rank": cfg.max_tokens_per_rank,
            "ranks": ranks,
            "stage": stage,
            "count_fuse": options[-1],
        }
        digest = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode("ascii")
        ).digest()
        # These switches select V4 numerical/capture routes that this strategy
        # does not implement. Check collectively before sending any payload.
        compatible = (
            dist.get_backend(group) == "nccl"
            and (context is None or ranks == context.world_ranks)
            and not any(options[:-1])
        )
        signature = torch.tensor(
            [*digest, int(compatible)], dtype=torch.uint8, device=device
        ).view(1, 33)
        gathered = torch.empty((world, 33), dtype=torch.uint8, device=device)
        dist.all_gather_into_tensor(gathered, signature, group=group)
        signatures = gathered.cpu()
        if not bool((signatures[:, :32] == signatures[0, :32]).all()) or not bool(
            (signatures[:, 32] == 1).all()
        ):
            self._failed = True
            raise RuntimeError(
                "V4.1 EP peer recipe, layout, bounds or transport option mismatch"
            )
        self._admitted_identity = identity
        return group, world, rank

    def synchronized_chunk_extent(self, local_tokens, device):
        try:
            self._admit(device)
            return super().synchronized_chunk_extent(local_tokens, device)
        except BaseException:
            self._failed = True
            raise

    def forward(self, x, weights, indices):
        pending = self._take_pending_counts()
        try:
            group, world, rank = self._admit(x.device)
            rows = int(x.shape[0])
            counts = self._counts_for_forward(rows, group, world, x.device, pending)
            # Every rank sees the same vector, so bound failures precede payload
            # collectives on all peers, including empty ranks.
            if (
                len(counts) != world
                or counts[rank] != rows
                or any(n < 0 or n > self.cfg.max_tokens_per_rank for n in counts)
            ):
                raise ValueError(
                    "V4.1 EP received rows exceed the admitted per-rank bound"
                )
            valid = torch.ones(rows, dtype=torch.bool, device=x.device)
            payload = pack_dispatch(x, weights, indices, valid, self.layout)
            received = self._all_to_all(
                payload.repeat(world, 1),
                [rows] * world,
                counts,
                self.layout.payload_bytes,
                group,
            )
            local = self._local.local
            layout = (
                "masked" if sum(counts) <= local.masked_token_limit else "contiguous"
            )
            partial = local.forward_received(received, self.layout, layout=layout)
            returned = self._all_to_all(
                partial.contiguous().view(torch.uint8),
                counts,
                [rows] * world,
                self.cfg.dim * 4,
                group,
            ).view(torch.float32)
            output = torch.empty(
                (rows, self.cfg.dim), dtype=torch.float32, device=x.device
            )
            return fp32_peer_sum(returned, world, output)
        except BaseException:
            self._failed = True
            raise
