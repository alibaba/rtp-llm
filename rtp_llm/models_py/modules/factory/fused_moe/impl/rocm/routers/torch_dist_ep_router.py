"""Torch distributed-based EP router.

Uses torch.distributed.all_to_all for expert parallel
dispatch and combine, without depending on DeepEP.

Suitable for single-node EP (e.g. 4tp4ep, 8tp8ep) where NCCL/RCCL
all_to_all provides sufficient performance.

Protocol:
  Dispatch:
    1. Flatten [tokens, topk] → [flat_total] pairs of (hidden, global_expert_id, weight)
    2. Classify each pair by destination rank (rank = global_expert_id // experts_per_rank)
    3. Build per-dst-rank send buffers, pad to max across all ranks
    4. all_to_all hidden/weights/ids/orig_idx → each rank gets its expert's tokens
       along with the original flattened index for combine
    5. Remap global expert IDs to local IDs for executor

  Combine:
    1. Weight expert outputs by router weights
    2. All-gather weighted outputs + orig_flat_idx from all ranks
    3. Scatter into [flat_total, H] using index_add, then sum over topk
"""

from typing import Any, Optional

import torch
import torch.distributed as dist

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    ExpertTokensMetadata,
    FusedMoeDataRouter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import RouterType


def _pad_and_cat(
    chunks: list[torch.Tensor],
    counts: list[int],
    max_c: int,
    extra_dim: int,
) -> torch.Tensor:
    """Pad each chunk to max_c along dim 0 and concatenate.

    For 2D tensors (e.g. hidden states), extra_dim is the hidden dimension.
    For 1D tensors (e.g. weights/indices), extra_dim is ignored.
    """
    padded = []
    for i, chunk in enumerate(chunks):
        deficit = max_c - counts[i]
        if deficit > 0 and max_c > 0:
            pad_shape = (deficit, extra_dim) if chunk.dim() == 2 else (deficit,)
            pad = torch.zeros(pad_shape, dtype=chunk.dtype, device=chunk.device)
            padded.append(torch.cat([chunk, pad], dim=0))
        else:
            padded.append(chunk)
    return torch.cat(padded, dim=0).contiguous()


class TorchDistEpRouter(FusedMoeDataRouter):
    """EP router using torch.distributed all_to_all for dispatch/combine."""

    @classmethod
    def router_type(cls):
        return RouterType.DEEPEP_NORMAL

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if TorchDistEpRouter can handle the configuration."""
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(resolver.is_ep_enabled(config))
        checker.check(not resolver.is_single_gpu(config))
        checker.check(not resolver.use_all_gather(config))
        checker.check(not resolver.use_low_latency(config))
        # DeepEP must NOT be available (otherwise prefer DeepEP version)
        try:
            import deep_ep  # noqa: F401

            checker.check(False)
        except ImportError:
            pass

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config)

        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.expert_num = config.expert_num
        self.expert_num_per_rank = self.expert_num // self.ep_size

        assert (
            dist.is_initialized()
        ), "torch.distributed must be initialized before using TorchDistEpRouter"
        self._ep_group = dist.group.WORLD

        # Metadata cached between prepare() and finalize()
        self._dispatch_meta: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    #  Dispatch
    # ------------------------------------------------------------------ #

    def _dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        """Route tokens to expert-owning ranks.

        Args:
            hidden_states: [num_tokens, hidden_size]
            topk_ids:      [num_tokens, topk] — global expert IDs
            topk_weights:  [num_tokens, topk]

        Returns:
            recv_hidden:  [recv_count, hidden_size] — tokens for our experts
            recv_weights: [recv_count] — router weights
            recv_lids:    [recv_count] — local expert IDs
            recv_oids:    [recv_count] — original flat indices
            send_counts:  per-dst-rank token counts
        """
        num_tokens, hidden_dim = hidden_states.shape
        topk = topk_ids.shape[1]
        flat_total = num_tokens * topk
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Flatten: [num_tokens, topk] → [flat_total]
        flat_h = (
            hidden_states.unsqueeze(1)
            .expand(-1, topk, -1)
            .reshape(flat_total, hidden_dim)
        )
        flat_ids = topk_ids.reshape(flat_total)
        flat_w = topk_weights.reshape(flat_total)
        flat_oi = torch.arange(flat_total, device=device, dtype=torch.int64)

        # Classify each (token, expert) pair by destination rank
        send_h_chunks: list[torch.Tensor] = []
        send_w_chunks: list[torch.Tensor] = []
        send_i_chunks: list[torch.Tensor] = []
        send_o_chunks: list[torch.Tensor] = []
        send_counts: list[int] = []

        for dst in range(self.ep_size):
            e_start = dst * self.expert_num_per_rank
            e_end = e_start + self.expert_num_per_rank
            mask = (flat_ids >= e_start) & (flat_ids < e_end)

            if mask.any():
                idx = mask.nonzero(as_tuple=False).squeeze(-1)
                local_ids = flat_ids[idx] - e_start
                send_h_chunks.append(flat_h[idx].contiguous())
                send_w_chunks.append(flat_w[idx].contiguous())
                send_i_chunks.append(local_ids.contiguous())
                send_o_chunks.append(flat_oi[idx].contiguous())
                send_counts.append(int(idx.numel()))
            else:
                send_h_chunks.append(
                    torch.empty((0, hidden_dim), dtype=dtype, device=device)
                )
                send_w_chunks.append(torch.empty(0, dtype=torch.float32, device=device))
                send_i_chunks.append(torch.empty(0, dtype=torch.int64, device=device))
                send_o_chunks.append(torch.empty(0, dtype=torch.int64, device=device))
                send_counts.append(0)

        # Gather max chunk size across ranks for padding alignment
        all_max_list: list[int] = [0] * self.ep_size
        dist.all_gather_object(all_max_list, max(send_counts), group=self._ep_group)
        max_count = max(all_max_list)

        # Pad and concatenate each buffer type
        cat_h = _pad_and_cat(send_h_chunks, send_counts, max_count, hidden_dim)
        cat_w = _pad_and_cat(send_w_chunks, send_counts, max_count, 0)
        cat_i = _pad_and_cat(send_i_chunks, send_counts, max_count, 0)
        cat_o = _pad_and_cat(send_o_chunks, send_counts, max_count, 0)

        # All-to-all: exchange padded chunks across all ranks
        r_h = torch.empty_like(cat_h)
        r_w = torch.empty_like(cat_w)
        r_i = torch.empty_like(cat_i)
        r_o = torch.empty_like(cat_o)

        for recv, send in [(r_h, cat_h), (r_w, cat_w), (r_i, cat_i), (r_o, cat_o)]:
            dist.all_to_all(
                list(recv.chunk(self.ep_size, dim=0)),
                list(send.chunk(self.ep_size, dim=0)),
                group=self._ep_group,
            )

        # Extract our rank's received tokens
        recv_total = sum(send_counts)

        def our_slice(t: torch.Tensor) -> torch.Tensor:
            chunk = t.chunk(self.ep_size, dim=0)[self.ep_rank]
            return chunk[:recv_total] if recv_total > 0 else chunk

        if recv_total > 0:
            return (
                our_slice(r_h),
                our_slice(r_w),
                our_slice(r_i),
                our_slice(r_o),
                send_counts,
            )
        else:
            return (
                torch.empty((0, hidden_dim), dtype=dtype, device=device),
                torch.empty(0, dtype=torch.float32, device=device),
                torch.empty(0, dtype=torch.int64, device=device),
                torch.empty(0, dtype=torch.int64, device=device),
                send_counts,
            )

    # ------------------------------------------------------------------ #
    #  Combine
    # ------------------------------------------------------------------ #

    def _combine(
        self,
        expert_output: torch.Tensor,
        recv_weights: torch.Tensor,
        recv_orig_flat_idx: torch.Tensor,
        num_tokens: int,
        topk: int,
        hidden_dim: int,
    ) -> torch.Tensor:
        """Scatter expert results back to original token positions.

        Args:
            expert_output:      [recv_count, hidden_size]
            recv_weights:       [recv_count]
            recv_orig_flat_idx: [recv_count] — original flat indices
            num_tokens:         original num_tokens
            topk:               original topk
            hidden_dim:         hidden dimension

        Returns:
            output: [num_tokens, hidden_size]
        """
        device = expert_output.device
        dtype = expert_output.dtype
        flat_total = num_tokens * topk
        recv_count = expert_output.shape[0]

        if recv_count == 0:
            return torch.zeros((num_tokens, hidden_dim), dtype=dtype, device=device)

        # Weight outputs by router scores
        weighted = expert_output * recv_weights.to(expert_output.dtype).unsqueeze(-1)

        # All-gather weighted outputs + original flat indices from all ranks
        all_gathered_h = [
            torch.empty((recv_count, hidden_dim), dtype=dtype, device=device)
            for _ in range(self.ep_size)
        ]
        all_gathered_o = [
            torch.empty(recv_count, dtype=torch.int64, device=device)
            for _ in range(self.ep_size)
        ]
        dist.all_gather(all_gathered_h, weighted, group=self._ep_group)
        dist.all_gather(all_gathered_o, recv_orig_flat_idx, group=self._ep_group)

        # Scatter into [flat_total, H] using index_add, then sum over topk
        output = torch.zeros((flat_total, hidden_dim), dtype=dtype, device=device)
        for src_h, src_o in zip(all_gathered_h, all_gathered_o):
            if src_h.numel() > 0:
                output.index_add_(0, src_o, src_h)

        return output.reshape(num_tokens, topk, hidden_dim).sum(dim=1)

    # ------------------------------------------------------------------ #
    #  FusedMoeDataRouter interface
    # ------------------------------------------------------------------ #

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        assert a1_scale is None and a2_scale is None, "quant not supported"

        recv_h, recv_w, recv_lids, recv_oids, send_counts = self._dispatch(
            a1, topk_ids, topk_weights
        )
        num_tokens, topk = topk_ids.shape

        self._dispatch_meta = {
            "num_tokens": num_tokens,
            "topk": topk,
            "recv_weights": recv_w,
            "recv_orig_flat_idx": recv_oids,
            "hidden_dim": a1.shape[-1],
        }

        # Compute per-expert token counts from received local expert IDs
        expert_num_tokens_cpu = [0] * self.expert_num_per_rank
        if recv_h.shape[0] > 0:
            lids_cpu = recv_lids.cpu().tolist()
            for lid in lids_cpu:
                expert_num_tokens_cpu[lid] += 1

        return ExpertForwardPayload(
            expert_x=recv_h,
            expert_x_origin_dtype=a1.dtype,
            expert_x_scale=None,
            expert_tokens_meta=ExpertTokensMetadata(
                expert_num_tokens=None,
                expert_num_tokens_cpu=expert_num_tokens_cpu,
            ),
            expert_topk_ids=recv_lids.reshape(-1, 1),
            expert_topk_weights=recv_w.reshape(-1, 1),
        )

    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[dict[str, Any]],
        skip_allreduce: bool = False,
    ) -> torch.Tensor:
        meta = self._dispatch_meta

        return self._combine(
            expert_output=payload.fused_expert_output,
            recv_weights=meta["recv_weights"],
            recv_orig_flat_idx=meta["recv_orig_flat_idx"],
            num_tokens=meta["num_tokens"],
            topk=meta["topk"],
            hidden_dim=meta["hidden_dim"],
        )
