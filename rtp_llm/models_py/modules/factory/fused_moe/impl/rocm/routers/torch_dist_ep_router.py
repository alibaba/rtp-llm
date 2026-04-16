"""Torch distributed-based EP router.

Uses torch.distributed.all_to_all for expert parallel
dispatch and combine, without depending on DeepEP.

Suitable for single-node EP (e.g. 4tp4ep, 8tp8ep) where NCCL/RCCL
all_to_all provides sufficient performance.

Protocol:
  Dispatch:
    1. Flatten [tokens, topk] -> [flat_total] pairs of (hidden, global_expert_id, weight)
    2. Classify each pair by destination rank (rank = global_expert_id // experts_per_rank)
    3. Build per-dst-rank send buffers, pad to max across all ranks
    4. all_to_all hidden/weights/ids/orig_idx -> each rank gets its expert's tokens
       along with the original flattened index for combine
    5. Remap global expert IDs to local IDs for executor

  Combine:
    1. Weight expert outputs by router weights (unless apply_router_weight_on_input)
    2. Reverse all_to_all: re-pad expert outputs, all_to_all back to source ranks
    3. Unpad using send_counts, scatter using saved orig flat indices
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
        import os
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(resolver.is_ep_enabled(config))
        checker.check(not resolver.is_single_gpu(config))
        checker.check(not resolver.use_all_gather(config))
        checker.check(not resolver.use_low_latency(config))
        # When USE_TORCH_DIST_EP=1, skip DeepEP check
        if os.environ.get('USE_TORCH_DIST_EP', '0') == '1':
            return
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
        self._debug_call_count = 0

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
            topk_ids:      [num_tokens, topk] -- global expert IDs
            topk_weights:  [num_tokens, topk]

        Returns:
            recv_hidden:  [recv_total, hidden_size] -- tokens for our experts
            recv_weights: [recv_total] -- router weights
            recv_lids:    [recv_total] -- local expert IDs
            recv_oids:    [recv_total] -- original flat indices (from source rank)
            send_counts:  per-dst-rank token counts (what THIS rank sent)
        """
        num_tokens, hidden_dim = hidden_states.shape
        topk = topk_ids.shape[1]
        flat_total = num_tokens * topk
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Flatten: [num_tokens, topk] -> [flat_total]
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

        # Exchange send_counts via all_to_all_single to get recv_counts
        send_counts_t = torch.tensor(send_counts, dtype=torch.int64, device=device)
        recv_counts_t = torch.empty(self.ep_size, dtype=torch.int64, device=device)
        dist.all_to_all_single(recv_counts_t, send_counts_t, group=self._ep_group)
        recv_counts: list[int] = recv_counts_t.cpu().tolist()

        # Gather max chunk size across all ranks for padding alignment
        local_max = max(max(send_counts), max(recv_counts)) if (send_counts or recv_counts) else 0
        all_max_list: list[int] = [0] * self.ep_size
        dist.all_gather_object(all_max_list, local_max, group=self._ep_group)
        max_count = max(all_max_list)

        if max_count == 0:
            # No tokens to route at all
            return (
                torch.empty((0, hidden_dim), dtype=dtype, device=device),
                torch.empty(0, dtype=torch.float32, device=device),
                torch.empty(0, dtype=torch.int64, device=device),
                torch.empty(0, dtype=torch.int64, device=device),
                send_counts,
            )

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

        # Extract received tokens from ALL source ranks (unpad each chunk)
        recv_total = sum(recv_counts)
        if recv_total > 0:
            recv_h_parts = []
            recv_w_parts = []
            recv_i_parts = []
            recv_o_parts = []
            chunks_h = r_h.chunk(self.ep_size, dim=0)
            chunks_w = r_w.chunk(self.ep_size, dim=0)
            chunks_i = r_i.chunk(self.ep_size, dim=0)
            chunks_o = r_o.chunk(self.ep_size, dim=0)
            for src in range(self.ep_size):
                cnt = recv_counts[src]
                if cnt > 0:
                    recv_h_parts.append(chunks_h[src][:cnt])
                    recv_w_parts.append(chunks_w[src][:cnt])
                    recv_i_parts.append(chunks_i[src][:cnt])
                    recv_o_parts.append(chunks_o[src][:cnt])

            recv_hidden = torch.cat(recv_h_parts, dim=0)
            recv_weights = torch.cat(recv_w_parts, dim=0)
            recv_lids = torch.cat(recv_i_parts, dim=0)
            recv_oids = torch.cat(recv_o_parts, dim=0)
        else:
            recv_hidden = torch.empty((0, hidden_dim), dtype=dtype, device=device)
            recv_weights = torch.empty(0, dtype=torch.float32, device=device)
            recv_lids = torch.empty(0, dtype=torch.int64, device=device)
            recv_oids = torch.empty(0, dtype=torch.int64, device=device)

        # Save metadata for combine phase
        self._dispatch_meta["send_counts"] = send_counts
        self._dispatch_meta["recv_counts"] = recv_counts
        self._dispatch_meta["max_count"] = max_count
        self._dispatch_meta["send_o_chunks"] = send_o_chunks

        return recv_hidden, recv_weights, recv_lids, recv_oids, send_counts

    # ------------------------------------------------------------------ #
    #  Combine
    # ------------------------------------------------------------------ #

    def _combine(
        self,
        expert_output: torch.Tensor,
        recv_weights: torch.Tensor,
        num_tokens: int,
        topk: int,
        hidden_dim: int,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        """Scatter expert results back to original token positions.

        Uses reverse all_to_all to send results back to the originating ranks,
        then scatter using the saved original flat indices.

        Args:
            expert_output:      [recv_total, hidden_size]
            recv_weights:       [recv_total]
            num_tokens:         original num_tokens
            topk:               original topk
            hidden_dim:         hidden dimension
            apply_router_weight_on_input: if True, weights already applied in executor

        Returns:
            output: [num_tokens, hidden_size]
        """
        device = expert_output.device
        dtype = expert_output.dtype
        flat_total = num_tokens * topk

        send_counts = self._dispatch_meta["send_counts"]
        recv_counts = self._dispatch_meta["recv_counts"]
        max_count = self._dispatch_meta["max_count"]
        send_o_chunks = self._dispatch_meta["send_o_chunks"]

        recv_total = sum(recv_counts)

        # Apply router weights if not already applied in executor
        if not apply_router_weight_on_input:
            weighted = expert_output * recv_weights.to(expert_output.dtype).unsqueeze(-1)
        else:
            weighted = expert_output

        # Split expert outputs by source rank using recv_counts, then re-pad
        if recv_total > 0:
            out_chunks = torch.split(weighted, recv_counts, dim=0)
        else:
            out_chunks = [
                torch.empty((0, hidden_dim), dtype=dtype, device=device)
                for _ in range(self.ep_size)
            ]

        # Pad each chunk back to max_count for all_to_all
        out_chunks_list = list(out_chunks)
        cat_send = _pad_and_cat(out_chunks_list, recv_counts, max_count, hidden_dim)

        # Reverse all_to_all: send results back to originating ranks
        cat_recv = torch.empty_like(cat_send)
        dist.all_to_all(
            list(cat_recv.chunk(self.ep_size, dim=0)),
            list(cat_send.chunk(self.ep_size, dim=0)),
            group=self._ep_group,
        )

        # Unpad using send_counts and scatter using saved orig flat indices
        output = torch.zeros((flat_total, hidden_dim), dtype=dtype, device=device)
        chunks_recv = cat_recv.chunk(self.ep_size, dim=0)
        for dst in range(self.ep_size):
            cnt = send_counts[dst]
            if cnt > 0:
                vals = chunks_recv[dst][:cnt]
                idx = send_o_chunks[dst]
                output.index_add_(0, idx, vals)

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

        self._dispatch_meta["num_tokens"] = num_tokens
        self._dispatch_meta["topk"] = topk
        self._dispatch_meta["recv_weights"] = recv_w
        self._dispatch_meta["hidden_dim"] = a1.shape[-1]

        # Compute per-expert token counts from received local expert IDs
        expert_num_tokens_cpu = [0] * self.expert_num_per_rank
        if recv_h.shape[0] > 0:
            lids_cpu = recv_lids.cpu().tolist()
            for lid in lids_cpu:
                expert_num_tokens_cpu[lid] += 1

        # Debug logging (first call only, rank 0 only)
        if self._debug_call_count == 0 and self.ep_rank == 0:
            recv_counts = self._dispatch_meta.get("recv_counts", [])
            print(f"[TorchDistEP DEBUG] rank={self.ep_rank} "
                  f"num_tokens={num_tokens} topk={topk} "
                  f"ep_size={self.ep_size} experts_per_rank={self.expert_num_per_rank}")
            print(f"[TorchDistEP DEBUG] send_counts={send_counts} recv_counts={recv_counts}")
            print(f"[TorchDistEP DEBUG] recv_h.shape={recv_h.shape} "
                  f"recv_lids range=[{recv_lids.min().item() if recv_lids.numel() > 0 else 'empty'}, "
                  f"{recv_lids.max().item() if recv_lids.numel() > 0 else 'empty'}]")
            print(f"[TorchDistEP DEBUG] recv_w stats: mean={recv_w.float().mean().item():.4f} "
                  f"min={recv_w.min().item():.4f} max={recv_w.max().item():.4f}")
            print(f"[TorchDistEP DEBUG] recv_h stats: mean={recv_h.float().mean().item():.6f} "
                  f"std={recv_h.float().std().item():.6f} "
                  f"has_nan={recv_h.isnan().any().item()} has_inf={recv_h.isinf().any().item()}")
            print(f"[TorchDistEP DEBUG] expert_num_tokens_cpu={expert_num_tokens_cpu}")

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

        # Debug logging (first call only, rank 0 only)
        if self._debug_call_count == 0 and self.ep_rank == 0:
            eo = payload.fused_expert_output
            print(f"[TorchDistEP DEBUG finalize] expert_output.shape={eo.shape} "
                  f"apply_router_weight_on_input={apply_router_weight_on_input}")
            print(f"[TorchDistEP DEBUG finalize] expert_output stats: "
                  f"mean={eo.float().mean().item():.6f} std={eo.float().std().item():.6f} "
                  f"has_nan={eo.isnan().any().item()} has_inf={eo.isinf().any().item()}")

        result = self._combine(
            expert_output=payload.fused_expert_output,
            recv_weights=meta["recv_weights"],
            num_tokens=meta["num_tokens"],
            topk=meta["topk"],
            hidden_dim=meta["hidden_dim"],
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        # Debug logging for output
        if self._debug_call_count == 0 and self.ep_rank == 0:
            print(f"[TorchDistEP DEBUG finalize] output.shape={result.shape} "
                  f"mean={result.float().mean().item():.6f} std={result.float().std().item():.6f} "
                  f"has_nan={result.isnan().any().item()} has_inf={result.isinf().any().item()}")
            self._debug_call_count += 1

        return result
