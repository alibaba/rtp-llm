"""DeepEPStrategy: ACCL-EP normal-mode dispatch + per-expert local compute + combine.

EP > 1 DeepEP implementation. DSV4 automatic strategy selection no longer
falls back here when Mega is unavailable; EP>1 requires Mega and fails fast.
This class is kept as an explicit implementation for targeted tests or
experiments. Composes ``LocalLoopStrategy`` for the local per-expert compute
on the dispatched recv tokens.

Direct port of the pre-refactor ``_routed_experts_deepep`` +
``_pad_topk_for_deepep`` + the ``_DEEPEP_SUPPORTED_TOPK`` constant.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Dict, Optional, Tuple

import torch

from .base import MoeCfg, RoutedExpertsStrategy, register_strategy
from .local_loop import LocalLoopStrategy
from .grouped_fp4 import GroupedFP4Strategy, _has_fp8_fp4_grouped_kernel


# ACCL-EP's intranode dispatch kernel has a compile-time switch over
# ``num_topk`` that only covers {2, 4, 8, 16} (asserts false on others —
# intranode.cu:2237 "Unsupported num_topk"). V4-Flash uses
# ``n_activated_experts = 6``; we pad both ``indices`` and ``weights``
# up to 8 slots with ``-1`` and ``0.0`` so the dispatch accepts them,
# and the padding slots are silently dropped by the per-expert loop
# (``torch.where(idx == -1)`` never matches a real expert index).
_DEEPEP_SUPPORTED_TOPK = (2, 4, 8, 16)

def _sm120_uses_replicated_tp_tokens(cfg: MoeCfg, world: int) -> bool:
    return cfg.tp_size > 1 and cfg.tp_size == cfg.ep_size == world

@register_strategy
class DeepEPStrategy(RoutedExpertsStrategy):
    name = "deepep"

    def __init__(self, cfg: MoeCfg):
        super().__init__(cfg)
        # Composition: hold a LocalLoopStrategy instance for the per-expert
        # local compute on dispatched recv tokens. Registered as a child
        # nn.Module so its ``experts`` ModuleList propagates through
        # ``MoE.to(device)`` / state_dict.
        self._sm120_grouped = None
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 12 \
                and _has_fp8_fp4_grouped_kernel():
            local_cfg = replace(
                cfg,
                ep_size=1,
                ep_rank=0,
                n_routed_experts=cfg.n_local_experts,
                local_expert_start=0,
                local_expert_end=cfg.n_local_experts,
            )
            self._sm120_grouped = GroupedFP4Strategy(local_cfg)
            self._local = self._sm120_grouped
        else:
            self._local = LocalLoopStrategy(cfg)

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        # ep_size > 1. Mega-vs-DeepEP priority is enforced by registry order
        # (Mega registered first).
        return cfg.ep_size > 1

    def setup_weights(self, layer_weights: Dict) -> None:
        """Delegates to ``LocalLoopStrategy.setup_weights`` — DeepEP has no
        weights of its own; it dispatches recv tokens to the per-expert loop
        owned by the inner ``LocalLoopStrategy``.
        """
        self._local.setup_weights(layer_weights)

    @staticmethod
    def _pad_topk_for_deepep(
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad ``(indices, weights)`` to the nearest supported topk width.

        See ``_DEEPEP_SUPPORTED_TOPK`` docstring above.
        """
        n_act = indices.size(-1)
        if n_act in _DEEPEP_SUPPORTED_TOPK:
            return indices, weights
        pad_to = next((k for k in _DEEPEP_SUPPORTED_TOPK if k > n_act), None)
        if pad_to is None:
            raise RuntimeError(
                f"n_activated_experts={n_act} exceeds largest DeepEP-supported "
                f"topk ({max(_DEEPEP_SUPPORTED_TOPK)})"
            )
        N = indices.size(0)
        pad_n = pad_to - n_act
        pad_idx = torch.full((N, pad_n), -1, dtype=indices.dtype, device=indices.device)
        pad_w = torch.zeros((N, pad_n), dtype=weights.dtype, device=weights.device)
        return (
            torch.cat([indices, pad_idx], dim=-1),
            torch.cat([weights, pad_w], dim=-1),
        )

    def forward(
        self,
        x: torch.Tensor,        # [N, D] local rank's tokens (BF16)
        weights: torch.Tensor,  # [N, k] fp32
        indices: torch.Tensor,  # [N, k] int64 global expert IDs
    ) -> torch.Tensor:
        """DP+EP path: DeepEP normal dispatch → local per-expert compute
        → DeepEP combine. Requires ``init_deepep_wrapper`` to have been
        called by the engine (``backend_manager.py``).
        """
        if x.is_cuda and torch.cuda.get_device_capability(x.device)[0] == 12:
            dist = torch.distributed
            replicated_tp_tokens = (
                dist.is_initialized()
                and _sm120_uses_replicated_tp_tokens(
                    self.cfg,
                    dist.get_world_size(dist.group.WORLD),
                )
            )
            if (
                not replicated_tp_tokens
                and torch.cuda.is_current_stream_capturing()
                and self._sm120_grouped is not None
                and x.size(0) <= 4
            ):
                return self._forward_sm120_fixed_ep(x, weights, indices)
            if (
                not torch.cuda.is_current_stream_capturing()
                and not replicated_tp_tokens
                and self._sm120_grouped is not None
            ):
                return self._forward_sm120_all_to_all(x, weights, indices)
            return self._forward_sm120_collective(x, weights, indices)
        from rtp_llm.models_py.distributed.deepep_wrapper import (
            DeepEPMode,
            DeepEPWrapper,
        )

        if DeepEPWrapper._instance is None:
            raise RuntimeError(
                "DeepEPWrapper not initialised; ep_size>1 requires "
                "init_deepep_wrapper() at engine startup (enable via "
                "--use_deepep_moe 1)."
            )
        wrapper = DeepEPWrapper._instance
        assert (
            wrapper.mode == DeepEPMode.NORMAL
        ), f"expected NORMAL DeepEP mode, got {wrapper.mode}"
        buf = wrapper.buffer
        cfg = self.cfg

        # Pad topk to nearest supported value (V4's 6 → 8).
        indices_p, weights_p = self._pad_topk_for_deepep(indices, weights)

        # 1. Dispatch layout. indices cast to int64 already.
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            _,
        ) = buf.get_dispatch_layout(indices_p, cfg.n_routed_experts)

        # 2. Dispatch the BF16 tokens + topk scaffolding.
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            _,
        ) = buf.dispatch(
            x,
            None,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            is_token_in_rank,
            num_tokens_per_expert,
            indices_p,
            weights_p,
            expert_alignment=1,
        )

        # 3. Local per-expert compute. ACCL-EP's dispatch returns
        # ``recv_topk_idx`` in the LOCAL index space ``[0, n_local_experts)``
        # (with -1 for tokens not destined for any local expert), NOT the
        # global expert id. Shift to global so the per-expert loop in
        # ``LocalLoopStrategy`` indexes ``self._local.experts[global_i]``
        # correctly. Also force int64 and contiguous — the ACCL tensor
        # sometimes comes back with a non-standard dtype that triggers
        # ``torch.where(idx == i)`` with "unknown parameter type".
        M = recv_x.size(0)
        if M > 0:
            global_topk_idx = recv_topk_idx.to(torch.int64).contiguous()
            # Shift local→global; keep -1 as -1 (won't match any expert id).
            global_topk_idx = torch.where(
                global_topk_idx == -1,
                global_topk_idx,
                global_topk_idx + cfg.local_expert_start,
            )
            # _local.forward() allocates its own y_local buffer (its
            # _local_y_buf), runs the [local_start, local_end) loop, and
            # returns the fp32 accumulator. We pass through.
            y_local = self._local._forward_into_buf(
                recv_x.contiguous(),
                recv_topk_weights.contiguous(),
                global_topk_idx,
                local_start=cfg.local_expert_start,
                local_end=cfg.local_expert_end,
            )
        else:
            # M == 0: no recv tokens this rank — produce a fresh empty
            # fp32 accumulator so combine still has a valid tensor to send.
            y_local = torch.zeros(M, cfg.dim, dtype=torch.float32, device=recv_x.device)

        # 4. Combine back to source ranks. combine expects the tensor
        # dtype to match x (BF16) — cast the fp32 accumulator.
        y_combined, _, _ = buf.combine(
            y_local.to(x.dtype),
            handle,
        )
        return y_combined.float()
    def _forward_sm120_fixed_ep(self, x, weights, indices) -> torch.Tensor:
        dist = torch.distributed
        group = dist.group.WORLD; world = dist.get_world_size(group)
        rank = dist.get_rank(group)
        n, d = x.shape
        topk = indices.size(1)
        x_bytes = d * x.element_size()
        weight_bytes = topk * weights.element_size()
        local_payload = torch.cat((x.contiguous().view(torch.uint8),
            weights.contiguous().view(torch.uint8).reshape(n, weight_bytes),
            indices.to(torch.int32).contiguous().view(torch.uint8).reshape(n, topk * 4)), dim=1)
        payload = torch.zeros((world, *local_payload.shape),
                              dtype=torch.uint8, device=x.device)
        payload[rank].copy_(local_payload)
        dist.all_reduce(payload.view(torch.int32), op=dist.ReduceOp.SUM, group=group)
        gathered = payload.view(world * n, -1)
        all_x = gathered[:, :x_bytes].contiguous().view(x.dtype).reshape(world * n, d)
        all_w = gathered[:, x_bytes:x_bytes + weight_bytes].contiguous() \
            .view(weights.dtype).reshape(world * n, topk)
        all_i = gathered[:, x_bytes + weight_bytes:].contiguous().view(torch.int32) \
            .to(torch.int64).reshape(world * n, topk)
        local_i = all_i - self.cfg.local_expert_start
        valid = (local_i >= 0) & (local_i < self.cfg.n_local_experts)
        local_w = all_w * valid.to(all_w.dtype)
        local_i.clamp_(0, self.cfg.n_local_experts - 1)
        partial = self._sm120_grouped._forward_capture_sm120(all_x, local_w, local_i) \
            .to(x.dtype).contiguous()
        dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=group)
        return partial.view(world, n, d)[rank].float()
    def _forward_sm120_all_to_all(self, x, weights, indices) -> torch.Tensor:
        dist = torch.distributed
        group = dist.group.WORLD; world = dist.get_world_size(group)
        cfg = self.cfg
        experts_per_rank = cfg.n_routed_experts // world
        from flashinfer import mxfp8_quantize
        x_fp8, x_scale = mxfp8_quantize(x.contiguous(), is_sf_swizzled_layout=False)
        scale_cols = x.size(1) // 32
        x_scale = x_scale.reshape(x.size(0), scale_cols)
        topk = weights.size(1)
        x_end = x.size(1)
        scale_end = x_end + scale_cols
        weight_end = scale_end + topk * 4
        payload_cols = weight_end + topk * 4
        indices_i32 = indices.to(torch.int32)
        send_counts = [x.size(0)] * world
        long_cp_allgather = x.size(0) > 4096
        if long_cp_allgather:
            local_payload = torch.cat([x_fp8.view(torch.uint8), x_scale,
                weights.contiguous().view(torch.uint8).reshape(x.size(0), topk * 4),
                indices_i32.contiguous().view(torch.uint8).reshape(x.size(0), topk * 4)], dim=1)
            recv_counts = send_counts
            recv_payload = torch.empty((world * x.size(0), payload_cols),
                                       dtype=torch.uint8, device=x.device)
            dist.all_gather_into_tensor(recv_payload, local_payload, group=group)
        else:
            local_count = torch.full((1,), x.size(0), dtype=torch.int64, device=x.device)
            gathered_counts = torch.empty(world, dtype=torch.int64, device=x.device)
            dist.all_gather_into_tensor(gathered_counts, local_count, group=group)
            recv_counts = [int(v) for v in gathered_counts.cpu().tolist()]
            recv_payload = torch.empty((sum(recv_counts), payload_cols),
                                       dtype=torch.uint8, device=x.device)
        destination = torch.div(indices, experts_per_rank,
                                rounding_mode="floor").clamp_(0, world - 1)
        if not long_cp_allgather:
            send_payload_by_peer = torch.empty((world, x.size(0), payload_cols),
                                               dtype=torch.uint8, device=x.device)
            send_payload_by_peer[:, :, :x_end].copy_(x_fp8.view(torch.uint8))
            send_payload_by_peer[:, :, x_end:scale_end].copy_(x_scale)
            send_weights = send_payload_by_peer[:, :, scale_end:weight_end].view(torch.float32)
            send_indices = send_payload_by_peer[:, :, weight_end:].view(torch.int32)
            for dst in range(world):
                owned = (destination == dst) & (indices >= 0)
                torch.where(owned, weights, torch.zeros_like(weights), out=send_weights[dst])
                torch.where(owned, indices_i32, torch.full_like(indices_i32, -1),
                            out=send_indices[dst])
            send_payload = send_payload_by_peer.view(-1, payload_cols)
            dist.all_to_all_single(recv_payload, send_payload,
                output_split_sizes=recv_counts, input_split_sizes=send_counts, group=group)
        recv_tokens = sum(recv_counts)
        recv_x = recv_payload[:, :x_end].contiguous().view(torch.float8_e4m3fn)
        recv_scale = recv_payload[:, x_end:scale_end].contiguous()
        recv_w = recv_payload[:, scale_end:weight_end].contiguous().view(torch.float32)
        recv_i = recv_payload[:, weight_end:].contiguous().view(torch.int32).to(torch.int64)
        local_i = recv_i - cfg.local_expert_start
        valid = (local_i >= 0) & (local_i < cfg.n_local_experts)
        local_w = recv_w * valid.to(recv_w.dtype)
        local_i = torch.where(valid, local_i, 0)
        if self._sm120_grouped is not None:
            output_parts = []
            chunk_tokens = int(os.environ.get("DSV4_MOE_CHUNK_TOKENS", "4096"))
            for begin in range(0, recv_tokens, chunk_tokens):
                end = min(begin + chunk_tokens, recv_tokens)
                output_parts.append(self._sm120_grouped._forward_sm120(
                    recv_x[begin:end], local_w[begin:end], local_i[begin:end],
                    input_scale=recv_scale[begin:end]).to(x.dtype))
            recv_output = torch.cat(output_parts, dim=0) if output_parts else \
                torch.empty((0, x.size(1)), dtype=x.dtype, device=x.device)
        else:
            raise RuntimeError("SM120 MXFP8 dispatch requires grouped FP4 MoE")
        combine_q, combine_scale = mxfp8_quantize(
            recv_output.contiguous(), is_sf_swizzled_layout=False)
        combine_scale = combine_scale.reshape(recv_tokens, scale_cols)
        combine_payload = torch.cat([combine_q.view(torch.uint8), combine_scale], dim=1).contiguous()
        returned_payload = torch.empty((world * x.size(0), combine_payload.size(1)),
            dtype=torch.uint8, device=x.device)
        dist.all_to_all_single(returned_payload, combine_payload,
            output_split_sizes=send_counts, input_split_sizes=recv_counts, group=group)
        from .._nccl_ep_combine_triton import mxfp8_dequant_peer_sum
        return mxfp8_dequant_peer_sum(returned_payload, x.size(0), x.size(1), world)

    def _forward_sm120_collective(self, x, weights, indices) -> torch.Tensor:
        dist = torch.distributed
        if not dist.is_initialized(): raise RuntimeError("SM120 EP fallback requires torch.distributed")
        group = dist.group.WORLD
        world = dist.get_world_size(group)
        if not _sm120_uses_replicated_tp_tokens(self.cfg, world): raise RuntimeError("non-replicated SM120 EP must use dispatch")
        if self._sm120_grouped is None: raise RuntimeError("SM120 requires grouped FP8xFP4 MoE")
        local_i = indices.to(torch.int64) - self.cfg.local_expert_start
        valid = (local_i >= 0) & (local_i < self.cfg.n_local_experts)
        local_w = weights * valid.to(weights.dtype)
        local_i.clamp_(0, self.cfg.n_local_experts - 1)
        output = torch.empty_like(x, dtype=torch.float32)
        for begin in range(0, x.size(0), 4096):
            end = min(begin + 4096, x.size(0))
            partial = self._sm120_grouped(x[begin:end], local_w[begin:end], local_i[begin:end])
            dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=group)
            output[begin:end].copy_(partial)
        return output
