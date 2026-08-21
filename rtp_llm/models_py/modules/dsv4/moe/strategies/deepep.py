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

def _sm120_uses_replicated_tp_tokens(
    cfg: MoeCfg, world: int, cuda_major: int
) -> bool:
    """True only when WORLD is the same TP+EP group with replicated tokens."""
    return (
        cuda_major == 12
        and cfg.tp_size > 1
        and cfg.tp_size == cfg.ep_size
        and cfg.ep_size == world
    )


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
            # The EP loader already gives each rank only its local expert
            # stack. Present that stack as a single-rank grouped problem;
            # global-to-local routing is handled after the collective below.
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
                    12,
                )
            )
            if (
                not torch.cuda.is_current_stream_capturing()
                and os.environ.get("DSV4_SM120_NCCL_EP", "1") != "0"
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

    def _forward_sm120_all_to_all(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch only routed tokens to each expert rank with NCCL.

        A token is sent at most once to a destination rank even when several
        of its top-k experts live there.  The destination computes all of its
        local expert contributions, sends one BF16 row back, and the source
        performs the final per-token sum.  This avoids both the full-token
        all-gather and the full global-output all-reduce used by bring-up.
        """
        dist = torch.distributed
        if not dist.is_initialized():
            raise RuntimeError("SM120 NCCL EP requires torch.distributed")
        group = dist.group.WORLD
        world = dist.get_world_size(group)
        cfg = self.cfg
        if cfg.ep_size != world:
            raise RuntimeError(
                f"SM120 NCCL EP requires ep_size ({cfg.ep_size}) == "
                f"world_size ({world})"
            )
        if cfg.n_routed_experts % world:
            raise RuntimeError(
                "SM120 NCCL EP requires an equal contiguous expert partition"
            )

        experts_per_rank = cfg.n_routed_experts // world
        from flashinfer import mxfp8_quantize

        x_fp8, x_scale = mxfp8_quantize(
            x.contiguous(), is_sf_swizzled_layout=False
        )
        scale_cols = x.size(1) // 32
        x_scale = x_scale.reshape(x.size(0), scale_cols)
        topk = weights.size(1)
        destination = torch.div(
            indices, experts_per_rank, rounding_mode="floor"
        ).clamp_(0, world - 1)
        # V4 top-k=6 reaches almost every EP4 rank. Fixed N-row peer blocks add
        # less than 2% traffic and avoid dynamic compaction and source index_add.
        token_ids = torch.arange(x.size(0), device=x.device)
        padded_all_to_all = True
        send_payload_parts = []
        send_counts = []
        for dst in range(world):
            owned = (destination == dst) & (indices >= 0)
            send_counts.append(token_ids.numel())

            token_owned = owned.index_select(0, token_ids)
            token_weights = weights.index_select(0, token_ids)
            token_indices = indices.index_select(0, token_ids)
            packed_weights = torch.where(
                token_owned, token_weights, torch.zeros_like(token_weights)
            )
            packed_indices = torch.where(
                token_owned,
                token_indices,
                torch.full_like(token_indices, -1),
            ).to(torch.int32)
            send_payload_parts.append(
                torch.cat(
                    [
                        x_fp8.index_select(0, token_ids).view(torch.uint8),
                        x_scale.index_select(0, token_ids),
                        packed_weights.contiguous().view(torch.uint8).reshape(
                            token_ids.numel(), topk * 4
                        ),
                        packed_indices.contiguous().view(torch.uint8).reshape(
                            token_ids.numel(), topk * 4
                        ),
                    ],
                    dim=1,
                )
            )

        send_payload = torch.cat(send_payload_parts, dim=0).contiguous()

        # DP ranks can own different active-token counts at batch boundaries.
        local_count = torch.full(
            (1,), x.size(0), dtype=torch.int64, device=x.device
        )
        gathered_counts = torch.empty(
            world, dtype=torch.int64, device=x.device
        )
        dist.all_gather_into_tensor(gathered_counts, local_count, group=group)
        recv_counts = [int(v) for v in gathered_counts.cpu().tolist()]
        recv_tokens = sum(recv_counts)

        recv_payload = torch.empty(
            (recv_tokens, send_payload.size(1)),
            dtype=torch.uint8,
            device=x.device,
        )
        dist.all_to_all_single(
            recv_payload,
            send_payload,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=group,
        )
        x_end = x.size(1)
        scale_end = x_end + scale_cols
        weight_end = scale_end + topk * 4
        recv_x = recv_payload[:, :x_end].contiguous().view(torch.float8_e4m3fn)
        recv_scale = recv_payload[:, x_end:scale_end].contiguous()
        recv_w = (
            recv_payload[:, scale_end:weight_end]
            .contiguous()
            .view(torch.float32)
        )
        recv_i = (
            recv_payload[:, weight_end:]
            .contiguous()
            .view(torch.int32)
            .to(torch.int64)
        )
        local_i = recv_i - cfg.local_expert_start
        valid = (local_i >= 0) & (local_i < cfg.n_local_experts)
        local_w = recv_w * valid.to(recv_w.dtype)
        local_i = local_i.clamp(0, cfg.n_local_experts - 1)

        equal_peer_counts = all(count == x.size(0) for count in recv_counts)
        fp8_combine = True
        pipeline_combine = (
            equal_peer_counts
            and os.environ.get("DSV4_SM120_NCCL_EP_PIPELINE", "0") != "0"
        )
        if pipeline_combine:
            if self._sm120_grouped is None:
                raise RuntimeError("SM120 MXFP8 dispatch requires grouped FP4 MoE")

            # recv_* is source-rank-major: [source_rank, local_token, ...].
            # Process an equally sized slice from every source rank, then send
            # that slice back immediately.  Keeping one asynchronous combine
            # in flight overlaps PCIe traffic and peer skew with the next
            # FlashInfer fused-MoE chunk without changing token ordering.
            chunk_tokens = int(os.environ.get("DSV4_MOE_CHUNK_TOKENS", "4096"))
            peer_chunk = max(chunk_tokens // world, 1)
            recv_x_by_peer = recv_x.view(world, x.size(0), x.size(1))
            recv_scale_by_peer = recv_scale.view(
                world, x.size(0), scale_cols
            )
            local_w_by_peer = local_w.view(world, x.size(0), topk)
            local_i_by_peer = local_i.view(world, x.size(0), topk)
            pending = []
            output_parts = []

            def finish_oldest_combine() -> None:
                work, returned_payload, _combine_payload, rows = pending.pop(0)
                work.wait()
                from .._nccl_ep_combine_triton import mxfp8_dequant_peer_sum

                output_parts.append(
                    mxfp8_dequant_peer_sum(
                        returned_payload, rows, x.size(1), world
                    )
                )

            for begin in range(0, x.size(0), peer_chunk):
                end = min(begin + peer_chunk, x.size(0))
                rows = end - begin
                chunk_x = recv_x_by_peer[:, begin:end].contiguous().view(
                    world * rows, x.size(1)
                )
                chunk_scale = recv_scale_by_peer[:, begin:end].contiguous().view(
                    world * rows, scale_cols
                )
                chunk_w = local_w_by_peer[:, begin:end].contiguous().view(
                    world * rows, topk
                )
                chunk_i = local_i_by_peer[:, begin:end].contiguous().view(
                    world * rows, topk
                )
                chunk_output = self._sm120_grouped.forward_sm120_mxfp8(
                    chunk_x, chunk_scale, chunk_w, chunk_i
                ).to(x.dtype)
                combine_q, combine_scale = mxfp8_quantize(
                    chunk_output, is_sf_swizzled_layout=False
                )
                combine_payload = torch.cat(
                    [
                        combine_q.view(torch.uint8),
                        combine_scale.reshape(world * rows, scale_cols),
                    ],
                    dim=1,
                ).contiguous()
                returned_payload = torch.empty_like(combine_payload)
                splits = [rows] * world
                work = dist.all_to_all_single(
                    returned_payload,
                    combine_payload,
                    output_split_sizes=splits,
                    input_split_sizes=splits,
                    group=group,
                    async_op=True,
                )
                # Keep both buffers alive until NCCL signals completion.
                pending.append((work, returned_payload, combine_payload, rows))
                if len(pending) == 2:
                    finish_oldest_combine()
            while pending:
                finish_oldest_combine()
            return torch.cat(output_parts, dim=0)

        if self._sm120_grouped is not None:
            output_parts = []
            chunk_tokens = int(os.environ.get("DSV4_MOE_CHUNK_TOKENS", "4096"))
            for begin in range(0, recv_tokens, chunk_tokens):
                end = min(begin + chunk_tokens, recv_tokens)
                output_parts.append(
                    self._sm120_grouped.forward_sm120_mxfp8(
                        recv_x[begin:end],
                        recv_scale[begin:end],
                        local_w[begin:end],
                        local_i[begin:end],
                    ).to(x.dtype)
                )
            recv_output = (
                torch.cat(output_parts, dim=0)
                if output_parts
                else torch.empty((0, x.size(1)), dtype=x.dtype, device=x.device)
            )
        else:
            raise RuntimeError("SM120 MXFP8 dispatch requires grouped FP4 MoE")

        reduce_scatter_combine = (
            padded_all_to_all
            and equal_peer_counts
            and os.environ.get("DSV4_SM120_NCCL_EP_REDUCE_SCATTER", "0") != "0"
        )
        if reduce_scatter_combine:
            reduced = torch.empty(
                (x.size(0), x.size(1)), dtype=x.dtype, device=x.device
            )
            dist.reduce_scatter_tensor(
                reduced, recv_output.contiguous(), group=group
            )
            return reduced.float()

        if fp8_combine:
            combine_q, combine_scale = mxfp8_quantize(
                recv_output.contiguous(), is_sf_swizzled_layout=False
            )
            combine_scale = combine_scale.reshape(recv_tokens, scale_cols)
            combine_payload = torch.cat(
                [combine_q.view(torch.uint8), combine_scale], dim=1
            ).contiguous()
            returned_payload = torch.empty(
                (send_payload.size(0), combine_payload.size(1)),
                dtype=torch.uint8,
                device=x.device,
            )
            dist.all_to_all_single(
                returned_payload,
                combine_payload,
                output_split_sizes=send_counts,
                input_split_sizes=recv_counts,
                group=group,
            )
            from .._nccl_ep_combine_triton import mxfp8_dequant_peer_sum

            return mxfp8_dequant_peer_sum(
                returned_payload, x.size(0), x.size(1), world
            )

        returned = torch.empty(
            (send_payload.size(0), x.size(1)), dtype=x.dtype, device=x.device
        )
        dist.all_to_all_single(
            returned,
            recv_output,
            output_split_sizes=send_counts,
            input_split_sizes=recv_counts,
            group=group,
        )
        if padded_all_to_all:
            # Every peer returns exactly one row per source token, in source
            # token order.  Invalid peer routes produced zero expert output.
            return returned.view(world, x.size(0), x.size(1)).float().sum(dim=0)
        output = torch.zeros(
            (x.size(0), x.size(1)), dtype=torch.float32, device=x.device
        )
        output.index_add_(0, send_token_ids, returned.float())
        return output

    def _forward_sm120_collective(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Correctness EP path for SM120 without an ACCL-EP cubin.

        Gather the small decode batch on every EP rank, compute only this
        rank's local expert slice, then sum the partial outputs.  This keeps
        weights EP-sharded and is intentionally used only for SM120 bring-up.
        """
        dist = torch.distributed
        if not dist.is_initialized():
            raise RuntimeError("SM120 EP fallback requires torch.distributed")
        group = dist.group.WORLD
        world = dist.get_world_size(group)
        rank = dist.get_rank(group)
        # Attention TP all-reduces its output, so when TP and EP are the same
        # rank group every rank enters MoE with an identical full token set.
        # Gathering those rows again creates ``tp_size`` duplicate copies and
        # multiplies routed GEMM work. CP reports effective tp_size=1 and DP
        # also remains on the disjoint-token gather path below.
        cuda_major = torch.cuda.get_device_capability(x.device)[0]
        replicated_tp_tokens = _sm120_uses_replicated_tp_tokens(
            self.cfg, world, cuda_major
        )
        if replicated_tp_tokens:
            counts = [x.size(0)]
            all_x = x
            all_w = weights
            all_i = indices.to(torch.int64)
            local_begin = 0
            local_end = x.size(0)
        else:
            if torch.cuda.is_current_stream_capturing():
                # Decode graphs are captured for a fixed per-rank batch size.
                # Avoid CPU scalar materialization and device-to-host reads inside
                # capture; every DP rank captures the same local batch shape.
                counts = [x.size(0)] * world
            else:
                n = torch.full((1,), x.size(0), dtype=torch.int64, device=x.device)
                counts_t = [torch.empty_like(n) for _ in range(world)]
                dist.all_gather(counts_t, n, group=group)
                counts = [int(v.item()) for v in counts_t]
            max_n = max(max(counts), 1)

            def gather_padded(t: torch.Tensor, fill: float | int) -> list[torch.Tensor]:
                padded = torch.full(
                    (max_n, *t.shape[1:]), fill, dtype=t.dtype, device=t.device
                )
                if t.size(0):
                    padded[: t.size(0)].copy_(t)
                if cuda_major == 12:
                    gathered = torch.empty(
                        (world * max_n, *t.shape[1:]), dtype=t.dtype, device=t.device
                    )
                    dist.all_gather_into_tensor(gathered, padded, group=group)
                    return list(
                        gathered.view(world, max_n, *t.shape[1:]).unbind(0)
                    )
                gathered_list = [torch.empty_like(padded) for _ in range(world)]
                dist.all_gather(gathered_list, padded, group=group)
                return gathered_list

            xs = gather_padded(x, 0.0)
            ws = gather_padded(weights, 0.0)
            ids = gather_padded(indices.to(torch.int64), -1)
            all_x = torch.cat([t[:c] for t, c in zip(xs, counts)], dim=0)
            all_w = torch.cat([t[:c] for t, c in zip(ws, counts)], dim=0)
            all_i = torch.cat([t[:c] for t, c in zip(ids, counts)], dim=0)
            local_begin = sum(counts[:rank])
            local_end = local_begin + counts[rank]
        if self._sm120_grouped is not None:
            local_i = all_i - self.cfg.local_expert_start
            valid = (local_i >= 0) & (local_i < self.cfg.n_local_experts)
            local_w = all_w * valid.to(all_w.dtype)
            local_i = local_i.clamp(0, self.cfg.n_local_experts - 1)
            try:
                # The FlashInfer SM120 path materialises routed activations for
                # its input rows.  Bound that model-forward workspace instead
                # of letting CP all-gathered 16k/32k warmup requests consume the
                # remaining HBM in one call.  Every EP rank has the same all_x
                # ordering, so the per-chunk all-reduces are collective-safe.
                chunk_tokens = 4096
                output = torch.empty(
                    (local_end - local_begin, all_x.size(1)),
                    dtype=torch.float32,
                    device=all_x.device,
                )
                for begin in range(0, all_x.size(0), chunk_tokens):
                    end = min(begin + chunk_tokens, all_x.size(0))
                    partial = self._sm120_grouped(
                        all_x[begin:end], local_w[begin:end], local_i[begin:end]
                    )
                    dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=group)
                    copy_begin = max(begin, local_begin)
                    copy_end = min(end, local_end)
                    if copy_begin < copy_end:
                        output[
                            copy_begin - local_begin : copy_end - local_begin
                        ].copy_(partial[copy_begin - begin : copy_end - begin])
                return output
            except BaseException:
                # PyWrappedModel crosses a C++ worker-thread boundary whose
                # terminate path otherwise hides the originating Python error.
                import traceback
                traceback.print_exc()
                raise
        else:
            partial = self._local._forward_into_buf(
                all_x,
                all_w,
                all_i,
                local_start=self.cfg.local_expert_start,
                local_end=self.cfg.local_expert_end,
            )
        dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=group)
        if replicated_tp_tokens:
            return partial.float()
        offset = sum(counts[:rank])
        return partial[offset : offset + counts[rank]].float()
