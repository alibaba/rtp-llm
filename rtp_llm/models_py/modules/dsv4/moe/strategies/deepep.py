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
            gathered = [torch.empty_like(padded) for _ in range(world)]
            dist.all_gather(gathered, padded, group=group)
            return gathered

        xs = gather_padded(x, 0.0)
        ws = gather_padded(weights, 0.0)
        ids = gather_padded(indices.to(torch.int64), -1)
        all_x = torch.cat([t[:c] for t, c in zip(xs, counts)], dim=0)
        all_w = torch.cat([t[:c] for t, c in zip(ws, counts)], dim=0)
        all_i = torch.cat([t[:c] for t, c in zip(ids, counts)], dim=0)
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
                local_begin = sum(counts[:rank])
                local_end = local_begin + counts[rank]
                output = torch.empty(
                    (counts[rank], all_x.size(1)),
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
        offset = sum(counts[:rank])
        return partial[offset : offset + counts[rank]].float()
