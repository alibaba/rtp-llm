"""Stage-local MXFP8 dispatch/return with ordered FP32 peer accumulation.

Partials round through BF16 and MXFP8; this is not BF16-backend bit equivalence."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Dict, Optional, Tuple

import torch

from ..._profiler import record_function_range
from .._nccl_ep_mxfp8_combine import SCALE_BLOCK, mxfp8_dequant_peer_sum
from .base import MoeCfg, RoutedExpertsStrategy, register_strategy
from .grouped_fp4 import GroupedFP4Strategy, _has_fp8_fp4_grouped_kernel
from .local_loop import LocalLoopStrategy

BACKEND_NAME = "fork_nccl_mxfp8"

_LOGGED = False

# Dispatch bytes: FP8 D + scales D/32 + weights topk*4 + IDs topk*4.
# Return bytes: D + D/32.
_WEIGHT_BYTES = 4
_ID_BYTES = 4


def _is_sm120_runtime() -> bool:
    from rtp_llm.models_py.utils.arch import is_sm120

    return is_sm120()


@register_strategy
class NcclEpMxfp8Strategy(RoutedExpertsStrategy):
    """Stage-local NCCL all_to_all_single with MXFP8 activation and return."""

    name = BACKEND_NAME
    requires_synchronized_chunk_schedule = True
    # The MoE layer adds the shared expert; this strategy returns routed only.
    routed_includes_shared = False

    def __init__(self, cfg: MoeCfg):
        super().__init__(cfg)
        if cfg.ep_size <= 1:
            raise RuntimeError(
                "%s requires ep_size > 1 (got %d)" % (BACKEND_NAME, cfg.ep_size)
            )
        # Local expert compute sees only this rank's shard, in a local [0, E_local)
        # id space; the dispatch path masks and shifts ids before calling it.
        local_cfg = replace(
            cfg,
            ep_size=1,
            ep_rank=0,
            n_routed_experts=cfg.n_local_experts,
            local_expert_start=0,
            local_expert_end=cfg.n_local_experts,
        )
        if _is_sm120_runtime() and _has_fp8_fp4_grouped_kernel():
            self._local: RoutedExpertsStrategy = GroupedFP4Strategy(local_cfg)
        else:
            self._local = LocalLoopStrategy(local_cfg)
        self._chunk_extent_tensor: Optional[torch.Tensor] = None
        self._count_tensor: Optional[torch.Tensor] = None
        self._count_gather: Optional[torch.Tensor] = None

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        return cfg.ep_size > 1

    def setup_weights(self, layer_weights: Dict) -> None:
        self._local.setup_weights(layer_weights)

    # ---- stage topology ---------------------------------------------------

    def _stage(self) -> Tuple[object, int, int]:
        """Resolve this stage's roster; legacy WORLD is valid only for a single expert stage."""
        dist = torch.distributed
        if not dist.is_initialized():
            raise RuntimeError("%s requires torch.distributed" % BACKEND_NAME)

        ctx = self.cfg.stage_context
        if ctx is not None:
            group = ctx.process_group
            world, rank = ctx.group_size, ctx.group_rank
        else:
            group = dist.group.WORLD
            world, rank = dist.get_world_size(group), dist.get_rank(group)
            if world != self.cfg.ep_size:
                raise RuntimeError(
                    "%s: no EpStageContext and world=%d != ep_size=%d, so WORLD is "
                    "not the expert roster. Build the stage context (DSV4_PP_EP_ENABLE=1) "
                    "instead of exchanging expert payloads across stages."
                    % (BACKEND_NAME, world, self.cfg.ep_size)
                )
        if world != self.cfg.ep_size or rank != self.cfg.ep_rank:
            raise RuntimeError(
                "%s: stage roster %d/rank %d does not match the expert partition "
                "ep_size/ep_rank=%d/%d"
                % (BACKEND_NAME, world, rank, self.cfg.ep_size, self.cfg.ep_rank)
            )
        return group, world, rank

    def synchronized_chunk_extent(self, local_tokens: int, device: torch.device) -> int:
        group, _, _ = self._stage()
        if (
            self._chunk_extent_tensor is None
            or self._chunk_extent_tensor.device != device
        ):
            self._chunk_extent_tensor = torch.empty(
                (1,), dtype=torch.int64, device=device
            )
        self._chunk_extent_tensor.fill_(int(local_tokens))
        torch.distributed.all_reduce(
            self._chunk_extent_tensor, op=torch.distributed.ReduceOp.MAX, group=group
        )
        return int(self._chunk_extent_tensor.item())

    # ---- forward ----------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        global _LOGGED
        if not x.is_cuda:
            raise RuntimeError("%s requires a CUDA device" % BACKEND_NAME)
        group, world, rank = self._stage()
        if not _LOGGED:
            _LOGGED = True
            logging.info(
                "[DSV4 MoE] %s: stage_world=%d stage_rank=%d ep=%d/%d topk=%d "
                "hidden=%d local_rows=%d",
                BACKEND_NAME,
                world,
                rank,
                self.cfg.ep_rank,
                self.cfg.ep_size,
                int(weights.size(1)),
                int(x.size(1)),
                int(x.size(0)),
            )

        hidden = int(x.size(1))
        topk = int(weights.size(1))
        scale_cols = (hidden + SCALE_BLOCK - 1) // SCALE_BLOCK
        payload_cols = hidden + scale_cols + topk * _WEIGHT_BYTES + topk * _ID_BYTES
        device = x.device

        n_local = int(x.size(0))
        counts = self._exchange_counts(n_local, group, world, device)
        n_total = sum(counts)

        with record_function_range("dsv4.moe.a2a.dispatch"):
            send_payload = self._pack_dispatch(
                x, weights, indices, hidden, scale_cols, topk, payload_cols, world
            )
            recv = self._all_to_all(
                send_payload, [n_local] * world, counts, payload_cols, group
            )
            del send_payload

        with record_function_range("dsv4.moe.a2a.compute"):
            local_out = self._compute_local(
                recv, counts, hidden, scale_cols, topk, payload_cols
            )

        del recv
        with record_function_range("dsv4.moe.a2a.return"):
            ret_payload = self._pack_return(local_out, hidden, scale_cols, n_total)
            del local_out
            ret_recv = self._all_to_all(
                ret_payload, counts, [n_local] * world, ret_payload.size(1), group
            )
            del ret_payload
            summed = mxfp8_dequant_peer_sum(
                ret_recv,
                n_rows=n_local,
                hidden_size=hidden,
                world_size=world,
                out_dtype=torch.float32,
            )
        return summed.to(x.dtype)

    # ---- steps ------------------------------------------------------------

    def _exchange_counts(self, n_local: int, group, world: int, device) -> list:
        """One stage-local all_gather of the local row count."""
        if (
            self._count_tensor is None
            or self._count_tensor.device != device
            or self._count_tensor.numel() != world
        ):
            self._count_tensor = torch.empty(
                (world, 1), dtype=torch.int64, device=device
            )
        self._count_tensor.fill_(n_local)
        torch.distributed.all_gather_into_tensor(
            self._count_tensor,
            torch.tensor([n_local], dtype=torch.int64, device=device),
            group=group,
        )
        return [int(v) for v in self._count_tensor.view(-1).cpu().tolist()]

    @staticmethod
    def _mask_for_peer(
        weights: torch.Tensor, indices: torch.Tensor, dst: int, experts_per_rank: int
    ):
        """Rows this destination owns; everything else becomes weight 0 / id -1."""
        destination = torch.div(indices, experts_per_rank, rounding_mode="floor")
        owned = (destination == dst) & (indices >= 0)
        w = torch.where(owned, weights, torch.zeros_like(weights))
        i = torch.where(owned, indices, torch.full_like(indices, -1))
        return w, i

    def _pack_dispatch(
        self, x, weights, indices, hidden, scale_cols, topk, payload_cols, world
    ) -> torch.Tensor:
        from flashinfer import mxfp8_quantize

        n_local = int(x.size(0))
        experts_per_rank = self.cfg.n_local_experts
        x_fp8 = x.new_empty((0,), dtype=torch.uint8)
        x_scale = x.new_empty((0,), dtype=torch.uint8)
        if n_local:
            x_fp8, x_scale = mxfp8_quantize(x.contiguous(), is_sf_swizzled_layout=False)
            x_fp8 = x_fp8.view(torch.uint8).reshape(n_local, hidden)
            x_scale = x_scale.view(torch.uint8).reshape(n_local, scale_cols)

        scale_end = hidden + scale_cols
        weight_end = scale_end + topk * _WEIGHT_BYTES
        buf = torch.empty(
            (world, n_local, payload_cols), dtype=torch.uint8, device=x.device
        )
        if n_local:
            buf[:, :, :hidden].copy_(x_fp8.unsqueeze(0))
            buf[:, :, hidden:scale_end].copy_(x_scale.unsqueeze(0))
            for dst in range(world):
                w, i = self._mask_for_peer(weights, indices, dst, experts_per_rank)
                buf[dst, :, scale_end:weight_end].view(torch.float32).copy_(w)
                buf[dst, :, weight_end:].view(torch.int32).copy_(i.to(torch.int32))
        return buf.view(-1, payload_cols)

    def _compute_local(self, recv, counts, hidden, scale_cols, topk, payload_cols):
        """Dequantize the received activations, mask/remap ids, run local experts.

        Returns FP32 partials for ALL owners, concatenated in owner order — the
        order the return splits expect.
        """
        from flashinfer import mxfp8_quantize  # noqa: F401  (availability check)

        n_total = sum(counts)
        if n_total == 0:
            return torch.zeros((0, hidden), dtype=torch.float32, device=recv.device)

        scale_end = hidden + scale_cols
        weight_end = scale_end + topk * _WEIGHT_BYTES

        x_q = recv[:, :hidden].contiguous()
        x_s = recv[:, hidden:scale_end].contiguous()
        w = recv[:, scale_end:weight_end].view(torch.float32).contiguous()
        ids = recv[:, weight_end:].view(torch.int32).contiguous().to(torch.int64)

        x = self._dequant_mxfp8(x_q, x_s, hidden)

        # The sender masked by destination, so every surviving id belongs to this
        # rank. Re-check rather than trust: a routing bug upstream would otherwise
        # silently index the wrong expert.
        local_begin = self.cfg.local_expert_start
        local_end = self.cfg.local_expert_end
        valid = (w != 0) & (ids >= local_begin) & (ids < local_end)
        ids = torch.where(valid, ids - local_begin, torch.full_like(ids, -1))
        w = torch.where(valid, w, torch.zeros_like(w))

        with record_function_range("dsv4.moe.a2a.local_experts"):
            partial = self._local(x, w, ids)
        return partial.to(torch.float32)

    @staticmethod
    def _dequant_mxfp8(
        x_q: torch.Tensor, x_s: torch.Tensor, hidden: int
    ) -> torch.Tensor:
        """Block32-MXFP8 -> BF16. The UE8M0 byte is a biased exponent."""
        n = x_q.size(0)
        q = x_q.view(torch.float8_e4m3fn).to(torch.float32)
        scale = torch.exp2(x_s.to(torch.float32) - 127.0).repeat_interleave(
            SCALE_BLOCK, dim=1
        )
        return (q * scale[:, :hidden]).to(torch.bfloat16)

    def _pack_return(
        self, local_out: torch.Tensor, hidden: int, scale_cols: int, n_total: int
    ) -> torch.Tensor:
        from flashinfer import mxfp8_quantize

        ret_cols = hidden + scale_cols
        if n_total == 0:
            return torch.empty(
                (0, ret_cols), dtype=torch.uint8, device=local_out.device
            )
        # Round the partial through BF16 before quantization.
        rounded = local_out.to(torch.bfloat16)
        q, s = mxfp8_quantize(rounded.contiguous(), is_sf_swizzled_layout=False)
        q = q.view(torch.uint8).reshape(n_total, hidden)
        s = s.view(torch.uint8).reshape(n_total, scale_cols)
        buf = torch.empty(
            (n_total, ret_cols), dtype=torch.uint8, device=local_out.device
        )
        buf[:, :hidden].copy_(q)
        buf[:, hidden:].copy_(s)
        return buf

    @staticmethod
    def _all_to_all(
        send: torch.Tensor, send_splits, recv_splits, cols: int, group
    ) -> torch.Tensor:
        """Exchange 2-D byte rows: split sizes count rows, never flattened elements."""
        if send.dim() != 2 or send.size(1) != cols:
            raise ValueError(
                "%s._all_to_all expects a [rows, %d] payload, got %s"
                % (BACKEND_NAME, cols, tuple(send.shape))
            )
        if sum(send_splits) != send.size(0):
            raise ValueError(
                "send splits %r sum to %d but the payload has %d rows"
                % (list(send_splits), sum(send_splits), send.size(0))
            )
        recv = torch.empty(
            (sum(recv_splits), cols), dtype=torch.uint8, device=send.device
        )
        torch.distributed.all_to_all_single(
            recv,
            send.contiguous(),
            output_split_sizes=list(recv_splits),
            input_split_sizes=list(send_splits),
            group=group,
        )
        return recv


__all__ = ["NcclEpMxfp8Strategy", "BACKEND_NAME"]
