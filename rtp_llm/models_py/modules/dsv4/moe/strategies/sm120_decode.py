"""Explicit SM120 DP/EP decode transport; PP uses the stage-local strategies."""

from __future__ import annotations

import logging
import os
from dataclasses import replace

import torch

from rtp_llm.models_py.utils.arch import is_sm120

from .._dispatch_quant_pack_triton import (
    dispatch_payload_layout,
    quant_pack_dispatch_payload,
    view_dispatch_payload,
)
from .._nccl_ep_combine_triton import fp32_peer_sum
from .base import MoeCfg, RoutedExpertsStrategy, register_strategy
from .sm120_decode_experts import Sm120DecodeExperts

_logger = logging.getLogger(__name__)
_A2A_PAYLOAD_TOKEN_BOUND = 65536
_A2A_OVERSIZE_WARN_CT = [0]
_A2A_OVERSIZE_WARN_MAX = 3
_EAGER_FIXED_EP_BOUND = int(os.environ.get("DSV4_SM120_EAGER_FIXED_EP", "0"))
_C2_PREPARE = int(os.environ.get("DSV4_C2_PREPARE_DISPATCH", "0"))


def _a2a_payload_bound_exceeded(recv_counts) -> bool:
    try:
        return sum(recv_counts) > _A2A_PAYLOAD_TOKEN_BOUND or any(
            (c < 0 or c > _A2A_PAYLOAD_TOKEN_BOUND for c in recv_counts)
        )
    except Exception:
        return True


def _warn_a2a_oversize(recv_counts) -> None:
    if _A2A_OVERSIZE_WARN_CT[0] >= _A2A_OVERSIZE_WARN_MAX:
        return
    _A2A_OVERSIZE_WARN_CT[0] += 1
    try:
        total = sum(recv_counts)
    except Exception:
        total = -1
    _logger.warning(
        "SM120 MoE all-to-all dispatch fell back to the fixed-EP path: recv_counts=%r (sum=%d) violates the %d-token payload bound. The fallback is correct but allocates an O(world * tokens * hidden) FP32 reduction buffer per layer and is much slower at prefill scale. On a prefill leg this means the context batch is larger than the guard allows, so cap the admission token bound (max_batch_tokens_size) to keep one batch under %d tokens. (warning %d/%d)",
        recv_counts,
        total,
        _A2A_PAYLOAD_TOKEN_BOUND,
        _A2A_PAYLOAD_TOKEN_BOUND,
        _A2A_OVERSIZE_WARN_CT[0],
        _A2A_OVERSIZE_WARN_MAX,
    )


def _stream_is_capturing() -> bool:
    try:
        import ctypes

        _rt = ctypes.CDLL("libcudart.so")
        cap = ctypes.c_int(0)
        _rt.cudaStreamIsCapturing(
            ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), ctypes.byref(cap)
        )
        return cap.value != 0
    except Exception:
        return torch.cuda.is_current_stream_capturing()


def _sm120_uses_replicated_tp_tokens(cfg: MoeCfg, world: int) -> bool:
    return cfg.tp_size > 1 and cfg.tp_size == cfg.ep_size == world


@register_strategy
class Sm120DecodeStrategy(RoutedExpertsStrategy):
    name = "sm120_decode"
    single_round_dispatch = True
    _sm120_fixed_ep_ws_cache: dict = {}

    @classmethod
    def can_handle(cls, cfg):
        return (
            cfg.ep_size > 1
            and cfg.stage_context is None
            and (not cfg.cp_enabled)
            and is_sm120()
        )

    def __init__(self, cfg):
        super().__init__(cfg)
        if not self.can_handle(cfg):
            raise RuntimeError("SM120 decode requires a non-PP DP/EP topology")
        local_cfg = replace(
            cfg,
            ep_size=1,
            ep_rank=0,
            n_routed_experts=cfg.n_local_experts,
            local_expert_start=0,
            local_expert_end=cfg.n_local_experts,
        )
        self._sm120_grouped = Sm120DecodeExperts(local_cfg)

    def setup_weights(self, layer_weights):
        self._sm120_grouped.setup_weights(layer_weights)

    def _validate_group(self):
        dist = torch.distributed
        if not dist.is_initialized():
            raise RuntimeError("SM120 decode requires an initialized EP group")
        group = dist.group.WORLD
        if (
            dist.get_world_size(group) != self.cfg.ep_size
            or dist.get_rank(group) != self.cfg.ep_rank
        ):
            raise RuntimeError(
                "SM120 decode WORLD group does not match the expert partition"
            )

    def forward(self, x, weights, indices):
        self._validate_group()
        if not is_sm120(x.device):
            raise RuntimeError("SM120 decode requires an exact SM120 device")
        dist = torch.distributed
        replicated = _sm120_uses_replicated_tp_tokens(self.cfg, dist.get_world_size())
        capturing = _stream_is_capturing()
        if not replicated and (capturing or isinstance(x.size(0), torch.SymInt)):
            return self._forward_sm120_fixed_ep(x, weights, indices, pad_floor=4)
        if replicated:
            return self._forward_sm120_collective(x, weights, indices)
        if _EAGER_FIXED_EP_BOUND > 0:
            extent = torch.tensor([int(x.size(0))], dtype=torch.int32, device=x.device)
            dist.all_reduce(extent, op=dist.ReduceOp.MAX)
            if int(extent.item()) <= _EAGER_FIXED_EP_BOUND:
                return self._forward_sm120_fixed_ep(
                    x, weights, indices, pad_floor=_EAGER_FIXED_EP_BOUND
                )
        return self._forward_sm120_all_to_all(x, weights, indices)

    def _ensure_sm120_fixed_ep_workspace(
        self, n_pad: int, world: int, d: int, topk: int, device: torch.device
    ) -> dict:
        """Reuse AG / All-to-All payload buffers so capture bakes stable pointers."""
        key = (n_pad, world, d, topk, device.index, torch.bfloat16)
        ws = getattr(self, "_sm120_fixed_ep_ws", None)
        if getattr(self, "_sm120_fixed_ep_ws_key", None) == key and ws is not None:
            return ws
        cached = Sm120DecodeStrategy._sm120_fixed_ep_ws_cache.get(key)
        if cached is not None:
            self._sm120_fixed_ep_ws = cached
            self._sm120_fixed_ep_ws_key = key
            return cached
        (payload_bytes, packed, _, _, _) = dispatch_payload_layout(d, topk)
        local_payload = torch.empty(
            (n_pad, payload_bytes), dtype=torch.uint8, device=device
        )
        gathered = torch.empty(
            (world * n_pad, payload_bytes), dtype=torch.uint8, device=device
        )
        partial = torch.empty((world * n_pad, d), dtype=torch.bfloat16, device=device)
        a2a_recv = torch.empty((world * n_pad, d), dtype=torch.bfloat16, device=device)
        local_out = torch.empty((n_pad, d), dtype=torch.float32, device=device)
        ws = {
            "packed": packed,
            "local_payload": local_payload,
            "gathered": gathered,
            "local_views": view_dispatch_payload(local_payload, d, topk, packed),
            "gathered_views": view_dispatch_payload(gathered, d, topk, packed),
            "partial": partial,
            "a2a_recv": a2a_recv,
            "local_out": local_out,
        }
        Sm120DecodeStrategy._sm120_fixed_ep_ws_cache[key] = ws
        self._sm120_fixed_ep_ws = ws
        self._sm120_fixed_ep_ws_key = key
        return ws

    def _forward_sm120_fixed_ep(
        self, x, weights, indices, pad_floor: int | None = None
    ) -> torch.Tensor:
        dist = torch.distributed
        group = dist.group.WORLD
        world = dist.get_world_size(group)
        (n, d) = x.shape
        topk = indices.size(1)
        floor = (
            int(pad_floor)
            if pad_floor is not None
            else int(self.cfg.max_tokens_per_rank)
        )
        n_pad = max(floor, n)
        ws = self._ensure_sm120_fixed_ep_workspace(
            n_pad, world, int(d), int(topk), x.device
        )
        local_payload = ws["local_payload"]
        gathered = ws["gathered"]
        quant_pack_dispatch_payload(
            x, weights, indices, local_payload, n_valid=n, views=ws["local_views"]
        )
        dist.all_gather_into_tensor(gathered, local_payload, group=group)
        (all_x, all_scale, all_w, all_i) = ws["gathered_views"]
        self._sm120_grouped._forward_sm120_deepgemm_masked(
            all_x,
            all_w,
            all_i,
            input_scale=all_scale,
            expert_start_id=self.cfg.local_expert_start,
            out=ws["partial"],
        )
        dist.all_to_all_single(ws["a2a_recv"], ws["partial"], group=group)
        fp32_peer_sum(ws["a2a_recv"], world, ws["local_out"])
        return ws["local_out"][:n]

    def _forward_sm120_all_to_all(self, x, weights, indices) -> torch.Tensor:
        try:
            return self._forward_sm120_all_to_all_impl(x, weights, indices)
        except Exception:
            _logger.exception("SM120 MoE all-to-all failed")
            raise

    def prepare_dispatch(self, x, weights, indices):
        self._validate_group()
        "Quantize, exchange counts and pack before shared-expert compute.\n\n        Only split the eager all-to-all path; preserve collective order and\n        return None for paths that require the normal forward ordering.\n        "
        if not _C2_PREPARE:
            return None
        if self._sm120_grouped is None:
            return None
        if _EAGER_FIXED_EP_BOUND > 0:
            return None
        if not (x.is_cuda and torch.cuda.get_device_capability(x.device)[0] == 12):
            return None
        if _stream_is_capturing() or isinstance(x.size(0), torch.SymInt):
            return None
        dist = torch.distributed
        if not dist.is_initialized():
            return None
        if _sm120_uses_replicated_tp_tokens(
            self.cfg, dist.get_world_size(dist.group.WORLD)
        ):
            return None
        prep = self._prepare_sm120_all_to_all(x, weights, indices)
        return prep

    def run_dispatch_prepared(self, prep: dict) -> torch.Tensor:
        """Execute the path selected by ``prepare_dispatch``."""
        mode = prep["mode"]
        if mode == "fixed_ep":
            return self._forward_sm120_fixed_ep(
                prep["x"], prep["weights"], prep["indices"], pad_floor=prep["pad_floor"]
            )
        return self._run_sm120_all_to_all_prepared(prep)

    def _forward_sm120_all_to_all_impl(self, x, weights, indices) -> torch.Tensor:
        return self.run_dispatch_prepared(
            self._prepare_sm120_all_to_all(x, weights, indices)
        )

    def _prepare_sm120_all_to_all(self, x, weights, indices) -> dict:
        dist = torch.distributed
        group = dist.group.WORLD
        world = dist.get_world_size(group)
        cfg = self.cfg
        experts_per_rank = cfg.n_routed_experts // world
        from flashinfer import mxfp8_quantize

        (x_fp8, x_scale) = mxfp8_quantize(x.contiguous(), is_sf_swizzled_layout=False)
        scale_cols = int(x.size(1)) // 32
        x_scale = x_scale.reshape(x.size(0), scale_cols)
        topk = weights.size(1)
        x_end = int(x.size(1))
        scale_end = x_end + scale_cols
        weight_end = scale_end + topk * 4
        payload_cols = int(weight_end + topk * 4)
        indices_i32 = indices.to(torch.int32)
        send_counts = [int(x.size(0))] * world
        pair = torch.tensor([send_counts[0]], dtype=torch.int64, device=x.device)
        gathered = torch.empty(world, 1, dtype=torch.int64, device=x.device)
        dist.all_gather_into_tensor(gathered, pair, group=group)
        recv_counts = [int(v) for v in gathered.view(-1).cpu().tolist()]
        try:
            _bad = _a2a_payload_bound_exceeded(recv_counts)
        except Exception:
            _bad = True
        if _bad:
            _warn_a2a_oversize(recv_counts)
            return {
                "mode": "fixed_ep",
                "x": x,
                "weights": weights,
                "indices": indices,
                "pad_floor": max(recv_counts),
            }
        destination = torch.div(
            indices, experts_per_rank, rounding_mode="floor"
        ).clamp_(0, world - 1)
        recv_payload = torch.empty(
            (sum(recv_counts), payload_cols), dtype=torch.uint8, device=x.device
        )
        send_payload_by_peer = torch.empty(
            (world, int(x.size(0)), payload_cols), dtype=torch.uint8, device=x.device
        )
        send_payload_by_peer[:, :, :x_end].copy_(x_fp8.view(torch.uint8))
        send_payload_by_peer[:, :, x_end:scale_end].copy_(x_scale)
        del x_fp8, x_scale
        send_weights = send_payload_by_peer[:, :, scale_end:weight_end].view(
            torch.float32
        )
        send_indices = send_payload_by_peer[:, :, weight_end:].view(torch.int32)
        for dst in range(world):
            owned = (destination == dst) & (indices >= 0)
            torch.where(
                owned, weights, torch.zeros_like(weights), out=send_weights[dst]
            )
            torch.where(
                owned,
                indices_i32,
                torch.full_like(indices_i32, -1),
                out=send_indices[dst],
            )
        send_payload = send_payload_by_peer.view(-1, payload_cols)
        return {
            "mode": "a2a",
            "x": x,
            "recv_counts": recv_counts,
            "send_counts": send_counts,
            "send_payload": send_payload,
            "recv_payload": recv_payload,
            "x_end": x_end,
            "scale_end": scale_end,
            "weight_end": weight_end,
            "scale_cols": scale_cols,
        }

    def _run_sm120_all_to_all_prepared(self, prep: dict) -> torch.Tensor:
        dist = torch.distributed
        group = dist.group.WORLD
        world = dist.get_world_size(group)
        cfg = self.cfg
        x = prep["x"]
        recv_counts = prep.pop("recv_counts")
        send_counts = prep.pop("send_counts")
        x_end = prep.pop("x_end")
        scale_end = prep.pop("scale_end")
        weight_end = prep.pop("weight_end")
        scale_cols = prep.pop("scale_cols")
        send_payload = prep.pop("send_payload")
        recv_payload = prep.pop("recv_payload")
        prep.clear()
        dist.all_to_all_single(
            recv_payload,
            send_payload,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=group,
        )
        del send_payload
        combine_payload = self._half_compute(
            recv_payload,
            sum(recv_counts),
            cfg,
            x,
            x_end,
            scale_end,
            weight_end,
            scale_cols,
        )
        del recv_payload
        returned_payload = torch.empty(
            (world * int(x.size(0)), int(combine_payload.size(1))),
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
        del combine_payload
        from .._nccl_ep_combine_triton import mxfp8_dequant_peer_sum

        result = mxfp8_dequant_peer_sum(
            returned_payload, x.size(0), x.size(1), world, out_dtype=x.dtype
        )
        return result

    def _half_compute(
        self, recv_buf, n_rows, cfg, x, x_end, scale_end, weight_end, scale_cols
    ):
        from flashinfer import mxfp8_quantize

        recv_x = recv_buf[:, :x_end].contiguous().view(torch.float8_e4m3fn)
        recv_scale = recv_buf[:, x_end:scale_end].contiguous()
        recv_w = recv_buf[:, scale_end:weight_end].contiguous().view(torch.float32)
        recv_i = recv_buf[:, weight_end:].contiguous().view(torch.int32).to(torch.int64)
        local_i = recv_i - cfg.local_expert_start
        valid = (local_i >= 0) & (local_i < cfg.n_local_experts)
        local_w = recv_w * valid.to(recv_w.dtype)
        local_i = torch.where(valid, local_i, 0)
        if self._sm120_grouped is None:
            raise RuntimeError("SM120 MXFP8 dispatch requires grouped FP4 MoE")
        chunk_tokens = int(os.environ.get("DSV4_MOE_CHUNK_TOKENS", "4096"))
        out = torch.empty((n_rows, int(x.size(1))), dtype=x.dtype, device=x.device)
        for begin in range(0, n_rows, chunk_tokens):
            end = min(begin + chunk_tokens, n_rows)
            out[begin:end] = self._sm120_grouped(
                recv_x[begin:end],
                local_w[begin:end],
                local_i[begin:end],
                input_scale=recv_scale[begin:end],
            ).to(x.dtype)
        del recv_x, recv_scale, recv_w, recv_i, local_w, local_i, valid
        (combine_q, combine_scale) = mxfp8_quantize(
            out.contiguous(), is_sf_swizzled_layout=False
        )
        combine_scale = combine_scale.reshape(n_rows, int(x.size(1)) // 32)
        payload = torch.cat(
            [combine_q.view(torch.uint8), combine_scale], dim=1
        ).contiguous()
        return payload

    def _forward_sm120_collective(self, x, weights, indices) -> torch.Tensor:
        dist = torch.distributed
        if not dist.is_initialized():
            raise RuntimeError("SM120 EP fallback requires torch.distributed")
        group = dist.group.WORLD
        world = dist.get_world_size(group)
        if not _sm120_uses_replicated_tp_tokens(self.cfg, world):
            raise RuntimeError("non-replicated SM120 EP must use dispatch")
        if self._sm120_grouped is None:
            raise RuntimeError("SM120 requires grouped FP8xFP4 MoE")
        local_i = indices.to(torch.int64) - self.cfg.local_expert_start
        valid = (local_i >= 0) & (local_i < self.cfg.n_local_experts)
        local_w = weights * valid.to(weights.dtype)
        local_i.clamp_(0, self.cfg.n_local_experts - 1)
        output = torch.empty_like(x, dtype=torch.float32)
        for begin in range(0, x.size(0), 4096):
            end = min(begin + 4096, x.size(0))
            partial = self._sm120_grouped(
                x[begin:end], local_w[begin:end], local_i[begin:end]
            )
            dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=group)
            output[begin:end].copy_(partial)
        return output
