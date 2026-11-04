# SM120 routed rail strategy: EP16 decode MoE over the routed rail transport (librail_routed2.so).
#
# Same contract as ``Sm120DecodeStrategy._forward_sm120_fixed_ep`` (fixed n_pad, CUDA-graph safe) except
# that dispatch is routed: each TP owner sends each of its ``cap = ceil(n_pad/tp)`` rows only to the ranks
# that host one of its topk experts, so every rank computes ``16 * cap`` gathered rows instead of
# ``16 * n_pad`` and the combine returns only those partial rows. The 4 TP mates end with the identical
# full ``y[n_pad]`` (or, with DSV4_TP_RSAG=1, each rank only its owned rows; see ``forward_owned_rows``).
#
# Enable with ``DSV4_MOE_STRATEGY=sm120_rail_routed``. Rows the transport did not fill carry topk ids
# = -1, so the masked grouped GEMM ignores them (``recompute_topk_ids_sum_expert_count`` keeps only
# ids in this rank's local expert range).

import logging
import os

import torch
import torch.distributed as dist

from rtp_llm.models_py.utils.arch import is_sm120

from .._dispatch_quant_pack_triton import (
    dispatch_payload_layout,
    quant_pack_dispatch_payload,
    view_dispatch_payload,
)
from .._rail_routed2_transport import get_routed2_transport
from .base import register_strategy
from .sm120_decode import Sm120DecodeStrategy, _stream_is_capturing

_logger = logging.getLogger(__name__)
_RR_WS = {}


@register_strategy
class Sm120RailRoutedStrategy(Sm120DecodeStrategy):
    """EP16 two-node decode strategy riding the routed rail transport."""

    name = "sm120_rail_routed"
    single_round_dispatch = True

    @classmethod
    def can_handle(cls, cfg):
        return (
            os.environ.get("DSV4_SM120_RAIL", "1") != "0"
            and cfg.ep_size > 1
            and cfg.stage_context is None
            and (not cfg.cp_enabled)
            and cfg.ep_size == 2 * int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
            and is_sm120()
        )

    def __init__(self, cfg):
        super().__init__(cfg)
        self._rr = get_routed2_transport(
            cfg, torch.device("cuda", torch.cuda.current_device())
        )

    def _routed_workspace(self, n_pad, d, topk, device):
        # Shared by every MoE layer: layers run back to back on one stream, so one set of
        # buffers per n_pad is enough (a per-layer cache costs ~1-2 GB over 43 layers x graph sizes).
        key = (n_pad, d, topk, device.index)
        ws = _RR_WS.get(key)
        if ws is not None:
            return ws
        world = self._rr.world
        cap = self._rr.cap(n_pad)
        (payload_bytes, packed, _, _, _) = dispatch_payload_layout(d, topk)
        local_payload = torch.empty(
            (n_pad, payload_bytes), dtype=torch.uint8, device=device
        )
        gathered = self._rr.gathered_view(world * cap)
        ws = {
            "cap": cap,
            "local_payload": local_payload,
            "local_views": view_dispatch_payload(local_payload, d, topk, packed),
            "gathered_views": view_dispatch_payload(gathered, d, topk, packed),
            "partial": torch.empty(
                (world * cap, d), dtype=torch.bfloat16, device=device
            ),
            "out": torch.empty((n_pad, d), dtype=torch.float32, device=device),
            "out_local": torch.empty((cap, d), dtype=torch.float32, device=device),
            "packed": packed,
        }
        _RR_WS[key] = ws
        return ws

    def forward_owned_rows(self, x, weights, indices):
        """Token-sharded decode (DSV4_TP_RSAG=1): x / weights / indices are this rank's owned rows [cap, ...] of the
        TP group's n_pad rows (distributed/tp_rsag.py). Pad rows (past the real tokens) carry routed id -1. Returns
        the routed output for the owned rows [cap, dim] fp32 (no intra-group y all-gather).
        """
        from rtp_llm.models_py.distributed import tp_rsag as _tp_rsag

        ctx = _tp_rsag.CTX
        n_pad, cap = ctx.n_pad, ctx.cap
        (n, d) = x.shape
        topk = indices.size(1)
        if n != cap or self._rr.cap(n_pad) != cap:
            raise RuntimeError(f"owned rows {n} vs cap {cap} (n_pad {n_pad})")
        ws = self._routed_workspace(n_pad, int(d), int(topk), x.device)
        n_valid = max(0, ctx.e - ctx.b)
        if "owned_payload" not in ws:
            # [tp * cap] rows so every rank's owned slice exists even when n_pad % tp != 0; the transport reads only
            # rows [own_base, own_base + t_own) of the payload it is given
            lp = torch.empty(
                (self._rr.tp * cap, ws["local_payload"].size(1)),
                dtype=torch.uint8,
                device=x.device,
            )
            own = lp[ctx.tp_rank * cap : ctx.tp_rank * cap + cap]
            ws["owned_payload"], ws["owned_views"] = lp, view_dispatch_payload(
                own, d, topk, ws["packed"]
            )
        lp = ws["owned_payload"]
        own = lp[ctx.tp_rank * cap : ctx.tp_rank * cap + cap]
        quant_pack_dispatch_payload(
            x, weights, indices, own, n_valid=n_valid, views=ws["owned_views"]
        )
        self._rr.dispatch(lp, n_pad)
        (all_x, all_scale, all_w, all_i) = ws["gathered_views"]
        self._sm120_grouped._forward_sm120_deepgemm_masked(
            all_x,
            all_w,
            all_i,
            input_scale=all_scale,
            expert_start_id=self.cfg.local_expert_start,
            out=ws["partial"],
        )
        out = ws["out_local"]
        self._rr.combine_local(ws["partial"], out, n_pad)
        return out

    def _forward_sm120_fixed_ep(self, x, weights, indices, pad_floor=None):
        self._validate_group()
        (n, d) = x.shape
        topk = indices.size(1)
        floor = (
            int(pad_floor)
            if pad_floor is not None
            else int(self.cfg.max_tokens_per_rank)
        )
        if _stream_is_capturing():
            n_pad = max(floor, n)
        else:
            extent = torch.tensor([max(floor, n)], device=x.device, dtype=torch.int64)
            dist.all_reduce(extent, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
            n_pad = int(extent.item())
        if n_pad > self._rr.n_pad_max:
            raise RuntimeError(
                f"routed rail n_pad {n_pad} exceeds slab capacity {self._rr.n_pad_max} "
                "(raise DSV4_SM120_RAIL_NPAD_MAX)"
            )
        ws = self._routed_workspace(n_pad, int(d), int(topk), x.device)
        quant_pack_dispatch_payload(
            x, weights, indices, ws["local_payload"], n_valid=n, views=ws["local_views"]
        )
        self._rr.dispatch(ws["local_payload"], n_pad)
        (all_x, all_scale, all_w, all_i) = ws["gathered_views"]
        self._sm120_grouped._forward_sm120_deepgemm_masked(
            all_x,
            all_w,
            all_i,
            input_scale=all_scale,
            expert_start_id=self.cfg.local_expert_start,
            out=ws["partial"],
        )
        self._rr.combine(ws["partial"], ws["out"], n_pad)
        return ws["out"][:n]
