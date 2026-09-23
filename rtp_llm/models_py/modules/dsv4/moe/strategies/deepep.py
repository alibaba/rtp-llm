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

import logging
import os
from dataclasses import replace
from typing import Dict, Tuple

import torch

_logger = logging.getLogger(__name__)

# All-to-all dispatch payload bound, in tokens. A batch whose summed recv_counts exceeds it cannot
# use the all-to-all path and falls back to fixed-EP. That fallback is correct, but it allocates an
# O(world * tokens * hidden) FP32 reduction buffer per layer, so at prefill scale it is a large
# latency cliff. Warn on fallback, rate-limited to the first few fires per process.
_A2A_PAYLOAD_TOKEN_BOUND = 65536
_A2A_OVERSIZE_WARN_CT = [0]
_A2A_OVERSIZE_WARN_MAX = 3


def _a2a_payload_bound_exceeded(recv_counts) -> bool:
    """True when a dispatch batch cannot use the all-to-all path.

    A negative count is a sentinel for a failed count exchange, so it is rejected too. Any exception
    while summing is treated as exceeded: the fixed-EP fallback is the safe branch.
    """
    try:
        return (sum(recv_counts) > _A2A_PAYLOAD_TOKEN_BOUND) or any(
            (c < 0) or (c > _A2A_PAYLOAD_TOKEN_BOUND) for c in recv_counts)
    except Exception:
        return True


def _warn_a2a_oversize(recv_counts) -> None:
    """Rate-limited, always-on warning for the fixed-EP fallback (see the bound's comment).

    Host-side logging only, so it is safe under CUDA-graph capture and costs nothing on the
    all-to-all path (it is called only from the fallback branch). Capped so a long-running prefill
    leg that repeatedly exceeds the bound cannot flood the log.
    """
    if _A2A_OVERSIZE_WARN_CT[0] >= _A2A_OVERSIZE_WARN_MAX:
        return
    _A2A_OVERSIZE_WARN_CT[0] += 1
    try:
        total = sum(recv_counts)
    except Exception:
        total = -1
    _logger.warning(
        "SM120 MoE all-to-all dispatch fell back to the fixed-EP path: recv_counts=%r (sum=%d) "
        "violates the %d-token payload bound. The fallback is correct but allocates an "
        "O(world * tokens * hidden) FP32 reduction buffer per layer and is much slower at prefill "
        "scale. On a prefill leg this means the context batch is larger than the guard allows, so "
        "cap the admission token bound (max_batch_tokens_size) to keep one batch under %d tokens. "
        "(warning %d/%d)",
        recv_counts, total, _A2A_PAYLOAD_TOKEN_BOUND, _A2A_PAYLOAD_TOKEN_BOUND,
        _A2A_OVERSIZE_WARN_CT[0], _A2A_OVERSIZE_WARN_MAX)


# Optional eager fixed-EP bound. Selection uses the world-max row count so
# every rank chooses the same path and padding. Zero disables this override.
_EAGER_FIXED_EP_BOUND = int(os.environ.get("DSV4_SM120_EAGER_FIXED_EP", "0"))
# Prepare host-synchronized dispatch before starting the shared expert, then
# overlap its compute with all-to-all. Disabled by default.
_C2_PREPARE = int(os.environ.get("DSV4_C2_PREPARE_DISPATCH", "0"))


def _stream_is_capturing() -> bool:
    # Driver-level capture check: raw C++ cudaStreamBeginCapture does not update
    # torch's per-stream capture cache, so torch.cuda.is_current_stream_capturing()
    # can return False during an active capture (mismatched collectives -> garbage).
    try:
        import ctypes
        _rt = ctypes.CDLL("libcudart.so")
        cap = ctypes.c_int(0)
        _rt.cudaStreamIsCapturing(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream), ctypes.byref(cap))
        return cap.value != 0
    except Exception:
        return torch.cuda.is_current_stream_capturing()


from .._dispatch_quant_pack_triton import (
    dispatch_payload_layout,
    quant_pack_dispatch_payload,
    view_dispatch_payload,
)
from .._nccl_ep_combine_triton import fp32_peer_sum
from .base import MoeCfg, RoutedExpertsStrategy, register_strategy
from .grouped_fp4 import GroupedFP4Strategy, _has_fp8_fp4_grouped_kernel
from .local_loop import LocalLoopStrategy


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
    # Shared across layers: decode AG/A2A payload buffers are sequential.
    _sm120_fixed_ep_ws_cache: dict = {}

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
            _capturing = _stream_is_capturing()
            _symbolic = isinstance(x.size(0), torch.SymInt)
            _use_fixed = not replicated_tp_tokens and (_capturing or _symbolic)
            try:
                local_rows = int(x.size(0))
            except Exception:
                local_rows = -1
            if (
                _use_fixed
                and self._sm120_grouped is not None
            ):
                return self._forward_sm120_fixed_ep(x, weights, indices, pad_floor=4)
            # Eager ranks can have different row counts. Select fixed-EP from
            # the world maximum and pad every rank to the same bound.
            _eager_fixed_bound = _EAGER_FIXED_EP_BOUND
            if (
                _eager_fixed_bound > 0
                and not _capturing
                and not _symbolic
                and not replicated_tp_tokens
                and self._sm120_grouped is not None
            ):
                _n_max_t = torch.tensor(
                    [local_rows if local_rows >= 0 else 0],
                    dtype=torch.int32,
                    device=x.device,
                )
                dist.all_reduce(_n_max_t, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
                if int(_n_max_t.item()) <= _eager_fixed_bound:
                    return self._forward_sm120_fixed_ep(
                        x, weights, indices, pad_floor=_eager_fixed_bound
                    )
            if (
                not _capturing
                and not _symbolic
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

    def _ensure_sm120_fixed_ep_workspace(
        self,
        n_pad: int,
        world: int,
        d: int,
        topk: int,
        device: torch.device,
    ) -> dict:
        """Reuse AG / All-to-All payload buffers so capture bakes stable pointers."""
        key = (n_pad, world, d, topk, device.index, torch.bfloat16)
        ws = getattr(self, "_sm120_fixed_ep_ws", None)
        if getattr(self, "_sm120_fixed_ep_ws_key", None) == key and ws is not None:
            return ws
        cached = DeepEPStrategy._sm120_fixed_ep_ws_cache.get(key)
        if cached is not None:
            self._sm120_fixed_ep_ws = cached
            self._sm120_fixed_ep_ws_key = key
            return cached
        payload_bytes, packed, _, _, _ = dispatch_payload_layout(d, topk)
        local_payload = torch.empty(
            (n_pad, payload_bytes), dtype=torch.uint8, device=device
        )
        gathered = torch.empty(
            (world * n_pad, payload_bytes), dtype=torch.uint8, device=device
        )
        partial = torch.empty(
            (world * n_pad, d), dtype=torch.bfloat16, device=device
        )
        a2a_recv = torch.empty(
            (world * n_pad, d), dtype=torch.bfloat16, device=device
        )
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
        DeepEPStrategy._sm120_fixed_ep_ws_cache[key] = ws
        self._sm120_fixed_ep_ws = ws
        self._sm120_fixed_ep_ws_key = key
        return ws

    def _forward_sm120_fixed_ep(
        self, x, weights, indices, pad_floor: int | None = None
    ) -> torch.Tensor:
        """SM120 decode MoE with DP: AllGather dispatch + All-to-All combine.

        Local FP8 quant is fused with pad/pack, then one AllGather moves
        ``(fp8 x, UE8M0 scale, weights, ids)``. Non-local ids become -1 inside
        ``recompute_topk_ids_sum_expert_count``. Combine All-to-Alls the
        bf16 gather outputs and fp32-sums the world shards so each DP rank
        keeps only its own token rows.
        """
        dist = torch.distributed
        group = dist.group.WORLD
        world = dist.get_world_size(group)
        n, d = x.shape
        topk = indices.size(1)
        # Pad every rank's payload to a rank-invariant token count so the
        # AllGather / All-to-All shapes always match, even when some ranks
        # execute the forward during a CUDA graph capture while others run it
        # eagerly (mismatched collectives would otherwise hang or read garbage).
        # pad_floor: capture callers pass a small floor — decode graphs capture
        # with n = bs*(sp+1) rows (4 at bs=1), and padding those to
        # max_tokens_per_rank (4096) multiplied every replayed MoE into a
        # world-sized AllGather plus a world*n_pad All-to-All per layer and
        # dominated decode GPU time. Captured n is uniform across ranks, so a
        # pad derived from max(floor, n) stays rank-invariant there; the eager
        # fallback (default floor) keeps the config floor for its mixed-shape
        # safety contract.
        floor = int(pad_floor) if pad_floor is not None else int(self.cfg.max_tokens_per_rank)
        n_pad = max(floor, n)
        ws = self._ensure_sm120_fixed_ep_workspace(
            n_pad, world, int(d), int(topk), x.device
        )
        # Quantize locally and pack ``(fp8 x, UE8M0 scale, weights, ids)``
        # plus pad rows in one Triton launch, then one AllGather.
        local_payload = ws["local_payload"]
        gathered = ws["gathered"]
        quant_pack_dispatch_payload(
            x,
            weights,
            indices,
            local_payload,
            n_valid=n,
            views=ws["local_views"],
        )
        dist.all_gather_into_tensor(gathered, local_payload, group=group)
        all_x, all_scale, all_w, all_i = ws["gathered_views"]
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
        """Quantize, exchange counts and pack before shared-expert compute.

        Only split the eager all-to-all path; preserve collective order and
        return None for paths that require the normal forward ordering.
        """
        if not _C2_PREPARE:
            return None
        if self._sm120_grouped is None:
            return None
        # Only the eager all_to_all path is split. Capture/symbolic calls go
        # to fixed_ep inside forward; replicated-TP goes to collective; a
        # positive eager bound changes the decision, so keep the stock order.
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
                self.cfg, dist.get_world_size(dist.group.WORLD)):
            return None
        prep = self._prepare_sm120_all_to_all(x, weights, indices)
        return prep

    def run_dispatch_prepared(self, prep: dict) -> torch.Tensor:
        """Execute the path selected by ``prepare_dispatch``."""
        mode = prep["mode"]
        if mode == "fixed_ep":
            return self._forward_sm120_fixed_ep(
                prep["x"], prep["weights"], prep["indices"])
        return self._run_sm120_all_to_all_prepared(prep)

    def _forward_sm120_all_to_all_impl(self, x, weights, indices) -> torch.Tensor:
        # Preparation can select fixed-EP, whose payload has no recv_counts.
        # Route by mode in both the split and normal forward paths.
        return self.run_dispatch_prepared(
            self._prepare_sm120_all_to_all(x, weights, indices))

    def _prepare_sm120_all_to_all(self, x, weights, indices) -> dict:
        dist = torch.distributed
        group = dist.group.WORLD; world = dist.get_world_size(group)
        cfg = self.cfg
        experts_per_rank = cfg.n_routed_experts // world
        from flashinfer import mxfp8_quantize
        x_fp8, x_scale = mxfp8_quantize(x.contiguous(), is_sf_swizzled_layout=False)
        scale_cols = int(x.size(1)) // 32
        x_scale = x_scale.reshape(x.size(0), scale_cols)
        topk = weights.size(1)
        x_end = int(x.size(1))
        scale_end = x_end + scale_cols
        weight_end = scale_end + topk * 4
        payload_cols = int(weight_end + topk * 4)
        indices_i32 = indices.to(torch.int32)
        # Exchange all local rows once per layer. Rank-local token chunking
        # would create unmatched collective rounds; only the GEMM may be tiled.
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
            # All ranks see the same counts and select the same fallback.
            return {"mode": "fixed_ep", "x": x, "weights": weights,
                    "indices": indices}
        # Per-peer destination rank for every token slot.
        destination = torch.div(indices, experts_per_rank,
                                rounding_mode="floor").clamp_(0, world - 1)
        recv_payload = torch.empty((sum(recv_counts), payload_cols),
                                   dtype=torch.uint8, device=x.device)
        send_payload_by_peer = torch.empty((world, int(x.size(0)), payload_cols),
                                           dtype=torch.uint8, device=x.device)
        send_payload_by_peer[:, :, :x_end].copy_(x_fp8.view(torch.uint8))
        send_payload_by_peer[:, :, x_end:scale_end].copy_(x_scale)
        del x_fp8, x_scale  # copied into the send payload; free before the a2a
        send_weights = send_payload_by_peer[:, :, scale_end:weight_end].view(torch.float32)
        send_indices = send_payload_by_peer[:, :, weight_end:].view(torch.int32)
        for dst in range(world):
            owned = (destination == dst) & (indices >= 0)
            torch.where(owned, weights, torch.zeros_like(weights), out=send_weights[dst])
            torch.where(owned, indices_i32, torch.full_like(indices_i32, -1),
                        out=send_indices[dst])
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
        group = dist.group.WORLD; world = dist.get_world_size(group)
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
        prep.clear()  # the locals above now own the refs (see the del below)
        dist.all_to_all_single(recv_payload, send_payload,
            output_split_sizes=recv_counts, input_split_sizes=send_counts, group=group)
        # The dispatch payload is the largest transient (world x T rows ~0.5
        # GiB at 32K ISL) — release it before the local GEMM allocations.
        # The by-peer base dies with its last view ref (prep was cleared).
        del send_payload
        combine_payload = self._half_compute(
            recv_payload, sum(recv_counts), cfg, x, x_end, scale_end,
            weight_end, scale_cols)
        del recv_payload
        returned_payload = torch.empty((world * int(x.size(0)), int(combine_payload.size(1))),
            dtype=torch.uint8, device=x.device)
        dist.all_to_all_single(returned_payload, combine_payload,
            output_split_sizes=send_counts, input_split_sizes=recv_counts, group=group)
        del combine_payload
        from .._nccl_ep_combine_triton import mxfp8_dequant_peer_sum
        # bf16 result (single rounding from the fp32 accumulator — the caller
        # casts to bf16 anyway): halves the largest post-combine buffer
        # (512 -> 256 MiB at 32K rows).
        result = mxfp8_dequant_peer_sum(returned_payload, x.size(0), x.size(1), world,
                                        out_dtype=x.dtype)
        return result

    def _half_compute(self, recv_buf, n_rows, cfg, x, x_end, scale_end,
                      weight_end, scale_cols):
        """Unpack a dispatch payload ([n_rows, payload_cols] uint8, peer-
        grouped), run the grouped-GEMM tiles and quantize into the combine
        payload. Main-stream only; row-identical to the pre-split body."""
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
            out[begin:end] = self._sm120_grouped._forward_sm120(
                recv_x[begin:end], local_w[begin:end], local_i[begin:end],
                input_scale=recv_scale[begin:end]).to(x.dtype)
        del recv_x, recv_scale, recv_w, recv_i, local_w, local_i, valid
        combine_q, combine_scale = mxfp8_quantize(
            out.contiguous(), is_sf_swizzled_layout=False)
        # The expert-output return payload uses per-32 MXFP8 scales.
        combine_scale = combine_scale.reshape(n_rows, int(x.size(1)) // 32)
        payload = torch.cat(
            [combine_q.view(torch.uint8), combine_scale], dim=1).contiguous()
        return payload


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
