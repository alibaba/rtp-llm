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
_DIAG_CT = [0]

# P0 slice 1: eager decode-round commit MoE via the fixed_ep path. 0 = off
# (legacy all_to_all for every eager call); N > 0 routes eager calls with
# n <= N rows through _forward_sm120_fixed_ep(pad_floor=N). Safe because the
# decode-round commit geometry bounds n identically on every rank (see the
# selector comment); rank-invariant padding keeps the collectives matching.
_EAGER_FIXED_EP_BOUND = int(os.environ.get("DSV4_SM120_EAGER_FIXED_EP", "0"))
# Lever 1 (Sep 1): real deep_ep intranode dispatch/combine for the eager
# prefill MoE — direct P2P kernels replace the NCCL-emulated a2a (NCCL runs
# every collective over host SHM on this box; see DSV4_NCCL_AB_20260901.md).
# 0 = off (canonical NCCL emulation).
_DEEPEP_REAL = int(os.environ.get("DSV4_DEEPEP_REAL", "0"))
_DEEPEP_REAL_MAX_TOKENS = int(os.environ.get("DSV4_DEEPEP_MAX_TOKENS", "8192"))
_DEEPEP_BUFFER = None
_DEEPEP_DISABLED = object()  # sentinel: fatal deep_ep failure -> NCCL fallback
_DIAG_DE = [0]


def _deepep_disable(reason: str):
    """Permanently fall back to the NCCL emulation after a deep_ep failure
    (a model exception otherwise kills the rank per the C++ supervisor)."""
    global _DEEPEP_BUFFER
    import sys
    import traceback
    print("[DIAGDE] DISABLING deep_ep path (%s) — NCCL fallback takes over" % reason,
          file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)
    _DEEPEP_BUFFER = _DEEPEP_DISABLED

_DIAG_ATA = [0]
_SERVE_PATH_CT = {"__total": 0}
_DIAG_FE = [0]
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

def _get_deep_ep_buffer(world: int, hidden: int, topk_pad: int):
    """Process-wide deep_ep intranode Buffer (one per engine — all 43 layer
    strategies share the WORLD EP group). Sized for
    world x max_tokens x topk x hidden bf16 + 25% routing-imbalance slack;
    the validated 24/24 run used the same magnitude (2e9 for 4x8192x8x7168)."""
    global _DEEPEP_BUFFER
    if _DEEPEP_BUFFER is None:
        import deep_ep
        num_nvl_bytes = int(world * _DEEPEP_REAL_MAX_TOKENS * topk_pad * hidden * 2 * 1.25)
        if os.environ.get("DSV4_DIAG"):
            import sys
            print("[DIAGDE] rank=%d creating deep_ep Buffer num_nvl_bytes=%d" % (
                torch.distributed.get_rank(torch.distributed.group.WORLD)
                if torch.distributed.is_initialized() else -1,
                num_nvl_bytes), file=sys.stderr, flush=True)
        _DEEPEP_BUFFER = deep_ep.Buffer(
            torch.distributed.group.WORLD, num_nvl_bytes=num_nvl_bytes)
    return _DEEPEP_BUFFER


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
            _capturing = _stream_is_capturing()
            _symbolic = isinstance(x.size(0), torch.SymInt)
            _use_fixed = not replicated_tp_tokens and (_capturing or _symbolic)
            _diag_path = "fixed_ep" if _use_fixed else ("all_to_all" if not replicated_tp_tokens else "collective")
            try:
                _diag_x0 = int(x.size(0))
            except Exception:
                _diag_x0 = -1
            if os.environ.get("DSV4_DIAG") and _DIAG_CT[0] < 40 and (
                    _diag_x0 > 4 or _diag_path != "all_to_all"):
                _DIAG_CT[0] += 1
                import sys
                print("[DIAG] rank=%d x0=%r type=%s cap=%s sym=%s path=%s x.shape=%s" % (
                    dist.get_rank(dist.group.WORLD) if dist.is_initialized() else -1,
                    x.size(0), type(x.size(0)).__name__,
                    _capturing, _symbolic, _diag_path,
                    tuple(x.shape)), file=sys.stderr, flush=True)
            if os.environ.get("DSV4_DIAG") and _capturing:
                # Capture-time ground truth: print EVERY call taken during a
                # graph capture (boot-only, ~45 lines/rank — no budget issue).
                import sys as _sys
                print("[DIAGCAP] rank=%d strat=%d x0=%r cap=%s sym=%s path=%s grouped=%r" % (
                    dist.get_rank(dist.group.WORLD) if dist.is_initialized() else -1,
                    id(self) % 100000, x.size(0), _capturing, _symbolic,
                    _diag_path, self._sm120_grouped is not None), file=_sys.stderr, flush=True)
            if os.environ.get("DSV4_DIAG") and not _capturing and not _symbolic and _diag_x0 <= 64:
                # Serve-time path distribution for small (decode-sized) calls:
                # count and dump every 5000 calls — proves eager-vs-replay mix.
                global _SERVE_PATH_CT
                try:
                    _SERVE_PATH_CT[_diag_path] += 1
                except KeyError:
                    _SERVE_PATH_CT[_diag_path] = 1
                if _SERVE_PATH_CT["__total"] % 5000 == 0:
                    import sys as _sys
                    print("[DIAGSERVE] rank=%d totals=%r" % (
                        dist.get_rank(dist.group.WORLD) if dist.is_initialized() else -1,
                        {k: v for k, v in _SERVE_PATH_CT.items() if k != "__total"}),
                        file=_sys.stderr, flush=True)
                _SERVE_PATH_CT["__total"] += 1
            if (
                _use_fixed
                and self._sm120_grouped is not None
            ):
                return self._forward_sm120_fixed_ep(x, weights, indices, pad_floor=4)
            # P0 slice 1 (Aug 31): decode-round commit forwards run eagerly and
            # took the all_to_all path (AG + 8 P2P SendRecv per layer = 12
            # collectives/layer, ~150 ms NCCL + host dispatch per commit round —
            # the dominant cost of the fused decode window). The fixed_ep path is
            # eager-safe only when the pad is RANK-INVARIANT, and eager T is NOT
            # locally known to match across ranks (DP + fake/warmup streams: a
            # rank can run a large seed/prefill commit while others run tiny
            # decode commits — a local-n guard deadlocked exactly there).
            # Decision = WORLD-MAX token count via one 1-element all_reduce per
            # MoE call (~25 us; 43/forward ~= 1 ms, trivial vs the ~150 ms it
            # saves). max_n <= bound => every rank pads to `bound`.
            _eager_fixed_bound = _EAGER_FIXED_EP_BOUND
            if (
                _eager_fixed_bound > 0
                and not _capturing
                and not _symbolic
                and not replicated_tp_tokens
                and self._sm120_grouped is not None
            ):
                _n_max_t = torch.tensor(
                    [_diag_x0 if _diag_x0 >= 0 else 0],
                    dtype=torch.int32,
                    device=x.device,
                )
                dist.all_reduce(_n_max_t, op=dist.ReduceOp.MAX, group=dist.group.WORLD)
                if int(_n_max_t.item()) <= _eager_fixed_bound:
                    return self._forward_sm120_fixed_ep(
                        x, weights, indices, pad_floor=_eager_fixed_bound
                    )
            if (
                _DEEPEP_REAL > 0
                and _DEEPEP_BUFFER is not _DEEPEP_DISABLED
                and not _capturing
                and not _symbolic
                and not replicated_tp_tokens
                and self._sm120_grouped is not None
                and 0 <= _diag_x0 <= _DEEPEP_REAL_MAX_TOKENS
            ):
                # Real deep_ep intranode kernels (prefill engine, flag ON).
                # Oversized calls fall through to the NCCL emulation below;
                # a deep_ep failure disables the path permanently (the
                # supervisor otherwise kills the rank on any exception).
                try:
                    return self._forward_sm120_deepep_real(x, weights, indices)
                except Exception:
                    _deepep_disable("dispatch/combine raised")
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
    def _forward_sm120_fixed_ep(
        self, x, weights, indices, pad_floor: int | None = None
    ) -> torch.Tensor:
        if os.environ.get("DSV4_DIAG") and _DIAG_FE[0] < 10:
            try:
                if int(x.size(0)) > 4:
                    _DIAG_FE[0] += 1
                    import sys
                    print("[DIAG6] fixed_ep rank=%d x0=%r type=%s" % (
                        torch.distributed.get_rank(torch.distributed.group.WORLD)
                        if torch.distributed.is_initialized() else -1,
                        x.size(0), type(x.size(0)).__name__), file=sys.stderr, flush=True)
            except Exception:
                pass
        dist = torch.distributed
        group = dist.group.WORLD; world = dist.get_world_size(group)
        rank = dist.get_rank(group)
        n, d = x.shape
        topk = indices.size(1)
        x_bytes = d * x.element_size()
        weight_bytes = topk * weights.element_size()
        # Pad every rank's payload to a rank-invariant token count so the
        # all_reduce shapes always match, even when some ranks execute the
        # forward during a CUDA graph capture while others run it eagerly
        # (mismatched collectives would otherwise hang or read garbage).
        # pad_floor: capture callers pass a small floor — decode graphs capture
        # with n = bs*(sp+1) rows (4 at bs=1), and padding those to
        # max_tokens_per_rank (4096) multiplied every replayed MoE into two
        # ~70/268 MB all-reduces per layer (~43 GB/step) and dominated decode
        # GPU time. Captured n is uniform across ranks, so a pad derived from
        # max(floor, n) stays rank-invariant there; the eager fallback (default
        # floor) keeps the config floor for its mixed-shape safety contract.
        floor = int(pad_floor) if pad_floor is not None else int(self.cfg.max_tokens_per_rank)
        n_pad = max(floor, n)
        local_payload = torch.cat((x.contiguous().view(torch.uint8),
            weights.contiguous().view(torch.uint8).reshape(n, weight_bytes),
            indices.to(torch.int32).contiguous().view(torch.uint8).reshape(n, topk * 4)), dim=1)
        if n < n_pad:
            local_payload = torch.cat((local_payload,
                local_payload.new_zeros(n_pad - n, local_payload.size(1))), dim=0)
        # All-gather the rank-ordered payloads directly (was: zero-slot buffer
        # + AllReduce(SUM) as a gather — the AR moves 2x the bytes of an AG and
        # needed a world-sized zero-fill + copy_ per layer).
        gathered = torch.empty((world * n_pad, local_payload.size(1)),
                               dtype=torch.uint8, device=x.device)
        dist.all_gather_into_tensor(gathered, local_payload, group=group)
        all_x = gathered[:, :x_bytes].contiguous().view(x.dtype).reshape(world * n_pad, d)
        all_w = gathered[:, x_bytes:x_bytes + weight_bytes].contiguous() \
            .view(weights.dtype).reshape(world * n_pad, topk)
        all_i = gathered[:, x_bytes + weight_bytes:].contiguous().view(torch.int32) \
            .to(torch.int64).reshape(world * n_pad, topk)
        local_i = all_i - self.cfg.local_expert_start
        valid = (local_i >= 0) & (local_i < self.cfg.n_local_experts)
        local_w = all_w * valid.to(all_w.dtype)
        local_i.clamp_(0, self.cfg.n_local_experts - 1)
        # Tile the grouped GEMM into 512-row chunks so the flashinfer workspace
        # (sized for max_tokens_per_rank up to 512) does not balloon.  With the
        # capture pad floor of 64 the loop runs ceil(world*64/512) = 1 tile for
        # decode graphs; prefill-fallback pads still tile at 512-row chunks.
        MAX_TILES = 512
        total_rows = world * n_pad
        partial = torch.empty(total_rows, d, dtype=torch.float32, device=x.device)
        for offset in range(0, total_rows, MAX_TILES):
            end = min(offset + MAX_TILES, total_rows)
            chunk_x = all_x[offset:end]
            chunk_w = all_w[offset:end]
            chunk_i = all_i[offset:end]
            cli = chunk_i - self.cfg.local_expert_start
            cv = (cli >= 0) & (cli < self.cfg.n_local_experts)
            cw = chunk_w * cv.to(chunk_w.dtype)
            cli.clamp_(0, self.cfg.n_local_experts - 1)
            partial[offset:end] = self._sm120_grouped._forward_capture_sm120(
                chunk_x, cw, cli).to(x.dtype).contiguous()
        dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=group)
        return partial.view(world, n_pad, d)[rank][:n].float()

    def _forward_sm120_deepep_real(self, x, weights, indices) -> torch.Tensor:
        """Real deep_ep intranode dispatch + grouped FP4 compute + combine.

        DSV4_DEEPEP_REAL=1 (prefill engine only): replaces the NCCL-emulated
        single-round a2a with deep_ep's intranode kernels — direct P2P
        transport (vs NCCL-over-SHM), device-side layout computation, and no
        host count exchange (the emulation's count-AG + .cpu().tolist() drain
        disappears). Mirrors the H100 path's DeepEP semantics (recv indices in
        LOCAL space with -1 pads) with the SM120 grouped-FP4 local compute.
        """
        dist = torch.distributed
        cfg = self.cfg
        world = dist.get_world_size(dist.group.WORLD)
        n, d = x.shape
        n_act = int(indices.size(-1))
        topk_pad = next(
            (k for k in _DEEPEP_SUPPORTED_TOPK if k >= n_act),
            _DEEPEP_SUPPORTED_TOPK[-1],
        )
        buf = _get_deep_ep_buffer(world, int(d), topk_pad)
        if os.environ.get("DSV4_DIAG") and _DIAG_DE[0] < 40:
            _DIAG_DE[0] += 1
            import sys
            print("[DIAGDE] rank=%d x0=%d deep_ep=real path=dispatch" % (
                dist.get_rank(dist.group.WORLD), n), file=sys.stderr, flush=True)
        # Pad topk 6 -> 8 (intranode dispatch kernel supports {2,4,8,16});
        # the -1 padding slots are dropped by the dispatch.
        indices_p, weights_p = self._pad_topk_for_deepep(indices, weights)
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            _,
        ) = buf.get_dispatch_layout(indices_p, cfg.n_routed_experts)
        recv_x, recv_topk_idx, recv_topk_weights, _per_expert, handle, _ev = buf.dispatch(
            x.contiguous(),
            None,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            is_token_in_rank,
            num_tokens_per_expert,
            indices_p,
            weights_p,
            expert_alignment=1,
        )
        M = int(recv_x.size(0))
        # recv_topk_idx is LOCAL index space ([0, n_local_experts), -1 = not
        # local) — same contract the H100 path documents. Mask + clamp so the
        # grouped GEMM sees exactly the emulation's post-recv contract.
        local_i = recv_topk_idx.to(torch.int64).contiguous()
        valid = (local_i >= 0) & (local_i < cfg.n_local_experts)
        local_w = recv_topk_weights * valid.to(recv_topk_weights.dtype)
        local_i = torch.where(valid, local_i, torch.zeros_like(local_i))
        recv_output = torch.empty((M, d), dtype=x.dtype, device=x.device)
        chunk_tokens = int(os.environ.get("DSV4_MOE_CHUNK_TOKENS", "4096"))
        for begin in range(0, M, chunk_tokens):
            end = min(begin + chunk_tokens, M)
            recv_output[begin:end] = self._sm120_grouped._forward_sm120(
                recv_x[begin:end].contiguous(), local_w[begin:end],
                local_i[begin:end]).to(x.dtype)
        del recv_x, local_w, local_i, valid
        y_combined, _, _ = buf.combine(recv_output.contiguous(), handle)
        return y_combined.to(x.dtype)

    def _forward_sm120_all_to_all(self, x, weights, indices) -> torch.Tensor:
        try:
            return self._forward_sm120_all_to_all_impl(x, weights, indices)
        except Exception:
            import sys as _sys, traceback as _tb
            try:
                def _t(v):
                    if v is None:
                        return "None"
                    try:
                        return "%s size=%s type0=%s" % (type(v).__name__, tuple(v.shape), type(v.size(0)).__name__)
                    except Exception:
                        return "%s" % type(v).__name__
                print("[DIAG4] all_to_all EXC rank=%d x=%s w=%s i=%s" % (
                    torch.distributed.get_rank(torch.distributed.group.WORLD)
                    if torch.distributed.is_initialized() else -1,
                    _t(x), _t(weights), _t(indices)), file=_sys.stderr, flush=True)
                _tb.print_exc(file=_sys.stderr)
            except Exception:
                pass
            raise

    def _forward_sm120_all_to_all_impl(self, x, weights, indices) -> torch.Tensor:
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
        # Single-round count exchange: every rank issues exactly ONE
        # count-AllGather + payload all_to_all + combine all_to_all per MoE
        # layer, whatever its local token count. The designs this replaces
        # were rank-asymmetric and deadlocked fused >= 8K ISL prefills:
        # chunking the owner's tokens multiplied its collective rounds
        # (1-token peers could not match them), and the plain
        # all_gather_into_tensor "long" branch required equal row counts on
        # every rank. Each rank now broadcasts its FULL local token count;
        # recv_counts is simply every rank's total ([1, T, 1, 1] for a
        # T-token owner with 1-placeholder peers), so the a2a split sizes
        # stay consistent on all ranks for any owner (it rotates per
        # request). The per-rank GEMM stays tiled further below, keeping
        # the flashinfer workspace bounded regardless of T.
        send_counts = [int(x.size(0))] * world
        pair = torch.tensor([send_counts[0]], dtype=torch.int64, device=x.device)
        gathered = torch.empty(world, 1, dtype=torch.int64, device=x.device)
        dist.all_gather_into_tensor(gathered, pair, group=group)
        recv_counts = [int(v) for v in gathered.view(-1).cpu().tolist()]
        try:
            _bad = (sum(recv_counts) > 65536) or any((c < 0) or (c > 65536) for c in recv_counts)
        except Exception:
            _bad = True
        if _bad:
            if os.environ.get("DSV4_DIAG"):
                import sys
                print("[DIAG5] rank=%d FALLBACK->fixed_ep counts=%r" % (
                    torch.distributed.get_rank(torch.distributed.group.WORLD)
                    if torch.distributed.is_initialized() else -1,
                    recv_counts), file=sys.stderr, flush=True)
            return self._forward_sm120_fixed_ep(x, weights, indices)
        try:
            _sz = int(x.size(0))
        except Exception:
            _sz = -1
        if os.environ.get("DSV4_DIAG") and _sz > 4 and _DIAG_ATA[0] < 400:
            _DIAG_ATA[0] += 1
            import sys
            try:
                print("[DIAG2] rank=%d x0=%r(%s) topk=%r pcols=%r counts=%r sum=%d path=single_round" % (
                    dist.get_rank(dist.group.WORLD) if dist.is_initialized() else -1,
                    x.size(0), type(x.size(0)).__name__,
                    weights.size(1), payload_cols,
                    recv_counts, sum(recv_counts)),
                    file=sys.stderr, flush=True)
            except Exception as _e3:
                print("[DIAG2] print-fail %r" % (_e3,), file=sys.stderr, flush=True)
        recv_payload = torch.empty((sum(recv_counts), payload_cols),
                                   dtype=torch.uint8, device=x.device)
        destination = torch.div(indices, experts_per_rank,
                                rounding_mode="floor").clamp_(0, world - 1)
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
        dist.all_to_all_single(recv_payload, send_payload,
            output_split_sizes=recv_counts, input_split_sizes=send_counts, group=group)
        # The dispatch payload is the largest transient (world x T rows ~0.5
        # GiB at 32K ISL) — release it before the local GEMM allocations.
        del send_payload_by_peer, send_payload
        recv_tokens = sum(recv_counts)
        recv_x = recv_payload[:, :x_end].contiguous().view(torch.float8_e4m3fn)
        recv_scale = recv_payload[:, x_end:scale_end].contiguous()
        recv_w = recv_payload[:, scale_end:weight_end].contiguous().view(torch.float32)
        recv_i = recv_payload[:, weight_end:].contiguous().view(torch.int32).to(torch.int64)
        del recv_payload  # recv_* hold contiguous copies
        local_i = recv_i - cfg.local_expert_start
        valid = (local_i >= 0) & (local_i < cfg.n_local_experts)
        local_w = recv_w * valid.to(recv_w.dtype)
        local_i = torch.where(valid, local_i, 0)
        if self._sm120_grouped is not None:
            chunk_tokens = int(os.environ.get("DSV4_MOE_CHUNK_TOKENS", "4096"))
            # Preallocate the full output instead of cat(output_parts): the
            # cat held both the parts list and the result simultaneously.
            recv_output = torch.empty((recv_tokens, int(x.size(1))),
                                      dtype=x.dtype, device=x.device)
            for begin in range(0, recv_tokens, chunk_tokens):
                end = min(begin + chunk_tokens, recv_tokens)
                recv_output[begin:end] = self._sm120_grouped._forward_sm120(
                    recv_x[begin:end], local_w[begin:end], local_i[begin:end],
                    input_scale=recv_scale[begin:end]).to(x.dtype)
            del recv_x, recv_scale, recv_w, recv_i, local_w, local_i, valid
        else:
            raise RuntimeError("SM120 MXFP8 dispatch requires grouped FP4 MoE")
        combine_q, combine_scale = mxfp8_quantize(
            recv_output.contiguous(), is_sf_swizzled_layout=False)
        combine_scale = combine_scale.reshape(recv_tokens, scale_cols)
        combine_payload = torch.cat([combine_q.view(torch.uint8), combine_scale], dim=1).contiguous()
        del recv_output, combine_q, combine_scale  # folded into combine_payload
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
