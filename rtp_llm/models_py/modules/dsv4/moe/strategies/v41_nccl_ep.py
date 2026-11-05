"""V4.1 NCCL row transport with peer-admitted FP32 or opt-in MXFP8 returns."""

import hashlib
import json
import os
from dataclasses import asdict, dataclass, replace

import torch

from rtp_llm.models_py.modules.dsv4._nvtx_phases import nvtx_phase
from rtp_llm.models_py.modules.dsv4._v41_n2_sparse import (
    _announce_n2,
    n2_sparse_dispatch_enabled,
    sparse_compaction_order,
    sparse_destination_mask,
)
from rtp_llm.models_py.modules.dsv4.moe._nccl_ep_combine_triton import fp32_peer_sum
from rtp_llm.models_py.modules.dsv4.v41_dispatch import V41DispatchLayout, pack_dispatch
from rtp_llm.models_py.modules.dsv4.v41_expert import V41_EXPERT_RECIPE
from rtp_llm.models_py.modules.dsv4.v41_return import V41ReturnLayout, pack_return

from .base import register_strategy
from .nccl_ep_mxfp8 import NcclEpMxfp8Strategy
from .v41_grouped_fp4 import V41GroupedFP4Strategy

_FUSED_TAIL_FLAG = "DSV41_EP_FUSED_COMBINE_TAIL"
_EPTAIL_ANNOUNCED = False
_EPTAIL_DEBUG_SITES: set = set()


def _eptail_debug_once(state: str) -> None:
    """One-shot-per-site [EPTAILDBG] state print (DSV41_EP_TAIL_DEBUG=1) —
    diagnostics for engagement troubleshooting; silent by default."""
    if os.environ.get("DSV41_EP_TAIL_DEBUG", "0") != "1":
        return
    site = state.split(":", 1)[0]
    if site in _EPTAIL_DEBUG_SITES:
        return
    _EPTAIL_DEBUG_SITES.add(site)
    import sys

    print(f"[EPTAILDBG] {state}", file=sys.stderr, flush=True)


def _fused_combine_tail_enabled() -> bool:
    """E8: fused EP combine tail (mxfp8 peer-sum dequant + shared add + cast
    in one kernel at moe_layer combine time). Default ON since Batch-4b
    decision 1: bitwise-equal to the two-kernel tail (identical per-element
    op order; microbench 0/26M mismatches, parity byte-exact with
    engagement proven, A/B median −12.1 ms 2/3). ``=0`` (any
    non-1 value) restores the two-kernel tail."""
    return os.environ.get(_FUSED_TAIL_FLAG, "1") == "1"


@dataclass(frozen=True)
class DeferredMXFP8PeerSum:
    """Returned MXFP8 payload whose peer-sum is deferred to the fused
    combine-tail kernel (the strategy cannot see the shared expert)."""

    payload: torch.Tensor
    n_rows: int
    world_size: int


def run_deferred_combine(deferred, shared, out_dtype, out):
    """Execute the fused tail for a DeferredMXFP8PeerSum (moe_layer side)."""
    global _EPTAIL_ANNOUNCED
    if not _EPTAIL_ANNOUNCED:
        _EPTAIL_ANNOUNCED = True
        import sys

        print(
            "[EPTAIL] fused mxfp8 peer-sum+shared epilogue engaged "
            f"(rows={int(deferred.n_rows)}, world={int(deferred.world_size)})",
            file=sys.stderr,
            flush=True,
        )
    from .._nccl_ep_mxfp8_combine import mxfp8_peer_sum_shared_epilogue

    return mxfp8_peer_sum_shared_epilogue(
        deferred.payload,
        shared,
        int(deferred.n_rows),
        int(shared.shape[1]),
        int(deferred.world_size),
        out_dtype,
        out=out,
    )


@register_strategy
class V41NcclEpStrategy(NcclEpMxfp8Strategy):
    name = "v41_nccl_ep"
    expert_recipe = V41_EXPERT_RECIPE

    @classmethod
    def can_handle(cls, cfg):
        context = cfg.stage_context
        return (
            cfg.expert_recipe == cls.expert_recipe
            and cfg.ep_size in (2, 4, 8)
            and 0 <= cfg.ep_rank < cfg.ep_size
            and cfg.n_routed_experts == cfg.n_local_experts * cfg.ep_size
            and cfg.local_expert_start == cfg.ep_rank * cfg.n_local_experts
            and cfg.local_expert_end == cfg.local_expert_start + cfg.n_local_experts
            and cfg.n_local_experts > 0
            and 0 < cfg.n_activated_experts <= cfg.n_routed_experts
            and cfg.max_tokens_per_rank > 0
            and cfg.tp_size == 1
            and cfg.dim % 128 == cfg.moe_inter_dim % 128 == 0
            and min(cfg.dim, cfg.moe_inter_dim) >= 128
            and (context is None or context.backend == "fork_nccl_mxfp8")
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability() == (12, 0)
        )

    def __init__(self, cfg):
        if not self.can_handle(cfg):
            raise ValueError("V4.1 NCCL requires a compatible SM120 expert shard")
        super().__init__(cfg)
        self.layout = V41DispatchLayout(
            cfg.dim, cfg.n_activated_experts, cfg.n_routed_experts
        )
        self._admitted_identity = None
        self._failed = False
        # Freeze the wire recipe for this strategy's lifetime. Invalid choices
        # participate in peer admission so one rank cannot strand the others.
        self._return_options = tuple(
            os.environ.get(name, "0")
            for name in ("DSV41_COMBINE_MXFP8", "DSV41_COMBINE_BF16")
        )
        self._return_options_valid = self._return_options in (("0", "0"), ("1", "0"))
        self.return_layout = V41ReturnLayout(
            cfg.dim, "mxfp8" if self._return_options == ("1", "0") else "fp32"
        )
        self._dispatch_pack_fused = (
            os.environ.get("DSV41_DISPATCH_PACK_FUSE", "0") == "1"
        )
        self._return_pack_fused = os.environ.get("DSV41_RETURN_PACK_FUSE", "0") == "1"

    def _make_local_strategy(self, local_cfg):
        # A receiver computes all peers' rows and keeps global expert IDs.
        return V41GroupedFP4Strategy(
            replace(
                local_cfg,
                local_expert_start=self.cfg.local_expert_start,
                local_expert_end=self.cfg.local_expert_end,
                max_tokens_per_rank=self.cfg.max_tokens_per_rank * self.cfg.ep_size,
            )
        )

    def _admit(self, device):
        if self._failed:
            raise RuntimeError("V4.1 EP failed; replace the worker/group before retry")
        if torch.cuda.is_current_stream_capturing():
            raise NotImplementedError(
                "V4.1 NCCL strategy currently requires eager execution"
            )
        if self._local.local is None or device != self._local.local.experts.device:
            raise RuntimeError(
                "V4.1 EP requires initialized local weights on the input device"
            )
        group, world, rank = self._stage()
        dist = torch.distributed
        if dist.get_world_size(group) != world or dist.get_rank(group) != rank:
            raise RuntimeError("V4.1 EP process group disagrees with its stage roster")
        ranks = tuple(dist.get_process_group_ranks(group))
        stream = int(torch.cuda.current_stream(device).cuda_stream)
        cfg, context = self.cfg, self.cfg.stage_context
        stage = None if context is None else (context.pp_rank, context.generation)
        options = tuple(
            os.environ.get(name, "0") == "1"
            for name in (
                "DSV4_MOE_PREQUANT_INPUT",
                "DSV4_MOE_PREQUANT_INPUT_REQUIRED",
                "DSV4_MOE_LOCAL_REPLAY",
                "DSV4_MOE_LOCAL_REPLAY_REQUIRED",
                "DSV4_NCCL_EP_MXFP8_DISPATCH_PACK",
                "DSV4_MOE_EXTENT_COUNT_FUSE",
            )
        )
        identity = (
            id(group),
            ranks,
            device,
            stream,
            id(self._local.local),
            cfg,
            self.layout,
            stage,
            options,
            self.return_layout,
            self._return_options,
        )
        if self._admitted_identity is not None:
            if self._admitted_identity != identity:
                raise RuntimeError(
                    "V4.1 EP configuration, group, weights, device or stream changed after admission"
                )
            return group, world, rank
        metadata = {
            "protocol": "v41-nccl-a2a-v2",
            "dispatch": asdict(self.layout),
            "return": self.return_layout.identity,
            "return_options": self._return_options,
            "layer": cfg.layer_id,
            "intermediate": cfg.moe_inter_dim,
            "swiglu_limit": cfg.swiglu_limit,
            "rows_per_rank": cfg.max_tokens_per_rank,
            "ranks": ranks,
            "stage": stage,
            "count_fuse": options[-1],
        }
        digest = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode("ascii")
        ).digest()
        # These switches select V4 numerical/capture routes that this strategy
        # does not implement. Check collectively before sending any payload.
        compatible = (
            dist.get_backend(group) == "nccl"
            and (context is None or ranks == context.world_ranks)
            and not any(options[:-1])
            and self._return_options_valid
        )
        signature = torch.tensor(
            [*digest, int(compatible)], dtype=torch.uint8, device=device
        ).view(1, 33)
        gathered = torch.empty((world, 33), dtype=torch.uint8, device=device)
        dist.all_gather_into_tensor(gathered, signature, group=group)
        signatures = gathered.cpu()
        if not bool((signatures[:, :32] == signatures[0, :32]).all()) or not bool(
            (signatures[:, 32] == 1).all()
        ):
            self._failed = True
            raise RuntimeError(
                "V4.1 EP peer recipe, layout, bounds or transport option mismatch "
                "(return supports FP32 or MXFP8; BF16/conflicting flags are unsupported)"
            )
        self._admitted_identity = identity
        return group, world, rank

    def _sparse_dispatch_exchange(
        self, payload, indices, rows, world, rank, group, dense_total
    ):
        """N2: send each row only to the EP ranks owning its experts.

        Returns the dense-shaped ``returned`` buffer ([world*rows, ret_cols],
        block d = rank d's partials, zero rows exactly where the dense path
        would carry a +0.0 partial), or None when the shard contract does
        not hold (caller falls back to the dense exchange). Row order per
        destination is ascending source-row order; per-row partial values
        are row-independent (the contiguous/masked routing computes each
        row's expert sum in a stable order), so outputs are bitwise
        identical to the dense dispatch.
        """

        def _dbg(stage):
            if os.environ.get("DSV41_N2_DEBUG") == "1":
                import sys as _sys

                print(f"[N2DBG] rank={rank} {stage}", file=_sys.stderr, flush=True)

        local = self._local.local
        per = self.layout.experts // world
        _dbg(f"entry rows={rows} world={world} payload={tuple(payload.shape)}")
        if (
            rows <= 0
            or world <= 1
            or self.layout.experts % world
            or getattr(local, "expert_start_id", None) != rank * per
            or indices.device != payload.device
            or indices.dtype not in (torch.int32, torch.int64)
        ):
            _dbg("contract-fallback")
            return None
        try:
            return self._sparse_exchange_body(
                payload, indices, rows, world, rank, group, dense_total, _dbg
            )
        except BaseException:
            import traceback

            traceback.print_exc()
            raise

    def _sparse_exchange_body(
        self, payload, indices, rows, world, rank, group, dense_total, _dbg
    ):
        local = self._local.local
        per = self.layout.experts // world
        contrib = sparse_destination_mask(indices, world, per)
        send_counts = contrib.sum(0, dtype=torch.int64)
        _dbg("mask-done")
        gathered = torch.empty(world * world, dtype=torch.int64, device=payload.device)
        torch.distributed.all_gather_into_tensor(gathered, send_counts, group=group)
        _dbg("count-gather-done")
        matrix = gathered.view(world, world)
        splits = torch.cat(
            [send_counts, matrix[:, rank].contiguous()]
        ).cpu()  # one host sync
        send_splits = splits[:world].tolist()
        recv_splits = splits[world:].tolist()
        _dbg(f"splits send={send_splits} recv={recv_splits}")
        order = sparse_compaction_order(contrib)
        total = sum(send_splits)
        dense_positions = order[:total]  # d-major flat index == d*rows + row
        idx_rows = dense_positions % rows
        _announce_n2(int(total), int(dense_total))
        sparse_payload = payload.index_select(0, idx_rows)
        recv_cols = self.layout.payload_bytes
        received = torch.empty(
            (sum(recv_splits), recv_cols), dtype=torch.uint8, device=payload.device
        )
        torch.distributed.all_to_all_single(
            received,
            sparse_payload,
            output_split_sizes=list(recv_splits),
            input_split_sizes=list(send_splits),
            group=group,
        )
        _dbg("dispatch-a2a-done")
        layout = "masked" if dense_total <= local.masked_token_limit else "contiguous"
        partial = local.forward_received(received, self.layout, layout=layout)
        _dbg("experts-done")
        packed = pack_return(partial, self.return_layout, fused=self._return_pack_fused)
        ret_cols = self.return_layout.payload_bytes
        dense_returned = torch.zeros(
            (world * rows, ret_cols), dtype=torch.uint8, device=payload.device
        )
        send_back = torch.empty(
            (sum(send_splits), ret_cols), dtype=torch.uint8, device=payload.device
        )
        torch.distributed.all_to_all_single(
            send_back,
            packed,
            output_split_sizes=list(send_splits),
            input_split_sizes=list(recv_splits),
            group=group,
        )
        dense_returned.index_copy_(0, dense_positions, send_back)
        _dbg("return-done")
        return dense_returned

    def _all_to_all_dense_send(self, payload, counts, cols, group):
        """E3 list-form exchange: every rank receives the full `payload`
        (uniform counts guaranteed by the caller); receives land in one
        contiguous buffer via row views."""
        from ...v41_dispatch import _announce_e3

        _announce_e3("dense-send")
        recv = torch.empty(
            (sum(counts), cols), dtype=torch.uint8, device=payload.device
        )
        outs = []
        off = 0
        for n in counts:
            outs.append(recv[off : off + n])
            off += n
        torch.distributed.all_to_all(outs, [payload] * len(counts), group=group)
        return recv

    def synchronized_chunk_extent(self, local_tokens, device):
        try:
            self._admit(device)
            return super().synchronized_chunk_extent(local_tokens, device)
        except BaseException:
            self._failed = True
            raise

    def forward(self, x, weights, indices):
        pending = self._take_pending_counts()
        try:
            group, world, rank = self._admit(x.device)
            rows = int(x.shape[0])
            with nvtx_phase("v41.moe.counts_sync"):
                counts = self._counts_for_forward(rows, group, world, x.device, pending)
            # Every rank sees the same vector, so bound failures precede payload
            # collectives on all peers, including empty ranks.
            if (
                len(counts) != world
                or counts[rank] != rows
                or any(n < 0 or n > self.cfg.max_tokens_per_rank for n in counts)
            ):
                raise ValueError(
                    "V4.1 EP received rows exceed the admitted per-rank bound"
                )
            valid = torch.ones(rows, dtype=torch.bool, device=x.device)
            with nvtx_phase("v41.moe.dispatch_prep"):
                payload = pack_dispatch(
                    x,
                    weights,
                    indices,
                    valid,
                    self.layout,
                    fused=self._dispatch_pack_fused,
                )
            from ...v41_dispatch import e3_dispatch_lean_enabled

            # N2 replaces the WHOLE exchange (dispatch + experts + return);
            # a dense `returned` must therefore SKIP the shared tail below.
            # A None result (contract mismatch fallback) drops into the
            # untouched dense path.
            returned = None
            if (
                n2_sparse_dispatch_enabled()
                and all(n == rows for n in counts)
                and not torch.cuda.is_current_stream_capturing()
            ):
                with nvtx_phase("v41.moe.exchange_sparse"):
                    returned = self._sparse_dispatch_exchange(
                        payload,
                        indices,
                        rows,
                        world,
                        rank,
                        group,
                        sum(counts),
                    )
            if returned is None:
                with nvtx_phase("v41.moe.dispatch"):
                    if e3_dispatch_lean_enabled() and all(n == rows for n in counts):
                        # E3: dense dispatch sends the SAME rows to every rank; the
                        # list-form all_to_all issues grouped NCCL sends straight from
                        # `payload` — identical wire bytes, no world-x repeat
                        # materialization (~22MB alloc+copy per layer at 4k chunks).
                        received = self._all_to_all_dense_send(
                            payload, counts, self.layout.payload_bytes, group
                        )
                    else:
                        received = self._all_to_all(
                            payload.repeat(world, 1),
                            [rows] * world,
                            counts,
                            self.layout.payload_bytes,
                            group,
                        )
                local = self._local.local
                layout = (
                    "masked"
                    if sum(counts) <= local.masked_token_limit
                    else "contiguous"
                )
                # R25/F1: offer the return layout so the local body can fuse
                # combine+pack into one pass (flag-gated; guards fall back to
                # the legacy path). A packet result skips pack_return — wire
                # bytes are bitwise-identical (synthetic compare + sha gate).
                return_hint = None
                if self.return_layout.mode == "mxfp8":
                    from ..._v41_combine_return_triton import (
                        combine_return_fused_enabled as _combret_enabled,
                    )

                    if _combret_enabled():
                        return_hint = self.return_layout
                with nvtx_phase("v41.moe.experts"):
                    partial = local.forward_received(
                        received, self.layout, layout=layout, return_layout=return_hint
                    )
                with nvtx_phase("v41.moe.return"):
                    if partial.dtype == torch.uint8:
                        # R25/F1 fused packet (the legacy partial is always FP32).
                        wire = partial
                    else:
                        wire = pack_return(
                            partial, self.return_layout, fused=self._return_pack_fused
                        )
                    returned = self._all_to_all(
                        wire,
                        counts,
                        [rows] * world,
                        self.return_layout.payload_bytes,
                        group,
                    )
            _eptail_debug_once(
                f"strategy-tail: cls={type(self).__name__} mode={self.return_layout.mode} "
                f"flag={_fused_combine_tail_enabled()} "
                f"defer={getattr(self, '_defer_tail', None)} "
                f"env={os.environ.get(_FUSED_TAIL_FLAG)}"
            )
            if self.return_layout.mode == "mxfp8":
                # Deferred tail ONLY for the moe_layer chunked path (the only
                # consumer wired for DeferredMXFP8PeerSum); _run_chunk sets
                # _defer_tail around its strategy calls. Warmup/decode/
                # single-round forwards keep the in-strategy peer sum.
                if _fused_combine_tail_enabled() and getattr(
                    self, "_defer_tail", False
                ):
                    return DeferredMXFP8PeerSum(returned, rows, world)
                from .._nccl_ep_mxfp8_combine import mxfp8_dequant_peer_sum

                return mxfp8_dequant_peer_sum(
                    returned,
                    n_rows=rows,
                    hidden_size=self.cfg.dim,
                    world_size=world,
                    out_dtype=torch.float32,
                )
            output = torch.empty(
                (rows, self.cfg.dim), dtype=torch.float32, device=x.device
            )
            return fp32_peer_sum(returned.view(torch.float32), world, output)
        except BaseException:
            self._failed = True
            raise
