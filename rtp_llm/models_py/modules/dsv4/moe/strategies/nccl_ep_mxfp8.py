"""Stage-local MXFP8 dispatch/return with ordered FP32 peer accumulation.

Partials round through BF16 and MXFP8; this is not BF16-backend bit equivalence."""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from dataclasses import replace
from typing import Dict, Optional, Tuple

import torch

from ..._profiler import record_function_range
from .._nccl_ep_mxfp8_combine import SCALE_BLOCK, mxfp8_dequant_peer_sum
from ..forward_ep_plan import current_scope
from ..warmup_sync import cuda_graph_warmup_forward_enabled
from .base import MoeCfg, RoutedExpertsStrategy, register_strategy
from .grouped_fp4 import GroupedFP4Strategy, _has_fp8_fp4_grouped_kernel
from .local_loop import LocalLoopStrategy

BACKEND_NAME = "fork_nccl_mxfp8"

_LOGGED = False

# Dispatch bytes: FP8 D + scales D/32 + weights topk*4 + IDs topk*4.
# Return bytes: D + D/32.
_WEIGHT_BYTES = 4
_ID_BYTES = 4

# Reuse one authoritative stage count vector for extent and dispatch.
# Reuse is single-call, rank-uniform and checked against local rows/device.
_COUNT_FUSE_FLAG = "DSV4_MOE_EXTENT_COUNT_FUSE"
_PREQUANT_INPUT_FLAG = "DSV4_MOE_PREQUANT_INPUT"
# A profile that explicitly requests the experimental prequantized entry must
# reject an unavailable route rather than silently measuring BF16 fallback.
# This is intentionally separate from the default-off capability flag.
_PREQUANT_INPUT_REQUIRED_FLAG = "DSV4_MOE_PREQUANT_INPUT_REQUIRED"
_LOCAL_REPLAY_FLAG = "DSV4_MOE_LOCAL_REPLAY"
_LOCAL_REPLAY_REQUIRED_FLAG = "DSV4_MOE_LOCAL_REPLAY_REQUIRED"
# The current 32K CP4 schedule presents 1024 owner rows/rank; 512 is retained
# for the declared smaller capture bucket.  Other shapes fall back unless the
# benchmark explicitly requests REQUIRED=1.
_LOCAL_REPLAY_OWNER_ROWS = (512, 1024)
_LOCAL_REPLAY_MAX_GRAPHS_PER_LAYER = 2
# A pool is shared only by graphs with the exact same device/current-stream/
# shape key.  PyTorch permits graph-pool sharing only when graphs replay in the
# same order they were captured.  Separating the 512/1024 and stream buckets
# prevents one request shape from aliasing another bucket's live graph storage.
_LOCAL_REPLAY_CAPTURE_STREAMS: dict[tuple[int, int], torch.cuda.Stream] = {}
_LOCAL_REPLAY_POOLS: dict[tuple, object] = {}


def _count_fuse_enabled() -> bool:
    return os.environ.get(_COUNT_FUSE_FLAG, "0") == "1"


def _prequant_input_enabled() -> bool:
    return os.environ.get(_PREQUANT_INPUT_FLAG, "0") == "1"


def _prequant_input_required() -> bool:
    return os.environ.get(_PREQUANT_INPUT_REQUIRED_FLAG, "0") == "1"


def _local_replay_enabled() -> bool:
    return os.environ.get(_LOCAL_REPLAY_FLAG, "0") == "1"


def _local_replay_required() -> bool:
    return os.environ.get(_LOCAL_REPLAY_REQUIRED_FLAG, "0") == "1"


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
        # Single-use carrier set by ``synchronized_chunk_extent`` when the fuse
        # is armed: (local_rows, device, counts tuple). Never read twice.
        self._pending_counts: Optional[Tuple[int, torch.device, Tuple[int, ...]]] = None
        # Per-layer graph objects.  Graph-private temporaries share one device
        # pool because layers replay serially on the same stream; each graph
        # keeps its own stable recv staging buffer outside that shared pool.
        self._local_replay_entries: dict[tuple, dict] = {}
        self._local_replay_captures = 0
        self._local_replay_replays = 0

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
        group, world, _ = self._stage()
        self._pending_counts = None
        local_tokens = int(local_tokens)
        scope = current_scope()
        if scope is not None:
            counts = scope.get_counts(
                self,
                local_tokens,
                group,
                world,
                device,
                lambda: self._gather_counts(local_tokens, group, world, device),
                full=True,
            )
            return max(counts)
        if _count_fuse_enabled():
            counts = self._gather_counts(local_tokens, group, world, device)
            self._pending_counts = (local_tokens, device, tuple(counts))
            return max(counts)
        if (
            self._chunk_extent_tensor is None
            or self._chunk_extent_tensor.device != device
        ):
            self._chunk_extent_tensor = torch.empty(
                (1,), dtype=torch.int64, device=device
            )
        self._chunk_extent_tensor.fill_(local_tokens)
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
        # Ownership transfer BEFORE anything else can raise: the fused extent
        # record may serve at most this forward call. A forward that fails or
        # is skipped before the count decision therefore cannot leave the
        # record behind for an unrelated later forward.
        pending = self._take_pending_counts()
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
        counts = self._counts_for_forward(n_local, group, world, device, pending)
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

    def _gather_counts(self, n_local: int, group, world: int, device) -> list:
        """One stage-local all_gather of the local row count."""
        if self._count_gather is None or self._count_gather.device != device:
            self._count_gather = torch.empty((1,), dtype=torch.int64, device=device)
        if (
            self._count_tensor is None
            or self._count_tensor.device != device
            or self._count_tensor.numel() != world
        ):
            self._count_tensor = torch.empty(
                (world, 1), dtype=torch.int64, device=device
            )
        self._count_gather.fill_(int(n_local))
        torch.distributed.all_gather_into_tensor(
            self._count_tensor, self._count_gather, group=group
        )
        return [int(v) for v in self._count_tensor.view(-1).cpu().tolist()]

    def _exchange_counts(self, n_local: int, group, world: int, device) -> list:
        """One stage-local all_gather of the local row count."""
        return self._gather_counts(n_local, group, world, device)

    def _take_pending_counts(self):
        """Transfer count ownership at forward entry so skipped paths cannot retain it."""
        pending = self._pending_counts
        self._pending_counts = None
        return pending

    def _counts_for_forward(
        self, n_local: int, group, world: int, device, pending=None
    ) -> list:
        """Reuse current-call counts only when rank-uniform and matching rows/device; else gather."""
        scope = current_scope()
        if scope is not None:
            return list(
                scope.get_counts(
                    self,
                    int(n_local),
                    group,
                    world,
                    device,
                    lambda: self._gather_counts(n_local, group, world, device),
                )
            )
        if pending is not None and _count_fuse_enabled():
            rec_local, rec_device, rec_counts = pending
            if (
                rec_device == device
                and rec_local == int(n_local)
                and len(set(rec_counts)) == 1
            ):
                return list(rec_counts)
        return self._exchange_counts(n_local, group, world, device)

    def forward_subchunk_scope(self, start: int, width: int, local_full_rows: int):
        scope = current_scope()
        return (
            nullcontext()
            if scope is None
            else scope.subchunk(self, start, width, local_full_rows)
        )

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

        # Keep quantization identical; only packet copies and destination masks
        # change under this independent, default-off experimental selector.
        if os.environ.get("DSV4_NCCL_EP_MXFP8_DISPATCH_PACK", "0") == "1":
            from .._nccl_ep_mxfp8_dispatch_pack import pack_dispatch_packet

            return pack_dispatch_packet(
                x_fp8.reshape(n_local, hidden),
                x_scale.reshape(n_local, scale_cols),
                weights,
                indices,
                experts_per_rank,
                world,
            )

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
        """Run the local compute eagerly or through the default-off replay island."""
        if _local_replay_required() and not _local_replay_enabled():
            raise RuntimeError(
                "DSV4_MOE_LOCAL_REPLAY_REQUIRED=1 requires DSV4_MOE_LOCAL_REPLAY=1"
            )
        if _local_replay_enabled():
            return self._compute_local_replay(
                recv, counts, hidden, scale_cols, topk, payload_cols
            )
        return self._compute_local_body(
            recv,
            counts,
            hidden,
            scale_cols,
            topk,
            payload_cols,
            allow_prequant_capture=False,
        )

    def _local_replay_eligible(self, recv, counts) -> tuple[bool, str]:
        if not isinstance(self._local, GroupedFP4Strategy):
            return False, "local strategy is not GroupedFP4"
        if os.environ.get("DSV4_MOE_DEVICE_META", "0") != "1":
            return False, "device metadata is required"
        if not _prequant_input_enabled() or not _prequant_input_required():
            return False, "received-MXFP8 prequant REQUIRED route is required"
        if torch.cuda.is_current_stream_capturing():
            return False, "nested CUDA graph capture is forbidden"
        if cuda_graph_warmup_forward_enabled():
            return False, "framework CUDA-graph warmup is a different route"
        if recv.dtype != torch.uint8 or recv.dim() != 2 or not recv.is_contiguous():
            return False, "recv must be contiguous uint8 [rows, payload]"
        if not counts or len(set(counts)) != 1:
            return False, "only rank-uniform stage counts are capture buckets"
        if int(counts[0]) not in _LOCAL_REPLAY_OWNER_ROWS:
            return False, "owner rows are outside the 512/1024 capture buckets"
        return True, ""

    def _compute_local_replay(
        self, recv, counts, hidden, scale_cols, topk, payload_cols
    ):
        eligible, reason = self._local_replay_eligible(recv, counts)
        if not eligible:
            if _local_replay_required():
                raise RuntimeError("local replay required but unavailable: " + reason)
            return self._compute_local_body(
                recv,
                counts,
                hidden,
                scale_cols,
                topk,
                payload_cols,
                allow_prequant_capture=False,
            )

        device_index = recv.device.index
        stream_id = int(torch.cuda.current_stream(recv.device).cuda_stream)
        key = (
            device_index,
            stream_id,
            tuple(int(v) for v in counts),
            int(hidden),
            int(scale_cols),
            int(topk),
            int(payload_cols),
            tuple(recv.shape),
            recv.dtype,
        )
        entry = self._local_replay_entries.get(key)
        if entry is None:
            if len(self._local_replay_entries) >= _LOCAL_REPLAY_MAX_GRAPHS_PER_LAYER:
                if _local_replay_required():
                    raise RuntimeError("local replay graph cache bound exceeded")
                return self._compute_local_body(
                    recv,
                    counts,
                    hidden,
                    scale_cols,
                    topk,
                    payload_cols,
                    allow_prequant_capture=False,
                )
            entry = self._capture_local_replay(
                key, recv, counts, hidden, scale_cols, topk, payload_cols
            )
            self._local_replay_entries[key] = entry
            self._local_replay_captures += 1

        # Stable-address ingress copy stays eager.  Replay contains route parsing,
        # local FI64 expert compute and gather, but no collectives.  Same-stream
        # ordering makes the stable output safe for the immediately following
        # return quantize/pack before the next layer can reuse the shared pool.
        entry["recv"].copy_(recv)
        with record_function_range("dsv4.moe.a2a.local_graph_replay"):
            entry["graph"].replay()
        entry["replays"] += 1
        self._local_replay_replays += 1
        if entry["replays"] == 1:
            logging.info(
                "[DSV4 MoE] local replay engaged: owner_rows=%d recv_rows=%d "
                "payload_cols=%d stream=%d",
                int(counts[0]),
                int(recv.size(0)),
                int(payload_cols),
                stream_id,
            )
        return entry["output"]

    def _capture_local_replay(
        self, key, recv, counts, hidden, scale_cols, topk, payload_cols
    ) -> dict:
        device = recv.device
        index = int(device.index)
        stream_key = (index, int(key[1]))
        capture_stream = _LOCAL_REPLAY_CAPTURE_STREAMS.get(stream_key)
        if capture_stream is None:
            capture_stream = torch.cuda.Stream(device=device)
            _LOCAL_REPLAY_CAPTURE_STREAMS[stream_key] = capture_stream
        # Graphs from different layers but the same exact bucket share this
        # pool and execute in model layer order.  A different owner-row bucket,
        # current stream, payload shape, or dtype receives a distinct pool.
        pool = _LOCAL_REPLAY_POOLS.get(key)
        if pool is None:
            pool = torch.cuda.graph_pool_handle()
            _LOCAL_REPLAY_POOLS[key] = pool

        current = torch.cuda.current_stream(device)
        static_recv = torch.empty_like(recv)
        static_recv.copy_(recv)
        capture_stream.wait_stream(current)
        # Warm the exact body on the future capture stream.  Temporary eager
        # allocations are released before capture; provider/JIT work must happen
        # here rather than inside the graph context.
        with torch.cuda.stream(capture_stream):
            for _ in range(2):
                warm = self._compute_local_body(
                    static_recv,
                    counts,
                    hidden,
                    scale_cols,
                    topk,
                    payload_cols,
                    allow_prequant_capture=True,
                )
                del warm
        capture_stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream, pool=pool):
            output = self._compute_local_body(
                static_recv,
                counts,
                hidden,
                scale_cols,
                topk,
                payload_cols,
                allow_prequant_capture=True,
            )
        capture_stream.synchronize()
        current.wait_stream(capture_stream)
        logging.info(
            "[DSV4 MoE] captured local replay graph: owner_rows=%d recv_rows=%d "
            "payload_cols=%d device=%d stream=%d",
            int(counts[0]),
            int(recv.size(0)),
            int(payload_cols),
            index,
            key[1],
        )
        return {"graph": graph, "recv": static_recv, "output": output, "replays": 0}

    def _compute_local_body(
        self,
        recv,
        counts,
        hidden,
        scale_cols,
        topk,
        payload_cols,
        *,
        allow_prequant_capture: bool,
    ):
        """Compute received rows in return order; prequantized capture is private to local replay."""
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

        # The sender masked by destination, so every surviving id belongs to this
        # rank. Re-check rather than trust: a routing bug upstream would otherwise
        # silently index the wrong expert.
        local_begin = self.cfg.local_expert_start
        local_end = self.cfg.local_expert_end
        valid = (w != 0) & (ids >= local_begin) & (ids < local_end)
        ids = torch.where(valid, ids - local_begin, torch.full_like(ids, -1))
        w = torch.where(valid, w, torch.zeros_like(w))

        with record_function_range("dsv4.moe.a2a.local_experts"):
            # Opt-in received block32 scales match GroupedFP4 input layout.
            # Framework capture keeps BF16 fallback; only private replay may use this route.
            prequant_requested = _prequant_input_enabled()
            prequant_required = _prequant_input_required()
            if prequant_required and not prequant_requested:
                raise RuntimeError(
                    "DSV4_MOE_PREQUANT_INPUT_REQUIRED=1 requires "
                    "DSV4_MOE_PREQUANT_INPUT=1; refusing an ambiguous BF16 fallback."
                )
            if prequant_requested:
                # Keep ordinary/default execution untouched: capture and warmup
                # queries are meaningful only for a requested experimental path.
                capturing = torch.cuda.is_current_stream_capturing()
                prequant_available = isinstance(self._local, GroupedFP4Strategy) and (
                    (not capturing and not cuda_graph_warmup_forward_enabled())
                    or (allow_prequant_capture and capturing)
                )
                if prequant_available:
                    partial = self._local.forward_sm120_eager(
                        x_q.view(torch.float8_e4m3fn), w, ids, input_scale=x_s
                    )
                    return partial.to(torch.float32)
                # The ordinary opt-in retains capture/warmup BF16 fallback. The
                # standard experimental TTFT bundle sets REQUIRED=1 so it
                # rejects, rather than timing, that different route.
                if prequant_required:
                    raise RuntimeError(
                        "DSV4_MOE_PREQUANT_INPUT_REQUIRED=1 requested the "
                        "received-MXFP8 eager GroupedFP4 route, but it is unavailable "
                        "(GroupedFP4/capture/warmup fallback would change the profile)."
                    )
            x = self._dequant_mxfp8(x_q, x_s, hidden)
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
