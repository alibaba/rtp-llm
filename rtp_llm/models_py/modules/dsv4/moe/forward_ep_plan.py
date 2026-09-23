"""Forward-scoped physical row counts, never cached routing histograms."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import get_ident
from typing import Any, Callable, Optional, Tuple

_FLAG = "DSV4_MOE_FORWARD_COUNT_PLAN"
_CURRENT: ContextVar[Optional["ForwardEpScope"]] = ContextVar(
    "dsv4_forward_ep", default=None
)
_SLICE: ContextVar[Optional["EpSubchunk"]] = ContextVar(
    "dsv4_ep_subchunk", default=None
)


def enabled() -> bool:
    value = os.environ.get(_FLAG, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{_FLAG} must be 0 or 1, got {value!r}")
    return value == "1"


def _nat(value: int, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer, got {value!r}")
    return value


def _stage_key(ctx: Any) -> tuple:
    return (
        id(ctx.process_group),
        tuple(ctx.world_ranks),
        ctx.group_rank,
        ctx.group_size,
        ctx.pp_rank,
        ctx.generation,
    )


def _device_key(device: Any) -> tuple:
    return (device.type, device.index)


@dataclass(frozen=True)
class ForwardEpPlan:
    epoch: object
    model_id: int
    stage: tuple
    device: tuple
    physical_rows: Tuple[int, ...]
    chunk_width: int

    @property
    def extent(self) -> int:
        return max(self.physical_rows)

    def counts(
        self, start: Optional[int] = None, width: Optional[int] = None
    ) -> Tuple[int, ...]:
        if start is None:
            if width is not None:
                raise ValueError("width without a subchunk start")
            return self.physical_rows
        _nat(start, "subchunk start")
        if type(width) is not int or width <= 0 or width != self.chunk_width:
            raise ValueError("subchunk width differs from the forward schedule")
        if start % width or start >= self.extent:
            raise ValueError("subchunk start is outside the declared schedule")
        return tuple(max(0, min(width, count - start)) for count in self.physical_rows)


@dataclass(frozen=True)
class EpSubchunk:
    epoch: object
    strategy_id: int
    start: int
    width: int


class ForwardEpScope:
    """One coordinator invocation; lazily binds one immutable authoritative plan."""

    def __init__(
        self,
        model: Any,
        strategies: tuple,
        ctx: Any,
        local_rows: int,
        device: Any,
        chunk_width: int,
    ):
        self.epoch = object()
        self.model_id = id(model)
        self.strategy_ids = frozenset(id(s) for s in strategies)
        self.stage = _stage_key(ctx)
        self.device = _device_key(device)
        self.local_rows = _nat(local_rows, "physical local rows")
        self.chunk_width = chunk_width
        self.plan: Optional[ForwardEpPlan] = None
        self.active = False
        self.closed = False
        self.thread_id = get_ident()

    def _validate(self, strategy: Any, group: Any, world: int, device: Any) -> None:
        if (
            not self.active
            or self.thread_id != get_ident()
            or _CURRENT.get() is not self
        ):
            raise RuntimeError("expired or foreign-thread forward EP scope")
        if id(strategy) not in self.strategy_ids:
            raise RuntimeError("strategy does not belong to the active model forward")
        ctx = strategy.cfg.stage_context
        if (
            ctx is None
            or self.stage != _stage_key(ctx)
            or group is not ctx.process_group
            or world != ctx.group_size
            or strategy.cfg.ep_rank != ctx.group_rank
            or strategy.cfg.ep_size != world
            or self.device != _device_key(device)
        ):
            raise RuntimeError("forward EP stage/group/device identity mismatch")

    def get_counts(
        self,
        strategy: Any,
        local_rows: int,
        group: Any,
        world: int,
        device: Any,
        gather: Callable[[], list],
        *,
        full: bool = False,
    ) -> Tuple[int, ...]:
        self._validate(strategy, group, world, device)
        _nat(local_rows, "caller local rows")
        if full and _SLICE.get() is not None:
            raise RuntimeError("full extent requested from inside a subchunk")
        subchunk = None if full else _SLICE.get()
        if subchunk is not None and (
            subchunk.epoch is not self.epoch or subchunk.strategy_id != id(strategy)
        ):
            raise RuntimeError("foreign subchunk cannot consume forward EP counts")
        if self.plan is None:
            if subchunk is not None:
                raise RuntimeError("extent must establish counts before a subchunk")
            if local_rows != self.local_rows:
                raise RuntimeError("physical row count changed before first MoE layer")
            counts = tuple(gather())
            if len(counts) != world or any(type(n) is not int or n < 0 for n in counts):
                raise RuntimeError("invalid authoritative stage row counts")
            if counts[strategy.cfg.ep_rank] != self.local_rows:
                raise RuntimeError("gathered local count disagrees with this forward")
            self.plan = ForwardEpPlan(
                self.epoch,
                self.model_id,
                self.stage,
                self.device,
                counts,
                self.chunk_width,
            )
        if self.plan.epoch is not self.epoch:
            raise RuntimeError("forward EP plan epoch mismatch")
        counts = (
            self.plan.counts()
            if subchunk is None
            else self.plan.counts(subchunk.start, subchunk.width)
        )
        if counts[strategy.cfg.ep_rank] != local_rows:
            raise RuntimeError(
                "MoE physical rows do not match current forward/subchunk"
            )
        return counts

    @contextmanager
    def subchunk(self, strategy: Any, start: int, width: int, local_full_rows: int):
        ctx = strategy.cfg.stage_context
        self._validate(
            strategy, ctx.process_group, ctx.group_size, _Device(*self.device)
        )
        _nat(local_full_rows, "full local rows")
        if self.plan is None or local_full_rows != self.local_rows:
            raise RuntimeError(
                "subchunk requires this forward's established full counts"
            )
        self.plan.counts(start, width)  # Validate the schedule before payload work.
        if _SLICE.get() is not None:
            raise RuntimeError("nested MoE subchunks are not supported")
        token = _SLICE.set(EpSubchunk(self.epoch, id(strategy), start, width))
        try:
            yield
        finally:
            _SLICE.reset(token)


@dataclass(frozen=True)
class _Device:
    type: str
    index: Optional[int]


def current_scope() -> Optional[ForwardEpScope]:
    return _CURRENT.get()


@contextmanager
def activate_scope(scope: Optional[ForwardEpScope]):
    # Even an ineligible nested forward masks its caller's scope.
    token = _CURRENT.set(scope)
    slice_token = _SLICE.set(None)
    if scope is not None:
        if scope.active or scope.closed:
            _SLICE.reset(slice_token)
            _CURRENT.reset(token)
            raise RuntimeError("cannot re-enter an active or consumed forward EP scope")
        scope.active = True
    try:
        yield scope
    finally:
        if scope is not None:
            scope.active = False
            scope.closed = True
            scope.plan = None
        _SLICE.reset(slice_token)
        _CURRENT.reset(token)


def make_prefill_scope(
    model: Any,
    cp_ctx: Any,
    local_rows: int,
    device: Any,
    *,
    requested: bool,
    capturing: bool = False,
    warming: bool = False,
) -> Optional[ForwardEpScope]:
    """Choose eligibility before the layer loop; reject malformed participating topology."""
    if not requested or capturing or warming or cp_ctx is None:
        return None
    if getattr(model, "commit_only", False) or getattr(cp_ctx, "cp_size", 1) != 4:
        return None
    strategies = []
    widths = set()
    for layer in model.layers:
        ffn = getattr(layer, "ffn", None)
        strategy = getattr(ffn, "_strategy", None)
        if ffn is None or getattr(ffn, "_is_decode_role", False):
            return None
        if getattr(strategy, "name", None) != "fork_nccl_mxfp8":
            return None
        strategies.append(strategy)
        widths.add(int(ffn.max_tokens_per_rank))
    if not strategies:
        return None
    if len(widths) != 1 or min(widths) <= 0:
        raise RuntimeError("inconsistent MoE subchunk policy in one stage")
    ctx = strategies[0].cfg.stage_context
    if ctx is None:
        return None
    key = _stage_key(ctx)
    if (
        ctx.group_size != 4
        or tuple(ctx.world_ranks) != tuple(range(ctx.pp_rank * 4, ctx.pp_rank * 4 + 4))
        or ctx.pp_rank not in (0, 1)
        or cp_ctx.cp_rank != ctx.group_rank
        or cp_ctx.chunk_length != local_rows
    ):
        raise RuntimeError("CP/EP stage geometry does not match physical forward rows")
    for strategy in strategies:
        other = strategy.cfg.stage_context
        if (
            other is None
            or _stage_key(other) != key
            or strategy.cfg.ep_size != 4
            or strategy.cfg.ep_rank != ctx.group_rank
        ):
            raise RuntimeError("layers disagree on the stage-local EP roster")
    return ForwardEpScope(
        model, tuple(strategies), ctx, local_rows, device, widths.pop()
    )


def prefill_forward_scope(model: Any, cp_ctx: Any, local_rows: int, device: Any):
    """Runtime entry; no CUDA/runtime queries at all when the flag is OFF."""
    if not enabled():
        return activate_scope(None)
    import torch

    from .warmup_sync import cuda_graph_warmup_forward_enabled

    capturing = device.type == "cuda" and torch.cuda.is_current_stream_capturing()
    scope = make_prefill_scope(
        model,
        cp_ctx,
        local_rows,
        device,
        requested=True,
        capturing=capturing,
        warming=cuda_graph_warmup_forward_enabled(),
    )
    return activate_scope(scope)
