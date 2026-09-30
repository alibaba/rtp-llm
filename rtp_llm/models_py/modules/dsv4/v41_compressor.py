"""Reference-ordered ratio-2 pooling with deferred native FP32 state writes.

Projection, CP predecessor exchange and request-generation admission belong to
the caller. The state is the native csa_state value/score ring, without a second
hidden history store. Prepare materializes old state before any commit writes.
"""

from dataclasses import dataclass

import torch

from rtp_llm.models.dsv41_kv_cache import dsv41_page_layouts


@dataclass(frozen=True)
class V41PairState:
    raw: torch.Tensor
    gen_num_per_cycle: int = 1

    def __post_init__(self):
        width = self.entries * 1024
        if (
            self.raw.ndim != 2
            or self.raw.dtype != torch.float32
            or self.raw.shape[1] != width
            or self.raw.stride(1) != 1
            or self.raw.stride(0) < width
        ):
            raise ValueError(
                "V4.1 pair state requires native FP32 [pages, entries*1024]"
            )

    @property
    def entries(self):
        return dsv41_page_layouts(self.gen_num_per_cycle)["csa_state"].entries

    def _slots(self, slots, backend):
        if backend not in ("eager", "triton"):
            raise ValueError("V4.1 pair-state backend must be eager or triton")
        if (
            slots.ndim != 1
            or slots.dtype not in (torch.int32, torch.int64)
            or slots.device != self.raw.device
            or not slots.is_contiguous()
        ):
            raise ValueError("V4.1 state slots must be contiguous device integers")
        if (
            slots.numel()
            and slots.untyped_storage().data_ptr()
            == self.raw.untyped_storage().data_ptr()
        ):
            raise ValueError("V4.1 state slots must not alias state storage")
        if backend == "eager":
            if bool(((slots < -1) | (slots >= self.raw.shape[0] * self.entries)).any()):
                raise ValueError("V4.1 state slot outside allocation")
        elif not self.raw.is_cuda:
            raise RuntimeError("V4.1 pair-state Triton operations require CUDA")

    def read(self, slots, *, backend="triton"):
        self._slots(slots, backend)
        out = torch.zeros(
            (slots.numel(), 1024), dtype=torch.float32, device=self.raw.device
        )
        if backend == "eager":
            active = slots >= 0
            selected = slots[active].long()
            out[active] = self.raw[
                selected[:, None] // self.entries,
                (selected[:, None] % self.entries) * 1024
                + torch.arange(1024, device=slots.device),
            ]
        elif slots.numel():
            from rtp_llm.models_py.modules.dsv4.fp8._v41_compressor_state_triton import (
                transfer,
            )

            transfer(self, slots, out, write=False)
        return out

    def write(self, slots, values, *, backend="triton"):
        self._slots(slots, backend)
        if (
            values.shape != (slots.numel(), 1024)
            or values.dtype != torch.float32
            or values.device != self.raw.device
            or not values.is_contiguous()
        ):
            raise ValueError("V4.1 state writes require contiguous device FP32 rows")
        if (
            values.numel()
            and values.untyped_storage().data_ptr()
            == self.raw.untyped_storage().data_ptr()
        ):
            raise ValueError("V4.1 pending state must not alias the destination")
        if backend == "eager":
            active = slots >= 0
            selected = slots[active].long()
            if selected.unique().numel() != selected.numel():
                raise ValueError("V4.1 state writes must have unique slots")
            self.raw[
                selected[:, None] // self.entries,
                (selected[:, None] % self.entries) * 1024
                + torch.arange(1024, device=slots.device),
            ] = values[active]
        elif slots.numel():
            from rtp_llm.models_py.modules.dsv4.fp8._v41_compressor_state_triton import (
                transfer,
            )

            transfer(self, slots, values, write=True)


@dataclass(frozen=True)
class V41PairMetadata:
    request_ids: torch.Tensor
    positions: torch.Tensor
    valid_rows: torch.Tensor
    previous_rows: torch.Tensor
    previous_slots: torch.Tensor
    write_slots: torch.Tensor


@dataclass(frozen=True)
class V41CompressionFrame:
    latent: torch.Tensor
    complete: torch.Tensor
    state: V41PairState
    pending_rows: torch.Tensor
    write_slots: torch.Tensor
    backend: str
    stream_id: int | None

    def commit(self):
        """Call after the corresponding GLOBAL/index stores have been enqueued."""
        if (
            self.stream_id is not None
            and torch.cuda.current_stream(self.state.raw.device).cuda_stream
            != self.stream_id
        ):
            raise RuntimeError("V4.1 state commit must use its preparation stream")
        self.state.write(self.write_slots, self.pending_rows, backend=self.backend)


def prepare_ratio2(values, scores, norm_weight, state, metadata, *, backend="triton"):
    """Return pre-RoPE BF16 latents and a separate pending state commit.

    Complete odd positions use either a same-request current-batch predecessor or
    an admitted old-state slot. Non-complete and dead rows return zero. Write slots
    are a unique retained suffix chosen by the caller; -1 means no state write.
    State reads and commit writes are ordered on the same CUDA stream.
    """
    if backend not in ("triton", "eager"):
        raise ValueError("V4.1 compressor backend must be triton or eager")
    if values.ndim != 2 or values.shape[1] != 512 or values.shape[0] > 128:
        raise ValueError("V4.1 pooling requires at most 128 projected 512-wide rows")
    rows = values.shape[0]
    for value in (values, scores):
        if (
            value.shape != (rows, 512)
            or value.dtype != torch.float32
            or value.device != state.raw.device
            or not value.is_contiguous()
        ):
            raise ValueError("V4.1 ratio-2 projections must be contiguous device FP32")
    if (
        norm_weight.shape != (512,)
        or norm_weight.dtype != torch.bfloat16
        or norm_weight.device != values.device
        or not norm_weight.is_contiguous()
    ):
        raise ValueError("V4.1 compressor norm must be contiguous device BF16 [512]")
    for name in (
        "request_ids",
        "positions",
        "valid_rows",
        "previous_rows",
        "previous_slots",
        "write_slots",
    ):
        value = getattr(metadata, name)
        dtypes = (torch.bool,) if name == "valid_rows" else (torch.int32, torch.int64)
        if (
            value.shape != (rows,)
            or value.dtype not in dtypes
            or value.device != values.device
            or not value.is_contiguous()
        ):
            raise ValueError("V4.1 pair metadata shape/dtype/device/stride mismatch")
    if backend == "triton":
        if not values.is_cuda or torch.cuda.get_device_capability(values.device) != (
            12,
            0,
        ):
            raise RuntimeError("V4.1 pooling requires an exact SM120 device")
    elif values.is_cuda and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("V4.1 eager compressor cannot be captured")
    complete = metadata.valid_rows & (metadata.positions % 2 == 1)
    write_slots = torch.where(metadata.valid_rows, metadata.write_slots, -1)
    if backend == "eager":
        _validate_pair_values(
            values, scores, norm_weight, state, metadata, complete, write_slots
        )
    pending = torch.cat((values, scores), -1)
    stream_id = (
        torch.cuda.current_stream(values.device).cuda_stream if values.is_cuda else None
    )
    if not rows:
        return V41CompressionFrame(
            torch.empty((0, 512), dtype=torch.bfloat16, device=values.device),
            complete,
            state,
            pending,
            write_slots,
            backend,
            stream_id,
        )
    from_batch = complete & (metadata.previous_rows >= 0)
    previous = state.read(
        torch.where(complete & ~from_batch, metadata.previous_slots, -1),
        backend=backend,
    )
    selected = metadata.previous_rows.long().clamp(0, rows - 1)
    prior_values = torch.where(from_batch[:, None], values[selected], previous[:, :512])
    prior_scores = torch.where(from_batch[:, None], scores[selected], previous[:, 512:])
    pair_values = torch.stack((prior_values, values), 1)
    pair_scores = torch.stack((prior_scores, scores), 1)
    pair_values = torch.where(complete[:, None, None], pair_values, 0)
    pair_scores = torch.where(complete[:, None, None], pair_scores, 0)
    pooled = (pair_values * pair_scores.softmax(1)).sum(1).bfloat16().float()
    normalized = pooled * torch.rsqrt(pooled.square().mean(-1, keepdim=True) + 1e-20)
    latent = (norm_weight * normalized).bfloat16()
    latent = torch.where(complete[:, None], latent, 0)
    return V41CompressionFrame(
        latent, complete, state, pending, write_slots, backend, stream_id
    )


def _validate_pair_values(values, scores, norm, state, meta, complete, write_slots):
    if (
        not bool(torch.isfinite(values[meta.valid_rows]).all())
        or not bool(torch.isfinite(scores[meta.valid_rows]).all())
        or not bool(torch.isfinite(norm).all())
    ):
        raise ValueError("V4.1 compressor live inputs must be finite")
    if bool(
        (
            (meta.positions[meta.valid_rows] < 0)
            | (meta.request_ids[meta.valid_rows] < 0)
        ).any()
    ):
        raise ValueError("V4.1 live pair identity must be nonnegative")
    state._slots(write_slots, "eager")
    writes = write_slots[write_slots >= 0]
    if writes.unique().numel() != writes.numel():
        raise ValueError("V4.1 state writes must have unique slots")
    for row in range(values.shape[0]):
        if not complete[row]:
            continue
        previous = int(meta.previous_rows[row])
        slot = int(meta.previous_slots[row])
        if previous < -1 or slot < -1 or (previous >= 0) == (slot >= 0):
            raise ValueError("V4.1 completed pair requires exactly one predecessor")
        if previous >= 0:
            if (
                previous >= values.shape[0]
                or not meta.valid_rows[previous]
                or meta.request_ids[previous] != meta.request_ids[row]
                or meta.positions[previous] != meta.positions[row] - 1
            ):
                raise ValueError(
                    "V4.1 predecessor must be the same request's preceding token"
                )
        else:
            state._slots(meta.previous_slots[row : row + 1], "eager")
            if not bool(
                torch.isfinite(
                    state.read(meta.previous_slots[row : row + 1], backend="eager")
                ).all()
            ):
                raise ValueError("V4.1 predecessor state must be finite")
