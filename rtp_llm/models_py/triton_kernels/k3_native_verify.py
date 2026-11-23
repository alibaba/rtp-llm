"""Use native SM10x target verification with the shared LINEAR replay logs."""

from functools import lru_cache

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.utils.cutlass import setup_cutlass_import_path

setup_cutlass_import_path()

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from ._kda_ws.checkpoint_tensor_map import checkpoint_tensor_map
from ._kda_ws.fused_ws import SMEM_LIMIT, KdaFusedWS, shared_bytes
from .linear_replay import (
    _load_conv_replay_history,
    _replay_row,
    _replay_trap,
    _require_tensor,
)

_DIM = 128
_DTYPE = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.int32: cutlass.Int32,
}


@triton.jit
def _prepare_verify(
    Projected,
    ConvPool,
    RawLog,
    StateReadSlots,
    StateWriteSlots,
    Pending,
    Banks,
    Slot,
    Source,
    Destination,
    Accept,
    HistoryCount,
    InitKind,
    HistoryBank,
    NSLOT: tl.constexpr,
    NBLOCK: tl.constexpr,
    T: tl.constexpr,
    C: tl.constexpr,
    CONV_STRIDE: tl.constexpr,
    PROJ_STRIDE: tl.constexpr,
    CHANNEL_TILE: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    active, slot, source, destination, accepted, init = _replay_row(
        Slot, Source, Destination, Accept, HistoryCount, InitKind, row, NSLOT, NBLOCK, T
    )
    bank = tl.load(HistoryBank + row)
    if active & ((bank < 0) | (bank > 1)):
        _replay_trap()
    if tile == 0:
        tl.store(StateReadSlots + row, tl.where(active & (init != 2), source, 0))
        tl.store(StateWriteSlots + row, tl.where(active, destination, 0))
        tl.store(Pending + row, tl.where(active, accepted, 0))
        tl.store(Banks + row, tl.where(active, bank, 0))
    if active:
        channel = tile * CHANNEL_TILE + tl.arange(0, CHANNEL_TILE)
        tap = tl.arange(0, 4)
        history = _load_conv_replay_history(
            ConvPool,
            RawLog,
            channel[:, None],
            tap[None, :],
            source,
            slot,
            accepted,
            init,
            CONV_STRIDE,
            C,
            T,
            C,
            4,
        )
        # Both the committed page and the previous raw log may be overwritten.
        tl.debug_barrier()
        tl.store(
            ConvPool + destination * CONV_STRIDE + tap[None, :] * C + channel[:, None],
            history,
            (channel[:, None] < C) & (tap[None, :] < 3),
        )
        for t in tl.static_range(T):
            raw = tl.load(
                Projected + (row * T + t) * PROJ_STRIDE + channel,
                mask=channel < C,
                other=0,
            )
            tl.store(RawLog + (slot * T + t) * C + channel, raw, mask=channel < C)


def _spec(tensor):
    return tuple(tensor.shape), tensor.stride(), tensor.dtype


@lru_cache(maxsize=None)
def _compile(specs, heads, steps, device, ctas, eps, lower_bound, state_tma):
    operands = tuple(
        make_fake_tensor(_DTYPE[dtype], shape, stride, assumed_align=16)
        for shape, stride, dtype in specs
    )
    kernel = KdaFusedWS(heads, steps, ctas, state_tma=state_tma)
    scalars = tuple(map(cutlass.Float32, (lower_bound, _DIM**-0.5, eps)))
    with torch.cuda.device(device):
        return cute.compile(
            kernel,
            *operands[:12],
            *scalars,
            *operands[12:],
            make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )


class K3NativeVerify:
    """Target verify with Triton-compatible token-major K/G/U logs.

    The engine chooses the history bank. CuTe reads it and writes normalized
    keys and activated log-decays to the opposite bank. U is private to a
    request/head, and the prepass preserves the raw convolution log format.
    A narrower verify can therefore use Triton on an existing lease.
    """

    def __init__(self, decoder, forget_weight, norm):
        self.decoder = decoder
        self.forget_weight = forget_weight
        self.norm = norm
        self.lower_bound = decoder.gate_lower_bound
        self.eps = norm.eps
        self._binding = None

    def bind(self, kv_cache):
        if kv_cache is None:
            return False
        cache, base = kv_cache.linear_replay, kv_cache.kv_cache_base
        if cache is None or base is None or kv_cache.group_id < 0:
            return False
        group = kv_cache.group_id
        if (
            self._binding is not None
            and group == self.group
            and base is self.base
            and all(
                getattr(cache, name) is getattr(self.logs, name)
                for name in ("k", "u", "g", "conv_inputs")
            )
        ):
            return self._binding

        if not base.is_cuda or torch.cuda.is_current_stream_capturing():
            # Creating a TMA descriptor copies host bytes. A new pool binds
            # during eager warmup so the graph owns stable device operands.
            return False
        self.base, self.logs, self.group = base, cache, group
        self._binding = False
        slots, steps, heads, dim = cache.u.shape
        capability = torch.cuda.get_device_capability(base.device)
        state_tma = capability == (10, 3) and (heads, steps) == (48, 4)
        width = heads * _DIM
        channels = 3 * width
        if (
            capability[0] != 10
            or dim != _DIM
            or min(slots, heads) <= 0
            or not 3 <= steps <= 8
            or shared_bytes(steps, steps, state_tma=state_tma) > SMEM_LIMIT
            or self.forget_weight.shape != (width, _DIM)
            or self.lower_bound is None
            or self.lower_bound >= 0
            or self.eps <= 0
        ):
            return False
        flat = base.reshape(base.shape[0], -1)
        state = self.decoder._get_ssm_states(flat)
        conv = self.decoder._get_conv_states(flat)
        blocks = state.shape[0]
        if (
            state.shape != (blocks, heads, _DIM, _DIM)
            or state.dtype != torch.float32
            or state.stride()[1:] != (_DIM * _DIM, _DIM, 1)
            or state.stride(0) < heads * _DIM * _DIM
            or state.stride(0) % 4
            or state.data_ptr() % 16
            or conv.shape != (blocks, 3, channels)
            or conv.dtype != torch.bfloat16
            or conv.stride()[1:] != (channels, 1)
            or conv.stride(0) < 3 * channels
            or conv.stride(0) % 8
            or conv.data_ptr() % 16
        ):
            return False
        kg_shape = (2 * slots, steps, heads, _DIM)
        for log, shape, dtype in (
            (cache.k, kg_shape, torch.float32),
            (cache.g, kg_shape, torch.float32),
            (cache.u, (slots, steps, heads, _DIM), torch.float32),
            (cache.conv_inputs, (slots, steps, channels), torch.bfloat16),
        ):
            if (
                log.shape != shape
                or log.dtype != dtype
                or log.device != base.device
                or not log.is_contiguous()
                or log.data_ptr() % 16
            ):
                return False
        weight = self.decoder.conv_weights
        if weight.shape != (channels, 4) or weight.dtype not in (
            torch.bfloat16,
            torch.float32,
        ):
            return False
        self.conv_weight = weight.reshape(3, width, 4).transpose(1, 2).contiguous()
        self.a_log = self.decoder.alog.reshape(-1).float().contiguous()
        self.dt_bias = self.decoder.dt_bias.reshape(-1).float().contiguous()
        self.forget = self.forget_weight.contiguous()
        self.norm_weight = self.norm.weight.detach().float().contiguous()
        for tensor, shape, dtype in (
            (self.a_log, (heads,), torch.float32),
            (self.dt_bias, (width,), torch.float32),
            (self.forget, (width, _DIM), torch.bfloat16),
            (self.norm_weight, (_DIM,), torch.float32),
        ):
            if (
                tensor.shape != shape
                or tensor.dtype != dtype
                or tensor.device != base.device
                or tensor.data_ptr() % 16
            ):
                return False
        self.state, self.conv = state, conv
        self.slots, self.steps, self.heads = slots, steps, heads
        self.sms = torch.cuda.get_device_properties(base.device).multi_processor_count
        self.state_map = checkpoint_tensor_map(state) if state_tma else None
        self._binding = True
        return True

    def __call__(self, projected, kv_cache, replay):
        if replay is None or not self.bind(kv_cache):
            return None
        batch = replay.slot_ids.numel()
        steps, heads = self.steps, self.heads
        width = heads * _DIM
        channels = 3 * width
        rows, logical_width = batch * steps, 4 * width + _DIM + heads
        if (
            batch == 0
            or projected.ndim != 2
            or projected.shape[0] != rows
            or projected.shape[1] < logical_width
            or projected.dtype != torch.bfloat16
            or projected.device != self.state.device
        ):
            return None
        projection = projected[:, :logical_width]
        if (
            projection.stride(1) != 1
            or projection.stride(0) % 8
            or projection.data_ptr() % 16
        ):
            packed = projected.new_empty((rows, triton.cdiv(logical_width, 8) * 8))
            packed[:, :logical_width].copy_(projection)
            projection = packed[:, :logical_width]

        device = projected.device
        for name in (
            "slot_ids",
            "prev_accept_lengths",
            "history_valid_lengths",
            "init_kinds",
            "history_log_banks",
        ):
            _require_tensor(name, getattr(replay, name), (batch,), device, torch.int32)
        for name in ("state_read_block_ids", "active_block_ids"):
            tensor = getattr(replay, name)
            if (
                tensor.ndim != 2
                or tensor.shape[0] <= self.group
                or tensor.stride(-1) != 1
            ):
                raise ValueError(
                    f"LINEAR replay {name} requires contiguous rows for the layer group"
                )
            _require_tensor(
                name,
                tensor,
                (tensor.shape[0], batch),
                device,
                torch.int32,
                contiguous=False,
            )
        if replay.slot_ids.data_ptr() % 16:
            return None

        source = replay.state_read_block_ids[self.group]
        destination = replay.active_block_ids[self.group]
        prepared = source.new_empty((4, triton.cdiv(batch, 4) * 4))
        read_slots, write_slots, pending, banks = (
            prepared[:, :batch].unsqueeze(-1).unbind(0)
        )
        out = projected.new_empty((rows, width))
        operands = (
            projection[:, :channels],
            self.conv_weight,
            self.conv,
            projection[:, 4 * width : 4 * width + _DIM],
            projection[:, 4 * width + _DIM :],
            self.a_log,
            self.dt_bias,
            write_slots,
            self.state,
            out,
            projection[:, channels : 4 * width],
            self.norm_weight,
            self.forget,
            read_slots,
            self.logs.u,
            self.logs.k,
            replay.slot_ids,
            pending,
            banks,
            self.logs.g,
        )
        state_tma = self.state_map is not None
        # The pinned CuTe compiler uses a fixed host signature. Non-TMA
        # kernels specialize away the descriptor operand.
        operands += (self.state_map if state_tma else write_slots,)
        items = batch * heads
        per_sm = max(2, triton.cdiv(items, self.sms))
        ctas = triton.cdiv(items, per_sm)
        # Resolve all eager fallback conditions and compile before mutating pages.
        kernel = _compile(
            tuple(map(_spec, operands)),
            heads,
            steps,
            device,
            ctas,
            self.eps,
            self.lower_bound,
            state_tma,
        )
        channel_tile = 128
        _prepare_verify[(batch, triton.cdiv(channels, channel_tile))](
            projection,
            self.conv,
            self.logs.conv_inputs,
            read_slots,
            write_slots,
            pending,
            banks,
            replay.slot_ids,
            source,
            destination,
            replay.prev_accept_lengths,
            replay.history_valid_lengths,
            replay.init_kinds,
            replay.history_log_banks,
            self.slots,
            self.state.shape[0],
            steps,
            channels,
            self.conv.stride(0),
            projection.stride(0),
            channel_tile,
        )
        kernel(*operands[:12], self.lower_bound, _DIM**-0.5, self.eps, *operands[12:])
        return out
