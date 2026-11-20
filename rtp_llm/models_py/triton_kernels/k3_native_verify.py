"""Adapt K3's paged LINEAR cache to the native KDA target verify kernel.

The rolling convolution window and replay records persist across calls.
Each live request owns a writable checkpoint page and a unique record row.
Read-only checkpoint pages may be shared. The previous accepted length must
be published before the next verify. Stream ordering protects those records
from concurrent launches.
"""

import logging
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
from .linear_replay import _replay_row


@triton.jit
def _prepare_verify(
    ConvPool,
    ConvWindow,
    ConvSlots,
    StateReadSlots,
    StateWriteSlots,
    Pending,
    Slot,
    Generation,
    Source,
    Destination,
    Accept,
    HistoryCount,
    HistoryEpoch,
    InitKind,
    PoolGeneration,
    PoolEpoch,
    PoolCount,
    Error,
    NSLOT: tl.constexpr,
    NBLOCK: tl.constexpr,
    T: tl.constexpr,
    C: tl.constexpr,
    CONV_STRIDE: tl.constexpr,
    CHANNEL_TILE: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    active, slot, source, dst, accepted, init = _replay_row(
        Slot, Generation, Source, Destination, Accept, HistoryCount, HistoryEpoch,
        InitKind, PoolGeneration, PoolEpoch, PoolCount, Error, row, NSLOT, NBLOCK, T,
    )
    if tile == 0:
        tl.store(ConvSlots + row, tl.where(active, slot + 1, 0))
        tl.store(StateReadSlots + row, tl.where(active & (init != 2), source, 0))
        tl.store(StateWriteSlots + row, tl.where(active, dst, 0))
        if active:
            tl.store(Pending + slot, accepted)
    if active:
        channel = tile * CHANNEL_TILE + tl.arange(0, CHANNEL_TILE)
        live_channel = channel < C
        window_base = ConvWindow + (slot * C + channel) * (T + 2)
        destination = ConvPool + dst * CONV_STRIDE + channel
        if init == 0:
            # CuTe stored [last two history taps, T projected rows] in the
            # window on the preceding verify. Mirror the accepted prefix to
            # the physical page, including when the destination is private.
            tap = tl.arange(0, 4)
            valid = live_channel[:, None] & (tap[None, :] < 3)
            accepted_window = tl.load(
                window_base[:, None] + (accepted - 1) + tap[None, :], mask=valid, other=0
            )
            tl.store(destination[:, None] + tap[None, :] * C, accepted_window, mask=valid)
        else:
            # A new generation seeds the window from its checkpoint (or zero),
            # even if it reuses a slot that still contains another request.
            cold_tap = tl.arange(0, triton.next_power_of_2(T + 2))
            valid_cold = live_channel[:, None] & (cold_tap[None, :] < 3)
            initial_window = tl.load(
                ConvPool + source * CONV_STRIDE + channel[:, None] + cold_tap[None, :] * C,
                mask=valid_cold & (init == 1), other=0,
            )
            tl.debug_barrier()
            tl.store(
                window_base[:, None] + cold_tap[None, :], initial_window,
                mask=live_channel[:, None] & (cold_tap[None, :] < T + 2),
            )
            tl.store(
                destination[:, None] + cold_tap[None, :] * C, initial_window, mask=valid_cold
            )


class K3NativeVerify:
    """Keep RTP pages authoritative and store raw native replay records in place.

    The native convolution window follows the replay record slot. The
    physical convolution checkpoint follows the accepted SSM state page.
    The layer uses one replay format for its cache pool's lifetime.
    """

    def __init__(self, decoder, f_b, norm):
        self.conv_weight = decoder.conv_weights.contiguous()
        self.a_log = decoder.alog.reshape(-1).float().contiguous()
        self.dt_bias = decoder.dt_bias.reshape(-1).float().contiguous()
        self.f_b = f_b.contiguous()
        self.norm_weight = norm.weight.detach().float().contiguous()
        self.lower_bound = decoder.gate_lower_bound
        self.eps = norm.eps
        self.decoder = decoder
        self.base = None
        self.layer_cache = None
        self._binding_key = None
        self.conv_window = None
        self._native_replay = None

    def bind(self, kv_cache):
        base, layer_cache = kv_cache.kv_cache_base, kv_cache.linear_replay
        record = layer_cache.k
        group_id = kv_cache.group_id
        record_key = (
            record.device, record.data_ptr(), record.shape, record.stride(),
            layer_cache.slot_generations.data_ptr(),
        )
        binding_key = (group_id, base.device, base.data_ptr(), base.shape, base.stride(), record_key)
        if self._binding_key == binding_key:
            return self._native_replay is not None
        self._binding_key = None
        self._native_replay = None
        self.base, self.layer_cache = base, layer_cache
        self.conv_window = None
        slots, steps, heads, dim = record.shape
        if not supports_verify_width(steps):
            self._binding_key = binding_key
            return False
        cache = base.reshape(base.shape[0], -1)
        self.ssm = self.decoder._get_ssm_states(cache)
        self.conv = self.decoder._get_conv_states(cache)
        if self.ssm.dtype != torch.float32 or self.conv.dtype != torch.bfloat16:
            self._binding_key = binding_key
            return False
        gate_log, conv_log = layer_cache.g, layer_cache.conv_inputs
        channels, window = 3 * heads * dim, steps + 2
        storage = conv_log.untyped_storage()
        if (
            conv_log.dtype != torch.bfloat16
            or conv_log.device != self.conv.device
            or gate_log.data_ptr() != storage.data_ptr()
            or conv_log.data_ptr() != gate_log.data_ptr() + gate_log.nbytes
            or storage.nbytes() != gate_log.nbytes + conv_log.nbytes
            or slots * channels * window * conv_log.element_size() > storage.nbytes()
        ):
            self._binding_key = binding_key
            return False

        # K packs K/G and U keeps the corrections. The legacy G and ConvLog
        # share one allocation, which now holds the per-record convolution windows.
        self.conv_window = conv_log.as_strided(
            (slots, channels, window), (channels * window, window, 1), storage_offset=0
        )
        self.key_gate = layer_cache.k.view(torch.bfloat16).view(slots, heads, steps, 2 * dim)
        self.correction = layer_cache.u.view(slots, heads, steps, dim)
        self.group_id = group_id
        if self.group_id < 0:
            raise ValueError("Native K3 verify requires a LINEAR replay cache group")
        self.steps, self.heads, self.slots = steps, heads, slots
        self._native_replay = NativeKdaReplay(
            self.conv_weight, self.f_b, self.a_log, self.dt_bias, self.norm_weight,
            self.ssm, self.conv_window, self.correction, self.key_gate,
            lower_bound=self.lower_bound, eps=self.eps,
        )
        self._binding_key = binding_key
        logging.getLogger(__name__).info(
            "Native K3 KdaFusedWS verify bound: group=%d heads=%d width=%d",
            self.group_id, heads, steps,
        )
        return True

    def __call__(self, projected, kv_cache, inputs):
        if not self.bind(kv_cache):
            return None
        replay = inputs.linear_replay
        batch = replay.slot_ids.numel()
        if not batch:
            return None
        dim, steps, heads = 128, self.steps, self.heads
        channels = 3 * heads * dim
        if projected.shape[0] != batch * steps or projected.dtype != torch.bfloat16:
            raise ValueError("Native K3 verify inputs must match the fixed replay width and BF16 weights")
        if projected.stride(1) != 1 or projected.stride(0) % 8 or projected.data_ptr() % 16:
            width = projected.shape[1]
            packed = projected.new_empty((batch * steps, triton.cdiv(width, 8) * 8))
            packed[:, :width].copy_(projected)
            projected = packed[:, :width]

        source = replay.state_read_block_ids[self.group_id]
        dst = replay.active_block_ids[self.group_id]
        cache = self.layer_cache
        conv_slots = source.new_empty((batch, 1))
        state_read_slots = source.new_empty((batch, 1))
        state_write_slots = source.new_empty((batch, 1))
        pending = cache.valid_counts.new_empty(self.slots)
        channel_tile = 512
        _prepare_verify[(batch, triton.cdiv(channels, channel_tile))](
            self.conv, self.conv_window,
            conv_slots, state_read_slots, state_write_slots, pending,
            replay.slot_ids, replay.slot_generations, source, dst,
            replay.prev_accept_lengths, replay.history_valid_lengths, replay.history_epochs,
            replay.init_kinds, cache.slot_generations, cache.log_epochs, cache.valid_counts,
            cache.error_flags, self.slots, self.ssm.shape[0], steps, channels,
            self.conv.stride(0), channel_tile,
        )
        return self._native_replay(
            projected, state_read_slots, replay.slot_ids, pending,
            state_write_slots=state_write_slots, conv_slots=conv_slots,
        )


_DIM = 128
_DTYPE = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.int32: cutlass.Int32,
}


def _spec(tensor):
    return tuple(tensor.shape), tensor.stride(), tensor.dtype


@lru_cache(maxsize=None)
def _compile(
    specs, heads, tokens, device, ctas, eps, lower_bound,
    state_tma,
):
    operands = tuple(
        make_fake_tensor(_DTYPE[dtype], shape, stride, assumed_align=16)
        for shape, stride, dtype in specs
    )
    kernel = KdaFusedWS(heads, tokens, tokens + 2, ctas, state_tma=state_tma)
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


def _require(name, tensor, shape, dtype, device):
    if (
        tensor.shape != shape or tensor.dtype != dtype
        or tensor.device != device or not tensor.is_contiguous()
        or tensor.data_ptr() % 16 != 0
    ):
        raise ValueError(f"Native KDA {name} requires aligned contiguous {shape}, {dtype}, {device}")


def supports_verify_width(tokens):
    return (
        3 <= tokens <= 8
        and shared_bytes(tokens, tokens, tokens + 2, 2) <= SMEM_LIMIT
    )


class NativeKdaReplay:
    """Run native Conv → F_B → replay/verify → gated RMSNorm on SM10x.

    Args:
        state: FP32 checkpoint pages in [K,V] layout, optionally pitched.
        conv_state: BF16 [record_slots, channels, tokens + 2], rolling raw inputs.
        correction: FP32 [record_slots, heads, tokens, 128].
        key_gate: BF16 [record_slots, heads, tokens, 256], one K/G copy.
        lower_bound: Negative bound of the raw F_B projection activation.

    The projected rows pack QKV, output gate, F_A and beta. A zero destination
    pads a request. A zero source starts a live page at zero.
    Live records are addressed through record_rows; pending[record] is zero
    for the first call and 1..tokens afterwards. When a checkpoint tensor map
    is available, [K,V] uses double-page TMA with hardware swizzle.
    Without TMA, the kernel uses one linearly laid out shared checkpoint page.
    """

    def __init__(
        self, conv_weight, f_b, a_log, dt_bias, norm_weight, state, conv_state,
        correction, key_gate, *, lower_bound=-5.0, eps=1e-6,
    ):
        capability = torch.cuda.get_device_capability(state.device) if state.is_cuda else (0, 0)
        if capability[0] != 10:
            raise ValueError("Native KDA requires CUDA SM10x")
        pages, heads = state.shape[:2]
        rows, _, tokens = correction.shape[:3]
        if min(pages, heads, rows) < 1 or not supports_verify_width(tokens):
            raise ValueError("Native KDA requires live pages and a supported verify width")
        if lower_bound >= 0 or eps <= 0:
            raise ValueError("Native KDA requires a bounded negative gate and positive epsilon")

        device, dim = state.device, heads * _DIM
        page_width = heads * _DIM * _DIM
        if (
            state.shape != (pages, heads, _DIM, _DIM)
            or state.dtype != torch.float32
            or state.stride()[1:] != (_DIM * _DIM, _DIM, 1)
            or state.stride(0) < page_width
            or state.stride(0) % 4
            or state.data_ptr() % 16
        ):
            raise ValueError("Native KDA requires aligned, FP32 K-major checkpoint pages")
        # Prepare the descriptor during eager warmup before CUDA Graph capture.
        self.state_tensor_map = (
            checkpoint_tensor_map(state)
            if capability == (10, 3) and (heads, tokens) == (48, 4)
            else None
        )
        _require("conv_state", conv_state, (rows, 3 * dim, tokens + 2), torch.bfloat16, device)
        _require("correction", correction, (rows, heads, tokens, _DIM), torch.float32, device)
        _require("key_gate", key_gate, (rows, heads, tokens, 2 * _DIM), torch.bfloat16, device)
        if conv_weight.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("Native KDA convolution weights must be BF16 or FP32")
        _require("conv_weight", conv_weight, (3 * dim, 4), conv_weight.dtype, device)
        _require("f_b", f_b, (dim, _DIM), torch.bfloat16, device)
        _require("a_log", a_log, (heads,), torch.float32, device)
        _require("dt_bias", dt_bias, (dim,), torch.float32, device)
        norm = norm_weight.detach().float().contiguous()
        _require("norm_weight", norm, (_DIM,), torch.float32, device)

        self.device, self.dim, self.heads, self.tokens = device, dim, heads, tokens
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.record_slots = rows
        self.conv_weight = conv_weight.view(3, dim, 4).transpose(1, 2).contiguous()
        self.a_log, self.dt_bias, self.f_b, self.norm_weight = a_log, dt_bias, f_b, norm
        self.conv_state, self.state = conv_state, state
        self.correction, self.key_gate = correction, key_gate
        self.eps, self.lower_bound = eps, lower_bound

    def __call__(
        self, projected, state_slots, record_rows, pending, *, state_write_slots, conv_slots,
    ):
        batch = state_slots.shape[0]
        rows, dim, heads = batch * self.tokens, self.dim, self.heads
        if (
            projected.shape != (rows, 4 * dim + _DIM + heads)
            and (projected.ndim != 2 or projected.shape[0] != rows
                 or projected.shape[1] < 4 * dim + _DIM + heads)
        ):
            raise ValueError("Native KDA requires packed QKV, output gate, F_A and beta rows")
        if (
            projected.device != self.device or projected.dtype != torch.bfloat16
            or projected.stride(1) != 1 or projected.stride(0) % 8 != 0
            or projected.data_ptr() % 16 != 0
        ):
            raise ValueError("Native KDA requires 16-byte-aligned BF16 projection rows")
        _require("state_slots", state_slots, (batch, 1), torch.int32, self.device)
        _require("state_write_slots", state_write_slots, (batch, 1), torch.int32, self.device)
        _require("conv_slots", conv_slots, (batch, 1), torch.int32, self.device)
        _require("record_rows", record_rows, (batch,), torch.int32, self.device)
        _require("pending", pending, (self.record_slots,), torch.int32, self.device)
        # The kernel writes live rows in epilogue and zeros padding rows in prep.
        out = projected.new_empty((rows, dim))

        # KdaFusedWS reads the accepted count by record id from pending.
        operands = (
            projected[:, :3 * dim], self.conv_weight, self.conv_state,
            projected[:, 4 * dim:4 * dim + _DIM],
            projected[:, 4 * dim + _DIM:4 * dim + _DIM + heads], self.a_log,
            self.dt_bias, conv_slots, self.state, out, projected[:, 3 * dim:4 * dim],
            self.norm_weight, self.f_b, state_slots, self.correction, self.key_gate,
            record_rows, pending, state_write_slots,
        )
        state_tma = self.state_tensor_map is not None
        if state_tma:
            operands += (self.state_tensor_map,)
        items = batch * heads
        per_sm = max(2, (items + self.sms - 1) // self.sms)
        ctas = (items + per_sm - 1) // per_sm
        kernel = _compile(
            tuple(map(_spec, operands)), heads, self.tokens, self.device, ctas,
            self.eps, self.lower_bound, state_tma,
        )
        kernel(
            *operands[:12], self.lower_bound, _DIM**-0.5, self.eps, *operands[12:]
        )
        return out
