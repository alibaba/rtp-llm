"""AITER FlyDSL GDN decode adapter for RTP-LLM's ROCm block cache."""

import functools
import inspect
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.triton_kernels.fla.utils import (
    env_flag,
    is_amd_cdna3,
    is_amd_cdna4,
)

_LOGGER = logging.getLogger(__name__)
_WARMED_DECODE_SIGNATURES: set[tuple[object, ...]] = set()
_LOGGED_BACKEND_DECISIONS: set[bool] = set()


def _flydsl_gdr_decode_contract(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    indices: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    use_qk_l2norm: bool,
    need_shuffle_state: bool,
    stream: torch.cuda.Stream | None = None,
    read_indices: torch.Tensor | None = None,
    write_indices: torch.Tensor | None = None,
) -> None:
    """Static mirror of the installed AITER wrapper's public call contract."""


@dataclass(frozen=True)
class AiterFlydslGdnDecodeStateMetadata:
    """Complete RTP block-cache metadata required by FlyDSL decode."""

    block_map: torch.Tensor
    sequence_lengths_plus_1: torch.Tensor
    seq_size_per_block: int
    host_sequence_lengths: torch.Tensor | None
    state_pool_size: int

    def cache_key(self, state_dtype: torch.dtype) -> tuple[object, ...]:
        """Return a per-forward cache key while retaining tensor identity."""
        return self.make_cache_key(
            self.block_map,
            self.sequence_lengths_plus_1,
            self.host_sequence_lengths,
            self.seq_size_per_block,
            self.state_pool_size,
            state_dtype,
        )

    @staticmethod
    def make_cache_key(
        block_map: torch.Tensor,
        sequence_lengths_plus_1: torch.Tensor,
        host_sequence_lengths: torch.Tensor | None,
        seq_size_per_block: int,
        state_pool_size: int,
        state_dtype: torch.dtype,
    ) -> tuple[object, ...]:
        """Build a layer-independent per-forward dispatch cache key."""
        return (
            block_map.device,
            block_map.data_ptr(),
            tuple(block_map.shape),
            tuple(block_map.stride()),
            sequence_lengths_plus_1.data_ptr(),
            tuple(sequence_lengths_plus_1.shape),
            tuple(sequence_lengths_plus_1.stride()),
            (
                None
                if host_sequence_lengths is None
                else (
                    host_sequence_lengths.device,
                    host_sequence_lengths.data_ptr(),
                    tuple(host_sequence_lengths.shape),
                    tuple(host_sequence_lengths.stride()),
                )
            ),
            seq_size_per_block,
            state_pool_size,
            state_dtype,
        )


@functools.cache
def _is_aiter_flydsl_gdn_decode_disabled() -> bool:
    """Return the optional emergency rollback setting.

    Dispatch remains automatic by default. This process-start setting lets
    operators roll back to Triton without rebuilding or replacing an image.
    Restart the serving process after changing it because the value is cached.
    """
    return env_flag("DISABLE_AITER_FLYDSL_GDN_DECODE")


def _log_backend_decision_once(reason: str | None) -> None:
    """Log the first selected and first fallback decision per process.

    The cache key is the bounded decision class rather than a dynamic reason
    containing batch or shape values, so decode traffic cannot cause log churn.
    """
    supported = reason is None
    if supported in _LOGGED_BACKEND_DECISIONS:
        return
    _LOGGED_BACKEND_DECISIONS.add(supported)
    if supported:
        _LOGGER.info("AITER FlyDSL GDN decode selected")
    else:
        _LOGGER.info("AITER FlyDSL GDN decode fallback to Triton: %s", reason)


@functools.cache
def _warn_host_validation_unavailable_once() -> None:
    _LOGGER.warning(
        "AITER FlyDSL GDN decode host-side block-table validation is disabled "
        "because no CPU sequence-length mirror is available"
    )


@functools.cache
def _get_aiter_flydsl_gdn_decode() -> Callable | None:
    """Resolve the optional AITER symbol once and fall back cleanly if absent."""
    try:
        from aiter.ops.flydsl.linear_attention_kernels import flydsl_gdr_decode
    except Exception as error:
        _LOGGER.warning(
            "AITER FlyDSL GDN decode is unavailable; falling back to Triton: %s",
            error,
            exc_info=True,
        )
        return None
    expected_signature = inspect.signature(_flydsl_gdr_decode_contract)
    actual_signature = inspect.signature(flydsl_gdr_decode)
    incompatible_reason = _callable_signature_incompatibility(
        expected_signature, actual_signature
    )
    if incompatible_reason is not None:
        _LOGGER.warning(
            "AITER FlyDSL GDN decode signature is incompatible; "
            "falling back to Triton: %s; expected=%s, actual=%s",
            incompatible_reason,
            expected_signature,
            actual_signature,
        )
        return None
    return flydsl_gdr_decode


def _callable_signature_incompatibility(
    expected: inspect.Signature,
    actual: inspect.Signature,
) -> str | None:
    """Return why ``actual`` cannot accept the adapter's keyword call.

    AITER may append optional parameters without breaking RTP's call contract.
    Required unknown parameters and missing/non-keyword expected parameters are
    incompatible.
    """
    actual_parameters = actual.parameters
    for name in expected.parameters:
        parameter = actual_parameters.get(name)
        if parameter is None:
            return f"missing parameter {name!r}"
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return f"parameter {name!r} does not accept keyword arguments"

    for name, parameter in actual_parameters.items():
        if name in expected.parameters:
            continue
        if (
            parameter.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            and parameter.default is inspect.Parameter.empty
        ):
            return f"unexpected required parameter {name!r}"
    return None


def _expected_state_inner_stride(state: torch.Tensor) -> tuple[int, int, int]:
    _, _, value_dim, key_dim = state.shape
    return (value_dim * key_dim, key_dim, 1)


def _state_inner_layout_is_contiguous(state: torch.Tensor) -> bool:
    return state.stride()[1:] == _expected_state_inner_stride(state)


def _aiter_flydsl_gdn_decode_unsupported_reason(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    state: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float | None,
) -> str | None:
    if _is_aiter_flydsl_gdn_decode_disabled():
        return "disabled by DISABLE_AITER_FLYDSL_GDN_DECODE"
    if not (is_amd_cdna3 or is_amd_cdna4):
        return "device is not AMD CDNA3/gfx942 or CDNA4/gfx950"
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or state.ndim != 4:
        return "q/k/v/state rank is unsupported"

    batch, query_length, key_heads, key_dim = q.shape
    value_heads = v.shape[2]
    value_dim = v.shape[3]
    expected_ab_shape = (batch * query_length, value_heads)
    expected_scale = key_dim**-0.5

    if batch < 1 or query_length < 1:
        return f"decode shape requires B>=1 and T>=1, got B={batch}, T={query_length}"
    if q.device.type != "cuda":
        return f"input device must be CUDA/HIP, got {q.device.type}"
    if not (
        q.device
        == k.device
        == v.device
        == a.device
        == b.device
        == state.device
        == A_log.device
        == dt_bias.device
    ):
        return "input, parameter, and state devices differ"
    if any(tensor.dtype != torch.bfloat16 for tensor in (q, k, v, a, b)):
        return "q/k/v/a/b must be bfloat16"
    if state.dtype not in (torch.float32, torch.bfloat16):
        return f"state dtype is unsupported: {state.dtype}"
    if state.data_ptr() % 16 != 0:
        return "state base pointer is not 16-byte aligned"
    if state.stride(0) * state.element_size() % 16 != 0:
        return "state row stride is not 16-byte aligned"
    if not _state_inner_layout_is_contiguous(state):
        return f"state inner stride is unsupported: {state.stride()[1:]}"
    if k.shape != q.shape or v.shape[:2] != (batch, query_length):
        return "q/k/v shapes are inconsistent"
    if key_heads < 1 or value_heads % key_heads != 0:
        return "value head count must be divisible by key head count"
    if q.stride(-1) != 1 or k.stride(-1) != 1:
        return "q/k head dimensions must be contiguous"
    key_tile = 32 if state.dtype == torch.float32 else 64
    if key_dim < key_tile or key_dim % key_tile != 0:
        return f"key dimension must be divisible by {key_tile}"
    if value_dim < 32 or value_dim % 32 != 0:
        return "value dimension must be divisible by 32"
    if a.shape != expected_ab_shape or b.shape != expected_ab_shape:
        return (
            f"a/b shape must be {expected_ab_shape}, "
            f"got {tuple(a.shape)} and {tuple(b.shape)}"
        )
    if state.shape[1:] != (value_heads, value_dim, key_dim):
        return f"state shape is inconsistent: {tuple(state.shape)}"
    if (
        A_log.ndim != 1
        or A_log.stride(0) != 1
        or A_log.dtype not in (torch.float32, torch.bfloat16)
    ):
        return (
            "A_log layout or dtype is unsupported: "
            f"shape={tuple(A_log.shape)}, stride={A_log.stride()}, "
            f"dtype={A_log.dtype}"
        )
    if A_log.numel() != value_heads:
        return f"A_log must contain {value_heads} values"
    if (
        dt_bias.ndim != 1
        or dt_bias.stride(0) != 1
        or dt_bias.dtype != q.dtype
        or dt_bias.numel() != value_heads
    ):
        return (
            f"dt_bias must be contiguous 1D {q.dtype} with {value_heads} values; "
            f"got shape={tuple(dt_bias.shape)}, stride={dt_bias.stride()}, "
            f"dtype={dt_bias.dtype}"
        )
    if scale is not None and not math.isclose(scale, expected_scale, rel_tol=1e-6):
        return f"scale must be {expected_scale}, got {scale}"
    if _get_aiter_flydsl_gdn_decode() is None:
        return "AITER flydsl_gdr_decode symbol is unavailable"
    return None


def is_aiter_flydsl_gdn_decode_supported(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    state: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float | None = None,
    *,
    block_map: torch.Tensor | None = None,
    sequence_lengths_plus_1: torch.Tensor | None = None,
    seq_size_per_block: int | None = None,
    host_sequence_lengths: torch.Tensor | None = None,
    state_pool_size: int | None = None,
) -> bool:
    """Return whether automatic AITER FlyDSL dispatch supports all inputs."""
    reason = _aiter_flydsl_gdn_decode_unsupported_reason(
        q,
        k,
        v,
        a,
        b,
        state,
        A_log,
        dt_bias,
        scale,
    )
    metadata_values = (
        block_map,
        sequence_lengths_plus_1,
        seq_size_per_block,
        host_sequence_lengths,
        state_pool_size,
    )
    metadata_supplied = any(value is not None for value in metadata_values)
    if reason is None and metadata_supplied:
        if (
            block_map is None
            or sequence_lengths_plus_1 is None
            or seq_size_per_block is None
            or state_pool_size is None
        ):
            reason = "incomplete decode state metadata"
        else:
            reason = _decode_state_metadata_unsupported_reason(
                block_map,
                sequence_lengths_plus_1,
                seq_size_per_block,
                host_sequence_lengths,
                state_pool_size,
                expected_batch=q.shape[0],
            )
            if reason is None and state_pool_size != state.shape[0]:
                reason = (
                    "state_pool_size differs from state.shape[0]: "
                    f"{state_pool_size} != {state.shape[0]}"
                )
    _log_backend_decision_once(reason)
    return reason is None


def _decode_state_metadata_unsupported_reason(
    block_map: torch.Tensor,
    sequence_lengths_plus_1: torch.Tensor,
    seq_size_per_block: int,
    host_sequence_lengths: torch.Tensor | None,
    state_pool_size: int,
    *,
    expected_batch: int | None = None,
) -> str | None:
    if block_map.ndim != 2:
        return f"block_map must be 2D, got {tuple(block_map.shape)}"
    if block_map.shape[1] == 0:
        return "block_map must contain at least one block column"
    if block_map.dtype != torch.int32:
        return f"block_map must be int32, got {block_map.dtype}"
    if block_map.stride(1) != 1:
        return f"block_map columns must be contiguous, got {block_map.stride()}"
    if sequence_lengths_plus_1.ndim != 1:
        return (
            "sequence_lengths_plus_1 must be 1D, "
            f"got {tuple(sequence_lengths_plus_1.shape)}"
        )
    if sequence_lengths_plus_1.dtype != torch.int32:
        return (
            "sequence_lengths_plus_1 must be int32, "
            f"got {sequence_lengths_plus_1.dtype}"
        )
    if sequence_lengths_plus_1.stride(0) != 1:
        return (
            "sequence_lengths_plus_1 must be contiguous, "
            f"got {sequence_lengths_plus_1.stride()}"
        )
    if block_map.device != sequence_lengths_plus_1.device:
        return "block_map and sequence lengths must be on the same device"
    if sequence_lengths_plus_1.numel() != block_map.shape[0]:
        return "sequence length count must equal block-map batch size"
    if expected_batch is not None and block_map.shape[0] != expected_batch:
        return (
            "block-map batch differs from decode input batch: "
            f"{block_map.shape[0]} != {expected_batch}"
        )
    if seq_size_per_block <= 0:
        return f"seq_size_per_block must be positive, got {seq_size_per_block}"
    if state_pool_size <= 0:
        return f"state_pool_size must be positive, got {state_pool_size}"
    if (
        isinstance(host_sequence_lengths, torch.Tensor)
        and host_sequence_lengths.device.type == "cpu"
        and host_sequence_lengths.numel() > 0
    ):
        max_sequence_length_plus_1 = int(host_sequence_lengths.max().item()) + 1
        max_write_pos = (max_sequence_length_plus_1 - 1) // seq_size_per_block
        if max_write_pos >= block_map.shape[1]:
            return "real request exceeds block-map width"
    else:
        _warn_host_validation_unavailable_once()
    return None


@triton.jit
def _prepare_decode_state_indices_kernel(
    block_map,
    sequence_lengths_plus_1,
    read_indices,
    write_indices,
    invalid_row_flags,
    block_map_row_stride: tl.int64,
    block_map_width: tl.int64,
    state_pool_size: tl.int64,
    seq_size_per_block: tl.constexpr,
):
    batch = tl.program_id(0)
    sequence_length = tl.load(sequence_lengths_plus_1 + batch).to(tl.int64)
    read_pos = (sequence_length - 2) // seq_size_per_block
    write_pos = (sequence_length - 1) // seq_size_per_block
    valid = (
        (sequence_length >= 2)
        & (read_pos >= 0)
        & (write_pos >= 0)
        & (read_pos < block_map_width)
        & (write_pos < block_map_width)
    )
    safe_read_pos = tl.minimum(tl.maximum(read_pos, 0), block_map_width - 1)
    safe_write_pos = tl.minimum(tl.maximum(write_pos, 0), block_map_width - 1)
    row_start = batch * block_map_row_stride
    read_id = tl.load(block_map + row_start + safe_read_pos)
    write_id = tl.load(block_map + row_start + safe_write_pos)
    valid_ids = (
        valid
        & (read_id > 0)
        & (write_id > 0)
        & (read_id < state_pool_size)
        & (write_id < state_pool_size)
    )
    # AITER treats negative state indices as CUDA/HIP Graph padding and skips
    # both the recurrent update and state store. Do not route padding through
    # block 0: concurrent padding rows would otherwise race on the dummy block.
    # Any invalid row with sequence_length >= 2 still sets the diagnostic flag.
    tl.store(
        invalid_row_flags + batch,
        ((sequence_length >= 2) & ~valid_ids).to(tl.int32),
    )
    tl.store(read_indices + batch, tl.where(valid_ids, read_id, -1))
    tl.store(write_indices + batch, tl.where(valid_ids, write_id, -1))


def _prepare_aiter_flydsl_gdn_decode_state_indices(
    state_metadata: AiterFlydslGdnDecodeStateMetadata,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve source/current block IDs for one decode token per request.

    ``sequence_lengths_plus_1`` must contain the post-decode length and normally
    be at least 2. Invalid graph-padding rows and rows whose read or write block
    is non-positive resolve both indices to ``-1``. AITER recognizes this
    sentinel and skips the row without reading or writing the state pool. A real
    request known from the CPU length mirror to exceed the block-table width
    fails fast. Valid rows load from ``read_indices`` and store the updated state
    directly to ``write_indices`` in the fused decode kernel.

    The returned int32 tensor flags real-request rows with invalid positions or
    state-pool IDs; graph padding is excluded. It is updated by the existing
    index kernel during Graph replay and retained by the model for diagnostics,
    without an additional hot-path kernel or device synchronization.
    """
    reason = _decode_state_metadata_unsupported_reason(
        state_metadata.block_map,
        state_metadata.sequence_lengths_plus_1,
        state_metadata.seq_size_per_block,
        state_metadata.host_sequence_lengths,
        state_metadata.state_pool_size,
    )
    if reason is not None:
        raise ValueError(reason)

    block_map = state_metadata.block_map
    sequence_lengths_plus_1 = state_metadata.sequence_lengths_plus_1
    batch = block_map.shape[0]
    read_indices = torch.empty(batch, device=block_map.device, dtype=torch.int32)
    write_indices = torch.empty_like(read_indices)
    invalid_row_flags = torch.empty(batch, device=block_map.device, dtype=torch.int32)
    if batch == 0:
        return read_indices, write_indices, invalid_row_flags
    _prepare_decode_state_indices_kernel[(batch,)](
        block_map,
        sequence_lengths_plus_1,
        read_indices,
        write_indices,
        invalid_row_flags,
        block_map_row_stride=block_map.stride(0),
        block_map_width=block_map.shape[1],
        state_pool_size=state_metadata.state_pool_size,
        seq_size_per_block=state_metadata.seq_size_per_block,
        num_warps=1,
    )
    return read_indices, write_indices, invalid_row_flags


def prepare_aiter_flydsl_gdn_decode_state_indices(
    state_metadata: AiterFlydslGdnDecodeStateMetadata,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve indices and return device-side invalid-row diagnostic flags.

    The flags are updated by the index kernel during graph replay. Consumers
    must inspect them outside the decode hot path because host inspection
    synchronizes the device.
    """
    return _prepare_aiter_flydsl_gdn_decode_state_indices(state_metadata)


def _decode_indices_unsupported_reason(
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    *,
    device: torch.device,
    expected_numel: int | None = None,
) -> str | None:
    if read_indices.shape != write_indices.shape:
        return (
            "read_indices and write_indices must have the same shape, "
            f"got {tuple(read_indices.shape)} and {tuple(write_indices.shape)}"
        )
    if (
        read_indices.ndim != 1
        or write_indices.ndim != 1
        or read_indices.dtype != torch.int32
        or write_indices.dtype != torch.int32
        or read_indices.stride(0) != 1
        or write_indices.stride(0) != 1
    ):
        return "read_indices and write_indices must be contiguous 1D int32 tensors"
    if read_indices.device != device or write_indices.device != device:
        return "state/input and decode indices must be on the same device"
    if expected_numel is not None and (
        read_indices.numel() != expected_numel
        or write_indices.numel() != expected_numel
    ):
        return "read_indices and write_indices must contain one element per batch row"
    return None


@torch.compiler.disable
def aiter_flydsl_gdn_decode(
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    state: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    already_validated: bool = False,
) -> torch.Tensor:
    """Run FlyDSL decode with independent state-cache read/write indices.

    Call this path once per graph-runner batch bucket before CUDA/HIP Graph
    capture so AITER can finish lazy module and kernel initialization. RTP's
    graph runner performs eager forwards before capture; capture and replay are
    supported after that warmup.
    """
    unsupported_reason = None
    if not already_validated:
        unsupported_reason = _aiter_flydsl_gdn_decode_unsupported_reason(
            q, k, v, a, b, state, A_log, dt_bias, scale
        )
    if unsupported_reason is not None:
        raise ValueError(
            "AITER FlyDSL GDN decode is unsupported: "
            f"{unsupported_reason}; got q={tuple(q.shape)}/{q.dtype}, "
            f"v={tuple(v.shape)}/{v.dtype}, a={a.dtype}, b={b.dtype}, "
            f"state={tuple(state.shape)}/{state.dtype}/{state.stride()}"
        )
    indices_reason = _decode_indices_unsupported_reason(
        read_indices,
        write_indices,
        device=q.device,
        expected_numel=q.shape[0],
    )
    if indices_reason is not None:
        raise ValueError(indices_reason)

    expected_scale = q.shape[-1] ** -0.5
    if scale is not None and not math.isclose(scale, expected_scale, rel_tol=1e-6):
        raise ValueError(
            "AITER FlyDSL GDN decode uses the fixed head-dimension scale "
            f"{expected_scale}, got {scale}"
        )

    decode_signature = (
        q.device.index,
        tuple(q.shape),
        tuple(q.stride()),
        tuple(k.stride()),
        tuple(v.shape),
        tuple(v.stride()),
        tuple(a.stride()),
        tuple(b.stride()),
        q.dtype,
        tuple(state.shape),
        tuple(state.stride()),
        state.dtype,
        A_log.dtype,
        use_qk_l2norm_in_kernel,
    )
    is_capturing = torch.cuda.is_current_stream_capturing()
    if is_capturing and decode_signature not in _WARMED_DECODE_SIGNATURES:
        raise RuntimeError(
            "AITER FlyDSL GDN decode must run once eagerly for this device, "
            "shape, and dtype before CUDA/HIP Graph capture; "
            f"current_signature={decode_signature}, "
            f"warmed_signatures={tuple(_WARMED_DECODE_SIGNATURES)}"
        )

    flydsl_gdr_decode = _get_aiter_flydsl_gdn_decode()
    if flydsl_gdr_decode is None:
        raise RuntimeError("AITER FlyDSL GDN decode became unavailable after dispatch")

    batch, query_length = q.shape[:2]
    value_heads = v.shape[2]
    output = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    flydsl_gdr_decode(
        query=q,
        key=k,
        value=v,
        a=a.reshape(batch, query_length, value_heads),
        b=b.reshape(batch, query_length, value_heads),
        dt_bias=dt_bias,
        A_log=A_log,
        indices=write_indices,
        read_indices=read_indices,
        write_indices=write_indices,
        state=state,
        out=output,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
        # RTP stores the persistent SSM cache in VK layout already.
        need_shuffle_state=False,
    )
    if not is_capturing:
        _WARMED_DECODE_SIGNATURES.add(decode_signature)
    return output
