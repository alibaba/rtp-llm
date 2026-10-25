# SPDX-License-Identifier: Apache-2.0
"""Native FP8 UVA lookup with a fused GPU-scale packing epilogue.

The default lookup gathers native FP8 bytes, then packs GPU scales in the same
kernel. It is a byte gather/permutation, with no BF16 round trip or requantization.
Valid FP8 bytes (including NaNs) and scale bytes (including 0/255) are preserved.
The legacy exponent-bitcast path treats scale0 as float zero and scale255 as
float infinity. Direct UE8M0 consumption must be checked separately at WKV;
neither exceptional-scale nor general BF16-round-trip equivalence is claimed.

Contract: native contiguous uint8 UVA weight[table_rows,256], contiguous GPU
uint8 scales[table_rows,8] or a ShardedScaleDescriptor, and strided int64
indices[M,24], M>0. A descriptor supplies ceil-divided scale partitions and
a CUDA pointer table of locally mapped peer addresses; no forward collective
or full-table scale replication is performed by this kernel. Values are
contiguous float8_e4m3fn[M,6144]. Scale words are int32[M,48] with strides
(1,align4(M)); all padded rows are written with neutral E8M0=127. Invalid IDs
produce zero FP8 values and neutral scales. The CPU table layout is unchanged.

Allocate once, prepare outside graph capture, then reuse launch functions and
fixed outputs with changing live IDs. Inputs, outputs and the registered UVA
table owner must outlive captured graphs.
SM100 and SM103 share the same capped grid configuration. No CUDA work is
performed on import. Fresh graph capture defers uncached carveout setup until
an eager call outside capture; preparation is not required for correctness.
"""

from dataclasses import dataclass
from functools import lru_cache

import torch
import triton
import triton.language as tl

HEADS = 24
DIM = 256
GRID = 128
NUM_WARPS = 1
CARVEOUT = 100
EPILOGUE_BLOCK_TOKENS = 32
_PREPARED_KERNELS = {}


@triton.jit(
    do_not_specialize=[
        "tokens",
        "table_rows",
        "stride_t",
        "stride_h",
        "scale_stride",
        "shard_rows",
    ]
)
def _lookup_quantized_epilogue_kernel(
    weight,
    gpu_scales,
    indices,
    values,
    packed_scales,
    tokens,
    table_rows,
    stride_t,
    stride_h,
    scale_stride,
    scale_ptrs,
    shard_rows,
    GRID: tl.constexpr,
    SHARDED: tl.constexpr,
):
    weight32 = weight.to(tl.pointer_type(tl.uint32))
    values32 = values.to(tl.pointer_type(tl.uint32))
    words = tl.arange(0, 64)
    for row in tl.range(tl.program_id(0), tokens * 24, GRID):
        row64 = tl.cast(row, tl.int64)
        token, head = row64 // 24, row64 % 24
        index = tl.load(indices + token * stride_t + head * stride_h).to(tl.int64)
        valid = (index >= 0) & (index < table_rows)
        index = tl.where(valid, index, 0)
        raw = tl.load(weight32 + index * 64 + words, valid, other=0)
        tl.store(values32 + row64 * 64 + words, raw)

    # Independent scale tasks only read live IDs and GPU scales. They never
    # read another CTA's raw output, so no inter-CTA synchronization is needed.
    scales64 = gpu_scales.to(tl.pointer_type(tl.uint64))
    lanes = tl.arange(0, 32)
    tasks = 24 * tl.cdiv(scale_stride, 32)
    for task in tl.range(tl.program_id(0), tasks, GRID):
        task64 = tl.cast(task, tl.int64)
        head = task64 % 24
        token = (task64 // 24) * 32 + lanes
        index = tl.load(
            indices + token * stride_t + head * stride_h, token < tokens, other=-1
        ).to(tl.int64)
        valid = (token < tokens) & (index >= 0) & (index < table_rows)
        index = tl.where(valid, index, 0)
        if SHARDED:
            owner = index // shard_rows
            local_row = index - owner * shard_rows
            # Mask before touching the pointer table. The global bound also
            # excludes the unused rows in the final ceil-divided partition.
            address = tl.load(scale_ptrs + owner, valid, other=0).to(tl.uint64)
            source = (address + local_row.to(tl.uint64) * 8).to(
                tl.pointer_type(tl.uint64)
            )
        else:
            source = scales64 + index
        bits = tl.load(source, valid, other=0x7F7F7F7F7F7F7F7F)
        low = bits.to(tl.uint32).to(tl.int32, bitcast=True)
        high = (bits >> 32).to(tl.uint32).to(tl.int32, bitcast=True)
        target = head * 2 * scale_stride + token
        tl.store(packed_scales + target, low, token < scale_stride)
        tl.store(packed_scales + target + scale_stride, high, token < scale_stride)


def _tensor(tensor, dtype, ndim):
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.is_cuda
        and tensor.layout == torch.strided
        and tensor.dtype == dtype
        and tensor.ndim == ndim
        and not tensor.requires_grad
        and not tensor.is_conj()
        and not tensor.is_neg()
    )


def _check_indices(indices):
    if not (
        _tensor(indices, torch.int64, 2)
        and indices.shape[0] > 0
        and indices.shape[1] == HEADS
        and all(stride >= 0 for stride in indices.stride())
    ):
        raise ValueError(
            "Expected CUDA int64 indices[M,24], M>0; arbitrary nonnegative strides are supported"
        )


@dataclass(frozen=True)
class QuantizedLookupOutputs:
    values: torch.Tensor
    scales: torch.Tensor
    scale_storage: torch.Tensor

    def pair(self):
        """Pass this pair to V41MXFP8Linear.forward_quantized after lookup joins."""
        return self.values, self.scales


def allocate_outputs(indices):
    """Allocate fixed output addresses; launches initialize all scale padding."""
    _check_indices(indices)
    tokens = indices.shape[0]
    aligned = triton.cdiv(tokens, 4) * 4
    values = torch.empty(
        (tokens, HEADS * DIM), dtype=torch.float8_e4m3fn, device=indices.device
    )
    storage = torch.empty((48, aligned), dtype=torch.int32, device=indices.device)
    scales = storage.as_strided((tokens, 48), (1, aligned))
    return QuantizedLookupOutputs(values, scales, storage)


def _as_outputs(out):
    if isinstance(out, QuantizedLookupOutputs):
        return out
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        raise ValueError("out must be QuantizedLookupOutputs or its (q, s) pair")
    values, scales = out
    if not (
        _tensor(scales, torch.int32, 2)
        and scales.shape[0] > 0
        and scales.shape[1] == 48
    ):
        raise ValueError("Expected int32[M,48] packed scale output")
    aligned = triton.cdiv(scales.shape[0], 4) * 4
    if scales.stride() != (1, aligned):
        raise ValueError("Packed scale output must have strides(1,align4(M))")
    storage = scales.as_strided((48, aligned), (aligned, 1))
    return QuantizedLookupOutputs(values, scales, storage)


def _check_outputs(indices, outputs, sources=()):
    _check_indices(indices)
    if not isinstance(outputs, QuantizedLookupOutputs):
        raise ValueError("Allocate QuantizedLookupOutputs with allocate_outputs")
    values, scales, storage = outputs.values, outputs.scales, outputs.scale_storage
    tokens = indices.shape[0]
    aligned = triton.cdiv(tokens, 4) * 4
    if not (
        _tensor(values, torch.float8_e4m3fn, 2)
        and values.shape == (tokens, 6144)
        and values.is_contiguous()
        and values.data_ptr() % 4 == 0
        and _tensor(scales, torch.int32, 2)
        and scales.shape == (tokens, 48)
        and scales.stride() == (1, aligned)
        and _tensor(storage, torch.int32, 2)
        and storage.shape == (48, aligned)
        and storage.is_contiguous()
        and storage.data_ptr() == scales.data_ptr()
        and storage.untyped_storage().data_ptr() == scales.untyped_storage().data_ptr()
        and all(t.device == indices.device for t in (values, scales, storage))
    ):
        raise ValueError(
            "Output geometry/device differs from the fixed FP8/packed UE8M0 contract"
        )
    destinations = (
        values.untyped_storage().data_ptr(),
        storage.untyped_storage().data_ptr(),
    )
    if destinations[0] == destinations[1] or any(
        t.untyped_storage().data_ptr() in destinations for t in (indices, *sources)
    ):
        raise ValueError("Lookup outputs must not alias each other or any input")


def _check_table(table, indices, columns, label, alignment=4):
    if not (
        _tensor(table, torch.uint8, 2)
        and table.shape[1] == columns
        and table.is_contiguous()
        and table.data_ptr() % alignment == 0
        and table.device == indices.device
    ):
        raise ValueError(
            f"Expected contiguous aligned CUDA uint8 {label}[table_rows,{columns}]"
        )


def _scale_arguments(scales_gpu, indices, table_rows):
    """Resolve metadata without reading the GPU pointer table on the host."""
    if isinstance(scales_gpu, torch.Tensor):
        _check_table(scales_gpu, indices, 8, "GPU scales", alignment=8)
        if scales_gpu.shape[0] != table_rows:
            raise ValueError("Weight/scales table row counts differ")
        # SHARDED=False removes the unused pointer-table argument and division.
        return scales_gpu, scales_gpu, max(1, table_rows), 1

    from ._engram_scale_shards import ShardedScaleDescriptor

    if not isinstance(scales_gpu, ShardedScaleDescriptor):
        raise ValueError("Expected a GPU scale tensor or ShardedScaleDescriptor")
    if scales_gpu.validate_metadata(indices.device, table_rows) is False:
        raise ValueError("Invalid sharded GPU scale metadata")
    return (
        scales_gpu.local_scales,
        scales_gpu.peer_ptrs,
        scales_gpu.rows_per_shard,
        scales_gpu.group_size,
    )


def _gather_grid(num_sms):
    if type(num_sms) is not int or num_sms <= 0:
        raise ValueError("num_sms must be a positive integer")
    return min(GRID, num_sms)


@lru_cache(maxsize=None)
def _supports_cuda_device(device):
    return torch.cuda.get_device_capability(device) in ((10, 0), (10, 3))


def supports_device(device):
    """Capability gate; owners can query/cache this at initialization."""
    device = torch.device(device)
    if device.type != "cuda":
        return False
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return _supports_cuda_device(device)


def is_supported(weight_uva, scales_gpu, indices, num_sms):
    """Metadata/capability gate for the shared SM100/SM103 implementation."""
    try:
        _gather_grid(num_sms)
        _check_indices(indices)
        _check_table(weight_uva, indices, 256, "native UVA weight")
        _scale_arguments(scales_gpu, indices, weight_uva.shape[0])
    except (ValueError, AttributeError, TypeError):
        return False
    return supports_device(indices.device)


def _prepare_kernel_cached(kernel):
    """Cache driver attributes by compiled handle; never mutate them in capture."""
    if kernel in _PREPARED_KERNELS:
        return _PREPARED_KERNELS[kernel]
    if torch.cuda.is_current_stream_capturing():
        return None
    from cuda.bindings import driver

    function = driver.CUfunction(kernel.function)
    attribute = (
        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
    )
    (status,) = driver.cuFuncSetAttribute(function, attribute, CARVEOUT)
    if status != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"Quantized lookup carveout setup failed: {status}")
    status, value = driver.cuFuncGetAttribute(attribute, function)
    if status != driver.CUresult.CUDA_SUCCESS or value != CARVEOUT:
        raise RuntimeError(
            f"Quantized lookup carveout readback failed: {status}, {value}"
        )
    # Keeping the compiled handle alive prevents reuse of its CUDA function ID.
    _PREPARED_KERNELS[kernel] = value
    return value


def launch_epilogue(weight_uva, scales_gpu, indices, outputs, num_sms=128):
    """Default single-kernel lookup into fixed output storage.

    The first eager launch prepares carveout for subsequent calls. Fresh JIT
    during capture skips driver mutation; an explicit prepare_lookup() outside capture
    establishes the desired carveout before performance-sensitive replays.
    """
    if not is_supported(weight_uva, scales_gpu, indices, num_sms):
        raise ValueError(
            "Native quantized lookup requires supported SM100/SM103 inputs"
        )
    outputs = _as_outputs(outputs)
    local_scales, peer_ptrs, shard_rows, group_size = _scale_arguments(
        scales_gpu, indices, weight_uva.shape[0]
    )
    _check_outputs(indices, outputs, (weight_uva, local_scales, peer_ptrs))
    grid = _gather_grid(num_sms)
    with torch.cuda.device(indices.device):
        kernel = _lookup_quantized_epilogue_kernel[(grid,)](
            weight_uva,
            local_scales,
            indices,
            outputs.values,
            outputs.scales,
            indices.shape[0],
            weight_uva.shape[0],
            indices.stride(0),
            indices.stride(1),
            outputs.scales.stride(1),
            peer_ptrs,
            shard_rows,
            GRID=grid,
            SHARDED=group_size > 1,
            num_warps=NUM_WARPS,
        )
        if not isinstance(scales_gpu, torch.Tensor):
            scales_gpu.record_stream(torch.cuda.current_stream(indices.device))
        _prepare_kernel_cached(kernel)
    return {"epilogue": kernel}


def lookup_host_rows(weight_uva, scales_gpu, indices, num_sms, *, prepare=False):
    """Allocate and return the native (FP8 values, packed GPU scales) pair."""
    if not is_supported(weight_uva, scales_gpu, indices, num_sms):
        raise ValueError(
            "Native quantized lookup requires supported SM100/SM103 inputs"
        )
    outputs = allocate_outputs(indices)
    if prepare:
        prepare_lookup(weight_uva, scales_gpu, indices, outputs, num_sms=num_sms)
    else:
        launch_epilogue(weight_uva, scales_gpu, indices, outputs, num_sms)
    return outputs.pair()


def prepare_lookup(weight_uva, scales_gpu, indices, outputs, *, num_sms=128):
    """Prepare the default kernel outside capture and report its resources."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Quantized lookup prepare must run outside capture")
    outputs = _as_outputs(outputs)
    kernel = launch_epilogue(weight_uva, scales_gpu, indices, outputs, num_sms)[
        "epilogue"
    ]
    value = _prepare_kernel_cached(kernel)
    grid = _gather_grid(num_sms)
    rows = indices.numel()
    loops = [max(0, triton.cdiv(rows - pid, grid)) for pid in range(grid)]
    tasks = HEADS * triton.cdiv(outputs.scales.stride(1), EPILOGUE_BLOCK_TOKENS)
    scale_loops = [max(0, triton.cdiv(tasks - pid, grid)) for pid in range(grid)]
    local_scales, peer_ptrs, shard_rows, group_size = _scale_arguments(
        scales_gpu, indices, weight_uva.shape[0]
    )
    resources = {
        "kernel_name": kernel.name,
        "grid": [grid, 1, 1],
        "num_warps": NUM_WARPS,
        "block_rows": 1,
        "threads_per_cta": 32,
        "n_regs": kernel.n_regs,
        "shared_bytes": kernel.metadata.shared,
        "n_spills": getattr(kernel, "n_spills", None),
        "preferred_shared_memory_carveout": value,
        "loop_rows_including_padding": rows,
        "loop_iterations_min": min(loops),
        "loop_iterations_max": max(loops),
        "raw_loop_iterations_per_cta": loops,
        "scale_epilogue_tasks": tasks,
        "scale_epilogue_tokens_per_task": EPILOGUE_BLOCK_TOKENS,
        "scale_epilogue_iterations_per_cta": scale_loops,
        "scale_epilogue_iterations_min": min(scale_loops),
        "scale_epilogue_iterations_max": max(scale_loops),
        "inter_cta_raw_dependency": False,
        "same_kernel_epilogue": True,
        "scale_group_size": group_size,
        "scale_rows_per_shard": shard_rows,
        "local_scale_bytes": local_scales.numel() * local_scales.element_size(),
        "scale_pointer_table_bytes": (
            peer_ptrs.numel() * peer_ptrs.element_size()
            if not isinstance(scales_gpu, torch.Tensor)
            else 0
        ),
        "scale_addressing": "peer_pointer_table" if group_size > 1 else "direct_local",
    }
    launch_epilogue(weight_uva, scales_gpu, indices, outputs, num_sms)
    torch.cuda.current_stream(indices.device).synchronize()
    return {
        "variant": "epilogue",
        "resources": {"epilogue": resources},
        "tokens": indices.shape[0],
        "values_shape": list(outputs.values.shape),
        "values_dtype": str(outputs.values.dtype),
        "scales_shape": list(outputs.scales.shape),
        "scales_stride": list(outputs.scales.stride()),
        "scale_storage_shape": list(outputs.scale_storage.shape),
        "padding_exponent": 127,
        "invalid_ids": "zero FP8 bytes with neutral E8M0=127; no invalid host/GPU-scale read",
        "valid_scale_0_255_policy": "preserve bytes; direct UE8M0 vs legacy exponent-bitcast semantics require WKV validation",
        "bf16_round_trip_equivalence": "NOT_VALIDATED_BY_THIS_MODULE",
        "cpu_layout": "unchanged native FP8 rows; separate GPU scales",
        "graph_reusable": True,
    }
