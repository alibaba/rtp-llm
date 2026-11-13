"""Fixed-shape, graph-capturable lookup into the TP-local PLE row shards."""

import ctypes

import torch
import triton
import triton.language as tl

from .ple_hash_triton import _hash_two_three


@triton.jit
def _lookup_shard_rows(
    ids, live, cols, shard_ptrs, scale_ptr,
    SHARD_ROWS: tl.constexpr, TOTAL_SHARDS: tl.constexpr,
    D: tl.constexpr, BF16: tl.constexpr, FP8: tl.constexpr,
):
    shard = ids // SHARD_ROWS
    local_row = ids % SHARD_ROWS
    in_range = live & (ids >= 0) & (shard < TOTAL_SHARDS)
    addr = tl.load(shard_ptrs + shard, mask=in_range, other=0)
    valid = in_range & (addr != 0)
    if FP8:
        base = addr.to(tl.pointer_type(tl.uint8))
    elif BF16:
        base = addr.to(tl.pointer_type(tl.bfloat16))
    else:
        base = addr.to(tl.pointer_type(tl.float32))
    values = tl.load(
        base[:, None] + local_row[:, None] * D + cols[None, :],
        mask=valid[:, None] & (cols[None, :] < D),
        other=0,
    )
    if FP8:
        values = values.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16) * tl.load(
            scale_ptr
        ).to(tl.bfloat16)
    return values


@triton.jit(do_not_specialize=["N"])
def _gather_shards(
    ids_ptr, shard_ptrs, scale_ptr, out_ptr, N,
    SHARD_ROWS: tl.constexpr, TOTAL_SHARDS: tl.constexpr,
    D: tl.constexpr, BD: tl.constexpr, BR: tl.constexpr,
    BF16: tl.constexpr, FP8: tl.constexpr,
):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.arange(0, BD)
    live = rows < N
    ids = tl.load(ids_ptr + rows, mask=live, other=0)
    values = _lookup_shard_rows(
        ids, live, cols, shard_ptrs, scale_ptr, SHARD_ROWS, TOTAL_SHARDS, D, BF16, FP8,
    )
    tl.store(
        out_ptr + rows[:, None] * D + cols[None, :], values,
        mask=live[:, None] & (cols[None, :] < D),
    )


@triton.jit
def _gather_hashed_decode(
    history, vocab, offsets, multipliers, shard_ptrs, out,
    BATCH: tl.constexpr, SHARD_ROWS: tl.constexpr, TOTAL_SHARDS: tl.constexpr,
    EOS: tl.constexpr, BR: tl.constexpr,
):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    request, head = rows // 16, rows % 16
    live = request < BATCH
    base = history + request * 3
    ids = _hash_two_three(
        tl.load(base + 2, live, EOS), tl.load(base + 1, live, EOS),
        tl.load(base, live, EOS), head, tl.load(vocab + head),
        tl.load(offsets + head), multipliers, EOS,
    )
    cols = tl.arange(0, 256)
    values = _lookup_shard_rows(
        ids, live, cols, shard_ptrs, None, SHARD_ROWS, TOTAL_SHARDS, 160, True, False,
    )
    tl.store(
        out + rows[:, None] * 160 + cols[None, :], values,
        mask=live[:, None] & (cols[None, :] < 160),
    )


def try_gather_hashed_decode(
    history, vocab, offsets, multipliers, shards, pointer_table, total_shards, eos,
):
    """Fuse SM90 single-token hashing and pinned BF16 shard lookup."""
    if not (
        history.ndim == 2
        and 16 <= history.shape[0] <= 128
        and history.shape[1] == 3
        and history.is_contiguous()
        and is_supported(history, shards)
        and shards[0].device.type == "cpu"
        and shards[0].dtype == torch.bfloat16
        and shards[0].shape[1] == 160
        and torch.cuda.get_device_capability(history.device) == (9, 0)
        and all(
            tensor.shape == shape and tensor.device == history.device
            and tensor.dtype == torch.int64 and tensor.is_contiguous()
            for tensor, shape in (
                (vocab, (16,)), (offsets, (16,)), (multipliers, (3,)),
                (pointer_table, (total_shards,)),
            )
        )
    ):
        return None
    batch = history.shape[0]
    out = torch.empty((batch, 1, 16, 160), device=history.device, dtype=torch.bfloat16)
    rows_per_cta = 8
    _gather_hashed_decode[(triton.cdiv(batch * 16, rows_per_cta),)](
        history, vocab, offsets, multipliers, pointer_table, out,
        BATCH=batch, SHARD_ROWS=shards[0].shape[0], TOTAL_SHARDS=total_shards,
        EOS=eos, BR=rows_per_cta, num_warps=4,
    )
    return out


def is_supported(ids: torch.Tensor, shards: list[torch.Tensor]) -> bool:
    if not ids.is_cuda or torch.version.hip is not None or ids.dtype != torch.int64:
        return False
    if not shards or shards[0].dtype not in (
        torch.bfloat16,
        torch.float32,
        torch.float8_e4m3fn,
    ):
        return False
    shape = shards[0].shape
    return (
        len(shape) == 2
        and 0 < shape[1] <= 256
        and all(
            (
                (shard.is_cuda and shard.device == ids.device)
                or (shard.device.type == "cpu" and shard.is_pinned())
            )
            and shard.dtype == shards[0].dtype
            and shard.shape == shape
            and shard.is_contiguous()
            for shard in shards
        )
    )


def gather_local(
    ids: torch.Tensor,
    shards: list[torch.Tensor],
    shard_indices: tuple[int, ...],
    total_shards: int,
    pointer_table: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return local shard values, leaving remote shard values at zero."""
    if not is_supported(ids, shards):
        raise ValueError("unsupported PLE Triton shard layout")
    if pointer_table.shape != (total_shards,) or pointer_table.dtype != torch.int64:
        raise ValueError("PLE pointer table must cover every global shard")
    if len(shard_indices) != len(shards):
        raise ValueError("PLE shard index count mismatch")
    if pointer_table.device != ids.device or not pointer_table.is_contiguous():
        raise ValueError("PLE pointer table must be contiguous on the ids device")
    ids = ids.contiguous()
    fp8 = shards[0].dtype == torch.float8_e4m3fn
    output_dtype = torch.bfloat16 if fp8 else shards[0].dtype
    if fp8 and (
        scale is None
        or scale.device != ids.device
        or scale.numel() != 1
        or scale.dtype not in (torch.bfloat16, torch.float32)
    ):
        raise ValueError("FP8 PLE gather requires one device-local floating scale")
    shape = (*ids.shape, shards[0].shape[1])
    if out is None:
        out = torch.empty(shape, device=ids.device, dtype=output_dtype)
    elif (
        out.shape != shape
        or out.device != ids.device
        or out.dtype != output_dtype
        or not out.is_contiguous()
    ):
        raise ValueError("PLE gather output must match ids and shard geometry")
    n = ids.numel()
    if n:
        # The per-row UVA layout follows vLLM's pinned lookup. Four rows per
        # CTA won the actual-table TP2 complete-consumer 4K/12K experiment;
        # one row and explicit packed-word variants regressed and stay out.
        rows_per_cta = (
            4
            if (
                fp8
                and shards[0].device.type == "cpu"
                and shards[0].shape[1] == 160
                and 65536 <= n <= 196608
                and torch.cuda.get_device_capability(ids.device) == (12, 0)
            )
            else 16
        )
        _gather_shards[(triton.cdiv(n, rows_per_cta),)](
            ids,
            pointer_table,
            scale if fp8 else pointer_table,
            out,
            N=n,
            SHARD_ROWS=shards[0].shape[0],
            TOTAL_SHARDS=total_shards,
            D=shards[0].shape[1],
            BD=triton.next_power_of_2(shards[0].shape[1]),
            BR=rows_per_cta,
            BF16=shards[0].dtype == torch.bfloat16,
            FP8=fp8,
            num_warps=4,
        )
    return out


def prepare_cpu_uva_shards(
    shards: list[torch.Tensor],
    shard_indices: list[int],
    total_shards: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Pin CPU tables and resolve mapped addresses once, before graph capture.

    cuMemHostGetDevicePointer also covers Torch allocators configured to use
    cudaHostRegister, where a host address need not equal the device address.
    The returned CPU tensors own the allocations for all gather/replay calls.
    """
    if torch.version.hip is not None or device.type != "cuda":
        raise RuntimeError("PLE CPU UVA requires NVIDIA CUDA")
    if not shards or any(
        shard.device.type != "cpu"
        or shard.dtype
        not in (
            torch.bfloat16,
            torch.float32,
            torch.float8_e4m3fn,
        )
        or shard.shape != shards[0].shape
        or not shard.is_contiguous()
        for shard in shards
    ):
        raise ValueError("PLE CPU UVA requires equal contiguous BF16/F32/FP8 shards")
    if (
        shards[0].ndim != 2
        or not 0 < shards[0].shape[1] <= 256
        or len(shards) != len(shard_indices)
        or len(set(shard_indices)) != len(shard_indices)
        or total_shards <= 0
        or any(i < 0 or i >= total_shards for i in shard_indices)
    ):
        raise ValueError("PLE CPU UVA shard geometry or indices are invalid")
    torch.cuda.init()
    # Torch wheels and compiled extensions may load different compatible
    # libcudart builds in one process. Pinned allocations belong to the CUDA
    # driver; resolving their mapped address through the driver avoids guessing
    # which runtime created Torch's allocation (or rejecting a valid process).
    driver = ctypes.CDLL("libcuda.so.1")
    get_pointer = driver.cuMemHostGetDevicePointer_v2
    get_pointer.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    get_pointer.restype = ctypes.c_int
    pinned = []
    pointers = [0] * total_shards
    with torch.cuda.device(device):
        for index, shard in zip(shard_indices, shards):
            owner = shard if shard.is_pinned() else shard.pin_memory()
            address = ctypes.c_uint64()
            code = get_pointer(ctypes.byref(address), owner.data_ptr(), 0)
            if code or not address.value:
                raise RuntimeError(
                    f"PLE CPU UVA pointer mapping failed: CUDA driver {code}"
                )
            pinned.append(owner)
            pointers[index] = address.value
        table = torch.tensor(pointers, dtype=torch.int64, device=device)
    return pinned, table
