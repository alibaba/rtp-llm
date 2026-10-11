"""Fixed-shape, graph-capturable lookup into the TP-local PLE row shards."""

import ctypes
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_shards(
    ids_ptr,
    shard_ptrs,
    out_ptr,
    N: tl.constexpr,
    SHARD_ROWS: tl.constexpr,
    TOTAL_SHARDS: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    BF16: tl.constexpr,
):
    rows = tl.program_id(0) * 16 + tl.arange(0, 16)
    cols = tl.arange(0, BD)
    ids = tl.load(ids_ptr + rows, mask=rows < N, other=0)
    shard = ids // SHARD_ROWS
    local_row = ids % SHARD_ROWS
    in_range = (rows < N) & (ids >= 0) & (shard < TOTAL_SHARDS)
    addr = tl.load(shard_ptrs + shard, mask=in_range, other=0)
    valid = in_range & (addr != 0)
    if BF16:
        base = addr.to(tl.pointer_type(tl.bfloat16))
    else:
        base = addr.to(tl.pointer_type(tl.float32))
    values = tl.load(
        base[:, None] + local_row[:, None] * D + cols[None, :],
        mask=valid[:, None] & (cols[None, :] < D),
        other=0,
    )
    tl.store(
        out_ptr + rows[:, None] * D + cols[None, :],
        values,
        mask=(rows[:, None] < N) & (cols[None, :] < D),
    )


def is_supported(ids: torch.Tensor, shards: list[torch.Tensor]) -> bool:
    if not ids.is_cuda or torch.version.hip is not None or ids.dtype != torch.int64:
        return False
    if not shards or shards[0].dtype not in (torch.bfloat16, torch.float32):
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
    shape = (*ids.shape, shards[0].shape[1])
    if out is None:
        out = torch.empty(shape, device=ids.device, dtype=shards[0].dtype)
    elif (
        out.shape != shape
        or out.device != ids.device
        or out.dtype != shards[0].dtype
        or not out.is_contiguous()
    ):
        raise ValueError("PLE gather output must match ids and shard geometry")
    n = ids.numel()
    if n:
        _gather_shards[(triton.cdiv(n, 16),)](
            ids,
            pointer_table,
            out,
            N=n,
            SHARD_ROWS=shards[0].shape[0],
            TOTAL_SHARDS=total_shards,
            D=shards[0].shape[1],
            BD=triton.next_power_of_2(shards[0].shape[1]),
            BF16=shards[0].dtype == torch.bfloat16,
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

    cudaHostGetDevicePointer also covers Torch allocators configured to use
    cudaHostRegister, where a host address need not equal the device address.
    The returned CPU tensors own the allocations for all gather/replay calls.
    """
    if torch.version.hip is not None or device.type != "cuda":
        raise RuntimeError("PLE CPU UVA requires NVIDIA CUDA")
    if not shards or any(
        shard.device.type != "cpu"
        or shard.dtype not in (torch.bfloat16, torch.float32)
        or shard.shape != shards[0].shape
        or not shard.is_contiguous()
        for shard in shards
    ):
        raise ValueError("PLE CPU UVA requires equal contiguous BF16/F32 shards")
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
    libraries = {
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if "/libcudart.so" in line and line.split()[-1].startswith("/")
    }
    if len(libraries) != 1:
        raise RuntimeError("PLE CPU UVA needs one loaded CUDA runtime")
    runtime = ctypes.CDLL(libraries.pop())
    get_pointer = runtime.cudaHostGetDevicePointer
    get_pointer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    get_pointer.restype = ctypes.c_int
    pinned = []
    pointers = [0] * total_shards
    with torch.cuda.device(device):
        for index, shard in zip(shard_indices, shards):
            owner = shard if shard.is_pinned() else shard.pin_memory()
            address = ctypes.c_void_p()
            code = get_pointer(ctypes.byref(address), owner.data_ptr(), 0)
            if code or address.value is None:
                raise RuntimeError(f"PLE CPU UVA pointer mapping failed: CUDA {code}")
            pinned.append(owner)
            pointers[index] = address.value
        table = torch.tensor(pointers, dtype=torch.int64, device=device)
    return pinned, table
