from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import TYPE_CHECKING, Protocol

import torch
import triton
import triton.language as tl

if TYPE_CHECKING:
    from rtp_llm.models_py.modules.offload.runtime import KVWrite
    from rtp_llm.ops import OffloadStorage


@dataclass(frozen=True)
class FetchRows:
    backing_ids: torch.Tensor
    resident_ids: torch.Tensor
    positions: torch.Tensor
    count: torch.Tensor


class LayoutOps(Protocol):
    def fetch(self, storage: OffloadStorage, misses: FetchRows) -> None: ...

    def store(
        self, storage: OffloadStorage, data: KVWrite, resident_ids: torch.Tensor
    ) -> None: ...


@triton.jit
def _row_offset(
    row, PAGE_ROWS: tl.constexpr, PAGE_STRIDE: tl.constexpr, ROW_STRIDE: tl.constexpr
):
    row = row.to(tl.int64)
    return row // PAGE_ROWS * PAGE_STRIDE + row % PAGE_ROWS * ROW_STRIDE


@triton.jit
def _payload_offset(x, SHAPE: tl.constexpr, STRIDES: tl.constexpr, START: tl.constexpr):
    offset = tl.full(x.shape, 0, tl.int64)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        offset += (x % SHAPE[axis]).to(tl.int64) * STRIDES[START + axis]
        x = x // SHAPE[axis]
    return offset


@triton.jit
def _fetch_tiled(
    Host,
    Resident,
    Ids,
    ResidentIds,
    Positions,
    Count,
    PAGE_ROWS: tl.constexpr,
    HBM_ROWS: tl.constexpr,
    SHAPE: tl.constexpr,
    HOST: tl.constexpr,
    GPU: tl.constexpr,
    WIDTH: tl.constexpr,
    UNIT: tl.constexpr,
    B: tl.constexpr,
    ROWS: tl.constexpr,
):
    pointer: tl.constexpr = tl.pointer_type(tl.int32 if UNIT == 4 else tl.uint8)
    host = Host.to(pointer)
    resident = Resident.to(pointer)
    count = tl.load(Count)
    for start in range(tl.program_id(0) * ROWS, count, tl.num_programs(0) * ROWS):
        entry = start + tl.arange(0, ROWS)
        valid = entry < count
        position = tl.load(Positions + entry, valid, other=0)
        token = tl.load(Ids + position, valid, other=HBM_ROWS).to(tl.int64)
        slot = tl.load(ResidentIds + position, valid, other=0).to(tl.int64)
        source = _row_offset(token - HBM_ROWS, PAGE_ROWS, HOST[0], HOST[1])
        destination = _row_offset(slot, PAGE_ROWS, GPU[0], GPU[1])
        x = tl.arange(0, B)
        mask = valid[:, None] & (x[None, :] < WIDTH)
        source_x = _payload_offset(x, SHAPE, HOST, 2)
        destination_x = _payload_offset(x, SHAPE, GPU, 2)
        data = tl.load(host + source[:, None] + source_x[None, :], mask, other=0)
        tl.store(resident + destination[:, None] + destination_x[None, :], data, mask)


@triton.jit
def _write(
    Host,
    Resident,
    Ids,
    ResidentIds,
    Values,
    PAGE_ROWS: tl.constexpr,
    HBM_ROWS: tl.constexpr,
    SHAPE: tl.constexpr,
    HOST: tl.constexpr,
    GPU: tl.constexpr,
    VALUE: tl.constexpr,
    WIDTH: tl.constexpr,
    UNIT: tl.constexpr,
    B: tl.constexpr,
):
    pointer: tl.constexpr = tl.pointer_type(tl.int32 if UNIT == 4 else tl.uint8)
    host = Host.to(pointer)
    resident = Resident.to(pointer)
    values = Values.to(pointer)
    i = tl.program_id(0).to(tl.int64)
    token = tl.load(Ids + i).to(tl.int64)
    if token >= 0:
        x = tl.arange(0, B)
        source_x = _payload_offset(x, SHAPE, VALUE, 1)
        host_x = _payload_offset(x, SHAPE, HOST, 2)
        gpu_x = _payload_offset(x, SHAPE, GPU, 2)
        data = tl.load(values + i * VALUE[0] + source_x, x < WIDTH, other=0)
        if token < HBM_ROWS:
            destination = _row_offset(token, PAGE_ROWS, GPU[0], GPU[1])
            tl.store(resident + destination + gpu_x, data, x < WIDTH)
        else:
            destination = _row_offset(token - HBM_ROWS, PAGE_ROWS, HOST[0], HOST[1])
            tl.store(host + destination + host_x, data, x < WIDTH)
            slot = tl.load(ResidentIds + i).to(tl.int64)
            if slot >= 0:
                destination = _row_offset(slot, PAGE_ROWS, GPU[0], GPU[1])
                tl.store(resident + destination + gpu_x, data, x < WIDTH)


def _copy_unit(*planes: torch.Tensor) -> int:
    for plane in planes:
        if plane.stride(-1) != 1 or plane.shape[-1] * plane.element_size() % 4:
            return 1
        if plane.data_ptr() % 4 or any(
            s * plane.element_size() % 4 for s in plane.stride()[: plane.ndim - 1]
        ):
            return 1
    return 4


def _copy_view(plane: torch.Tensor, unit: int):
    shape = tuple(plane.shape)
    strides = tuple(s * plane.element_size() // unit for s in plane.stride())
    if unit == 4:
        return (*shape[:-1], shape[-1] * plane.element_size() // unit), (
            *strides[:-1],
            1,
        )
    # Byte coordinates preserve native dtype bits, including strided head axes.
    return (*shape, plane.element_size()), (*strides, 1)


class PagedRowLayout:
    """Copy group-provided entry views, preserving head strides and page padding."""

    def fetch(self, storage: OffloadStorage, misses: FetchRows) -> None:
        for host, resident in zip(storage.host_planes, storage.resident_planes):
            unit = _copy_unit(host, resident)
            shape, host_strides = _copy_view(host, unit)
            _, gpu_strides = _copy_view(resident, unit)
            _fetch_tiled[(min(triton.cdiv(misses.backing_ids.numel(), 4), 128),)](
                host,
                resident,
                misses.backing_ids,
                misses.resident_ids,
                misses.positions,
                misses.count,
                storage.rows_per_page,
                storage.hbm_blocks * storage.rows_per_block,
                shape[2:],
                host_strides,
                gpu_strides,
                prod(shape[2:]),
                unit,
                triton.next_power_of_2(prod(shape[2:])),
                4,
            )

    def store(
        self, storage: OffloadStorage, data: KVWrite, resident_ids: torch.Tensor
    ) -> None:
        for host, resident, values in zip(
            storage.host_planes, storage.resident_planes, data.planes
        ):
            unit = _copy_unit(host, resident, values)
            shape, host_strides = _copy_view(host, unit)
            _, gpu_strides = _copy_view(resident, unit)
            _, value_strides = _copy_view(values, unit)
            _write[(data.backing_ids.numel(),)](
                host,
                resident,
                data.backing_ids,
                resident_ids,
                values,
                storage.rows_per_page,
                storage.hbm_blocks * storage.rows_per_block,
                shape[2:],
                host_strides,
                gpu_strides,
                value_strides,
                prod(shape[2:]),
                unit,
                triton.next_power_of_2(prod(shape[2:])),
            )
