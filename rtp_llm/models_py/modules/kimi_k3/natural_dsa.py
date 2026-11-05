"""Preparation contracts shared by the natural-outer decoder and DSA."""

import os

import torch


def outer_natural_enabled(environ=None):
    value = (os.environ if environ is None else environ).get("KIMI_K31_OUTER_NATURAL", "0")
    if value not in ("0", "1"):
        raise ValueError("KIMI_K31_OUTER_NATURAL must be exactly 0 or 1")
    return value == "1"


def validate_shared_restore(layout, plan):
    """Compare the actual shared restore and padding, not just their extents."""
    if not plan.lengths or any(n <= 0 for n in plan.lengths):
        raise ValueError("natural outer Prefill requires nonempty real requests")
    if layout.physical_rows != plan.physical_rows or layout.local_rows != plan.local_rows:
        raise ValueError("shared restore extent differs from natural routing plan")
    expected = torch.empty(plan.physical_rows, dtype=torch.long)
    for rank in range(plan.size):
        incoming = type(plan)(plan.lengths, plan.size, rank)
        for segment in incoming.to_zigzag:
            natural_start = segment.peer * plan.local_rows + segment.source_offset
            zigzag_start = rank * plan.local_rows + segment.destination_offset
            expected[natural_start:natural_start + segment.rows] = torch.arange(
                zigzag_start, zigzag_start + segment.rows)
    if not torch.equal(layout.restore.cpu(), expected):
        raise ValueError("shared restore permutation differs from natural routing plan")
    padding = torch.cat([
        torch.arange(padded) < length
        for length, padded in zip(plan.lengths, plan.padded_lengths)
    ])
    if not torch.equal(layout.padding_mask.cpu(), padding):
        raise ValueError("shared restore padding differs from natural routing plan")


def route_allocation_budget(exchange):
    """Conservative device-local accounting for Torch's raw CUDA VMM source.

    Read the actual allocation properties: granularity is driver/device/handle
    specific, not a portable 2 MiB constant. Charge pointer arrays as independent
    granularity-sized allocations, a conservative bound on cudaMalloc backing.
    Receive slots use the caching allocator and are rounded to its 512 B quantum.
    This is a planning bound, not a measurement of allocator or NVML peaks.
    """
    import ctypes

    class Location(ctypes.Structure):
        _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]

    class Flags(ctypes.Structure):
        _fields_ = [("compression", ctypes.c_ubyte), ("rdma", ctypes.c_ubyte),
                    ("usage", ctypes.c_ushort), ("reserved", ctypes.c_ubyte * 4)]

    class Properties(ctypes.Structure):
        _fields_ = [("type", ctypes.c_int), ("handles", ctypes.c_int),
                    ("location", Location), ("metadata", ctypes.c_void_p),
                    ("flags", Flags)]

    driver = ctypes.CDLL("libcuda.so.1")
    # CUDA 13.2 cuda.h: CUmemGenericAllocationHandle is unsigned long long;
    # CUmemAllocationProp_v1 is the 32-byte struct declared above.
    driver.cuMemRetainAllocationHandle.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_void_p]
    driver.cuMemGetAllocationPropertiesFromHandle.argtypes = [ctypes.POINTER(Properties), ctypes.c_ulonglong]
    driver.cuMemGetAllocationGranularity.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(Properties), ctypes.c_int]
    driver.cuMemRelease.argtypes = [ctypes.c_ulonglong]
    for name in ("cuMemRetainAllocationHandle", "cuMemGetAllocationPropertiesFromHandle",
                 "cuMemGetAllocationGranularity", "cuMemRelease"):
        getattr(driver, name).restype = ctypes.c_int
    handle = ctypes.c_ulonglong()

    def check(result):
        if result:
            raise RuntimeError(f"cannot account natural route CUDA allocation (driver error {result})")

    check(driver.cuMemRetainAllocationHandle(ctypes.byref(handle),
                                            ctypes.c_void_p(exchange.source.data_ptr())))
    try:
        properties, granularity = Properties(), ctypes.c_size_t()
        check(driver.cuMemGetAllocationPropertiesFromHandle(ctypes.byref(properties), handle))
        # Pinned Torch 70d99e998b49 CUDASymmetricMemory.cu uses RECOMMENDED
        # granularity (flag 1), not MINIMUM, for align16(payload)+signal.
        check(driver.cuMemGetAllocationGranularity(ctypes.byref(granularity),
                                                  ctypes.byref(properties), ctypes.c_int(1)))
    finally:
        check(driver.cuMemRelease(handle))
    granularity = granularity.value
    if granularity <= 0:
        raise RuntimeError("CUDA returned invalid route allocation granularity")
    align = lambda n, a: (n + a - 1) // a * a
    payload = exchange.max_rows * exchange.hidden_size * 2
    source = align(align(payload, 16) + exchange.signal_pad_bytes, granularity)
    pointers = 2 * align(exchange.size * 8, granularity)
    slots = align(4 * exchange.chunk_capacity * exchange.hidden_size, 512)
    return source + pointers + slots, granularity
