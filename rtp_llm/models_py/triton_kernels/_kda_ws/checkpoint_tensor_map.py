"""Build the immutable CUDA tensor map for a KDA checkpoint pool."""

import ctypes
from functools import lru_cache
from threading import Lock

import torch

_MAP_BYTES = 128
_MAP_ALIGNMENT = 128
_TENSOR_MAP_FLOAT32 = 7
_TENSOR_MAP_SWIZZLE_128B = 3
_CACHE_LOCK = Lock()


@lru_cache(maxsize=1)
def _encode_tiled():
    encode = ctypes.CDLL("libcuda.so.1").cuTensorMapEncodeTiled
    encode.argtypes = (
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    )
    encode.restype = ctypes.c_int
    return encode


def _checkpoint_map_bytes(address: int, slots: int, heads: int, page_stride: int) -> bytes:
    raw_map = ctypes.create_string_buffer(_MAP_BYTES + _MAP_ALIGNMENT - 1)
    map_address = ctypes.addressof(raw_map)
    map_address += (-map_address) % _MAP_ALIGNMENT

    # Coordinates from fastest to slowest are (v, k % 4, k // 4, head, slot).
    # Each tile fixes a K residue and spans one V stripe, letting the hardware
    # swizzle use k // 4 to spread the scan's eight K lanes across banks.
    dimensions = (ctypes.c_uint64 * 5)(128, 4, 32, heads, slots)
    strides = (ctypes.c_uint64 * 4)(512, 2048, 65536, page_stride * 4)
    tile = (ctypes.c_uint32 * 5)(32, 1, 32, 1, 1)
    element_strides = (ctypes.c_uint32 * 5)(1, 1, 1, 1, 1)
    error = _encode_tiled()(
        map_address,
        _TENSOR_MAP_FLOAT32,
        5,
        address,
        dimensions,
        strides,
        tile,
        element_strides,
        0,  # CU_TENSOR_MAP_INTERLEAVE_NONE
        _TENSOR_MAP_SWIZZLE_128B,
        0,  # CU_TENSOR_MAP_L2_PROMOTION_NONE
        0,  # CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    )
    if error != 0:
        raise RuntimeError(f"cuTensorMapEncodeTiled failed for the KDA checkpoint pool: {error}")
    return ctypes.string_at(map_address, _MAP_BYTES)


def checkpoint_tensor_map(state: torch.Tensor) -> torch.Tensor:
    """Return a 128-byte device map owned by the bound FP32 [K,V] state tensor."""
    if not state.is_cuda or state.dtype != torch.float32 or state.ndim != 4:
        raise ValueError("KDA TMA requires CUDA FP32 [slot, head, K, V] checkpoint pages")
    slots, heads, k, v = state.shape
    address, stride = state.data_ptr(), state.stride()
    if (
        min(slots, heads) < 1
        or (k, v) != (128, 128)
        or stride[1:] != (128 * 128, 128, 1)
        or stride[0] < heads * 128 * 128
        or stride[0] % 4
        or address % 16
    ):
        raise ValueError("KDA TMA requires 16-byte-aligned, pitched [K,V] checkpoint pages")

    key = (state.device, address, tuple(state.shape), stride)
    cached = getattr(state, "_kda_checkpoint_tensor_map", None)
    if cached is not None and cached[0] == key:
        return cached[1]

    with _CACHE_LOCK:
        cached = getattr(state, "_kda_checkpoint_tensor_map", None)
        if cached is not None and cached[0] == key:
            return cached[1]

        with torch.cuda.device(state.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up the KDA TMA checkpoint pool before CUDA Graph capture")
            image = _checkpoint_map_bytes(address, slots, heads, stride[0])
            descriptor = torch.frombuffer(bytearray(image), dtype=torch.int32).to(state.device)

        if descriptor.data_ptr() % _MAP_ALIGNMENT:
            raise RuntimeError("The KDA TMA descriptor must be 128-byte aligned")
        state._kda_checkpoint_tensor_map = (key, descriptor)
        return descriptor
