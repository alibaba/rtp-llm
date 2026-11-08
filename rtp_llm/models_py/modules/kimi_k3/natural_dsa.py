"""Preparation contracts shared by the natural-outer decoder and DSA."""

import torch
from functools import lru_cache


class Fp8NaturalProjection(torch.nn.Module):
    """Project aligned output rows using the loaded FP8 weight and scales.

    The KV slice includes the fused input's original N padding. Only the
    logical result is returned. Packed UE8M0 scales keep their MN-major stride
    and the same row offset as the weight; no dequantization is involved.
    """

    def __init__(self, projection, start, end, logical_width):
        super().__init__()
        if not (0 <= start < end <= projection.weight.shape[0]
                and start % 128 == end % 128 == 0
                and 0 < logical_width <= end - start):
            raise ValueError("FP8 natural projection must preserve complete output blocks")
        if projection.bias is not None or projection.weight.dtype != torch.float8_e4m3fn:
            raise ValueError("FP8 natural projection requires bias-free E4M3 weights")
        self.weight = projection.weight[start:end]
        scales = projection.weight_scales
        if scales.dtype == torch.int32:
            self.weight_scales = scales[start:end]
        elif scales.dtype == torch.float32:
            self.weight_scales = scales[start // 128:end // 128]
        else:
            raise ValueError("Unsupported FP8 natural weight scales")
        self.logical_width = logical_width
        self._quantize_input = projection._deepgemm_linear.quantize_input
        self._disable_ue8m0_cast = not projection.scale_ue8m0

    def forward(self, hidden):
        return self.forward_quantized(*self.quantize_input(hidden))

    def quantize_input(self, hidden):
        return self._quantize_input(hidden)

    def forward_quantized(self, values, scales):
        """Consume an existing activation encoding without requantizing it."""
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import fp8_gemm_nt

        output = torch.empty((values.shape[0], self.weight.shape[0]),
                             dtype=torch.bfloat16, device=values.device)
        fp8_gemm_nt((values, scales), (self.weight, self.weight_scales), output,
                    disable_ue8m0_cast=self._disable_ue8m0_cast)
        return output[:, :self.logical_width]


class NaturalPairedProjection(torch.nn.Module):
    """Pack two compatible output intervals once for a single GEMM."""

    def __init__(self, q, gate):
        super().__init__()
        self.q_width = q.weight.shape[0]
        self.gate_width = gate.weight.shape[0]
        self.fp8 = isinstance(q, Fp8NaturalProjection)
        self.nvfp4 = None
        if getattr(q, "dynamic_input_scale", False):
            from rtp_llm.models.kimi_k3.nvfp4_tensor import repack_nvfp4_projections

            self.nvfp4 = repack_nvfp4_projections([
                (q, 0, self.q_width), (gate, 0, self.gate_width),
            ])
            self.weight = self.nvfp4.weight
            return
        # Preserve the backend's scale orientation, especially packed UE8M0's
        # MN-major stride. All packing is model initialization work.
        def pack(left, right):
            rows = left.shape[0] + right.shape[0]
            shape = (rows, left.shape[1])
            out = (torch.empty_strided(shape, (1, rows), dtype=left.dtype, device=left.device)
                   if left.stride(0) == 1 else left.new_empty(shape))
            if left.dtype == torch.float8_e4m3fn:
                out[:left.shape[0]].view(torch.uint8).copy_(left.view(torch.uint8))
                out[left.shape[0]:].view(torch.uint8).copy_(right.view(torch.uint8))
            else:
                out[:left.shape[0]].copy_(left)
                out[left.shape[0]:].copy_(right)
            return out
        self.weight = pack(q.weight, gate.weight)
        if self.fp8:
            if (not isinstance(gate, Fp8NaturalProjection)
                    or q.weight_scales.dtype != gate.weight_scales.dtype
                    or q.weight_scales.stride(0) != gate.weight_scales.stride(0)):
                raise ValueError("paired FP8 projections require matching scale layouts")
            self.weight_scales = pack(q.weight_scales, gate.weight_scales)
            self._quantize_input = q.quantize_input
            self._disable_ue8m0_cast = q._disable_ue8m0_cast

    def forward_quantized(self, values, scales):
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import fp8_gemm_nt

        out = torch.empty((values.shape[0], self.q_width + self.gate_width),
                          device=values.device, dtype=torch.bfloat16)
        fp8_gemm_nt((values, scales), (self.weight, self.weight_scales), out,
                    disable_ue8m0_cast=self._disable_ue8m0_cast)
        return out.split((self.q_width, self.gate_width), dim=-1)

    def forward_nvfp4(self, activation):
        out = self.nvfp4.forward_quantized(activation)
        return out.split((self.q_width, self.gate_width), dim=-1)

    def forward(self, hidden):
        if self.nvfp4 is not None:
            out = self.nvfp4(hidden)
        elif self.fp8:
            return self.forward_quantized(*self._quantize_input(hidden))
        else:
            out = torch.nn.functional.linear(hidden, self.weight)
        return out.split((self.q_width, self.gate_width), dim=-1)


def split_natural_projection(projection, q_width, kv_width, gate_width, padding):
    """Keep Q/KV/gate views aligned with the fused projection's physical ABI."""
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import CudaFp8GEMMLinear

    kv_end = q_width + kv_width
    gate_start = kv_end + padding
    if projection.weight.shape[0] != gate_start + gate_width:
        raise ValueError("natural projection widths disagree with the fused input")
    if getattr(projection, "dynamic_input_scale", False):
        if padding:
            raise ValueError("natural NVFP4 input does not use the FP8 KV alignment gap")
        from rtp_llm.models.kimi_k3.nvfp4_tensor import repack_nvfp4_projections

        return tuple(repack_nvfp4_projections([(projection, start, end)]) for start, end in (
            (0, q_width), (q_width, kv_end), (gate_start, gate_start + gate_width),
        ))
    if isinstance(projection, CudaFp8GEMMLinear):
        return (
            Fp8NaturalProjection(projection, 0, q_width, q_width),
            Fp8NaturalProjection(projection, q_width, gate_start, kv_width),
            Fp8NaturalProjection(projection, gate_start, gate_start + gate_width, gate_width),
        )
    if not isinstance(projection, CudaF16Linear) or padding:
        raise ValueError("Unsupported natural input projection")
    weight = projection.weight
    return (CudaF16Linear(weight[:q_width].T),
            CudaF16Linear(weight[q_width:kv_end].T),
            CudaF16Linear(weight[gate_start:].T))


@lru_cache(maxsize=16)
def _shared_restore_reference(plan_type, lengths, size):
    plan = plan_type(lengths, size, 0)
    expected = torch.empty(plan.physical_rows, dtype=torch.long)
    for rank in range(plan.size):
        incoming = plan_type(plan.lengths, plan.size, rank)
        for segment in incoming.to_zigzag:
            natural_start = segment.peer * plan.local_rows + segment.source_offset
            zigzag_start = rank * plan.local_rows + segment.destination_offset
            expected[natural_start:natural_start + segment.rows] = torch.arange(
                zigzag_start, zigzag_start + segment.rows)
    padding = torch.cat([
        torch.arange(padded) < length
        for length, padded in zip(plan.lengths, plan.padded_lengths)
    ])
    return expected, padding


def validate_shared_restore(layout, plan, cache=None):
    """Validate each current mapping, without a device-to-host synchronization.

    Only expected geometry is cached. Native producers can rewrite a tensor
    without advancing its Python version, so input identity never skips checks.
    """
    if not plan.lengths or any(n <= 0 for n in plan.lengths):
        raise ValueError("natural outer Prefill requires nonempty real requests")
    if (layout.physical_rows != plan.physical_rows or layout.local_rows != plan.local_rows
            or layout.restore.numel() != plan.physical_rows
            or layout.padding_mask.numel() != plan.physical_rows):
        raise ValueError("shared restore extent differs from natural routing plan")
    key = (type(plan), tuple(plan.lengths), plan.size, str(layout.restore.device))
    expected, padding = _shared_restore_reference(key[0], key[1], key[2])
    if layout.restore.is_cuda:
        from rtp_llm.models_py.triton_kernels.common.validate_zigzag import validate_zigzag

        cache = {} if cache is None else cache
        tensors = cache.pop(key, None)
        if tensors is None:
            device = layout.restore.device
            tensors = (expected.to(device, non_blocking=True), padding.to(device, non_blocking=True),
                       torch.empty((plan.physical_rows + 4095) // 4096, dtype=torch.bool, device=device))
        cache[key] = tensors
        while len(cache) > 8:
            cache.pop(next(iter(cache)))
        validate_zigzag(layout.restore, layout.padding_mask, *tensors)
        return
    if not torch.equal(layout.restore, expected):
        raise ValueError("shared restore permutation differs from natural routing plan")
    if not torch.equal(layout.padding_mask, padding):
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
