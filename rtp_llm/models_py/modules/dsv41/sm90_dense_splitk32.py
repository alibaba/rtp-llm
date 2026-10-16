"""Experimental zero-padded sub-group adapter for the existing SM90 FP8 GEMM.

This is NOT conventional split-K output reduction. Each logical g32 group is
represented by two (or four) physical g32 groups containing respectively 16
(or 8) original elements and zeros. The original scale is duplicated, never
recomputed. The existing DeepGEMM sidecar performs every matrix multiply.

The hypothesis under test is that fewer nonzero products per native FP8 WGMMA
instruction reduce its partial-dot error. This does not establish numerical
acceptance or a performance improvement. Weight payload, activation workspace,
scale storage, and physical GEMM K all grow by ``splits``. Activation packing
must remain inside any prepared/full-chain latency measurement.

Warm every row count and selected GEMM configuration before graph capture.
One adapter has reusable workspaces for at most eight row counts and is for
sequential execution only; concurrent streams/graphs need separate instances.
No production factory selects this feasibility candidate.
"""

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv41.sm90_dense import unpack_ue8m0_input_scales


@triton.jit(do_not_specialize=["ELEMENTS"])
def _pack_subgroups_bytes(
    Source, Destination, ELEMENTS, K: tl.constexpr, SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Copy bytes, not float values, so even E4M3 encodings remain unchanged."""
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    physical_k = K * SPLITS
    row = index // physical_k
    column = index % physical_k
    physical_group = column // 32
    lane = column % 32
    chunk = 32 // SPLITS
    source_column = (physical_group // SPLITS) * 32 + (physical_group % SPLITS) * chunk + lane
    value = tl.load(
        Source + row * K + source_column,
        mask=(index < ELEMENTS) & (lane < chunk), other=0,
    )
    tl.store(Destination + index, value, mask=index < ELEMENTS)


@triton.jit(do_not_specialize=["ROWS"])
def _pack_subgroup_scales(
    Source, Destination, ROWS, STRIDE_M: tl.constexpr, STRIDE_K: tl.constexpr,
    GROUPS: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr,
):
    """Duplicate FP32 scale values into fully allocated M-major TMA storage."""
    padded_m = tl.cdiv(ROWS, 4) * 4
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = index % padded_m
    physical_group = index // padded_m
    valid = index < padded_m * GROUPS * SPLITS
    value = tl.load(
        Source + row * STRIDE_M + (physical_group // SPLITS) * STRIDE_K,
        mask=valid & (row < ROWS), other=0.0,
    )
    tl.store(Destination + index, value, mask=valid)


class Sm90SplitK32Linear:
    """Exact operand repacking around the unchanged native-FP8 g32 sidecar."""

    MAX_CACHED_ROWS = 8

    def __init__(self, weight: torch.Tensor, scales: torch.Tensor, *, splits=2, config="auto"):
        from rtp_llm.models_py.kernels.cuda.sm90_dense_deepgemm import CONFIGS

        if not weight.is_cuda or torch.cuda.get_device_capability(weight.device) != (9, 0):
            raise ValueError("Sub-group FP8 candidate requires CUDA SM90")
        if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn:
            raise TypeError("Expected 2-D E4M3 weights")
        if splits not in (2, 4):
            raise ValueError("Bounded candidate supports splits=2 or splits=4")
        self.N, self.K = weight.shape
        self.splits = splits
        self.physical_K = self.K * splits
        if (self.N <= 0 or self.N % 32 or self.K <= 0 or self.K % 128
                or self.physical_K > 16384):
            raise ValueError("Requires N%32=0, K%128=0 and expanded K<=16384")
        if self.N * self.physical_K >= 2**31:
            raise ValueError("Expanded weight exceeds 32-bit indexing bound")
        if (scales.device != weight.device or scales.dtype != torch.float8_e8m0fnu
                or tuple(scales.shape) != (self.N // 32, self.K // 32)):
            raise ValueError("Expected same-device UE8M0 32x32 weight scales")
        if config != "auto" and config not in CONFIGS:
            raise ValueError(f"Unknown sidecar config {config!r}")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Construct candidate outside CUDA graph capture")
        self.config = config
        self.weight = torch.empty((self.N, self.physical_K), device=weight.device, dtype=weight.dtype)
        source = weight.contiguous()
        _pack_subgroups_bytes[(triton.cdiv(self.weight.numel(), 512),)](
            source.view(torch.uint8), self.weight.view(torch.uint8), self.weight.numel(),
            K=self.K, SPLITS=self.splits, BLOCK=512,
        )
        # UE8M0 -> FP32 is lossless, including its zero exponent and NaN code.
        self.weight_scale = scales.float().repeat_interleave(self.splits, dim=1).contiguous()
        self._workspaces = {}
        self.last_pack_kernel = None
        self.last_scale_pack_kernel = None
        self.extension_binary = None

    def _output(self, q, out):
        if (q.ndim < 1 or q.dtype != torch.float8_e4m3fn or q.shape[-1] != self.K
                or q.device != self.weight.device or not q.is_contiguous()):
            raise ValueError("Expected contiguous same-device E4M3 activations with original K")
        rows = q.numel() // self.K
        if max(rows * self.physical_K, rows * self.N) >= 2**31:
            raise ValueError("Expanded activation/output exceeds 32-bit indexing bound")
        shape = (*q.shape[:-1], self.N)
        if out is None:
            out = torch.empty(shape, device=q.device, dtype=torch.bfloat16)
        if (out.shape != shape or out.dtype != torch.bfloat16 or out.device != q.device
                or not out.is_contiguous()):
            raise ValueError("out must be contiguous same-device BF16 with matching shape")
        if any(torch._C._overlaps(out, value) for value in (q, self.weight, self.weight_scale)):
            raise ValueError("out must not overlap inputs or weights")
        return rows, out

    def _workspace(self, rows):
        cached = self._workspaces.get(rows)
        if cached is not None:
            return cached
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm this row count before CUDA graph capture")
        if len(self._workspaces) >= self.MAX_CACHED_ROWS:
            raise RuntimeError("Bounded candidate workspace cache is full; use a new instance")
        expanded_q = torch.empty((rows, self.physical_K), device=self.weight.device, dtype=self.weight.dtype)
        padded_m = (rows + 3) // 4 * 4
        storage = torch.empty((self.physical_K // 32, padded_m), device=self.weight.device, dtype=torch.float32)
        expanded_scale = storage.t()[:rows]
        cached = (expanded_q, storage, expanded_scale)
        self._workspaces[rows] = cached
        return cached

    def forward_quantized(self, q, packed_scales, out=None):
        rows, out = self._output(q, out)
        if (packed_scales.device != q.device or packed_scales.dtype != torch.int32
                or tuple(packed_scales.shape) != (rows, self.K // 128)):
            raise ValueError("Expected original packed group32 UE8M0 activation scales")
        if torch._C._overlaps(out, packed_scales):
            raise ValueError("out must not overlap input scales")
        if not rows:
            return out
        return self.prepared(q, unpack_ue8m0_input_scales(packed_scales), out)

    def prepared(self, q, float_scales, out=None):
        """Pack unchanged operands/scales, then run the existing sidecar GEMM."""
        from rtp_llm.models_py.kernels.cuda.sm90_dense_deepgemm import fp8_gemm_nt

        rows, out = self._output(q, out)
        if (float_scales.device != q.device or float_scales.dtype != torch.float32
                or tuple(float_scales.shape) != (rows, self.K // 32)
                or any(stride <= 0 for stride in float_scales.stride())):
            raise ValueError("Expected positive-stride original [M,K/32] FP32 scales")
        if torch._C._overlaps(out, float_scales):
            raise ValueError("out must not overlap input scales")
        if not rows:
            return out
        expanded_q, scale_storage, expanded_scale = self._workspace(rows)
        for buffer in (expanded_q, scale_storage):
            if any(torch._C._overlaps(buffer, value) for value in (q, float_scales, out)):
                raise ValueError("Caller tensors must not alias cached packing workspaces")
        self.last_pack_kernel = _pack_subgroups_bytes[(triton.cdiv(expanded_q.numel(), 512),)](
            q.view(torch.uint8), expanded_q.view(torch.uint8), expanded_q.numel(),
            K=self.K, SPLITS=self.splits, BLOCK=512,
        )
        self.last_scale_pack_kernel = _pack_subgroup_scales[(triton.cdiv(scale_storage.numel(), 256),)](
            float_scales, scale_storage, rows,
            STRIDE_M=float_scales.stride(0), STRIDE_K=float_scales.stride(1),
            GROUPS=self.K // 32, SPLITS=self.splits, BLOCK=256,
        )
        fp8_gemm_nt(expanded_q, expanded_scale, self.weight, self.weight_scale,
                    out.view(rows, self.N), config=self.config)
        if self.extension_binary is None:
            # The call above has already compiled/warmed the unchanged sidecar.
            # Expose its real binary for SASS audit, not the Triton pack kernels.
            from rtp_llm.models_py.kernels.cuda.sm90_dense_deepgemm import _extension

            self.extension_binary = _extension(q.device.index, self.N, self.physical_K).__file__
        return out
