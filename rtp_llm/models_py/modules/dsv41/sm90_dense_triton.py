# SPDX-License-Identifier: Apache-2.0
# Copyright 2023-2026 SGLang Team
"""Isolated SGLang Triton g32 Dense candidates, not production dispatch.

Kernel bodies are ported from SGLang revision
27b68303d043ddbddf9861c55c26bc8617ea7c35 without arithmetic changes.
Only launch specialization metadata changes: M and reduction ELEMENTS are
runtime values. See kernels/cuda/sm90_dense_triton/PROVENANCE.md.
"""

from collections.abc import Mapping

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.modules.dsv41.sm90_dense import unpack_ue8m0_input_scales


@triton.jit(do_not_specialize=["M"])
def _w8a8_block_fp8_matmul(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    needs_masking: tl.constexpr,
):
    """Triton-accelerated function used to perform linear operations (dot
    product) on input tensors `A` and `B` with block-wise quantization, and store the result in output
    tensor `C`.
    """

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n
    n_tiles_k_per_group_k = group_k // BLOCK_SIZE_K

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if needs_masking:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

        a_s = tl.load(As_ptrs)
        b_s = tl.load(Bs_ptrs)

        scale_step_k = tl.where((k + 1) % n_tiles_k_per_group_k == 0, 1, 0)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        As_ptrs += scale_step_k * stride_As_k
        Bs_ptrs += scale_step_k * stride_Bs_k

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit(do_not_specialize=["M"])
def _w8a8_block_fp8_matmul_hopper(
    # Pointers to inputs and output
    A,
    B,
    C,
    As,
    Bs,
    # Shape for matmul
    M,
    N,
    K,
    # Block size for block-wise quantization
    group_n,
    group_k,
    # Stride for inputs and output
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_As_m,
    stride_As_k,
    stride_Bs_k,
    stride_Bs_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    needs_masking: tl.constexpr,
    SWAP_AB: tl.constexpr = False,
    SPLIT_K: tl.constexpr = 1,
):
    pid = tl.program_id(axis=0)
    split = tl.program_id(axis=1)
    tiles_per_split = tl.cdiv(tl.cdiv(K, BLOCK_SIZE_K), SPLIT_K)
    first_tile = split * tiles_per_split
    C += split * M * N
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    As_ptrs = As + offs_am * stride_As_m
    offs_bsn = offs_bn // group_n
    Bs_ptrs = Bs + offs_bsn * stride_Bs_n
    n_tiles_k_per_group_k = group_k // BLOCK_SIZE_K

    a_ptrs += first_tile * BLOCK_SIZE_K * stride_ak
    b_ptrs += first_tile * BLOCK_SIZE_K * stride_bk
    As_ptrs += (first_tile // n_tiles_k_per_group_k) * stride_As_k
    Bs_ptrs += (first_tile // n_tiles_k_per_group_k) * stride_Bs_k

    # Small-M Hopper configs transpose the MMA so the weight tile occupies M.
    if SWAP_AB:
        accumulator = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(
        first_tile, tl.minimum(first_tile + tiles_per_split, tl.cdiv(K, BLOCK_SIZE_K))
    ):
        if needs_masking:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        else:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)

        a_s = tl.load(As_ptrs)
        b_s = tl.load(Bs_ptrs)

        scale_step_k = tl.where((k + 1) % n_tiles_k_per_group_k == 0, 1, 0)
        if SWAP_AB:
            accumulator += (
                tl.dot(tl.trans(b), tl.trans(a)) * b_s[:, None] * a_s[None, :]
            )
        else:
            accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        As_ptrs += scale_step_k * stride_As_k
        Bs_ptrs += scale_step_k * stride_Bs_k

    if SWAP_AB:
        accumulator = tl.trans(accumulator)

    if C.dtype.element_ty == tl.bfloat16:
        c = accumulator.to(tl.bfloat16)
    elif C.dtype.element_ty == tl.float16:
        c = accumulator.to(tl.float16)
    else:
        c = accumulator.to(tl.float32)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit(do_not_specialize=["ELEMENTS"])
def _reduce_block_fp8_split_k(
    Parts, Out, ELEMENTS, SPLITS: tl.constexpr, BLOCK: tl.constexpr
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, SPLITS)
    values = tl.load(
        Parts + splits[:, None] * ELEMENTS + offsets[None, :],
        offsets[None, :] < ELEMENTS,
        0.0,
    )
    tl.store(Out + offsets, tl.sum(values, axis=0), offsets < ELEMENTS)


def _config(block_m, block_n, stages=3, *, swap=False, splits=1):
    return dict(
        BLOCK_SIZE_M=block_m, BLOCK_SIZE_N=block_n, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=32, num_warps=4, num_stages=stages,
        SWAP_AB=swap, SPLIT_K=splits,
    )


# Explicit bounded candidates, not an auto-tuning or performance claim. M32
# variants change only an existing generic kernel's tile metadata; the compiler
# may lower its FP8 dot to FP16 MMA rather than native FP8 (inspect PTX).
TRITON_CONFIGS = {
    "generic_m64n32": _config(64, 32),  # upstream block32 fallback
    "generic_m32n64": _config(32, 64),
    "generic_m32n128": _config(32, 128),
    "swap_m16n64": _config(16, 64, 4, swap=True),
    "swap_m16n64_split8": _config(16, 64, 4, swap=True, splits=8),
    "swap_m16n64_split16": _config(16, 64, 3, swap=True, splits=16),
}


def _resolve_config(config):
    if config is None:
        config = "generic_m64n32"
    if isinstance(config, str):
        if config not in TRITON_CONFIGS:
            raise ValueError(f"Unknown Triton group32 config: {config!r}")
        return dict(TRITON_CONFIGS[config])
    if isinstance(config, Mapping):
        result = dict(config)
        result.setdefault("SWAP_AB", False)
        result.setdefault("SPLIT_K", 1)
        if result not in TRITON_CONFIGS.values():
            raise ValueError("Only the explicitly exposed Triton configurations are supported")
        return result
    raise TypeError("config must be a TRITON_CONFIGS name or configuration mapping")


class Sm90TritonMXFP8Linear:
    """Candidate using original FP8 payloads and losslessly decoded g32 scales.

    Input quantization is owned by RTP. There is no BF16 input quantizer here
    and no re-quantization of weights. ``prepared`` excludes scale conversion;
    split-K still includes allocation, partial GEMM and reduction. A compiler
    handle is exposed for instruction audit outside timed forwards.
    """

    def __init__(self, weight: torch.Tensor, scales: torch.Tensor, config=None):
        if not weight.is_cuda or torch.cuda.get_device_capability(weight.device) != (9, 0):
            raise ValueError("Triton group32 candidate requires a CUDA SM90 device")
        if weight.ndim != 2 or weight.dtype != torch.float8_e4m3fn:
            raise ValueError("Expected two-dimensional E4M3 weights")
        self.N, self.K = weight.shape
        if self.N <= 0 or self.K <= 0 or self.K % 32:
            raise ValueError("Expected positive N and positive K divisible by 32")
        if (scales.device != weight.device or scales.dtype != torch.float8_e8m0fnu
                or tuple(scales.shape) != ((self.N + 31) // 32, self.K // 32)):
            raise ValueError("Expected same-device UE8M0 32x32 block weight scales")
        self.weight = weight.contiguous()
        self.weight_scale = scales.float().contiguous()
        self.config = _resolve_config(config)
        self.last_compiled_kernel = None
        self.last_reduce_kernel = None

    def _output(self, q, out):
        if (q.ndim < 1 or q.dtype != torch.float8_e4m3fn
                or q.shape[-1] != self.K or q.device != self.weight.device
                or not q.is_contiguous()):
            raise ValueError("Expected same-device contiguous E4M3 activations with matching K")
        rows = q.numel() // self.K
        shape = (*q.shape[:-1], self.N)
        if out is None:
            out = torch.empty(shape, dtype=torch.bfloat16, device=q.device)
        elif (tuple(out.shape) != shape or out.dtype != torch.bfloat16
              or out.device != q.device or not out.is_contiguous()):
            raise ValueError("out must be same-device contiguous BF16 with matching shape")
        if any(torch._C._overlaps(out, tensor)
               for tensor in (q, self.weight, self.weight_scale)):
            raise ValueError("output must not overlap any GEMM input")
        return rows, out

    def forward_quantized(self, q: torch.Tensor, packed: torch.Tensor, out=None):
        """Consume RTP packed UE8M0 scales, including conversion in this call."""
        rows, out = self._output(q, out)
        if (packed.dtype != torch.int32 or packed.device != q.device
                or tuple(packed.shape) != (rows, (self.K + 127) // 128)):
            raise ValueError("Expected packed per-token group32 UE8M0 activation scales")
        if torch._C._overlaps(out, packed):
            raise ValueError("output must not overlap packed activation scales")
        if rows == 0:
            return out
        # Preserve exponent bytes including 0/255 semantics. Slicing only drops
        # packed tail padding groups, never re-estimates or rounds scales.
        float_scales = unpack_ue8m0_input_scales(packed)[:, :self.K // 32]
        return self.prepared(q, float_scales, out)

    def prepared(self, q: torch.Tensor, float_scales: torch.Tensor, out, config=None):
        """Execute with FP32 [M,K/32] scales in either positive-stride layout."""
        rows, out = self._output(q, out)
        if (float_scales.dtype != torch.float32 or float_scales.device != q.device
                or tuple(float_scales.shape) != (rows, self.K // 32)
                or any(stride <= 0 for stride in float_scales.stride())):
            raise ValueError("Expected same-device positive-stride FP32 [M,K/32] scales")
        if torch._C._overlaps(out, float_scales):
            raise ValueError("output must not overlap activation scales")
        if rows == 0:
            return out
        selected = dict(self.config) if config is None else _resolve_config(config)
        split_k = selected["SPLIT_K"]
        if max(rows * self.K, self.N * self.K, split_k * rows * self.N) >= 2**31:
            raise ValueError("Candidate requires fewer than 2^31 indexed elements")
        hopper = selected["SWAP_AB"] or split_k > 1
        if hopper:
            kernel = _w8a8_block_fp8_matmul_hopper
        else:
            kernel = _w8a8_block_fp8_matmul
            selected.pop("SWAP_AB")
            selected.pop("SPLIT_K")
        a, c = q.view(rows, self.K), out.view(rows, self.N)
        partials = (torch.empty((split_k, rows, self.N), dtype=torch.float32, device=q.device)
                    if split_k > 1 else c)
        blocks = triton.cdiv(rows, selected["BLOCK_SIZE_M"]) * triton.cdiv(
            self.N, selected["BLOCK_SIZE_N"])
        grid = (blocks, split_k) if hopper else (blocks,)
        self.last_compiled_kernel = kernel[grid](
            a, self.weight, partials, float_scales, self.weight_scale,
            rows, self.N, self.K, 32, 32,
            a.stride(0), a.stride(1), self.weight.stride(1), self.weight.stride(0),
            c.stride(0), c.stride(1), float_scales.stride(0), float_scales.stride(1),
            self.weight_scale.stride(1), self.weight_scale.stride(0),
            **selected, needs_masking=False,
        )
        self.last_reduce_kernel = None
        if split_k > 1:
            self.last_reduce_kernel = _reduce_block_fp8_split_k[(triton.cdiv(rows * self.N, 256),)](
                partials, c, rows * self.N, split_k, 256
            )
        return out


__all__ = ["Sm90TritonMXFP8Linear", "TRITON_CONFIGS"]
