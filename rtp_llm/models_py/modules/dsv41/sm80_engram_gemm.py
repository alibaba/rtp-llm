"""sm80 reference WKV projection with host-resident weights.

The Engram contract keeps every Engram checkpoint tensor in host RAM with no
persistent GPU mirror. The dequantized BF16 WKV weight therefore lives in
pinned host memory; this kernel reads it directly over UVA (the same
mechanism the Engram table prefetch kernels use) instead of cuBLAS, which
requires device-resident operands.

Compute is weight-bandwidth bound: each call streams N*K*2 bytes over PCIe
(~314 MB -> ~16-40 ms), which is amortized across the chunk's M rows.
num_stages=1 keeps Triton off cp.async pipelining (host-memory sources).
The launch is CUDA-graph capturable: pointers are stable for the process
lifetime and replays re-read the immutable host weights.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _linear_host_weight_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    sam,
    sak,
    sbn,
    sbk,
    scm,
    scn,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    # M-inner ordering: consecutive programs share the same B tile rows, so
    # the (dominant) host-weight traffic gets L2 reuse across the M sweep.
    pid_m = pid % grid_m
    pid_n = pid // grid_m

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    mask_m = rm < M
    mask_n = rn < N

    A_ptr = A + rm[:, None] * sam + rk[None, :] * sak
    # B is [N, K] row-major in pinned host memory; rk is the contiguous axis.
    B_ptr = B + rn[:, None] * sbn + rk[None, :] * sbk

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, K, BK):
        mask_k = (k + rk) < K
        a = tl.load(A_ptr, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(B_ptr, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc += tl.dot(a, tl.trans(b))
        A_ptr += BK * sak
        B_ptr += BK * sbk

    c = acc.to(tl.bfloat16)
    C_ptr = C + rm[:, None] * scm + rn[None, :] * scn
    tl.store(C_ptr, c, mask=mask_m[:, None] & mask_n[None, :])


def linear_host_weight(x: torch.Tensor, w_host: torch.Tensor) -> torch.Tensor:
    """``x [M,K] (cuda, bf16) @ w_host [N,K]^T (pinned cpu, bf16) -> [M,N]``."""
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    if not x.is_cuda:
        raise ValueError("activations must be device-resident")
    if w_host.device.type != "cpu" or not w_host.is_pinned():
        raise ValueError("host weight must be pinned CPU memory for UVA reads")
    if not w_host.is_contiguous():
        w_host = w_host.contiguous().pin_memory()
    M, K = x.shape
    N = w_host.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    if M == 0 or N == 0:
        return out
    x = x.contiguous()
    BM = 32 if M <= 256 else 64
    BN, BK = 128, 64
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _linear_host_weight_kernel[grid](
        x,
        w_host,
        out,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w_host.stride(0),
        w_host.stride(1),
        out.stride(0),
        out.stride(1),
        BM=BM,
        BN=BN,
        BK=BK,
        num_warps=4,
        num_stages=1,
    )
    return out
