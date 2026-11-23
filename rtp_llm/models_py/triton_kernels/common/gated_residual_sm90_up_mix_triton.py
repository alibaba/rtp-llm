"""SM90 fused Hyper-Connection BF16 up-projection + 4-branch sigmoid mix-reduce.

Eliminates the intermediate ``[M, 10240]`` BF16 ``mix_logits`` HBM write/read
and extra kernel launch between ``F.linear(mix, bf16_up)`` and
``fused_mix_reduce`` across decode, MTP draft, and target-verify batch sizes
``1 <= M <= 128`` while preserving exact BF16 gate/product rounding boundaries
(larger prefill/verify shapes defer to Hopper TMA cuBLASLt ``nvjet``).
"""

import os

import torch
import triton
import triton.language as tl

from rtp_llm.models_py.utils.device_capability import (
    cached_device_capability,
    sm90_pdl_enabled,
)


@triton.jit(do_not_specialize=["M"])
def _sm90_bf16_up_mix_kernel(
    X,
    W,
    NORM,
    Y,
    M,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    PDL: tl.constexpr = False,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    mask_m = rm < M

    packed = tl.arange(0, 4 * BN)
    col_4b = pid_n * BN + (packed % BN) + (packed // BN) * 2560
    acc = tl.zeros((BM, 4 * BN), dtype=tl.float32)
    if PDL:
        tl.extra.cuda.gdc_wait()
    for kb in tl.static_range(320 // BK):
        kk = kb * BK + rk
        a = tl.load(
            X + rm[:, None] * 320 + kk[None, :],
            mask=mask_m[:, None],
            other=0.0,
        )
        b = tl.load(W + col_4b[None, :] * 320 + kk[:, None])
        acc = tl.dot(a, b, acc)

    logits = acc.to(tl.bfloat16).to(tl.float32)
    gate = (1.0 / (1.0 + tl.exp(-logits))).to(tl.bfloat16).to(tl.float32)
    norm = tl.load(
        NORM + rm[:, None] * 10240 + col_4b[None, :],
        mask=mask_m[:, None],
        other=0.0,
    ).to(tl.float32)
    product = (gate * norm).to(tl.bfloat16).to(tl.float32).reshape(BM, 4, BN)
    branch_ids = tl.arange(0, 4)
    output = tl.zeros((BM, BN), dtype=tl.float32)
    for branch in tl.static_range(4):
        output += tl.sum(
            tl.where(branch_ids[None, :, None] == branch, product, 0.0), 1
        )
    if PDL:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(
        Y + rm[:, None] * 2560 + rn[None, :],
        (output * 0.25).to(tl.bfloat16),
        mask=mask_m[:, None],
    )


def try_sm90_bf16_up_mix(x, bf16_up, normed, branches):
    if os.getenv("RTP_QWEN4_DISABLE_SM90_BF16_UP_MIX", "0") == "1":
        return None
    if not (
        cached_device_capability(x.device) == (9, 0)
        and x.is_cuda
        and torch.version.hip is None
        and x.ndim >= 2
        and x.shape[-1] == 320
        and 1 <= x.numel() // 320 <= 128
        and branches == 4
        and x.dtype == torch.bfloat16
        and tuple(bf16_up.shape) == (10240, 320)
        and bf16_up.dtype == torch.bfloat16
        and normed.shape == (*x.shape[:-1], 10240)
        and normed.dtype == torch.bfloat16
        and x.device == bf16_up.device == normed.device
        and all(
            t.is_contiguous() and not t.requires_grad
            for t in (x, bf16_up, normed)
        )
    ):
        return None

    rows_2d = x.reshape(-1, 320)
    m = rows_2d.shape[0]
    output = x.new_empty((m, 2560))
    pdl = sm90_pdl_enabled(x.device)
    if m <= 64:
        bm, bn, stages = 16, 32, 2
    else:
        bm, bn, stages = 64, 32, 3
    _sm90_bf16_up_mix_kernel[(triton.cdiv(m, bm), 2560 // bn)](
        rows_2d,
        bf16_up,
        normed.reshape(-1, 10240),
        output,
        m,
        BM=bm,
        BN=bn,
        BK=64,
        PDL=pdl,
        num_warps=4,
        num_stages=stages,
        enable_fp_fusion=False,
        launch_pdl=pdl,
    )
    return output.reshape(*x.shape[:-1], 2560)
