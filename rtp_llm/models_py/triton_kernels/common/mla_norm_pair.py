"""One launch for the Q/CKV norms consuming the fused MLA projection."""

import torch
import triton as tr
import triton.language as tl
from rtp_llm.models_py import prefill_optimization as _prefill_optimization
from rtp_llm.models_py.kernels.cuda import prefill_cpu_ops as _cpu_ops
from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
)
from rtp_llm.models_py.triton_kernels.common import fast_launch as _cpu_launch
from rtp_llm.models_py.triton_kernels.common.fused_strided_rmsnorm import (
    _fused_strided_rmsnorm_singlepass_kernel,
)

_kernels = {}
_CPU_PATH = _prefill_optimization.ENABLED


@tr.jit
def _pair(
    Q,
    KV,
    QW,
    KW,
    QB,
    QF,
    QS,
    KB,
    QE,
    KE,
    QSTRIDE,
    KSTRIDE,
    SST,
    SSG,
    H: tl.constexpr,
    KH: tl.constexpr,
    FP8: tl.constexpr,
    UE8: tl.constexpr,
):
    _fused_strided_rmsnorm_singlepass_kernel(
        Q,
        QW,
        QB,
        QF,
        QS,
        H,
        QE,
        448.0,
        -448.0,
        QSTRIDE,
        H,
        H,
        SST,
        SSG,
        H,
        128,
        True,
        FP8,
        UE8,
    )
    # Preserve the existing H=512, two-warp kernel's reduction: eight
    # consecutive BF16 squares per lane, then a 64-lane butterfly. Letting
    # the four-warp compiler choose this layout changes the sum ordering.
    token = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, 64)
    acc = tl.full((64,), 0.0, tl.float32)
    for i in tl.static_range(8):
        x = tl.load(KV + token * KSTRIDE + lane * 8 + i).to(tl.float32)
        acc = acc + x * x
    inv = tl.rsqrt(tl.sum(acc, 0) / 512 + KE)
    col = tl.arange(0, 512)
    x = tl.load(KV + token * KSTRIDE + col).to(tl.float32)
    w = tl.load(KW + col).to(tl.float32)
    tl.store(KB + token * 512 + col, (x * inv * w).to(tl.bfloat16))


def try_mla_norm_pair(q, kv, qw, kw, q_eps, kv_eps, *, fp8=True, scale_ue8m0=True):
    if not (
        q.is_cuda
        and q.ndim == kv.ndim == 2
        and q.shape[1] == 2048
        and kv.shape == (q.shape[0], 512)
        and q.stride(1) == kv.stride(1) == 1
        and qw.shape == (2048,)
        and kw.shape == (512,)
        and q.stride(0) % 8 == kv.stride(0) % 8 == 0
        and all(t.data_ptr() % 16 == 0 for t in (q, kv, qw, kw))
        and qw.is_contiguous()
        and kw.is_contiguous()
        and all(
            t.device == q.device and t.dtype == torch.bfloat16 and not t.requires_grad
            for t in (q, kv, qw, kw)
        )
    ):
        return None
    n = q.shape[0]
    allocator = _cpu_ops.native() if _cpu_ops.NATIVE_ALLOC and fp8 else None
    if allocator is not None:
        qb, qf, qs, kb = allocator.norm_outputs(
            n, 2048, 512, q.device.index, scale_ue8m0, True
        )
    else:
        qb = torch.empty((n, 2048), device=q.device, dtype=torch.bfloat16)
        kb = torch.empty((n, 512), device=q.device, dtype=torch.bfloat16)
    if fp8 and allocator is None:
        qf = torch.empty((n, 2048), device=q.device, dtype=torch.float8_e4m3fn)
        qs = create_per_token_group_quant_fp8_output_scale(
            (n, 2048),
            q.device,
            group_size=128,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=scale_ue8m0,
        )
    elif not fp8:
        qf, qs = qb, qb
    if n:
        args = (
            q,
            kv,
            qw,
            kw,
            qb,
            qf,
            qs,
            kb,
            q_eps,
            kv_eps,
            q.stride(0),
            kv.stride(0),
            qs.stride(0),
            qs.stride(1),
            2048,
            512,
            fp8,
            scale_ue8m0,
        )
        if _CPU_PATH:
            # The support gate already fixes input types and 16-byte alignment;
            # all outputs are freshly allocated and aligned. Retain every scalar
            # specialization and scale layout, without re-inspecting 8 tensors.
            key = (
                q.device.index,
                q_eps,
                kv_eps,
                args[10],
                args[11],
                args[12],
                args[13],
                fp8,
                scale_ue8m0,
            )
        else:
            key = tuple(
                (
                    (t.dtype, t.device.index, t.data_ptr() % 16)
                    if isinstance(t, torch.Tensor)
                    else t
                )
                for t in args
            )
        kernel = _kernels.get(key)
        if kernel is None:
            kernel = _pair[(n, 1, 1)](*args, num_warps=4)
            if len(_kernels) >= 128:
                _kernels.pop(next(iter(_kernels)))
            _kernels[key] = kernel
        else:
            _cpu_launch.launch_compiled(kernel, (n, 1, 1), args)
    return qb, (qf if fp8 else None), (qs if fp8 else None), kb
