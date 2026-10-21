from dataclasses import dataclass
import importlib

import torch
import triton
import triton.language as tl


@triton.jit
def _blockwise_tf32(x_ptr, w_ptr, qs_ptr, out_ptr, scale_const, M,
                    BLOCK_M: tl.constexpr = 32, BLOCK_K: tl.constexpr = 128):
    rows = tl.program_id(0).to(tl.int64)*BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    heads = tl.arange(0, 32)
    acc = tl.zeros((BLOCK_M, 32), tl.float32)
    for k0 in range(0, 6144, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(x_ptr + rows[:, None]*6144 + kk[None, :], rows[:, None] < M, other=0).to(tl.float32)
        w = tl.load(w_ptr + heads[:, None]*6144 + kk[None, :]).to(tl.float32)
        local = tl.dot(x, tl.trans(w), out_dtype=tl.float32, input_precision="tf32")
        # TF32x3 adds each K128 subtotal separately; prevent dot/add contraction.
        local = tl.inline_asm_elementwise("mov.b32 $0, $1;", constraints="=f,f", args=[local],
                                          dtype=tl.float32, is_pure=False, pack=1)
        acc = local + acc
    qs = tl.load(qs_ptr + rows[:, None]*32 + heads[None, :], rows[:, None] < M, other=0)
    tl.store(out_ptr + rows[:, None]*32 + heads[None, :], acc*qs*scale_const, rows[:, None] < M)


def _version(t):
    try:
        return t._version
    except RuntimeError:
        return None


def _signature(t):
    return (t.data_ptr(), t.device, t.dtype, tuple(t.shape), tuple(t.stride()))


@dataclass(frozen=True)
class Plan:
    hidden: torch.Tensor
    weight: torch.Tensor
    ones: torch.Tensor
    output: torch.Tensor
    signatures: tuple
    module: object
    raw_dot_policy: str = "native"
    requested_dot_policy: str = "native"
    weight_version: int = 0

    @property
    def metadata(self):
        return {"raw_precompute": True, "qkv_auxiliary_fused": False,
                "native_reduction": "sum" if self.hidden.shape[0] <= 32 else "tf32x3",
                "kernel_launches": 1, "pdl": False, "output_stride": self.output.stride(),
                "raw_dot_policy": self.raw_dot_policy,
                "requested_dot_policy": self.requested_dot_policy,
                "native_fallback": self.raw_dot_policy != self.requested_dot_policy,
                "effective_dot_precision": "sum" if self.hidden.shape[0] <= 32 else
                    "tf32" if self.raw_dot_policy != "native" else "tf32x3",
                "separate_block_accumulation": self.raw_dot_policy == "exact_tf32"}


def prepare_plan(hidden, weight, out=None, *, raw_dot_policy="native"):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare outside capture")
    if raw_dot_policy not in ("native", "bf16_tf32", "exact_tf32"):
        raise ValueError("unknown raw_dot_policy")
    if raw_dot_policy == "bf16_tf32" and (weight.dtype != torch.bfloat16 or hidden.dtype != torch.bfloat16):
        raise ValueError("bf16_tf32 requires BF16 hidden and BF16 weights")
    if (not hidden.is_cuda or hidden.dtype != torch.bfloat16 or hidden.ndim != 2
            or hidden.shape[1] != 6144 or not 1 <= hidden.shape[0] <= 256
            or not hidden.is_contiguous()):
        raise ValueError("hidden requires contiguous CUDA BF16 [M,6144]")
    if (weight.device != hidden.device or weight.dtype not in (torch.bfloat16, torch.float32)
            or weight.shape != (32, 6144) or not weight.is_contiguous()):
        raise ValueError("weight requires contiguous BF16 or FP32 [32,6144]")
    requested_dot_policy = raw_dot_policy
    if raw_dot_policy == "exact_tf32" and weight.dtype == torch.float32:
        exact = torch.isfinite(weight) & ((weight.view(torch.int32) & 8191) == 0)
        if not bool(exact.all().item()):
            raw_dot_policy = "native"
    if out is None:
        out = torch.empty((hidden.shape[0], 32), device=hidden.device, dtype=torch.float32)
    if out.device != hidden.device or out.dtype != torch.float32 or out.shape != (hidden.shape[0], 32) or out.stride() != (32, 1):
        raise ValueError("out requires rowmajor FP32 [M,32]")
    tensors = (hidden, weight, out)
    for i, a in enumerate(tensors):
        for b in tensors[:i]:
            if not (a.data_ptr()+a.numel()*a.element_size() <= b.data_ptr()
                    or b.data_ptr()+b.numel()*b.element_size() <= a.data_ptr()):
                raise ValueError("input/output overlap")
    module = importlib.import_module("rtp_llm.models_py.triton_kernels.sparse_mla.fused_logits_head_gate")
    ones = torch.ones_like(out)
    plan = Plan(hidden, weight, ones, out, tuple(_signature(t) for t in (*tensors, ones)), module,
                raw_dot_policy, requested_dot_policy, _version(weight))
    precompute(hidden, weight, out=out, plan=plan)
    return plan


def precompute(hidden, weight, *, out, plan):
    signatures = tuple(_signature(t) for t in (hidden, weight, out, plan.ones))
    if signatures[0][1:] != plan.signatures[0][1:] or signatures[1:] != plan.signatures[1:]:
        raise ValueError("prepared allocation/metadata changed")
    if plan.raw_dot_policy == "exact_tf32" and weight.dtype == torch.float32 and _version(weight) != plan.weight_version:
        raise ValueError("FP32 weights changed after exact-TF32 eligibility check; prepare again")
    for other in (weight, out, plan.ones):
        if not (hidden.data_ptr()+hidden.nbytes <= other.data_ptr()
                or other.data_ptr()+other.nbytes <= hidden.data_ptr()):
            raise ValueError("dynamic hidden overlaps static input/output")
    m, k = hidden.shape
    if m <= 32:
        plan.module._fused_logits_head_gate_small_t_kernel[(m, 32)](
            hidden, weight, plan.ones, out, 1.0, k, hidden.stride(0), weight.stride(0),
            weight.stride(1), 32, 32, BLOCK_K=8192, num_warps=4)
    elif plan.raw_dot_policy == "exact_tf32":
        _blockwise_tf32[((m+31)//32,)](hidden, weight, plan.ones, out, 1.0, m,
                                      BLOCK_M=32, BLOCK_K=128, num_warps=4, num_stages=3)
    else:
        plan.module._fused_logits_head_gate_kernel[((m+31)//32,)](
            hidden, weight, plan.ones, out, 1.0, m, k, 32, hidden.stride(0),
            weight.stride(0), weight.stride(1), 32, 32, BLOCK_M=32, BLOCK_K=128,
            BLOCK_N=32, W_IS_TRANSPOSED=False, HIGH_PRECISION=plan.raw_dot_policy == "native", num_warps=4, num_stages=3)
    return out
