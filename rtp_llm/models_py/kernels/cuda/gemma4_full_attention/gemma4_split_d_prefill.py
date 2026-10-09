# Copyright (c) 2026 Alibaba Cloud.
#
# Adapted for RTP-LLM from the FlashInfer CuTe-DSL wrapper reference:
#   build_logs/.../references/flashinfer_attention/wrappers/batch_prefill.py
#   (functools.cache'd compile over symbolic-dimension fake tensors,
#    plan()/run() split, TVM-FFI launch options).
# The kernel itself lives in cute_fmha_split_d.py.

"""PyTorch wrapper for the Gemma4 two-phase split-D attention kernels.

Phase 1 computes one FP32 causal online-softmax ``(m, l)`` trajectory over the
full D=512 QK reduction.  Phase 2 consumes that exact stats tensor, recomputes
QK, forms BF16 probabilities, and evaluates both N=256 P@V output halves.  No
output half computes an independent max or normalization sum.

Prototype scope: one contiguous NHD causal sequence, BF16 full output, and
SM100/SM103 tcgen05.  Unsupported inputs raise; there is no fallback path.
"""

import functools

import torch
from rtp_llm.models_py.utils.cutlass import setup_cutlass_import_path

setup_cutlass_import_path()

import cutlass
import cutlass.cute as cute

from .cute_fmha_split_d import LOG2_E, Gemma4SplitDApplyKernel, Gemma4SplitDStatsKernel

_DTYPE_MAP = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}

HEAD_DIM = 512


def gemma4_split_d_support(
    q: torch.Tensor,
    k: torch.Tensor,
) -> bool:
    """Whether the split-D stats kernel supports this q/k pair.

    Structural checks only (Python-level, no GPU work): dtype, head_dim=512,
    GQA divisibility, matching sequence lengths and devices.
    """
    if q.dtype not in _DTYPE_MAP or k.dtype != q.dtype:
        return False
    if q.dim() != 3 or k.dim() != 3:
        return False
    if q.shape[-1] != HEAD_DIM or k.shape[-1] != HEAD_DIM:
        return False
    num_qo_heads, num_kv_heads = q.shape[1], k.shape[1]
    if num_qo_heads < 1 or num_kv_heads < 1:
        return False
    if num_qo_heads % num_kv_heads != 0:
        return False
    if q.shape[0] < 1 or q.shape[0] != k.shape[0]:
        return False
    if not (q.is_cuda and k.is_cuda) or q.device != k.device:
        return False
    return True


def gemma4_split_d_attention_support(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> bool:
    """Whether the full fixed-statistics BF16 path supports ``q/k/v``."""
    if not gemma4_split_d_support(q, k) or q.dtype != torch.bfloat16:
        return False
    if v.dtype != torch.bfloat16 or v.dim() != 3:
        return False
    if v.shape != k.shape or v.device != q.device:
        return False
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        return False
    return True


@functools.cache
def _get_compiled_stats_kernel(in_dtype, num_qo_heads, num_kv_heads):
    """Compile and cache the stats kernel (vendor batch_prefill.py pattern).

    Sequence length stays symbolic (``sym_int``) so one compiled kernel serves
    every T; head counts and head_dim are fixed per cache entry.  All of
    ``problem_size`` remains a dynamic Int32 argument, exactly like the
    vendor's compile path.
    """
    kernel = Gemma4SplitDStatsKernel()

    sym_t = cute.sym_int()
    q_fake = cute.runtime.make_fake_compact_tensor(
        in_dtype,
        (sym_t, num_qo_heads, HEAD_DIM),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    k_fake = cute.runtime.make_fake_compact_tensor(
        in_dtype,
        (sym_t, num_kv_heads, HEAD_DIM),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    stats_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (sym_t, num_qo_heads, 2),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )

    problem_size = (1, num_qo_heads, num_kv_heads, HEAD_DIM)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        kernel,
        q_fake,
        k_fake,
        stats_fake,
        problem_size,
        1.0,
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )


@functools.cache
def _get_compiled_apply_kernel(num_qo_heads, num_kv_heads):
    """Compile/cache Phase 2; sequence length stays a symbolic runtime value."""
    kernel = Gemma4SplitDApplyKernel()
    in_dtype = cutlass.BFloat16

    sym_t = cute.sym_int()
    q_fake = cute.runtime.make_fake_compact_tensor(
        in_dtype,
        (sym_t, num_qo_heads, HEAD_DIM),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    k_fake = cute.runtime.make_fake_compact_tensor(
        in_dtype,
        (sym_t, num_kv_heads, HEAD_DIM),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    v_fake = cute.runtime.make_fake_compact_tensor(
        in_dtype,
        (sym_t, num_kv_heads, HEAD_DIM),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    stats_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (sym_t, num_qo_heads, 2),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        in_dtype,
        (sym_t, num_qo_heads, HEAD_DIM),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )

    problem_size = (1, num_qo_heads, num_kv_heads, HEAD_DIM)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        q_fake,
        k_fake,
        v_fake,
        stats_fake,
        out_fake,
        problem_size,
        1.0,
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )


class Gemma4SplitDPrefillWrapper:
    """Plan/run wrapper around both split-D phases.

    ``run`` preserves the Phase-1 stats API.  ``run_attention`` computes those
    stats once and passes the same tensor to the two-half Phase-2 apply kernel.
    """

    def __init__(self) -> None:
        self._compiled = None
        self._compiled_apply = None
        self._num_qo_heads = None
        self._num_kv_heads = None
        self._in_dtype = None
        self._q_dtype_torch = None
        self._sm_scale = 1.0
        self._scale_softmax_log2 = 1.0

    def plan(
        self,
        num_qo_heads: int,
        num_kv_heads: int,
        sm_scale: float = 1.0,
        q_data_type: torch.dtype = torch.bfloat16,
        head_dim: int = HEAD_DIM,
    ) -> None:
        """Compile the kernel for this head configuration (cached per combo)."""
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required for the split-D stats kernel")
        if head_dim != HEAD_DIM:
            raise ValueError(
                f"head_dim={head_dim} unsupported; the split-D kernel is "
                f"hardcoded to {HEAD_DIM} (four 128-wide D chunks)"
            )
        if num_qo_heads < 1 or num_kv_heads < 1:
            raise ValueError("num_qo_heads and num_kv_heads must both be positive")
        if num_qo_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_qo_heads={num_qo_heads} must be divisible by "
                f"num_kv_heads={num_kv_heads} (GQA)"
            )
        if q_data_type not in _DTYPE_MAP:
            raise ValueError(f"Unsupported q dtype: {q_data_type}")

        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._q_dtype_torch = q_data_type
        self._in_dtype = _DTYPE_MAP[q_data_type]
        self._sm_scale = sm_scale
        self._scale_softmax_log2 = sm_scale * LOG2_E

        self._compiled = _get_compiled_stats_kernel(
            self._in_dtype, num_qo_heads, num_kv_heads
        )
        self._compiled_apply = (
            _get_compiled_apply_kernel(num_qo_heads, num_kv_heads)
            if q_data_type == torch.bfloat16
            else None
        )

    def _validate_run_inputs(self, q: torch.Tensor, k: torch.Tensor) -> None:
        if self._compiled is None:
            raise RuntimeError("Call plan() before run()")
        for name, tensor in (("q", q), ("k", k)):
            if tensor.dtype != self._q_dtype_torch:
                raise ValueError(
                    f"{name}.dtype={tensor.dtype} does not match planned "
                    f"q_data_type={self._q_dtype_torch}"
                )
            if not tensor.is_cuda:
                raise ValueError(f"{name} must be a CUDA tensor")
            if tensor.dim() != 3:
                raise ValueError(
                    f"{name} must be [T, H, {HEAD_DIM}], got {tuple(tensor.shape)}"
                )
        if q.shape[1] != self._num_qo_heads:
            raise ValueError(
                f"q.shape[1]={q.shape[1]} does not match planned num_qo_heads={self._num_qo_heads}"
            )
        if k.shape[1] != self._num_kv_heads:
            raise ValueError(
                f"k.shape[1]={k.shape[1]} does not match planned num_kv_heads={self._num_kv_heads}"
            )
        if q.shape[-1] != HEAD_DIM or k.shape[-1] != HEAD_DIM:
            raise ValueError(
                f"head_dim must be {HEAD_DIM}, got q={q.shape[-1]}, k={k.shape[-1]}"
            )
        if q.shape[0] < 1 or q.shape[0] != k.shape[0]:
            raise ValueError(
                "Single-sequence prototype requires matching positive lengths, "
                f"got q.shape[0]={q.shape[0]}, k.shape[0]={k.shape[0]}"
            )
        if q.device != k.device:
            raise ValueError(
                f"q and k must be on the same device, got {q.device} and {k.device}"
            )
        # The kernel's layouts assume NHD row-major (D contiguous).
        if not q.is_contiguous() or not k.is_contiguous():
            raise ValueError("q and k must be contiguous [T, H, D] tensors")

    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        out: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute causal GQA softmax statistics for one sequence.

        :param q: ``[T, Hq, 512]`` queries (bf16/fp16, contiguous)
        :param k: ``[T, Hkv, 512]`` keys (bf16/fp16, contiguous)
        :param out: optional preallocated ``[T, Hq, 2]`` FP32 stats tensor
        :returns: ``[T, Hq, 2]`` FP32; ``[..., 0]`` = row max of scaled
            scores, ``[..., 1]`` = row sum of exp((S - m) * sm_scale)
        """
        self._validate_run_inputs(q, k)

        seq_len = q.shape[0]
        if out is None:
            stats = torch.empty(
                (seq_len, self._num_qo_heads, 2),
                dtype=torch.float32,
                device=q.device,
            )
        else:
            if (
                out.shape != (seq_len, self._num_qo_heads, 2)
                or out.dtype != torch.float32
            ):
                raise ValueError(
                    f"out must be [{seq_len}, {self._num_qo_heads}, 2] FP32, "
                    f"got {tuple(out.shape)} {out.dtype}"
                )
            if out.device != q.device:
                raise ValueError(f"out must be on {q.device}, got {out.device}")
            if not out.is_contiguous():
                raise ValueError("out must be contiguous")
            stats = out

        self._compiled(
            q,
            k,
            stats,
            (seq_len, self._num_qo_heads, self._num_kv_heads, HEAD_DIM),
            self._scale_softmax_log2,
        )
        return stats

    def _validate_attention_inputs(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        self._validate_run_inputs(q, k)
        if self._compiled_apply is None:
            raise ValueError("Full split-D attention requires a BF16 plan")
        if v.dtype != torch.bfloat16:
            raise ValueError(f"v.dtype must be torch.bfloat16, got {v.dtype}")
        if v.shape != k.shape:
            raise ValueError(
                f"v must match k shape {tuple(k.shape)}, got {tuple(v.shape)}"
            )
        if not v.is_cuda or v.device != q.device:
            raise ValueError(f"v must be a CUDA tensor on {q.device}, got {v.device}")
        if not v.is_contiguous():
            raise ValueError("v must be a contiguous [T, Hkv, 512] tensor")

    def run_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor = None,
        stats_out: torch.Tensor = None,
    ) -> torch.Tensor:
        """Run Phase 1 once, then apply its fixed stats to both V halves.

        :param q: contiguous BF16 ``[T,Hq,512]`` queries
        :param k: contiguous BF16 ``[T,Hkv,512]`` keys
        :param v: contiguous BF16 ``[T,Hkv,512]`` values
        :param out: optional contiguous BF16 ``[T,Hq,512]`` output
        :param stats_out: optional contiguous FP32 ``[T,Hq,2]`` workspace/output
        :returns: ``out`` containing the normalized attention result
        """
        self._validate_attention_inputs(q, k, v)
        seq_len = q.shape[0]
        if out is None:
            output = torch.empty(
                (seq_len, self._num_qo_heads, HEAD_DIM),
                dtype=torch.bfloat16,
                device=q.device,
            )
        else:
            expected_shape = (seq_len, self._num_qo_heads, HEAD_DIM)
            if out.shape != expected_shape or out.dtype != torch.bfloat16:
                raise ValueError(
                    f"out must be {list(expected_shape)} BF16, "
                    f"got {tuple(out.shape)} {out.dtype}"
                )
            if out.device != q.device:
                raise ValueError(f"out must be on {q.device}, got {out.device}")
            if not out.is_contiguous():
                raise ValueError("out must be contiguous")
            output = out

        # This is the only max/sum computation.  The apply grid's two output
        # halves both read this same tensor and only evaluate exp(S-m) / l.
        stats = self.run(q, k, out=stats_out)
        self._compiled_apply(
            q,
            k,
            v,
            stats,
            output,
            (seq_len, self._num_qo_heads, self._num_kv_heads, HEAD_DIM),
            self._scale_softmax_log2,
        )
        return output


def gemma4_split_d_stats(
    q: torch.Tensor,
    k: torch.Tensor,
    sm_scale: float = 1.0,
) -> torch.Tensor:
    """One-shot helper: plan (cached) + run for a q/k pair.

    Raises ``ValueError`` for unsupported inputs instead of silently falling
    back to another path (per the task contract: no hidden two-softmax
    substitution).
    """
    if not gemma4_split_d_support(q, k):
        raise ValueError(
            f"Unsupported inputs for the split-D stats kernel: "
            f"q={tuple(q.shape)} {q.dtype}, k={tuple(k.shape)} {k.dtype}. "
            f"Requires CUDA [T, H, 512] bf16/fp16 tensors with T matched, "
            f"Hq % Hkv == 0."
        )
    wrapper = Gemma4SplitDPrefillWrapper()
    wrapper.plan(
        q.shape[1],
        k.shape[1],
        sm_scale=sm_scale,
        q_data_type=q.dtype,
    )
    return wrapper.run(q, k)


def gemma4_split_d_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float = 1.0,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """One-shot full BF16 split-D attention with one shared stats trajectory."""
    if not gemma4_split_d_attention_support(q, k, v):
        raise ValueError(
            "Unsupported inputs for the full split-D attention kernel: "
            f"q={tuple(q.shape)} {q.dtype}, k={tuple(k.shape)} {k.dtype}, "
            f"v={tuple(v.shape)} {v.dtype}. Requires same-device contiguous "
            "CUDA BF16 q=[T,Hq,512], k/v=[T,Hkv,512], and Hq % Hkv == 0."
        )
    wrapper = Gemma4SplitDPrefillWrapper()
    wrapper.plan(
        q.shape[1],
        k.shape[1],
        sm_scale=sm_scale,
        q_data_type=torch.bfloat16,
    )
    return wrapper.run_attention(q, k, v, out=out)


def run_gemma4_split_d_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float = 1.0,
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Compatibility-named full-output entry point for model integration."""
    return gemma4_split_d_attention(q, k, v, sm_scale=sm_scale, out=out)
