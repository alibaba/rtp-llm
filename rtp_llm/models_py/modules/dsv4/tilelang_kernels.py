"""Vendored TileLang kernels from DeepSeek-V4's ``inference/kernel.py``.

Keeps only what dsv4/attention.py uses:
- ``sparse_attn_kernel`` — top-k sparse attention + per-head learned sink.

The math is bit-for-bit from the upstream V4 release at
``/mnt/nas1/hf/DeepSeek-V4-Flash/inference/kernel.py``.  No modifications.
Wrapped in a lazy-import guard so dsv4 stays importable on hosts without
tilelang installed (fallbacks to the PyTorch reference at call site).

V4 is MQA + Q-LoRA (NOT MLA): single 512-dim shared K=V across 64 Q
heads; the kernel is the author-authored path for exactly this shape,
so unlike ``flash_mla.flash_mla_sparse_fwd`` there is no "misleading
name" concern.
"""

from typing import Optional

import torch


_TILELANG_AVAILABLE: bool = False
_SPARSE_ATTN_KERNEL_CACHE: dict = {}

try:  # noqa: broad except — ImportError / OSError (CDLL symbol miss) / RuntimeError
    import tilelang
    import tilelang.language as T

    _TILELANG_AVAILABLE = True
    tilelang.set_log_level("WARNING")

    _PASS_CONFIGS = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }

    BF16 = "bfloat16"
    FP32 = "float32"
    INT32 = "int32"

    @tilelang.jit(pass_configs=_PASS_CONFIGS)
    def _sparse_attn_kernel(h: int, d: int, scale=None):
        """Verbatim port of V4 inference/kernel.py:sparse_attn_kernel.

        Top-k sparse MQA attention with per-head learned attn_sink,
        FlashAttention-style online softmax. Block of 64 along topk,
        d along KV dim (512 for V4), h along Q heads (V4: 64).
        """
        b = T.symbolic("b")
        m = T.symbolic("m")
        n = T.symbolic("n")
        topk = T.symbolic("topk")
        if scale is None:
            scale = (1.0 / d) ** 0.5

        num_stages = 2
        threads = 256
        block = 64
        num_blocks = tilelang.cdiv(topk, block)

        @T.prim_func
        def sparse_attn_kernel_(
            q: T.Tensor[(b, m, h, d), BF16],
            kv: T.Tensor[(b, n, d), BF16],
            o: T.Tensor[(b, m, h, d), BF16],
            attn_sink: T.Tensor[(h,), FP32],
            topk_idxs: T.Tensor[(b, m, topk), INT32],
        ):
            with T.Kernel(m, b, threads=threads) as (bx, by):
                q_shared = T.alloc_shared((h, d), BF16)
                kv_shared = T.alloc_shared((block, d), BF16)
                o_shared = T.alloc_shared((h, d), BF16)
                acc_s_cast = T.alloc_shared((h, block), BF16)

                idxs = T.alloc_fragment(block, INT32)
                acc_s = T.alloc_fragment((h, block), FP32)
                acc_o = T.alloc_fragment((h, d), FP32)
                scores_max = T.alloc_fragment(h, FP32)
                scores_max_prev = T.alloc_fragment(h, FP32)
                scores_scale = T.alloc_fragment(h, FP32)
                scores_sum = T.alloc_fragment(h, FP32)
                sum_exp = T.alloc_fragment(h, FP32)

                T.clear(acc_o)
                T.clear(sum_exp)
                T.fill(scores_max, -T.infinity(FP32))
                T.copy(q[by, bx, :, :], q_shared)

                for t in T.Pipelined(num_blocks, num_stages=num_stages):
                    for i in T.Parallel(block):
                        idxs[i] = T.if_then_else(
                            t * block + i < topk,
                            topk_idxs[by, bx, t * block + i], -1,
                        )
                    for i, j in T.Parallel(block, d):
                        kv_shared[i, j] = T.if_then_else(
                            idxs[i] != -1, kv[by, idxs[i], j], 0,
                        )
                    for i, j in T.Parallel(h, block):
                        acc_s[i, j] = T.if_then_else(
                            idxs[j] != -1, 0, -T.infinity(FP32),
                        )
                    T.gemm(q_shared, kv_shared, acc_s,
                           transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(h, block):
                        acc_s[i, j] *= scale
                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(h):
                        scores_scale[i] = T.exp(scores_max_prev[i] - scores_max[i])
                    for i, j in T.Parallel(h, block):
                        acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(h):
                        sum_exp[i] = sum_exp[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(h, d):
                        acc_o[i, j] *= scores_scale[i]
                    T.gemm(acc_s_cast, kv_shared, acc_o,
                           policy=T.GemmWarpPolicy.FullRow)

                for i in T.Parallel(h):
                    sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
                for i, j in T.Parallel(h, d):
                    acc_o[i, j] /= sum_exp[i]
                T.copy(acc_o, o_shared)
                T.copy(o_shared, o[by, bx, :, :])

        return sparse_attn_kernel_

except (ImportError, OSError, RuntimeError) as _e:  # pragma: no cover
    # tilelang can fail to import with OSError when libstdc++ symbols
    # don't match between the pip-installed libtvm.so and the toolchain
    # (e.g. bazel runfiles with __cxa_call_terminate undefined).  Fall
    # back to the other sparse-attn paths silently.
    import logging as _logging
    _logging.getLogger(__name__).info(
        "tilelang unavailable (%s); dsv4 sparse attention will fall "
        "back to FlashMLA or the Python reference", type(_e).__name__,
    )
    _TILELANG_AVAILABLE = False


def tilelang_available() -> bool:
    return _TILELANG_AVAILABLE


def sparse_attn(
    q: torch.Tensor,            # [B, S, H, D] bf16
    kv: torch.Tensor,           # [B, T, D] bf16  (single KV head, shared across H)
    attn_sink: torch.Tensor,    # [H] fp32
    topk_idxs: torch.Tensor,    # [B, S, K] int — any integer dtype; cast to int32
    softmax_scale: float,
) -> torch.Tensor:
    """V4's native top-k sparse MQA attention with per-head learned sink.

    Exact port of ``inference/kernel.py:sparse_attn``. Pads the Q-head
    dimension up to 16 when ``H < 16`` for kernel throughput (trims after).
    Caches the compiled kernel per ``(h_padded, d, softmax_scale)`` triple.
    """
    if not _TILELANG_AVAILABLE:
        raise RuntimeError(
            "tilelang is not available; cannot run V4 TileLang sparse_attn. "
            "Install tilelang>=0.1.7 (pip install --no-build-isolation tilelang).",
        )

    b, s, h, d = q.size()
    h_padded = max(h, 16)
    if h < 16:
        q = torch.cat([q, q.new_zeros(b, s, 16 - h, d)], dim=2)
        attn_sink = torch.cat([attn_sink, attn_sink.new_zeros(16 - h)])

    if topk_idxs.dtype != torch.int32:
        topk_idxs = topk_idxs.to(torch.int32)

    key = (h_padded, d, float(softmax_scale))
    kernel = _SPARSE_ATTN_KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = _sparse_attn_kernel(h_padded, d, softmax_scale)
        _SPARSE_ATTN_KERNEL_CACHE[key] = kernel

    o = torch.empty_like(q)
    kernel(q, kv, o, attn_sink, topk_idxs)
    if h < 16:
        o = o.narrow(2, 0, h).contiguous()
    return o
