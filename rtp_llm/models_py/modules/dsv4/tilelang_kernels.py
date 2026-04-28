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

import logging as _logging
from typing import Optional

import torch

_log = _logging.getLogger(__name__)

_TILELANG_AVAILABLE: bool = False
_SPARSE_ATTN_KERNEL_CACHE: dict = {}


def _ensure_libz3_loadable() -> None:
    """z3-solver ships ``libz3.so`` inside its site-packages lib dir and
    tilelang's internal Rewriter dlopens it by bare name (`libz3.so`),
    which only works if the lib is on LD_LIBRARY_PATH or loaded eagerly.
    Under bazel runfiles the pip_parse for z3-solver puts the library at
    ``pip_cuda12_arm_torch_z3_solver/site-packages/z3/lib/libz3.so`` —
    not on LD_LIBRARY_PATH — so the bare-name dlopen fails.  We pre-load
    it via ctypes before tilelang imports; if the host has a system
    libz3 this is a no-op."""
    import ctypes

    try:
        ctypes.CDLL("libz3.so", mode=ctypes.RTLD_GLOBAL)
        return
    except OSError:
        pass
    try:
        import os

        import z3

        lib_dir = os.path.join(os.path.dirname(z3.__file__), "lib")
        for name in ("libz3.so", "libz3.so.4.13", "libz3.so.4"):
            path = os.path.join(lib_dir, name)
            if os.path.exists(path):
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                return
    except Exception:
        pass


_ensure_libz3_loadable()

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
                # Identity element of max — standard online-softmax init.
                # NaN would arise only if block 0 is entirely masked
                # (exp(−∞ − (−∞)) = exp(NaN)), which requires topk = 0.
                # V4's Indexer always returns index_topk ≥ 1 valid keys,
                # so block 0 always has ≥ 1 valid entry and this is safe.
                T.fill(scores_max, -T.infinity(FP32))
                T.copy(q[by, bx, :, :], q_shared)

                for t in T.Pipelined(num_blocks, num_stages=num_stages):
                    for i in T.Parallel(block):
                        idxs[i] = T.if_then_else(
                            t * block + i < topk,
                            topk_idxs[by, bx, t * block + i],
                            -1,
                        )
                    for i, j in T.Parallel(block, d):
                        kv_shared[i, j] = T.if_then_else(
                            idxs[i] != -1,
                            kv[by, idxs[i], j],
                            0,
                        )
                    for i, j in T.Parallel(h, block):
                        acc_s[i, j] = T.if_then_else(
                            idxs[j] != -1,
                            0,
                            -T.infinity(FP32),
                        )
                    T.gemm(
                        q_shared,
                        kv_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
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
                    T.gemm(
                        acc_s_cast, kv_shared, acc_o, policy=T.GemmWarpPolicy.FullRow
                    )

                for i in T.Parallel(h):
                    sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
                for i, j in T.Parallel(h, d):
                    acc_o[i, j] /= sum_exp[i]
                T.copy(acc_o, o_shared)
                T.copy(o_shared, o[by, bx, :, :])

        return sparse_attn_kernel_

except (ImportError, OSError, RuntimeError) as _e:  # pragma: no cover
    _log.warning(
        "[dsv4] tilelang unavailable (%s: %s); sparse attention falls "
        "back to the Python reference",
        type(_e).__name__,
        _e,
    )
    _TILELANG_AVAILABLE = False

_log.info("[dsv4] tilelang_kernels init: _TILELANG_AVAILABLE=%s", _TILELANG_AVAILABLE)


def tilelang_available() -> bool:
    return _TILELANG_AVAILABLE


def prewarm(
    n_heads: int, head_dim: int, softmax_scale: float, device: str = "cuda"
) -> None:
    """Pre-warm (JIT-compile and cache) the TileLang sparse_attn kernel.

    Must be called before CUDA graph capture.  The first call triggers JIT
    compilation; subsequent calls with the same (h_padded, d, scale) hit the
    module-level _SPARSE_ATTN_KERNEL_CACHE and skip compilation.
    """
    if not _TILELANG_AVAILABLE:
        return
    h_padded = max(n_heads, 16)
    _log.info(
        "[dsv4] pre-warming TileLang sparse_attn h=%d d=%d scale=%.6f on %s",
        h_padded,
        head_dim,
        softmax_scale,
        device,
    )
    q = torch.zeros((1, 1, h_padded, head_dim), dtype=torch.bfloat16, device=device)
    kv = torch.zeros((1, 1, head_dim), dtype=torch.bfloat16, device=device)
    attn_sink = torch.zeros(h_padded, dtype=torch.float32, device=device)
    topk_idxs = torch.zeros((1, 1, 1), dtype=torch.int32, device=device)
    sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)
    _log.info("[dsv4] TileLang sparse_attn prewarm done")


def sparse_attn(
    q: torch.Tensor,  # [B, S, H, D] bf16
    kv: torch.Tensor,  # [B, T, D] bf16  (single KV head, shared across H)
    attn_sink: torch.Tensor,  # [H] fp32
    topk_idxs: torch.Tensor,  # [B, S, K] int — any integer dtype; cast to int32
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
        _log.warning(
            "[dsv4] JIT-compiling V4 TileLang sparse_attn h=%d d=%d", h_padded, d
        )
        kernel = _sparse_attn_kernel(h_padded, d, softmax_scale)
        _SPARSE_ATTN_KERNEL_CACHE[key] = kernel

    o = torch.empty_like(q)
    kernel(q, kv, o, attn_sink, topk_idxs)
    if h < 16:
        o = o.narrow(2, 0, h).contiguous()
    return o
