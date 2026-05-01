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
# Cache for paged variant (output: acc_o + sum_exp + scores_max, no sink, no normalize).
# Key: (h_padded, d, softmax_scale).  KV is consumed as flat ``[N, D]`` so the
# pool's tokens_per_block doesn't enter the cache key — caller views the BlockPool
# tensor as ``[num_blocks * tpb, head_dim]`` and translates global slot indices to
# physical flat indices before invoking.
_SPARSE_ATTN_LSE_KERNEL_CACHE: dict = {}


def _ensure_tvm_tmpdir_writable() -> None:
    """Route TVM debug tempdirs away from a root-owned /tmp parent."""
    import os
    import tempfile

    default_parent = os.path.join(tempfile.gettempdir(), "tvm-debug-mode-tempdirs")
    if os.path.exists(default_parent) and os.access(default_parent, os.W_OK):
        return
    if not os.path.exists(default_parent):
        try:
            os.makedirs(default_parent, exist_ok=True)
            return
        except OSError:
            pass

    fallback = os.path.join("/tmp", f"rtp_tvm_tmp_{os.getuid()}")
    try:
        os.makedirs(fallback, exist_ok=True)
    except OSError:
        return
    os.environ["TMPDIR"] = fallback
    tempfile.tempdir = None


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
    import os
    import sys

    try:
        ctypes.CDLL("libz3.so", mode=ctypes.RTLD_GLOBAL)
        return
    except OSError:
        pass

    try:
        import z3
        lib_dir = os.path.join(os.path.dirname(z3.__file__), "lib")
    except Exception:
        return

    # In Bazel runfiles, tilelang and z3-solver live in separate pip_parse
    # repositories. tilelang/lib/libtvm.so has RUNPATH
    # $ORIGIN/../../z3/lib, so make that path exist next to tilelang.
    # Loading z3 by absolute path is not enough because the z3 wheel's SONAME
    # is versioned (for example libz3.so.4.15) while libtvm needs libz3.so.
    for site_dir in list(sys.path):
        tilelang_lib_dir = os.path.join(site_dir, "tilelang", "lib")
        if not os.path.isdir(tilelang_lib_dir):
            continue
        dst_dir = os.path.join(site_dir, "z3", "lib")
        try:
            os.makedirs(dst_dir, exist_ok=True)
            for name in os.listdir(lib_dir):
                if not name.startswith("libz3.so"):
                    continue
                src = os.path.join(lib_dir, name)
                dst = os.path.join(dst_dir, name)
                if os.path.exists(dst):
                    continue
                os.symlink(src, dst)
        except OSError:
            # Some runfiles layouts may be read-only. Fall through to the
            # absolute-path preload attempt; regular site-packages installs do
            # not need the symlink path.
            pass

    for name in ("libz3.so", "libz3.so.4.15", "libz3.so.4.13", "libz3.so.4"):
        path = os.path.join(lib_dir, name)
        if os.path.exists(path):
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                return
            except OSError:
                pass


_ensure_libz3_loadable()
_ensure_tvm_tmpdir_writable()

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

    @tilelang.jit(pass_configs=_PASS_CONFIGS)
    def _sparse_attn_kernel_lse(h: int, d: int, scale=None):
        """Branchwise sparse attention that outputs *un-normalized* online
        softmax state instead of a normalized output.  Used by the paged
        prefill flow: each branch (SWA pool + CMP pool) runs this kernel
        against its own KV pool, then a Python merge step combines the
        states with the per-head ``attn_sink`` to produce the final
        normalized output.

        Differences from ``sparse_attn_kernel_``:
          * ``kv`` is flat ``[n, d]`` (not ``[b, n, d]``).  Caller views the
            paged ``[num_blocks, tokens_per_block, d]`` BlockPool tensor as
            ``[num_blocks * tokens_per_block, d]`` (zero-copy reshape) and
            remaps the global topk slot indices into physical flat indices
            (``block_table[b, slot // tpb] * tpb + slot % tpb``) before the
            launch.  Same KV is shared across all queries in the batch.
          * Outputs three FP32 buffers per (b, m, h) instead of one BF16
            normalized output:
              * ``o_acc[b, m, h, d]``     — Σ exp(s_i − M) · v_i
              * ``sum_exp[b, m, h]``      — Σ exp(s_i − M)
              * ``scores_max[b, m, h]``   — M = max_i s_i  (clamped to ``-1e30``
                so a fully-masked branch produces sum_exp=0 + max=-1e30 that
                merges harmlessly).
          * Does NOT apply ``attn_sink``: the merge step adds it once to the
            combined denominator across both branches.
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
        def sparse_attn_kernel_lse_(
            q: T.Tensor[(b, m, h, d), BF16],
            kv: T.Tensor[(n, d), BF16],
            o_acc: T.Tensor[(b, m, h, d), FP32],
            sum_exp_out: T.Tensor[(b, m, h), FP32],
            scores_max_out: T.Tensor[(b, m, h), FP32],
            topk_idxs: T.Tensor[(b, m, topk), INT32],
        ):
            with T.Kernel(m, b, threads=threads) as (bx, by):
                q_shared = T.alloc_shared((h, d), BF16)
                kv_shared = T.alloc_shared((block, d), BF16)
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
                # Init scores_max to a *finite* sentinel ``-1e30`` instead of
                # ``-inf``.  Reason: when the entire branch is masked (every
                # ``topk_idxs`` entry == -1), ``acc_s`` stays ``-inf`` after the
                # GEMM; if ``scores_max`` were also ``-inf`` then
                # ``scores_max_prev - scores_max == NaN`` on the first iteration,
                # poisoning ``sum_exp`` for the rest of the run.  ``-1e30`` is far
                # below any plausible logit (V4 logits are ~O(D**-0.5) bounded), so
                # the first valid score still replaces it; for fully-masked
                # branches we end up with scores_max=-1e30, sum_exp=0, acc_o=0 —
                # exactly the identity element the merge step expects.
                T.fill(scores_max, T.Cast(FP32, -1.0e30))
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
                            kv[idxs[i], j],
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

                T.copy(acc_o, o_acc[by, bx, :, :])
                T.copy(sum_exp, sum_exp_out[by, bx, :])
                T.copy(scores_max, scores_max_out[by, bx, :])

        return sparse_attn_kernel_lse_

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


def sparse_attn_branch_lse(
    q: torch.Tensor,             # [B, S, H, D] bf16
    kv_flat: torch.Tensor,       # [N, D] bf16 — flat view of one paged pool
    topk_idxs_flat: torch.Tensor,  # [B, S, K] int — physical flat indices into kv_flat (-1 = mask)
    softmax_scale: float,
):
    """Run one branch of paged sparse attention.

    Returns un-normalized online-softmax state ``(o_acc, sum_exp,
    scores_max)`` so the caller can merge multiple branches (typically
    SWA + CMP for V4 attention) into a single normalized output that
    includes ``attn_sink``.

    The kernel does NOT apply ``attn_sink``.  ``topk_idxs_flat`` is
    expected to already be the physical row index into ``kv_flat`` —
    callers translate global slot indices through their pool's
    block_table before calling this.

    Returns:
        Tuple ``(o_acc, sum_exp, scores_max)`` of dtype FP32:
          * ``o_acc``      — ``[B, S, H, D]`` un-normalized weighted sum
          * ``sum_exp``    — ``[B, S, H]`` Σ exp(s_i − M)
          * ``scores_max`` — ``[B, S, H]`` per-(b,s,h) max score (clamped
            ≥ -1e30 so empty branches merge harmlessly).
    """
    if not _TILELANG_AVAILABLE:
        raise RuntimeError(
            "tilelang is not available; cannot run V4 paged sparse_attn LSE branch."
        )

    b, s, h, d = q.size()
    h_padded = max(h, 16)
    if h < 16:
        q = torch.cat([q, q.new_zeros(b, s, 16 - h, d)], dim=2)

    if topk_idxs_flat.dtype != torch.int32:
        topk_idxs_flat = topk_idxs_flat.to(torch.int32)
    topk_idxs_flat = topk_idxs_flat.contiguous()

    key = (h_padded, d, float(softmax_scale))
    kernel = _SPARSE_ATTN_LSE_KERNEL_CACHE.get(key)
    if kernel is None:
        _log.warning(
            "[dsv4] JIT-compiling V4 TileLang sparse_attn LSE-branch h=%d d=%d",
            h_padded,
            d,
        )
        kernel = _sparse_attn_kernel_lse(h_padded, d, softmax_scale)
        _SPARSE_ATTN_LSE_KERNEL_CACHE[key] = kernel

    o_acc = torch.empty(b, s, h_padded, d, dtype=torch.float32, device=q.device)
    sum_exp = torch.empty(b, s, h_padded, dtype=torch.float32, device=q.device)
    scores_max = torch.empty(b, s, h_padded, dtype=torch.float32, device=q.device)

    kernel(q, kv_flat, o_acc, sum_exp, scores_max, topk_idxs_flat)

    if h < 16:
        o_acc = o_acc.narrow(2, 0, h).contiguous()
        sum_exp = sum_exp.narrow(2, 0, h).contiguous()
        scores_max = scores_max.narrow(2, 0, h).contiguous()
    return o_acc, sum_exp, scores_max


def merge_branches_with_sink(
    o_accs,            # list of [B, S, H, D] fp32
    sum_exps,          # list of [B, S, H] fp32
    scores_maxs,       # list of [B, S, H] fp32
    attn_sink: torch.Tensor,   # [H] fp32
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Merge multiple paged sparse-attn branches with V4's per-head sink.

    Reproduces the math of the unified ``sparse_attn_kernel_`` finalize
    step, but applied across branches.  Bit-equal to the single-pool
    kernel's output when given a single branch (modulo FP32→BF16 cast).

    For each (b, s, h):
        M = max(scores_max_branch_i ..., attn_sink[h])
        e_i = sum_exp_i * exp(scores_max_i - M)        (per branch)
        e_sink = exp(attn_sink[h] - M)
        denom = Σ_i e_i + e_sink
        o_acc_total = Σ_i o_acc_i * exp(scores_max_i - M)
        out[b,s,h,:] = o_acc_total / denom
    """
    assert len(o_accs) >= 1
    assert len(o_accs) == len(sum_exps) == len(scores_maxs)
    H = attn_sink.numel()
    sink_view = attn_sink.view(1, 1, H).to(torch.float32)

    M = scores_maxs[0]
    for sm in scores_maxs[1:]:
        M = torch.maximum(M, sm)
    M = torch.maximum(M, sink_view)

    e_sink = torch.exp(sink_view - M)
    denom = e_sink.clone()
    o_total = None
    for o_acc_i, sum_exp_i, scores_max_i in zip(o_accs, sum_exps, scores_maxs):
        e_i = torch.exp(scores_max_i - M)  # [B, S, H]
        denom = denom + sum_exp_i * e_i
        contrib = o_acc_i * e_i.unsqueeze(-1)
        o_total = contrib if o_total is None else (o_total + contrib)

    out = (o_total / denom.unsqueeze(-1)).to(out_dtype)
    return out


def _flat_view_pool(pool_tensor: torch.Tensor, tokens_per_block: int, head_dim: int, dtype: torch.dtype) -> torch.Tensor:
    """View a ``[num_blocks, stride_bytes] uint8`` BlockPool tensor as
    ``[num_blocks * tokens_per_block, head_dim]`` of ``dtype`` — zero-copy.
    """
    if pool_tensor.dim() != 2 or pool_tensor.dtype != torch.uint8:
        raise ValueError(
            f"pool_tensor must be [num_blocks, stride_bytes] uint8, got "
            f"shape={tuple(pool_tensor.shape)} dtype={pool_tensor.dtype}"
        )
    num_blocks = pool_tensor.size(0)
    expected_stride = tokens_per_block * head_dim * torch.empty((), dtype=dtype).element_size()
    if pool_tensor.size(1) != expected_stride:
        raise ValueError(
            f"pool stride {pool_tensor.size(1)} != tokens_per_block({tokens_per_block}) "
            f"* head_dim({head_dim}) * esize({torch.empty((), dtype=dtype).element_size()}) "
            f"= {expected_stride}"
        )
    return pool_tensor.view(dtype).view(num_blocks * tokens_per_block, head_dim)


def _global_to_phys_flat(
    topk_global: torch.Tensor,    # [B, S, K] int — global slot indices, -1 = mask
    block_table: torch.Tensor,    # [B, max_blocks_per_req] int — physical block ids
    tokens_per_block: int,
) -> torch.Tensor:
    """Translate ``[B, S, K]`` global slot indices into physical flat
    indices into the pool's flat view ``[num_blocks * tokens_per_block, D]``.

    Three masking conditions force the result to ``-1`` (kernel-side mask):
      1. Caller passed ``-1`` already.
      2. ``slot // tokens_per_block >= max_blocks_per_req`` — caller asked
         for a slot past the request's allocated block range.  This is
         legal in the V4 attention because callers (e.g. ``Indexer.forward``)
         add a fixed ``offset`` to all topk indices regardless of the actual
         valid CMP entry count, then rely on the kernel masking past-end
         entries (the legacy dense path "succeeds" because the dense buffer
         is zero-initialized in those slots; the paged kernel needs an
         explicit mask).
      3. ``block_table[b, slot // tpb] <= 0`` — the framework reserves
         block_id 0 as a padding sentinel; reading from it would land
         in some other request's data.

    Otherwise: ``phys[b,s,k] = block_table[b, slot // tpb] * tpb + slot % tpb``.
    """
    if topk_global.dtype != torch.long:
        topk_global = topk_global.to(torch.long)
    if block_table.dtype != torch.long:
        block_table = block_table.to(torch.long)
    valid = topk_global >= 0
    slot = topk_global.clamp(min=0)
    block_idx = slot // int(tokens_per_block)
    in_block = slot % int(tokens_per_block)

    B, S, K = topk_global.shape
    max_blocks = block_table.size(1)
    valid = valid & (block_idx < max_blocks)
    block_idx_clamped = block_idx.clamp(max=max_blocks - 1)
    # gather: per-batch lookup
    flat_idx = block_idx_clamped.view(B, S * K)
    phys_block = torch.gather(block_table, 1, flat_idx).view(B, S, K)
    valid = valid & (phys_block > 0)
    phys = phys_block * int(tokens_per_block) + in_block
    return torch.where(valid, phys, torch.full_like(phys, -1))


def sparse_attn_paged_single_pool(
    q: torch.Tensor,                 # [B, S, H, D] bf16
    pool_tensor: torch.Tensor,       # [num_blocks, stride_bytes] uint8
    block_table: torch.Tensor,       # [B, max_blocks_per_req] int32
    attn_sink: torch.Tensor,         # [H] fp32
    topk_global: torch.Tensor,       # [B, S, K] int — global slot indices, -1 = mask
    softmax_scale: float,
    tokens_per_block: int,
    head_dim: int,
) -> torch.Tensor:
    """V4 paged sparse attention against a single KV pool (SWA-only layers).

    Input is the framework's BlockPool tensor + per-request block table;
    no intermediate dense gather.  Numerically equivalent (within bf16
    precision) to ``sparse_attn(q, kv_dense, attn_sink, topk_dense, scale)``
    where ``kv_dense[b, slot, :] = pool_view[block_table[b, slot//tpb],
    slot % tpb, :]`` and ``topk_dense = topk_global``.
    """
    kv_flat = _flat_view_pool(pool_tensor, tokens_per_block, head_dim, q.dtype)
    phys_flat = _global_to_phys_flat(topk_global, block_table, tokens_per_block)
    o_acc, sum_exp, scores_max = sparse_attn_branch_lse(q, kv_flat, phys_flat, softmax_scale)
    return merge_branches_with_sink([o_acc], [sum_exp], [scores_max], attn_sink, out_dtype=q.dtype)


def sparse_attn_paged_two_pool(
    q: torch.Tensor,                  # [B, S, H, D] bf16
    swa_pool: torch.Tensor,           # [num_blocks_swa, stride_bytes] uint8
    swa_block_table: torch.Tensor,    # [B, max_blocks_swa] int32
    swa_topk_global: torch.Tensor,    # [B, S, K_swa] int — global SWA slot, -1 mask
    swa_tokens_per_block: int,
    cmp_pool: torch.Tensor,           # [num_blocks_cmp, stride_bytes] uint8
    cmp_block_table: torch.Tensor,    # [B, max_blocks_cmp] int32
    cmp_topk_global: torch.Tensor,    # [B, S, K_cmp] int — global CMP slot, -1 mask
    cmp_tokens_per_block: int,
    attn_sink: torch.Tensor,          # [H] fp32
    softmax_scale: float,
    head_dim: int,
) -> torch.Tensor:
    """V4 paged sparse attention against TWO KV pools (CSA / HCA layers).

    SWA and compressed-K live in separate BlockPool pools with different
    ``tokens_per_block``.  Run the LSE-branch kernel against each, then
    merge with the per-head ``attn_sink``.

    Numerically equivalent (within bf16 precision) to the dense flow:

        kv_swa = gather_dense(swa_pool, swa_bt, [0..win))
        kv_cmp = gather_dense(cmp_pool, cmp_bt, [0..n_cmp))
        kv_cat = torch.cat([kv_swa, kv_cmp], dim=1)
        topk = torch.cat([swa_topk_global, cmp_topk_global + win], dim=-1)
        out = sparse_attn(q, kv_cat, attn_sink, topk, scale)

    but without materializing ``kv_swa`` / ``kv_cmp`` / ``kv_cat``.
    """
    # SWA branch
    swa_kv_flat = _flat_view_pool(swa_pool, swa_tokens_per_block, head_dim, q.dtype)
    swa_phys = _global_to_phys_flat(swa_topk_global, swa_block_table, swa_tokens_per_block)
    o_acc_swa, sum_exp_swa, max_swa = sparse_attn_branch_lse(q, swa_kv_flat, swa_phys, softmax_scale)
    # CMP branch
    cmp_kv_flat = _flat_view_pool(cmp_pool, cmp_tokens_per_block, head_dim, q.dtype)
    cmp_phys = _global_to_phys_flat(cmp_topk_global, cmp_block_table, cmp_tokens_per_block)
    o_acc_cmp, sum_exp_cmp, max_cmp = sparse_attn_branch_lse(q, cmp_kv_flat, cmp_phys, softmax_scale)
    return merge_branches_with_sink(
        [o_acc_swa, o_acc_cmp],
        [sum_exp_swa, sum_exp_cmp],
        [max_swa, max_cmp],
        attn_sink,
        out_dtype=q.dtype,
    )
