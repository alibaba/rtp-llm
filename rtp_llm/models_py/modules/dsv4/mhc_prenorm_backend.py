"""Backend selection and split-K arithmetic for the mHC pre-norm GEMM.

Single source of truth for two decisions that used to be made twice, once in the
runtime wrapper (``3rdparty/tile_kernels/modeling/mhc/ops/pre_big_fuse.py``) and
once in the JIT warmup (``dsv4_kernel_jit_warmup.py``):

1. **Which backend runs.** The warmup's copy resolved an unset or ``auto``
   ``DSV4_MHC_PRE_GEMM_BACKEND`` to ``deepgemm`` unconditionally, while the
   runtime probes for ``deep_gemm.tf32_hc_prenorm_gemm`` and falls back to
   ``tilelang_splitk`` when the build does not export it. On such a build the two
   disagreed and both halves were wrong: warmup drove the DeepGEMM path and
   failed during startup, and the kernel requests actually run was never compiled
   ahead of time.

2. **How many K-splits it runs with.** Every distinct count is a distinct
   ``tl.constexpr``, hence a distinct compile. The warmup re-derived DeepGEMM's
   heuristic without the TileLang divisor clamp, so it named counts the TileLang
   backend never launches -- and warmed exactly one of them.

This module deliberately imports nothing beyond the standard library at module
scope. It is imported by the vendored TileLang wrapper (function-locally, as that
file already does for ``deepgemm_wrapper``) and must stay importable in a CPU-only
test lane, where neither ``tilelang`` nor ``deep_gemm`` is installed.
"""

import functools
import logging
import os

_logger = logging.getLogger(__name__)

BACKEND_ENV = "DSV4_MHC_PRE_GEMM_BACKEND"

# Canonical names. ``deepgemm`` needs an optional DeepGEMM symbol; the two
# TileLang backends are always available wherever the vendored kernels are.
DEEPGEMM = "deepgemm"
TILELANG_SINGLE = "tilelang_single"
TILELANG_SPLITK = "tilelang_splitk"

_ALIASES = {
    "dg": DEEPGEMM,
    "tilelang": TILELANG_SINGLE,
    "single": TILELANG_SINGLE,
    "splitk": TILELANG_SPLITK,
}

# Backends whose split count is shape-dependent; the rest run one K-slice.
_SPLIT_BACKENDS = (DEEPGEMM, TILELANG_SPLITK)

# The TileLang kernel slices K in units of this many elements (its hidden_block),
# so its split count has to divide the resulting block count evenly.
_TILELANG_HIDDEN_BLOCK = 256


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def largest_divisor_le(n: int, want: int) -> int:
    """Largest divisor of ``n`` that is ``<= want`` (and ``>= 1``).

    DeepGEMM's heuristic returns ``num_sms // grid_size`` capped at
    ``num_block_k // 4``, which is not in general a divisor of the TileLang
    kernel's K block count -- at 78 SMs and m=512 it returns 9. The split-K
    kernel needs the K blocks to divide evenly, and rounding down to a divisor
    keeps every block's slice the same length, which is what lets its ``pz`` loop
    stay a compile-time constant.
    """
    d = min(max(int(want), 1), n)
    while d > 1 and n % d:
        d -= 1
    return max(d, 1)


def num_split_for(
    *, m: int, k: int, num_sms: int, block_m: int = 64, block_k: int = 64
) -> int:
    """DeepGEMM's split-K heuristic as a pure function of ``(m, k, num_sms)``.

    Device-free on purpose: the warmup enumerates candidate M values and already
    knows ``num_sms``, so neither caller needs a device query per candidate.
    """
    grid_size = _ceil_div(max(int(m), 1), block_m)
    split_k = max(int(num_sms), 1) // max(grid_size, 1)
    num_block_k = _ceil_div(int(k), block_k)
    split_k = min(split_k, num_block_k // 4)
    return max(split_k, 1)


def resolve_n_splits(
    *, mhc_hidden_size: int, num_tokens: int, num_sms: int, backend: str
) -> int:
    """The split count ``backend`` actually launches for this shape.

    At decode (``num_tokens`` a couple of dozen) the grid is one block and the
    count lands on the cap -- 64 for a 16384-wide ``fn``. At prefill the token
    grid already fills the GPU and the heuristic returns 1, which makes the
    split-K kernel identical in shape to the single-GEMM one.
    """
    if backend not in _SPLIT_BACKENDS:
        return 1
    n_splits = num_split_for(m=num_tokens, k=mhc_hidden_size, num_sms=num_sms)
    if backend == TILELANG_SPLITK:
        n_splits = largest_divisor_le(
            int(mhc_hidden_size) // _TILELANG_HIDDEN_BLOCK, n_splits
        )
    return n_splits


@functools.cache
def has_deepgemm_prenorm() -> bool:
    """Whether the installed DeepGEMM exports the mHC pre-norm GEMM.

    Delegates to the wrapper's own capability probe rather than reading its
    private impl global: that global is populated lazily on the first
    ``tf32_hc_prenorm_gemm`` call, so before then it reads ``None`` even on builds
    that *do* export the symbol -- which is every time this runs, since the
    decision precedes the first call.

    Deliberately not gated on device capability. What decides this is which
    symbols the installed DeepGEMM exports, not which architecture is running: the
    H20 build this PR targets exports none, but a Hopper build that did should
    still use it, and an SM100 build without it should still fall back.
    """
    try:
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            has_tf32_hc_prenorm_gemm,
        )
    except (ImportError, ModuleNotFoundError):
        return False
    return has_tf32_hc_prenorm_gemm()


@functools.cache
def _log_backend_choice(requested: str, resolved: str, probed: bool) -> None:
    """One line per distinct outcome per process, not per call."""
    _logger.info(
        "[DSV4 mHC] pre-norm GEMM backend: requested=%r resolved=%r "
        "(deep_gemm.tf32_hc_prenorm_gemm exported=%s)",
        requested,
        resolved,
        probed,
    )


def resolve_backend() -> str:
    """Canonical backend name for ``DSV4_MHC_PRE_GEMM_BACKEND``.

    An unrecognised value is returned as-is so the runtime raises naming it,
    rather than being silently coerced to a default.
    """
    requested = os.environ.get(BACKEND_ENV, "").strip().lower()
    if requested in ("", "auto"):
        # DeepGEMM's split-K kernel when the build has it: it is the fastest of
        # the three and the SM100 smoke suite validates DSV4 greedy/golden
        # semantics against it. Otherwise tilelang_splitk -- same math as
        # tilelang_single and the same number of kernels, but the K loop is spread
        # over the grid instead of walked by one CUDA block, which is worth ~20%
        # of decode TPOT.
        probed = has_deepgemm_prenorm()
        resolved = DEEPGEMM if probed else TILELANG_SPLITK
    else:
        probed = False
        resolved = _ALIASES.get(requested, requested)
    _log_backend_choice(requested, resolved, probed)
    return resolved
