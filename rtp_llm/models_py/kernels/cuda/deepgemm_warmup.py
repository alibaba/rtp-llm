import logging
import os
import time
import torch

logger = logging.getLogger(__name__)

_FP8_GEMM_NT_WARMED: set[tuple[int, int]] = set()  # (N, K)
_GROUPED_GEMM_WARMED: set[tuple[int, int, int]] = set()  # (E, N, K)


def reset_warmed_caches() -> None:
    """Clear warmup dedup caches (for model reload or tests)."""
    _FP8_GEMM_NT_WARMED.clear()
    _GROUPED_GEMM_WARMED.clear()


BLOCK_M = 128

def _get_local_rank_info() -> tuple[int, int]:
    """Return (local_rank, local_world_size) from env vars.

    Same-node ranks share DG_JIT_CACHE_DIR, so we split M values by
    local_rank and let the shared file cache avoid redundant compilations.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    return local_rank, local_world_size


def _split_m_values(
    m_values: list[int], local_rank: int, local_world_size: int
) -> list[int]:
    """Interleaved split: rank i gets m_values[i::world_size]."""
    if local_world_size <= 1:
        return m_values
    return m_values[local_rank::local_world_size]


# ---------------------------------------------------------------------------
# M-value generation
# ---------------------------------------------------------------------------
def _generate_m_values(max_tokens: int, mode: str, n: int) -> list[int]:
    """Generate M values that cover all possible DeepGEMM kernel configurations.

    Reference: https://github.com/deepseek-ai/DeepGEMM/blob/79f48ee/csrc/jit_kernels/heuristics/common.hpp
    """
    if mode == "full":
        return list(range(1, max_tokens + 1))

    block_ms = [64, 128, 256]
    block_ns = list(range(16, min(257, n + 1), 16))
    try:
        num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:
        num_sms = 132  # H20 default

    m_values: set[int] = set()

    # Always include small cases 
    m_values.update([1, 2, 4] + list(range(8, min(65, max_tokens + 1), 8)))

    for block_m in block_ms:
        for block_n in block_ns:
            if block_n > n:
                continue
            # Add M boundaries for wave transitions
            n_blocks = -(-n // block_n)  # cdiv(n, block_n)
            for wave in range(1, 11):
                target_blocks = wave * num_sms
                m = target_blocks * block_m // n_blocks
                if 1 <= m <= max_tokens:
                    m_values.add(m)

            # Add block_m multiples
            for multiple in range(1, max_tokens // block_m + 1):
                m = multiple * block_m
                if m <= max_tokens:
                    m_values.add(m)

    return sorted(m_values)


def _generate_grouped_m_values(
    max_m_per_expert: int, block_m: int = BLOCK_M
) -> list[int]:
    values = list(range(block_m, max_m_per_expert + 1, block_m))
    return values or [block_m]


# ---------------------------------------------------------------------------
# Multi-rank warmup helper
# ---------------------------------------------------------------------------
def _rank_warmup(
    func,
    all_values: list[int],
    my_values: list[int],
    local_world_size: int,
):
    """Phase 1: compile this rank's slice; Phase 2: load the rest from cache."""
    # Phase 1: each rank compiles its own slice (JIT compile → shared cache)
    for v in my_values:
        func(v)
    # Barrier: wait for all ranks to finish Phase 1
    if local_world_size > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()
    # Phase 2: load remaining kernels (all cache hits after barrier)
    if local_world_size > 1:
        remaining = [v for v in all_values if v not in set(my_values)]
        for v in remaining:
            func(v)


# ---------------------------------------------------------------------------
# Scan FP8 linear shapes from LinearBase
# ---------------------------------------------------------------------------
def _find_fp8_linear_weights(
    model: torch.nn.Module,
) -> dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]]:
    from rtp_llm.models_py.modules.factory.linear import LinearBase

    shape_to_tensors: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
    for m in model.modules():
        if not isinstance(m, LinearBase):
            continue
        w = getattr(m, "weight", None)
        ws = getattr(m, "weight_scales", None)
        if w is None or w.dtype != torch.float8_e4m3fn or w.dim() != 2:
            continue
        key = (w.shape[0], w.shape[1])  # (N, K)
        if key not in shape_to_tensors:
            shape_to_tensors[key] = (w, ws)
    return shape_to_tensors


# ---------------------------------------------------------------------------
# Dense FP8 Linear warmup
# ---------------------------------------------------------------------------
def _warmup_fp8_linear(
    model: torch.nn.Module,
    max_tokens: int,
    mode: str,
    local_rank: int,
    local_world_size: int,
):
    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import fp8_gemm_nt

    shape_to_tensors = _find_fp8_linear_weights(model)
    if not shape_to_tensors:
        return

    for (N, K), (w, ws) in shape_to_tensors.items():
        if (N, K) in _FP8_GEMM_NT_WARMED:
            continue

        all_m_values = [
            m for m in _generate_m_values(max_tokens, mode, n=N) if m <= max_tokens
        ]
        my_m_values = _split_m_values(all_m_values, local_rank, local_world_size)

        device = w.device
        dummy_a = torch.empty(max_tokens, K, device=device, dtype=torch.float8_e4m3fn)
        dummy_a_scales = torch.empty(
            max_tokens, K // BLOCK_M, device=device, dtype=torch.float32
        )
        dummy_out = torch.empty(max_tokens, N, device=device, dtype=torch.bfloat16)

        def _warmup_one_m(num_tokens, _a=dummy_a, _as=dummy_a_scales,
                          _w=w, _ws=ws, _out=dummy_out):
            fp8_gemm_nt(
                (_a[:num_tokens], _as[:num_tokens]),
                (_w, _ws),
                _out[:num_tokens],
            )

        _rank_warmup(_warmup_one_m, all_m_values, my_m_values, local_world_size)
        _FP8_GEMM_NT_WARMED.add((N, K))


# ---------------------------------------------------------------------------
# Scan MoE weights from FusedMoe modules
# ---------------------------------------------------------------------------
def _make_dummy_scale(
    shape_prefix: tuple[int, ...], inner_dim: int, use_e8m0: bool, device: torch.device
) -> torch.Tensor:
    """Create dummy input scale tensor for grouped GEMM warmup.

    For e8m0: column-major packed layout via transpose, matching
    create_packed_scale_tensor / deep_gemm expected format.
    """
    groups = inner_dim // BLOCK_M
    if use_e8m0:
        packed = (groups + 3) // 4
        storage = torch.ones(
            (*shape_prefix[:-1], packed, shape_prefix[-1]),
            dtype=torch.int32,
            device=device,
        )
        storage.fill_(0x7F7F7F7F)
        return storage.transpose(-2, -1)
    else:
        return torch.ones(
            (*shape_prefix, groups),
            dtype=torch.float32,
            device=device,
        )


def _find_moe_weights(
    model: torch.nn.Module,
) -> dict[tuple[int, int, int], tuple]:
    """Scan FusedMoe modules, return unique (E, N, K) with weight tensors.

    FusedMoe is an nn.Module so model.modules() finds it directly.
    We access its .fused_experts attribute to extract DeepGEMM executor weights.

    Returns dict of (E, N, K) -> (w1, w1_scale, w2, w2_scale, needs_contiguous, top_k).
    """
    from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import FusedMoe
    from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_hybrid_executor import (
        DeepGemmHybridExecutor,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_masked_executor import (
        DeepGemmMaskedExecutor,
    )

    result: dict[tuple[int, int, int], tuple] = {}
    for m in model.modules():
        if not isinstance(m, FusedMoe):
            continue
        ex = m.fused_experts
        # DeepGemmMaskedExecutor: uses _w1/_w2/_E/_N/_K naming
        if isinstance(ex, DeepGemmMaskedExecutor):
            if not ex._use_fp8:
                continue
            key = (ex._E, ex._N, ex._K)
            if key in result:
                continue
            needs_contiguous = ex.executor_type() == ExecutorType.DEEPGEMM_CONTINUOUS
            top_k = getattr(ex, "top_k", None)
            if top_k is None:
                top_k = getattr(getattr(ex, "config", None), "moe_k", 1)
            result[key] = (ex._w1, ex._w1_scale, ex._w2, ex._w2_scale, needs_contiguous, top_k)
        # DeepGemmHybridExecutor (and V2 which inherits it): uses w13_weight/w2_weight/E/N/K
        # V2.executor_type() returns DEEPGEMM_MASKED so needs_contiguous=False, correct.
        elif isinstance(ex, DeepGemmHybridExecutor):
            key = (ex.E, ex.N, ex.K)
            if key in result:
                continue
            needs_contiguous = ex.executor_type() == ExecutorType.DEEPGEMM_CONTINUOUS
            top_k = getattr(ex, "top_k", None)
            if top_k is None:
                top_k = getattr(getattr(ex, "config", None), "moe_k", 1)
            result[key] = (
                ex.w13_weight, ex.w13_weight_scale_inv,
                ex.w2_weight, ex.w2_weight_scale_inv,
                needs_contiguous, top_k,
            )
    return result


# ---------------------------------------------------------------------------
# MoE grouped GEMM warmup
# ---------------------------------------------------------------------------
def _warmup_moe_grouped_gemm(
    model: torch.nn.Module,
    max_tokens: int,
    mode: str,
    local_rank: int,
    local_world_size: int,
):
    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
        is_deep_gemm_e8m0_used,
        m_grouped_fp8_gemm_nt_contiguous,
        m_grouped_fp8_gemm_nt_masked,
    )

    shape_to_info = _find_moe_weights(model)
    if not shape_to_info:
        return

    use_e8m0 = is_deep_gemm_e8m0_used()
    disable_ue8m0_cast = not use_e8m0

    for (E, N, K), (w1, w1s, w2, w2s, needs_contiguous, top_k) in shape_to_info.items():
        if (E, N, K) in _GROUPED_GEMM_WARMED:
            continue

        device = w1.device
        weight_pairs = [(w1, w1s, K, N), (w2, w2s, N // 2, K)]

        # --- Masked warmup ---
        max_m_per_expert = min((max_tokens + E - 1) // E, max_tokens)
        max_m_aligned = (
            (max_m_per_expert + BLOCK_M - 1) // BLOCK_M
        ) * BLOCK_M or BLOCK_M
        masked_m_values = _generate_grouped_m_values(max_m_aligned, BLOCK_M)

        def _warmup_masked(expected_m, _wp=weight_pairs, _E=E, _dev=device):
            masked_m = torch.full((_E,), expected_m, dtype=torch.int32, device=_dev)
            for w, ws, in_dim, out_dim in _wp:
                x = torch.empty(_E, expected_m, in_dim, dtype=torch.float8_e4m3fn, device=_dev)
                x_scale = _make_dummy_scale((_E, expected_m), in_dim, use_e8m0, _dev)
                out = torch.empty(_E, expected_m, out_dim, dtype=torch.bfloat16, device=_dev)
                m_grouped_fp8_gemm_nt_masked(
                    (x, x_scale), (w, ws), out, masked_m, expected_m,
                    disable_ue8m0_cast=disable_ue8m0_cast,
                )
                del x, x_scale, out

        my_masked = _split_m_values(masked_m_values, local_rank, local_world_size)
        _rank_warmup(_warmup_masked, masked_m_values, my_masked, local_world_size)

        # --- Contiguous warmup (HybridExecutor only) ---
        if needs_contiguous:
            max_m_cont = max_tokens * top_k + E * (BLOCK_M - 1)
            max_m_cont = ((max_m_cont + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
            cont_m_values = list(range(BLOCK_M, max_m_cont + 1, BLOCK_M))

            def _warmup_contiguous(total_tokens, _wp=weight_pairs, _E=E, _dev=device):
                m_indices = torch.arange(total_tokens, dtype=torch.int32, device=_dev) % _E
                for w, ws, in_dim, out_dim in _wp:
                    x = torch.empty(total_tokens, in_dim, dtype=torch.float8_e4m3fn, device=_dev)
                    x_scale = _make_dummy_scale((total_tokens,), in_dim, use_e8m0, _dev)
                    out = torch.empty(total_tokens, out_dim, dtype=torch.bfloat16, device=_dev)
                    m_grouped_fp8_gemm_nt_contiguous(
                        (x, x_scale), (w, ws), out, m_indices,
                        disable_ue8m0_cast=disable_ue8m0_cast,
                    )
                    del x, x_scale, out

            my_cont = _split_m_values(cont_m_values, local_rank, local_world_size)
            _rank_warmup(_warmup_contiguous, cont_m_values, my_cont, local_world_size)

        _GROUPED_GEMM_WARMED.add((E, N, K))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def deep_gemm_warmup(
    model: torch.nn.Module,
    max_tokens: int,
    mode: str = "skip",
):
    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import has_deep_gemm

    _VALID_MODES = {"skip", "relax", "full"}
    if mode not in _VALID_MODES:
        logger.warning(f"Unknown deepgemm_warmup_mode='{mode}', falling back to 'skip'")
        return
    if mode == "skip" or not has_deep_gemm() or model is None:
        return

    local_rank, local_world_size = _get_local_rank_info()
    t0 = time.time()
    _warmup_fp8_linear(model, max_tokens, mode, local_rank, local_world_size)
    _warmup_moe_grouped_gemm(model, max_tokens, mode, local_rank, local_world_size)
    logger.info(f"DeepGEMM warmup completed in {time.time() - t0:.1f}s")
