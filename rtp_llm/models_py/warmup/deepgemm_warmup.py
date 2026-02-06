# SPDX-License-Identifier: Apache-2.0
"""
DeepGEMM JIT warmup implementation with multi-process parallel execution.

This implementation is inspired by vLLM's deep_gemm_warmup.py:
https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/warmup/deep_gemm_warmup.py

Key differences from vLLM:
- Uses multi-process parallel execution instead of sequential execution
- Adapts to RTP-LLM's module structure (CudaFp8DeepGEMMLinear, DeepGemmContinousExecutor, DeepGemmMaskedExecutor)
- Supports RTP-LLM-specific features (UE8M0, masked executor)

The shape selection logic for warmup M values is directly borrowed from vLLM's
_generate_optimal_warmup_m_values function to ensure compatibility.
"""

import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from tqdm import tqdm

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    fp8_gemm_nt,
    is_deep_gemm_e8m0_used,
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nt_masked,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import FusedMoe
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_continous_executor import (
    DeepGemmContinousExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_masked_executor import (
    DeepGemmMaskedExecutor,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import (
    CudaFp8DeepGEMMLinear,
)
from rtp_llm.models_py.utils.math import ceil_div

logger = logging.getLogger(__name__)

# Cache for warmup tracking (shared across processes via Manager)
_warmup_cache_manager = None
_warmup_cache_lock = None
FP8_GEMM_NT_WARMUP_CACHE: Set[Tuple[int, int]] = None
GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE: Set[Tuple[int, int, int]] = None
GROUPED_FP8_GEMM_NT_MASKED_WARMUP_CACHE: Set[Tuple[int, int, int]] = None


def _init_warmup_cache():
    """Initialize shared cache for multi-process warmup."""
    global _warmup_cache_manager, _warmup_cache_lock
    global FP8_GEMM_NT_WARMUP_CACHE, GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE
    global GROUPED_FP8_GEMM_NT_MASKED_WARMUP_CACHE

    if _warmup_cache_manager is None:
        _warmup_cache_manager = multiprocessing.Manager()
        _warmup_cache_lock = _warmup_cache_manager.Lock()
        FP8_GEMM_NT_WARMUP_CACHE = _warmup_cache_manager.dict()
        GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE = _warmup_cache_manager.dict()
        GROUPED_FP8_GEMM_NT_MASKED_WARMUP_CACHE = _warmup_cache_manager.dict()


@dataclass
class WarmupTask:
    """Warmup task definition for multi-process execution."""

    task_type: str  # "fp8_gemm_nt", "grouped_contiguous", "grouped_masked"
    weight_shape: Tuple[int, ...]
    weight_scale_shape: Tuple[int, ...]
    m_values: List[int]
    device_id: int  # CUDA device ID
    disable_ue8m0_cast: bool
    # Additional fields for grouped tasks
    num_experts: Optional[int] = None
    num_topk: Optional[int] = None
    expected_m: Optional[int] = None  # For masked tasks
    max_m: Optional[int] = None  # For grouped contiguous tasks
    expert_ids: Optional[List[int]] = None  # For grouped contiguous tasks


def _generate_optimal_warmup_m_values(
    max_tokens: int, n: int, device: torch.device
) -> List[int]:
    """
    Generate M values that cover all possible DeepGEMM kernel configurations.

    This function is directly borrowed from vLLM's deep_gemm_warmup.py:
    https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/warmup/deep_gemm_warmup.py

    Reference: https://github.com/deepseek-ai/DeepGEMM/blob/79f48ee15a82dd5fad5cd9beaa393c1f755e6b55/csrc/jit_kernels/heuristics/common.hpp

    Args:
        max_tokens: Maximum number of tokens to warmup for
        n: The actual N dimension from the weight tensor
        device: The torch device to get properties from.

    Returns:
        Sorted list of M values to warmup.
    """
    # DeepGEMM's possible block sizes
    block_ms = [64, 128, 256]
    block_ns = list(range(16, min(257, n + 1), 16))
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count

    m_values = set()

    # Always include small cases
    m_values.update([1, 2, 4] + [i for i in range(8, 65, 8)])

    # Collect M values where different wave patterns occur
    for block_m in block_ms:
        for block_n in block_ns:
            if block_n > n:
                continue

            # Add key M boundaries for this block combination
            for wave in range(1, 11):  # Up to 10 waves
                # M where this block config transitions to next wave
                target_blocks = wave * num_sms
                # Use ceil_div (equivalent to vLLM's cdiv)
                m = target_blocks * block_m // ceil_div(n, block_n)
                if 1 <= m <= max_tokens:
                    m_values.add(m)

            # Add block_m boundaries
            for multiple in range(1, max_tokens // block_m + 1):
                m = multiple * block_m
                if m <= max_tokens:
                    m_values.add(m)

    return sorted(m_values)


def _get_fp8_gemm_nt_m_values(w: torch.Tensor, max_tokens: int, mode: str) -> List[int]:
    """
    Get the M values to warmup for a given weight tensor.

    This logic is borrowed from vLLM's _get_fp8_gemm_nt_m_values.

    Args:
        w: Weight tensor (N, K)
        max_tokens: Maximum tokens
        mode: "relax" or "full"

    Returns:
        List of M values to warmup.
    """
    n, _ = w.size()
    device = w.device

    if mode == "relax":
        return _generate_optimal_warmup_m_values(max_tokens, n, device)
    elif mode == "full":
        return list(range(1, max_tokens + 1))
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _get_grouped_gemm_params(
    num_experts: int,
    num_topk: int,
    max_tokens: int,
    block_m: int = 128,
) -> Tuple[int, List[int]]:
    """
    Get parameters for grouped GEMM warmup.

    This function is inspired by vLLM's _get_grouped_gemm_params, adapted for RTP-LLM.

    Args:
        num_experts: Number of experts
        num_topk: Top-k value
        max_tokens: Maximum tokens
        block_m: Block size for M dimension

    Returns:
        Tuple of (MAX_M, expert_ids_list).
    """
    # Simplified version: use max_tokens * num_topk as upper bound
    # In vLLM, this considers DP group size and chunk size, but for warmup we simplify
    MAX_M = min(max_tokens * num_topk, max_tokens * 4)  # Reasonable upper bound
    MAX_M = ((MAX_M + block_m - 1) // block_m) * block_m  # Align to block_m

    # Distribute expert-ids evenly (will be converted to tensor in worker process)
    MAX_BLOCKS = MAX_M // block_m
    expert_ids_block = [i % num_experts for i in range(MAX_BLOCKS)]  # Even distribution
    expert_ids = []
    for expert_id in expert_ids_block:
        expert_ids.extend([expert_id] * block_m)

    return MAX_M, expert_ids[:MAX_M]


def _rtp_llm_fp8_linear_may_use_deep_gemm(module: torch.nn.Module) -> bool:
    """Check if the module uses DeepGEMM for FP8 Linear.

    Args:
        module: PyTorch module to check.

    Returns:
        True if module uses DeepGEMM.
    """
    return isinstance(module, CudaFp8DeepGEMMLinear)


def _extract_data_from_rtp_llm_linear(
    module: torch.nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor, bool]:
    """Extract weights and weight scales from RTP-LLM Linear module.

    Args:
        module: CudaFp8DeepGEMMLinear module.

    Returns:
        Tuple of (weight, weight_scales, disable_ue8m0_cast).
    """
    assert isinstance(module, CudaFp8DeepGEMMLinear)
    w = module.weight  # Shape: (N, K)
    ws = module.weight_scales  # Shape: (scale_N, scale_K)
    disable_ue8m0_cast = not module.scale_ue8m0
    return w, ws, disable_ue8m0_cast


def _rtp_llm_fused_moe_may_use_deep_gemm(module: torch.nn.Module) -> bool:
    """Check if the FusedMoe module uses DeepGEMM executor.

    Args:
        module: PyTorch module to check.

    Returns:
        True if module uses DeepGEMM executor.
    """
    if not isinstance(module, FusedMoe):
        return False

    executor = module.fused_experts
    return isinstance(executor, (DeepGemmContinousExecutor, DeepGemmMaskedExecutor))


def _extract_data_from_rtp_llm_fused_moe(
    module: torch.nn.Module,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    str,
    bool,
    Optional[int],
]:
    """Extract weights and config from RTP-LLM FusedMoe module.

    Args:
        module: FusedMoe module.

    Returns:
        Tuple of (w13, w13_scale, w2, w2_scale, num_topk, executor_type,
                  disable_ue8m0_cast, expected_m).
    """
    assert isinstance(module, FusedMoe)
    executor = module.fused_experts

    if isinstance(executor, DeepGemmContinousExecutor):
        w13 = executor.w13_weight  # Shape: (num_experts, N, K)
        w13_scale = executor.w13_weight_scale_inv
        w2 = executor.w2_weight  # Shape: (num_experts, K, N//2)
        w2_scale = executor.w2_weight_scale_inv
        num_topk = executor.top_k
        disable_ue8m0_cast = not is_deep_gemm_e8m0_used()
        return (
            w13,
            w13_scale,
            w2,
            w2_scale,
            num_topk,
            "grouped_contiguous",
            disable_ue8m0_cast,
            None,
        )
    elif isinstance(executor, DeepGemmMaskedExecutor):
        w1 = executor._w1  # Shape: (E, N, K)
        w1_scale = executor._w1_scale
        w2 = executor._w2  # Shape: (E, K, N//2)
        w2_scale = executor._w2_scale
        # For masked executor, we need expected_m
        # Use a reasonable default based on typical usage
        expected_m = 128  # Default expected_m for warmup
        disable_ue8m0_cast = not is_deep_gemm_e8m0_used()
        # Get num_topk from config
        num_topk = executor.config.moe_k if hasattr(executor, "config") else 2
        return (
            w1,
            w1_scale,
            w2,
            w2_scale,
            num_topk,
            "grouped_masked",
            disable_ue8m0_cast,
            expected_m,
        )
    else:
        raise ValueError(f"Unsupported executor type: {type(executor)}")


def _generate_warmup_tasks(
    model: torch.nn.Module, max_tokens: int, mode: str
) -> List[WarmupTask]:
    """Generate all warmup tasks from the model.

    Args:
        model: PyTorch model to scan.
        max_tokens: Maximum tokens for warmup.
        mode: "relax" or "full".

    Returns:
        List of warmup tasks.
    """
    tasks: List[WarmupTask] = []
    seen_shapes: Set[Tuple[str, Tuple[int, ...]]] = set()

    # Scan for FP8 Linear modules
    for module in model.modules():
        if _rtp_llm_fp8_linear_may_use_deep_gemm(module):
            w, ws, disable_ue8m0_cast = _extract_data_from_rtp_llm_linear(module)
            n, k = w.shape
            cache_key = ("fp8_gemm_nt", (n, k))

            if cache_key in seen_shapes:
                continue
            seen_shapes.add(cache_key)

            # Check cache
            with _warmup_cache_lock:
                if (n, k) in FP8_GEMM_NT_WARMUP_CACHE:
                    continue

            m_values = _get_fp8_gemm_nt_m_values(w, max_tokens, mode)
            if m_values:
                device_id = w.device.index if w.device.index is not None else 0
                tasks.append(
                    WarmupTask(
                        task_type="fp8_gemm_nt",
                        weight_shape=(n, k),
                        weight_scale_shape=ws.shape,
                        m_values=m_values,
                        device_id=device_id,
                        disable_ue8m0_cast=disable_ue8m0_cast,
                    )
                )

    # Scan for FusedMoe modules
    for module in model.modules():
        if _rtp_llm_fused_moe_may_use_deep_gemm(module):
            (
                w1_or_w13,
                w1_or_w13_scale,
                w2,
                w2_scale,
                num_topk,
                executor_type,
                disable_ue8m0_cast,
                expected_m,
            ) = _extract_data_from_rtp_llm_fused_moe(module)

            if executor_type == "grouped_contiguous":
                # w13 shape: (num_experts, N, K)
                num_experts, n, k = w1_or_w13.shape
                cache_key = ("grouped_contiguous", (num_experts, n, k))

                if cache_key in seen_shapes:
                    continue
                seen_shapes.add(cache_key)

                # Check cache
                with _warmup_cache_lock:
                    if (
                        num_experts,
                        n,
                        k,
                    ) in GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE:
                        continue

                # Get grouped gemm params (inspired by vLLM)
                block_m = 128
                max_m, expert_ids = _get_grouped_gemm_params(
                    num_experts, num_topk, max_tokens, block_m
                )
                m_values = list(range(block_m, max_m + 1, block_m))
                if not m_values:
                    m_values = [block_m]

                device_id = (
                    w1_or_w13.device.index if w1_or_w13.device.index is not None else 0
                )

                # Add w13 task
                tasks.append(
                    WarmupTask(
                        task_type="grouped_contiguous",
                        weight_shape=(num_experts, n, k),
                        weight_scale_shape=w1_or_w13_scale.shape,
                        m_values=m_values,
                        device_id=device_id,
                        disable_ue8m0_cast=disable_ue8m0_cast,
                        num_experts=num_experts,
                        num_topk=num_topk,
                        max_m=max_m,
                        expert_ids=expert_ids,
                    )
                )

                # Add w2 task (different shape)
                w2_n = w2.shape[2]  # N//2
                cache_key_w2 = ("grouped_contiguous", (num_experts, w2_n, k))
                if cache_key_w2 not in seen_shapes:
                    seen_shapes.add(cache_key_w2)
                    with _warmup_cache_lock:
                        if (
                            num_experts,
                            w2_n,
                            k,
                        ) not in GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE:
                            tasks.append(
                                WarmupTask(
                                    task_type="grouped_contiguous",
                                    weight_shape=(num_experts, w2_n, k),
                                    weight_scale_shape=w2_scale.shape,
                                    m_values=m_values,
                                    device_id=device_id,
                                    disable_ue8m0_cast=disable_ue8m0_cast,
                                    num_experts=num_experts,
                                    num_topk=num_topk,
                                    max_m=max_m,
                                    expert_ids=expert_ids,
                                )
                            )

            elif executor_type == "grouped_masked":
                # w1 shape: (E, N, K)
                num_experts, n, k = w1_or_w13.shape
                cache_key = ("grouped_masked", (num_experts, n, k))

                if cache_key in seen_shapes:
                    continue
                seen_shapes.add(cache_key)

                # Check cache
                with _warmup_cache_lock:
                    if (num_experts, n, k) in GROUPED_FP8_GEMM_NT_MASKED_WARMUP_CACHE:
                        continue

                # For masked, use key expected_m values
                if expected_m is None:
                    expected_m = 128
                m_values = [64, 128, 256]  # Key expected_m values
                m_values = [m for m in m_values if m <= max_tokens]
                if not m_values:
                    m_values = [64]

                device_id = (
                    w1_or_w13.device.index if w1_or_w13.device.index is not None else 0
                )

                # Add w1 task
                tasks.append(
                    WarmupTask(
                        task_type="grouped_masked",
                        weight_shape=(num_experts, n, k),
                        weight_scale_shape=w1_or_w13_scale.shape,
                        m_values=m_values,
                        device_id=device_id,
                        disable_ue8m0_cast=disable_ue8m0_cast,
                        num_experts=num_experts,
                        num_topk=num_topk,
                        expected_m=expected_m,
                    )
                )

                # Add w2 task
                w2_n = w2.shape[2]  # N//2
                cache_key_w2 = ("grouped_masked", (num_experts, w2_n, k))
                if cache_key_w2 not in seen_shapes:
                    seen_shapes.add(cache_key_w2)
                    with _warmup_cache_lock:
                        if (
                            num_experts,
                            w2_n,
                            k,
                        ) not in GROUPED_FP8_GEMM_NT_MASKED_WARMUP_CACHE:
                            tasks.append(
                                WarmupTask(
                                    task_type="grouped_masked",
                                    weight_shape=(num_experts, w2_n, k),
                                    weight_scale_shape=w2_scale.shape,
                                    m_values=m_values,
                                    device_id=device_id,
                                    disable_ue8m0_cast=disable_ue8m0_cast,
                                    num_experts=num_experts,
                                    num_topk=num_topk,
                                    expected_m=expected_m,
                                )
                            )

    return tasks


def _execute_warmup_task_worker(task_dict: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Worker function to execute a warmup task in a separate process.

    This function runs in a worker process and initializes CUDA context for that process.

    Args:
        task_dict: Dictionary representation of WarmupTask (for pickling).

    Returns:
        Tuple of (success, error_message).
    """
    import torch

    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
        fp8_gemm_nt,
        m_grouped_fp8_gemm_nt_contiguous,
        m_grouped_fp8_gemm_nt_masked,
    )
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
        create_per_token_group_quant_fp8_output_scale,
    )

    try:
        # Initialize CUDA device in this process
        device = torch.device(f"cuda:{task_dict['device_id']}")
        torch.cuda.set_device(device)

        task_type = task_dict["task_type"]
        weight_shape = tuple(task_dict["weight_shape"])
        weight_scale_shape = tuple(task_dict["weight_scale_shape"])
        m_values = task_dict["m_values"]
        disable_ue8m0_cast = task_dict["disable_ue8m0_cast"]

        if task_type == "fp8_gemm_nt":
            n, k = weight_shape
            block_m = 128
            max_m = max(m_values) if m_values else 128

            # Create dummy tensors
            w = torch.empty((n, k), device=device, dtype=torch.float8_e4m3fn)
            ws = torch.empty(weight_scale_shape, device=device, dtype=torch.float32)
            a1q = torch.empty((max_m, k), device=device, dtype=torch.float8_e4m3fn)

            if disable_ue8m0_cast:
                a1q_scales = torch.empty(
                    (max_m, k // block_m), device=device, dtype=torch.float32
                )
            else:
                a1q_scales = create_per_token_group_quant_fp8_output_scale(
                    x_shape=(max_m, k),
                    device=device,
                    group_size=block_m,
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=True,
                )
                a1q_scales.fill_(0x7F7F7F7F)

            out = torch.empty((max_m, n), device=device, dtype=torch.bfloat16)

            # Warmup for each M value
            for m in m_values:
                fp8_gemm_nt(
                    (a1q[:m], a1q_scales[:m]),
                    (w, ws),
                    out[:m],
                    c=None,
                    disable_ue8m0_cast=disable_ue8m0_cast,
                )

            return True, ""

        elif task_type == "grouped_contiguous":
            num_experts, n, k = weight_shape
            block_m = 128
            max_m = task_dict.get("max_m", max(m_values) if m_values else block_m)
            expert_ids_list = task_dict.get("expert_ids", [])

            # Create dummy tensors
            w = torch.empty(
                (num_experts, n, k), device=device, dtype=torch.float8_e4m3fn
            )
            scale_dtype = torch.int32 if not disable_ue8m0_cast else torch.float32
            ws = torch.empty(weight_scale_shape, device=device, dtype=scale_dtype)
            a1q = torch.empty((max_m, k), device=device, dtype=torch.float8_e4m3fn)

            if disable_ue8m0_cast:
                a1q_scales = torch.empty(
                    (max_m, k // block_m), device=device, dtype=torch.float32
                )
            else:
                a1q_scales = create_per_token_group_quant_fp8_output_scale(
                    x_shape=(max_m, k),
                    device=device,
                    group_size=block_m,
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=True,
                )
                a1q_scales.fill_(0x7F7F7F7F)

            out = torch.empty((max_m, n), device=device, dtype=torch.bfloat16)

            # Convert expert_ids list to tensor
            expert_ids = torch.tensor(
                expert_ids_list[:max_m], device=device, dtype=torch.int32
            )

            # Warmup for each M value
            for m in m_values:
                m_indices = expert_ids[:m]
                m_grouped_fp8_gemm_nt_contiguous(
                    (a1q[:m], a1q_scales[:m]),
                    (w, ws),
                    out[:m],
                    m_indices,
                    disable_ue8m0_cast=disable_ue8m0_cast,
                )

            return True, ""

        elif task_type == "grouped_masked":
            num_experts, n, k = weight_shape
            expected_m = task_dict.get("expected_m", 128)

            # Create dummy tensors
            w = torch.empty(
                (num_experts, n, k), device=device, dtype=torch.float8_e4m3fn
            )
            scale_dtype = torch.int32 if not disable_ue8m0_cast else torch.float32
            ws = torch.empty(weight_scale_shape, device=device, dtype=scale_dtype)

            max_m = expected_m
            a1q = torch.empty(
                (num_experts, max_m, k), device=device, dtype=torch.float8_e4m3fn
            )

            if disable_ue8m0_cast:
                a1q_scales = torch.empty(
                    (num_experts, max_m, k // 128), device=device, dtype=torch.float32
                )
            else:
                a1q_scales = create_per_token_group_quant_fp8_output_scale(
                    x_shape=(num_experts, max_m, k),
                    device=device,
                    group_size=128,
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=True,
                )
                a1q_scales.fill_(0x7F7F7F7F)

            out = torch.empty(
                (num_experts, max_m, n), device=device, dtype=torch.bfloat16
            )

            # Create masked_m tensor
            masked_m = torch.full(
                (num_experts,), expected_m, device=device, dtype=torch.int32
            )

            # Warmup for each expected_m value
            for m in m_values:
                masked_m.fill_(m)
                m_grouped_fp8_gemm_nt_masked(
                    (a1q[:, :m], a1q_scales[:, :m]),
                    (w, ws),
                    out[:, :m],
                    masked_m,
                    m,
                    disable_ue8m0_cast=disable_ue8m0_cast,
                )

            return True, ""

        else:
            return False, f"Unknown task type: {task_type}"

    except Exception as e:
        return False, str(e)


def deepgemm_warmup(
    model: torch.nn.Module,
    max_tokens: int,
    mode: str = "relax",
    num_workers: int = 4,
    show_progress: bool = True,
) -> None:
    """
    Perform DeepGEMM warmup with multi-process parallel execution.

    This function generates warmup tasks and executes them in parallel using
    multiple processes. Each process initializes its own CUDA context.

    The shape selection logic is borrowed from vLLM's deep_gemm_warmup.py.

    Args:
        model: PyTorch model to warmup.
        max_tokens: Maximum tokens for warmup.
        mode: "relax" or "full" (default: "relax").
        num_workers: Number of parallel worker processes (default: 4).
        show_progress: Whether to show progress bar (default: True).
    """
    # Initialize shared cache
    _init_warmup_cache()

    # Generate all warmup tasks
    tasks = _generate_warmup_tasks(model, max_tokens, mode)

    if not tasks:
        logger.info("No DeepGEMM warmup tasks found")
        return

    logger.info(
        f"DeepGEMM warmup: {len(tasks)} tasks, mode={mode}, "
        f"max_tokens={max_tokens}, workers={num_workers}"
    )

    # Adjust num_workers
    num_workers = min(num_workers, len(tasks), os.cpu_count() or 1)

    # Execute tasks in parallel using ProcessPoolExecutor
    success_count = 0
    fail_count = 0
    failed_tasks: List[Tuple[str, str]] = []

    # Create progress bar (only if show_progress)
    pbar = tqdm(total=len(tasks), desc="DeepGEMM warmup", disable=not show_progress)

    # Convert tasks to dictionaries for pickling
    task_dicts = []
    for task in tasks:
        task_dict = {
            "task_type": task.task_type,
            "weight_shape": task.weight_shape,
            "weight_scale_shape": task.weight_scale_shape,
            "m_values": task.m_values,
            "device_id": task.device_id,
            "disable_ue8m0_cast": task.disable_ue8m0_cast,
            "num_experts": task.num_experts,
            "num_topk": task.num_topk,
            "expected_m": task.expected_m,
            "max_m": task.max_m,
            "expert_ids": task.expert_ids,
        }
        task_dicts.append(task_dict)

    # Use spawn method for multiprocessing (required for CUDA)
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(_execute_warmup_task_worker, task_dict): task_dict
            for task_dict in task_dicts
        }

        # Process completed tasks
        for future in as_completed(future_to_task):
            task_dict = future_to_task[future]
            try:
                success, error_msg = future.result()
                if success:
                    success_count += 1
                    # Update cache
                    weight_shape = tuple(task_dict["weight_shape"])
                    task_type = task_dict["task_type"]
                    with _warmup_cache_lock:
                        if task_type == "fp8_gemm_nt":
                            FP8_GEMM_NT_WARMUP_CACHE[weight_shape] = True
                        elif task_type == "grouped_contiguous":
                            GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE[
                                weight_shape
                            ] = True
                        elif task_type == "grouped_masked":
                            GROUPED_FP8_GEMM_NT_MASKED_WARMUP_CACHE[weight_shape] = True
                else:
                    fail_count += 1
                    failed_tasks.append((task_type, str(weight_shape)))
                    if error_msg:
                        logger.debug(
                            f"Task failed: {task_type} {weight_shape}: {error_msg}"
                        )
            except Exception as e:
                fail_count += 1
                failed_tasks.append(
                    (task_dict["task_type"], str(task_dict["weight_shape"]))
                )
                logger.warning(
                    f"Task exception: {task_dict['task_type']} {task_dict['weight_shape']}: {e}"
                )

            pbar.update(1)

    pbar.close()

    # Summary
    logger.info(
        f"DeepGEMM warmup completed: {success_count} succeeded, "
        f"{fail_count} failed out of {len(tasks)} tasks"
    )
    if failed_tasks:
        logger.warning(f"Failed tasks: {failed_tasks[:10]}")  # Show first 10
