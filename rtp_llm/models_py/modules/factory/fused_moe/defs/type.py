from enum import Enum


class RouterType(Enum):
    """Router implementation type

    Ordered by communication efficiency (higher value = better performance).
    """

    BATCHED_DATA = 0  # Basic batched data router
    DEEPGEMM_CONTINUOUS = 1  # DeepGEMM continuous router
    DEEPEP_NORMAL = 2  # DeepEP normal mode
    DEEPEP_LOW_LATENCY = 4  # DeepEP low latency mode (best communication)
    PURE_TP = 5  # optimize when EP=TP, use all_reduce as gather


class ExecutorType(Enum):
    """Executor implementation type

    Ordered by computation efficiency (higher value = better performance).
    """

    BATCHED_TRITON = 0  # Triton-based batched executor
    FUSED_MOE = 2  # ROCm fused MoE executor (same level)
    DEEPGEMM_CONTINUOUS = 1  # DeepGEMM continuous executor
    DEEPGEMM_MASKED = 2  # DeepGEMM masked executor
    CUTLASS_FP8 = 3  # Cutlass FP8 executor (specialized)
    CUTLASS_BATCHED_FP8 = 4  # Cutlass batched FP8 (most optimized)
