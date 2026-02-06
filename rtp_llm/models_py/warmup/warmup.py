"""Unified kernel warmup entry point for RTP-LLM."""

import logging
import os

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import has_deep_gemm
from rtp_llm.models_py.warmup.deepgemm_warmup import deepgemm_warmup

logger = logging.getLogger(__name__)


def _is_rank0() -> bool:
    """Check if current process is rank 0.

    Returns:
        True if rank 0, False otherwise.
    """
    try:
        from rtp_llm.distribute.worker_info import g_parallel_info

        return g_parallel_info.world_rank == 0
    except (ImportError, AttributeError):
        # If parallel info is not available, assume single process (rank 0)
        return True


def kernel_warmup(
    model: torch.nn.Module,
    max_generate_batch_size: int,
    max_batch_tokens_size: int,
) -> None:
    """Unified kernel warmup entry point.

    This function checks conditions and calls appropriate warmup functions.

    Args:
        model: The PyTorch model to warmup.
        max_generate_batch_size: Maximum batch size for generation (decode).
        max_batch_tokens_size: Maximum batch tokens size for prefill.
    """
    # Check if CUDA is available
    if not torch.cuda.is_available():
        logger.debug("CUDA not available, skipping kernel warmup")
        return

    # Check if DeepGEMM is available
    if not has_deep_gemm():
        logger.debug("DeepGEMM not available, skipping kernel warmup")
        return

    # Check environment variable for warmup mode
    warmup_mode = os.getenv("RTP_LLM_DEEP_GEMM_WARMUP", "relax")
    if warmup_mode == "skip":
        logger.info("DeepGEMM warmup skipped (RTP_LLM_DEEP_GEMM_WARMUP=skip)")
        return

    if warmup_mode not in ("relax", "full"):
        logger.warning(f"Unknown warmup mode: {warmup_mode}, using 'relax' instead")
        warmup_mode = "relax"

    # Calculate max_tokens from parameters
    try:
        max_tokens_raw = max(max_generate_batch_size, max_batch_tokens_size)

        # Apply cap
        cap = int(os.getenv("RTP_LLM_DEEP_GEMM_WARMUP_MAX_TOKENS_CAP", "8192"))
        max_tokens = min(max_tokens_raw, cap)

        logger.info(
            f"DeepGEMM warmup: mode={warmup_mode}, max_tokens={max_tokens} "
            f"(raw={max_tokens_raw}, cap={cap})"
        )
    except Exception as e:
        logger.warning(f"Failed to calculate max_tokens: {e}, using default 2048")
        max_tokens = 2048

    # Get number of workers
    num_workers = int(os.getenv("RTP_LLM_DEEP_GEMM_WARMUP_WORKERS", "4"))

    # Call deepgemm warmup
    # Only show progress bar on rank 0 to avoid cluttered output
    show_progress = _is_rank0()
    try:
        deepgemm_warmup(
            model=model,
            max_tokens=max_tokens,
            mode=warmup_mode,
            num_workers=num_workers,
            show_progress=show_progress,
        )
    except Exception as e:
        logger.error(f"DeepGEMM warmup failed: {e}", exc_info=True)
        # Don't raise - warmup failure shouldn't block model loading
