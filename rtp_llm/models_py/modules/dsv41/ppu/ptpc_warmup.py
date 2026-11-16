"""Opt-in pre-readiness warmup of the resident FP8 PTPC providers.

The generic PPU warmup covers BF16/INT8, not this model's FP8 providers.
Keep the vendor's exact M-dependent selection and use independent, valid
scratch tensors because compile-only mode may still launch on some wheels.
"""

import fcntl
import logging
import os
import time
from pathlib import Path

import torch

from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUPTPCLinear

logger = logging.getLogger(__name__)
MAX_ROWS_ENV = "RTP_V41_PPU_PTPC_WARMUP_MAX_ROWS"


def resident_families(model):
    """Include packed Q/KV weights created outside SharedWeightViews."""
    operations = list(model.shared_weight_views._linears.values())
    for block in model.v4.layers:
        packed = getattr(block.attn, "_local_prefill_qkv", None)
        if packed is not None:
            operations.append(packed._linear)
    dense, grouped = {}, {}
    for op in operations:
        if not isinstance(op, PPUPTPCLinear) or not op.weight.is_cuda:
            continue
        dense.setdefault(tuple(op.weight.shape), op)
        if op._group_direct_gemm is not None:
            grouped.setdefault(tuple(op._group_weight.shape), op)
    return dense, grouped


def warmup_resident_ptpc(model):
    """Run on Prefill rank zero; caller synchronizes the role before serving."""
    maximum = int(os.environ.get(MAX_ROWS_ENV, "0"))
    if maximum < 0:
        raise ValueError(f"{MAX_ROWS_ENV} must be nonnegative")
    if maximum == 0:
        return
    import deep_gemm

    dense, grouped = resident_families(model)
    if not dense:
        raise RuntimeError("PTPC warmup requested without resident CUDA PTPC weights")
    # Serialize the whole enumeration, including the C++ provider whose build
    # function is not covered by the Python JIT compiler's per-build lock.
    root = Path(os.environ.get("DG_JIT_CACHE_DIR", str(Path.home() / ".deep_gemm")))
    root.mkdir(parents=True, exist_ok=True)
    with (root / "rtp_v41_ptpc_warmup.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        original = deep_gemm.get_compile_mode()
        deep_gemm.set_compile_mode(1)
        try:
            with torch.inference_mode():
                for (n, k), op in sorted(dense.items()):
                    started = time.monotonic()
                    lhs = torch.zeros((maximum, k), device=op.weight.device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
                    scales = torch.ones((maximum, 1), device=op.weight.device, dtype=torch.float32)
                    out = torch.empty((maximum, n), device=op.weight.device, dtype=torch.bfloat16)
                    for m in range(1, maximum + 1):
                        op._gemm((lhs[:m], scales[:m]), (op.weight, op.scale), out[:m])
                        if m % 256 == 0:
                            logger.info("PTPC warmup dense N=%d K=%d M=%d/%d", n, k, m, maximum)
                    torch.cuda.synchronize(op.weight.device)
                    logger.info("PTPC warmup dense N=%d K=%d complete %.3fs", n, k, time.monotonic() - started)
                    del lhs, scales, out
                for (groups, n, k), op in sorted(grouped.items()):
                    # The direct per-channel BMM selects configurations in
                    # Python. Deduplicate its exact vendor keys before any
                    # device calls; M remains the unchanged runtime argument.
                    from deep_gemm.jit_kernels.einsum import GemmType, get_best_int8_configs, get_num_sms

                    configs = {}
                    for m in range(1, maximum + 1):
                        key = get_best_int8_configs(m, n, k, groups, get_num_sms(), gemm_type=GemmType.Batched)
                        configs.setdefault(tuple(key[:-1]) + (tuple(key[-1]),), m)
                    for key, m in configs.items():
                        lhs = torch.zeros((groups, m, k), device=op.weight.device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
                        scales = torch.ones((groups, m, 1), device=op.weight.device, dtype=torch.float32)
                        out = torch.empty((m, groups, n), device=op.weight.device, dtype=torch.bfloat16)
                        op._group_direct_gemm(lhs, scales, op._group_weight, op._group_scale, out)
                        torch.cuda.synchronize(op.weight.device)
                    logger.info("PTPC warmup grouped H=%d N=%d K=%d configurations=%d complete", groups, n, k, len(configs))
        finally:
            deep_gemm.set_compile_mode(original)
