"""Deterministic M890P HC PRE adapter over the existing DeepGEMM kernel.

HcPrenormGemm already supports disjoint split-K output planes. Use that
specialization and PyTorch's fixed-axis reduction instead of global FP32
atomicAdd. No GEMM, quantization, or HC-mix kernel is reimplemented here.
Explicit opt-in via DSV4_MHC_PRE_GEMM_BACKEND=deepgemm_deterministic.
"""

import logging
from functools import lru_cache

import torch

_TEMPLATE = """
using gemm_t = deep_gemm::HcPrenormGemm<
    {N}, {K}, {BLOCK_M}, {BLOCK_N}, {BLOCK_K}, {NUM_SPLITS},
    {FAST_BF16_TO_TF32}, false>;
gemm_t::run(d, sqr_sum, m, a, b, stream, num_sms, smem_size);
"""


def launch_geometry(m: int, k: int, n: int) -> dict:
    """Match the frozen PPU DeepGEMM TF32 HC launch policy, not its atomics."""
    if m <= 0 or k != 16384 or n != 24:
        raise ValueError(
            "deterministic PPU HC requires M>0, K=16384, N=24 (DSV4 Flash)"
        )
    block_k = 128 if m <= 256 else 64
    splits = 64 if m <= 256 else (32 if m <= 512 or m >= 8192 else 16)
    return {
        "N": n,
        "K": k,
        "BLOCK_M": 64 if m <= 256 or (m <= 4096 and k <= 8192) else 128,
        "BLOCK_N": min(((n + 7) // 8) * 8, 32),
        "BLOCK_K": block_k,
        "NUM_SPLITS": max(1, min(splits, k // block_k)),
        "FAST_BF16_TO_TF32": "true" if m <= 256 or k <= 8192 or m >= 8192 else "false",
    }


@lru_cache(maxsize=None)
def _require_device(index: int) -> None:
    name = torch.cuda.get_device_name(index)
    if name != "ZW-M890P":
        raise RuntimeError(f"deterministic PPU HC requires ZW-M890P, got {name}")
    logging.info(
        "DSV4_HC_BACKEND deepgemm_deterministic: existing DeepGEMM partials + torch.sum"
    )


def tf32_hc_prenorm_gemm(x, weight, out, sqrsum, requested_splits=None):
    """Fill the existing single-plane FP32 output ABI on the current stream.

    Like the vendor PPU function, actual split count is selected from M/K, not
    requested_splits. Scratch is call-owned (no cross-stream singleton); graph
    capture retains stable allocation addresses through the normal graph pool.
    """
    config = _check_inputs(x, weight)
    m, n = x.shape[0], weight.shape[0]
    tensors = (x, weight, out, sqrsum)
    if any(
        t.device != x.device or not t.is_cuda or not t.is_contiguous() for t in tensors
    ):
        raise ValueError("HC tensors must be contiguous and on the same PPU")
    if any(t.dtype != torch.float32 for t in (out, sqrsum)):
        raise TypeError("HC requires FP32 outputs")
    if tuple(out.shape) != (1, m, n) or tuple(sqrsum.shape) != (1, m):
        raise ValueError("HC outputs must retain the single-plane ABI")
    storage = [t.untyped_storage().data_ptr() for t in tensors]
    if len(set(storage)) != len(storage):
        raise ValueError("HC input, weight and outputs must not alias")
    partials, squares = _launch_partials(x, weight, config)
    torch.sum(partials, dim=0, out=out[0])
    torch.sum(squares, dim=0, out=sqrsum[0])
    return out


def _check_inputs(x, weight):
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("HC input and weight must be matrices")
    m, k = x.shape
    n, weight_k = weight.shape
    if k != weight_k:
        raise ValueError("HC input/weight K mismatch")
    config = launch_geometry(m, k, n)
    tensors = (x, weight)
    if any(
        t.device != x.device or not t.is_cuda or not t.is_contiguous() for t in tensors
    ):
        raise ValueError("HC tensors must be contiguous and on the same PPU")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.float32:
        raise TypeError("HC requires BF16 input and FP32 weight")
    storage = [t.untyped_storage().data_ptr() for t in tensors]
    if len(set(storage)) != len(storage):
        raise ValueError("HC input and weight must not alias")
    _require_device(x.device.index)
    return config


def tf32_hc_prenorm_partials(x, weight):
    """Expose existing split planes for consumers with their own reduction."""
    return _launch_partials(x, weight, _check_inputs(x, weight))


def _launch_partials(x, weight, config):

    from deep_gemm.jit_kernels.tuner import jit_tuner
    from deep_gemm.jit_kernels.utils import get_num_sms

    splits = config["NUM_SPLITS"]
    m, n = x.shape[0], weight.shape[0]
    partials = torch.empty((splits, m, n), dtype=torch.float32, device=x.device)
    squares = torch.empty((splits, m), dtype=torch.float32, device=x.device)
    smem = 2 * (
        config["BLOCK_M"] * (config["BLOCK_K"] + 8) * 2
        + config["BLOCK_N"] * (config["BLOCK_K"] + 4) * 4
    )
    args = (
        x,
        weight,
        partials,
        squares,
        m,
        torch.cuda.current_stream(x.device),
        get_num_sms(),
        smem,
    )
    runtime = jit_tuner.compile_and_tune(
        # Vendor's in-memory tuner omits template from its key. A distinct
        # name is mandatory: otherwise it can return the cached atomic kernel.
        name="rtp_dsv4_hc_prenorm_partials",
        keys=config,
        space=(),
        includes=('"../deep_gemm/tf32_hc_prenorm_gemm.cuh"',),
        arg_defs=(
            ("a", torch.bfloat16),
            ("b", torch.float32),
            ("d", torch.float32),
            ("sqr_sum", torch.float32),
            ("m", int),
            ("stream", torch.cuda.Stream),
            ("num_sms", int),
            ("smem_size", int),
        ),
        template=_TEMPLATE,
        args=args,
        jit_include_dir="cutlass3",
    )
    runtime(*args)
    return partials, squares
