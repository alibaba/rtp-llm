"""Qualified small-batch SM120 FP8 projections with Torch's scale ordering."""

import functools
import hashlib
from pathlib import Path

import torch

from rtp_llm.models_py.utils.arch import get_sm

_MAX_ROWS = {
    (3072, 5120): 64,
    (4096, 5120): 64,
    (5120, 3072): 64,
    (8192, 5120): 32,
}


@functools.lru_cache(maxsize=1)
def _module():
    from flashinfer.jit import env, gen_jit_spec

    source = Path(__file__).with_name("sm120_fp8_skinny.cu").resolve()
    # FlashInfer retains Ninja files across temporary runfiles and redeploys.
    source_id = hashlib.sha256(
        str(source).encode() + b"\0" + source.read_bytes()
    ).hexdigest()[:16]
    return gen_jit_spec(
        f"rtp_sm120_fp8_skinny_v4_{source_id}",
        [source],
        extra_include_paths=[str(p) for p in env.CUTLASS_INCLUDE_DIRS],
        extra_cuda_cflags=[
            "--ftz=false",
            "--prec-div=true",
            "--prec-sqrt=true",
            "-gencode=arch=compute_120a,code=sm_120a",
        ],
    ).build_and_load()


def skinny_fp8_mm(input, weight, scale_a, scale_b):
    """Return None outside the qualified inference shapes and layouts.

    The caller checks the existing fusion gate. Activation and weight scales
    remain device-resident and may differ between rows and columns.
    """
    if (
        not input.is_cuda
        or input.ndim != 2
        or input.dtype != torch.float8_e4m3fn
        or not input.is_contiguous()
        or weight.ndim != 2
        or weight.dtype != input.dtype
        or weight.device != input.device
        or weight.shape[0] != input.shape[1]
        or weight.stride() != (1, input.shape[1])
        or (weight.shape[1], input.shape[1]) not in _MAX_ROWS
        or not 0 < input.shape[0] <= _MAX_ROWS[(weight.shape[1], input.shape[1])]
        or scale_a.device != input.device
        or scale_a.dtype != torch.float32
        or scale_a.numel() != input.shape[0]
        or not scale_a.is_contiguous()
        or scale_b.device != input.device
        or scale_b.dtype != torch.float32
        or scale_b.numel() != weight.shape[1]
        or not scale_b.is_contiguous()
        or get_sm(input.device) != (12, 0)
    ):
        return None
    output = torch.empty(
        (input.shape[0], weight.shape[1]),
        device=input.device,
        dtype=torch.bfloat16,
    )
    _module().run(input, weight.T, scale_a, scale_b, output)
    return output
