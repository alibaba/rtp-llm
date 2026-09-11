"""Repository-owned H8 Q-B/RoPE launch over the pinned RTP GEMM headers.

This is an opt-in helper, not a replacement installed into rtp_kernel. Call it
once before capture to build/load the two kernel variants. The existing managed
TORCH_EXTENSIONS_DIR supplies the cross-process build cache and build lock.
"""

import hashlib
import importlib.metadata
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _load_extension():
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm up fused H8 Q-B before CUDA graph capture")
    import rtp_kernel.glm5 as glm5
    from torch.utils.cpp_extension import load

    source_dir = Path(__file__).with_name("glm5_cmp_q_b_src")
    package_include = Path(glm5.__file__).parent / "include"
    includes = [package_include, package_include / "third_party"]
    required = [
        package_include / "rtp_kernel/glm5/q_b_proj.cuh",
        package_include / "rtp_kernel/glm5/deep_gemm/fp8_gemm.cuh",
        includes[1] / "deep_gemm",
        includes[1] / "cutlass",
    ]
    if not all(path.exists() for path in required):
        raise RuntimeError(
            "Fused H8 Q-B requires the pinned RTP wheel's bundled JIT headers"
        )
    # torch's in-process source versioner does not hash included headers. Give
    # each pinned package/header/source combination an explicit module name.
    digest = hashlib.sha256(importlib.metadata.version("rtp_kernel").encode())
    digest.update(torch.__version__.encode())
    digest.update(Path(__file__).read_bytes())
    for path in sorted(package_include.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(package_include)).encode())
            digest.update(path.read_bytes())
    for path in sorted(source_dir.iterdir()):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        raise ValueError(f"Fused H8 Q-B requires SM100/SM103, got {capability}")
    return load(
        name="rtp_glm5_q_b_h8_" + digest.hexdigest()[:16],
        sources=[str(source_dir / "binding.cpp"), str(source_dir / "kernel.cu")],
        extra_include_paths=[str(path) for path in includes],
        extra_cflags=["-O2", "-std=c++17"],
        extra_cuda_cflags=[
            # CUDA13 accepts the header's template-lambda device extension
            # in C++17 mode, avoiding a CCCL concepts/GCC10 host-compiler ICE.
            "-O3",
            "-std=c++17",
            "-gencode=arch=compute_100f,code=sm_100f",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--diag-suppress=39,161,174,177,186,940",
            "--ptxas-options=--register-usage-level=10",
        ],
        extra_ldflags=["-lcuda"],
    )


@lru_cache(maxsize=None)
def _prepare_device(device_index):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "Warm up fused H8 Q-B on this device before CUDA graph capture"
        )
    extension = _load_extension()
    # Materialize both CUDA functions and their dynamic-smem attributes before
    # capture, including when the first captured row count differs from warmup.
    extension.initialize()
    return extension


def q_b_proj_h8(
    activation,
    activation_scale,
    weight,
    weight_scale,
    cos_sin,
    positions,
    *,
    out,
    is_neox_style,
    enable_pdl=None,
):
    """FP8 [M,2048] -> BF16 NoPE [M,8,192] / Q-RoPE [M,8,576].

    M is 1..256. Scales are native packed UE8M0 int32 [M,4]/[2048,4],
    column-major with M rounded to four for activation-scale storage. Inputs
    and outputs must not overlap. Positions are device int32/int64, and their
    values must address cos_sin rows. No host tensor values are read, no KV is
    touched, and the caller-owned Q latent prefix is left unchanged.
    """
    if not isinstance(is_neox_style, bool):
        raise ValueError("is_neox_style must be bool")
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        raise ValueError("out must contain caller-owned NoPE and sparse-query tensors")
    if enable_pdl is None:
        from rtp_kernel.glm5 import get_pdl

        enable_pdl = get_pdl()
    if not isinstance(enable_pdl, bool):
        raise ValueError("enable_pdl must be bool")
    tensors = (
        activation,
        activation_scale,
        weight,
        weight_scale,
        cos_sin,
        positions,
        *out,
    )
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise ValueError("all Q-B operands must be tensors")
    storage_ids = [tensor.untyped_storage().data_ptr() for tensor in tensors]
    for index, storage_id in enumerate(storage_ids[6:]):
        # Conservatively reject shared storage, including disjoint views;
        # read each storage pointer once, using metadata only during capture.
        if storage_id in storage_ids[: 6 + index]:
            raise ValueError("Q-B output storage must not alias any input/output")
    if not activation.is_cuda:
        raise ValueError("activation must be CUDA")
    with torch.cuda.device(activation.device):
        _prepare_device(activation.device.index).q_b_proj_h8(
            activation,
            activation_scale,
            weight,
            weight_scale,
            cos_sin,
            positions,
            out[0],
            out[1],
            is_neox_style,
            enable_pdl,
        )
    return out
