"""GLM5 CMP producers for DeepGEMM's group-32 MXFP8 indexer contract."""

from functools import lru_cache
from pathlib import Path
from types import ModuleType

import torch


@lru_cache(maxsize=1)
def prepare() -> ModuleType:
    """Compile before CUDA Graph capture; reuse PyTorch's persistent JIT cache."""
    import rtp_kernel.glm5
    from torch.utils.cpp_extension import load

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("GLM5 MXFP8 producers must be prepared before graph capture")
    major, _ = torch.cuda.get_device_capability()
    if major != 10:
        raise RuntimeError("GLM5 MXFP8 fused K producer requires SM10.x")
    package = Path(rtp_kernel.glm5.__file__).resolve().parent
    include = package / "include"
    dependencies = include / "third_party"
    if (
        not (dependencies / "deep_gemm").is_dir()
        or not (dependencies / "cute").is_dir()
    ):
        raise RuntimeError("rtp-kernel wheel is missing bundled GLM5 CUDA headers")
    source = Path(__file__).resolve().parent / "cuda"
    return load(
        name="rtp_glm5_cmp_mxfp8_sm10",
        sources=[
            str(source / "glm5_cmp_mxfp8.cpp"),
            str(source / "glm5_cmp_mxfp8_q.cu"),
            str(source / "glm5_cmp_mxfp8_k.cu"),
        ],
        extra_include_paths=[str(include), str(dependencies)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-gencode=arch=compute_100f,code=sm_100f",
        ],
        extra_ldflags=["-lcuda"],
    )


def indexer_q_rope_quant(
    projected: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    gate_raw: torch.Tensor,
    *,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    notify_event: torch.cuda.Event | None = None,
    pdl: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    event_handle = 0 if notify_event is None else int(notify_event.cuda_event)
    if notify_event is not None and event_handle == 0:
        raise RuntimeError("notify_event must be initialized before use")
    prepare().indexer_q_rope_quant(
        projected,
        cos_sin,
        positions,
        gate_raw,
        *out,
        event_handle,
        pdl,
    )
    return out


def indexer_k_cache(
    activation: torch.Tensor,
    activation_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    cache: torch.Tensor,
    *,
    epsilon: float,
    pdl: bool = True,
) -> None:
    prepare().indexer_k_cache(
        activation,
        activation_scale,
        weight,
        weight_scale,
        norm_weight,
        norm_bias,
        cos_sin,
        positions,
        slot_mapping,
        cache,
        epsilon,
        pdl,
    )
