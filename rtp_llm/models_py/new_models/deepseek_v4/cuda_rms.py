"""Pin V4's ARM CUDA13 SM100 RMS reduction independently of import order."""

import functools
import platform

import torch


def use_native_rms(device):
    return (
        torch.device(device).type == "cuda"
        and torch.version.cuda is not None
        and torch.version.cuda.split(".", 1)[0] == "13"
        and platform.machine().lower() in ("aarch64", "arm64")
        and torch.cuda.get_device_capability(device) == (10, 0)
    )


@functools.lru_cache(None)
def _native_module():
    from flashinfer.jit.norm import gen_norm_module

    # Same JIT spec and native entry as FlashInfer's CUDA rmsnorm backend.
    # Don't mutate FLASHINFER_USE_CUDA_NORM or process-wide CuTe selection.
    return gen_norm_module().build_and_load()


@torch.library.custom_op("rtp_llm_v4::native_rms_out", mutates_args=("out",))
def _native_out(
    out: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, eps: float, pdl: bool
) -> None:
    _native_module().rmsnorm(out, x, weight, eps, pdl)


@_native_out.register_fake
def _fake_out(out, x, weight, eps, pdl):
    return None


def native_rmsnorm(x, weight, *, eps, out=None):
    from flashinfer.utils import device_support_pdl

    if out is None:
        out = torch.empty_like(x)
    _native_out(out, x, weight, eps, device_support_pdl(x.device))
    return out
