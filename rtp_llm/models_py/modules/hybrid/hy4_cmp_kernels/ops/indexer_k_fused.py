"""Single-launch HY4 WK MX32 / BF16 / LN / RoPE / K-cache candidate.

Diagnostic BF16 boundaries are real epilogue stores included in timing. Static
weights may be prefetched before GDC. Activation and request metadata remain
behind dependency waits. Early notification only permits GDC consumers to start.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib.util import find_spec
import json
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def prepare_extension():
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare_extension must run outside capture")
    from torch.utils.cpp_extension import CUDA_HOME, load

    spec = find_spec("rtp_kernel")
    if spec is None or spec.origin is None or CUDA_HOME is None:
        raise RuntimeError("CUDA13 and pinned RTP kernel headers required")
    include = Path(spec.origin).parent / "glm5/include"
    root = Path(__file__).resolve().parents[1]
    private = root / "cuda/gemm_include"
    version = Path(CUDA_HOME) / "version.json"
    if json.loads(version.read_text())["cuda"]["version"].split(".")[0] != "13":
        raise RuntimeError("CUDA13 required")
    sources = [root / "cuda/indexer_k_fused.cpp", root / "cuda/indexer_k_fused.cu"]
    flags = ["-O3", "-std=c++20", "-lineinfo", "--fmad=true", "--ftz=true",
             "--prec-div=true", "--prec-sqrt=true", "--expt-relaxed-constexpr",
             "--expt-extended-lambda", "-gencode=arch=compute_103a,code=sm_103a",
             "--ptxas-options=--register-usage-level=10",
             "--diag-suppress=39,161,174,177,186,940",
             "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
             "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__"]
    digest = hashlib.sha256(repr(flags).encode() + torch.__version__.encode() + version.read_bytes())
    dependencies = sources + [root / "cuda/indexer_k_fused.cuh"]
    for folder in (private, include / "third_party", include / "rtp_kernel"):
        dependencies += sorted(p for p in folder.rglob("*") if p.is_file())
    for path in dependencies:
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    name = "hy4_indexer_k_fused_" + digest.hexdigest()[:20]
    from torch.utils.cpp_extension import _get_build_directory
    build = Path(_get_build_directory(name, verbose=False))
    build.mkdir(parents=True, exist_ok=True)
    return load(name=name, sources=list(map(str, sources)), build_directory=str(build),
                extra_include_paths=[str(private), str(include), str(include / "third_party"),
                                     str(Path(CUDA_HOME) / "include/cccl")],
                extra_cflags=["-O3", "-std=c++20"], extra_cuda_cflags=flags,
                extra_ldflags=["-lcuda"], verbose=True)


@dataclass(frozen=True)
class Outputs:
    projected_bf16: torch.Tensor
    normed_bf16: torch.Tensor
    key_bf16: torch.Tensor


def allocate_outputs(rows, *, device="cuda"):
    if not 1 <= rows <= 256 or torch.cuda.is_current_stream_capturing():
        raise ValueError("allocate outside capture with 1<=rows<=256")
    return Outputs(*(torch.empty((rows, 128), dtype=torch.bfloat16, device=device) for _ in range(3)))


def _signature(t):
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)


@dataclass(frozen=True)
class Plan:
    handle: object
    tensors: tuple
    signatures: tuple
    metadata: dict


def validate_metadata(positions, cos_sin, slots, cache):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("validate_metadata outside capture required")
    if bool(((positions < 0) | (positions >= cos_sin.shape[0])).any().item()):
        raise ValueError("position out of table bounds")
    if bool((slots >= cache.shape[0] * 64).any().item()):
        raise ValueError("slot beyond cache capacity")
    live = slots[slots >= 0]
    if live.unique().numel() != live.numel():
        raise ValueError("duplicate live cache slots would race")


def _tensors(x8, xs, w8, ws, weight, bias, positions, cos_sin, slots, cache, out):
    if not isinstance(out, Outputs):
        raise TypeError("out must be allocate_outputs result")
    return (x8, xs, w8, ws, weight, bias, positions, cos_sin, slots, cache,
            out.projected_bf16, out.normed_bf16, out.key_bf16)


def prepare_plan(x8, xs, w8, ws, norm_weight, norm_bias, positions, cos_sin, slots, cache, *, out):
    validate_metadata(positions, cos_sin, slots, cache)
    tensors = _tensors(x8, xs, w8, ws, norm_weight, norm_bias, positions, cos_sin, slots, cache, out)
    handle = prepare_extension().prepare_plan(*tensors)
    metadata = dict(handle.metadata())
    metadata.update(fusion="WK_MX32_BF16_LN_RoPE_KQuantCache", epsilon=1e-6,
                    native_scale_floor=1e-4, fallback=False,
                    request_bounds_checked_at_prepare_only=True,
                    core_revision="bdafb874f725dd1aebf587c3b99d8c87ea77d42d")
    return Plan(handle, tensors, tuple(map(_signature, tensors)), metadata)


def run(x8, xs, w8, ws, norm_weight, norm_bias, positions, cos_sin, slots, cache, *, out, plan,
        pdl=False, notify_event=None):
    tensors = _tensors(x8, xs, w8, ws, norm_weight, norm_bias, positions, cos_sin, slots, cache, out)
    signatures = tuple(map(_signature, tensors))
    dynamic = (0, 1, 6, 8, 9)
    if (not isinstance(plan, Plan)
            or any(signatures[i][1:] != plan.signatures[i][1:] for i in dynamic)
            or any(signatures[i] != plan.signatures[i] for i in range(13) if i not in dynamic)):
        raise ValueError("prepared plan pointer/shape/stride/device changed")
    event = 0
    if notify_event is not None:
        event = int(notify_event.cuda_event)
        if not event or notify_event.device != x8.device:
            raise ValueError("notify event must be initialized on input device")
    plan.handle.run(tensors, bool(pdl), event)
    return out
