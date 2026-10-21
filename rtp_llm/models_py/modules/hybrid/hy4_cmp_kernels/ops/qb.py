"""HY4 group32 QB GEMM + BF16 materialization + interleaved Q RoPE.

Only transformed[...,512:] is written; the absorbed BMM owns its first512.
Positions must stay in [0,cos_sin.shape[0]) on every replay. Request metadata
validation belongs outside capture. Early notifications require a GDC consumer.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib.util import find_spec
import json
from pathlib import Path
from types import ModuleType
import torch

_EXTENSION = None

def _runtime_include():
    spec = find_spec("rtp_kernel")
    if spec is None or spec.origin is None:
        raise RuntimeError("pinned rtp_kernel headers required")
    return Path(spec.origin).parent / "glm5/include"

@lru_cache(maxsize=1)
def prepare_extension() -> ModuleType:
    global _EXTENSION
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare_extension must run outside CUDA Graph capture")
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("CUDA13 toolkit is required")
    version_file = Path(CUDA_HOME) / "version.json"
    if not version_file.is_file():
        raise RuntimeError("CUDA toolkit version.json is required for explicit CUDA13 eligibility")
    toolkit_version = json.loads(version_file.read_text())["cuda"]["version"]
    if toolkit_version.split(".", 1)[0] != "13":
        raise RuntimeError(f"CUDA13 toolkit is required; selected CUDA_HOME={CUDA_HOME} ({toolkit_version})")
    root = Path(__file__).resolve().parents[1]
    include = _runtime_include()
    private = root / "cuda/qb_include"
    sources = [root / "cuda/qb.cpp", root / "cuda/qb.cu"]
    flags = [
        "-O3", "-std=c++20", "-lineinfo",
        "-gencode=arch=compute_103a,code=sm_103a",
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "--ptxas-options=--register-usage-level=10",
        "--diag-suppress=39,161,174,177,186,940",
    ]
    digest = hashlib.sha256(repr(flags).encode() + torch.__version__.encode() + version_file.read_bytes())
    # Hash all transitive template dependencies to avoid stale local extension reuse.
    dependencies = sources + sorted((root / "cuda").glob("qb*.cuh"))
    dependencies += sorted(p for p in private.rglob("*") if p.is_file())
    dependencies += sorted(p for p in (include / "third_party").rglob("*") if p.is_file())
    dependencies += sorted(p for p in (include / "rtp_kernel").rglob("*") if p.is_file())
    for path in dependencies:
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    name = "hy4_qb_rope_" + digest.hexdigest()[:20]
    from torch.utils.cpp_extension import _get_build_directory
    build = Path(_get_build_directory(name, verbose=False))
    build.mkdir(parents=True, exist_ok=True)
    _EXTENSION = load(
        name=name,
        sources=[str(p) for p in sources],
        build_directory=str(build),
        extra_include_paths=[str(private), str(include), str(include / "third_party"),
                             str(Path(CUDA_HOME) / "include/cccl")],
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=flags,
        extra_ldflags=["-lcuda"],
        verbose=True,
    )
    return _EXTENSION


@dataclass(frozen=True)
class Outputs:
    q_bf16: torch.Tensor
    transformed: torch.Tensor


def allocate_outputs(rows, heads=64, *, device="cuda"):
    if rows not in (1, 2, 4, 8, 16, 17, 32, 64, 128, 256) or heads not in (8, 64):
        raise ValueError("unsupported HY4 rows/heads")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("allocate_outputs must run outside capture")
    return Outputs(torch.empty((rows, heads, 256), dtype=torch.bfloat16, device=device),
                   torch.empty((rows, heads, 576), dtype=torch.bfloat16, device=device))


def _signature(t):
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)


@dataclass(frozen=True)
class Plan:
    handle: object
    tensors: tuple
    signatures: tuple
    metadata: dict


def prepare_plan(q8, qs, w8, ws, positions, cos_sin, out, *, heads=64):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare_plan must run outside capture")
    if not isinstance(out, Outputs):
        raise TypeError("out must be allocate_outputs result")
    # One explicit metadata validation before capture, no hidden replay sync.
    if positions.numel() == 0 or positions.min().item() < 0 or positions.max().item() >= cos_sin.shape[0]:
        raise ValueError("positions out of cos_sin bounds")
    tensors = (q8, qs, w8, ws, positions, cos_sin, out.q_bf16, out.transformed)
    handle = prepare_extension().prepare_plan(*tensors, heads)
    metadata = dict(handle.metadata())
    metadata.update({"fusion": "QB_MX32_BF16_QRoPE_direct_PE", "fallback": False,
                     "kv_write": False, "absorbed_bmm": False, "neox": False,
                     "positions_bounds_validated_at_prepare_only": True,
                     "core_revision": "bdafb874f725dd1aebf587c3b99d8c87ea77d42d"})
    return Plan(handle, tensors, tuple(map(_signature, tensors)), metadata)


def qb_rope(q8, qs, w8, ws, positions, cos_sin, *, out, plan, pdl=False, notify_event=None):
    if not isinstance(plan, Plan) or not isinstance(out, Outputs):
        raise ValueError("explicit prepared plan and allocated outputs required")
    tensors = (q8, qs, w8, ws, positions, cos_sin, out.q_bf16, out.transformed)
    signatures = tuple(map(_signature, tensors))
    if (signatures[4][1:] != plan.signatures[4][1:]
            or any(signatures[i] != plan.signatures[i] for i in (0,1,2,3,5,6,7))):
        raise ValueError("plan pointer/shape/stride/dtype/device changed")
    event = 0
    if notify_event is not None:
        event = int(notify_event.cuda_event)
        if not event or notify_event.device != q8.device:
            raise ValueError("notify_event must be initialized on input device before capture")
    plan.handle.run(positions, bool(pdl), event)
    return out
