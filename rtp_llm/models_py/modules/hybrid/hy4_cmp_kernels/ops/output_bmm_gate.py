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
    private = root / "cuda/output_bmm_gate_include"
    sources = [root / "cuda/output_bmm_gate.cpp", root / "cuda/output_bmm_gate.cu"]
    flags = [
        "-O3", "-std=c++20", "-lineinfo", "--fmad=false", "--ftz=false",
        "-gencode=arch=compute_103a,code=sm_103a",
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
        "--ptxas-options=--register-usage-level=10",
        "--diag-suppress=39,161,174,177,186,940",
    ]
    digest = hashlib.sha256(repr(flags).encode() + torch.__version__.encode() + version_file.read_bytes())
    # Hash all transitive template dependencies to avoid stale local extension reuse.
    dependencies = sources + sorted((root / "cuda").glob("output_bmm_gate*.cuh"))
    dependencies += sorted(p for p in private.rglob("*") if p.is_file())
    dependencies += sorted(p for p in (include / "third_party").rglob("*") if p.is_file())
    dependencies += sorted(p for p in (include / "rtp_kernel").rglob("*") if p.is_file())
    for path in dependencies:
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    name = "hy4_output_bmm_gate_" + digest.hexdigest()[:20]
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
    expanded_bf16: torch.Tensor
    q_fp8: torch.Tensor
    q_scale: torch.Tensor


def allocate_outputs(rows, heads=64, *, device="cuda"):
    if not 1 <= rows <= 256 or heads not in (8, 64):
        raise ValueError("requires 1<=rows<=256 and heads in (8,64)")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("allocate outside capture required")
    return Outputs(
        torch.empty((rows, heads*256), device=device, dtype=torch.bfloat16),
        torch.empty((rows, heads*256), device=device, dtype=torch.float8_e4m3fn),
        torch.empty((heads*2, (rows+3)//4*4), device=device, dtype=torch.int32).T[:rows],
    )


def _signature(t):
    return (t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)


@dataclass(frozen=True)
class Plan:
    handle: object
    tensors: tuple
    signatures: tuple
    metadata: dict


def prepare_plan(raw, weight, gate, *, out, tuning="epilogue4"):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare_plan outside capture required")
    if not isinstance(out, Outputs):
        raise TypeError("out must be allocated Outputs")
    tensors = (raw, weight, gate, out.expanded_bf16, out.q_fp8, out.q_scale)
    if tuning not in ("baseline", "epilogue4", "tile64_epilogue4", "epilogue2"):
        raise ValueError("tuning must be baseline, epilogue4, tile64_epilogue4 or epilogue2")
    handle = prepare_extension().prepare(*tensors, tuning)
    return Plan(handle, tensors, tuple(map(_signature, tensors)), dict(handle.metadata()))


def output_bmm_gate(raw, weight, gate, *, out, plan, pdl=False, notify_event=None):
    # Both dynamic raw and gate producers must be dependencies of this stream.
    # Early notification grants launch permission; the O-project consumer must GDC wait.
    tensors = (raw, weight, gate, out.expanded_bf16, out.q_fp8, out.q_scale)
    signatures = tuple(map(_signature, tensors))
    if (any(signatures[i][1:] != plan.signatures[i][1:] for i in (0, 2))
            or any(signatures[i] != plan.signatures[i] for i in (1, 3, 4, 5))):
        raise ValueError("plan tensor identity/layout changed; prepare a new plan outside capture")
    event = 0
    if notify_event is not None:
        event = int(notify_event.cuda_event)
        if not event or notify_event.device != raw.device:
            raise ValueError("notify_event must be initialized on input device")
    plan.handle.run(raw, gate, bool(pdl), event)
    return out
