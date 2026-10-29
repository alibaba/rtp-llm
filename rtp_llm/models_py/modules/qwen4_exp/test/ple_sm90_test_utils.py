"""Independent PLE consumer arithmetic and invocation-bound diagnostics."""

import contextlib
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.hybrid.gated_residual import grouped_rms_norm


def save(name, record):
    path = os.path.join(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"], name + ".json")
    with open(path, "w") as handle:
        json.dump(record, handle, indent=2)


def sha(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def metadata():
    import triton

    p = torch.cuda.get_device_properties(0)
    cublas_lt = {"status": "NOT_LOADED"}
    maps = Path("/proc/self/maps")
    if maps.exists():
        paths = {
            line.split()[-1]
            for line in maps.read_text().splitlines()
            if "libcublasLt.so" in line and line.split()[-1].startswith("/")
        }
        if len(paths) == 1:
            path = paths.pop()
            library = ctypes.CDLL(path)
            get_version = library.cublasLtGetVersion
            get_version.restype = ctypes.c_size_t
            get_version.argtypes = []
            cublas_lt = {
                "status": "LOADED",
                "path": path,
                "version": int(get_version()),
            }
        elif paths:
            cublas_lt = {"status": "MULTIPLE_LOADED", "paths": sorted(paths)}
    return {
        "torch_git_version": torch.version.git_version,
        "cublas_lt": cublas_lt,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "device": p.name,
        "capability": list(torch.cuda.get_device_capability()),
        "uuid": str(getattr(p, "uuid", "unavailable")),
        "fixture": "synthetic; not checkpoint activations",
    }


def exact(actual, expected):
    if actual is None:
        return {"pass": False, "reason": "unexpected fallback"}
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {
            "pass": False,
            "reason": "shape/dtype mismatch",
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
            "actual_dtype": str(actual.dtype),
            "expected_dtype": str(expected.dtype),
        }
    a, b = actual.float(), expected.float()
    mismatch = actual != expected
    count = int(mismatch.sum().item())
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    record = {
        "pass": count == 0 and finite,
        "mismatches": count,
        "elements": actual.numel(),
        "finite": finite,
        "max_abs": float((a - b).abs().max().item()) if a.numel() else 0.0,
    }
    if count:
        index = mismatch.nonzero()[0].tolist()
        record["first_index"] = index
        record["actual"] = float(a[tuple(index)].item())
        record["expected"] = float(b[tuple(index)].item())
    return record


def reference_gate(key, value, hidden, gk, gq, gc, eps=1e-6):
    # Frozen arithmetic from the pre-integration _gated_values consumer.
    kn = grouped_rms_norm(key, gk, 2560, eps).unflatten(-1, (4, 2560))
    qn = grouped_rms_norm(hidden, gq, 2560, eps).unflatten(-1, (4, 2560))
    gate = (kn * qn).sum(-1, keepdim=True) / math.sqrt(2560)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated = (torch.sigmoid(gate) * value.unsqueeze(-2)).flatten(-2)
    return gated, grouped_rms_norm(gated, gc, 2560, eps)


def reference_conv(gated, normed, initial, weight, residual=None):
    if initial is None:
        initial = torch.zeros(
            normed.shape[0], 9, 10240, device=normed.device, dtype=normed.dtype
        )
    history = torch.cat((initial, normed), 1)
    conv = F.silu(
        F.conv1d(history.transpose(1, 2), weight, groups=10240, dilation=3)
    ).transpose(1, 2)
    output = gated + conv
    return output if residual is None else residual + output


@contextlib.contextmanager
def capture_kernel(module, name):
    original, calls = getattr(module, name), []

    class Capture:
        def __getitem__(self, grid):
            launcher = original[grid]

            def launch(*args, **kwargs):
                result = launcher(*args, **kwargs)
                meta = getattr(result, "metadata", None)
                calls.append(
                    {
                        "name": name,
                        "grid": list(grid),
                        "compiled_hash": str(getattr(result, "hash", "unavailable")),
                        "n_regs": getattr(result, "n_regs", None),
                        "shared_bytes": getattr(meta, "shared", None),
                    }
                )
                return result

            return launch

    with patch.object(module, name, Capture()):
        yield calls
