"""Flash1017 checkpoint arithmetic, before concatenation or backend packing.

All conversion happens on CPU in bounded row chunks. In particular the FP32
reciprocal of 448 is rounded before multiplication: dividing by 448 with a
different backend can produce a scale one ULP away from the reference file.
Native MXFP4 is never decoded and re-encoded by this module.
"""

import hashlib
import json
import os
from pathlib import Path
import threading

import torch

FP8_INV_MAX = 0.0022321429569274187  # float32(1 / 448)
_audit_lock = threading.Lock()
_audit_writers = {}
_audit_identity = None


def audit_loaded_tensor(name, tensor, rank=0, *, layout=False):
    """Optional validation receipt from the actual loader's output buffer."""
    directory = os.environ.get("RTP_V41_PPU_WEIGHT_AUDIT_DIR")
    if not directory:
        return
    root = Path(directory).resolve()
    payload = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, payload.numel(), 8 * 1024 * 1024):
        digest.update(payload[start:start + 8 * 1024 * 1024].cpu().numpy().tobytes())
    dtype = {
        torch.float8_e4m3fn: "F8_E4M3", torch.float8_e8m0fnu: "F8_E8M0",
        torch.float32: "F32", torch.bfloat16: "BF16", torch.uint8: "U8", torch.int8: "I8",
        torch.int32: "I32", torch.int64: "I64", torch.float16: "F16",
        torch.int16: "I16", torch.uint16: "U16", torch.uint32: "U32", torch.uint64: "U64",
    }[tensor.dtype]
    global _audit_identity
    if _audit_identity is None:
        manifest = Path(__file__).resolve().parents[5] / ".v41_snapshot.json"
        _audit_identity = json.loads(manifest.read_text()) if manifest.is_file() else {}
    record = dict(name=name, dtype=dtype, shape=list(tensor.shape),
                  sha256=digest.hexdigest(), rank=int(rank),
                  origin="production_backend_layout" if layout else "production_loader_output",
                  source_sha256=_audit_identity.get("source_sha256"))
    if layout:
        record.update(stride=list(tensor.stride()), storage_offset=tensor.storage_offset())
    with _audit_lock:
        key = (str(root), os.getpid(), int(rank), layout)
        if key not in _audit_writers:
            root.mkdir(parents=True, exist_ok=True)
            kind = "layouts" if layout else "weights"
            _audit_writers[key] = (root / f"{kind}-rank{rank:05d}.jsonl").open("x")
        writer = _audit_writers[key]
        writer.write(json.dumps(record) + "\n")
        writer.flush()


def decode_block32_rows(weight, scale, start=0, end=None):
    if weight.ndim != 2:
        raise ValueError("Flash1017 dense checkpoint weights must be [N, K]")
    n, k = weight.shape
    end = n if end is None else end
    if tuple(scale.shape) != ((n + 31) // 32, (k + 31) // 32):
        raise ValueError("Flash1017 source scale must describe 32x32 blocks")
    if scale.element_size() != 1:
        raise TypeError("Flash1017 block scales must retain UE8M0 bytes")
    values = weight[start:end].contiguous().view(torch.float8_e4m3fn).float()
    rows = torch.arange(start, end, device=scale.device) // 32
    exponents = scale.view(torch.uint8).index_select(0, rows)
    scales = torch.exp2(exponents.float() - 127).repeat_interleave(32, 1)[:, :k]
    return values * scales


def convert_dense_ptpc(weight, scale, *, chunk_rows=128):
    """Return checkpoint-order E4M3 [N,K] and FP32 [N,1] scale.

    Already-channelwise input passes through without another quantization.
    Keep CPU arithmetic identical for CPU and accelerator model loading.
    """
    if weight.ndim != 2 or weight.element_size() != 1:
        raise TypeError("PTPC requires a byte-sized [N, K] FP8 weight")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    weight, scale = weight.cpu(), scale.cpu()
    n, k = weight.shape
    if scale.dtype == torch.float32:
        if tuple(scale.shape) != (n, 1):
            raise ValueError("PTPC checkpoint scales must be FP32 [N, 1]")
        return weight.view(torch.float8_e4m3fn), scale
    result = torch.empty((n, k), dtype=torch.uint8)
    scales = torch.empty((n, 1), dtype=torch.float32)
    for start in range(0, n, chunk_rows):
        end = min(n, start + chunk_rows)
        values = decode_block32_rows(weight, scale, start, end)
        amax = values.abs().amax(-1, keepdim=True).clamp_min(1e-12)
        row_scale = amax * FP8_INV_MAX
        quantized = (values / row_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        result[start:end].copy_(quantized.view(torch.uint8))
        scales[start:end].copy_(row_scale)
    return result.view(torch.float8_e4m3fn), scales


def convert_engram_wkv(weight, scale, *, chunk_rows=128):
    if scale is None:
        if weight.dtype != torch.bfloat16:
            raise TypeError("unscaled Engram wkv must already be BF16")
        return weight
    weight, scale = weight.cpu(), scale.cpu()
    result = torch.empty(weight.shape, dtype=torch.bfloat16)
    for start in range(0, weight.shape[0], chunk_rows):
        end = min(weight.shape[0], start + chunk_rows)
        result[start:end].copy_(decode_block32_rows(weight, scale, start, end))
    return result
