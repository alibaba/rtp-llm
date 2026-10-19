"""External DeepGEMM wo_a + group32 output quantization for Blackwell.

Requires deep_gemm.epilogue.FP8Group32. Missing API, unsupported input or cold
CUDA capture falls back to the caller's BF16 path. Import is CUDA-lazy.
"""

from __future__ import annotations

import ctypes
import logging
import threading
from functools import lru_cache

import torch

_READY = {}
_LOCK = threading.Lock()


@lru_cache(maxsize=16)
def _device_geometry(index):
    prop = torch.cuda.get_device_properties(index)
    return prop.major, prop.minor, prop.multi_processor_count


def _device_supported(device):
    if device.type != "cuda":
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    major, _, sms = _device_geometry(index)
    # The external backend selects the kernel and cluster geometry. Retain the
    # qualified Blackwell device envelope of the wo_a integration.
    return major == 10 and sms >= 2 and sms % 2 == 0


def _weight_supported(weight, scale):
    return (
        weight.is_cuda
        and weight.shape == (8, 1024, 4096)
        and weight.dtype == torch.float8_e4m3fn
        and weight.is_contiguous()
        and scale.device == weight.device
        and scale.dtype == torch.int32
        and scale.shape == (8, 1024, 32)
        and scale.stride() == (32768, 1, 1024)
        and all(not t.requires_grad and t.data_ptr() % 16 == 0 for t in (weight, scale))
        and _device_supported(weight.device)
    )


def is_supported(a, weight):
    aq, sa = a
    w, sw = weight
    if aq.ndim != 3 or not 0 < aq.shape[0] < 2**31 - 128:
        return False
    m = aq.shape[0]
    stride = (m + 3) // 4 * 4
    return (
        aq.device == sa.device == w.device
        and aq.dtype == torch.float8_e4m3fn
        and aq.shape == (m, 8, 4096)
        # DeepGEMM encodes both runtime strides in its A-side TMA map.
        # The inverse-RoPE quantizer emits group-major [G,M,K]; native mega
        # attention emits token-major [M,G,K]. Both keep K contiguous and
        # require no copy or different CUDA specialization. Other padded,
        # overlapping or broadcast layouts remain outside the contract.
        and aq.stride() in ((4096, m * 4096, 1), (32768, 4096, 1))
        and sa.dtype == torch.int32
        and sa.shape == (m, 8, 32)
        and sa.stride() == (1, stride * 32, stride)
        and all(not t.requires_grad and t.data_ptr() % 16 == 0 for t in (aq, sa))
        and _weight_supported(w, sw)
    )


def _external_api():
    # Feature detection replaces the private vendor-header SHA/ABI checks.
    try:
        import deep_gemm
    except ImportError:
        return None
    if not (
        hasattr(getattr(deep_gemm, "epilogue", None), "FP8Group32")
        and callable(getattr(deep_gemm, "get_jit_config_fingerprint", None))
    ):
        return None
    return deep_gemm


@lru_cache(maxsize=1)
def _driver():
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    driver.cuCtxGetCurrent.restype = ctypes.c_int
    return driver


def _context():
    current = ctypes.c_void_p()
    if _driver().cuCtxGetCurrent(ctypes.byref(current)):
        return None
    return current.value


def _output(m, device):
    q = torch.empty((m, 8192), dtype=torch.float8_e4m3fn, device=device)
    padded_m = (m + 3) // 4 * 4
    s = torch.empty((64, padded_m), dtype=torch.int32, device=device).T
    return q, s if padded_m == m else s[:m]


def _check_output(out, m, device, inputs):
    q, sf = out
    if not (
        q.shape == (m, 8192)
        and q.dtype == torch.float8_e4m3fn
        and q.is_contiguous()
        and sf.shape == (m, 64)
        and sf.dtype == torch.int32
        and sf.stride() == (1, (m + 3) // 4 * 4)
        and all(
            t.device == device and not t.requires_grad and t.data_ptr() % 16 == 0
            for t in out
        )
    ):
        raise ValueError("Invalid wo_a FP8 output or packed group32 scale layout")
    # Outputs cannot alias inputs (or each other), even through dtype views.
    storage = [t.untyped_storage().data_ptr() for t in (*inputs, *out)]
    if storage[-1] in storage[:-1] or storage[-2] in storage[:-2]:
        raise ValueError("wo_a outputs must not alias inputs or each other")


def is_ready(weight, scale):
    """Never import, load or compile a missing backend from the serving path."""
    entry = _READY.get(weight.device.index)
    if entry is None or not _weight_supported(weight, scale):
        return False
    with torch.cuda.device(weight.device):
        return entry["context"] == _context()


def _signature(a, api):
    # DeepGEMM owns the list of public configuration knobs affecting its JIT.
    return (a[0].shape[0], a[0].stride(), api.get_jit_config_fingerprint())


def _launch(entry, a, weight, out):
    api = entry["api"]
    api.fp8_einsum(
        "bhr,hdr->bhd",
        a,
        weight,
        out[0].view(-1, 8, 1024),
        recipe=(1, 1, 32),
        epilogue=api.epilogue.FP8Group32(out[1]),
    )
    stream = torch.cuda.current_stream(a[0].device)
    for tensor in (*a, *weight, *out):
        tensor.record_stream(stream)


def try_grouped_quant(a, weight, *, out=None):
    """Return FP8 + packed group32 scales, or None for the BF16 fallback.

    Eager calls after startup may JIT additional shapes. CUDA capture only
    launches signatures already prepared eagerly; it never compiles here.
    Output quantization always uses legacy, independent of global quant knobs.
    Inputs must be finite. Outputs are caller-owned and may be reused.
    """
    if not is_supported(a, weight) or a[0].device.index not in _READY:
        return None
    aq, _ = a
    entry = _READY[aq.device.index]
    with torch.cuda.device(aq.device):
        if entry["context"] != _context():
            return None
        signature = _signature(a, entry["api"])
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and signature not in entry["prepared"]:
            return None
        if out is None:
            out = _output(aq.shape[0], aq.device)
        _check_output(out, aq.shape[0], aq.device, (*a, *weight))
        _launch(entry, a, weight, out)
        if not capturing:
            entry["prepared"].add(signature)
    return out


@torch.inference_mode()
def warmup(weight, scale):
    """Prepare all decode M=1..512, both layouts, and prefill anchors.

    Other prefill shapes must run eagerly before capture. A cold signature
    safely falls back; no startup compilation is hidden inside capture.
    """
    if not _weight_supported(weight, scale):
        return False
    device = weight.device
    with torch.cuda.device(device), _LOCK:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("V41 wo_a JIT warmup must precede CUDA graph capture")
        if device.index in _READY:
            return is_ready(weight, scale)
        api = _external_api()
        if api is None:
            logging.info(
                "[DSV41 WoAQuant] external FP8Group32 unavailable; using BF16 fallback"
            )
            return False
        context = _context()
        if context is None:
            return False
        entry = dict(api=api, context=context, prepared=set())
        rows = (*range(1, 513), 4096, 12288)
        # Publish readiness only after all warmup launches have succeeded.
        for m in rows:
            aligned = (m + 3) // 4 * 4
            scale_a = (
                torch.full(
                    (8, 32, aligned), 0x7F7F7F7F, device=device, dtype=torch.int32
                )
                .transpose(1, 2)[:, :m]
                .transpose(0, 1)
            )
            out = _output(m, device)
            for token_major in (False, True):
                q = torch.zeros(
                    (m, 8, 4096) if token_major else (8, m, 4096),
                    device=device,
                    dtype=torch.float8_e4m3fn,
                )
                if not token_major:
                    q = q.transpose(0, 1)
                a = (q, scale_a)
                _launch(entry, a, (weight, scale), out)
                entry["prepared"].add(_signature(a, api))
        torch.cuda.current_stream(device).synchronize()
        _READY[device.index] = entry
        logging.info(
            "[DSV41 WoAQuant] external DeepGEMM ready device=%s signatures=%d",
            device,
            len(entry["prepared"]),
        )
    return True
