"""PPU SDK adapters shared by every model; importing this module probes no device."""

import fcntl
import importlib
import os
from functools import wraps


class PpuStreamPool:
    """Auxiliary streams owned by one model provider, shared across its layers.

    Callers must prepare each role before capture and join its work before
    consuming results. No stream or device state is created on construction.
    """

    def __init__(self):
        self._streams = {}

    def get(self, role, device):
        import torch

        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("PPU auxiliary streams require a CUDA/PPU device")
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        key = (index, role)
        if key not in self._streams:
            with torch.cuda.device(index):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("PPU streams must be prepared before capture")
                self._streams[key] = torch.cuda.Stream(device=index)
        return self._streams[key]


def require_ppu_op(namespace, name):
    import torch

    from rtp_llm.ops import ensure_compute_ops_loaded

    ensure_compute_ops_loaded()
    try:
        return getattr(getattr(torch.ops, namespace), name)
    except AttributeError as error:
        raise RuntimeError(
            f"PPU operator {namespace}::{name} requires an RTP-LLM build with --config=ppu"
        ) from error


def native_row_gemv_enabled():
    return os.getenv("DSV4_PPU_DECODE_ROW_GEMV_NATIVE", "0") == "1"


def try_fp32_rowwise_linear(x, weight, stream_pool, lanes):
    """Run independent row GEMVs and join before returning the contiguous output."""
    import torch

    if (
        stream_pool is None
        or x.ndim != 2
        or x.shape[0] not in (3, 24)
        or x.dtype not in (torch.bfloat16, torch.float32)
        or not x.is_cuda
        or not x.is_contiguous()
        or weight.dtype != torch.float32
        or weight.device != x.device
        or not weight.is_contiguous()
        or (torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad))
        or torch.backends.cuda.matmul.allow_tf32
        or torch.cuda.get_device_name(x.device) != "ZW-M890P"
    ):
        return None

    if (
        native_row_gemv_enabled()
        and x.shape[0] == 24
        and x.shape[1] == 4096
        and weight.ndim == 2
        and weight.shape[1] == 4096
        and weight.shape[0] in (256, 1024, 2048)
        and x.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
    ):
        rows_per_cta = 2 if x.shape[0] == 24 and weight.shape[0] == 256 else 4
        return torch.ops.fp32_row_probe.forward(x.float(), weight, rows_per_cta)

    if (
        native_row_gemv_enabled()
        and x.shape[0] == 24
        and x.shape[1] == 16384
        and weight.shape == (24, 16384)
        and x.data_ptr() % 8 == 0
        and weight.data_ptr() % 8 == 0
    ):
        return torch.ops.hc_projection_probe.forward(x.float(), weight, 4)

    rows = x.float().split(1)
    output = torch.empty(
        (len(rows), weight.shape[0]), device=x.device, dtype=torch.float32
    )
    destinations = output.split(1)
    transposed = weight.t()
    main = torch.cuda.current_stream(x.device)
    streams = [
        stream_pool.get(("rowwise_gemv", lane), x.device)
        for lane in range(min(lanes, len(rows)))
    ]
    try:
        for lane, stream in enumerate(streams):
            stream.wait_stream(main)
            with torch.cuda.stream(stream):
                for index in range(lane, len(rows), len(streams)):
                    torch.mm(rows[index], transposed, out=destinations[index])
    finally:
        for stream in streams:
            main.wait_stream(stream)
    return output


def require_symbol(module_name, symbol_name):
    """Resolve a selected SDK operation, failing at initialization on ABI gaps."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise RuntimeError(f"PPU requires the SDK package {module_name}") from error
    symbol = getattr(module, symbol_name, None)
    if not callable(symbol):
        raise RuntimeError(f"PPU requires callable {module_name}.{symbol_name}")
    return symbol


_SERIALIZED_BUILD_MARKER = "_rtp_llm_cross_process_serialized"


def install_deep_gemm_build_lock(compiler=None) -> None:
    """Serialize DeepGEMM cache misses shared by local worker processes.

    DeepGEMM atomically publishes ``kernel.so`` but does not lock the compile
    step.  On an EP/CP launch every rank can therefore invoke nvcc for the same
    new shape at once.  Holding one cache-local flock around ``build`` makes
    the first rank compile and lets waiters re-run DeepGEMM's filesystem cache
    check after the artifact has been published.
    """

    patch_bound_references = compiler is None
    if patch_bound_references:
        compiler = importlib.import_module("deep_gemm.jit.compiler")
    original_build = compiler.build
    if getattr(original_build, _SERIALIZED_BUILD_MARKER, False):
        return

    @wraps(original_build)
    def serialized_build(*args, **kwargs):
        cache_dir = compiler.get_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        lock_path = os.path.join(cache_dir, ".rtp_llm_build.lock")
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            return original_build(*args, **kwargs)
        finally:
            os.close(fd)

    setattr(serialized_build, _SERIALIZED_BUILD_MARKER, True)
    compiler.build = serialized_build
    if patch_bound_references:
        # DeepGEMM imports ``build`` by value in both of these modules, so
        # replacing compiler.build alone would leave the hot path unlocked.
        importlib.import_module("deep_gemm.jit").build = serialized_build
        importlib.import_module("deep_gemm.jit_kernels.tuner").build = serialized_build
