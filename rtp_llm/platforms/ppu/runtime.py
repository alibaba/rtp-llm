"""PPU SDK adapters shared by every model; importing this module probes no device."""

import fcntl
import importlib
import os
import sys
from functools import wraps


def prepare_model_runtime(model_config, engine_config):
    if model_config.model_type == "deepseek_v4":
        from .models.dsv4.communication import maybe_warmup_ppu_tp_communication

        context = engine_config.module_build_context
        options = (
            context.selection.model_metadata["execution_options"]
            if context is not None
            else os.environ
        )
        maybe_warmup_ppu_tp_communication(
            engine_config.parallelism_config, options=options
        )


def configure_model_weight_loader(model_config, loader):
    config = loader.get_load_config()
    if model_config.model_type == "deepseek_v4" and config.moe_pure_tp_mode:
        from .models.dsv4.resources import routed_tp_preparation

        config.weight_preparation = routed_tp_preparation()


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


def configure_runtime_paths():
    """Restore packaged JIT dependency paths in multiprocessing.spawn workers."""
    for path in os.environ.get("_JIT_CACHE_PATHS", "").split(os.pathsep):
        if path and path not in sys.path:
            sys.path.insert(0, path)


def require_symbol(module_name, symbol_name):
    """Resolve a selected SDK operation, failing at initialization on ABI gaps."""
    configure_runtime_paths()
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
