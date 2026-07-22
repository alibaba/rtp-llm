import ctypes
import glob
import logging
import os
import site
import sys
import traceback
from typing import List

import torch


def _preload_nvidia_deps():
    """Preload nvidia wheel libraries that torch doesn't cover.

    For regular uv/pip install, RPATH in our .so resolves to
    site-packages/nvidia/*/lib/ automatically.  For editable install
    the .so stays in the repo dir so RPATH can't resolve.

    Same pattern as torch._preload_cuda_deps / torch._load_global_deps.
    Harmless if libs are already loaded or not installed.
    """
    _NVIDIA_DEPS = {
        "nvtx": "libnvtx*.so*",
        "cuda_cupti": "libcupti.so*",
        "cudnn": "libcudnn.so.*[0-9]",
        "nccl": "libnccl.so*",
        "cusparselt": "libcusparseLt.so*",
        "cufile": "libcufile.so*",
    }
    search_paths = []
    try:
        usp = site.getusersitepackages()
        if isinstance(usp, str):
            search_paths.append(usp)
    except Exception:
        pass
    try:
        search_paths.extend(site.getsitepackages())
    except Exception:
        pass

    for folder, pattern in _NVIDIA_DEPS.items():
        for sp in search_paths:
            lib_dir = os.path.join(sp, "nvidia", folder, "lib")
            if not os.path.isdir(lib_dir):
                continue
            matches = glob.glob(os.path.join(lib_dir, pattern))
            if matches:
                try:
                    ctypes.CDLL(matches[0], mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break


_preload_nvidia_deps()

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
libs_path = os.path.join(parent_dir, "libs")
SO_NAME = "libth_transformer_config.so"


def _find_so_in_bazel_bin() -> str:
    """Dev / `bazel test` fallback: locate SO_NAME under the workspace bazel-bin
    tree (setup.py does not populate libs/ in those flows). Returns the
    containing directory, or "" if not found / not a bazel workspace."""
    bazel_bin = os.path.normpath(os.path.join(parent_dir, "..", "bazel-bin"))
    if not os.path.isdir(bazel_bin):
        return ""
    for root, _, files in os.walk(bazel_bin):
        if SO_NAME in files:
            return root
    return ""


# All .so files are in rtp_llm/libs/ (copied by setup.py during uv/pip install).
so_path = libs_path
_so_available = os.path.exists(os.path.join(so_path, SO_NAME))
if not _so_available:
    # Restore the dev/bazel-test fallback removed in the pyproject migration.
    bazel_so_path = _find_so_in_bazel_bin()
    if bazel_so_path:
        so_path = bazel_so_path
        _so_available = True

if _so_available:
    logging.info(f"so path: {so_path}")
    sys.path.append(so_path)
elif os.environ.get("RTP_LLM_ALLOW_MISSING_SO") == "1":
    # Explicit collection-only mode (e.g. pytest collection with no build).
    logging.warning(
        f"{SO_NAME} not found in {libs_path} or bazel-bin; running collection-only "
        f"(RTP_LLM_ALLOW_MISSING_SO=1). C++ extensions are unavailable."
    )
else:
    # Fail fast by default so a missing/broken build is not silently degraded.
    raise ImportError(
        f"{SO_NAME} not found in {libs_path} or bazel-bin. Build the C++ extensions "
        f"(e.g. `pip install -e .` or `bazel build ...`), or set "
        f"RTP_LLM_ALLOW_MISSING_SO=1 to allow collection-only mode."
    )


# hack for amd rocm 6.3.0.2 test, libcaffe2_nvrtc.so should have been automatically loaded via torch
try:
    logging.info(f"torch path: {torch.__path__}")
    so_load_path = f"{torch.__path__[0]}/lib/libcaffe2_nvrtc.so"
    if os.path.exists(so_load_path):
        from ctypes import cdll

        cdll.LoadLibrary(so_load_path)
        logging.info(f"loaded libcaffe2_nvrtc.so from {so_load_path}")
except BaseException as e:
    logging.info(f"Exception: {e}, traceback: {traceback.format_exc()}")

# frontend cannot load libpython, so we need to load it manually
import sys
import sysconfig
from ctypes import cdll

try:
    _pyver = f"{sys.version_info.major}.{sys.version_info.minor}"
    cdll.LoadLibrary(sysconfig.get_config_var("LIBDIR") + f"/libpython{_pyver}.so")
except (OSError, TypeError):
    pass


# Stub used for frontend / standalone / collection-only modes where the C++
# extension is unavailable. Defined before the import blocks so they can fall
# back to it.
class EmptyClass:
    def __init__(self, **kwargs):
        pass


# Symbols imported from libth_transformer_config; stubbed with EmptyClass in
# collection-only mode (RTP_LLM_ALLOW_MISSING_SO=1) when the .so is missing.
_LIBTH_CONFIG_SYMBOLS = [
    "ArpcConfig", "AttentionConfigs", "GrpcConfig", "BatchDecodeSchedulerConfig",
    "CacheStoreConfig", "ConcurrencyConfig", "DeviceResourceConfig", "EplbMode",
    "FfnDisAggregateConfig", "FIFOSchedulerConfig", "FMHAConfig", "HWKernelConfig",
    "KVCacheConfig", "MiscellaneousConfig", "MlaOpsType", "ModelConfig",
    "ModelSpecificConfig", "MoeConfig", "NcclCommConfig", "PDSepConfig",
    "ParallelismConfig", "ProfilingDebugLoggingConfig", "RopeCache", "RopeConfig",
    "RopeStyle", "TaskType", "VitConfig", "VitSeparation", "check_rope_cache",
    "get_rope_cache", "get_rope_cache_once", "CPRotateMethod", "PrefillCPConfig",
    "QuantAlgo", "RoleType", "RuntimeConfig", "SpecialTokens",
    "SpeculativeExecutionConfig", "SpeculativeType", "EPLBConfig", "ActivationType",
    "DataType", "KvCacheDataType", "HybridAttentionConfig", "HybridAttentionType",
    "LinearAttentionConfig", "MultimodalInput", "MMPreprocessConfig",
    "EplbConfig", "cpp_get_block_cache_keys",
]

try:
    from libth_transformer_config import (
        ArpcConfig,
        AttentionConfigs,
        GrpcConfig,
        BatchDecodeSchedulerConfig,
        CacheStoreConfig,
        ConcurrencyConfig,
        DeviceResourceConfig,
        EplbMode,
        FfnDisAggregateConfig,
        FIFOSchedulerConfig,
        FMHAConfig,
        HWKernelConfig,
        KVCacheConfig,
        MiscellaneousConfig,
        MlaOpsType,
        ModelConfig,
        ModelSpecificConfig,
        MoeConfig,
        NcclCommConfig,
        PDSepConfig,
        ParallelismConfig,
        ProfilingDebugLoggingConfig,
        RopeCache,
        RopeConfig,
        RopeStyle,
        TaskType,
        VitConfig,
        VitSeparation,
        check_rope_cache,
        get_rope_cache,
        get_rope_cache_once,
        CPRotateMethod,
        PrefillCPConfig,
    )
    # Alias for backward compatibility
    from libth_transformer_config import (
        QuantAlgo,
        RoleType,
        RuntimeConfig,
        SpecialTokens,
        SpeculativeExecutionConfig,
        SpeculativeType,
        EPLBConfig,
        ActivationType,
        DataType,
        KvCacheDataType,
        ModelConfig,
        HybridAttentionConfig,
        HybridAttentionType,
        LinearAttentionConfig,
    )
    # Alias for backward compatibility
    EplbConfig = EPLBConfig
    from libth_transformer_config import (
        get_block_cache_keys as cpp_get_block_cache_keys,
    )
    from libth_transformer_config import MultimodalInput, MMPreprocessConfig

except BaseException as e:
    logging.info(f"Exception: {e}, traceback: {traceback.format_exc()}")
    if os.environ.get("RTP_LLM_ALLOW_MISSING_SO") != "1":
        raise e
    # Collection-only mode: stub every libth_transformer_config symbol with
    # EmptyClass so `import rtp_llm` succeeds without the C++ extension. Access
    # to these types is non-functional (pytest collection / frontend only).
    logging.warning(
        "RTP_LLM_ALLOW_MISSING_SO=1: stubbing libth_transformer_config symbols "
        "with EmptyClass (collection-only mode; C++ config types non-functional)."
    )
    for _sym in _LIBTH_CONFIG_SYMBOLS:
        globals()[_sym] = EmptyClass


def get_block_cache_keys(token_ids: List[int], block_size: int) -> List[int]:
    try:
        # split token_ids into chunks of size block_size, dropping the last chunk if it is smaller than block_size
        token_ids_list: List[List[int]] = []
        for i in range(0, len(token_ids), block_size):
            chunk = token_ids[i : i + block_size]
            if len(chunk) == block_size:
                token_ids_list.append(chunk)
        return cpp_get_block_cache_keys(token_ids_list)  # type: ignore
    except Exception as e:
        logging.error(f"get block ids error: {e}")
        # If an error occurs, return an empty list
        return []


try:
    import librtp_compute_ops
    from .compute_ops import rtp_llm_ops
    # Export LayerKVCache and other types from librtp_compute_ops
    from librtp_compute_ops import LayerKVCache, KVCache, PyAttentionInputs, PyModelInputs, PyModelOutputs, PyModelInitResources, PyCacheStoreInputs
except BaseException as e:
    logging.info(f"Exception: {e}, traceback: {traceback.format_exc()}")
    rtp_llm_ops = EmptyClass
    LayerKVCache = KVCache = PyAttentionInputs = PyModelInputs = PyModelOutputs = PyModelInitResources = PyCacheStoreInputs = EmptyClass

try:

    from libth_transformer import RtpEmbeddingOp, RtpLLMOp
    from libth_transformer import EmbeddingCppOutput

    libth_transformer_imported = True
except BaseException as e:
    EmbeddingCppOutput = RtpEmbeddingOp = RtpLLMOp = EmptyClass

    logging.warning(f"libth_transformer import failed: {type(e).__name__}: {e}")
    logging.info(
        "libth_transformer not imported, you may under python standalone mode or frontend mode now."
    )
