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

# All .so files are in rtp_llm/libs/ (copied by setup.py during uv/pip install)
so_path = libs_path
_so_available = os.path.exists(os.path.join(so_path, SO_NAME))
if not _so_available:
    logging.warning(
        f"{SO_NAME} not found in {libs_path}. "
        f"C++ extensions not available (collection-only mode)."
    )
else:
    logging.info(f"so path: {so_path}")
    sys.path.append(so_path)


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
except OSError:
    pass

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
        FMHAType,
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

except BaseException as e:
    logging.info(f"Exception: {e}, traceback: {traceback.format_exc()}")
    raise e

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


# Frontend not related
class EmptyClass:
    def __init__(self, **kwargs):
        pass

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

    from libth_transformer import MultimodalInput as MultimodalInputCpp
    from libth_transformer import RtpEmbeddingOp, RtpLLMOp
    from libth_transformer import EmbeddingCppOutput

    libth_transformer_imported = True
except BaseException as e:
    MultimodalInputCpp = EmbeddingCppOutput = (
        EmptyClass
    )
    RtpEmbeddingOp = RtpLLMOp = EmptyClass

    logging.warning(f"libth_transformer import failed: {type(e).__name__}: {e}")
    logging.info(
        "libth_transformer not imported, you may under python standalone mode or frontend mode now."
    )
