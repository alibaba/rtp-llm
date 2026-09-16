import ctypes
import glob
import importlib
import logging
import os
import site
import sys
import threading
import traceback
from enum import IntEnum
from typing import List, Optional

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
from rtp_llm.utils import torch_patch  # noqa: F401

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
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


# Symbols imported from libth_transformer_config; stubbed with EmptyClass in
# collection-only mode (RTP_LLM_ALLOW_MISSING_SO=1) when the .so is missing.
_LIBTH_CONFIG_SYMBOLS = [
    "ArpcConfig", "AttentionConfigs", "GrpcConfig", "BatchDecodeSchedulerConfig",
    "CacheCapacityPolicyDesc", "CacheCpPolicyDesc", "CacheEvictPolicy",
    "CacheGroupType", "CacheMemoryPlacement", "CacheMemoryPolicyDesc",
    "CacheReusePolicy", "CacheReusePolicyDesc", "CacheStoreConfig",
    "CacheTailPolicyDesc", "ConcurrencyConfig", "CpBlockMappingMode",
    "CpBlockSliceMode", "CpPrefillSliceLayout", "DashScGrpcConfig",
    "DeviceResourceConfig", "EplbMode",
    "FfnDisAggregateConfig", "FIFOSchedulerConfig", "FMHAConfig", "HWKernelConfig",
    "GrammarConfig", "KVCacheConfig", "KVCacheSpecDesc", "KVCacheSpecType",
    "MiscellaneousConfig", "MlaOpsType", "ModelConfig",
    "ModelSpecificConfig", "MoeConfig", "NcclCommConfig", "PDSepConfig",
    "ParallelismConfig", "ProfilingDebugLoggingConfig", "RopeCache", "RopeConfig",
    "RopeStyle", "TaskType", "VitConfig", "VitSeparation", "check_rope_cache",
    "get_rope_cache", "get_rope_cache_once", "CPRotateMethod", "PrefillCPConfig",
    "QuantAlgo", "RoleType", "RuntimeConfig", "SpecialTokens",
    "SpeculativeExecutionConfig", "SpeculativeType", "EPLBConfig", "ActivationType",
    "DataType", "KvCacheDataType", "HybridAttentionConfig", "HybridAttentionType",
    "LinearAttentionConfig", "MultimodalInput", "MultimodalInputCpp",
    "MMPreprocessConfig", "OpaqueBlockEntryCountMode", "EplbConfig",
    "cpp_get_block_cache_keys",
]

try:
    from libth_transformer_config import (
        ArpcConfig,
        AttentionConfigs,
        GrpcConfig,
        BatchDecodeSchedulerConfig,
        CacheCapacityPolicyDesc,
        CacheCpPolicyDesc,
        CacheEvictPolicy,
        CacheGroupType,
        CacheMemoryPlacement,
        CacheMemoryPolicyDesc,
        CacheReusePolicy,
        CacheReusePolicyDesc,
        CacheStoreConfig,
        CacheTailPolicyDesc,
        CpBlockMappingMode,
        CpBlockSliceMode,
        CpPrefillSliceLayout,
        DashScGrpcConfig,
        KVCacheSpecType,
        OpaqueBlockEntryCountMode,
        ConcurrencyConfig,
        DeviceResourceConfig,
        EplbMode,
        FfnDisAggregateConfig,
        FIFOSchedulerConfig,
        FMHAConfig,
        GrammarConfig,
        HWKernelConfig,
        KVCacheConfig,
        KVCacheSpecDesc,
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

    class _CollectionRoleType(IntEnum):
        PDFUSION = 0
        PREFILL = 1
        DECODE = 2
        VIT = 3
        FRONTEND = 4

    RoleType = _CollectionRoleType

    class _CollectionTaskType(IntEnum):
        DENSE_EMBEDDING = 0
        ALL_EMBEDDING = 1
        SPARSE_EMBEDDING = 2
        COLBERT_EMBEDDING = 3
        LANGUAGE_MODEL = 4
        SEQ_CLASSIFICATION = 5
        RERANKER = 6
        LINEAR_SOFTMAX = 7
        BGE_M3 = 8

    class _CollectionVitSeparation(IntEnum):
        VIT_SEPARATION_LOCAL = 0
        VIT_SEPARATION_ROLE = 1
        VIT_SEPARATION_REMOTE = 2

    class _CollectionMMPreprocessConfig:
        def __init__(
            self,
            width=-1,
            height=-1,
            min_pixels=-1,
            max_pixels=-1,
            fps=-1,
            min_frames=-1,
            max_frames=-1,
            crop_positions=None,
            mm_timeout_ms=-1,
        ):
            self.width = width
            self.height = height
            self.min_pixels = min_pixels
            self.max_pixels = max_pixels
            self.fps = fps
            self.min_frames = min_frames
            self.max_frames = max_frames
            self.crop_positions = [] if crop_positions is None else crop_positions
            self.mm_timeout_ms = mm_timeout_ms

    class _CollectionMultimodalInput:
        def __init__(self, url, mm_type=0, tensor=None, mm_preprocess_config=None):
            self.url = url
            self.mm_type = mm_type
            self.tensor = tensor
            self.mm_preprocess_config = (
                _CollectionMMPreprocessConfig()
                if mm_preprocess_config is None
                else mm_preprocess_config
            )

    TaskType = _CollectionTaskType
    VitSeparation = _CollectionVitSeparation
    MMPreprocessConfig = _CollectionMMPreprocessConfig
    MultimodalInput = _CollectionMultimodalInput


def serialize_grammar_tokenizer_info(
    encoded_vocab: List[str],
    tokenizer_metadata_json: str,
) -> str:
    from libth_grammar_tokenizer_info import (
        serialize_grammar_tokenizer_info as serialize,
    )

    return serialize(encoded_vocab, tokenizer_metadata_json)


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


_COMPUTE_SYMBOLS = {
    "compute_ops",
    "KVCache",
    "LayerKVCache",
    "PyAttentionInputs",
    "PyCacheStoreInputs",
    "PyModelInitResources",
    "PyModelInputs",
    "PyModelOutputs",
    "CacheStoreWriter",
    "rtp_llm_ops",
}
_ENGINE_SYMBOLS = {
    "EmbeddingCppOutput",
    "MultimodalInputCpp",
    "RtpEmbeddingOp",
    "RtpLLMOp",
}
_RDMA_SYMBOLS = {"MMRdmaExporter"}
_compute_ops_lock = threading.RLock()
_compute_ops_loaded = False
_compute_ops_error: Optional[BaseException] = None
_engine_ops_lock = threading.RLock()
_engine_ops_loaded = False
_engine_ops_error: Optional[BaseException] = None
_rdma_ops_lock = threading.RLock()
_rdma_ops_loaded = False
_rdma_ops_error: Optional[BaseException] = None


def _set_compute_fallbacks() -> None:
    globals()["rtp_llm_ops"] = EmptyClass
    globals()["LayerKVCache"] = EmptyClass
    globals()["KVCache"] = EmptyClass
    globals()["PyAttentionInputs"] = EmptyClass
    globals()["PyModelInputs"] = EmptyClass
    globals()["PyModelOutputs"] = EmptyClass
    globals()["PyModelInitResources"] = EmptyClass
    globals()["PyCacheStoreInputs"] = EmptyClass
    globals()["CacheStoreWriter"] = EmptyClass

def _raise_required_load_error(kind: str, error: BaseException) -> None:
    raise RuntimeError(f"failed to load required RTP-LLM {kind} ops") from error


def _load_compute_ops(required: bool = False) -> None:
    global _compute_ops_error, _compute_ops_loaded
    with _compute_ops_lock:
        if _compute_ops_loaded:
            if required and _compute_ops_error is not None:
                _raise_required_load_error("compute", _compute_ops_error)
            return
        try:
            import librtp_compute_ops

            globals()["KVCache"] = librtp_compute_ops.KVCache
            globals()["LayerKVCache"] = librtp_compute_ops.LayerKVCache
            globals()["CacheStoreWriter"] = librtp_compute_ops.CacheStoreWriter
            globals()["PyAttentionInputs"] = librtp_compute_ops.PyAttentionInputs
            globals()["PyCacheStoreInputs"] = librtp_compute_ops.PyCacheStoreInputs
            globals()["PyModelInitResources"] = librtp_compute_ops.PyModelInitResources
            globals()["PyModelInputs"] = librtp_compute_ops.PyModelInputs
            globals()["PyModelOutputs"] = librtp_compute_ops.PyModelOutputs

            compute_ops = importlib.import_module(f"{__name__}.compute_ops")
            globals()["compute_ops"] = compute_ops
            globals()["rtp_llm_ops"] = compute_ops.rtp_llm_ops
            _compute_ops_error = None
        except BaseException as e:
            _compute_ops_error = e
            logging.info(f"Exception: {e}, traceback: {traceback.format_exc()}")
            _set_compute_fallbacks()
            if required:
                _raise_required_load_error("compute", e)
        _compute_ops_loaded = True


class DisabledMMRdmaExporter(EmptyClass):
    """Stand-in for optional RDMA capability probes."""

    @staticmethod
    def available() -> bool:
        return False

    def enabled(self) -> bool:
        return False


def _set_engine_fallbacks() -> None:
    globals()["MultimodalInputCpp"] = EmptyClass
    globals()["EmbeddingCppOutput"] = EmptyClass
    globals()["RtpEmbeddingOp"] = EmptyClass
    globals()["RtpLLMOp"] = EmptyClass


def _load_rdma_ops(required: bool = False) -> None:
    global _rdma_ops_error, _rdma_ops_loaded
    with _rdma_ops_lock:
        if _rdma_ops_loaded:
            if required and _rdma_ops_error is not None:
                _raise_required_load_error("RDMA exporter", _rdma_ops_error)
            return
        try:
            from libmm_rdma_exporter import MMRdmaExporter

            globals()["MMRdmaExporter"] = MMRdmaExporter
            _rdma_ops_error = None
        except BaseException as e:
            _rdma_ops_error = e
            globals()["MMRdmaExporter"] = DisabledMMRdmaExporter
            logging.warning("libmm_rdma_exporter could not be imported: %s", e)
            if required:
                _raise_required_load_error("RDMA exporter", e)
        _rdma_ops_loaded = True


def _load_engine_ops(required: bool = False) -> None:
    global _engine_ops_error, _engine_ops_loaded
    with _engine_ops_lock:
        if _engine_ops_loaded:
            if required and _engine_ops_error is not None:
                _raise_required_load_error("engine", _engine_ops_error)
            return
        # libth_transformer has historically been loaded after librtp_compute_ops.
        # Keep that order even with lazy imports; loading it first can corrupt
        # process teardown in the current binary build.
        _load_compute_ops(required=required)
        try:
            from libth_transformer import EmbeddingCppOutput

            # MultimodalInput is registered by the config module
            # (libth_transformer_config) in this build; alias it here so
            # callers keep using the historical MultimodalInputCpp name.
            from libth_transformer_config import MultimodalInput as MultimodalInputCpp
            from libth_transformer import RtpEmbeddingOp, RtpLLMOp

            globals()["EmbeddingCppOutput"] = EmbeddingCppOutput
            globals()["MultimodalInputCpp"] = MultimodalInputCpp
            globals()["RtpEmbeddingOp"] = RtpEmbeddingOp
            globals()["RtpLLMOp"] = RtpLLMOp
            _engine_ops_error = None
        except BaseException as e:
            _engine_ops_error = e
            _set_engine_fallbacks()
            logging.info(
                "libth_transformer not imported, you may under python standalone mode or frontend mode now."
            )
            if required:
                _raise_required_load_error("engine", e)
        _engine_ops_loaded = True


def ensure_compute_ops_loaded() -> None:
    _load_compute_ops(required=True)


def ensure_engine_ops_loaded() -> None:
    _load_engine_ops(required=True)


def ensure_rdma_ops_loaded() -> None:
    _load_rdma_ops(required=True)


def __getattr__(name: str):
    if name in _COMPUTE_SYMBOLS:
        _load_compute_ops()
        if name in globals():
            return globals()[name]
    if name in _ENGINE_SYMBOLS:
        _load_engine_ops()
        if name in globals():
            return globals()[name]
    if name in _RDMA_SYMBOLS:
        _load_rdma_ops()
        if name in globals():
            return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
