import importlib
import logging
import os
import pathlib
import sys
import threading
import traceback
from typing import List, Optional

import torch

from rtp_llm.utils import torch_patch  # noqa: F401

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
libs_path = os.path.join(parent_dir, "libs")
SO_NAME = "libth_transformer_config.so"


# for py test
def find_upper_so(current_dir: str):
    logging.info(f"find_upper_so: {current_dir}")
    p = pathlib.Path(current_dir).resolve()
    while p != p.parent:
        if p.exists():
            for root, _, files in os.walk(p):
                logging.info(f"find_upper_so: {root}, {files}")
                if SO_NAME in files:
                    return root
        p = p.parent
    raise Exception(f"failed to find {SO_NAME} in {current_dir}")


def find_th_transformer(current_dir: str):
    logging.info(f"find_th_transformer: {current_dir}")
    if not os.path.exists(current_dir):
        return None
    dir_path = pathlib.Path(current_dir)
    for file in dir_path.iterdir():
        if file.is_file() and file.name == SO_NAME:
            return os.path.join(current_dir)

    # 检查下一级目录中的文件
    for subdir in dir_path.iterdir():
        if subdir.is_dir():  # 确保是目录
            for file in subdir.iterdir():
                if file.is_file() and file.name == SO_NAME:
                    return os.path.join(current_dir, subdir.name)

    # 检查更下一级目录中的文件
    for subdir in dir_path.iterdir():
        if subdir.is_dir():  # 确保是目录
            for subsubdir in subdir.iterdir():
                if subsubdir.is_dir():
                    for file in subsubdir.iterdir():
                        if file.is_file() and file.name == SO_NAME:
                            return os.path.join(
                                os.path.join(current_dir, subdir.name), subsubdir.name
                            )
    return None


so_path = os.path.join(libs_path)
if not os.path.exists(os.path.join(so_path, SO_NAME)):
    logging.info(
        f"failed to load libth_transformer_config.so from libs, try use another path"
    )
    # for debug useage, read in bazel-bin and bazel-bin's subdir
    bazel_bin_dir = os.path.join(parent_dir, "../bazel-bin")
    so_path = find_th_transformer(bazel_bin_dir)
    logging.info(f"failed to find {SO_NAME} in {bazel_bin_dir}")
    if not so_path:
        so_path = find_upper_so(current_dir)

logging.info(f"so path: {so_path}")
sys.path.append(so_path)

# load intel xft lib
xft_loaded = False
# for path in sys.path:
#     try:
#         if "xfastertransformer-devel" in os.listdir(path):
#             xft_lib_path = f"{path}/xfastertransformer-devel/lib"
#             from ctypes import cdll
#             cdll.LoadLibrary(f"{xft_lib_path}/libxfastertransformer.so")
#             xft_loaded = True
#             logging.info(f"loaded libxfastertransformer.so from {xft_lib_path}")
#             break
#         else:
#             logging.debug(f"checked path [{path}] for xft, not found.")
#     except:
#         pass
# if not xft_loaded:
#     logging.info("xfastertransformer-devel package not loaded, this won't affect run.")

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

# frontend cannot load libpython3.10.so, so we need to load it manually
import sysconfig
from ctypes import cdll

cdll.LoadLibrary(sysconfig.get_config_var("LIBDIR") + "/libpython3.10.so")

try:
    # Alias for backward compatibility
    from libth_transformer_config import (
        ActivationType,
        ArpcConfig,
        AttentionConfigs,
        BatchDecodeSchedulerConfig,
        CacheStoreConfig,
        ConcurrencyConfig,
        CPRotateMethod,
        DashScGrpcConfig,
        DataType,
        DeviceResourceConfig,
        EPLBConfig,
        EplbMode,
        FfnDisAggregateConfig,
        FIFOSchedulerConfig,
        FMHAConfig,
        FMHAType,
        GrammarConfig,
        GrpcConfig,
        HWKernelConfig,
        HybridAttentionConfig,
        HybridAttentionType,
        KVCacheConfig,
        KvCacheDataType,
        LinearAttentionConfig,
        MiscellaneousConfig,
        MlaOpsType,
        ModelConfig,
        ModelSpecificConfig,
        MoeConfig,
        NcclCommConfig,
        ParallelismConfig,
        PDSepConfig,
        PrefillCPConfig,
        ProfilingDebugLoggingConfig,
        QuantAlgo,
        RoleType,
        RopeCache,
        RopeConfig,
        RopeStyle,
        RuntimeConfig,
        SpecialTokens,
        SpeculativeExecutionConfig,
        SpeculativeType,
        TaskType,
        VitConfig,
        VitSeparation,
        check_rope_cache,
        get_rope_cache,
        get_rope_cache_once,
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


_COMPUTE_SYMBOLS = {
    "compute_ops",
    "KVCache",
    "LayerKVCache",
    "PyAttentionInputs",
    "PyCacheStoreInputs",
    "PyModelInitResources",
    "PyModelInputs",
    "PyModelOutputs",
    "rtp_llm_ops",
}
_ENGINE_SYMBOLS = {
    "EmbeddingCppOutput",
    "MultimodalInputCpp",
    "RtpEmbeddingOp",
    "RtpLLMOp",
    "build_xgrammar_tokenizer_info_json",
}
_compute_ops_lock = threading.RLock()
_compute_ops_loaded = False
_compute_ops_error: Optional[BaseException] = None
_engine_ops_lock = threading.RLock()
_engine_ops_loaded = False
_engine_ops_error: Optional[BaseException] = None


def _set_compute_fallbacks() -> None:
    globals()["rtp_llm_ops"] = EmptyClass
    globals()["LayerKVCache"] = EmptyClass
    globals()["KVCache"] = EmptyClass
    globals()["PyAttentionInputs"] = EmptyClass
    globals()["PyModelInputs"] = EmptyClass
    globals()["PyModelOutputs"] = EmptyClass
    globals()["PyModelInitResources"] = EmptyClass
    globals()["PyCacheStoreInputs"] = EmptyClass


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


def _set_engine_fallbacks() -> None:
    globals()["MultimodalInputCpp"] = EmptyClass
    globals()["EmbeddingCppOutput"] = EmptyClass
    globals()["build_xgrammar_tokenizer_info_json"] = EmptyClass
    globals()["RtpEmbeddingOp"] = EmptyClass
    globals()["RtpLLMOp"] = EmptyClass


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
            from libth_transformer import MultimodalInput as MultimodalInputCpp
            from libth_transformer import (
                RtpEmbeddingOp,
                RtpLLMOp,
                build_xgrammar_tokenizer_info_json,
            )

            globals()["EmbeddingCppOutput"] = EmbeddingCppOutput
            globals()["MultimodalInputCpp"] = MultimodalInputCpp
            globals()["RtpEmbeddingOp"] = RtpEmbeddingOp
            globals()["RtpLLMOp"] = RtpLLMOp
            globals()[
                "build_xgrammar_tokenizer_info_json"
            ] = build_xgrammar_tokenizer_info_json
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


def __getattr__(name: str):
    if name in _COMPUTE_SYMBOLS:
        _load_compute_ops()
        if name in globals():
            return globals()[name]
    if name in _ENGINE_SYMBOLS:
        _load_engine_ops()
        if name in globals():
            return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
