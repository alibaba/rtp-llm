import logging
import os
import pathlib
import sys
import traceback
from typing import List

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
libs_path = os.path.join(parent_dir, "libs")
SO_NAME = "libth_transformer_config.so"


def _preload_shared_lib(lib_name: str, search_roots: List[str]) -> bool:
    """Preload a runtime dependency that may live outside rpath in Bazel runfiles."""
    from ctypes import CDLL, RTLD_GLOBAL

    seen = set()
    for root in search_roots:
        if not root or root in seen or not os.path.exists(root):
            continue
        seen.add(root)
        for dirpath, _, filenames in os.walk(root):
            if lib_name not in filenames:
                continue
            lib_path = os.path.join(dirpath, lib_name)
            try:
                CDLL(lib_path, mode=RTLD_GLOBAL)
                logging.info(f"preloaded {lib_name} from {lib_path}")
                return True
            except BaseException as e:
                logging.info(
                    f"failed to preload {lib_name} from {lib_path}: {e}, "
                    f"traceback: {traceback.format_exc()}"
                )
    return False


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

# XGrammar's Python wheel ships shared libraries under the pip repository in
# Bazel runfiles.  libth_transformer.so links against them, but its rpath differs
# between the source tree, runfiles, and wheel layouts.  Preloading by filename
# keeps the engine import independent of that layout.
runfiles_root = os.path.dirname(parent_dir)
runfiles_parent = os.path.dirname(runfiles_root)
_preload_shared_lib(
    "libtvm_ffi.so",
    [
        os.path.join(runfiles_root, "external/pip_gpu_cuda13_torch_apache_tvm_ffi/site-packages/tvm_ffi/lib"),
        os.path.join(runfiles_root, "external/pip_cuda13_arm_torch_apache_tvm_ffi/site-packages/tvm_ffi/lib"),
        os.path.join(runfiles_parent, "pip_gpu_cuda13_torch_apache_tvm_ffi/site-packages/tvm_ffi/lib"),
        os.path.join(runfiles_parent, "pip_cuda13_arm_torch_apache_tvm_ffi/site-packages/tvm_ffi/lib"),
    ],
)
_preload_shared_lib(
    "libxgrammar_bindings.so",
    [
        os.path.join(runfiles_root, "external/pip_gpu_cuda13_torch_xgrammar/site-packages/xgrammar"),
        os.path.join(runfiles_root, "external/pip_cuda13_arm_torch_xgrammar/site-packages/xgrammar"),
        os.path.join(runfiles_parent, "pip_gpu_cuda13_torch_xgrammar/site-packages/xgrammar"),
        os.path.join(runfiles_parent, "pip_cuda13_arm_torch_xgrammar/site-packages/xgrammar"),
    ],
)

# frontend cannot load libpython3.10.so, so we need to load it manually
import sysconfig
from ctypes import cdll

cdll.LoadLibrary(sysconfig.get_config_var("LIBDIR") + "/libpython3.10.so")

try:
    from libth_transformer_config import (
        ArpcConfig,
        AttentionConfigs,
        DashScGrpcConfig,
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

    logging.info(
        "libth_transformer not imported, you may under python standalone mode or frontend mode now."
    )
