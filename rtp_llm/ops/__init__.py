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


# for py test
def find_upper_file(current_dir: str, file_name: str):
    logging.info(f"find_upper_file: {current_dir}, target: {file_name}")
    p = pathlib.Path(current_dir).resolve()
    while p != p.parent:
        if p.exists():
            for root, _, files in os.walk(p, followlinks = True):
                if file_name in files:
                    return os.path.join(root, file_name)
        p = p.parent
    raise Exception(f"failed to find {file_name} in {current_dir}")


def find_upper_so(current_dir: str):
    so_file = find_upper_file(current_dir, SO_NAME)
    return str(pathlib.Path(so_file).parent)


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
ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = ":".join([p for p in [so_path, ld_library_path] if p])
logging.info(f"updated LD_LIBRARY_PATH for ops: {os.environ['LD_LIBRARY_PATH']}")
sys.path.insert(0, so_path)

# glibc does not reliably honor runtime LD_LIBRARY_PATH updates for the current
# process, so preload transformer runtime deps by absolute path before importing
# libth_transformer.
from ctypes import CDLL, RTLD_GLOBAL

_PRELOAD_SO_NAMES = [
    "libicudata.so.66",
    "libicuuc.so.66",
    "libicui18n.so.66",
    "libunwind.so.8",
    "libcrypto.so.1.1",
    "libssl.so.1.1",
    "libboost_context.so.1.71.0",
    "libboost_filesystem.so.1.71.0",
    "libboost_program_options.so.1.71.0",
    "libboost_regex.so.1.71.0",
    "libboost_system.so.1.71.0",
    "libboost_thread.so.1.71.0",
    "libboost_atomic.so.1.71.0",
    "libdouble-conversion.so.3",
    "libgflags.so.2.2",
    "libglog.so.0",
    "libevent-2.1.so.7",
    "libdwarf.so.1",
    "libhf3fs_api_shared.so",
    "kv_cache_manager_client.so",
]


def _resolve_dep_path(dep_name: str):
    dep_path = os.path.join(so_path, dep_name)
    if os.path.exists(dep_path):
        return dep_path
    return find_upper_file(current_dir, dep_name)


failed_dep_names = list(_PRELOAD_SO_NAMES)
last_errors = {}
for _ in range(4):
    remaining = []
    progress = False
    for dep_name in failed_dep_names:
        try:
            dep_path = _resolve_dep_path(dep_name)
        except Exception:
            continue
        try:
            CDLL(dep_path, mode=RTLD_GLOBAL)
            logging.info(f"preloaded {dep_name} from {dep_path}")
            progress = True
        except OSError as e:
            last_errors[dep_name] = (dep_path, str(e))
            remaining.append(dep_name)
    if not remaining or not progress:
        failed_dep_names = remaining
        break
    failed_dep_names = remaining

for dep_name in failed_dep_names:
    dep_path, err = last_errors.get(dep_name, ("", "unknown error"))
    logging.info(f"failed to preload {dep_name} from {dep_path}: {err}")

_CORE_EXTENSION_NAMES = [
    "librtp_compute_ops.so",
    "libth_transformer_config.so",
    "libth_transformer.so",
]
for ext_name in _CORE_EXTENSION_NAMES:
    if ext_name == "libth_transformer.so" and failed_dep_names:
        logging.info(f"skip preloading {ext_name} because unresolved deps remain: {failed_dep_names}")
        continue
    try:
        ext_path = _resolve_dep_path(ext_name)
    except Exception:
        continue
    try:
        CDLL(ext_path, mode=RTLD_GLOBAL)
        logging.info(f"preloaded core extension {ext_name} from {ext_path}")
    except OSError as e:
        logging.info(f"failed to preload core extension {ext_name} from {ext_path}: {e}")

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
    # Export KVCache and other types from librtp_compute_ops
    from librtp_compute_ops import KVCache, PyAttentionInputs, PyModelInputs, PyModelOutputs, PyModelInitResources, PyCacheStoreInputs
except BaseException as e:
    logging.info(f"Exception: {e}, traceback: {traceback.format_exc()}")
    rtp_llm_ops = EmptyClass
    KVCache = PyAttentionInputs = PyModelInputs = PyModelOutputs = PyModelInitResources = PyCacheStoreInputs = EmptyClass

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
        f"libth_transformer not imported: {e}, traceback: {traceback.format_exc()}"
    )
