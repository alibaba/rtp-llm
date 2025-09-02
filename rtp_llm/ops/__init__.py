import logging
import os
import pathlib
import sys
import traceback

import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
libs_path = os.path.join(parent_dir, "libs")
SO_NAME = "libth_transformer.so"


# for py test
def find_upper_so(current_dir: str):
    p = pathlib.Path(current_dir).resolve()
    while p != p.parent:
        if p.exists():
            for root, _, files in os.walk(p):
                if SO_NAME in files:
                    return root
        p = p.parent
    raise Exception(f"failed to find {SO_NAME} in {current_dir}")


def find_th_transformer(current_dir: str):
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
    logging.info(f"failed to load libth_transformer.so from libs, try use another path")
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


try:
    from libth_transformer import (
        ArpcConfig,
        BatchDecodeSchedulerConfig,
        CacheStatusInfo,
        CacheStoreConfig,
        ConcurrencyConfig,
        DeviceExporter,
        DeviceResourceConfig,
        DeviceType,
        EngineScheduleInfo,
        EplbConfig,
        EplbMode,
        FfnDisAggregateConfig,
        FIFOSchedulerConfig,
        FMHAConfig,
        FMHAType,
        GptInitParameter,
        HWKernelConfig,
        KVCache,
        KVCacheConfig,
        KVCacheInfo,
        LoadBalanceInfo,
        MiscellaneousConfig,
        MlaOpsType,
        ModelSpecificConfig,
        MoeConfig,
    )
    from libth_transformer import MultimodalInput as MultimodalInputCpp
    from libth_transformer import (
        ParallelismDistributedConfig,
        ProfilingDebugLoggingConfig,
        PyAttentionInputs,
        PyModelInitResources,
        PyModelInputs,
        PyModelOutputs,
        QuantAlgo,
        RoleType,
        RtpEmbeddingOp,
        RtpLLMOp,
        SamplerConfig,
        SchedulerConfig,
        ServiceDiscoveryConfig,
        SpecialTokens,
        SpeculativeExecutionConfig,
        WorkerStatusInfo,
        get_block_cache_keys,
        get_device,
    )

except BaseException as e:
    import traceback

    print(f"Exception: {e}, traceback: {traceback.format_exc()}")
    logging.info(f"Exception: {e}, traceback: {traceback.format_exc()}")
