import torch
import pathlib
import os
import sys
import logging
import traceback

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
libs_path = os.path.join(parent_dir, "libs")
SO_NAME = 'libth_transformer.so'

def find_th_transformer(current_dir: str):
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

    raise Exception(f"failed to find {SO_NAME} in {current_dir}")

so_path = os.path.join(libs_path)
if not os.path.exists(os.path.join(so_path, SO_NAME)):
    logging.info(f"failed to load libth_transformer.so from libs, try use another path")
    # for debug useage, read in bazel-bin and bazel-bin's subdir
    bazel_bin_dir = os.path.join(parent_dir, "../bazel-bin")
    so_path = find_th_transformer(bazel_bin_dir)
print("so path: ", so_path)
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


try:
    # CUDA GPU build
    from libth_transformer import GptInitParameter, RtpEmbeddingOp, RtpLLMOp, SpecialTokens, LoadBalanceInfo
except BaseException as e:
    import traceback
    logging.info(f"Exception: {e}, traceback: {traceback.format_exc()}")
