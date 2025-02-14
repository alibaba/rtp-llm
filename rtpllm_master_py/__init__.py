import torch
import os
import sys
import logging
logging.basicConfig(level="INFO",
                format="[process-%(process)d][%(name)s][%(asctime)s][%(filename)s:%(funcName)s():%(lineno)s][%(levelname)s] %(message)s",
                datefmt='%m/%d/%Y %H:%M:%S')

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
# used when 
# load from libs dir
SO_RELATIVE_PATH = "stub/librtpllm_master.so"
# used when load fro
# m bazel-bin
WORKERSPACE = os.path.dirname(CURRENT_PATH)
DIR_NAME = "rtpllm_master_py"

so_path = CURRENT_PATH
if not os.path.exists(os.path.join(so_path, SO_RELATIVE_PATH)):
    logging.info(f"failed to load {SO_RELATIVE_PATH} from {so_path}, try use another path")
    # for debug useage, read in bazel-bin and bazel-bin's subdir
    so_path = os.path.join(WORKERSPACE, "bazel-bin", DIR_NAME)
    if not os.path.exists(os.path.join(so_path, SO_RELATIVE_PATH)):
        raise Exception(f"failed to load {SO_RELATIVE_PATH} from bazel-bin: {so_path}")
logging.info(f"so path: {so_path}")
sys.path.append(so_path)