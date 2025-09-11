# Building and Running RTP-LLM from Scratch
* Operating System: Linux
* Python: 3.10
* NVIDIA GPU: Compute Capability 7.0 or higher (e.g., V100, T4, RTX20xx, A100, L4, H100, etc.)

## 1. Environment Setup
In this article, we will introduce the complete deployment and usage path of the RTP-LLM inference engine system. This article uses a single machine with 4 A10 cards as an example. First, let's look at our machine configuration, with the GPU configuration as follows:
```shell
$nvidia-smi
Fri May 16 14:52:19 2025
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.82.01    Driver Version: 470.82.01    CUDA Version: 12.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A10          Off  | 00000000:00:09.0 Off |                 Off* |
|  0%   26C    P8    19W / 150W |      2MiB / 24258MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA A10          Off  | 00000000:00:0A.0 Off |                 Off* |
|  0%   27C    P8    20W / 150W |      2MiB / 24258MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  NVIDIA A10          Off  | 00000000:00:0B.0 Off |                 Off* |
|  0%   28C    P8    20W / 150W |      2MiB / 24258MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  NVIDIA A10          Off  | 00000000:00:0C.0 Off |                 Off* |
|  0%   30C    P8    21W / 150W |      2MiB / 24258MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
Next, we need to pull the container environment for configuring and installing RTP-LLM. The container we prepared is as follows:
registry.cn-hangzhou.aliyuncs.com/havenask/rtp_llm:2025_06_03_10_12_c02cc34

Image name: registry.cn-hangzhou.aliyuncs.com/havenask/rtp_llm
Version: 2025_06_03_10_12_c02cc34

### 1.1 Pull Container
```shell
docker pull registry.cn-hangzhou.aliyuncs.com/havenask/rtp_llm:2025_06_03_10_12_c02cc34
```

### 1.2 Create Docker Container, Mount Disks According to Local Machine Storage
```shell
## 1. The -v below mounts your storage with the external host storage. Users can flexibly adjust according to their local storage.
## 2. --cap-add SYS_ADMIN is for advanced users, such as users with source code level development needs.
## 3. --device <host_path>:<container_path> ensures correct mapping of devices and host containers.
## 4. --volume mounts the Docker volume to the /usr/local/nvidia directory in the container and sets it to read-only mode.
## 5. -runtime=nvidia specifies that the container uses the NVIDIA runtime to support GPU accelerated computing.
## 6. --gpus all ensures the container shares GPU with the host.
## 7. --net=host allows the host and container to share the network, which is useful for advanced users who need SSH remote connection development.
docker run \
  --cap-add SYS_ADMIN --device /dev/fuse \
  -v /mnt/:/mnt/ \
  -v /ssd/1:/ssd/1 \
  -v /ssd/2:/ssd/2 \
  -v /data0:/data0 \
  -v /data1:/data1 \
  -v /dev/shm:/dev/shm \
  -v /hio_disk:/hio_disk \
  --device /dev/nvidia0:/dev/nvidia0 --device /dev/nvidiactl:/dev/nvidiactl \
  --volume=nvidia_driver_volume:/usr/local/nvidia:ro \
  --runtime=nvidia --gpus all \
  --name user_gpu \
  --net=host -dit \
  registry.cn-hangzhou.aliyuncs.com/havenask/rtp_llm:2025_06_03_10_12_c02cc34 \
  /bin/bash
```

### 1.3 Create User in Container (Optional, Default is root. Advanced developers can choose to create their own user. It is not recommended to use root user for source code development)
```shell
## Create personal development user
docker exec -i user_gpu /usr/sbin/useradd -MU -u <your_UID> <your_username>
## Assign personal user to root group and add to /etc/sudoers
your_username ALL=(ALL) NOPASSWD:ALL
```

### 1.4 Enter Container
```shell
## 1. For regular users
docker exec -it container_name /bin/bash
## 2. For advanced users, need to enter their personal user account
## Get container PID
docker inspect --format="{{ .State.Pid }}" user_gpu
## Use nsenter to enter container. If you directly use docker exec -it to enter the container after container creation, there will be pam_session related permission issues. The <your_username> used in this example is tanboyu.tby
sudo nsenter --target PID --mount --uts --ipc --net --pid /usr/bin/su <your_username>
```
## 2. Compilation and Execution
In the previous step, we have completed the environment setup. Now we start the formal compilation and execution of RTP-LLM.
### 2.1 Code Pull
```shell
git clone https://github.com/alibaba/rtp-llm.git FastTransformer
```
### 2.2 Dependency Installation Related Issues
```shell
## nsightsystem (optional, for advanced users with source code performance debugging development needs)
The nsys in the container is located at /usr/local/cuda-12.6/bin/nsys. If needed, this path should be added to the execution path.
```
In addition, if network issues are encountered during compilation, it is recommended to change the source:
```shell
[global]
index-url = http://mirrors.aliyun.com/pypi/simple
trusted-host = mirrors.aliyun.com
```
### 2.3 Compile and Start Service
For demonstration convenience, we are using the Qwen2-0.5B small model:
```python
from transformers import AutoTokenizer, AutoModel

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")

# Load model
model = AutoModel.from_pretrained("Qwen/Qwen2-0.5B")

# Use the model for inference
inputs = tokenizer("This is a test sentence", return_tensors="pt")
outputs = model(**inputs)

# Output some results for verification
print(outputs)
```
Next, prepare the service startup script:
```python
### /hio_disk/tanboyu.tby/local_runner.py
import os
for key, value in os.environ.items():
    print(f"start env {key}={value}")

import sys
import signal
from threading import Thread
import requests
import time
import pathlib
import logging
import socket
from transformers import PreTrainedTokenizer

from typing import List, Dict, Optional

current_file_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(str(current_file_path.parent.parent.absolute()))

from maga_transformer.start_server import main as server_main
from maga_transformer.distribute.gang_info import members_from_test_env

tokenizer: Optional[PreTrainedTokenizer] = None
next_random_seed = int(os.environ.get("RANDOM_SEED", 114514))
max_seq_len = int(os.environ["MAX_SEQ_LEN"])

def wait_server_start(server_thread: Optional[Thread], port: int):
    start_time = time.time()
    while True:
        time.sleep(1)
        try:
            if server_thread and not server_thread.is_alive():
                raise SystemExit("Server thread dead!")
            res = requests.get(f"http://localhost:{port}/status")
            print(f"loop res: {res.text}")
            break
        except Exception as e:
            print(f"Waiting server on {port}, used {time.time() - start_time}s: {e}")
            continue

def wait_master_done(env_dict: Dict[str, str] = {}) -> None:
    dist_config_str = env_dict.get("GANG_CONFIG_STRING", os.environ.get("GANG_CONFIG_STRING"))
    if (not dist_config_str):
        raise RuntimeError("no gang config string, unexpected!")
    dist_members = members_from_test_env(dist_config_str)
    master_member = dist_members[0]
    master_host = master_member.ip
    master_port = master_member.server_port
    world_rank = int(os.environ.get("WORLD_RANK", 0))
    while True:
        logging.info(f"rank [{world_rank}] waiting for master {master_host}:{master_port} done")
        time.sleep(10)
        try:
            sock = socket.create_connection((master_host, master_port), timeout=1000)
            sock.close()
        except (socket.error, ConnectionRefusedError):
            break
    logging.info(f"rank [{world_rank}] master {master_host}:{master_port} done, this worker exit!")
    return

def script_exit(pgrp_set: bool = False):
    sys.stdout.flush()
    if pgrp_set:
        os.killpg(0, signal.SIGKILL)
        os._exit(0)
    else:
        os._exit(0)

if __name__ == '__main__':
    port = int(os.environ["START_PORT"])
    world_rank = int(os.environ.get("WORLD_RANK", 0))

    pgrp_set = False
    try:
        os.setpgrp()
        pgrp_set = True
    except Exception as e:
        logging.info(f"setpgrp error: {e}")

    server_thread = Thread(target=server_main)
    server_thread.start()
    print(f"server thread started, waiting...")
    wait_server_start(None, port)
    print(f"server start done!")

    import openai # you want `pip install openai==1.3.9`
    from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam
    openai.base_url = f"http://localhost:{int(os.environ['START_PORT'])}/"
    openai.api_key = "none"

    typed_messages: List[ChatCompletionMessageParam] = [
        ChatCompletionUserMessageParam(content="Who are you", role="user")
    ]

    response1 = openai.chat.completions.create(
        model="whatever",
        messages=typed_messages
    )
    print(f"response: {response1}")
    script_exit(pgrp_set)
```
Then start compilation and execution:
```shell
#!/bin/bash
set -x;
## Set Python execution command path
export PYTHON_BIN=/opt/conda310/bin/python;
## Specify user working directory
export USER_HOME=/hio_disk/tanboyu.tby;
## Can ensure immediate log printing during execution
export PYTHONUNBUFFERED=TRUE;

## Python path used for project imports
export PYTHONPATH=${USER_HOME}/FasterTransformer/:${PYTHONPATH}
## Specify log path
export PY_LOG_PATH=${USER_HOME}/FasterTransformer/logs

cd ${USER_HOME}/FasterTransformer

## Code compilation
bazelisk build //maga_transformer:maga_transformer --verbose_failures --config=cuda12_6 --keep_going || {
    echo "bazel build failed";
    exit 1;
};

## Create symbolic links since these two files are generated during compilation
ln -s ${USER_HOME}/bazel-bin/maga_transformer/cpp/proto/model_rpc_service_pb2_grpc.py maga_transformer/cpp/proto/;
ln -s ${USER_HOME}/bazel-bin/maga_transformer/cpp/proto/model_rpc_service_pb2.py maga_transformer/cpp/proto/;

## Users need to download https://huggingface.co/Qwen/Qwen2-0.5B on their own
export CHECKPOINT_PATH="/mnt/nas1/hf/Qwen2-0.5B";
export TOKENIZER_PATH=${CHECKPOINT_PATH}

export MODEL_TYPE="qwen_2";
echo $MODEL_TYPE
export LD_LIBRARY_PATH=/opt/conda310/lib/:/usr/local/nvidia/lib64:/usr/lib64:/usr/local/cuda/lib64:/usr/local/cuda-12.6/extras/CUPTI/lib64/

# export FT_SERVER_TEST=1
## Set TP parallelism
export TP_SIZE=1
## Set DP parallelism
export DP_SIZE=1
## Set EP parallelism
export EP_SIZE=$((TP_SIZE * DP_SIZE))
## See docs/MultiGPU.md
export WORLD_SIZE=$EP_SIZE
## See docs/MultiGPU.md
export LOCAL_WORLD_SIZE=$EP_SIZE
## Maximum number of text tokens requested by users
export MAX_SEQ_LEN=8192
## Maximum context size processed by the model at once
export MAX_CONTEXT_BATCH_SIZE=1
## Concurrency limit
export CONCURRENCY_LIMIT=8

## RUNTIME memory capacity limit
export RESERVER_RUNTIME_MEM_MB=4096
## Used to partition GPU memory into two parts, one for KV-Cache and one for computation
export WARM_UP=1
## Service startup port number
export START_PORT=61348
## Whether to enable performance profiling
export NSIGHT_PERF=0
## Whether to enable CUDA ASAN for memory detection
export CUDA_ASAN=0

echo "" > logs/engine.log;

export DEVICE_RESERVE_MEMORY_BYTES=-2048000000;

if [ $NSIGHT_PERF -eq 1 ]; then
    ## Choose to enable NSIGHT performance profiling
    NSIGHT_CMD="$PYTHON_BIN ${USER_HOME}/local_runner.py";
    rm -rf report*.nsys-rep;
    /usr/local/bin/nsys profile \
    -c cudaProfilerApi \
    -b none \
    --wait=primary \
    --cpuctxsw=none \
    --sample=none \
    --trace='cuda,nvtx' \
    --trace-fork-before-exec=true $NSIGHT_CMD;
elif [ $CUDA_ASAN -eq 1 ]; then
    ## Need to perform CUDA ASAN memory detection
    /usr/local/cuda/compute-sanitizer/compute-sanitizer --print-limit 100000 --target-processes all \
    $PYTHON_BIN ${USER_HOME}/local_runner.py;
else
    ## Most basic service startup solution
    $PYTHON_BIN ${USER_HOME}/local_runner.py;
fi

## Kill processes after script execution
ps xauww  | grep maga_ft | awk '{print $2}' | xargs kill -9;
```
### 2.4 Running Results
You can consider it successful when you see similar responses returned as below:
![](pics/response_success_example.png)