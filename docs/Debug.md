# RTP-LLM Code Debugging

The RTP-LLM project uses bazel for compilation. In the previous article [Building and Running RTP-LLM from Scratch](./Build.md), we learned how to compile and run RTP-LLM locally from scratch. In this article, we will teach you how to debug the code. RTP-LLM is mainly composed of Python and C++ code, so we will introduce Python source code debugging and C++ source code debugging in this article.

## 1. Test Code Preparation
For the example used in this article, all code in this chapter is placed under hio_disk/tanboyu.tby/backend_front.

Backend service startup code:
```python
## backend.py
import os
import sys
from threading import Thread
import requests
import time
import pathlib
import logging
import socket
from transformers import PreTrainedTokenizer
from typing import Dict, Optional

from maga_transformer.start_server import main as server_main
from maga_transformer.distribute.gang_info import members_from_test_env

for key, value in os.environ.items():
    print(f"start env {key}={value}")

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

if __name__ == "__main__":
    port = int(os.environ["START_PORT"])
    world_rank = int(os.environ.get("WORLD_RANK", 0))

    server_thread = Thread(target=server_main)
    server_thread.start()
    print(f"server thread started, waiting...")
    wait_server_start(None, port)
    print(f"server start done!")
```

Frontend request code:
```python
## fronted.py
import os
import sys
import logging
import signal
import openai # you want `pip install openai==1.3.9`
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam
from typing import List
def script_exit(pgrp_set: bool = False):
    sys.stdout.flush()
    if pgrp_set:
        os.killpg(0, signal.SIGKILL)
        os._exit(0)
    else:
        os._exit(0)


if __name__ == '__main__':
    pgrp_set = False
    try:
        os.setpgrp()
        pgrp_set = True
    except Exception as e:
        logging.info(f"setpgrp error: {e}")

    openai.base_url = f"http://localhost:{int(os.environ['START_PORT'])}/"
    openai.api_key = "none"
    typed_messages: List[ChatCompletionMessageParam] = [
        ChatCompletionUserMessageParam(content="What is the capital of China?", role="user")
    ]

    response = openai.chat.completions.create(
        model="whatever",
        messages=typed_messages
    )
    print(f"response: {response}")
    script_exit(pgrp_set)
```
Environment variable loading script:
```shell
## prev.sh
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

if [ $USE_COMPILE eq 1 ]; then
    ## Code compilation
    bazelisk build //maga_transformer:maga_transformer --compilation_mode=dbg --verbose_failures --config=cuda12_6 --keep_going --jobs=64 || {
        echo "bazel build failed";
        exit 1;
    };
fi

## Create symbolic links, as these two files are generated during compilation
ln -s ${USER_HOME}/bazel-bin/maga_transformer/cpp/proto/model_rpc_service_pb2_grpc.py maga_transformer/cpp/proto/;
ln -s ${USER_HOME}/bazel-bin/maga_transformer/cpp/proto/model_rpc_service_pb2.py maga_transformer/cpp/proto/;

## Users need to download https://huggingface.co/Qwen/Qwen2-0.5B on their own
export CHECKPOINT_PATH="/mnt/nas1/hf/Qwen2-0.5B";
export TOKENIZER_PATH=${CHECKPOINT_PATH}

export MODEL_TYPE="qwen_2";
export LD_LIBRARY_PATH=/opt/conda310/lib/:/usr/local/nvidia/lib64:/usr/lib64:/usr/local/cuda/lib64:/usr/local/cuda-12.6/extras/CUPTI/lib64/

# export FT_SERVER_TEST=1
## Set TP parallelism
export TP_SIZE=2
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
```

Backend startup script:
```shell
## backend.sh
export USE_COMPILE=1
source prev.sh
echo "user_home: $USER_HOME"
$PYTHON_BIN ${USER_HOME}/backend_front/backend.py
```

Frontend startup script:
```shell
## fronted.sh
export USE_COMPILE=0
source prev.sh
echo $USER_HOME
$PYTHON_BIN ${USER_HOME}/backend_front/fronted.py
```
## 2. Python Code Debugging
We use VSCode for code development and debugging. Python code debugging is convenient with a graphical interface, making it more intuitive. Below is the configuration file for launch debugging:
```json
// launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Run FasterTransformer with Qwen2-0.5B",
            "type": "debugpy",
            "request": "launch",
            // "processId": "${command:pickProcess}", // Use the process selected by the user
            "program": "${file}",
            // "console": "integratedTerminal",
            "cwd": "/hio_disk/tanboyu.tby/FasterTransformer",
            "env": {
                "CUDA_VISIBLE_DEVICES": "2,3",
                "CHECKPOINT_PATH": "/mnt/nas1/hf/Qwen2-0.5B",
                "TOKENIZER_PATH": "/mnt/nas1/hf/Qwen2-0.5B",
                "MODEL_TYPE": "qwen_2",
                "LD_LIBRARY_PATH": "/opt/conda310/lib/:/usr/local/nvidia/lib64:/usr/lib64:/usr/local/cuda/lib64:/usr/local/cuda-12.6/extras/CUPTI/lib64/",
                "TP_SIZE": "2",
                "DP_SIZE": "1",
                "EP_SIZE": "1",  // Manually calculate EP_SIZE = TP_SIZE * DP_SIZE
                "WORLD_SIZE": "2",  // Consistent with EP_SIZE
                "LOCAL_WORLD_SIZE": "1",  // Consistent with EP_SIZE
                "MAX_SEQ_LEN": "1024",
                "MAX_CONTEXT_BATCH_SIZE": "1",
                "CONCURRENCY_LIMIT": "8",
                "RESERVER_RUNTIME_MEM_MB": "4096",
                "WARM_UP": "1",
                "START_PORT": "61348",
                "NSIGHT_PERF": "0",
                "CUDA_ASAN": "0"
            },
            "args": []
        },
    ],
}
```
When debugging, you need to switch to /backend_front/backend.py in the interface, and then we can debug normally.
![](./pics/python_debug.png)
Actually, when the code executes normally, it will come here, where the underlying implementation calls the C++ code under maga_transformer/cpp to start LocalRpcServer.
![](./pics/launch_rpc_local_server.png)

## 3. Server Code Debugging
First, start the service: bash backend.sh. When it looks like the following, the service is successfully started.
![](./pics/backend_server_success.png)
View the relevant startup processes as follows:
```shell
[tanboyu.tby@mainse-buffer011161048115.na132 /hio_disk/tanboyu.tby/FasterTransformer]
$ps aux | grep maga_ft
tanboyu+  21090  5.7  0.1 41536432 1105896 pts/20 Sl+ 23:55   0:15 maga_ft_backend_server
tanboyu+  21251 54.8  0.8 136379296 6902104 pts/20 Sl+ 23:55   2:24 maga_ft_rank-0
tanboyu+  21252  122  0.8 142631432 6953340 pts/20 Sl+ 23:55   5:22 maga_ft_rank-1
tanboyu+  22451  5.8  0.1 50260920 1167256 pts/20 Sl+ 23:56   0:11 maga_ft_frontend_server_0
tanboyu+  22452  5.8  0.1 50260936 1175892 pts/20 Sl+ 23:56   0:11 maga_ft_frontend_server_1
tanboyu+  22453  5.6  0.1 50260920 1172820 pts/20 Sl+ 23:56   0:11 maga_ft_frontend_server_2
tanboyu+  22454  5.5  0.1 50259900 1167200 pts/20 Sl+ 23:56   0:10 maga_ft_frontend_server_3
tanboyu+  23916  0.0  0.0   7996   900 pts/16   S+   23:59   0:00 grep --color=auto maga_ft
```
After the service starts, we will find that there is a maga_ft_backend_server process, which is the main process of the inference service startup, while maga_ft_rank-0 and maga_ft_rank-1 are the corresponding child processes. This number is determined by our configuration. We set TP_SIZE = 2, so there will be two corresponding processes here. Additionally, four maga_ft_frontend_server frontend service processes will be started by default to receive external requests.

Next, we start gdb debugging: gdb -p 21251. After setting breakpoints, execute bash fronted.sh, which will hit the breakpoint. Then we can view the code path according to the stack.
![](./pics/rtp-llm_backend_gdb_debug.png)