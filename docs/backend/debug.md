# Debugging RTP-LLM

The RTP-LLM project uses the Bazel build system. After learning how to compile and run RTP-LLM locally from scratch, this article will explain how to debug the code. Since RTP-LLM is primarily composed of a combination of Python and C++ code, we will introduce several commonly used debugging methods in this guide.

## Part 1: debug Python Code

### Method 1: logging or print

When running the debug mode locally, modify the self.frontend_server_count = 4 to 1 in the ServerConfig class within rtp_llm/config/py_config_modules.py. This change ensures only one frontend server is launched, allowing print statements and logging outputs to be directly displayed in the terminal.


### Method 2: python debugger
Assuming we already have an existing container, access it and set up an SSH port mapping to enable remote connections.
```bash
sudo ssh-keygen -A
sudo /usr/sbin/sshd -p 37228
```
Access the container via the SSH extension in VS Code.

```bash
Host dev-container
    HostName ip
    User user
    Port 37228
```
After accessing the container via VS Code, ensure the Python debugger extension is installed within the container.
![alt text](../pics/debug_image-debugger.png)

Write the launch.json configuration file for debugging.
Here’s an example of a VS Code launch.json configuration file for debugging Python and C++ code in the RTP-LLM project (adjust paths as needed):
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Run RTP-LLM with Qwen2-0.5B",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "cwd": "${path/to/project}/RTP-LLM/",
            "env": {
                "PYTHONPATH": "$PYTHONPATH:${path/to/project}/RTP-LLM",
                "PYTHON_BIN": "/opt/conda310/bin/python",
                "CUDA_VISIBLE_DEVICES": "2,3",
                "CHECKPOINT_PATH": "/mnt/nas1/hf/Qwen2-0.5B",
                "TOKENIZER_PATH": "/mnt/nas1/hf/Qwen2-0.5B",
                "MODEL_TYPE": "qwen_2",
                "LD_LIBRARY_PATH": "",
                "TP_SIZE": "2",
                "DP_SIZE": "1",
                "EP_SIZE": "2", 
                "WORLD_SIZE": "2",  
                "LOCAL_WORLD_SIZE": "1", 
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
Set up the required environment variables before running or debugging the code.

```bash
#!/bin/bash
set -x;
## set python path
export PYTHON_BIN=/opt/conda310/bin/python;
## set user home
export USER_HOME=${/path/to/home};

export PYTHONUNBUFFERED=TRUE;
export PYTHONPATH=${path/to/project}/:${PYTHONPATH}

export PY_LOG_PATH=${path/to/project}/logs
export CHECKPOINT_PATH="/mnt/nas1/hf/Qwen2-0.5B";
export TOKENIZER_PATH=${CHECKPOINT_PATH}
export MODEL_TYPE="qwen_2";
export LD_LIBRARY_PATH=/opt/conda310/lib/:/usr/local/nvidia/lib64:/usr/lib64:/usr/local/cuda/lib64:/usr/local/cuda-12.6/extras/CUPTI/lib64/

export TP_SIZE=2
export DP_SIZE=1
export EP_SIZE=$((TP_SIZE * DP_SIZE))
export WORLD_SIZE=$EP_SIZE
export LOCAL_WORLD_SIZE=$EP_SIZE
## request max token number
export MAX_SEQ_LEN=8192
export MAX_CONTEXT_BATCH_SIZE=1
export CONCURRENCY_LIMIT=8

export RESERVER_RUNTIME_MEM_MB=4096
export WARM_UP=1
export START_PORT=61348
export NSIGHT_PERF=0
export CUDA_ASAN=0
export DEVICE_RESERVE_MEMORY_BYTES=-20480000

```

open the file containing start_server.py, set breakpoints, and begin debugging.

![alt text](../pics/debug_image-1.png)


## Part 1: debug c++ code
### Method 1: logging
Add the following log statements for output:
```cpp
RTP_LLM_LOG_DEBUG("request [%ld] enqueue success", request_id);
```
other similar functions include:
```cpp
RTP_LLM_LOG_INFO
RTP_LLM_LOG_WARNING
RTP_LLM_LOG_ERROR
```
Set the log level using the LOG_LEVEL="INFO" environment variable.

### Method 2: GDB debug
#### GDB debug core
When the code crashes with a core dump in the container, a core file is generated (e.g., core-rtp_llm_backend-78933-1757510512). To debug:
```
gdb /opt/conda310/bin/python3 core-rtp_llm_backend-78933-1757510512
```
After loading the core file into GDB, run the bt (backtrace) command to display the error stack trace.

![alt text](../pics/debug_image-2.png)

```bash
f 4       # check rtp_llm::ScoreStream::ScoreStream info
info locals     
p stream         
```


![alt text](../pics/debug_image-3.png)

check propose_stream_ info

```
p *(this->propose_stream_._M_ptr->sp_output_buffer_._M_ptr->tokens._M_ptr) 
```

check tokens info

![alt text](../pics/debug_image-4.png)

A null pointer (data_ = 0) was detected, causing a memcpy error.

#### GDB debug process

```bash
MODEL_TYPE=qwen_7b     \
CHECKPOINT_PATH=/mnt/nas1/hf/Qwen-7B-Chat/     \
TOKENIZER_PATH=/mnt/nas1/dm/qwen_sp/qwen_tokenizer     \
TP_SIZE=1     \
SP_TYPE=vanilla     \
GEN_NUM_PER_CIRCLE=5     \
SP_MODEL_TYPE=qwen_1b8     \
SP_CHECKPOINT_PATH=/mnt/nas1/hf/qwen_1b8_sft/     \
WARM_UP=1     \
INT8_MODE=1     \
SP_INT8_MODE=1     \
REUSE_CACHE=1     \
START_PORT=26666   \
/opt/conda310/bin/python3 -m rtp_llm.start_server 
```

After starting the service, you can view the relevant processes as follows:


A rtp_llm_backend_server process will be running as the main process for the inference service. If TP_SIZE=2 is set, you will see two child processes (e.g., rank-0 and rank-1) for tensor parallelism.
A rtp_llm_frontend_server_0 frontend service process will be active to handle external requests.

```bash
yanxi.w+  40954  40801 44 14:03 pts/8    00:00:41 rtp_llm_backend_server
yanxi.w+  41356  40801 20 14:04 pts/8    00:00:11 rtp_llm_frontend_server_0
```

To begin debugging with GDB:
Attach GDB to the target process (e.g., PID 40954):
```
gdb -p 40954  
```
Set breakpoints in the code
Use curl to send a test request and trigger the breakpoint:

``` bash
curl -X POST http://127.0.0.1:26000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "messages": [
      {
        "role": "user",
        "content": "杭州的天气怎么样？"
      }
    ],
    "stream": false,
    "aux_info": true,
    "max_tokens": 10
  }'
```
The breakpoint will be triggered, and you can then examine the code path by inspecting the stack trace.

![alt text](../pics/debug_image-5.png)

### Method 3: Unit Test

Example: Unit Testing for the ViT Module

File Structure of the ViT Module

![alt text](../pics/debug_image-6.png)

Steps to Create Unit Tests
Create Test Files:

Add a .cc test file (e.g., multimodal_processor_test.cc) under the test directory.
Write Google Test (gtest) cases using assertions like EXPECT_EQ to validate behavior.
```cpp
#include <gtest/gtest.h>  
TEST(MultimodalProcessorTest, BasicFunctionality) {  
  // Test logic here  
  EXPECT_EQ(result, expected_value);  
}  
```
Define the BUILD File:

```python
cc_test(  
    name = "multimodal_processor_test",  
    srcs = ["multimodal_processor_test.cc"],  
    deps = [  
        "//rtp_llm/cpp/multimodal_processor:main_lib",  
        "@gtest//:gtest_main",  
    ],  
)  

```
Run the Test:
Execute the following command in the project’s container:
```bash
bazelisk test  rtp_llm/cpp/multimodal_processor/test:multimodal_processor_test   --jobs=48 --test_output=streamed --config=cuda12_6
```

## Part 3: bazel smoke test

Add Smoke Test Cases:

In rtp_llm/test/smoke/BUILD, define smoke test targets.
Extend case_runner.py to include new API endpoints and comparers.

```python
...
    elif request_endpoint.startswith("/rtp_llm/worker_status"):
        comparer_cls = WorkerStatusComparer
...
```

Implement the Comparer:

```python
    def compare_result(
        self, expect_result: WorkStatus, actual_result: WorkStatus
    ) -> None:
```       
Prepare Test Data:

Place JSON files (e.g., expected_worker_status.json) in the smoke test directory for result validation.

Run Smoke Tests:

```
bazelisk test internal_source/rtp_llm/test/smoke:worker_status_reuse_cache  --config=ppu --test_env='CUDA_VISIBLE_DEVICES=11,12,13' --cache_test_results=no 
```
