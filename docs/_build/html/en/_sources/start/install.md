# Install RTP-LLM

We provide multiple ways to install RTP-LLM.
* If you need to run **DeepSeek V3/R1**, it is recommended to refer to [DeepSeek V3/R1 Support](../references/deepseek/index.rst) and use Docker to run
* If you need to run **Kimi-K2**, it is recommended to refer to [Kimi-K2 Support](../references/kimi/index.rst) and use Docker to run
* If you need to run **QwenMoE**, it is recommended to refer to [Qwen MoE Support](../references/qwen/index.rst) and use Docker to run


To speed up installation, it is recommended to use pip to install dependencies:

## Method 1: With pip

```bash
pip install --upgrade pip
pip install "rtp_llm>=0.0.1"
```


## Method 2: From source
| os | Python | NVIDIA GPU | AMD | Compile Tools|
| -------| -----| ----| ----|----|
| Linux | 3.10 | Compute Capability 7.0 or higher <br> ✅ RTX20xx<br>  ✅RTX30xx<br>  ✅RTX40xx<br>  ✅V100<br>  ✅T4<br>  ✅A10/A30/A100<br>  ✅L40/L20<br>  ✅H100/H200/H20/H800.. <br> | ✅MI308X | bazelisk |


```bash
# Use the last release branch
git clone -b release/0.0.1 git@gitlab.alibaba-inc.com:foundation_models/RTP-LLM.git
cd RTP-LLM

# build RTP-LLM whl target
# --config=cuda12_6 build target for NVIDIA GPU with cuda12_6
# --config=rocm build target for AMD
bazelisk build //rtp_llm:rtp_llm --verbose_failures --config=cuda12_6 --test_output=errors --test_env="LOG_LEVEL=INFO"  --jobs=64

ln  -sf `pwd`/bazel-out/k8-opt/bin/rtp_llm/cpp/proto/model_rpc_service_pb2.py  `pwd`/rtp_llm/cpp/proto/

```


## Method 3: Using docker
More Docker versions can be obtained from [RTP-LLM Release](../release/index.rst)
```bash
docker run --gpus all \
 --shm-size 32g \
 -p 30000:30000 \
 -v /mnt:/mnt \
 -v /home:/home \
 --ipc=host \
 hub.docker.alibaba-inc.com/isearch/rtp_llm_gpu_cuda12:2025_07_08_21_00_a1ed8e8 \
  /opt/conda310/bin/python -m rtp_llm.start_server \
   --checkpoint_path=/mnt/nas1/hf/models--Qwen--Qwen1.5-0.5B-Chat/snapshots/6114e9c18dac0042fa90925f03b046734369472f/ \
    --model_type=qwen_2 --start_port=30000

```