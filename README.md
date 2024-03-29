[English](README.md) [中文](README_cn.md)

## About
* rtp-llm is a Large Language Model (LLM) inference acceleration engine developed by Alibaba's Foundation Model Inference Team. It is widely used within Alibaba Group, supporting LLM service across multiple business units including Taobao, Tmall, Idlefish, Cainiao, Amap, Ele.me, AE, and Lazada.
* The rtp-llm project is a sub-project of the [havenask](https://github.com/alibaba/havenask)

## Features
### Production Proven
Applied in numerous LLM scenarios, such as:
* Taobao Wenwen
* Alibaba's international AI platform, [Aidge](https://aidc-ai.com/)
* [OpenSearch LLM Smart Q&A Edition](https://www.aliyun.com/activity/bigdata/opensearch/llmsearch)
* [Large Language Model based Long-tail Query Rewriting in Taobao Search](https://arxiv.org/abs/2311.03758)

### High Performance
* Utilizes high-performance CUDA kernels, including PagedAttention, FlashAttention, FlashDecoding, etc.
* Implements WeightOnly INT8 Quantization with automatic quantization at load time
* Adaptive KVCache Quantization
* Detailed optimization of dynamic batching overhead at the framework level
* Specially optimized for the V100 GPU

### Flexibility and Ease of Use
* Seamless integration with the HuggingFace models, supporting multiple weight formats such as SafeTensors, Pytorch, and Megatron
* Deploys multiple LoRA services with a single model instance
* Handles multimodal inputs (combining images and text)
* Enables multi-machine/multi-GPU tensor parallelism
* Supports P-tuning models

### Advanced Acceleration Techniques
* Loads pruned irregular models
* Contextual Prefix Cache for multi-turn dialogues
* System Prompt Cache
* Speculative Decoding
* Medusa for advanced parallelization strategies

## How to Use
### Requirements
* Operating System: Linux
* Python: 3.10
* NVIDIA GPU: Compute Capability 7.0 or higher (e.g., RTX20xx, RTX30xx, RTX40xx, V100, T4, A10/A30/A100, L4, H100, etc.)
### Startup example
1. docker
```bash
cd rtp-llm/docker
# IMAGE_NAME =
# if cuda11: registry.cn-hangzhou.aliyuncs.com/havenask/rtp_llm:deploy_image_cuda11
# if cuda12: registry.cn-hangzhou.aliyuncs.com/havenask/rtp_llm:deploy_image_cuda12
sh ./create_container.sh <CONTAINER_NAME> <IMAGE_NAME>
sh CONTAINER_NAME/sshme.sh

cd ../
# start http service
TOKENIZER_PATH=/path/to/tokenizer CHECKPOINT_PATH=/path/to/model MODEL_TYPE=your_model_type FT_SERVER_TEST=1 python3 -m maga_transformer.start_server
# request to server
curl -XPOST http://localhost:8088 -d '{"prompt": "hello, what is your name", "generate_config": {"max_new_tokens": 1000}}'
```

2. whl
```bash
# Install rtp-llm
cd rtp-llm
# For cuda12 environment, please use requirements_torch_gpu_cuda12.txt
pip3 install -r ./open_source/deps/requirements_torch_gpu.txt
# Use the corresponding whl from the release version, here's an example for the cuda11 version 0.1.0, for the cuda12 whl package please check the release page.
pip3 install maga_transformer-0.1.9+cuda118-cp310-cp310-manylinux1_x86_64.whl
# start http service

cd ../
TOKENIZER_PATH=/path/to/tokenizer CHECKPOINT_PATH=/path/to/model MODEL_TYPE=your_model_type FT_SERVER_TEST=1 python3 -m maga_transformer.start_server
# request to server
curl -XPOST http://localhost:8088 -d '{"prompt": "hello, what is your name", "generate_config": {"max_new_tokens": 1000}}'
```

### FAQ
1. libcufft.so
    
    **Error log**: `OSError: libcufft.so.11: cannot open shared object file: No such file or directory`

    **Resolution**: Please check whether cuda and rtp-llm versions are matched

2. libth_transformer.so

    **Error log**: `OSError: /rtp-llm/maga_transformer/libs/libth_transformer.so: cannot open shared object file: No such file or directory`

    **Resolution**: If installed via whl or docker(which means not a bazel build), please check your current directory is not rtp-llm, or python will use relative path package instead of installed whl

3. Bazel build time out

    **Error log**: `ERROR: no such package '@pip_gpu_cuda12_torch//': rules_python_external failed: (Timed out)`

    **Resolution**:
     1. change pip mirror repository in open_source/deps/pip.bzl, add extra_pip_args=["--index_url=xxx"]
     2. pip install requirements manually, especially for pytorch, for that bazel build has a 600s timeout by default, which may not be enough for pytorch downloading

## Documentation
* [Test in Deploy Docker](docs/DeployDocker.md)
* [Serving Example](docs/OpenAI-Tutorial.md)
* [RWKV-Runner Example](docs/RWKV-Runner.md)
* [Python Library Example](docs/HF.md)
* [Using RTP-LLm in Aliyun Ecs](https://zhuanlan.zhihu.com/p/679610919)
* [Configuration Parameters](docs/Config.md)
* [Source Code Build](docs/Build.md)
* [Request Format](docs/Request.md)
* [Multi GPU Inference](docs/MultiGPU.md)
* [LoRA](docs/LoRA-Tutorial.md)
* [PTuning](docs/PTuning-Tutorial.md)
* [SystemPrompt](docs/SystemPrompt-Tutorial.md)
* [ReuseKVCache](docs/ReuseKVCache-Tutorial.md)
* [Multimodal](docs/Multimodal-Tutorial.md)
* [Structured Pruning](docs/Sparse-Tutorial.md)
* [Speculative Sampling](docs/SpeculativeDecoding-Tutroial.md)
* [Roadmap](docs/Roadmap.md)
* [Contributing](docs/Contributing.md)
* [Benchmark&Performance](benchmark/README.md)

## Acknowledgments
Our project is mainly based on [FasterTransformer](https://github.com/NVIDIA/FasterTransformer), and on this basis, we have integrated some kernel implementations from [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM). FasterTransformer and TensorRT-LLM have provided us with reliable performance guarantees. [Flash-Attention2](https://github.com/Dao-AILab/flash-attention) and [cutlass](https://github.com/NVIDIA/cutlass) have also provided a lot of help in our continuous performance optimization process. Our continuous batching and increment decoding draw on the implementation of [vllm](https://github.com/vllm-project/vllm); sampling draws on [transformers](https://github.com/huggingface/transformers), with speculative sampling integrating [Medusa](https://github.com/FasterDecoding/Medusa)'s implementation, and the multimodal part integrating implementations from [llava](https://github.com/haotian-liu/LLaVA) and [qwen-vl](https://github.com/QwenLM/Qwen-VL). We thank these projects for their inspiration and help.
## External Application Scenarios (Continuously Updated)
* Taobao Wenda
* Alibaba's International AI Platform [Aidge](https://aidc-ai.com/)
* [OpenSearch LLM Smart Q&A Edition](https://www.aliyun.com/activity/bigdata/opensearch/llmsearch)
* [Large Language Model based Long-tail Query Rewriting in Taobao Search](https://arxiv.org/abs/2311.03758)
## Supported Model List
### LLM
* Aquila and Aquila2 (BAAI/AquilaChat2-7B, BAAI/AquilaChat2-34B, BAAI/Aquila-7B, BAAI/AquilaChat-7B, etc.)
* Baichuan and Baichuan2 (baichuan-inc/Baichuan2-13B-Chat, baichuan-inc/Baichuan-7B)
* Bloom (bigscience/bloom, bigscience/bloomz)
* ChatGlm (THUDM/chatglm2-6b, THUDM/chatglm3-6b)
* Falcon (tiiuae/falcon-7b, tiiuae/falcon-40b, tiiuae/falcon-rw-7b, etc.)
* GptNeox (EleutherAI/gpt-neox-20b)
* GPT BigCode (bigcode/starcoder, bigcode/starcoder2)
* LLaMA and LLaMA-2 (meta-llama/Llama-2-7b, meta-llama/Llama-2-13b-hf, meta-llama/Llama-2-70b-hf, lmsys/vicuna-33b-v1.3, 01-ai/Yi-34B, xverse/XVERSE-13B, etc.)
* MPT (mosaicml/mpt-30b-chat, etc.)
* Phi (microsoft/phi-1_5, etc.)
* Qwen (Qwen/Qwen-7B, Qwen/Qwen-7B-Chat, Qwen/Qwen-14B, Qwen/Qwen-14B-Chat, Qwen1.5, etc.)
* InternLM (internlm/internlm-7b, internlm/internlm-chat-7b, etc.)
* Gemma (google/gemma-it, etc)
* Mixtral (mistralai/Mixtral-8x7B-v0.1, etc)

### LLM + Multimodal
* LLAVA (liuhaotian/llava-v1.5-13b, liuhaotian/llava-v1.5-7b)
* Qwen-VL (Qwen/Qwen-VL)

## Contact Us
#### DingTalk Group
<img src=picture/dingding.png width="200px">

#### WeChat Group
<img src=picture/weixin.png width="200px">
