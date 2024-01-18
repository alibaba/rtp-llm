[English](README.md) [中文](README_cn.md)

## About
* rtp-llm is an LLM inference acceleration engine developed by Alibaba's big model prediction team. rtp-llm is widely used within Alibaba and supports the large model inference operations of various departments, including Taobao, Tmall, Cainiao, Amap, Ele.me, AE, Lazada, and others.
* rtp-llm offers high-performance, low-cost, and user-friendly inference services, helping customers and developers tailor inference services suitable for their own businesses, thus boosting business growth.
* rtp-llm is a subproject of the [havenask](https://github.com/alibaba/havenask) project.
## Features
### High Performance
* Utilizes high-performance cuda kernels.
* The framework has finely optimized the overhead of dynamic batching.
* Supports paged attention and kv cache quantization.
* Supports flash attention2.
* Supports weight only INT8 quantization with automatic quantization at load time.
* Specially optimized for the V100.
### Extremely Flexible and Easy to Use
* Seamlessly interfaces with popular HuggingFace models, supporting multiple weight formats without the need for additional conversion processes.
* Supports deployment of multiple LoRA services with a single model instance.
* Supports multimodal inputs (mixed image and text).
* Supports multi-machine/multi-card tensor parallelism.
* Supports loading P-tuning models.
### Advanced Inference Acceleration Methods
* Supports loading of irregular models after pruning.
* Supports multi-round dialogue context Cache.
* Supports Speculative Decoding acceleration.
* Supports Medusa acceleration.
## How to Use
### Requirements
* Operating System: Linux
* Python: 3.10
* NVIDIA GPU: Compute Capability 7.0 or higher (e.g., V100, T4, RTX20xx, A100, L4, H100, etc.)
### Installation and Startup
```bash
# Install rtp-llm
cd rtp-llm
# For cuda12 environment, please use requirements_torch_gpu_cuda12.txt
pip3 install -r ./maga_transformer/requirements_torch_gpu.txt
# Use the corresponding whl from the release version, here's an example for the cuda11 version 0.1.0, for the cuda12 whl package please check the release page.
pip3 install maga_transformer-0.0.1+cuda118-cp310-cp310-manylinux1_x86_64.whl
# Modify the model path in test.py, and start the program directly
python3 example/test.py
# Or start the http service
# rtp-llm uses fastapi to build high-performance model services and uses asynchronous programming to minimize CPU thread pressure interference with efficient GPU operation
export TOKENIZER_PATH=/path/to/tokenizer
export CHECKPOINT_PATH=/path/to/model
export FT_SERVER_TEST=1
python3 -m maga_transformer.start_server
# request to server
curl -XPOST http://localhost:8088 -d '{"prompt": "hello, what is your name", "generate_config": {"max_new_tokens": 1000}}'
```
Refer to the detailed documentation below for model configuration parameters, etc.
## Documentation
* [Configuration Parameters](docs/Config.md)
* [Source Code Build](docs/Build.md)
* [Request Format](docs/Request.md)
* [OpenAI Interface](docs/OpenAI-Tutorial.md)
* [LoRA](docs/LoRA-Tutorial.md)
* [PTuning](docs/PTuning-Tutorial.md)
* [Multimodal](docs/Multimodal-Tutorial.md)
* [Structured Pruning](docs/Sparse-Tutorial.md)
* [Speculative Sampling](docs/SpeculativeDecoding-Tutorial.md)
* [Using as a Python Library](docs/HF.md)
* [Roadmap](docs/Roadmap.md)
* [Contributing](docs/Contributing.md)
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
* GPT BigCode (bigcode/starcoder, etc.)
* LLaMA and LLaMA-2 (meta-llama/Llama-2-7b, meta-llama/Llama-2-13b-hf, meta-llama/Llama-2-70b-hf, lmsys/vicuna-33b-v1.3, 01-ai/Yi-34B, xverse/XVERSE-13B, etc.)
* MPT (mosaicml/mpt-30b-chat, etc.)
* Phi (microsoft/phi-1_5, etc.)
* Qwen (Qwen/Qwen-7B, Qwen/Qwen-7B-Chat, Qwen/Qwen-14B, Qwen/Qwen-14B-Chat, etc.)
* InternLM (internlm/internlm-7b, internlm/internlm-chat-7b, etc.)
### LLM + Multimodal
* LLAVA (liuhaotian/llava-v1.5-13b, liuhaotian/llava-v1.5-7b)
* Qwen-VL (Qwen/Qwen-VL)
## Contact Us
#### DingTalk Group
<img src=picture/dingding.png width="200px">
#### WeChat Group
<img src=picture/weixin.JPG width="200px">