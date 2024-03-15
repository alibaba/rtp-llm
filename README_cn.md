[English](README.md) [中文](README_cn.md)

## 介绍

* rtp-llm 是阿里巴巴大模型预测团队开发的 LLM 推理加速引擎。rtp-llm 在阿里巴巴内部被广泛使用，支持了包括淘宝、天猫、闲鱼、菜鸟、高德、饿了么、AE、Lazada 等多个部门的大模型推理业务。
* rtp-llm项目是[havenask](https://github.com/alibaba/havenask)项目的子项目。

## 特点
### 实战验证
在众多LLM场景中得到实际应用与检验
* 淘宝问问
* 阿里巴巴国际AI平台[Aidge](https://aidc-ai.com/)
* [OpenSearch LLM智能问答版](https://www.aliyun.com/activity/bigdata/opensearch/llmsearch)
* [Large Language Model based Long-tail Query Rewriting in Taobao Search](https://arxiv.org/abs/2311.03758)

### 高性能
* 使用高性能的 CUDA kernel, 包括 PagedAttention、FlashAttention、FlashDecoding 等
* WeightOnly INT8 量化，加载时自动量化
* 自适应 KVCache 量化
* 框架上对动态凑批的 overhead 进行了细致优化
* 对 V100 进行了特别优化

### 灵活易用
* 和 HuggingFace 无缝对接，支持 SafeTensors/Pytorch/Megatron 等多种权重格式
* 单模型实例同时部署多 LoRA 服务
* 多模态(图片和文本混合输入)
* 多机/多卡 Tensor 并行
* P-tuning 模型

### 高级加速
* 剪枝后的不规则模型加载
* 多轮对话上下文 Prefix Cache
* System Prompt Cache
* Speculative Decoding
* Medusa

## 使用方法
### 需求
* 操作系统: Linux
* Python: 3.10
* NVIDIA GPU: Compute Capability 7.0 或者更高 (e.g., RTX20xx, RTX30xx, RTX40xx, V100, T4, A10/A30/A100, L4, H100, etc.)

### 安装和启动
```bash
# 安装rtp-llm
cd rtp-llm
# cuda12的环境请使用 requirements_torch_gpu_cuda12.txt
pip3 install -r ./open_source/deps/requirements_torch_gpu.txt
# 使用release版本中对应的whl, 这里以0.1.0的cuda11版本为例子，cuda12的whl包请查看release发布页。
pip3 install maga_transformer-0.0.1+cuda118-cp310-cp310-manylinux1_x86_64.whl
# 启动 http 服务
TOKENIZER_PATH=/path/to/tokenizer CHECKPOINT_PATH=/path/to/model MODEL_TYPE=your_model_type FT_SERVER_TEST=1 python3 -m maga_transformer.start_server
# request to server
curl -XPOST http://localhost:8088 -d '{"prompt": "hello, what is your name", "generate_config": {"max_new_tokens": 1000}}'
```

## 文档
* [在Deploy Docker中测试](docs/DeployDocker.md)
* [启动服务样例](docs/OpenAI-Tutorial.md)
* [RWKV-Runner 样例](docs/RWKV-Runner.md)
* [Python Library 样例](docs/HF.md)
* [在Aliyun Ecs中使用RTP-LLm](https://zhuanlan.zhihu.com/p/679610919)
* [配置参数](docs/Config.md)
* [源码构建](docs/Build.md)
* [内置请求格式](docs/Request.md)
* [多卡推理](docs/MultiGPU.md)
* [LoRA](docs/LoRA-Tutorial.md)
* [PTuning](docs/PTuning-Tutorial.md)
* [SystemPrompt](docs/SystemPrompt-Tutorial.md)
* [多轮会话](docs/ReuseKVCache-Tutorial.md)
* [多模态](docs/Multimodal-Tutorial.md)
* [结构化剪枝](docs/Sparse-Tutorial.md)
* [投机采样](docs/SpeculativeDecoding-Tutroial.md)
* [Roadmap](docs/Roadmap.md)
* [Contributing](docs/Contributing.md)
* [Benchmark&Performance](benchmark/README.md)

## 致谢
我们的项目主要基于[FasterTransformer](https://github.com/NVIDIA/FasterTransformer)，并在此基础上集成了[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)的部分kernel实现。FasterTransformer和TensorRT-LLM为我们提供了可靠的性能保障。[Flash-Attention2](https://github.com/Dao-AILab/flash-attention)和[cutlass](https://github.com/NVIDIA/cutlass)也在我们持续的性能优化过程中提供了大量帮助。我们的continuous batching和increment decoding参考了[vllm](https://github.com/vllm-project/vllm)的实现；采样参考了[transformers](https://github.com/huggingface/transformers)，投机采样部分集成了[Medusa](https://github.com/FasterDecoding/Medusa)的实现，多模态部分集成了[llava](https://github.com/haotian-liu/LLaVA)和[qwen-vl](https://github.com/QwenLM/Qwen-VL)的实现。感谢这些项目对我们的启发和帮助。

## 支持模型列表

### LLM
* Aquila 和 Aquila2(BAAI/AquilaChat2-7B, BAAI/AquilaChat2-34B, BAAI/Aquila-7B, BAAI/AquilaChat-7B等)
* Baichuan 和 Baichuan2 (baichuan-inc/Baichuan2-13B-Chat, baichuan-inc/Baichuan-7B)
* Bloom (bigscience/bloom, bigscience/bloomz)
* ChatGlm (THUDM/chatglm2-6b, THUDM/chatglm3-6b)
* Falcon (tiiuae/falcon-7b, tiiuae/falcon-40b, tiiuae/falcon-rw-7b等)
* GptNeox (EleutherAI/gpt-neox-20b)
* GPT BigCode (bigcode/starcoder等)
* LLaMA 和 LLaMA-2 (meta-llama/Llama-2-7b, meta-llama/Llama-2-13b-hf, meta-llama/Llama-2-70b-hf, lmsys/vicuna-33b-v1.3, 01-ai/Yi-34B, xverse/XVERSE-13B等)
* MPT (mosaicml/mpt-30b-chat等)
* Phi (microsoft/phi-1_5等)
* Qwen (Qwen/Qwen-7B, Qwen/Qwen-7B-Chat, Qwen/Qwen-14B, Qwen/Qwen-14B-Chat等)
* InternLM (internlm/internlm-7b, internlm/internlm-chat-7b等)

### LLM + 多模态
* LLAVA (liuhaotian/llava-v1.5-13b, liuhaotian/llava-v1.5-7b)
* Qwen-VL (Qwen/Qwen-VL)

## 联系我们
#### 钉钉群
<img src=picture/dingding.png width="200px">

#### 微信群
<img src=picture/weixin.png width="200px">
