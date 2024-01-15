## 项目介绍

* rtp-llm 是阿里巴巴大模型预测团队开发的大模型推理加速引擎，也是阿里巴巴内部广泛使用的大规模分布式大模型推理引擎，支持了包括淘宝、天猫、菜鸟、高德、饿了么、AE、Lazada 等多个部门的大模型推理业务。
* 为用户提供高性能、低成本、易用的推理服务，帮助客户和开发者量身定做适合自身业务的推理服务，助力业务增长。
* rtp-llm项目是[havenask](https://github.com/alibaba/havenask)项目的子项目。

## 核心能力
### 高性能
* 使用高性能的 cuda kernel。
* 框架上对动态凑批的 overhead 进行了细致优化。
* 支持 paged attention 和 kv cache 量化。
* 支持 flash attention2。
* 支持 weight only INT8 量化。支持加载时自动量化。
* 对 V100 进行了特别优化。

### 非常灵活并且易于使用
* 和流行的HuggingFace模型无缝对接，支持多种权重格式。无需额外转换流程。
* 支持单模型实例同时部署多 LoRA 服务。
* 支持多模态(图片和文本混合输入)
* 支持多机/多卡 tensor 并行。
* 支持加载 P-tuning 模型。

### 高级推理加速方法
* 支持剪枝后的不规则模型加载
* 支持多轮对话上下文 Cache。
* 支持 Speculative Decoding 加速。
* 支持 Medusa 加速。

## 使用方法
### 需求
* 操作系统: Linux
* Python: 3.10
* NVIDIA GPU: Compute Capability 7.0 或者更高 (例如V100, T4, RTX20xx, A100, L4, H100等)

### 安装和启动
```bash
cd rtp-llm
pip3 install -r ./maga_transformer/requirements_torch_gpu.txt
# 使用release版本中对应的whl, 这里以0.1.0版本为例子
wget https://github.com/alibaba/rtp-llm/releases/download/v0.1.0/maga_transformer-0.1.0-py3-none-any.whl
pip3 install maga_transformer-0.1.0-py3-none-any.whl
export TOKENIZER_PATH=/path/to/tokenizer
export CHECKPOINT_PATH=/path/to/model
export FT_SERVER_TEST=1
python3 -m maga_transformer.start_server

# request to server
curl -XPOST http://localhost:8088 -d '{"prompt": "hello, what is your name", "generate_config: {"max_new_tokens": 1000}}'
```

## 文档
* [配置参数](docs/Config.md)
* [请求格式](docs/Request.md)
* [OpenAI接口](docs/OpenAI-Tutorial.md)
* [LoRA](docs/LoRA-Tutorial.md)
* [PTuning](docs/PTuning-Tutorial.md)
* [多模态](docs/Multimodal-Tutorial.md)
* [结构化剪枝](docs/Sparse-Tutorial.md)
* [投机采样](docs/SpeculativeDecoding-Tutroial.md)
* [作为 Python 库引用](docs/HF.md)
* [源码构建](docs/Build.md)
* [Roadmap](docs/Roadmap.md)
* [Contributing](docs/Contributing.md)

## 致谢
我们的项目主要基于[FasterTransformer](https://github.com/NVIDIA/FasterTransformer)，并在此基础上集成了[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)的部分kernel实现。FasterTransformer和TensorRT-LLM为我们提供了可靠的性能保障。[Flash-Attention2](https://github.com/Dao-AILab/flash-attention)和[cutlass](https://github.com/NVIDIA/cutlass)也在我们持续的性能优化过程中提供了大量帮助。我们的continuous batching和increment decoding参考了[vllm](https://github.com/vllm-project/vllm)的实现；采样参考了[transformers](https://github.com/huggingface/transformers)，投机采样部分集成了[Medusa](https://github.com/FasterDecoding/Medusa)的实现，多模态部分集成了[llava](https://github.com/haotian-liu/LLaVA)和[qwen-vl](https://github.com/QwenLM/Qwen-VL)的实现。感谢这些项目对我们的启发和帮助。

## 对外应用场景（持续更新）
* 淘宝问问
* 阿里巴巴国际AI平台[Aidge](https://aidc-ai.com/)
* [OpenSearch LLM智能问答版](https://www.aliyun.com/activity/bigdata/opensearch/llmsearch)
* [Large Language Model based Long-tail Query Rewriting in Taobao Search](https://arxiv.org/abs/2311.03758)

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
<img src=picture/weixin.JPG width="200px">