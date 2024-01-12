## 项目介绍

* rtp-llm 是阿里巴巴大模型预测团队开发的大模型推理加速引擎，也是阿里巴巴内部广泛使用的大规模分布式大模型推理引擎，支持了包括淘宝、天猫、菜鸟、高德、饿了么、AE、Lazada 等多个部门的大模型推理业务。
* 为用户提供高性能、低成本、易用的推理服务，帮助客户和开发者量身定做适合自身业务的推理服务，助力业务增长。
* rtp-llm项目是[havenask](https://github.com/alibaba/havenask)项目的子项目。

## 核心能力
### rtp-llm 的高性能体现在
* 使用优化过的高性能的 cuda 内核。
* 框架上对动态凑批的 overhead 进行了细致优化。
* 支持 kv cache 优化管理, 支持 page attention。
* 支持flash attention2，trt flash attention(仅在cuda12以上enable)。
* 支持multi_blcok_mode(类似flash decoding)。
* 支持weight only INT8 量化。支持加载时自动量化。
* 对 V100 进行了特别优化。

### rtp-llm 非常灵活并且易于使用
* 和流行的HuggingFace模型无缝对接，支持多种权重格式。无需额外转换流程。
* 高吞吐，并且有多种decode算法：包括并行采样，beam search等等。
* 支持单模型实例同时部署多 LoRA 服务。
* 支持加载 P-tuning 模型。
* 支持多机/多卡 tensor 并行。
* 支持流式输出。
* 支持多种Nvidia gpus卡类型。
* 支持多模态(图片和文本混合输入)

### rtp-llm 支持的高级推理加速方法
* 支持剪枝后的不规则模型加载
* 支持多轮对话上下文 Cache。
* 支持 Speculative Decoding 加速。
* 支持 Medusa 加速。

### rtp-llm 无缝的支持以下Hugging Face模型（持续更新中）。

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

### rtp-llm支持的多模态模型：
* LLAVA (liuhaotian/llava-v1.5-13b, liuhaotian/llava-v1.5-7b)
* Qwen-VL (Qwen/Qwen-VL)

## 安装：
rtp-llm 是一个python库，包含了编译好的c++库，cuda库以及python库等。

### 需求：
* 操作系统: Linux
* Python: 3.10
* NVIDIA GPU: Compute Capability 7.0 或者更高 (例如V100, T4, RTX20xx, A100, L4, H100等)

### 进入docker
```
$ git clone https://github.com/alibaba/rtp-llm.git
$ cd docker
$ python3 ./create_container.py create <CONTAINER_NAME> --gpu
$ python3 ./create_container.py enter <CONTAINER_NAME> --gpu
```

### 使用pip来安装
您可以使用pip来安装rtp-llm：
```
$ # 安装rtp-llm（cuda = 11.4）
$ cd rtp-llm
$ pip3 install -r ./maga_transformer/requirements_torch_gpu.txt
$ # 使用release版本中对应的whl, 这里以0.1.0版本为例子
$ wget https://github.com/alibaba/rtp-llm/releases/download/v0.1.0/maga_transformer-0.1.0-py3-none-any.whl
$ pip3 install maga_transformer-0.1.0-py3-none-any.whl
$ # 修改test.py中的模型路径
$ python3 example/test.py
```

### 从源代码开始构建
您也可以通过源代码来进行编译。源码构建使用bazel作为构建系统，推荐版本`5.2.0`。
```
$ cd rtp-llm
$ pip3 install -r ./maga_transformer/requirements_torch_gpu.txt
$ bazel build //maga_transformer:maga_transformer --jobs 100 --verbose_failures
$ # 修改test.py中的模型路径，运行一个实际的模型
$ bazel test //example:test --jobs 100
$ # 单元测试
$ bazel test //maga_transformer/test/model_test/fake_test:all_fake_model_test --jobs 100  --test_output=all
```

## 大模型预测
当前rtp-llm支持两种加载模型方式：
### 读取Huggingface模型
huggingface模型支持从通过模型名从远程下载模型，代码如下：(如果无法访问huggingface需要配置环境变量`HF_ENDPOINT`)
``` python
from maga_transformer.pipeline import Pipeline
from maga_transformer.model_factory import ModelFactory

if __name__ == '__main__':
    model = ModelFactory.from_huggingface("Qwen/Qwen-7B-Chat")
    pipeline = Pipeline(model, model.tokenizer)
    for res in pipeline(["hello, what's your name"], max_new_tokens = 100):
        print(res.batch_response)
    pipeline.stop()

```
也支持通过模型路径加载
``` python
model = ModelFactory.from_huggingface("/path/to/dir")
```
构建模型时默认使用基础配置参数，也可以通过构建`ModelConfig`自行修改配置，`ModelConfig`参数的介绍在下一节
``` python
from maga_transformer.utils.util import WEIGHT_TYPE

model_config = ModelConfig(
    async_mode=True,
    weight_type=WEIGHT_TYPE.INT8,
    max_seq_len=2000,
    ...
)
```

如果存在框架无法推断出模型类型，但是已经适配实现的case，可以自行指定模型类型
``` python
model_config = ModelConfig(
    model_type='chatglm',
    ckpt_path='/path/to/ckpt',
    tokenizer_path='/path/to/tokenizer',
    async_mode=True,
    weight_type=WEIGHT_TYPE.INT8,
    max_seq_len=2000,
    ...
)
```
### ModelConfig

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `model_type` | `str, default=''` | 模型类型 |
| `ckpt_path` | `str, default=''` | 模型路径 |
| `tokenizer_path` | `str, default=''` | tokenizer路径 |
| `async_mode` | `bool, default=False`| 是否开启异步凑批模型 |
| `weight_type` | `WEIGHT_TYPE, default=WEIGHT_TYPE.FP16` | 模型weights量化类型 |
| `act_type` | `WEIGHT_TYPE, default=WEIGHT_TYPE.FP16` | 模型weights存储类型 |
| `max_seq_len` | `bool, default=0` | beam search的个数 |
| `seq_size_per_block` | `int, default=8` | async模式下每个block的序列长度 |
| `gen_num_per_circle` | `int, default=1` | 每轮可能新增的token数，仅在投机采样情况下>1 |
| `ptuning_path` | `Optional[str], default=None` | ptuning ckpt的存储路径 |
| `lora_infos` | `Optional[Dict[str, str]]` | lora ckpt存储路径 |

目前我们支持的所有模型列表可以在`maga_transformer/models/__init__.py`查看，具体模型对应的`model_type`可以查看模型文件的`register_model`

## 大模型服务
rtp-llm使用fastapi构建了高性能模型服务，使用异步编程尽量避免cpu线程压力过大干扰gpu高效运行
### 服务Demo
``` shell
export MODEL_TYPE=model_type
export TOKENIZER_PATH=/path/to/tokenizer
export CHECKPOINT_PATH=/path/to/model
export FT_SERVER_TEST=1
export START_PORT=12345

python3 -m maga_transformer.start_server

# request to server
curl -XPOST http://localhost:12345 -d '{"prompt": "hello, what is your name", "generate_config: {"max_new_tokens": 1000}}'
```
默认服务的log会写到启动路径的logs文件夹下，可以添加环境变量`FT_SERVER_TEST=1`把日志写到shell

### 配置的含义

#### 环境变量说明：

常用选项：
| 环境变量名 | 类型 | 说明 |
| --- | --- | --- |
| `TOKENIZER_PATH` | `str`, required | tokenizer路径  |
| `CHECKPOINT_PATH` | `str`, required | checkpoint路径 |
| `MODEL_TYPE` | `str`, required | 模型类型 |
| `MAX_SEQ_LEN` | `str`, optional | 输入+输出最大长度 |
| `WEIGHT_TYPE` | `str`, optional | 模型加载使用的weight 类型:FP16/INT8 |
| `ASYNC_MODE` | `str`, optional | 异步模式（1:打开，0:关闭），支持dynamic batching、paged (token) attention等优化 |
| `CONCURRENCY_LIMIT` | `str`, optional | 模型最大并发数 |

* TOKENIZER_PATH和CHECKPOINT_PATH必须为本地路径。
* 当前支持的模型类型为：
* chatglm/chat_glm/chatglm2/chat_glm_2/chatglm3/chat_glm_3/glm_130b/gpt_bigcode/wizardcoder/sgpt_bloom/sgpt_bloom_vector/
* bloom/llama/xverse/llava/baichuan/gpt_neox/
* qwen_7b/qwen_13b/qwen_1b8/qwen_vl/falcon/mpt/internlm/phi/aquila

高级选项：
| 环境变量名 | 类型 | 说明 |
| --- | --- | --- |
| `INT8_KV_CACHE` | `str`, optional | 高级选项:kv cache 使用int8类型,可节省显存 |
| `KV_CACHE_MEM_MB` | `str`, optional | kv cache 预留显存大小，单位(MB) |
| `PRE_ALLOCATE_OP_MEM` | `str`, optional | 是否提前预分配显存,与KV_CACHE_MEM_MB配合使用 |
| `TP_SPLIT_EMB_AND_LMHEAD` | `str`, optional | TensorParallel时是否切分Emb和LmHead计算(1:打开，0:关闭) |
| `USE_BLOCK_CACHE` | `str`, optional | query之间复用kvcache |
| `EXTRA_DATA_PATH` | `str`, optional | 除了ckpt/tokenizer,额外需要的数据,比如LLAVA的 VIT数据 |

#### requests的组成说明：

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `prompt` / `prompt_batch` (二选一) | `str` / `List[str]`, required | prompt |
| `generate_config` | `dict`, optional, default=`{}` | 生成参数，目前支持如下参数 |
| `using_hf_sampling` | `bool`, optional, default=`False` | 是否使用hf采样 |


#### generate_config参数说明：

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `max_new_tokens` | `int` | 最大生成token数 |
| `top_k` | `int` | top_k采样 |
| `top_p` | `float` | top_p采样 |
| `temperature` | `float` | logits计算参数(温度) |
| `repetition_penalty` | `float` | 重复token惩罚系数 |
| `random_seed` | `long` | 随机种子 |
| `num_beams` | `bool` | beam search的个数 |
| `calculate_loss` | `bool` | 是否计算loss |
| `return_hidden_states`/`output_hidden_states` | `bool` | 是否返回hidden_states |
| `return_logits`/`output_logits` | `bool` | 是否返回logits |
| `yield_generator` | `bool` | 是否流式输出 |

#### 使用 openai api 接口访问

rtp-llm同时提供了openai风格服务接口，详见[OpenAI接口使用文档](docs/OpenAI-Tutorial.md)。

## 注意事项
1. 默认模型运行时的log_level=WARNING，可以添加环境变量`PY_LOG_LEVEL=INFO` 显示更多日志
2. 可以配置环境变量`LOAD_CKPT_NUM_PROCESS=x`多进程加载模型。多进程加载的时候需要使用`if __name__ == '_main__':`作为入口，因为默认程序会使用spawn的方式起多进程；同时进程数过多可能会导致cuda out of memory

## 相关文档
* [LoRA使用文档](docs/LoRA-Tutorial.md)
* [PTuning使用文档](docs/PTuning-Tutorial.md)
* [多模态使用文档](docs/Multimodal-Tutorial.md)
* [结构化剪枝使用文档](docs/Sparse-Tutorial.md)
* [投机采样使用文档](docs/SpeculativeDecoding-Tutroial.md)

## 致谢：
我们的项目主要基于[FasterTransformer](https://github.com/NVIDIA/FasterTransformer)，并在此基础上集成了[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)的部分kernel实现。FasterTransformer和TensorRT-LLM为我们提供了可靠的性能保障。[Flash-Attention2](https://github.com/Dao-AILab/flash-attention)和[cutlass](https://github.com/NVIDIA/cutlass)也在我们持续的性能优化过程中提供了大量帮助。我们的continuous batching和increment decoding参考了[vllm](https://github.com/vllm-project/vllm)的实现；采样参考了[hf transformers](https://github.com/huggingface/transformers)，投机采样部分集成了[Medusa](https://github.com/FasterDecoding/Medusa)的实现，多模态部分集成了[llava](https://github.com/haotian-liu/LLaVA)和[qwen-vl](https://github.com/QwenLM/Qwen-VL)的实现。感谢这些项目对我们的启发和帮助。

## 对外应用场景（持续更新）
* 淘宝问问
* 阿里巴巴国际AI平台[Aidge](https://aidc-ai.com/)
* [OpenSearch LLM智能问答版](https://www.aliyun.com/activity/bigdata/opensearch/llmsearch)
* [Large Language Model based Long-tail Query Rewriting in Taobao Search](https://arxiv.org/abs/2311.03758)

## 联系我们
#### 钉钉群
<img src=picture/dingding.png width="200px">

#### 微信群
<img src=picture/weixin.JPG width="200px">