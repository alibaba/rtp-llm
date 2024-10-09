# 环境变量配置

## 常用选项
| 环境变量名 | 类型 | 说明 |
| --- | --- | --- |
| `TOKENIZER_PATH` | `str`, required | tokenizer路径  |
| `CHECKPOINT_PATH` | `str`, required | checkpoint路径 |
| `MODEL_TYPE` | `str`, required | 模型类型 |
| `MAX_SEQ_LEN` | `str`, optional | 输入+输出最大长度 |
| `WEIGHT_TYPE` | `str`, optional | 模型加载使用的weight 类型:FP16/INT8 |
| `CONCURRENCY_LIMIT` | `str`, optional | 模型最大并发数 |


* `TOKENIZER_PATH` 和 `CHECKPOINT_PATH` 必须为本地路径。
* `MODEL_TYPE` 目前支持 `chatglm``chat_glm``chatglm2``chat_glm_2``chatglm3``chat_glm_3``glm_130b``gpt_bigcode``starcoder2``wizardcoder``sgpt_bloom``sgpt_bloom_vector``bloom``llama``gemma``xverse``llava``baichuan``gpt_neox``qwen_7b``qwen_13b``qwen_1b8``qwen_2``qwen_vl``falcon``mpt``internlm``phi``aquila``cogvlm2``chatglm4v`

## 高级选项
| 环境变量名 | 类型 | 说明 |
| --- | --- | --- |
| `INT8_KV_CACHE` | `str`, optional | 高级选项:kv cache 使用int8类型,可节省显存 |
| `KV_CACHE_MEM_MB` | `str`, optional | kv cache 预留显存大小，单位(MB) |
| `PRE_ALLOCATE_OP_MEM` | `str`, optional | 是否提前预分配显存,与KV_CACHE_MEM_MB配合使用 |
| `TP_SPLIT_EMB_AND_LMHEAD` | `str`, optional | TensorParallel时是否切分Emb和LmHead计算(1:打开，0:关闭) |
| `REUSE_CACHE` | `str`, optional | query之间复用kvcache |
| `EXTRA_DATA_PATH` | `str`, optional | 除了ckpt/tokenizer,额外需要的数据,比如LLAVA的 VIT数据 |
| `VIT_TRT` | `int`, optional | 是否使用TRT来加速VIT模型(1:打开，0:关闭) |
| `FT_DISABLE_CUSTOM_AR` | `int`, optional | 是否关闭Custom All Reduce(1:关闭，其他打开) |

## 注意事项
1. 默认模型运行时的 log_level=WARNING，可以添加环境变量`LOG_LEVEL=INFO` 显示更多日志
2. 可以配置环境变量`LOAD_CKPT_NUM_PROCESS=x`多进程加载模型。多进程加载的时候需要使用`if __name__ == '_main__':`作为入口，因为默认程序会使用spawn的方式起多进程；同时进程数过多可能会导致cuda out of memory

# ModelConfig

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `model_type` | `str, default=''` | 模型类型 |
| `ckpt_path` | `str, default=''` | 模型路径 |
| `tokenizer_path` | `str, default=''` | tokenizer路径 |
| `weight_type` | `WEIGHT_TYPE, default=WEIGHT_TYPE.FP16` | 模型weights量化类型 |
| `act_type` | `WEIGHT_TYPE, default=WEIGHT_TYPE.FP16` | 模型weights存储类型 |
| `max_seq_len` | `bool, default=0` | beam search的个数 |
| `seq_size_per_block` | `int, default=8` | async模式下每个block的序列长度 |
| `gen_num_per_circle` | `int, default=1` | 每轮可能新增的token数，仅在投机采样情况下>1 |
| `ptuning_path` | `Optional[str], default=None` | ptuning ckpt的存储路径 |
| `lora_infos` | `Optional[Dict[str, str]]` | lora ckpt存储路径 |

目前我们支持的所有模型列表可以在`maga_transformer/models/__init__.py`查看，具体模型对应的`model_type`可以查看模型文件的`register_model`
