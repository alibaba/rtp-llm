# Environment Variable Configuration

## Common Options
| Environment Variable Name | Type | Description |
| --- | --- | --- |
| `TOKENIZER_PATH` | `str`, required | Tokenizer path |
| `CHECKPOINT_PATH` | `str`, required | Checkpoint path |
| `MODEL_TYPE` | `str`, required | Model type |
| `MAX_SEQ_LEN` | `str`, optional | Maximum input+output length |
| `WEIGHT_TYPE` | `str`, optional | Weight type used for model loading: FP16/INT8 |
| `CONCURRENCY_LIMIT` | `str`, optional | Maximum concurrency for the model |

* `TOKENIZER_PATH` and `CHECKPOINT_PATH` must be local paths.
* `MODEL_TYPE` currently supports `chatglm``chat_glm``chatglm2``chat_glm_2``chatglm3``chat_glm_3``glm_130b``gpt_bigcode``starcoder2``wizardcoder``sgpt_bloom``sgpt_bloom_vector``bloom``llama``gemma``xverse``llava``baichuan``gpt_neox``qwen_7b``qwen_13b``qwen_1b8``qwen_2``qwen_vl``falcon``mpt``internlm``phi``aquila``chatglm4v`

## Advanced Options
| Environment Variable Name | Type | Description |
| --- | --- | --- |
| `INT8_KV_CACHE` | `str`, optional | Advanced option: Use int8 type for kv cache to save GPU memory |
| `KV_CACHE_MEM_MB` | `str`, optional | Reserved GPU memory size for kv cache, unit (MB) |
| `PRE_ALLOCATE_OP_MEM` | `str`, optional | Whether to pre-allocate GPU memory, used in conjunction with KV_CACHE_MEM_MB |
| `TP_SPLIT_EMB_AND_LMHEAD` | `str`, optional | Whether to split Emb and LmHead computation during TensorParallel (1: enable, 0: disable) |
| `REUSE_CACHE` | `str`, optional | Reuse kvcache between queries |
| `EXTRA_DATA_PATH` | `str`, optional | Additional data needed besides ckpt/tokenizer, such as LLAVA's VIT data |
| `VIT_TRT` | `int`, optional | Whether to use TRT to accelerate VIT model (1: enable, 0: disable) |
| `FT_DISABLE_CUSTOM_AR` | `int`, optional | Whether to disable Custom All Reduce (1: disable, others: enable) |

## Notes
1. The default log_level for model runtime is WARNING. You can add the environment variable `LOG_LEVEL=INFO` to display more logs.
2. You can configure the environment variable `LOAD_CKPT_NUM_PROCESS=x` to load the model with multiple processes. When loading with multiple processes, you need to use `if __name__ == '__main__':` as the entry point, because the default program will use spawn to start multiple processes; at the same time, too many processes may cause cuda out of memory.

# ModelConfig

| Parameter Name | Type | Description |
| --- | --- | --- |
| `model_type` | `str, default=''` | Model type |
| `ckpt_path` | `str, default=''` | Model path |
| `tokenizer_path` | `str, default=''` | Tokenizer path |
| `weight_type` | `WEIGHT_TYPE, default=WEIGHT_TYPE.FP16` | Model weights quantization type |
| `act_type` | `WEIGHT_TYPE, default=WEIGHT_TYPE.FP16` | Model weights storage type |
| `max_seq_len` | `bool, default=0` | Number of beam search |
| `seq_size_per_block` | `int, default=8` | Sequence length per block in async mode |
| `gen_num_per_circle` | `int, default=1` | Number of tokens that may be added per round, only >1 in speculative sampling cases |
| `ptuning_path` | `Optional[str], default=None` | Storage path for ptuning ckpt |
| `lora_infos` | `Optional[Dict[str, str]]` | Storage path for lora ckpt |

The list of all models we currently support can be viewed in `rtp_llm/models/__init__.py`. The corresponding `model_type` for specific models can be viewed in the model file's `register_model`.