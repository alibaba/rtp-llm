# 背景
rtp-llm当前支持weight only量化，包含int8和int4；可以显著减少显存占用，并加速decode阶段。
已知问题：Weight Only量化在Prefill阶段，长sequance时可能会导致性能下降
其中，weight only int8量化load float32/float16/bfloat16的weight，并对称量化得到int8 weight和scales；int4量化支持GPTQ和AWQ，需要load经由[AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ)/[AutoAWQ](https://github.com/casper-hansen/AutoAWQ)转化的ckpt。

## Weight Only Int8量化
设置环境变量： INT8_MODE=1 或 WEIGHT_TYPE=int8 即可
Weight Only Int8量化仅支持SM70及以上。

## Weight Only Int4量化
不需要设置环境。
模型config需要包含量化相关config，包含bits, group_size, quant_method
``` json
"quantization_config": {
    "bits": 4,
    "group_size": 128,
    "quant_method": "gptq"
}
```
Weight Only Int4量化仅支持SM80及以上。
当前在Qwen/Qwen2支持。