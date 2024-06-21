# 背景
rtp-llm当前支持weight only量化，包含int8和int4；可以显著减少显存占用，并加速decode阶段。
已知问题：Weight Only量化在Prefill阶段，长sequence时可能会导致性能下降
当前所有量化方式在SM70及以上支持

## Weight Only Int8量化
设置环境变量： INT8_MODE=1 或 WEIGHT_TYPE=int8 即可
Weight Only Int8量化仅支持SM70及以上。

## GPTQ和AWQ
不需要设置环境。
支持int4和int8。
模型config需要包含量化相关config，包含bits, group_size, quant_method
GPTQ config示例：
``` json
"quantization_config": {
    "bits": 4,
    "group_size": 128,
    "quant_method": "gptq"
}
```
AWQ config示例：
``` json
"quantization_config": {
    "bits": 4,
    "group_size": 128,
    "quant_method": "awq"
}
```

## W8A8
支持smoothquant和omniquant
需要在ckpt路径下包含一个名为“smoothquant.ini”的文件，或者写config
``` json
"quantization_config": {
    "bits": 8,
    "quant_method": "omni_quant"
}
```
支持llama，qwen，starcoder；保存在ckpt中的tensor name可参考相关模型文件。