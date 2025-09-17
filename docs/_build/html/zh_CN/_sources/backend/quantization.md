# Quantization
RTP-LLM currently supports weight only quantization, including int8 and int4. It can significantly reduce the video memory footprint and accelerate the decoding phase.
Known issues: Weight Only quantization may cause performance degradation for long sequences during the Prefill phase
Currently, all quantization methods are supported in SM70 and above

## Support Quant

| **CardType** | **Int8WeightOnly** | **Int8W8A8** | **BlockWiseFp8** | **PerTensorFp8** |**INT4** | **PTPC** |
|--------------|--------------------|--------------|------------------|------------------|---------|----------|
| **CUDA**     | ✅                 | ✅            | ✅               | ✅               |✅        | ❌       |
| **AMD**      | ❌                 | ✅            | ✅               | ✅               |✅        | ✅       |

## GPTQ/AWQ
Supports int4 and int8. Model weights needs to be quantified in advance(use AutoGPTQForCausalLM/AutoAWQForCausalLM).<br>
The model config needs to contain quantization related config, containing bits, group_size, quant_method.<br>
GPTQ config example:
``` json
"quantization_config": {
"bits": 4,
"group_size": 128,
"quant_method": "gptq"
}
```
Example AWQ config:
``` json
"quantization_config": {
"bits": 4,
"group_size": 128,
"quant_method": "awq"
}
```

## W8A8
smoothquant and omniquant are supported
You need to include a file called "smoothquant.ini" under the ckpt path, or write config

``` json
"quantization_config": {
    "bits": 8,
    "quant_method": "omni_quant"
}
```
Supports llama, qwen, starcoder. The name of the tensor stored in ckpt is referred to the associated model file.


## BlockWiseFp8
Support Load Quant or PreQuantified.<br>
You can use Load Quant by set args, Example<br>
```
python3 -m rtp_llm.start_server --checkpoint_path XXXX  --model_type qwen_3  --quantization fp8_per_block
```

You can Provide PreQuantified Model Weight, The model config needs to contain quantization related config<br>


```
"quantization_config": {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [
        128,
        128
    ]
}
```

## PerTensorFp8
Support Load Quant or PreQuantified by TRT-LLM/TransformerEngine.<br>
You can use Load Quant by set args, Example<br>
```
python3 -m rtp_llm.start_server --checkpoint_path XXXX  --model_type qwen_3  --quantization fp8
```

You can Provide PreQuantified Model Weight, The model config needs to contain quantization related config<br>
```
"quantization_config": {
    "quant_method": "FP8",
    "bits": 8
}
```


## Int8WeightOnly
Support Load Quant.You can use Load Quant by set args, Example<br>
```
python3 -m rtp_llm.start_server --checkpoint_path XXXX  --model_type qwen_3  --quantization int8
```
