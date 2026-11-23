# Quantization
RTP-LLM currently supports weight only quantization, including int8 and int4. It can significantly reduce the video memory footprint and accelerate the decoding phase.
Known issues: Weight Only quantization may cause performance degradation for long sequences during the Prefill phase
CUDA architecture requirements depend on the quantization method and its runtime kernel.

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

For the Llama SmoothQuant conversion tool, supply the original architecture
metadata (e.g. `config.json`) along with `llama2.pt` and `smoothquant.ini`.
Use the converter's default `--tensor-parallelism 1` to write complete
matrices: the loader partitions them for the requested inference TP size.

Calibrated decoder INT8 is a separate format, selected by `pertensorquant.ini`.
For HF-style Llama/Qwen decoder projections, store signed INT8 `.qweight`
matrices as [input, output], plus floating-point `.scales` (one positive
step per matrix or per output channel). Each matmul needs its calibrated
input step, e.g. `self_attn.qkv_input_scale`, `self_attn.o_proj_input_scale`,
`mlp.gate_proj_input_scale` (shared by gate and up), and
`mlp.down_proj_input_scale`. Missing calibration is an error; the loader
does not substitute a guessed step. Routed experts remain floating point.
Dense, unswizzled FT export preserves the calibration and inverse for each
projection. If an older FT archive lacks the QKV input step, re-export it
from the original calibrated checkpoint.


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
For dense projections, the loader quantizes FP16/BF16 checkpoint weights to INT8
with one scale per output channel. SM8.x uses PyTorch's CUDA mixed-dtype linear
operator. On SM90 and SM100 the loader checks that the installed Triton compiler
can run the packed-weight kernel before publishing the model. Other devices
(including SM120) are not enabled for this dense path; CPU loading is for
staging only. This does not imply support for every MoE executor.

Example:
```
python3 -m rtp_llm.start_server --checkpoint_path XXXX  --model_type qwen_3  --quantization int8
```
