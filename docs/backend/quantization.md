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
For inference, the packed dense and expert kernels accept FP16 or BF16
activations on CUDA SM80+ or ROCm gfx942/gfx950. A supported MoE router is
also required. On V100 (SM70) and T4 (SM75), the offline converter can write
the corresponding CUDA weight layout, but there is no AWQ/GPTQ inference
kernel. The CUDA 12 build target alone does not provide one.

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

For dense Llama/Qwen HF feed-forward layers with tensor parallelism (TP),
the intermediate width need not be a multiple of `128 × TP`. The loader
pads the gate, up, and down projections together **before** splitting them
across ranks, so the 128-channel scale blocks keep their checkpoint positions.
This does not relax the block-alignment requirements on TP-sharded Q/K/V.

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
For dense projections, the loader stages floating checkpoint weights in the
configured compute dtype (FP16 or BF16, not FP32) and quantizes to INT8 with one
scale per output channel. SM8.x uses PyTorch's CUDA mixed-dtype linear
operator. On SM90 and SM100 the loader checks that the installed Triton compiler
can run the packed-weight kernel before publishing the model. Other devices
(including SM120) are not enabled for this dense path; CPU loading is for
staging only. INT8 expert projections use the grouped integer kernel on
CUDA SM80+ or ROCm gfx942/gfx950, subject to router support. A model that
also quantizes dense layers must meet the dense requirements above.
On V100/T4, INT8 weights can be converted offline, but neither dense nor
expert W8A16 inference is available.

Example:
```
python3 -m rtp_llm.start_server --checkpoint_path XXXX  --model_type qwen_3  --quantization int8
```
