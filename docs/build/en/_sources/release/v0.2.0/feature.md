
## Overview
RTP-LLM First Release Version:0.2.0(2025.09)
## Features
### Framkework  Advanced Feature
* [PD Disaggregation](../../backend/pd_disaggregation.ipynb) && [PD Entrance Transpose](../../backend/pd_entrance_transpose.md)
* [Attention Support more Backend](../../backend/attention_backend.md): XQA, FlashInfer
* [Speculative Decoding](../../backend/speculative_decoding.md)
* [EPLB](../../references/deepseek/reporter.md#eplb)
* [MicroBatch & Overlapping](../../references/deepseek/reporter.md#microbatch-overlapping)
* [MTP](../../references/deepseek/reporter.md#mtp)
* [DeepEP](../../references/deepseek/reporter.md#deepep-network)
* [LoadBalance](../../backend/flexlb.md)
* [3FS](../../backend/3fs.md)
* [FP8 KVCache](../../backend/KvCache.md)
* [REUSE KV CACHE](../../backend/reuse_kv_cache.md)
* [Quantization](../../backend/quantization.md)
* [MultiLoRA](../../backend/lora.ipynb)
* [Attention FFN Disaggregation](../../backend/af_disaggregation.md)
* [Frontend/Backend Disaggregation](../../backend/Frontend.md)


### New Models
| **Model Family (Variants)** | **Example HuggingFace Identifier**  | **Description** | **Support CardType** |
|-----------------------------|-------------------------------------|-----------------|------------------|
| **DeepSeek** (v1, v2, v3/R1)| `deepseek-ai/DeepSeek-R1`  | Series of advanced reasoning-optimized models (including a 671B MoE) trained with reinforcement learning; <br>top performance on complex reasoning, math, and code tasks.<br> [RTP-LLM provides Deepseek v3/R1 model-specific optimizations](../../references/deepseek/reporter.md)| NV ✅<br> AMD ✅|
| **Kimi** (Kimi-K2) | `moonshotai/Kimi-K2-Instruct`  | Moonshot's MoE LLMs with 1 trillion parameters, exceptional on agentic intellegence| NV ✅<br> AMD ✅|
| **Qwen** (v1, v1.5, v2, v2.5, v3, QWQ, Qwen3-Coder)| `Qwen/Qwen3-235B-A22B`  | Series of advanced reasoning-optimized models, <br>Significantly improved performance on reasoning tasks,<br> including logical reasoning, mathematics, science, coding, and academic benchmarks that typically require human expertise — achieving state-of-the-art results among open-source thinking models.<br>Markedly better general capabilities, such as instruction following, tool usage, text generation, and alignment with human preferences.<br>Enhanced 256K long-context understanding capabilities.| NV ✅<br> AMD ✅|
| **QwenVL** (VL2, VL2.5, VL3)| `Qwen/Qwen2-VL-2B`  | Series of advanced  Vision-language model series based on Qwen2.5/Qwen3| NV ✅<br> AMD ❌|
| **Llama**         | `meta-llama/Llama-4-Scout-17B-16E-Instruct`  | Meta’s open LLM series, spanning 7B to 400B parameters (Llama 2, 3, and new Llama 4) with well-recognized performance.  | NV ✅<br> AMD ✅ |

### Bug Fixs

## Question of omission
* PD Entrance Transpose not worker with front app
* metrics of 3fs cache hit ratio is not accurate
* too many dynamic lora need more **reserver_runtime_mem_mb**
* AMD not support MoE models
* MoE model without shared_experter cannot use enable-layer-micro-batch


## Performance

## Compatibility
