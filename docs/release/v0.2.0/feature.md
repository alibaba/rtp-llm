
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
* P/D Disaggregation dead lock casuse by request cancel/failed before remote running
* Raw Request stream stop_words cause fake hang
* some speculative decoding bugs
* Warmup produce nan maybe influence kvcache
* Not success query make bad kvcache case wrong answer
* UseAllGather takes effect automatically according to the DP/TP 
* UseAllGather with deepgemm coredump cause by topk type is bad.
* FlexLb too many log cause bad performance
* Flexlb support PD_FUSION

## Question of omission
* In 3fs Case need more MEM or set FRONTEND_SERVER_COUNT=1 to reduce frontend_server mem usage in P/D when Use Frontend Disaggregation.
* too many dynamic lora need more reserver_runtime_mem_mb
* AMD not support MoE models
* MoE model without shared_experter cannot use enable-layer-micro-batch
* P/D Disaggregation with EPLB and MTP step > 1 may cause Prefill Hang
* Embedding of VL Model is not ok cause by position id is wrong
* FlexLb: Frequent switching of a large number of machines results in the performance degradation of flexlb


## Performance

## Compatibility
