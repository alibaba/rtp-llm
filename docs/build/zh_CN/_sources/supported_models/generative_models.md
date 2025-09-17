# Large Language Models

These models accept text input and produce text output (e.g., chat completions). They are primarily large language models (LLMs), some with mixture-of-experts (MoE) architectures for scaling.

## Example launch Command

```shell
/opt/conda310/bin/python3 -m rtp_llm.start_server \
--checkpoint_path /models/qwen \
--model_type qwen_2 \
--act_type fp16 \
--start_port 8088
```

## Supported models

Below the supported models are summarized in a table.

If you are unsure if a specific architecture is implemented, you can search for it via GitHub. For example, to search for `Qwen3ForCausalLM`, use the expression:

```
repo:foundation_models/RTP-LLM- path:/^rtp_llm/models\// Qwen3ForCausalLM
```

in the GitHub search bar.

| Model Family (Variants)             | Example HuggingFace Identifier         | ModelType            | Description                                                                            |
|-------------------------------------|--------------------------------------------------|----------| ----------------------------------------------------------------------------------------|
| **DeepSeek** (v1, v2, v3/R1)        | `deepseek-ai/DeepSeek-R1`                        | deepseek_v3 |Series of advanced reasoning-optimized models (including a 671B MoE) trained with reinforcement learning; top performance on complex reasoning, math, and code tasks. [RTP-LLM provides Deepseek v3/R1 model-specific optimizations](../references/deepseek/reporter.md)|
| **DeepSeek** (v1, v2)        | `deepseek-ai/DeepSeek-V2`                        | deepseek_v2 |Series of advanced reasoning-optimized models (including a 671B MoE) trained with reinforcement learning; top performance on complex reasoning, math, and code tasks. |
| **Qwen** (3MoE, 2.5MoE, Coder)       | `Qwen/Qwen3-30B-A3B`, `Qwen/Qwen3-Coder-480B-A35B-Instruct`  | qwen_3_moe      | Alibaba’s latest Qwen3Moe series for complex reasoning, language understanding, and generation tasks; Support for MoE variants along with previous generation 3, etc. |
| **Qwen** (3 series)       | `Qwen/Qwen3-32B`  | qwen_3      | Alibaba’s latest Qwen3 series for complex reasoning, language understanding, and generation tasks; Support for dense variants along with previous generation 3,  etc. |
| **Qwen** (2.5, 2, 1.5, QWQ series)       | `Qwen/Qwen2-72B`  | qwen_2      | Alibaba’s latest Qwen2 series for complex reasoning, language understanding, and generation tasks; Support fo dense along with previous generation 2.5, 2, 1.5, etc. |
| **Qwen** (1 series)       | `Qwen/Qwen-72B`  | qwen      | Alibaba’s latest Qwen3 series for complex reasoning, language understanding, and generation tasks; Support for MoE variants along with previous generation 2.5, 2, etc. |
| **Llama** (2, 3.x, 4 series)        | `meta-llama/Llama-4-Scout-17B-16E-Instruct`  | llama      | Meta’s open LLM series, spanning 7B to 400B parameters (Llama 2, 3, and new Llama 4) with well-recognized performance.  |
| **Mistral** (Mixtral, NeMo, Small3) | `mistralai/Mistral-7B-Instruct-v0.2`     | mistral         | Open 7B LLM by Mistral AI with strong performance; extended into MoE (“Mixtral”) and NeMo Megatron variants for larger scale. |
| **Gemma** (v1, v2, v3)              | `google/gemma-3-1b-it`             | gemma                | Google’s family of efficient multilingual models (1B–27B); Gemma 3 offers a 128K context window, and its larger (4B+) variants support vision input. |
| **Phi** (Phi-1.5, Phi-2, Phi-3, Phi-4, Phi-MoE series) | `microsoft/Phi-4-multimodal-instruct`, `microsoft/Phi-3.5-MoE-instruct`  | phi| Microsoft’s Phi family of small models (1.3B–5.6B); Phi-4-multimodal (5.6B) processes text, images, and speech, Phi-4-mini is a high-accuracy text model and Phi-3.5-MoE is a mixture-of-experts model. |
| **DBRX** (Databricks)              | `databricks/dbrx-instruct`             | Dbrx           | Databricks’ 132B-parameter MoE model (36B active) trained on 12T tokens; competes with GPT-3.5 quality as a fully open foundation model. |
| **ChatGLM2**       | `zai-org/chatglm2-6b`                   | chat_glm_2            | Zhipu AI’s bilingual chat model (6B) excelling at Chinese-English dialogue; fine-tuned for conversational quality and alignment. |
| **ChatGLM3**       | `zai-org/chatglm3-6b`                   | chat_glm_3            | Zhipu AI’s bilingual chat model (6B) excelling at Chinese-English dialogue; fine-tuned for conversational quality and alignment. |
| **GLM4**       | `zai-org/glm-4-9b-hf`                   | chat_glm_4            | Zhipu AI’s bilingual chat model (6B) excelling at Chinese-English dialogue; fine-tuned for conversational quality and alignment. |
| **InternLM 2** (7B, 20B)           | `internlm/internlm2-7b`                | internlm2           | Next-gen InternLM (7B and 20B) from SenseTime, offering strong reasoning and ultra-long context support (up to 200K tokens). |
| **Baichuan 2** (7B, 13B)           | `baichuan-inc/Baichuan2-13B-Chat`      | baichuan2           | BaichuanAI’s second-generation Chinese-English LLM (7B/13B) with improved performance and an open commercial license. |
| **XVERSE**                    | `xverse/XVERSE-13B`               | llama           | Yuanxiang’s open LLM supporting ~40 languages; delivers 100B+ dense-level performance via expert routing. |
