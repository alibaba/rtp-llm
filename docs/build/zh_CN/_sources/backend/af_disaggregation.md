# Attention FFN Disaggregation (AFD)

Attention FFN Disaggregation is a technique that separates the attention and feed-forward network (FFN) computations in transformer models to optimize performance. Currently, this feature only supports Qwen3 dense models (Qwen3-32B and Qwen3-8B), with support for MOE versions planned for future releases.

## Example launch Command

```shell
LOAD_PYTHON_MODEL=1 /opt/conda310/bin/python3 -m rtp_llm.start_server \
--checkpoint_path Qwen/Qwen3-32B \
--model_type qwen_3 \
--enable_ffn_disaggregate 1 \
--enable_layer_micro_batch 1 \
--world_size 2 \
--dp_size 2 \
--reserver_runtime_mem_mb 8000 \
--device_reserve_memory_bytes -8192000000 \
--start_port 8088
```

## Supported models

Below the supported models are summarized in a table.

If you are unsure if a specific architecture is implemented, you can search for it via GitHub. For example, to search for `Qwen3ForCausalLM`, use the expression:

```
repo:foundation_models/RTP-LLM- path:/^rtp_llm\/models\// Qwen3ForCausalLM
```

in the GitHub search bar.

| Model Family (Variants)             | Example HuggingFace Identifier         | ModelType            | Description                                                                            |
|-------------------------------------|--------------------------------------------------|----------| ----------------------------------------------------------------------------------------|
| **Qwen** (3 series)       | `Qwen/Qwen3-32B`, `Qwen/Qwen3-8B`  | qwen_3      | Alibaba's latest Qwen3 series for complex reasoning, language understanding, and generation tasks; Currently supports dense variants (Qwen3-32B and Qwen3-8B) with AFD. MOE versions will be supported in future releases. |