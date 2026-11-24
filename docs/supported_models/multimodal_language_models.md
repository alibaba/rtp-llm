# Multimodal language models

A multimodal service needs both a text checkpoint owner and a compatible vision owner. This also applies when the vision process runs on a separate worker.

## Example: Qwen2.5-VL on CUDA

```shell
python3 -m rtp_llm.start_server \
  --checkpoint_path /models/Qwen2.5-VL-7B-Instruct \
  --model_type qwen2_5_vl \
  --act_type bf16 \
  --start_port 8088
```

| Checkpoint family | Model type |
| --- | --- |
| Qwen2-VL / Qwen2.5-VL | `qwen2_vl` / `qwen2_5_vl` |
| Qwen3-VL dense / MoE | `qwen3_vl` / `qwen3_vl_moe` |
| DeepSeek-VL2 Tiny | `deepseek_vl_v2` (Tiny's separate Q/K/V layout; not the Small/Full MLA variants) |

**ROCm:** Qwen2-VL and Qwen2.5-VL checkpoints use non-interleaved MRoPE by default. The ROCm fused attention path does not implement that layout. Use CUDA; setting `mrope_interleaved=true` would change the model's semantics.

Historical registrations such as original Qwen-VL (`qwen_vl`), Qwen2-Audio, LLaVA and ChatGLM4V do not provide a direct vision checkpoint owner. Do not use those names as aliases for the Qwen2/3-VL owners. See [text generation models](generative_models.md) for how to query the installed language owners.
