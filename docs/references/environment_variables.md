# Environment variables

## Qwen3-VL vision attention

`VIT_ATTN_IMPLEMENTATION` selects the attention backend for the Hugging Face
Qwen3-VL vision encoder (including the encoder shared by Qwen3-VL MoE).
Set it before starting the service.

| Value | Behavior |
| --- | --- |
| `auto` (default) | Use FlashAttention2 when the GPU is supported and the dependency is available and importable; otherwise use SDPA. |
| `sdpa` | Use PyTorch scaled dot product attention without requiring `flash_attn`. |
| `eager` | Use the Transformers eager attention implementation. |
| `flash_attention_2` | Require FlashAttention2; fail with an explicit error if its checks fail. |

For example, `export VIT_ATTN_IMPLEMENTATION=sdpa` explicitly selects SDPA on MI308.
Without this setting, an MI308 environment missing `flash_attn` automatically
selects SDPA and logs at INFO level:

```text
Qwen3-VL ViT attention: requested=auto selected=sdpa reason=flash_attn_unavailable
```

This setting controls the Qwen3-VL vision encoder only; the language model's
attention backend and other vision encoders keep their existing configuration.
