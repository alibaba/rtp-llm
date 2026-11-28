# Text generation models

Architecture detection and checkpoint loading are separate. The service can recognize an old model name even when it has no direct checkpoint owner. To see which model types this installation can load, run this inside the serving environment:

```shell
python3 -c 'from rtp_llm.models_py.registry import list_models; print("\n".join(list_models()))'
```

Registration is a first check, not a guarantee for every quantization format, device or parallel configuration.

## Example: Qwen3 dense

```shell
python3 -m rtp_llm.start_server \
  --checkpoint_path /models/Qwen3-8B \
  --model_type qwen_3 \
  --act_type bf16 \
  --start_port 8088
```

| Checkpoint family | Example model type | What to check |
| --- | --- | --- |
| Qwen3 dense / MoE | `qwen_3` / `qwen_3_moe` | Use the MoE owner only for MoE checkpoints. |
| Qwen2 / Qwen2.5 dense | `qwen_2` | Qwen2-MoE is a different architecture. |
| Original Qwen | `qwen` | Its projection names are different. Do not assume that Qwen2/3 prequantized formats work with original Qwen. |
| DeepSeek V2 / V3 | `deepseek2` / `deepseek3` | Use the model type for the corresponding attention and expert layout. See the [DeepSeek guide](../references/deepseek/reporter.md). |
| Llama | `llama` | For compatible bias-free decoders; not a generic owner for all models descended from Llama. |

## Historical model names

Earlier releases documented service registrations including Mistral/Mixtral, Gemma, Phi, DBRX, ChatGLM, InternLM, Baichuan, Bloom, Falcon, MPT, GPT-NeoX and StarCoder. Recognition by the configuration or tokenizer does **not** supply their checkpoint implementation. Consult the installed registry before deployment; using another architecture's name does not convert the weights.
