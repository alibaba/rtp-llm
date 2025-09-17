# Multimodal Language Models

These models accept multi-modal inputs (e.g., images/video and text) and generate text output. They augment language models with multimodal encoders.

## Example launch Command

```shell
python3 -m rtp_llm.start_server \
--checkpoint_path /mnt/nas1/hf/Qwen-VL-Chat \  # example local path
--model_type qwen_vl \
--act_type bf16 \
--port 8088 \
```

## Supported models

Below the supported models are summarized in a table.

If you are unsure if a specific architecture is implemented, you can search for it via GitHub. For example, to search for `Qwen2_5_VLForConditionalGeneration`, use the expression:

```
repo:foundation_models/RTP-LLM- path:/^rtp_llm\/models\/qwen_vl.py\// Qwen2_5_VLForConditionalGeneration
```

in the GitHub search bar.


| Model Family (Variants)    | Example HuggingFace Identifier             | Chat Template    |  Model Type|Description                                                                                                                                                                                                     |
|----------------------------|--------------------------------------------|------------------|-----------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| **Qwen3-VL** (Qwen3 series) | `Qwen/Qwen3-VL-7B-Instruct`              | `qwen3_vl_moe`        | `qwen3_vl_moe`       | Alibaba’s vision-language extension of Qwen3MoE; for example, Qwen3-VL (7B and larger variants) can analyze and converse about image content.                                                                     |
| **Qwen-VL** (Qwen2.5 series) | `Qwen/Qwen2.5-VL-7B-Instruct`              | `qwen2_5_vl`        | `qwen2_5_vl`       | Alibaba’s vision-language extension of Qwen; for example, Qwen2.5-VL (7B and larger variants) can analyze and converse about image content.                                                                     |
| **Qwen-VL** (Qwen2 series) | `Qwen/Qwen2-VL-7B-Instruct`              | `qwen2_vl`        | `qwen2_vl`       | Alibaba’s vision-language extension of Qwen; for example, Qwen2-VL (7B and larger variants) can analyze and converse about image content.                                                                     |
| **Qwen-VL** (Qwen series) | `Qwen/Qwen2-VL-7B-Instruct`              | `qwen_vl`        | `qwen_vl`       | Alibaba’s vision-language extension of Qwen; for example, Qwen-VL (7B and larger variants) can analyze and converse about image content.                                                                     |
| **DeepSeek-VL2**           | `deepseek-ai/deepseek-vl2`                 | `deepseek-vl2`    | `qwen2-vl`       | Vision-language variant of DeepSeek (with a dedicated image processor), enabling advanced multimodal reasoning on image and text inputs.                                                                        |
| **MiniCPM-V / MiniCPM-o**  | `openbmb/MiniCPM-V-2_6`                    | `minicpmv`        | `minicpmv`       | MiniCPM-V (2.6, ~8B) supports image inputs, and MiniCPM-o adds audio/video; these multimodal LLMs are optimized for end-side deployment on mobile/edge devices.                                                 |
| **Llama 3.2 Vision** (11B) | `meta-llama/Llama-3.2-11B-Vision-Instruct` | `llama_3_vision`  | `llava`       | Vision-enabled variant of Llama 3 (11B) that accepts image inputs for visual question answering and other multimodal tasks.                                                                                     |
| **LLaVA** (v1.5 & v1.6)    | *e.g.* `liuhaotian/llava-v1.5-13b`         | `vicuna_v1.1`    | `llava`        | Open vision-chat models that add an image encoder to LLaMA/Vicuna (e.g. LLaMA2 13B) for following multimodal instruction prompts.                                                                               |
| **LLaVA-NeXT** (8B, 72B)   | `lmms-lab/llava-next-72b`                  | `chatml-llava`    | `llava	`       | Improved LLaVA models (with an 8B Llama3 version and a 72B version) offering enhanced visual instruction-following and accuracy on multimodal benchmarks.                                                       |
| **LLaVA-OneVision**        | `lmms-lab/llava-onevision-qwen2-7b-ov`     | `chatml-llava`   | `llava`        | Enhanced LLaVA variant integrating Qwen as the backbone; supports multiple images (and even video frames) as inputs via an OpenAI Vision API-compatible format.                                                 |
| **ChatGlmV4Vision** |  `zai-org/glm-4v-9b` | `chatglm4v` | `chatglm4v` | GLM-4V is a multimodal language model with visual understanding capabilities. |
| **InternVL** |`OpenGVLab/InternVL3-78B`  | `internvl` | `internvl` | A pioneering open-source alternative to GPT-4V |