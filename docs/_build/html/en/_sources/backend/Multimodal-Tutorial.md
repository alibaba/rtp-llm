# Background

Multimodal models refer to models that communicate through multiple modalities with computers, aiming to enable models to process and understand multi-modal information. Currently, common multimodal research directions include images, videos, audio, etc.

In rtp-llm, the currently supported multimodal models are mainly models that accept images as input, such as [qwen-vl](https://github.com/QwenLM/Qwen-VL) and [llava](https://github.com/haotian-liu/LLaVA).

## Usage

### LLaVA

The config.json of LLaVA in HF format contains the mm_vision_tower keyword as the path to the ViT, typically using OpenAI's pretrained CLIP.

#### Invocation

Consistent with HF format, when calling, use the `<image>` tag in the prompt to indicate the image insertion position, and insert the image sequence in List[str] format: rtp-llm's multimodal interface allows inserting multiple images in a single prompt, but the effect of current supported models on multiple images is not good. Additionally, it is necessary to strictly ensure that the number of image tags matches the number of images.

### Qwen-VL

Slightly different from LLaVA, although Qwen-VL's ViT also uses CLIP, its parameters are written together with the LLM part, so the ViT part parameters will be read from the checkpoint.

#### Invocation

Consistent with HF format, when calling, directly use the `<img>{img_url}</img>` tag to mark images in the prompt; additionally, you can also directly use the `<img/>` image placeholder to achieve separation of URL and prompt input.

## Demo

``` python
import os
from rtp_llm.pipeline import Pipeline
from rtp_llm.model_factory import ModelFactory, ModelConfig

model_name = "Qwen/Qwen-VL-Chat"
model = ModelFactory.from_huggingface(model_name)
pipeline = Pipeline(model, model.tokenizer)

generate_config = {
    "top_k": 1,
    "max_new_tokens": 100
}
```
Request response in the following way:
``` python
for res in pipeline("Picture 1:<img/>\nWhat is this?", urls = ["https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"], generate_config = generate_config):
    print(res.batch_response)

pipeline.stop()
```

Or

``` python
for res in pipeline("Picture 1:<img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>\nWhat is this?", generate_config = generate_config):
    print(res.batch_response)
```

Additionally, if starting as a service:
``` shell
export MODEL_TYPE=qwen_vl
export TOKENIZER_PATH=/path/to/tokenizer
export CHECKPOINT_PATH=/path/to/model
export FT_SERVER_TEST=1

python3 -m rtp_llm.start_server

# request to server
```
``` python
import requests
chat_messages = [
    {
        "role": "user",
        "content": [
        {
            "image_url": {
            "url": image_path
            },
            "type": "image_url"
        },
        {
            "text": "Describe this image",
            "type": "text"
        }
        ],
        "name": None
  }
]
openai_request = {
    "temperature": 0.0,
    "top_p": 0.1,
    "messages": chat_messages
}
response = requests.post(f"http://localhost:{port}/v1/chat/completions", json=openai_request)
print(response.json())

```