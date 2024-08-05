# 背景
多模态模型是指通过多种载体和计算机进行交流，旨在让模型实现处理和理解多源模态信息的能力。目前比较常见的多模态研究方向包括图像、视频、音频等。

在rtp-llm中，目前支持的多模态模型主要都是接受图片作为输入的模型，例如[qwen-vl](https://github.com/QwenLM/Qwen-VL)和[llava](https://github.com/haotian-liu/LLaVA)。

## 使用方法

### LLaVA

hf格式下的LLaVA的config.json包含mm_vision_tower关键词作为vit的路径，一般是使用openai的预训练clip。

#### 调用

和hf格式一致，调用时，在prompt里用`<image>`标签注明图片插入位置，以List[str]的格式插入图片序列：rtp-llm的多模态接口允许一条prompt插入多张图片，但目前支持的模型对于多图的效果并不好。此外，需要严格保证图片标签数量和图片数量一致。

### Qwen-VL

和LLaVA略微有所不同，Qwen-VL的vit虽然也使用了clip，其参数是和LLM部分写在一起的，因此会从ckpt读取vit部分参数。

#### 调用

和hf格式一致，调用时，直接在prompt里用`<img>{img_url}</img>`标记图片；此外，也可以直接使用`<img/>`图片占位符来实现url和prompt的分离输入。

## Demo

``` python
import os
from maga_transformer.pipeline import Pipeline
from maga_transformer.model_factory import ModelFactory, ModelConfig

model_name = "Qwen/Qwen-VL-Chat"
model = ModelFactory.from_huggingface(model_name)
pipeline = Pipeline(model, model.tokenizer)

generate_config = {
    "top_k": 1,
    "max_new_tokens": 100
}
```
按以下方式请求response：
``` python
for res in pipeline("Picture 1:<img/>\n这是什么", urls = ["https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"], generate_config = generate_config):
    print(res.batch_response)

pipeline.stop()
```

或者

``` python
for res in pipeline("Picture 1:<img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>\n这是什么", generate_config = generate_config):
    print(res.batch_response)
```

此外，如果以服务的方式启动：
``` shell
export MODEL_TYPE=qwen_vl
export TOKENIZER_PATH=/path/to/tokenizer
export CHECKPOINT_PATH=/path/to/model
export FT_SERVER_TEST=1

python3 -m maga_transformer.start_server

# request to server
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
            "text": "描述一下这张图片",
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