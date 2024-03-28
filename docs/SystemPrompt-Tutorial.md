# 背景
对长文本的System Prompt创建静态cache，在每次请求时直接从静态cache读取kvcache，而非重复计算，这个方法能够大幅减短模型的First Token Time

## 使用方法
### MultiTaskPrompt
rtp-llm通过环境变量`MULTI_TASK_PROMPT`指定需要做静态缓存的system prompt信息文件，格式类似如下:
``` json
[
    {"task_id": 1, "prompt": " <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>"},
    {"task_id": 2, "prompt": "你是一个严谨的程序员，你接下来需要非常谨慎的思考并回答以下问题:"}
]
```
也可以使用`MULTI_TASK_PROMPT_STR`环境变量将上述json传递进去。

模型在启动时后会运行以上system prompt并缓存kvcache在显存中，后续运行中如果指定task_id，就能使用这部分前缀，demo如下:
**注意：MultiTaskPrompt由于q和kv长度不一致，目前无法使用flash attention，需要在运行代码前配置环境变量`export ENABLE_FMHA=OFF`**

``` python
import os
from maga_transformer.pipeline import Pipeline
from maga_transformer.model_factory import ModelFactory, ModelConfig

os.environ["MULTI_TASK_PROMPT"] = "/path/to/file"
# os.environ["MULTI_TASK_PROMPT_STR"] = "{json str}"
model_name = "Qwen/Qwen-7B-Chat"
model_config = ModelConfig()
model = ModelFactory.from_huggingface(model_name, model_config)
pipeline = Pipeline(model, model.tokenizer)

# 使用task_id=1的system prompt拼接请求
generate_config = {
    "top_k": 1,
    "max_new_tokens": 100,
    "task_id": "1"
}

for res in pipeline("hello, what's your name", generate_config = generate_config):
    print(res.batch_response)

# 不使用system prompt
generate_config = {
    "top_k": 1,
    "max_new_tokens": 100,
}

for res in pipeline("hello, what's your name", generate_config = generate_config):
    print(res.batch_response)

pipeline.stop()

```

### 注意：
在使用MULTI_TASK_PROMPT的情况下，REUSE_CACHE功能被打开，那么kv cache就可以复用，参考文档[ReuseKVCache](docs/ReuseKVCache-Tutorial.md)。
在指定task id的情况下，使用task_id的system prompt拼接请求，并且在kv cache里面寻找最长匹配的历史请求，复用kv cache。
在不指定task id的情况下，使用用户的prompt，在kv cache里面寻找最长匹配的历史请求，复用kv cache。