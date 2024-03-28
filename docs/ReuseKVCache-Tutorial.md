# 背景
多轮会话场景下，多轮的prompt之间存在共同前缀，这些前缀token对应的kv cache是相同的，kv cache的reuse，可以减少这些重复部分的计算时间，降低First Token Time。

### ReuseCache
使用环境变量REUSE_CACHE=1，打开kv cache的复用，启动日志中会出现"reuse_cache: True"的字眼。
环境变量SEQ_SIZE_PER_BLOCK指定每个kv cache block对应的seq的数量。
**注意：ReuseCache由于q和kv长度不一致，目前无法使用flash attention，需要在运行代码前配置环境变量`export ENABLE_FMHA=OFF`**

``` python
import os
from maga_transformer.pipeline import Pipeline
from maga_transformer.model_factory import ModelFactory, ModelConfig

model_name = "Qwen/Qwen-7B-Chat"
model_config = ModelConfig()
model = ModelFactory.from_huggingface(model_name, model_config)
pipeline = Pipeline(model, model.tokenizer)

generate_config = {
    "top_k": 1,
    "max_new_tokens": 100,
}

for res in pipeline("hello, what's your name", generate_config = generate_config):
    print(res.batch_response)

# 这个query就可以复用前面query对应的kv cache。
# 在response的aux info的reuse len指示出复用了多长的kv cache。reuse len必定是SEQ_SIZE_PER_BLOCK的整数倍。
for res in pipeline("hello, what's your name? how old are you?", generate_config = generate_config):
    print(res.batch_response)

pipeline.stop()

```