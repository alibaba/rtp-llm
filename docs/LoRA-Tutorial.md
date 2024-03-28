# 背景
框架目前提供了两种高性能的解决方案：静态LoRA和动态LoRA。静态LoRA致力于满足极致性能要求，而动态LoRA则能灵活适应多样化的应用场景

# 静态LoRA
   静态lora能够实现单一基座模型和特定的单一LoRA模型的融合，在推理过程前完成基座模型参数和lora模型参数的叠加（该过程不可逆！），因此执行静态LoRA后，无法返回融合前的基座模型推理，该方式一般用于加速单一LoRA模型的推理效率。推理的结果也是apply LoRA后的结果，无法获取apply LoRA之前结果
## 使用方法
通过环境变量指定LoRA信息,示例
``` python
### load model ###
from maga_transformer.pipeline import Pipeline
from maga_transformer.model_factory import ModelFactory, ModelConfig

model_name = "Qwen/Qwen-7B-Chat"
model_config = ModelConfig(lora_infos={"default": "/data/lora_1"})
model = ModelFactory.from_huggingface(model_name, model_config)

generate_config = {
    "top_k": 1,
    "max_new_tokens": 100
}
pipeline = Pipeline(model, model.tokenizer)

for res in pipeline("hello, what's your name", max_new_tokens = 
generate_config):
    print(res.batch_response)
pipeline.stop()

```
lora_infos是一个字典，key 为adapter名字，value 为LoRA ckpt文件所在的地址。元素个数为`1`时使用静态模式，调用inference接口时，即是apply LoRA之后的结果.

# 动态LoRA
    动态LoRA以插件式架构与基座模型相结合，在推理过程中，会根据用户请求中的adapter参数，激活相应的LoRA模型进行推理计算。这种动态LoRA流程保障了模型推理的灵活性和高效性，能适应各种不同的需求和场景处理。

## 使用方法
通过环境变量指定LoRA信息,示例
``` python
### load model ###
from maga_transformer.pipeline import Pipeline
from maga_transformer.model_factory import ModelFactory, ModelConfig

model_name = "Qwen/Qwen-7B-Chat"
model_config = ModelConfig(lora_infos={"lora_1": "/data/lora_1", "lora_2": "/data/lora_2", "lora_3": "/data/lora_3"})
model = ModelFactory.from_huggingface(model_name, model_config)
pipeline = Pipeline(model, model.tokenizer)

....
### inference ###

# 指定LoRA的adpater名
prompt = "你是谁？"
generate_config = {
    "top_k":1,
    "adapter_name": "lora_1"
}
results = [res for res in pipline(prompt, generate_config=generate_config)]
print(results[-1])

# 不指定LoRA的adpater名，访问底座模型
prompt = "你是谁？"
generate_config = {
    "top_k":1
}
results = [res for res in pipeline(prompt, generate_config=generate_config)]
print(results[-1])
```
LORA_INFO 为json字符串，其内容是一个dict，key 为adapter名字，value 为LoRA ckpt文件所在的地址。元素个数>`1`generate_config 指定LoRA的adpater名。当不指定adapter_name的时候则返回底座模型预测结果.