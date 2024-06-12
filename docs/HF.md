# 读取Huggingface模型

huggingface模型支持从通过模型名从远程下载模型，代码如下：(如果无法访问huggingface需要配置环境变量`HF_ENDPOINT`)
``` python
from maga_transformer.pipeline import Pipeline
from maga_transformer.model_factory import ModelFactory

if __name__ == '__main__':
    model = ModelFactory.from_huggingface("Qwen/Qwen-1_8B-Chat")
    pipeline = Pipeline(model, model.tokenizer)
    for res in pipeline("<|im_start|>user\nhello, what's your name<|im_end|>\n<|im_start|>assistant\n", max_new_tokens = 100):
        print(res.batch_response)
    pipeline.stop()
```
其中pipeline中prompt格式是qwen模型的prompt格式，您需要换成您的模型的prompt格式。

也支持通过模型路径加载
``` python
model = ModelFactory.from_huggingface("/path/to/dir")
```
构建模型时默认使用基础配置参数，也可以通过构建`ModelConfig`自行修改配置，`ModelConfig`参数的介绍在下一节
``` python
from maga_transformer.utils.weight_type import WEIGHT_TYPE

model_config = ModelConfig(
    weight_type=WEIGHT_TYPE.INT8,
    max_seq_len=2000,
    ...
)
```

如果存在框架无法推断出模型类型，但是已经适配实现的case，可以自行指定模型类型
``` python
model_config = ModelConfig(
    model_type='chatglm',
    ckpt_path='/path/to/ckpt',
    tokenizer_path='/path/to/tokenizer',
    weight_type=WEIGHT_TYPE.INT8,
    max_seq_len=2000,
    ...
)
```