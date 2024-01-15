# 读取Huggingface模型
huggingface模型支持从通过模型名从远程下载模型，代码如下：(如果无法访问huggingface需要配置环境变量`HF_ENDPOINT`)
``` python
from maga_transformer.pipeline import Pipeline
from maga_transformer.model_factory import ModelFactory

if __name__ == '__main__':
    model = ModelFactory.from_huggingface("Qwen/Qwen-7B-Chat")
    pipeline = Pipeline(model, model.tokenizer)
    for res in pipeline(["hello, what's your name"], max_new_tokens = 100):
        print(res.batch_response)
    pipeline.stop()

```
也支持通过模型路径加载
``` python
model = ModelFactory.from_huggingface("/path/to/dir")
```
构建模型时默认使用基础配置参数，也可以通过构建`ModelConfig`自行修改配置，`ModelConfig`参数的介绍在下一节
``` python
from maga_transformer.utils.util import WEIGHT_TYPE

model_config = ModelConfig(
    async_mode=True,
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
    async_mode=True,
    weight_type=WEIGHT_TYPE.INT8,
    max_seq_len=2000,
    ...
)
```
# ModelConfig

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `model_type` | `str, default=''` | 模型类型 |
| `ckpt_path` | `str, default=''` | 模型路径 |
| `tokenizer_path` | `str, default=''` | tokenizer路径 |
| `async_mode` | `bool, default=False`| 是否开启异步凑批模型 |
| `weight_type` | `WEIGHT_TYPE, default=WEIGHT_TYPE.FP16` | 模型weights量化类型 |
| `act_type` | `WEIGHT_TYPE, default=WEIGHT_TYPE.FP16` | 模型weights存储类型 |
| `max_seq_len` | `bool, default=0` | beam search的个数 |
| `seq_size_per_block` | `int, default=8` | async模式下每个block的序列长度 |
| `gen_num_per_circle` | `int, default=1` | 每轮可能新增的token数，仅在投机采样情况下>1 |
| `ptuning_path` | `Optional[str], default=None` | ptuning ckpt的存储路径 |
| `lora_infos` | `Optional[Dict[str, str]]` | lora ckpt存储路径 |

目前我们支持的所有模型列表可以在`maga_transformer/models/__init__.py`查看，具体模型对应的`model_type`可以查看模型文件的`register_model`