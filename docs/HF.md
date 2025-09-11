# Loading Huggingface Models

Huggingface models support downloading models remotely by model name. The code is as follows: (If you cannot access Huggingface, you need to configure the environment variable `HF_ENDPOINT`)
``` python
from rtp_llm.pipeline import Pipeline
from rtp_llm.model_factory import ModelFactory

if __name__ == '__main__':
    model = ModelFactory.from_huggingface("Qwen/Qwen-1_8B-Chat")
    pipeline = Pipeline(model, model.tokenizer)
    for res in pipeline("<|im_start|>user\nhello, what's your name<|im_end|>\n<|im_start|>assistant\n", max_new_tokens = 100):
        print(res.batch_response)
    pipeline.stop()
```
The prompt format in the pipeline is the Qwen model's prompt format. You need to replace it with your model's prompt format.

It also supports loading through model path
``` python
model = ModelFactory.from_huggingface("/path/to/dir")
```
When building the model, basic configuration parameters are used by default. You can also modify the configuration by building `ModelConfig`. The introduction to `ModelConfig` parameters is in the next section.
``` python
from rtp_llm.utils.weight_type import WEIGHT_TYPE

model_config = ModelConfig(
    weight_type=WEIGHT_TYPE.INT8,
    max_seq_len=2000,
    ...
)
```

If there are cases where the framework cannot infer the model type but the implementation has already been adapted, you can specify the model type manually
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