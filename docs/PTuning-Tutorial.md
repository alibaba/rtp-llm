# 背景
P-Tuning 论文：[GPT Understands, Too](https://arxiv.org/abs/2103.10385)，该方法将Prompt转换为可以学习的Embedding层，并用MLP+LSTM的方式来对Prompt Embedding进行一层处理。这几乎可以获得了与全参数一致的效果。甚至在某些任务上优于全参数微调。效果上P-tuning对GPT这类单项语言模型的效果提升显著，显著优于人工构建模板和直接微调，使得GPT在不擅长的知识抽取任务中可以BERT的效果持平。

另外还有一篇论文是[P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/pdf/2110.07602.pdf), 提出了P-Tuning新方法，用于对预训练语言模型进行自然语言理解（NLU）。该方法使用可训练的连续提示嵌入与离散提示相结合的方式，可以稳定地提高NLU任务的表现。实验结果表明，P-TuningV2不仅能够缩小不同离散提示之间的差距，还能够在LAMA和SuperGLUE等广泛的任务上显著提高性能。此外，P-TuningV2对于冻结或微调的语言模型，在完全监督和少样本设置下都具有有效性。

最后是一个普遍的需求：对长文本的System Prompt创建静态cache，在每次请求时直接从静态cache读取kvcache，而非重复计算，这个方法能够大幅减短模型的First Token Time

这三种方法在模型层的表达是一致的，就是在真正的请求前拼接一段kvcache，区别只在于kvcache的来源。我们在rtp-llm中对这三种需求都进行了一定的支持

## 使用方法
首先对于P-TuningV2微调模型（如chatglm和chatglm2）rtp-llm通过环境变量`PTUNING_PATH`指定微调的ckpt 地址,示例：
``` python
import os
os.environ["TOKENIZER_PATH"] = "/data/base"
os.environ["MODEL_PATH"] = "/mnt/nas1/dm//turing_005_3b"
os.environ["MODEL_TYPE"] = "qwen_14b"
os.environ['PTUNING_PATH'] =  '/data/p-tuniing'
### load model ###
worker = InferenceWorker()
...
### inference ###
prompt = "你是谁？"
generate_config = {
    "top_k":1
}
result = worker.inference(prompt=prompt, generate_config=generate_config)
print(result)

```
`PTUNING_PATH` 为字符串，其内容是PTUNING_PATH ckpt文件所在的地址. 

其次对于P-tuning和静态cache的功能，rtp-llm在async模式下通过环境变量`MULTI_TASK_PROMPT`指定需要做静态缓存的prompt信息文件，格式类似如下:
``` json
[
    {"task_id": 1, "prompt": " <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>"},
    {"task_id": 2, "prompt": "你是一个严谨的程序员，你接下来需要非常谨慎的思考并回答以下问题:"}
]
```
模型在启动时后会运行以下prompt并缓存kvcache在显存中，后续运行中如果指定task_id，就能使用这部分前缀，demo如下:
``` python
import os
os.environ["TOKENIZER_PATH"] = "/data/base"
os.environ["MODEL_PATH"] = "/mnt/nas1/dm//turing_005_3b"
os.environ["ASYNC_MODE"] = "1"
os.environ["MODEL_TYPE"] = "qwen_14b"
os.environ["MULTI_TASK_PROMPT"] = "/path/to/file"
### load model ###
worker = InferenceWorker()
...
### inference ###
prompt = "你是谁？"
generate_config = {
    "top_k":1,
    "task_id": 1
}
result = worker.inference(prompt=prompt, generate_config=generate_config)
print(result)
```