# 背景
P-Tuning 论文：[GPT Understands, Too](https://arxiv.org/abs/2103.10385)，该方法将Prompt转换为可以学习的Embedding层，并用MLP+LSTM的方式来对Prompt Embedding进行一层处理。这几乎可以获得了与全参数一致的效果。甚至在某些任务上优于全参数微调。效果上P-tuning对GPT这类单项语言模型的效果提升显著，显著优于人工构建模板和直接微调，使得GPT在不擅长的知识抽取任务中可以BERT的效果持平。

另外还有一篇论文是[P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks](https://arxiv.org/pdf/2110.07602.pdf), 提出了P-Tuning新方法，用于对预训练语言模型进行自然语言理解（NLU）。该方法使用可训练的连续提示嵌入与离散提示相结合的方式，可以稳定地提高NLU任务的表现。实验结果表明，P-TuningV2不仅能够缩小不同离散提示之间的差距，还能够在LAMA和SuperGLUE等广泛的任务上显著提高性能。此外，P-TuningV2对于冻结或微调的语言模型，在完全监督和少样本设置下都具有有效性。

这两种需求在模型层的表达是一致的，就是在真正的请求前拼接一段kvcache，区别只在于kvcache的来源。我们在rtp-llm中对这三种需求都进行了一定的支持

## 使用方法
### PtuingV2
首先对于P-TuningV2微调模型（如chatglm和chatglm2）rtp-llm通过`ptuning_path`参数指定微调的ckpt 地址,示例：
``` python
from maga_transformer.pipeline import Pipeline
from maga_transformer.model_factory import ModelFactory, ModelConfig

model_name = "Qwen/Qwen-7B-Chat"
model_config = ModelConfig(ptuning_path="/path/to/ptuning")
model = ModelFactory.from_huggingface(model_name, model_config)

generate_config = {
    "top_k": 1,
    "max_new_tokens": 100
}
pipeline = Pipeline(model, model.tokenizer)

for res in pipeline("hello, what's your name", generate_config = generate_config):
    print(res.batch_response)
pipeline.stop()

```