## 框架架构

整个框架可以拆分为三层
1. python 入口：负责weights加载、启动请求server、调用tokenizer等逻辑。这部分以`maga_transformer/start_server.py`为入口。
2. c++ 调度层：这部分主要位于maga_transformer/cpp/目录下，入口为`maga_transformer/cpp/th_op/multi_gpu_gpt/RtpLLMOp.cc`。
3. 模型计算层：入口为`maga_transformer/cpp/models/GptModel.cc`，这一层组装模型计算逻辑，并调用device的相应算子进行运算。

## 硬件开发指南

框架在设计中期望能够分离请求调度等业务逻辑和模型计算逻辑，使得硬件后端开发者无需关心调度逻辑，只需专注于实现计算逻辑。
每种硬件的实现目录应当在`maga_transformer/cpp/devices/some_hw_impl`目录下。
要完整运行模型，需要实现以下基础算子:

 - copy
 - transpose
 - convert
 - layernorm
 - gemm
 - embeddingLookup
 - activation
 - softmax
 - contextAttention
 - decoderSelfAttention
 - sampleGreedy

如果需要支持多机通信，还需要实现以下通信算子：
 - broadcast
 - allReduce
 - allGather

完整的算子定义位于`maga_transformer/cpp/devices/DeviceOps.h`。

其中Attention部分需要特别注意

### 算子层级设计

不同硬件的op实现粒度会有所不同。
比如，gpu对ffn layer有gemm级别的实现，而有的硬件可能在attention layer/ffn layer这一粒度上实现op。
因此op会有多个层级的设计，ffn / attention layer会有一个默认实现，如果硬件支持细粒度op，可以从细粒度开始实现。
而如果硬件希望从更粗的layer粒度实现，也可以直接overwrite默认实现。
对于部分操作，如concat/select，默认通过copy进行实现，也可以通过调用dedicated kernel实现。

### 硬件后端测试目标

在`maga_transformer/cpp/devices/base_tests/`目录下有一些基础测试，开发者可以include这些测试，依赖目标并运行，保证op级别的结果正确性。

在所有op实现完成之后，应当运行
`bazel test   //maga_transformer/cpp/test:gpt_model_test --config=[device_type]`
验证模型的结果。这个测试运行了一个qwen 0.5b的模型，比较模型的输出分数。以及
`//maga_transformer/cpp/test:sampler_test --config=[device_type]`
测试sampler的结果。
