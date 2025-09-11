## Framework Architecture

The entire framework can be divided into three layers:
1. Python entry: Responsible for logic such as weights loading, starting the request server, and calling the tokenizer. This part uses `rtp_llm/start_server.py` as the entry point.
2. C++ scheduling layer: This part is mainly located in the rtp_llm/cpp/ directory, with the entry point at `rtp_llm/cpp/th_op/multi_gpu_gpt/RtpLLMOp.cc`.
3. Model computation layer: The entry point is `rtp_llm/cpp/models/GptModel.cc`. This layer assembles the model computation logic and calls the corresponding operators of the device for computation.

## Hardware Development Guide

The framework is designed to separate business logic such as request scheduling from model computation logic, so that hardware backend developers do not need to concern themselves with scheduling logic and can focus solely on implementing computation logic.
The implementation directory for each hardware should be under the `rtp_llm/cpp/devices/some_hw_impl` directory.
To fully run the model, the following basic operators need to be implemented:

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

If multi-machine communication support is needed, the following communication operators also need to be implemented:
 - broadcast
 - allReduce
 - allGather

The complete operator definitions are located in `rtp_llm/cpp/devices/DeviceOps.h`.

Special attention is needed for the Attention part.

### Operator Hierarchy Design

The implementation granularity of ops will vary across different hardware.
For example, GPUs have gemm-level implementations for FFN layers, while some hardware may implement ops at the granularity of attention layers/FFN layers.
Therefore, ops have a multi-level design. FFN/attention layers will have a default implementation. If the hardware supports fine-grained ops, implementation can start from the fine-grained level.
If the hardware wants to implement from a coarser layer granularity, it can also directly overwrite the default implementation.
For some operations, such as concat/select, the default implementation is through copy, but they can also be implemented by calling dedicated kernels.

### Hardware Backend Testing Targets

There are some basic tests in the `rtp_llm/cpp/devices/base_tests/` directory. Developers can include these tests, depend on the targets, and run them to ensure op-level result correctness.

After all op implementations are completed, the following should be run:
`bazel test   //rtp_llm/cpp/test:gpt_model_test --config=[device_type]`
to verify the model results. This test runs a Qwen 0.5B model and compares the model's output scores. Also:
`//rtp_llm/cpp/test:sampler_test --config=[device_type]`
to test the sampler results.