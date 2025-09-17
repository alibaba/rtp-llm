# How to Support New Models

This document explains how to add support for new language models and multimodal large language models (MLLMs) in
RTP-LLM. It also covers how to test new models and register external implementations.

## How to Support a New Language Model

To support a new model in RTP-LLM, you only need to add a single file under
the [RTP-LLM Models Directory](http://gitlab.alibaba-inc.com/foundation_models/RTP-LLM/tree/main/rtp_llm/models_py/). You can learn
from existing model implementations and create a new file for your model. For most models, you should be able to find a
similar model to start with (e.g., starting from Llama). Also refer how
to [port a Model from vLLM to RTP-LLM](#port-a-model-from-vllm-to-RTP-LLM)

## How to Support a New Multimodal Large Language Model

To support a new multimodal large language model (MLLM) in RTP-LLM, there are several key components in addition to the
standard LLM support:

1. **Register your new model as multimodal**:
   Extend `is_multimodal_model`
   in [model_config.py](http://gitlab.alibaba-inc.com/foundation_models/RTP-LLM/blob/0ab3f437aba729b348a683ab32b35b214456efc7/python/RTP-LLM/srt/configs/model_config.py#L561)
   to return `True` for your model.

2. **Register a new chat-template**
   See [conversation.py](http://gitlab.alibaba-inc.com/foundation_models/RTP-LLM/blob/86a779dbe9e815c02f71ea82574608f6eae016b5/python/RTP-LLM/srt/conversation.py)

3. **Multimodal Data Processor**:
   Define a new `Processor` class that inherits from `BaseMultimodalProcessor` and register this processor as your
   model’s dedicated processor.
   See [multimodal_processor.py](http://gitlab.alibaba-inc.com/foundation_models/RTP-LLM/blob/main/python/RTP-LLM/srt/managers/multimodal_processor.py)
   for more details.

4. **Handle Multimodal Tokens**:
   Implement a `pad_input_ids` function for your new model. In this function, multimodal tokens in the prompt should be
   expanded (if necessary) and padded with multimodal-data-hashes so that RTP-LLM can recognize different multimodal data
   with `RadixAttention`.

5. **Adapt to Vision Attention**:
   Adapt the multi-headed `Attention` of ViT with RTP-LLM’s `VisionAttention`.

You can refer to [Qwen2VL](http://gitlab.alibaba-inc.com/foundation_models/RTP-LLM/blob/main/python/RTP-LLM/srt/models/qwen2_vl.py) or
other mllm implementations. These models demonstrate how to correctly handle both multimodal and textual inputs.

You should test the new MLLM locally against Hugging Face models. See the [
`mmmu`](http://gitlab.alibaba-inc.com/foundation_models/RTP-LLM/tree/main/benchmark/mmmu) benchmark for an example.

## Test the Correctness

### Interactive Debugging

For interactive debugging, compare the outputs of Hugging Face/Transformers and RTP-LLM. The following two commands
should give the same text output and very similar prefill logits:

- Get the reference output:
  ```bash
  python3 scripts/playground/reference_hf.py --model-path [new model] --model-type {text,mllm}
  ```
- Get the RTP-LLM output:
  ```bash
  python3 -m RTP-LLM.bench_one_batch --correct --model [new model]
  ```

### Add the Model to the Test Suite

To ensure the new model is well maintained, add it to the test suite by including it in the `ALL_OTHER_MODELS` list in
the [test_generation_models.py](http://gitlab.alibaba-inc.com/foundation_models/RTP-LLM/blob/main/test/srt/models/test_generation_models.py)
file, test the new model on your local machine and report the results on demonstrative benchmarks (GSM8K, MMLU, MMMU,
MMMU-Pro, etc.) in your PR.

This is the command to test a new model on your local machine:

```bash
ONLY_RUN=Qwen/Qwen2-1.5B python3 -m unittest test_generation_models.TestGenerationModels.test_others
```

## Port a Model from vLLM to RTP-LLM

The [vLLM Models Directory](https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/models) is a valuable
resource, as vLLM covers many models. RTP-LLM reuses vLLM’s interface and some layers, making it easier to port models
from vLLM to RTP-LLM.

To port a model from vLLM to RTP-LLM:

- Compare these two files for guidance:
    - [RTP-LLM Llama Implementation](http://gitlab.alibaba-inc.com/foundation_models/RTP-LLM/blob/main/python/RTP-LLM/srt/models/llama.py)
    - [vLLM Llama Implementation](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/llama.py)
- The major differences include:
    - **Replace vLLM’s `Attention` with `RadixAttention`** (ensure you pass `layer_id` to `RadixAttention`).
    - **Replace vLLM’s `LogitsProcessor` with RTP-LLM’s `LogitsProcessor`.**
    - **Replace the multi-headed `Attention` of ViT with RTP-LLM’s `VisionAttention`.**
    - **Replace other vLLM layers** (such as `RMSNorm`, `SiluAndMul`) with RTP-LLM layers.
    - **Remove `Sample`.**
    - **Change the `forward()` functions** and add a `forward_batch()` method.
    - **Add `EntryClass`** at the end.
    - **Ensure that the new implementation uses only RTP-LLM components** and does not rely on any vLLM components.

Note: make sure you add your new model to the supported models list in the supported models documentation.

## Registering an External Model Implementation

In addition to the methods above, you can register your new model with the `ModelRegistry` before launching the server.
This allows you to integrate your model without modifying the source code.

For example:

```python
from RTP-LLM.srt.models.registry import ModelRegistry
from RTP-LLM.srt.entrypoints.http_server import launch_server

# For a single model, add it to the registry:
ModelRegistry.models[model_name] = model_class

# For multiple models, you can imitate the import_model_classes() function:
from functools import lru_cache

@lru_cache()
def import_new_model_classes():
    model_arch_name_to_cls = {}
    # Populate model_arch_name_to_cls with your new model classes.
    ...
    return model_arch_name_to_cls

ModelRegistry.models.update(import_new_model_classes())

# Launch the server with your server arguments:
launch_server(server_args)
```

---

By following these guidelines, you can add support for new language models and multimodal large language models in
RTP-LLM and ensure they are thoroughly tested and easily integrated into the system.
