# How to Support New Models

This guide describes the model registration, configuration, weight loading and multimodal interfaces in this repository. Start with an existing model that matches the checkpoint architecture; a new architecture can also require model-body or operator changes.

## How to Support a New Language Model

Add the model entry under the [models directory](https://github.com/alibaba/rtp-llm/tree/main/rtp_llm/models). Read [BaseModel](https://github.com/alibaba/rtp-llm/blob/main/rtp_llm/models/base_model.py) and an existing entry such as [Qwen2](https://github.com/alibaba/rtp-llm/blob/main/rtp_llm/models/qwen_v2.py).

Follow the existing entry's configuration parsing, tokenizer construction and weight mapping. Where a Python model body is required, connect it through the entry's `_create_python_model` implementation. Check tensor shapes, attention configuration, quantization and supported parallelism against the checkpoint.

Register the model type and supported Hugging Face architecture names with [register_model](https://github.com/alibaba/rtp-llm/blob/main/rtp_llm/model_factory_register.py). Keep registration in the model module so discovery and model loading use the same implementation. See [Registering an External Model Implementation](#registering-an-external-model-implementation) for an application-owned module.

For an in-tree built-in model, also add a matching `register_lazy_model` entry to `_register_builtin_lazy_models` in the same registry file. Specify the model type, importable module path and applicable Hugging Face architecture names or repository IDs. This metadata lets the factory discover and import the module before its module-level `register_model` call runs; use the same model type in both registrations.

## How to Support a New Multimodal Large Language Model

Use [Qwen2-VL](https://github.com/alibaba/rtp-llm/blob/main/rtp_llm/models/qwen2_vl.py) and the [multimodal interfaces](https://github.com/alibaba/rtp-llm/blob/main/rtp_llm/multimodal/multimodal_mixins/base_multimodal_mixin.py) as references. Follow the existing model's preprocessing, embedding and vision-weight loading paths.

Preserve the checkpoint's tokenizer and chat-template conventions, including image or video placeholders and position information. Verify how multimodal embeddings are passed to the language model; do not assume that another serving framework's processor or attention APIs apply here.

See the [multimodal development guide](../backend/Multimodal.md) for the existing multimodal entry points and separate-deployment examples.

## Test the Correctness

### Interactive Debugging

Prepare the environment using the [installation guide](../start/install.md). Compare the new implementation against the checkpoint's reference implementation with the same inputs, tokenizer, dtype and generation settings. Check preprocessing and weight mapping before investigating output differences.

### Add the Model to the Test Suite

Add focused configuration, weight-loading or operator tests beside the corresponding implementation. Add the model's request and expected-result fixtures to the existing Smoke framework and register the matching target in the [Smoke BUILD file](https://github.com/alibaba/rtp-llm/blob/main/rtp_llm/test/smoke/BUILD).

Use the original comparison tolerances and declare the required hardware, GPU count and timeout. Run the complete applicable test target and retain its logs and reports. Add the model to the [supported models list](generative_models.md) when its implementation and coverage are ready.

## Port a Model from vLLM to RTP-LLM

When porting a model, use the source implementation to understand checkpoint names, tensor layouts and numerical behavior. Adapt those details to RTP-LLM's configuration, weight-loading and model interfaces rather than copying framework-specific registries or runtime APIs.

Reuse existing RTP-LLM operators where they implement the required behavior. Validate any new operator independently, then test the complete model with its original precision and generation expectations.

## Registering an External Model Implementation

An application can import its own model module and call `register_model` before the model is resolved or loaded. The following example assumes that `my_model_plugin` provides a compatible `MyModel` implementation:

```python
from my_model_plugin import MyModel
from rtp_llm.model_factory_register import register_model

register_model(
    "my_model",
    MyModel,
    support_architectures=["MyModelForCausalLM"],
)
```

Run this registration in the same process that loads the model. Registration in an unrelated Python process does not configure a subsequently launched server. Follow the existing entry-point and model-factory code when integrating the application bootstrap.
