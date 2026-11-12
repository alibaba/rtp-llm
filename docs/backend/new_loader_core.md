# NewLoader core

This foundation introduces checkpoint loading into registered `torch.nn.Module`
parameters and buffers without enabling a replacement serving path.

## Scope

- `NewModelLoader`: streamed checkpoint reads, rank-local filtering, runtime
  preflight, integrity validation and post-load hooks.
- `RtpModule`: explicit weight dispatch, duplicate-target detection, shape/dtype
  validation and shared-tensor identity preservation during migration.
- `WeightsMapper` and checkpoint discovery: indexed safetensors/PyTorch shards,
  path validation, filtering and deterministic name mapping.
- Registry primitives and dependency-light quantization exclusion matching.

There are deliberately **no built-in model registrations** in this batch. Model
implementations, their registrations and smoke tests will arrive together in
later changes. Existing serving code and loader defaults are unchanged. This
does not implement fastsafetensors/FST loading or enable online weight updates.

## Validation

The CPU foundation tests write real temporary safetensors and PyTorch checkpoints
and call `NewModelLoader.load()`. They check missing, duplicated and malformed
weights, shared parameters, per-rank files, filtering and lifecycle hooks. A
separate fresh-process test verifies that importing the core does not import
concrete model implementations or native inference backends.

```sh
python -m unittest \
  rtp_llm.models_py.new_loader_test.foundation_test \
  rtp_llm.models_py.new_loader_test.quantization_exclusion_test \
  rtp_llm.models_py.new_loader_test.core_boundary_test
```

Bazel targets are `//rtp_llm/models_py/new_loader_test:foundation_test`,
`//rtp_llm/models_py/new_loader_test:quantization_exclusion_test`, and
`//rtp_llm/models_py/new_loader_test:core_boundary_test`. They do not require GPUs.

Concrete layer duplicate-dispatch tests remain with the later shared-layer
batch; this foundation tests the same dispatch contract using small modules.
