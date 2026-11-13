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

## Shared-layer batch

The next foundation batch adds Column/Row/Merged/QKV Linear, vocab-parallel
Embedding/LMHead, RMSNorm/RMSResNorm and SwiGLU, plus quantization interfaces and
unquantized methods. It does not register models, change serving routes, or ship
FP8/W4A8 execution backends. Unsupported quantization methods fail explicitly.

Collective operations resolve the existing runtime only when communication is
actually required; TP=1 and CPU layer imports do not load native inference ops.
Accelerator norm/activation paths retain their existing optional-backend fallback.
The public layer tests cover CPU numerics, TP shard algebra and collective-boundary
layout using controlled results, not a real distributed/GPU smoke.

## Two-stage post-load contract

The two stages serve different purposes; they are not two quantization passes.

1. Stream all checkpoint tensors into CPU modules and validate completeness.
   Run backend preflight before device migration.
2. For each leaf opting into `requires_staged_device_postprocess()`, move that
   leaf to the target device and immediately call its post-load hook. This avoids
   keeping every uncompressed staging weight on the accelerator at once. Staged
   modules must be leaves and must not share registered tensor aliases.
3. Move the remaining model to the target device and validate completeness again.
4. Visit post-load hooks through `model.modules()`, then put the model in eval
   mode. This final pass is necessary for non-staged layers and model-level
   derived state.

A staged hook is called again in step 4. Its implementation must make completed
conversion a no-op on subsequent calls, for example with a per-instance
`_post_load_done` flag set only after successful conversion. The same weights
must not be quantized or have scales fused twice.

The final traversal visits parents before children. A parent may consume staged
children that completed step 2; it must not assume that every non-staged child
has already finalized. This is not a general bottom-up initialization barrier.

Staged parameter replacement must retain checkpoint provenance for the second
completeness check. Derived-only buffers should normally be registered with
`persistent=False`; specialized loaders own any different completeness contract.

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
  rtp_llm.models_py.new_loader_test.core_boundary_test \
  rtp_llm.models_py.layers.test.shared_layers_test
```

Bazel targets are `//rtp_llm/models_py/new_loader_test:foundation_test`,
`//rtp_llm/models_py/new_loader_test:quantization_exclusion_test`, and
`//rtp_llm/models_py/new_loader_test:core_boundary_test`. They do not require GPUs.

Concrete layer duplicate-dispatch tests remain with the later shared-layer
batch; this foundation tests the same dispatch contract using small modules.
