# NewLoader core

NewLoader loads checkpoints into registered `torch.nn.Module` parameters and
buffers. The foundation and shared-layer commits are followed by the opt-in
dense serving integration described below.

## Scope

- `NewModelLoader`: streamed checkpoint reads, rank-local filtering, runtime
  preflight, integrity validation and post-load hooks.
- `RtpModule`: explicit weight dispatch, duplicate-target detection, shape/dtype
  validation and shared-tensor identity preservation during migration.
- `WeightsMapper` and checkpoint discovery: indexed safetensors/PyTorch shards,
  path validation, filtering and deterministic name mapping.
- Registry primitives and dependency-light quantization exclusion matching.

The foundation commit deliberately has no built-in model registrations. The
dense serving batch now registers unquantized `qwen_2`, `qwen_3` and standard HF
`llama`, together with their load tests and smoke targets. Loader defaults remain
legacy; automatic NewLoader routing requires `--require_weight_update false`,
and `--use_new_loader true` explicitly requests it.

Dense TP supports both explicit `ep_size=1` and the server's default resolved
`ep_size=tp_size` with `dp_size=1`; neither implies expert sharding for a dense
model. CP (including `PREFILL_CP`) and layer-count overrides such as
`--hack_layer_num` are not supported in this batch: automatic selection falls
back to legacy, while explicit NewLoader selection fails before reading weights.
Layer-count overrides are detected by comparing the runtime layer count against
`num_hidden_layers` in the checkpoint's HF `config.json`.

Eagle/MTP registration and tests belong to a later batch and are not included
in the dense test targets. This batch does not implement fastsafetensors/FST
loading or enable online weight updates.

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

The following contract is also exercised by the per-tensor FP8 batch below.

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

## Per-tensor FP8 method batch

This commit adds the **layer implementation**, not a new serving rollout.
`BaseModel` still rejects all model-level quantization for NewLoader (automatic
mode uses legacy; explicit NewLoader raises). Model-level quantization config
translation, per-channel/block FP8 and MoE integration remain separate batches.
The already registered Qwen2/Qwen3/Llama serving scope is unchanged.

The method API now accepts `fp8` / `FP8_PER_TENSOR_COMPRESSED` for prequantized
E4M3 weights, and `fp8_online` / `FP8_DYNAMIC_PER_TENSOR` for staged BF16/FP16
weights. Static activation quantization requires a finite positive scalar
`input_scale`; dynamic mode computes it per invocation. Weight scales are scalar
and must be finite and positive for prequantized checkpoints. Quantization
exclusions still select the unquantized method before backend resolution.

Local Linear N/K dimensions must both be divisible by 16, including TP shards.
Runtime preflight requires CUDA SM89+ or ROCm gfx942/gfx950, `torch._scaled_mm`,
and the platform's native per-tensor quantization operators. Unsupported backend
or missing symbols fails before device migration. Imports alone do not initialize
native operators or import DeepGEMM/DeepEP. ROCm gfx942 converts checkpoint FN to
FNUZ with matching scale adjustment; gfx950 retains FN. Runtime dtype conversion
of finalized FP8 parameters is not supported and is diagnosed before GEMM.

`quant_methods/test:test_fp8_per_tensor` covers real temporary checkpoint reads,
streamed merged/QKV scale handling at TP2, missing/invalid scales, FN/FNUZ
conversion and once-only conversion across staged/final hooks. CPU backend
doubles validate loader contracts, **not accelerator correctness**.
`test_fp8_per_tensor_gpu` (H20_CU13) and `test_fp8_per_tensor_gpu_rocm` (MI308X)
execute native quantization and scaled GEMM with no backend mocks. They fail if
the required accelerator is absent. These targets are not marked `open_skip`.
