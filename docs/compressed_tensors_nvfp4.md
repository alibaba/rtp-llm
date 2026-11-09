# compressed-tensors NVFP4 checkpoint adapter

This adapter initially targets dense Qwen3 checkpoints matching
[RedHatAI/Qwen3-8B-NVFP4](https://huggingface.co/RedHatAI/Qwen3-8B-NVFP4),
pinned to `e391349c110709b87bfc2ad2fde3f50dc5839fd8`.
It reuses RTP-LLM's native ModelOpt FP4 execution backend; it does not expand
weights to a resident BF16 copy or introduce a new CUDA kernel.

## Storage and scale conversion

| Checkpoint tensor | Native loader meaning |
| --- | --- |
| `weight_packed`, uint8 `[N,K/2]` | packed E2M1 codes, unchanged before backend layout conversion |
| `weight_scale`, E4M3 `[N,K/16]` | local dequantization scales, unchanged before layout conversion |
| `weight_global_scale`, FP32 scalar | `weight_scale_2 = reciprocal(weight_global_scale)` |
| `input_global_scale`, FP32 scalar | `input_scale = reciprocal(input_global_scale)` |

Runtime activation quantization receives `1/input_scale`; GEMM uses
`alpha = weight_scale_2 * input_scale`. Thus alpha equals
`1/(weight_global_scale * input_global_scale)`.
QKV and gate/up fusion require identical respective global scales.
Unequal, nonpositive or nonfinite global scales are rejected; choosing their
maximum without rewriting packed weights would change the represented matrix.
Down-projection padding converts channel alignment to packed-byte and scale-group
units separately. Existing tensor-parallel splitting and backend swizzling are reused.

## Supported scope

- One `Linear` config group, `nvfp4-pack-quantized`, compressed status.
- Symmetric static float4/group16 weights and local dynamic float4/group16 activations.
- Dense `qwen3`; unquantized `lm_head`; exact module and `re:` ignore patterns.
- No transform recipes, sparsity recipes, output/KV quantization, Hybrid models or MoE.
- Native execution requires a supported Blackwell backend and built Linux RTP-LLM environment.

Checkpoint detection is automatic through `QuantizationConfig.load_from_ckpt`.
Download the pinned checkpoint with `hf download RedHatAI/Qwen3-8B-NVFP4
--revision e391349c110709b87bfc2ad2fde3f50dc5839fd8 --local-dir /models/qwen3-nvfp4`
(on one command line), then use that directory in the existing Qwen3 launch flow.
No serving launch has been verified for this adapter yet.

## Validation

CPU parser and scale tests:

```sh
python -m unittest rtp_llm.config.test.compressed_tensors_nvfp4_test -v
```

Loader tests in the built Linux environment:

```sh
bazel test //rtp_llm/model_loader/test:test_compressed_tensors_nvfp4
```

Native numerical test on the existing SM100 test lane:

```sh
bazel test //rtp_llm/config/test:compressed_tensors_nvfp4_gpu_test
```

The native test uses a real, attributed 128x128 checkpoint tile and compares
GEMM with a dequantized reference using the same quantized activations.
This isolates adapter/layout errors from activation quantization error.

Local validation (2026-11-09): 10 new CPU tests pass; 5 existing ModelOpt parser
regression cases pass with filesystem reads mocked to avoid Windows temporary-directory
ACL failures. The native GPU test skips on this RTX 3050 (SM86).
Loader integration tests are authored but not executed locally: the Windows environment
lacks the built Linux extensions and its Torch version predates required dtype APIs.
Full-model load, TP execution, generation quality and performance remain unverified.
See `rtp_llm/config/test/fixtures/nvfp4/README.md` for checkpoint evidence.

## Comparison with existing MXFP4 and NVFP4 MoE paths

Reviewed against upstream base `97ebd38f8b6f223ffeb0d98d99705e9478cc8237`.
These are code-path observations, not newly executed GPU benchmarks.

- `model_loader/mixed_fp4_quant_weight.py`: Quark MXFP4 selects uint8 E8M0
  scales and drops ModelOpt's `scale_2`/`input_scale` subweights. The ROCm
  executor uses `aiter.fused_moe` with `per_1x32` and requires gfx950.
  Its passthrough router forwards activations without quantizing them there;
  this does not imply that the eventual kernel performs no activation conversion.
- `device/device_impl.py::shuffle_moe_weight`: reorders gate/up rows for the
  ROCm backend and shuffles both weights and E8M0 scales. Scales follow the
  same expert/row permutation as the corresponding weights.
- `model_loader/per_group_fp4_quant_weight.py::V4PerGroupFp4Weight`: a separate
  FP4/UE8M0 routed-expert path preserves packed int8 weights and E8M0 scales
  and deliberately bypasses ModelOpt NVFP4 layout preparation. DeepGEMM's
  FP8/FP4 path has a different execution contract from NVFP4 W4A4.
- `model_loader/online_modelopt_fp4_quant_weight.py`: online NVFP4 loads
  BF16/FP16 expert weights, quantizes each expert into packed E2M1 plus
  group16 E4M3 scales and an FP32 outer multiplier, then reuses the existing
  FP4 MoE layout preparation hook. It quantizes using the rounded E4M3
  scales, not the unrounded intermediate scales. This is distinct from
  loading an already quantized compressed-tensors checkpoint.
- `device/device_impl.py::maybe_prepare_static_weights_for_fp4_moe`: selects
  CuTeDSL gate/up reordering plus scale swizzling, or TRT-LLM expert-wise
  weight permutation plus scale interleaving. It is a no-op for dense keys.
- `TrtllmFp4Executor`: forms a shared input dequantization scale from the
  maximum of expert input scales, quantizes the input with its reciprocal,
  and recomputes first-GEMM alpha from that shared scale and each expert's
  weight outer multiplier. The second GEMM uses its own scale combination.
  This choice changes activation quantization; it does not justify changing
  stored weight multipliers while keeping packed weight codes unchanged.

For this adapter, reuse the loader/TP machinery and existing backend layout
hooks; keep compressed-tensors name mapping and reciprocal global-scale
conversion in the checkpoint adapter. The dense implementation already calls
these inherited hooks. Do not enable the MoE executor merely by widening its
quant-method allowlist: a real CT MoE checkpoint must establish per-expert
scale shapes, gate/up global compatibility, expert order, EP/TP slicing,
router contracts and native numerical agreement first.
