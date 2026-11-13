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

Global multipliers are normalized to zero-dimensional FP32 tensors.
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
- Dense `qwen3`; unquantized `lm_head`; only `ignore=["lm_head"]` (other BF16 islands are rejected).
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
Full built-Linux loader integration remains unexecuted locally. Isolated Python-loader
unit checks with explicit native/database substitutes are recorded below.
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

## RTX 4060 Ti software validation

On 2026-11-09, the actual RTX 4060 Ti 8GB host completed 12 software checks
with Torch 2.5.1+cu124. The raw report is
[ct_nvfp4_4060_result.json](validation/ct_nvfp4_4060_result.json).

The independent reference decodes E2M1/E4M3 fields with scalar arithmetic
and computes in FP64. The GPU path uses the adapter's reciprocal-scale
helper and FP32 CUDA GEMM with TF32 disabled. Both use a real checkpoint
weight tile; activations are synthetic, exactly representable quantized values.

- Weight reconstruction maximum absolute error: `2.980232241545089e-09`.
- GPU GEMM relative L2 error against FP64: `1.1664547436050209e-07`.
- All 256 packed byte patterns and 254 finite E4M3 byte patterns checked.
- Group-aligned partition and zero-padding algebra checked.
- Wrong reciprocal, nibble ordering and scale grouping negative controls detected.
- Unequal fused globals rejected.

Reproduce the software checks from the repository with a CUDA-enabled Torch:

```sh
python scripts/validate_ct_nvfp4_software.py --output /path/to/result.json
```

The committed runner adjusts fixture/helper paths and report output arguments
from the standalone host runner, retaining the validation logic. The executed
standalone runner SHA256 was
`cfd56216f73ce4b90fc7829a8affa4575d044087cd5bac92a4616d7f8987f1f4`;
the exact adapter helper supplied to the host had SHA256
`4f8ffbb2ee2ed20476e63e04861a90aea652833a3c8cafd87d7aff9164880620`.

This is not native NVFP4 kernel, production loader, TP/EP, whole-model,
quantization-quality or performance validation. Peak Torch allocated memory
in the report excludes the CUDA context and other processes. The existing
host environment was not modified; files were placed in an isolated run directory.

## Python loader pipeline checks

An isolated Torch 2.8.0 CPU dependency directory (no changes to the existing
CUDA/FracDTI environments) ran 25 unit tests: 15 compressed-tensors NVFP4
loader tests and 10 existing FP4 exclusion regression tests, all passing.
The report is [ct_nvfp4_loader_unit_result.json](validation/ct_nvfp4_loader_unit_result.json).

The tests execute the actual Python `WeightModule.create()` and `load()`
methods, checkpoint-name formatting, fusion, padding and TP slicing code.
A recording TensorSource supplies real q_proj fixture bytes; structural
QKV/gate-up cases reuse or slice those bytes and are explicitly synthetic
assemblies, not additional real projection checkpoints. A recording device
checks that the layout hook receives the correct tensors after splitting;
it does not implement or validate native swizzling.

The new down-projection negative control first failed because independently
padding malformed packed/scales inputs could hide their original mismatch.
A per-load source proxy now checks paired raw per-shard shapes before padding,
without changing tensor contents or mutating shared atomic weights. Final
post-padding shape checks remain in place. Original tests also now construct
wrappers through the real factory, which supplies required framework metadata.

Optional reproduction without built native extensions:

```sh
python scripts/validate_ct_nvfp4_loader_units.py --isolated-boundaries --output /path/to/loader_result.json
```

The explicit isolation flag substitutes `rtp_llm.ops` and the disk database
boundary, and bypasses eager `model_loader` package initialization. Real Python
loader, tensor-source and tensor-operation code is used. This does not test
complete package initialization/registry, a real disk database, compiled Linux
integration, distributed communication or native layout/GEMM. In a built
Linux environment use the normal Bazel target instead of the isolation runner.

PR #1664 status checked on 2026-11-09: CLA success, no submitted reviews/comments;
the CI Request Trigger run reported `action_required` with no jobs. The repository
workflow documents a fresh LGTM gate before internal CI is triggered. No CI pass
or GPU result is inferred from those statuses.

## Review audit (2026-11-12)

### Merge conflict cause

Upstream advanced from `97ebd38f8b` to `823f0173c1`. PR #1543 added the
Qwen3.8-Flash-Next recipe/configuration and tests; #1521 also added test targets.
The only conflicts were the compressed-tensors parsing hunk in
`config/quant_config.py` (upstream formatting versus the adapter insertion),
and appended targets in `config/test/BUILD` and `model_loader/test/BUILD`.
Both sides' functionality/targets are preserved. These were overlapping edits,
not a second implementation of this checkpoint adapter.

### Five review findings

1. **Local-scale layout: upstream contract confirmed for the pinned version.**
   The exact checkpoint-declared package, normalized as
   `compressed-tensors==0.11.1a20250917`, was downloaded from PyPI and SHA256
   verified. Its compressor preserves scale tensors without swizzling; its
   GROUP dequantization unflattens the last dimension and applies
   `scale.unsqueeze(-1)`. Selected upstream pack/unpack/dequantize function
   bodies were executed eagerly (compile decorators omitted), not reimplemented
   as our reference. The real tile matches upstream dequantization within FP32
   rounding. See [upstream format results](validation/ct_nvfp4_upstream_format_result.json).
   This is not a full-package/native execution or original-BF16 quality test.
2. **Mixed BF16 exclusions: confirmed scope/pricing gap, addressed by rejection.**
   Arbitrary layer exclusions could leave BF16 linears under FP4 memory pricing.
   The initial adapter now requires exactly `ignore=["lm_head"]` and rejects
   extra exact names, regexes, empty lists and other forms before loading. It
   does not change generic memory estimation or claim mixed-precision support.
3. **Equal fused globals: intentional guard, now backed by a complete audit.**
   Fresh bounded-range reads of both pinned safetensors shards confirm all
   504 global values are positive F32 `[1]` scalars; all 144 QKV/gate-up pairs
   of equality checks pass. All 252 local-scale tensor shapes/dtypes were
   checked. [checkpoint_audit.json](../rtp_llm/config/test/fixtures/nvfp4/checkpoint_audit.json)
   includes every scalar, source header hashes, raw scalar-region bytes/hashes
   and layer-0 metadata. CPU tests independently repeat all equality checks.
   This proves compatibility only for this pinned checkpoint, not all CT models.
   Unequal globals remain rejected; errors now include actual layer/module
   names and values. No silent max-scale substitution is introduced.
4. **Scalar rank: normalized proactively; native rejection was not established.**
   Source globals are `[1]` and the generic loader unsqueezes them to `[1,1]`.
   The adapter now returns FP32 scalars with shape `[]`, matching the usual
   fused ModelOpt path. Unit tests cover raw `[]`, `[1]` and `[1,1]`, and assert
   actual loaded scalar shapes. The native test covers these ranks too, but
   still needs execution on Blackwell.
5. **Nibble-order oracle: test blind spot confirmed and corrected.**
   The pinned upstream packer puts even K in the low nibble; all 256 packed
   bytes round-trip, and the asymmetric pair `[0.5,6]` packs to `0x71`.
   Earlier native-test references decoded both operands with one helper,
   allowing a common adjacent-pair permutation to cancel. The revised native
   test uses known asymmetric exactly representable activations (global=448)
   and requires their quantization round-trip to match the pre-quantization
   input exactly. Its expected GEMM uses that known input, and a weight-only
   nibble-swap must change output substantially. This strengthened native test
   is authored, not executed; CT-to-kernel compatibility still needs hardware.

Reproduce external-source checks (network required; small bounded downloads):

```sh
python scripts/audit_ct_nvfp4_checkpoint.py --output /path/to/checkpoint_audit.json
python scripts/audit_ct_nvfp4_format.py --wheel /path/to/compressed_tensors.whl --output /path/to/format_result.json
```

After merging upstream, 12 parser/scale/audit CPU tests and 25 isolated
Python-loader/regression tests pass. The one native GPU test skips on the
available hardware. The original 4060 report remains historical evidence
for its original helper version; it is not presented as a rerun of the new code.

The revised scalar helper also passed the 12 software checks again on the
local RTX 3050 SM86; the new report is
[ct_nvfp4_software_review_local_result.json](validation/ct_nvfp4_software_review_local_result.json).
This rerun is separate from the historical RTX 4060 result and does not
execute a native FP4 instruction/kernel.
