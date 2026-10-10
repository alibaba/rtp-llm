# DeepSelect K512 subsets

Vendored from [vllm-project/DeepSelect](https://github.com/vllm-project/DeepSelect)
at commit `d96d33afe1fab0d6066da49cdc91e64c2bee65ea`, the revision pinned by
vLLM's `cmake/external_projects/deepselect.cmake`. See `LICENSE` (MIT,
Copyright 2025 DeepSeek).

The selecting kernels were also checked against the official
[deepseek-ai/DeepSelect](https://github.com/deepseek-ai/DeepSelect) commit
`0f03b68748b304863fdf0181a11458d04ae533a9`. Its `cuda_kernels`, `structs.h`,
and kerutils sources match the pinned vLLM fork; the fork changes the host API
to the Torch stable ABI. The RTP adaptations below remain necessary. This
integration does not convert FP32 dense or candidate scores to BF16.

The original BF16 subset includes the normal selector and its required
headers. The cluster selector, Torch stable-ABI API, tests, generated
instantiations, and CUTLASS submodule are excluded. CUTLASS comes from RTP's
existing Bazel dependency. `deepselect_bf16.cu` instantiates the upstream
K512 configurations (512 threads / occupancy 1 for one wave, 256 threads /
occupancy 2 otherwise) for int32 output without sorting or value output.

Local changes to the vendored headers:

- Use workspace-qualified includes to avoid exporting generic header names.
- Apply the repository's clang-format and trailing-whitespace hooks. These
  formatting changes preserve C++ tokens and macro definitions.
- Replace three explicit-template lambdas with equivalent C++17 generic
  lambdas/integral constants and spell out two dependent `typename` aliases.
  RTP's Bazel CUDA wrapper only forwards language standards through C++17;
  these syntax changes preserve the same compile-time specialization.
- Keep only the two used `KU_LDG_256` / `KU_STG_256` macros from the kerutils
  SM100 helper. Unused TMEM/MMA helpers require a newer CUTLASS than RTP's
  existing CUDA 13 dependency and are not needed by the BF16 selector.
- Clamp the signed per-row end to `[0, width]` inside the selecting CTA.
- Initialize **all** 512 indices to `-1` when the long-row scan sees NaN.
  Upstream only initializes slot zero to a sentinel. Short rows retain the
  upstream prefix shortcut, and the caller filters nonfinite selected values
  during remapping. This exception contract does not match Torch NaN ordering.

The native wrapper accepts widths divisible by 512, 1024-byte-aligned row
strides and 16-byte-aligned input bases. This guarantees the TMA row padding
is backed by input storage, including the last row of a strided view.
Outputs have 32-byte-aligned bases/row strides. Indices are unsorted, with
unspecified membership among equal values. Finite negative values and
infinities use upstream BF16 ordering. The caller rejects selected nonfinite
values in its fused remap. No FP32 score tensor is materialized.

The native translation unit enables CuTe's SM90 TMA feature for the exact
SM103a device pass: RTP's CUTLASS 3.8 recognizes SM100a but predates SM103a.
This local compatibility definition does not enable unrelated MMA features.

## FP32 token selector

The `cuda_kernels/v3_fp32/topk_select.{h,cuh}` files are separately vendored
from official [deepseek-ai/DeepSelect](https://github.com/deepseek-ai/DeepSelect)
commit `bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18`, under the same MIT
license above. They reuse the existing shared headers, kerutils subset and
RTP CUTLASS dependency. The BF16 kernel and shared algorithms are unchanged.
`deepselect_fp32.cu` instantiates FP32/int32 K512 with 512 threads,
occupancy 1, 8192 elements per round and three TMA buffers for at most one
wave of rows. Larger row counts use 256 threads, occupancy 2, 4096 elements
per round and two TMA buffers. Both configurations use reconstruction
threshold 4096; output sorting and value output remain disabled.
The FP32 prefill integration is opt-in with `DSV41_PREFILL_DEEPSELECT=1`;
the default retains the existing FP32 selector.

FP32-specific adaptations:

- Workspace-qualified includes, C++17 generic lambdas in place of two
  explicit-template lambdas, and omission of the unused CUTLASS launch
  header for CUDA 12.9 compatibility.
- Clamp signed `ends` to `[0, width]` within the selecting CTA.
- Canonicalize both signs/payloads of NaN to positive infinity at every
  original-score register load: both initial-window passes, the main-loop
  hit predicate, and the first/subsequent incoming-pair loads. All finite
  FP32 bit patterns are retained; no BF16 conversion occurs.
- Remove the upstream NaN trap/sentinel path. NaN and positive infinity
  occupy selected slots, then `filter_finite=true` rejects selected
  nonfinite values using the final pair's FP32 exponent bits. The short-row
  prefix path applies the same predicate directly. The unfiltered variant
  is only for callers completing the established finite-filter epilogue.
- Write indices directly from the final survivor pairs and retain an
  unconditional CTA barrier before those reads. This replaces the barrier
  formerly supplied by `__syncthreads_or(have_nan)` and preserves the final
  reconstruction's producer/consumer ordering without another GPU launch.

FP32 inputs have row strides divisible by **256 elements** (1024 bytes),
and bases aligned to 16 bytes. The TMA descriptor rounds each logical row
up to 32 FP32 elements (128 bytes). For nonmultiple widths the native and
Python admission checks verify that the actual storage, including the
view's storage offset and final-row stride, backs those trailing elements;
row stride alone is insufficient. The native check uses division to avoid
overflow in the last-row address calculation. Widths already divisible by
32 need no additional allocation padding. The logical width remains the
bound for clamping `ends`, so padded values are never valid selections.
TMA's out-of-bounds zero fill handles the remainder of a 512-element box,
which the selector subsequently masks to negative infinity. Other layouts
use the existing selector. Outputs retain the BF16 wrapper's 32-byte
base/row-stride alignment rule.

Both native translation units must retain `--ftz=false`: the upstream
index arithmetic adds FP32 subnormal bit patterns as exact small integers.
The FP32 wrapper uses an input-device CUDA guard, that device's current
stream, no-overlap checks and launch-error propagation. CUDA 12.9/13 builds
for x86 and ARM share the same SM100a/SM103a device implementation.
