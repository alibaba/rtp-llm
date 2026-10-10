# DeepSelect BF16 K512 subset

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

Only the BF16 normal selector and its required headers are included. The
FP32 selector, cluster selector, Torch stable-ABI API, tests, generated
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
