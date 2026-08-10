# Attention Backend

## Supporting matrix for different attention backends

| **Backend**           | **Page Size > 1** | **Spec Decoding** | **MLA** | **Sliding Window** |         **Device Support**         |         **Server Args**         |         **Stage**         |
|-----------------------|-------------------|-------------------|---------|--------------------|------------------------------------|---------------------------------|---------------------------|
| **FLASHINFER_TRTLLM_GEN**        | ✅                | ✅                 | ❌      | ❌                 | NV SM100 ✅<br> AMD ❌ | --enable_flashinfer_trtllm_gen        | PREFILL ✅ <br>  DECODE✅  |
| **FLASHINFER_TRT_FMHA_V2**       | ❌                | ❌                 | ❌      | ❌                 | NV SM90/SM12x ✅<br> AMD ❌ | --enable_flashinfer_trt_fmha_v2       | PREFILL ✅ <br>  DECODE❌  |
| **PAGED_FLASHINFER_TRT_FMHA_V2** | ✅                | ❌                 | ❌      | ❌                 | NV SM90/SM12x ✅<br> AMD ❌ | --enable_paged_flashinfer_trt_fmha_v2 | PREFILL ✅ <br>  DECODE❌  |
| **FA4_TARGET_VERIFY**            | ✅                | ✅                 | ❌      | ❌                 | NV SM90 + CUDA 12.9 ✅<br> AMD ❌ | --enable_fa4_target_verify<br>AND --enable_paged_open_source_fmha | PREFILL ✅ <br>  DECODE❌  |
| **FLASHINFER_FA2_TARGET_VERIFY** | ✅                | ✅                 | ❌      | ❌                 | NV SM90 ✅<br> AMD ❌ | --enable_flashinfer_fa2_target_verify<br>AND --disable_flashinfer_native=false | PREFILL ✅ <br>  DECODE❌  |
| **OPEN_SOURCE**       | ❌                | ❌                 | ❌      | ❌                 | NV ✅<br> AMD ❌        | --enable_open_source_fmha       | PREFILL ✅ <br>  DECODE❌  |
| **PAGED_OPEN_SOURCE** | ✅                | ❌                 | ❌      | ❌                 | NV ✅<br> AMD ❌        | --enable_paged_open_source_fmha | PREFILL ✅ <br>  DECODE❌  |
| **CKFMHA**            | ❌                | ❌                 | ✅      | ✅                 | NV ❌<br> AMD ✅        | None                            | PREFILL ✅ <br>  DECODE❌  |
| **FLASHINFER_NATIVE** | ✅                | ✅                 | ✅      | ✅                 | NV ✅<br> AMD ✅        | --disable_flashinfer_native     | PREFILL ✅ <br>  DECODE✅  |
| **XQA**               | ✅                | ❌                 | ❌      | ❌                 | NV Hopper ✅<br> AMD ❌ | --enable_xqa                    | PREFILL ❌ <br>  DECODE✅  |
| **FlashMLA**          | ✅                | ✅                 | ✅      | ❌                 | NV Hopper ✅<br> AMD ❌ | None                            | PREFILL ❌ <br>  DECODE✅  |
| **MMHA**              | ✅                | ❌                 | ❌      | ❌                 | NV ✅<br> AMD ✅        | None                            | PREFILL ❌ <br>  DECODE✅  |
| **AiterPA**           | ✅                | ❌                 | ❌      | ❌                 | NV ❌<br> AMD ✅        | None                            | PREFILL ❌ <br>  DECODE✅  |

The FA4 target-verify row is limited to CUDA Graph prefill with BF16 query/base KV cache, head dimension 256, and page size 64. FlashInfer FA2 target verify accepts standard RoPE and MRoPE under one rollback gate, with FP16/BF16 query, base/FP8 KV cache, head dimension 64/128/256, and a positive power-of-two page size. Candidate order, cold-start guidance, and rollback behavior are documented in [Server arguments](./server_arguments.md#target-verify-backend-rollout-and-rollback).

FA4 runtime loading requires `nvidia-cutlass-dsl>=4.5.3,<4.6`, `apache-tvm-ffi>=0.1.12,<0.2`, `quack-kernels>=0.5.0,<0.6`, and `torch-c-dlpack-ext>=0.1.5,<0.2`. A Bazel consistency test reads the CUDA 12.9 requirements file and requires those declared ranges to match the runtime loader gate, so dependency changes must update both contracts together.
