# Attention Backend

## Supporting matrix for different attention backends

| **Backend**           | **Page Size > 1** | **Spec Decoding** | **MLA** | **Sliding Window** |         **Device Support**         |         **Server Args**         |         **Stage**         |
|-----------------------|-------------------|-------------------|---------|--------------------|------------------------------------|---------------------------------|---------------------------|
| **FLASHINFER_TRTLLM_GEN**        | ✅                | ✅                 | ❌      | ❌                 | NV SM100 ✅<br> AMD ❌ | --enable_flashinfer_trtllm_gen        | PREFILL ✅ <br>  DECODE✅  |
| **FLASHINFER_TRT_FMHA_V2**       | ❌                | ❌                 | ❌      | ❌                 | NV SM90/SM12x ✅<br> AMD ❌ | --enable_flashinfer_trt_fmha_v2       | PREFILL ✅ <br>  DECODE❌  |
| **PAGED_FLASHINFER_TRT_FMHA_V2** | ✅                | ❌                 | ❌      | ❌                 | NV SM90/SM12x ✅<br> AMD ❌ | --enable_paged_flashinfer_trt_fmha_v2 | PREFILL ✅ <br>  DECODE❌  |
| **OPEN_SOURCE**       | ❌                | ❌                 | ❌      | ❌                 | NV ✅<br> AMD ❌        | --enable_open_source_fmha       | PREFILL ✅ <br>  DECODE❌  |
| **PAGED_OPEN_SOURCE** | ✅                | ❌                 | ❌      | ❌                 | NV ✅<br> AMD ❌        | --enable_paged_open_source_fmha | PREFILL ✅ <br>  DECODE❌  |
| **CKFMHA**            | ❌                | ❌                 | ✅      | ✅                 | NV ❌<br> AMD ✅        | None                            | PREFILL ✅ <br>  DECODE❌  |
| **FLASHINFER_NATIVE** | ✅                | ✅                 | ✅      | ✅                 | NV ✅<br> AMD ✅        | --disable_flashinfer_native     | PREFILL ✅ <br>  DECODE✅  |
| **XQA**               | ✅                | ❌                 | ❌      | ❌                 | NV Hopper ✅<br> AMD ❌ | --enable_xqa                    | PREFILL ❌ <br>  DECODE✅  |
| **FlashMLA**          | ✅                | ✅                 | ✅      | ❌                 | NV Hopper ✅<br> AMD ❌ | None                            | PREFILL ❌ <br>  DECODE✅  |
| **MMHA**              | ✅                | ❌                 | ❌      | ❌                 | NV ✅<br> AMD ✅        | None                            | PREFILL ❌ <br>  DECODE✅  |
| **AiterPA**           | ✅                | ❌                 | ❌      | ❌                 | NV ❌<br> AMD ✅        | None                            | PREFILL ❌ <br>  DECODE✅  |
