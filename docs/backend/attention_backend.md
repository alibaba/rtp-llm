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

## ROCm KV-cache V layout and PA flag combinations

The decode PA kernel must read the same physical V layout that the RoPE/cache op writes. BASE cache
uses either linear `head_dim × page` or vectorized `page/width × head_dim × width`; FP8 cache is
always vectorized. `width` is `16 // itemsize` (8 for BF16/FP16, 16 for FP8).

- `--use_asm_pa 1` selects a vectorized prefill writer. With BASE cache,
  `--use_asm_pa 0` selects the linear non-ASM writer.
- `--use_triton_pa 1` makes Triton decode use the writer selected above. Without Triton, ASM decode
  is vectorized and supports `size_per_head=128`; non-ASM decode is linear.
- `head_dim` and `page` must be multiples of `width`. At `page == width`, linear and vectorized
  addressing coincide. `page` is `--kernel_seq_size_per_block`, falling back to
  `--seq_size_per_block`, then 16.
- Non-ASM decode requires `page` to divide its 512-token partition for BASE cache with
  `head_dim <= 128` and `max_seq_len <= 16384`, or its 256-token partition otherwise.

The factory rejects incompatible layouts before constructing an implementation. For BASE cache
with `size_per_head=256`, enable Triton or disable ASM so prefill and decode both use linear V.
Prefill/decode-disaggregated roles must use the same PA flags; cross-process layout negotiation is
not provided. These checks apply to full-attention MHA with a RoPE KV cache, not MLA.
