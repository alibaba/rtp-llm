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

The decode PA kernel reads V pages written by prefill and decode RoPE/cache ops. A reader must use
the layout produced by its writer: linear `head_dim × page`, or vectorized
`page/width × head_dim × width`, where `width` is `16 // itemsize` (8 for BF16/FP16, 16 for FP8).
Both `head_dim` and `page` must be a multiple of `width`: the Python CK prefill reader reshapes
every V page into `page // width` groups for both linear and vectorized layouts, so an unaligned
`page` fails the reshape regardless of the stored V layout. At `page == width`, linear and
vectorized addressing coincide. `page` is `--kernel_seq_size_per_block` (falling back to
`--seq_size_per_block`, then 16).

Non-ASM decode requires `page` to divide its runtime partition size: BF16/FP16 requests with
`head_dim ≤ 128` and `max_seq_len ≤ 16384` use the 512-token path, all others use the 256-token
path, and both raise (pointing at `--use_triton_pa 1`) when `page` does not divide that partition.
Only full-attention MHA with a RoPE KV cache is checked; MLA uses another factory. This matrix does
not cover PD-disaggregated role coordination.

Rows are `--use_aiter_pa`/`--use_asm_pa`/`--use_triton_pa`; columns are `size_per_head` × KV dtype.
Each cell shows `[prefill] (decode)` with the implementation and V layout (`V` vectorized, `L` linear)
at `page=16`. `❌` means the factory raises instead of returning an implementation; the cell says why —
`(none)` is no implementation for that phase, otherwise the two layouts disagree. `page=width` marks a
pair that only agrees because the layouts coincide at that page size. ASM decode requires
`size_per_head=128`. The table assumes no prefix reuse; with prefix reuse, `--use_asm_pa 1`
additionally selects the capturable `AiterPrefillImplPaged` path.

| aiter/asm/triton | 128 BASE | 128 FP8 | 256 BASE | 256 FP8 | fix for ❌ |
|---|---|---|---|---|---|
| `1/1/1` | ✅ `[Asm:V] (Triton:V)` | ✅ `[Asm:V] (Triton:V)` | ✅ `[Asm:V] (Triton:V)` | ✅ `[Asm:V] (Triton:V)` | — |
| `1/1/0` **(default)** | ✅ `[Asm:V] (Asm:V)` | ✅ `[Asm:V] (Asm:V)` | ❌ `[Asm:V] (NonAsm:L)` | ✅ `[Asm:V] (NonAsm:L)` `page=width` | `--use_triton_pa 1`, or `--use_asm_pa 0` on BASE |
| `1/0/1` | ✅ `[NonAsm:L] (TritonLin:L)` | ✅ `[NonAsm:V] (Triton:V)` | ✅ `[NonAsm:L] (TritonLin:L)` | ✅ `[NonAsm:V] (Triton:V)` | — |
| `1/0/0` | ✅ `[NonAsm:L] (NonAsm:L)` | ✅ `[NonAsm:V] (NonAsm:L)` `page=width` | ✅ `[NonAsm:L] (NonAsm:L)` | ✅ `[NonAsm:V] (NonAsm:L)` `page=width` | `--use_triton_pa 1` |
| `0/1/1` | ✅ `[Asm:V] (Triton:V)` | ✅ `[Asm:V] (Triton:V)` | ✅ `[Asm:V] (Triton:V)` | ✅ `[Asm:V] (Triton:V)` | — |
| `0/1/0` | ✅ `[Asm:V] (Asm:V)` | ✅ `[Asm:V] (Asm:V)` | ❌ `[Asm:V] (none)` | ❌ `[Asm:V] (none)` | `--use_triton_pa 1` |
| `0/0/1` | ❌ `[none] (TritonLin:L)` | ❌ `[none] (Triton:V)` | ❌ `[none] (TritonLin:L)` | ❌ `[none] (Triton:V)` | `--use_asm_pa 1`, or `--use_aiter_pa 1` |
| `0/0/0` | ❌ `[none] (none)` | ❌ `[none] (none)` | ❌ `[none] (none)` | ❌ `[none] (none)` | `--use_aiter_pa 1` |
