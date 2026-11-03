# DeepSeek V4 Flash PPU optional features

These options belong to the Flash PPU model/provider. They do not change
the shared GPU model, loader, scheduler or CUDA Graph runner. Presence in
this table describes code availability, not hardware qualification.

| Function | Selection | Status |
| --- | --- | --- |
| Decode operator fusion, MTP input fusion, ordered C4 queries | `DSV4_PPU_DECODE_FUSIONS=1` | Optional runtime path, default off |
| MegaMoE | `DSV4_PPU_DECODE_MOE_BACKEND=megamoe` | Optional runtime path; compatible native package required |
| Metadata graph and fused compressor state slots | `DSV4_PPU_MTP_METADATA_GRAPH=1` | Optional runtime path for q_len 1 through 4, default off |
| Inactive-route masking | `DSV4_PPU_MTP_MASK_INACTIVE=1` with metadata graph | Optional runtime path, default off |
| Prefix reuse | `DSV4_PPU_PREFIX_REUSE=1` with reuse-cache configuration | Experimental runtime path; correctness still needs PPU qualification |

The metadata graph uses its own updater so fused state-slot computation
replaces the reference computation. It retains the shared runner's normal
host mirrors. Turning the option off retains the existing metadata factory.
MegaMoE requires a compatible PPU DeepGEMM build exposing
`get_symm_buffer_for_mega_moe`, `transform_weights_for_mega_moe`,
`mega_moe_pre_dispatch` and `fp4_fp4_mega_moe`. The strategy uses
`preserve_bf16=True` and separate reduction (`fuse_reduce=False`). Selecting
`DSV4_PPU_MEGAMOE_PACKAGE=deep_gemm_mega` only changes the imported module;
it does not install a package or establish runtime compatibility.

The slim source baseline is RTP-LLM
[`70d80563221682bd2b5b008074922b38e7f7df37`](https://github.com/yukavio/rtp-llm/commit/70d80563221682bd2b5b008074922b38e7f7df37)
and DeepGEMM
[`b20e8efa892625c2ea867a447f1bd8ac50c7d9f3`](https://github.com/yukavio/DeepGEMM/commit/b20e8efa892625c2ea867a447f1bd8ac50c7d9f3).
Build the optional MegaMoE package from that DeepGEMM revision. Its BM64 LUT
is in `configs/dsv4_flash_bm64`; `SAIL_DEEPGEMM_LUT_PATH` can select that
directory at runtime. The regular SDK wheel is not evidence that this
optional API is available. This migration does not build or replace that wheel.

The slim baseline removes dense projection tiling, standalone HC post tiling,
staged shared experts and streamed MegaMoE. Their implementations and dedicated
tests are removed here too; attention tiles and the fused MoE-sum/HC update remain.

## Source features not admitted under the shared-code boundary

The source PR's `DSV4_PPU_PREFIX_TAIL_ALIGN` scheduler option and device-only
graph replay are runtime optimizations, not test-only code. The current
scheduler and CUDA Graph runner have no platform extension point for these
changes. They remain unported: enabling either would require changing a
shared execution path. No unused PPU callback or scheduler helper is added
to pretend that they are available.

## Test-only material

Historical LM-head/sampling benchmarks, projection benchmark results and
the D64 MTP-depth-two tuning profile are not runtime implementations and
are not migrated. Existing regression and numerical tests for retained
optional features remain necessary; a `manual` Bazel tag means the test
requires explicit execution, not that its feature is test-only. M890P
numerical tests must run on that hardware before production enablement.
