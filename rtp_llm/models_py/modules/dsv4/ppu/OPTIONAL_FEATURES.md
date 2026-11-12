# DeepSeek V4 Flash PPU optional features

These options belong to the Flash PPU model/provider. They do not change
the shared GPU model, loader, scheduler or CUDA Graph runner. Presence in
this table describes code availability, not hardware qualification.

| Function | Selection | Status |
| --- | --- | --- |
| Decode operator fusion, MTP input fusion, ordered C4 queries | `DSV4_PPU_DECODE_FUSIONS=1` | Optional runtime path, default off |
| MegaMoE | `DSV4_PPU_DECODE_MOE_BACKEND=megamoe` | Optional runtime path; compatible native package required |
| MegaMoE overlap | `DSV4_PPU_MEGAMOE_MODE=plain|streamed|staged` | Default `plain`; `streamed` needs the packaged kernel and falls back on unsupported tiles; `staged` is experimental |
| Decode metadata graph and shared RoPE tables | Always on for q_len=1 decode | Captured on first preparation; replay keeps the runner's host mirrors |
| MTP metadata graph (q_len 2 through 4) and fused compressor state slots | `DSV4_PPU_MTP_METADATA_GRAPH=1` | Optional runtime path, default off |
| Inactive-route masking | `DSV4_PPU_MTP_MASK_INACTIVE=1` with the MTP metadata graph | Optional runtime path, default off |
| Prefix reuse | `DSV4_PPU_PREFIX_REUSE=1` with reuse-cache configuration | Experimental runtime path; correctness still needs PPU qualification |

For q_len=1 decode the PPU provider always builds the metadata graph with
shared RoPE tables; there is no separate reference metadata mode to select
at startup. The graph is captured on its first preparation, before replay
reuses it, and the shared runner keeps its normal host mirrors. q_len 2
through 4 needs `DSV4_PPU_MTP_METADATA_GRAPH=1`; without it the framework
metadata factory is used for verify and draft catch-up, and turning the
option off does not change q_len=1 decode.
MegaMoE requires a compatible PPU DeepGEMM build exposing
`get_symm_buffer_for_mega_moe`, `transform_weights_for_mega_moe`,
`mega_moe_pre_dispatch` and `fp4_fp4_mega_moe`. The strategy uses
`preserve_bf16=True` and separate reduction (`fuse_reduce=False`). Selecting
`DSV4_PPU_MEGAMOE_PACKAGE=deep_gemm_mega` only changes the imported module;
it does not install a package or establish runtime compatibility. Exactly
one overlap mode is active per process: streamed dispatch and staged shared
expert are alternatives, not composable.
