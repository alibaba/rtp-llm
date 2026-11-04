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
