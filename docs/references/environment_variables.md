# Environment Variables

## DeepSeek-V4 on SM120

Set these variables before starting the server. Kernel modules may read them
at import or weight initialization. CLI-backed settings follow the precedence
in [server arguments](../backend/server_arguments.md). Kernel controls are
read directly and are not automatically converted to command-line arguments.

| Variable | Default | Scope and behavior |
| --- | --- | --- |
| `DSV4_STARTUP_REAL_WARMUP` | `true` | Additional real-request warmup for DeepSeek-V4 PREFILL when ordinary warmup is enabled. CLI values override this default. |
| `DSV4_MOE_STRATEGY` | unset | Compatibility alias for `sm120_fused_moe` or `sm120_decode` when the public MoE strategy is `auto`. Prefer `--moe_strategy`. |
| `DSV4_INDEXER_CACHE_MODE` | `fp8` | Only FP8 indexer cache storage is supported. Other values fail validation. |
| `DSV4_MOE_CHUNK_TOKENS` | `4096` for SM120 decode | Positive capacity for local expert computation chunks after eager all-to-all dispatch. It does not change the number of collectives per layer. |
| `DSV4_MOE_RETURN_DTYPE` | `mxfp8` | Intermediate MoE return representation: `mxfp8`, `bf16` or `fp32`. Changing it affects numerical and communication behavior and requires revalidation. |

SM120 decode requires eager execution, a separate DECODE role, TP1 and
DP=EP=WORLD. Every rank participates even when idle. The aggregate dispatch
limit is 65536 tokens across all EP ranks per step; exceeding it fails before
payload allocation. Size per-rank batch limits accordingly, including MTP
verification rows. Local expert chunking does not raise this transport limit.

Use the FP8 indexer with CP8+EP8 prefill and DP8+EP8 decode on two nodes.
CUDA-graph decode, FP4 indexer caches and optional SM120 kernel tuning paths
are not part of this baseline. Performance must be measured on the chosen
provider packages and deployment configuration.
