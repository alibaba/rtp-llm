# Environment Variables

## DeepSeek-V4 on SM120

Set these variables before starting the server. Some kernel modules read them
at import or weight initialization; changing the process environment after
startup is not a supported reconfiguration mechanism. CLI-backed settings
follow the precedence described in [server arguments](../backend/server_arguments.md).
Kernel flags in the tables below are read directly by their consumers and
are not automatically converted to command-line arguments.

Eligible SM120 prefill metadata sharing, indexer-tail overlap and CP4 head
prebuild are selected automatically. Head prebuild requires one request,
4096 global tokens, 1024 local tokens and unsharded KV cache; other geometries
use eager preparation. Workspace capacity stays fixed at its startup reservation.
The `sm120_decode` strategy releases allocator cache at prefill entry above
8192 tokens and permits one retry after an eligible sparse-attention OOM.

### Runtime Controls

| Variable | Default | Scope and behavior |
| --- | --- | --- |
| `DSV4_STARTUP_REAL_WARMUP` | `true` | Additional real-request warmup for DeepSeek-V4 PREFILL when ordinary warmup is enabled. The CLI option overrides this default. |
| `DSV4_MOE_STRATEGY` | unset | Compatibility alias for `sm120_fused_moe` or `sm120_decode` when the public MoE strategy is `auto`. Prefer `--moe_strategy`. |
| `DSV4_MOE_CHUNK_TOKENS` | `4096` for SM120 decode | Positive chunk capacity for eager all-to-all decode MoE. The SM120 decode executor defaults it to `4096`, while the model-side `moe_chunk_tokens_from_env` reader defaults to `16384` and uses the value to cap `max_tokens_per_rank` for PREFILL and PDFUSION. Validated at weight initialization. This is not a universal default for other MoE executors. |
| `DSV4_MOE_RETURN_DTYPE` | `fp32` | Intermediate MoE return representation, shared by eager and fixed EP execution. Accepts `bf16`, `mxfp8` or `fp32`; changing it changes numerical and communication behavior. |
| `DSV4_GATE_OUTPUT_FP32` | `0` | `1` retains FP32 MoE gate output instead of BF16. Changes routing numerics. |
| `DSV4_SM120_WOA_EINSUM` | `0` | `1` selects DeepGEMM WOA einsum on SM120 instead of FlashInfer grouped GEMM, including matching JIT warmup. Requires a provider supporting that kernel. |

### Optional Kernel Paths

These controls select narrower execution paths. Validate the actual topology,
numerical results and performance before enabling them. A CP8 serving run does
not exercise CP4-only compact transport or asynchronous head prebuild.

| Variable | Default | Scope and interaction |
| --- | --- | --- |
| `DSV4_CP_COMPACT_COMPRESSOR` | `0` | Enables the specialized compact CP4 compressor path with cached geometry plans and deferred verification where supported. Unsupported geometry falls back to the ordinary path. |
| `DSV4_FP8_GATHER` | `0` | Master switch for FP8 CP transport. `DSV4_FP8_GATHER_Q` and `DSV4_FP8_GATHER_KV` each default to `1` under the master switch and may independently disable a site with `0`. With KV transport enabled the persistent KV-pool write consumes the gathered `kv_full` after its e4m3 round-trip, so pool rows no longer match pre-gather rank-local values byte for byte. |
| `DSV4_INDEXER_CACHE_MODE` | `fp8` | Experimental `fp4` mode requires DeepSeek-V4, SP0 or the three-slot DSpark configuration, prefix reuse disabled, and 256-token allocation and kernel blocks. It is not a general FP8-cache replacement. |
| `DSV4_FP4_PAGED_BACKEND` | `triton` | `triton` or `native` paged indexer scoring for FP4 cache mode. Both read the same packed cache layout. Resolved at indexer construction. |
| `DSV4_SM120_INDEXER_FUSED_GATHER` | `0` | `1` uses fused gathering in the SM120 paged indexer fallback for contiguous CUDA cache pools. |
| `DSV4_SM120_MHC_COLUMN_TILES` | `0` | `1` selects the column-tiled mHC prenorm kernel for supported SM120 decode shapes. |
| `DSV4_SM120_PACK_DECODE_SLOTS` | `0` | `1` packs decode cache slots for the supported SM120 sparse-attention route. |
| `DSV4_SM120_PAGED_CACHE_PREFILL` | `0` | `1` enables the SM120 paged-cache prefill route. Validate cache-hit indexing before enabling it. |
| `DSV4_SM120_PREFILL_INDEX_ADAPTER` | `off` | `ready` enables the eligible prefill index producer/consumer adapter. Read at import. |
| `DSV4_SM120_PREFILL_SPLIT_FUSION` | `0` | `1` enables fused prefill index splitting; accepts only `0` or `1`. Read at import. |

### Engine Scheduling

| Variable | Default | Behavior |
| --- | --- | --- |
| `RTP_LLM_DECODE_FIXED_BS` | `0` | Selects fixed decode padding when positive; zero disables the override. Read at engine initialization, and only in the SM120 aligned DP decode profile (`sm120_decode` MoE with CUDA graph, TP=1, DP=EP=world size above 1). Must be at least `max_generate_batch_size` when enabled. Other engines log a warning and ignore it. |
| `RTP_LLM_DP_FAKE_STREAM_WAIT_MS` | `10` | Idle-DP fake-stream wait in milliseconds, clamped to 1 through 10. Reducing it trades CPU activity for scheduling latency. Read once by the scheduler. |
| `RTP_LLM_STOP_TIMEOUT_MS` | unset | Positive values override, in milliseconds, how long stop waits for in-flight RPC requests before calling gRPC `Shutdown()`. Otherwise `SHUTDOWN_TIMEOUT` supplies seconds, defaulting to 600. Neither value bounds engine thread joins, and gRPC `Shutdown()` itself has no deadline. |

Use RTP's timeline profiling controls for performance analysis. Removed
prefill print/dump probes and import-path overrides are not deployment settings.
