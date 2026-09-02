# ServerArgs

This page lists server arguments used to configure the behavior and performance of the language model server via command line. These parameters allow users to customize key server functionalities, including model selection, parallel strategies, memory management, and optimization techniques.

## Parallelism and Distributed Setup Configuration

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--worker_info_port_num` | Stride between port **bases** for each `rank_id`: `base = start_port + rank_id * worker_info_port_num`. Offsets under each base include RPC, HTTP, DashSc gRPC (`base + 8`), etc. **Breaking change:** default was **8**, now **9**. Services with DashSc gRPC enabled require this value to be at least `9`; multi-rank deployments that relied on the old stride must re-check ports. See [breaking-changes.md](../release/breaking-changes.md). | 9 |
| `--tp-size` | Specifies the tensor parallelism degree. | None |
| `--ep-size` | Defines the number of model instances for expert parallelism. | None |
| `--dp-size` | Sets the number of replicas or group size for data parallelism. | None |
| `--world-size` | Total number of GPUs used in distributed setup (WORLD_SIZE = TP_SIZE * DP_SIZE). | None |
| `--world-rank` | Global unique ID of the current process/GPU in the distributed system. | None |
| `--local-world-size` | Number of GPU devices used on the current node. | None |
| `--enable_ffn_disaggregate` | Enables FFN disaggregation feature to separate attention and feed-forward network computations for performance optimization. | None |

## Concurrency Control

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--concurrency-with-block` | Controls blocking behavior for concurrent requests. | False |
| `--concurrency-limit` | Maximum number of concurrent requests allowed by the system. | 32 |

## [Attention Optimization](./attention_backend.md)

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--enable_fmha` | Enables Fused Multi-Head Attention (FMHA) feature. | True |
| `--enable_flashinfer_trtllm_gen` | Enables FlashInfer TRT-LLM Gen attention on SM100. | True |
| `--enable_flashinfer_trt_fmha_v2` | Enables FlashInfer TRT-LLM FMHA v2 contiguous prefill. | True |
| `--enable_paged_flashinfer_trt_fmha_v2` | Enables FlashInfer TRT-LLM FMHA v2 paged prefill. | True |
| `--enable_open_source_fmha` | Enables open-source FMHA implementation. | True |
| `--enable_paged_open_source_fmha` | Enables Paged open-source FMHA implementation. | True |
| `--disable_flashinfer_native` | Disables FlashInfer native attention backends. | False |
| `--disable_flashinfer_hybrid_prefill` | Disables FlashInfer native Hybrid Prefill implementation. | True |
| `--enable_xqa` | Enables XQA feature (requires SM90+ GPU). | True |

### Removed FMHA options

The following legacy options and environment variables were removed and must no longer be used:

| Removed CLI option | Removed environment variable | Replacement |
|--------------------|------------------------------|-------------|
| `--enable_trt_fmha` | `ENABLE_TRT_FMHA` | `--enable_flashinfer_trt_fmha_v2` / `ENABLE_FLASHINFER_TRT_FMHA_V2` for contiguous prefill |
| `--enable_paged_trt_fmha` | `ENABLE_PAGED_TRT_FMHA` | `--enable_paged_flashinfer_trt_fmha_v2` / `ENABLE_PAGED_FLASHINFER_TRT_FMHA_V2` for paged prefill |
| `--enable_trtv1_fmha` | `ENABLE_TRTV1_FMHA` | None; select another supported attention backend |
| `--disable_flash_infer` | `DISABLE_FLASH_INFER` | To preserve global disable behavior, set `--disable_flashinfer_native=true`, `--enable_flashinfer_trtllm_gen=false`, `--enable_flashinfer_trt_fmha_v2=false`, and `--enable_paged_flashinfer_trt_fmha_v2=false` |

## KV Cache Configuration

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--reuse-cache` | Activates KV Cache reuse mechanism. | False |
| `--multi-task-prompt` | Multi-task prompt file path. | None |
| `--multi-task-prompt-str` | Multi-task prompt JSON string. | None |

## Hardware/Kernel Optimization

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--deep-gemm-num-sm` | Number of SMs used for DeepGEMM. | None |
| `--arm-gemm-use-kai` | Enables KleidiAI support for ARM GEMM. | False |
| `--enable-stable-scatter-add` | Enables stable scatter add operation. | False |
| `--enable-multi-block-mode` | Enables multi-block mode for MMHA. | True |
| `--rocm-hipblaslt-config` | hipBLASLt GEMM configuration file path. | gemm_config.csv |
| `--ft-disable-custom-ar` | Disables custom AllReduce implementation. | True |

## Device Resource Management

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--device-reserve-memory-bytes` | Amount of GPU memory to reserve (bytes). | 0 |
| `--host-reserve-memory-bytes` | Amount of CPU memory to reserve (bytes). | 4GB |
| `--overlap-math-sm-count` | Number of SMs for compute-communication overlap optimization. | 0 |
| `--overlap-comm-type` | Compute-communication overlap strategy type. | 0 |
| `--m-split` | M_SPLIT parameter for device operations. | 0 |
| `--enable-comm-overlap` | Enables compute-communication overlapping execution. | True |
| `--enable-layer-micro-batch` | Enables layer-level micro-batching. | 0 |
| `--not-use-default-stream` | Do not use default CUDA stream. | False |

## DeepEP Configuration

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--use-deepep-moe` | Enables DeepEP for MoE processing. Single EP shoude set be false  | False |
| `--use-deepep-internode` | Enables inter-node communication optimization. | False |
| `--use-deepep-low-latency` | Enables DeepEP low-latency mode. | True |
| `--use-deepep-p2p-low-latency` | Enables P2P low-latency mode. | False |
| `--deep-ep-num-sm` | Number of SMs for DeepEPBuffer. | 0 |

## EPLB Configuration

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--eplb_mode` | EPLB mode | "NONE" |
| `--balance_method` | EPLB load balancing method. | "mix" |
| `--redundant_expert` | Number of redundant experts. | 0 |
| `--eplb_update_time` | EPLB execution cycle. | 5000 |
| `--eplb_balance_layer_per_step` | Number of layers updated per EPLB update. | 1 |
| `--eplb_force_repack` | Globally repack EPLB experts. | False |
| `--eplb_stats_window_size` | EPLB statistics window size. | 10 |
| `--eplb_control_step` | (DEBUG) EPLB synchronization control parameter cycle. | 100 |
| `--eplb_test_mode` | (DEBUG) Enables ExpertBalancer test mode | False |
| `--fake_balance_expert` | (DEBUG) Enables expert pseudo-balancing mechanism. | False |

## Sampling Configuration

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--max-batch-size` | Override system maximum batch size. | 0 |
| `--enable-flashinfer-sample-kernel` | Enables FlashInfer sampling kernel. | True |

## Logging & Profiling

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--ft-nvtx` | Enables NVTX performance profiling. | False |
| `--py-inference-log-response` | Logs inference response content. | False |
| `--trace-memory` | Enables memory tracing. | False |
| `--trace-malloc-stack` | Enables malloc stack tracing. | False |
| `--enable-device-perf` | Collects device performance metrics. | False |
| `--ft-core-dump-on-exception` | Generates core dump on exception. | False |
| `--ft-alog-conf-path` | Log configuration file path. | None |
| `--log-level` | Log level (ERROR/WARN/INFO/DEBUG). | INFO |
| `--gen-timeline-sync` | Collects Timeline analysis data. | False |
| `--torch-cuda-profiler-dir` | Torch Profiler output directory. | "" |

## Speculative Decoding

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--sp-model-type` | Specifies draft model type (e.g.  "deepseek-v3-mtp") | "" |
| `--sp-type` | Controls speculative sampling type ("vanilla" disables, "mtp" enables) | "" |
| `--sp-min-token-match` | Minimum token match length | 2 |
| `--sp-max-token-match` | Maximum token match length | 2 |
| `--tree-decode-config` | Tree decode mapping configuration file | "" |
| `--gen-num-per-cycle` | Maximum number of tokens generated per cycle | 1 |
| `--force-stream-sample` | Forces streaming sampling | False |
| `--force-score-context-attention` | Forces context attention scoring | True |

## RPC and Service Discovery

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--use-local` | Uses local service discovery | False |
| `--remote-rpc-server-ip` | Remote RPC server address | None |
| `--decode-cm2-config` | Decode service discovery configuration | None |
| `--remote-vit-server-ip` | Remote ViT server address | None |
| `--multimodal-part-cm2-config` | Multimodal service discovery configuration | None |

## Cache Store

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--cache-store-rdma-mode` | Enables RDMA mode | False |
| `--wrr-available-ratio` | WRR load balancing availability threshold | 80 |
| `--rank-factor` | WRR ranking factor (0=KV_CACHE usage, 1=in-flight requests) | 0 |

## Scheduler Configuration

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--use-batch-decode-scheduler` | Enables batch decode scheduler | False |
| `--max-context-batch-size` | Maximum context batch size | 1 |
| `--scheduler-reserve-resource-ratio` | Reserved resource percentage | 5 |
| `--batch-decode-scheduler-batch-size` | Decode batch size | 1 |

## Load Balancing and Performance Optimization Configuration

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--load-balance` | Enables dynamic load balancing | False |
| `--step-records-time-range` | Performance record retention time window (microseconds) | 60000000 |
| `--step-records-max-size` | Maximum performance record count | 1000 |
| `--disable-pdl` | Disables PDL feature | False |

## 3FS Configuration

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--enable-3fs` | Enables 3FS for managing KVCache | False |

## Model Adaptation

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--max-lora-model-size` | Maximum size limit for LoRA models | -1 |

## System Debugging

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--gen-timeline-sync` | Collects Timeline analysis data | False |
| `--torch-cuda-profiler-dir` | Torch Profiler output directory | "" |

## Load Config

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--load_method` | Specify the weight loading method.<br>Options: auto, fastsafetensors, scratch (`LOAD_METHOD`) | auto |
| `--force_cpu_load_weights` | Load weights on CPU to reduce device memory usage (`FORCE_CPU_LOAD_WEIGHTS`) | False |
| `--loader_recycle_handles` | ROCm + safetensors only: close consumed main-model shard handles to release mmap memory. Requires layer-numbered tensors and copies safetensors data out before closing; no effect on fastsafetensors, ViT, EPLB, or .bin weights. (`LOADER_RECYCLE_HANDLES`) | True |
| `--moe_pure_tp_preshard` | Disabled by default. Set true to pre-shard supported Qwen3-Next / Qwen3.5 MoE and offline FP8 weights under pure TP (`tp>1, dp=1, ep=1`) before device copy. Unsupported sources or layouts warn and use legacy full reads. (`MOE_PURE_TP_PRESHARD`) | False |

### FastSafeTensors loader configuration

When `LOAD_METHOD=fastsafetensors`, or when the default `auto` mode selects the
FastSafeTensors path, RTP-LLM uses the config-driven `AutoLoader`. RTP checks
the installed package capabilities before loading: the full capability set uses
bounded `per-expert` delivery, a package without `stacked_moe_tensors` (or the
legacy `dim0_split_templates` alias) falls
back to the higher-memory `full-stacked` compatibility path, and a package
without `local_copyout_filter` continues with full materialization and RTP
consumer-side filtering. A missing package/`AutoLoader`, an import/ABI failure,
an unmet AUTO prerequisite, or an insufficient AUTO memory preflight falls back
to `scratch`. Explicit `LOAD_METHOD=fastsafetensors` treats `per-expert` as a
user override and skips the memory preflight; explicit `full-stacked` keeps the
preflight when RTP detects a raw stacked MoE checkpoint, because that path
materializes a complete stacked tensor. Dense and already per-expert checkpoints
do not run this stacked-tensor preflight. Clear import, API, constructor, and ABI
compatibility failures fall back to scratch in either entry mode, while
checkpoint/data errors remain fail-fast.
The two optional keywords control independent optimizations:

| Capability | Present | Missing |
|---|---|---|
| `local_copyout_filter` | rank-local copy-out | full materialization, RTP consumer filtering |
| `stacked_moe_tensors` (legacy alias: `dim0_split_templates`) | bounded `per-expert` MoE delivery | `full-stacked` MoE delivery |

When a degraded FastSafeTensors mode remains usable, RTP logs
`requested_mode`, `effective_mode` and `degraded_reason`; rank-local copy-out
degradation uses the same fields with `effective_mode=consumer-filter`. Every
scratch fallback contains `falls back to scratch`; package absence is INFO and
normal AUTO prerequisite rejection is also INFO. Compatibility, capability and
memory-preflight fallback causes are WARNING. CI or image builds that require
both optimizations must install the matching wheel and treat a missing
capability as a packaging failure. Set
`RTP_LLM_EXPECT_FASTSAFETENSORS_TIER=per-expert` for the installed wheel
contract test to turn a lower tier into a test failure; supported tiers are
`scratch`, `consumer-filter`, `full-stacked`, and `per-expert`.

Pass the standard fastsafetensors configuration as either an inline JSON string
or a JSON file path. The installed FastSafeTensors version defines the precise
configuration defaults and precedence:

```bash
# Inline JSON string; progress is controlled by the upstream parallel config.
export FASTSAFETENSORS_CONFIG_JSON='{"loader":"base","base":{"copier_type":"nogds"},"parallel":{"use_tqdm_on_load":true}}'

# JSON file path; the file contains the same JSON object
export FASTSAFETENSORS_CONFIG=/path/to/fastsafetensors.json
```

The same configuration also affects `auto` selection. RTP reads
`estimated_peak_device_bytes` from the installed package; missing or invalid
values use the historical `3 × max checkpoint shard` estimate. RTP then adds a
2 GiB empirical reserve for TensorCollector inputs that overlap final weight
materialization. This reserve is an integration estimate pending calibration
with stacked-MoE peak-memory measurements. Larger buffers, queues or producer
counts can raise `transient_mem` enough for `auto` to choose `scratch`. Inspect
the `fastsafetensor memory check` log and its `enough` field; this log is not
emitted for explicit per-expert loading because that mode skips preflight.

For compatibility with existing development environments,
`FASTSAFETENSORS_NOGDS=1` remains supported. Before memory preflight or
constructing `AutoLoader`, RTP-LLM overrides `FASTSAFETENSORS_CONFIG_JSON`
process-wide with
`{"loader":"base","base":{"copier_type":"nogds"}}`. This compatibility switch
therefore remains in effect for subsequent loaders in the same process. Prefer
one of the standard configuration variables above for new deployments. When
`FASTSAFETENSORS_CONFIG` is also set, the final precedence remains an upstream
package contract; current pinned wheels prefer the inline JSON value.

Stacked MoE checkpoints use bounded-memory per-expert delivery by default: the
source rank slices the stacked tensor first, then every rank broadcasts one
expert at a time. The higher-memory full-stacked path is a temporary
compatibility rollback for wheels or deployments that cannot use the bounded
split path, and it may also be used for controlled performance comparisons:

```bash
export RTP_FASTSAFETENSORS_STACKED_MOE_MODE=full-stacked
```

The accepted values are `per-expert` (default) and `full-stacked`; an empty
value also selects the default. When the checkpoint actually contains raw
stacked MoE tensors, `full-stacked` adds a conservative extra shard to the
FastSafeTensors memory preflight because it materializes a whole stacked tensor
before RTP clones expert slices. Dense or already per-expert checkpoints do not
pay this add-on. A passive downgrade logs a warning with
`degraded_reason`; an explicit request is reported as the selected mode. The
additional warning is emitted only when the checkpoint actually contains raw
stacked MoE tensors. Use `LOAD_METHOD=scratch` as the more conservative
rollback. This transitional RTP switch only selects stacked MoE delivery.
Bucket size, copier/backend, queue depth, producer count, loading progress and
tensor ordering are otherwise owned by the installed FastSafeTensors
configuration; rank-local copy-out is supplied by RTP's local checkpoint-key
predicate.

`RTP_FASTSAFETENSORS_STACKED_MOE_MODE` is a transitional, environment-only
switch: it has no command-line flag, is not shown by `--help`, and is not part
of the startup config dump. It is read only when the FastSafeTensors path is
considered. Values are case-sensitive and use a hyphen; any non-empty value
other than `per-expert` or `full-stacked` raises `ValueError` during
FastSafeTensors selection. It has no effect for `LOAD_METHOD=scratch` or for
weights that cannot use the FastSafeTensors path.
