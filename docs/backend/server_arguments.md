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
| `--eplb_mode` | `NONE`, `STATS`, `EPLB`, or `ALL`. Set the mode at startup; online changes are disabled. `STATS` collects counts without migration. Both `EPLB` and `ALL` collect counts and rearrange experts. With EP > 1, enabled runtimes prepare shuffle resources and warm up planning at startup, including `STATS` and a paused update interval. This preserves the initial weights, routing maps and statistics. Monitoring is configured independently. | "NONE" |
| `--redundant_expert` | Number of redundant experts. | 0 |
| `--eplb_update_time` | Rearrangement interval in eligible model steps. `0` disables rearrangement; the first update runs after `max(1, interval // 4)` eligible steps. | 0 |
| `--eplb_stats_window_size` | Rearrangement count window. Must be non-negative; a positive update interval requires `1 <= window <= interval`. | 100 |
| `--eplb_rearrange_after_prefill` | On PDFUSION workers, collect prefill counts and rearrange on the next decode step. Otherwise PDFUSION prefill is excluded from rearrangement scheduling. Dedicated P/D workers retain their own eligible step schedule. | False |
| `--eplb_rearrange_p2p_groups` | Positive P2P bucket count for rearrangement outside a shared NVLink domain. | 1 |
| `--eplb_metrics_report_interval` | Report `moe_lb` every N target-model prefill/decode forwards (target verify during speculative decoding); draft models are excluded. `0` disables collection; `-1` selects 100 for PREFILL and 1000 for DECODE or PDFUSION. Other values must be positive. Unsupported devices or topologies (including PP, FFN disaggregation and MoE context parallelism without EP) log a warning and skip monitoring. Dense models retain empty-window `moe_lb=1` without a collective or device-to-host copy. This does not relax dynamic EPLB requirements. | -1 |
| `--eplb_metrics_snapshot_max_files` | Snapshots retained per model instance in circular slots under `LOG_PATH/eplb_snapshot` (`LOG_PATH` defaults to `logs`). `0` disables snapshots. | 0 |
| `--fake_balance_expert` | (DEBUG) Enables expert pseudo-balancing mechanism. | False |

Each rearrangement handles every layer in a homogeneous weight-layout group. The former CPU layer-selection and debug options (`balance_method`, `eplb_balance_layer_per_step`, `eplb_force_repack`, and `eplb_test_mode`) have been removed.

Set `EPLB_MODE` and `EPLB_UPDATE_TIME` at startup; automatic rearrangement follows that fixed configuration.

`moe_lb` follows DeepInfer: sum the maximum EP-rank load for each step and layer,
then divide by the sum of their mean EP-rank loads. `1.0` means balanced; larger
values indicate more imbalance. A window with no routed tokens reports `1.0`.
The model-level `rtp_llm_moe_lb` gauge and `[EPLB]` log use this same ratio.
Completed windows use a blocking GPU-to-CPU copy and synchronous gathering;
snapshot schema 4 stores per-step layer sums for independent recomputation.

## Sampling Configuration

| Arguments | Description | Defaults |
|-----------|-------------|----------|
| `--max-batch-size` | Override system maximum batch size. | 0 |

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
