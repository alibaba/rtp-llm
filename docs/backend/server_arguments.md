# ServerArgs

This page lists server arguments used to configure the behavior and performance of the language model server via command line. These parameters allow users to customize key server functionalities, including model selection, parallel strategies, memory management, and optimization techniques.

## Parallelism and Distributed Setup Configuration

| Arguments | Description | Defaults |
|-----------|-------------|----------|
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
| `--enable_xqa` | Enables XQA feature (requires SM90+ GPU). | True |

### Removed FMHA options

The following legacy options and environment variables were removed and must no longer be used:

| Removed CLI option | Removed environment variable | Replacement |
|--------------------|------------------------------|-------------|
| `--enable_trt_fmha` | `ENABLE_TRT_FMHA` | `--enable_flashinfer_trt_fmha_v2` / `ENABLE_FLASHINFER_TRT_FMHA_V2` for contiguous prefill |
| `--enable_paged_trt_fmha` | `ENABLE_PAGED_TRT_FMHA` | `--enable_paged_flashinfer_trt_fmha_v2` / `ENABLE_PAGED_FLASHINFER_TRT_FMHA_V2` for paged prefill |
| `--enable_trtv1_fmha` | `ENABLE_TRTV1_FMHA` | None; select another supported attention backend |
| `--disable_flash_infer` | `DISABLE_FLASH_INFER` | To preserve global disable behavior, set `--disable_flashinfer_native=true`, `--enable_flashinfer_trtllm_gen=false`, `--enable_flashinfer_trt_fmha_v2=false`, and `--enable_paged_flashinfer_trt_fmha_v2=false` |

#### Compatibility and rollout contract

This migration intentionally establishes a new FMHA configuration contract and is a required breaking change. The embedded TensorRT cubin implementations and their configuration fields were removed in favor of FlashInfer TRT-LLM FMHA v2. The new contiguous and paged implementations have different support constraints and are not aliases for the removed implementations, so the legacy CLI options and environment variables are deliberately not preserved, translated, or warned on. FlashInfer TRT-LLM FMHA v2 requires CUDA 12.9 or later and an SM90 or SM12x GPU.

RTP-LLM deployments are released as versioned, immutable container images, with the matching configuration delivered as part of the same deployment revision. Image and configuration updates therefore form one atomic rollout unit; reusing an older FMHA configuration with a newer image is not a supported deployment mode. Rollback likewise restores the previous image and its configuration together. This image-scoped migration model has also been used for previous backend configuration changes, so no cross-version compatibility aliases are required for this removal.

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
| `--load_method` | Specify the weight loading method.<br>Options: auto, fastsafetensors, scratch | auto |
