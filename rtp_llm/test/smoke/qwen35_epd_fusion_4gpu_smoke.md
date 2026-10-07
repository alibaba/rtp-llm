# Manual Qwen3.5 EPD / PD Fusion harness

This harness runs Qwen3.5-397B-A17B-FP8 with four PD Fusion ranks and optionally one or two separate Encoder GPUs. It is an opt-in experiment tool for the configured CUDA 13 / SM10x environment, not a production deployment preset.

## Dependencies and topology

The complete harness requires both the scheduling-observation and global-cadence commits: its source snapshot includes coordinator files even when coordination is disabled. It also requires the passive GPU lock implementation in `test/utils/device_resource.py`.

| Mode | Encoder GPUs | PD GPUs | Total GPUs | Bazel target |
|---|---|---|---|---|
| Text PD Fusion | none | four | 4 | `qwen35_pd_fusion_4gpu_smoke` |
| Video with one Encoder | one | four distinct GPUs | 5 | `qwen35_e1pd4_smoke` |
| Video with two Encoders | two | four distinct GPUs | 6 | `qwen35_e2pd4_smoke` |

Use `--encoder-proxy 1` for the proxy/worker topology. `--encoder-dp2` remains an alias for compatibility; the number of workers comes from `--encoder-gpus`, not the option name. A single worker sets `RTP_VIT_SINGLE_WORKER_PROXY=1` explicitly. Normal server startup retains its existing single-worker standalone default.

## Launch preparation

Run the repository's test-execution precheck and use the designated workspace cache and GPU locks. The outer launcher expects the internal workspace layout and its precheck script. It supplies CUDA 13, SM10x, compute capability 10.3 and internal cache defaults; review generated commands before running on a different host. `--bazel-option` can append site-specific Bazel options.

From `github-opensource/`, prepare a one-Encoder run with explicit paths and an unused output directory:

```bash
python rtp_llm/test/smoke/qwen35_epd_fusion_4gpu_smoke.py \
  --model-dir /path/to/model \
  --cache-root /path/to/dedicated/bazel-cache \
  --data-dir /path/to/qwen35_e2p4d2_data \
  --gpus 4,5,6,7 --encoder-gpus 0 --encoder-proxy 1 \
  --encoder-profile candidate-a \
  --output /path/to/new-run
```

Without `--execute`, the launcher prepares the command and evidence snapshot. Inspect them before adding `--execute`. For two Encoders, change only `--encoder-gpus` to `0,1`; the target and GPU count change together. GPU lists must be unique and disjoint. Passive locking refuses busy or unqueryable GPUs without signalling unknown processes.

The Encoder profile controls concurrency, preprocessing and GPU batching. `--encoder-rdma-pool-bytes` is a per-worker allocation cap, not measured occupancy. The harness records actual service env/argv and requires evidence from every configured Encoder worker.

## Scheduling and measurements

Coordination defaults to `--coord-mode off`. The experimental global cadence mode requires `--coord-mode cadence`, `--scheduler-policy prefill-first`, `--schedule-trace 1` and a unique shared `--trace-run-id`. `--decode-prefill-ratio N` controls the target decode rounds after a real prefill. The coordinator supports only single-host synchronous TP1/DP4/EP4 execution, rejects unsupported modes, and terminates the group on control failure.

`--coord-checks 1` runs coordination correctness checks; peer-exit fault injection is separately enabled by `--coord-fault peer-exit`. Do not mix fault injection with performance measurements.

For steady video load, enable the Encoder proxy, set `--client-mode measured`, choose `--steady-concurrency`, and configure a sufficient `--rank-concurrency`. Warmup, window duration, window count and process count are explicit options. Use natural output when comparing equal work; input frames/tokens, output lengths, failures, queue inventory, TTFT, throughput and GPU memory all belong in the comparison. A shortened output is not evidence of equal-work improvement.

`RTP_STEP_MEASUREMENT`, scheduling trace and Graph/FP8 execution logs are diagnostic opt-ins. Host executor timings are not pure GPU kernel timings. `pdfusion_schedule_trace.py` analyzes trace alignment and distinguishes local observations from coordinated control epochs.

## Regression tests and limits

The CPU harness target is `//rtp_llm/test/smoke:qwen35_epd_fusion_4gpu_harness_test`. Trace analysis has its own `pdfusion_schedule_trace_test` target. The live multi-GPU targets are manual and must use the repository GPU-lock workflow.

Historical single-Encoder short runs used five GPUs and produced throughput close to the two-Encoder baseline, with greater tail latency and higher per-Encoder memory. These results do not establish strictly unchanged throughput or long-term stability. A later two-Encoder FP8 long run failed with grid sync timeout before a valid long comparison; reducing Encoder count has not been established as the cause. The separate video-decoder pixel-consistency issue is also unresolved. Preserve full run evidence and exclude failed windows from performance conclusions.
