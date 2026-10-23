# Manual Qwen3.5 EPD / PD Fusion harness

This harness runs Qwen3.5-397B-A17B-FP8 with two or four PD Fusion ranks and optional separate Encoder GPUs. It is an opt-in experiment tool for the configured CUDA 13 / SM10x environment, not a production deployment preset.

## Dependencies and topology

The complete harness requires both the scheduling-observation and global-cadence commits: its source snapshot includes coordinator files even when coordination is disabled. It also requires the passive GPU lock implementation in `test/utils/device_resource.py`.

| Mode | Encoder GPUs | PD GPUs | Total GPUs | Bazel target |
|---|---|---|---|---|
| Text PD Fusion | none | two or four | 2 or 4 | `qwen35_pd_fusion_4gpu_smoke` |
| Video with one Encoder | one | two distinct GPUs | 3 | `qwen35_e1pd2_smoke` |
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

Coordination defaults to `--coord-mode off`. The experimental global cadence mode requires `--coord-mode cadence`, `--scheduler-policy prefill-first`, `--schedule-trace 1` and a unique shared `--trace-run-id`. `--decode-prefill-ratio N` controls the target decode rounds after a real prefill. The coordinator supports single-host synchronous TP1/DP2/EP2 or TP1/DP4/EP4 execution, rejects unsupported modes, and terminates the group on control failure.

`--coord-checks 1` runs coordination correctness checks; peer-exit fault injection is separately enabled by `--coord-fault peer-exit`. Do not mix fault injection with performance measurements.

With `PDFUSION_CADENCE_ADAPTIVE=1`, each PD rank packs Prefill requests by
`sum(contextLength * currentBatchSize) < max_batch_tokens_size`, subject to total
in-flight concurrency and KV admission. `MAX_CONTEXT_BATCH_SIZE` is not an
independent request-count cap in this mode, even when explicitly set to 1 or 4.
It is normalized to total concurrency for capacity sizing. The original strict
token boundary and standalone long-request exception remain unchanged. Unknown
prefix hits are charged conservatively until the allocator determines reuse.

Ready requests can accumulate across Decode epochs while admission is allowed.
The packed budget includes already reserved and loading Prefill requests exactly
once. Loading requests execute only after becoming ready. An oversized candidate
can be skipped so a later short request fits; there is no wait to fill either a
request count or the token budget. All ranks share the Prefill phase, but their
real batch sizes may differ, including zero with a matching fake Prefill.

Set the token budget and omit `--max-context-batch-size`. The harness retains
32768 tokens by default, or 65536 with `--max-input-tokens 65536`. Without an
explicit token budget, adaptive model construction uses one maximum sequence's
capacity. Legacy cadence still requires a request limit of 1 and keeps padded
token accounting. Trace summaries include `prefill_batch_hist` and
`multi_rank_multi_request_prefill_steps`; compare equal work at fixed token
budgets using throughput, TTFT, decode latency and GPU memory.
For a text-only DP2/EP2 run, pass `--workload text --gpus 2,3` without Encoder
GPUs; the same correctness and steady-load entry points support DP4/EP4.

For steady video load, enable the Encoder proxy, set `--client-mode measured`, choose `--steady-concurrency`, and configure a sufficient `--rank-concurrency`. Warmup, window duration, window count and process count are explicit options. Use natural output when comparing equal work; input frames/tokens, output lengths, failures, queue inventory, TTFT, throughput and GPU memory all belong in the comparison. A shortened output is not evidence of equal-work improvement.

`RTP_STEP_MEASUREMENT`, scheduling trace and Graph/FP8 execution logs are diagnostic opt-ins. Host executor timings are not pure GPU kernel timings. `pdfusion_schedule_trace.py` analyzes trace alignment and distinguishes local observations from coordinated control epochs.

## Regression tests and limits

The CPU harness target is `//rtp_llm/test/smoke:qwen35_epd_fusion_4gpu_harness_test`. Trace analysis has its own `pdfusion_schedule_trace_test` target. The live multi-GPU targets are manual and must use the repository GPU-lock workflow.

Historical single-Encoder short runs used five GPUs and produced throughput close to the two-Encoder baseline, with greater tail latency and higher per-Encoder memory. These results do not establish strictly unchanged throughput or long-term stability. A later two-Encoder FP8 long run failed with grid sync timeout before a valid long comparison; reducing Encoder count has not been established as the cause. The separate video-decoder pixel-consistency issue is also unresolved. Preserve full run evidence and exclude failed windows from performance conclusions.

## Frozen single-Encoder performance preset

For the complete measured C512/N7 configuration (Encoder 1 GPU + PD Fusion 4 GPUs),
use [qwen35_e1pd4_perf_smoke](qwen35_e1pd4_perf_smoke.py). It includes the
required environment, the pinned DeepGEMM wheel and dependency checks.
