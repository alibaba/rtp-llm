# Single Encoder + PD Fusion 4 GPU performance smoke

`qwen35_e1pd4_perf_smoke.py` freezes the settings from the measured
`br-e1-n7-c512-01` run (2026-10-07, mean 4.286667 requests/s). It does not read
commands from `build_logs`. This is a performance reproduction preset, not a
short health check or a guarantee of that throughput on another revision.

Run inside `tuoyu.ty`, from the `epd-fusion` internal worktree:

```bash
cd /ssd/7/tuoyu.ty/workspace/RTP-LLM-epd-fusion
/opt/conda310/bin/python github-opensource/rtp_llm/test/smoke/qwen35_e1pd4_perf_smoke.py \
  --output "$PWD/build_logs/e1pd4-perf-$(date +%Y%m%d-%H%M%S)" \
  --execute
```

Omit `--execute` to print the complete Bazel command without acquiring GPUs,
creating output directories, loading the model, or requiring runtime assets.
The launcher runs the repository precheck, checks GPU availability, and uses
`//rtp_llm/test/utils:gpu_lock`. The dedicated manual target is
`//rtp_llm/test/smoke:qwen35_e1pd4_perf_smoke`; use the launcher so its environment,
resource paths, timeout, build flags and lock are all supplied together.
A new output directory is required; the trace ID is automatically unique.

## Frozen configuration

| Area | Setting |
| --- | --- |
| Topology | Encoder GPU 0, one proxy/one worker; PD GPUs 4,5,6,7, TP1/DP4/EP4 |
| Model/input | Qwen3.5-397B-A17B-FP8; existing NVDEC video fixture, FPS 6, 46 frames, 24,601 prompt tokens; thinking off |
| Output/cache | Natural output, max_tokens 4096; model/MM/URL reuse off; MM output CPU/GPU cache 0, hash cache 1 GiB |
| Encoder | candidate-a: concurrency 64, queue 512, preprocessing workers 8, GPU batch 4, batch wait 10 ms, MM queue 1024 |
| Transport | RDMA, pool cap 128 GiB per Encoder worker, read timeout 30,000 ms |
| PD scheduler | Ratio scheduler + global cadence N7; trace on; correctness fault injection off |
| Capacity | rank concurrency 256, automatic KV sizing, runtime reserve 57,344 MiB |
| Kernels | optimized mega_moe_fp8, decode graph + FP8 KV + native FP8 attention on; baseline fusion flags 0, native GDN, pre-kernel barrier 0 |
| Graph buckets | 1,2,4,8,16,32,64,96,128,160,192,224,256 |
| Tokens | seq block 4096 / kernel block 64, context batch 1, token budget 32768 |
| Client | Total C512, 4 measured client processes, SSE chunk 4096, request profiling off |
| Measurement | 240 s warmup, two 300 s windows, then drain; step/frontend measurement on |
| Build/startup | CUDA13, SM10x / 10.3, jobs 32; startup 2400 s, test budget 21600 s |

Changing performance options is rejected. For experiments use the generic
`qwen35_epd_fusion_4gpu_smoke.py` entry. Placement and asset paths can be relocated
with `--gpus`, `--encoder-gpus`, `--model-dir`, `--data-dir`, `--cache-root`,
`--dg-jit-cache` and `--pycparser-path`; retain the same
hardware and asset contents when comparing throughput.

## Required runtime assets

DeepGEMM is resolved through the normal Bazel CUDA 13 dependencies. Both the
open-source and internal requirements/lock files pin `2.8.0+58a6d07.cu132`, with
separate wheel SHA256 values for x86_64 and aarch64:

- [x86_64 wheel](http://artlab.alibaba-inc.com/1/pypi/rtp_llm/deep_gemm/deep_gemm-2.8.0+58a6d07.cu132-cp310-cp310-linux_x86_64.whl)
- [aarch64 wheel](http://artlab.alibaba-inc.com/1/pypi/rtp_llm/deep_gemm/deep_gemm-2.8.0+58a6d07.cu132-cp310-cp310-linux_aarch64.whl)

These wheels contain the FP8 Graph execution-epoch fix. No private package copy,
`--deep-gemm-path`, or `RTP_DEEP_GEMM_DIAGNOSTIC_PATH` hook is needed. The launcher
does not import DeepGEMM from its own Python environment. Before starting the
services, the Bazel worker verifies the resolved package's FP8 kernel header
and architecture-specific extension hashes, including after normal `gpu_lock`
JIT staging. An old, missing or modified package fails validation. This avoids
relying on dist-info metadata, which is not copied during JIT staging.

The historical reference used a private `83961ec` package with the epoch fix
and additional timeout diagnostics. Its recorded command is preserved as
historical evidence; the new published wheel does not include those private
diagnostics. The model, workload and performance settings remain fixed, but
throughput with the new wheel must be measured again before claiming parity
with the historical 4.286667 requests/s result.

The default JIT directory remains
`build_logs/epd-concurrency-bottleneck-20261005/jit-cache`; pycparser defaults to
`/root/.cache/epd-fusion-deps/pycparser`. These are external writable-cache/build
assets; on another machine supply your own paths below.

On another checkout, prepare the same model and video fixture,
create an empty private JIT cache, then pass the relocated paths explicitly:

```bash
mkdir -p /path/to/private/jit-cache
/opt/conda310/bin/python github-opensource/rtp_llm/test/smoke/qwen35_e1pd4_perf_smoke.py \
  --model-dir /path/to/Qwen3.5-397B-A17B-FP8 \
  --dg-jit-cache /path/to/private/jit-cache \
  --pycparser-path /path/to/pycparser-bazel-repository \
  --output /path/to/new-run-directory --execute
```

The pycparser path must contain a Bazel-compatible external repository, as does
`/root/.cache/epd-fusion-deps/pycparser` on the reference machine. GPU placement
can be changed with `--encoder-gpus` (one GPU) and `--gpus` (four distinct PD GPUs).

`RTP_STEP_MEASUREMENT=1`, `RTP_FRONTEND_MEASUREMENT=1`,
`DG_MEGA_MOE_FP8_GRID_SYNC_DIAG=0`, `MM_RDMA_READ_TIMEOUT_MS=30000`, the JIT
cache path and pycparser override are explicitly supplied by the launcher.
The grid-sync diagnostic switch is retained from the reference environment;
it has no effect in the published wheel without those private diagnostics.
The newer `MEGA_MOE_LOG_INPUTS` diagnostic is explicitly disabled.
No manual `export` is needed for these settings.

## Evidence and verification

The output includes the full command, source revision/status, source snapshots
and SHA256 manifest, `preset.json`, service configs, request/steady-load results,
GPU observations and server logs. The snapshot includes this preset;
`preset.json` records the expected wheel version/hashes and, in the worker
output, the resolved package path. Compare steady measurement QPS (excluding
warmup and drain), errors and input/output validity; do not use process wall
time as QPS.

`qwen35_e1pd4_perf_reference.json` preserves the measured command and service
configs, with dynamic Encoder diagnostic output paths removed. The CPU target
`//rtp_llm/test/smoke:qwen35_e1pd4_perf_preset_test` compares every service env and
argument against that record and checks launcher-to-worker parameter forwarding,
GPU allocation, rejected configuration drift, dry-run behavior and runtime
dependency failures. The historical reference is not a newly measured result on
the rebased branch. The earlier short windows do not establish long-run stability.
