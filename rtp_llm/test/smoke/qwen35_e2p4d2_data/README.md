# Qwen3.5-397B-A17B-FP8 E2/P4/D2 C96 reproduction

This reproduces the best configuration measured on 2026-09-11. The manual target is
//rtp_llm/test/smoke:qwen35_e2p4d2_repro. The original historical targets are retained.

## Configuration

| Role | Physical GPU | Settings |
| --- | --- | --- |
| E0/E1 | 5 / 0 | Two independent single-GPU services, TP1/DP1, embedding batch 1 |
| P | 1-4 | TP1/DP4/EP4, DeepEP Normal, eager, token budget 20000, reserve 24576 MiB |
| D | 6-7 | TP1/DP2/EP2, DeepEP Low-Latency, CUDA Graph, explicit KV 49152 MiB per worker |

Client concurrency is 96 across the whole pipeline. Each D worker has concurrency_limit=96.
D graph capture sizes: 1,2,4,8,16,32,48,64,96. D reserve remains 8192 MiB in the command,
but explicit kv_cache_mem_mb bypasses automatic pool sizing; this is not a claim of an
actual 8 GiB reserve. BF16 activation, BASE KV; prefix/MM/URL reuse disabled.
max_seq_len=30720, max_tokens=4096, temperature=0, thinking disabled.

The committed video hook is enabled only by QWEN35_BENCH_NATIVE_VIDEO=1. It decodes
and samples the original video for every request: 46 frames, grid [23,44,80],
20240 visual tokens. Each successful request must report input_tokens=24422,
video_tokens=20240, reuse_len=0 and pd_sep=true. Truncations and failures fail the run;
requests are not automatically retried. This is one repeated fixed video workload.

## Environment

Use the original SM103a development container and its internal Bazel dependency overlay.
Historical Python: 3.10; PyTorch: 2.11.0+cu130; Transformers: 5.2.0;
rtp_kernel: 0.1.0+3bc0ca45.cu13sm103a. Model weights and kernel wheels are not in Git.
The launcher uses cuda13/sm10x, compute capability 10.3 and an explicit kernel repository.
Repository AGENTS.md requires the test-execution workflow for Bazel operations. The
launcher retains its pre_build_check.sh and gpu_lock integration. Use the configured
container user and cache settings for the environment; do not launch alongside occupied GPUs.

## Run from the RTP-LLM repository root

CPU-only fixture and historical data verification:

    python3 rtp_llm/test/smoke/qwen35_e2p4d2_data/verify.py

Prepare a command without starting any GPU service:

    python3 rtp_llm/test/smoke/qwen35_e2p4d2_data/run.py       --model-dir /ssd/7/tuoyu.ty/workspace/model/Qwen3.5-397B-A17B-FP8       --kernel-repository /ssd/7/tuoyu.ty/workspace/qwen35-epd-validation/kernel-repository       --cache-root /ssd/7/tuoyu.ty/workspace/qwen35-epd-validation/bazel-cache       --output /ssd/7/tuoyu.ty/workspace/qwen35-e2p4d2-new-run       --mode smoke

Add --execute to run after the environment's test-execution checks. The output directory
must not exist. Add --bazel-option=--config=daily_aone_bazel_cache if the environment
requires the internal cache configuration. Supply authentication through environment
configuration; remote headers are redacted from the saved command and console output.

smoke: one request, concurrency 4, and confirmation that both E replicas handled requests.
benchmark: the same functional gate, a C96 pre-run of about 180 seconds, then three C96
long runs with at least 600 seconds of steady measurement each, plus startup/drain time.
Set --mode benchmark and choose a new output directory. It does not rerun E1 or C32 baselines
and does not run a profiler. All eight GPUs are required even for functional smoke.

## Historical results included

| Long run | Steady Total TPS | Steady Output TPS | Successful cohort requests |
| --- | ---: | ---: | ---: |
| 1 | 41370.99 | 1553.47 | 1079 |
| 2 | 44309.60 | 1654.32 | 1145 |
| 3 | 43720.62 | 1646.37 | 1136 |
| Mean / sum | 43133.74 | 1618.06 | 3360 |

All 3360 requests succeeded, with no errors or truncation. The earlier C96 pre-run is
included separately and excluded from the three-run mean. Steady metrics count whole
requests completing from 30 seconds after dispatch begins until dispatch stops.
Total TPS=(input+output tokens)/window seconds; Output TPS=output tokens/window seconds.
Cohort metrics include request drain time and must not be mixed with steady TPS.

results/*.requests.jsonl.gz retain per-request usage, timings, status and identity, with
generated text removed. Summary files include the SHA256 of the original complete request
log. manifest.json records asset SHA256 and historical harness provenance. verify.py
independently recomputes the summaries from these request records.

New run outputs include result.json, per-request JSONL and summaries, GPU resource samples,
and P/D worker_status snapshots about every 5 seconds. All task phases are preserved;
one aggregate endpoint per role avoids duplicate polling. Snapshot sampling can miss short
states. New E RPC instrumentation is outside this reproduction change.

## Validation of this packaging change

CPU fixture/hash reconciliation and Python/command checks are performed without GPUs.
The historical measurements above came from the archived harness; they are not new runs
of the reorganized entry. GPU smoke has not been rerun while the machine is serving vLLM.
