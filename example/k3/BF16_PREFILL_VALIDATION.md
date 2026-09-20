# K3 BF16 Prefill validation

The implementation is based on `origin/feat/k3_dev` at
`1527d574f6fd21c4c61e80884cd4b6dc9ac0b2bb`. Validation uses the
`codex/k3-bf16-prefill-fp32-decode` branch. Default service configuration
remains FP32; see [deployment and rollback](BF16_PREFILL_STATE.md).

## Environment

Two StarAgent hosts in the same l20-b cluster: L20D-dev-144 (Prefill) and
L20D-dev-145 (Decode), eight B300 GPUs each. Both hosts built their own service
inside `lhc_GPU`, as `luohaocheng.lhc`, with `--config=cuda13 --config=sm10x`.
No binaries or Bazel outputs were copied between hosts. RDMA services use
existing containers with the same CUDA13 base image:
`rtp_llm_base_gpu_cuda13:2026_04_30_00_05_eba6d8a`.

Full weights are local at `/ssd/5/kimi-k3` on both hosts (93 layers, 96 shards).
Checkpoint preflights passed, and service logs positively identify
FastSafetensors for main and MTP weights. Config and index hashes agree between
hosts. GPU selection found the selected hosts free; no external job was killed.

Remote evidence root on each host:
`/ssd/5/luohaocheng.lhc/k3-bf16-state-transfer-20260920`.

## Completed correctness checks

| Check | Result |
| --- | --- |
| BF16/FP32 gather/scatter, padded physical blocks, invalid slots | Passed |
| Seven fused-prefill GPU tests, including exact explicit BF16-boundary reference | Passed |
| Six current-state registry tests, including unrounded FP32 internal continuation | Passed |
| Actual terminal transfer: FP32/BF16, synchronous/background writer, conv contents, callback-independent ownership/release | Passed |
| Failed store callback, same-request/key retry, independent snapshots and outstanding-reader release (sync/background) | Passed |
| Protocol presence, legacy fallback and rejected dtype combinations | Passed |
| Publication planner and both pool dtypes with 1/2/4/8/16 partitions | Passed |
| Two-host driver tests on both Linux hosts | 33 passed per host |
| Smoke case fixture tests | 18 passed |
| Four-layer TP8→TP8, P BF16/D FP32, MTP3, chunkwise RDMA | 10 cases passed |
| Full model TP8→DP2/TP4, P BF16/D FP32, MTP3, chunkwise RDMA | 172 cases passed |
| Full model TP8→DP2/TP4, P BF16/D FP32, ordinary Decode, chunkwise RDMA | 172 cases passed |

The full run `k3-state-full-comparison-20260920` includes 86 page-boundary
cases, 20 actual Decode page-crossing cases, 35 cache hits, seven cases that
require MTP acceptance, one multimodal case and 41 concurrent stages. It
checks cold/repeated requests around 64K and 128K, plus a 110K cached dialogue
with 106496 reused prefix tokens. Both peers exited their smoke controllers
with status 0. Runtime checks passed for graph buckets 1/2/4/8, all Decode
ranks, TP4 DCP geometry and Prefill padding/chunk rounds.

An earlier four-layer run exposed an existing fixture problem: its prompt had
4991 tokens, shorter than the 8192-token KDA reuse stripe, but the repeat
required a cache hit. The fixture now fits prompts beyond the stripe and
retains the hit assertion. The complete rerun passed.

## Scope of these results

Semantic smoke passing is not a claim of bitwise FP32 equivalence. The final
Prefill state is intentionally rounded to BF16. Full-model comparisons with
unrounded FP32 storage and an explicitly rounded FP32 reference are recorded
separately when available.

The existing RDMA registration path copies unregistered FP32 staging into
registered pinned host memory. It is included in service execution; no claim
of direct GPU RDMA or a handoff speedup is made. Projection-KTP GPU coverage,
exhaustive failure/cancellation/retry injection and isolated handoff/staging
peak instrumentation are not established by these smoke results.

## Pool size evidence

The BF16 run logs Prefill layer-0 backing storage `[986, 210432]` (BF16 base
words). There are 69 KDA layers, 12 local heads at TP8 and 128×128 SSM entries
per head. Each block therefore contains 393216 SSM bytes plus 27648 conv
bytes. Across 69 layers and 986 blocks this is 24.9148 GiB of SSM and 26.6666
GiB of LINEAR storage per Prefill rank. With the **same block count**, FP32
would require 49.8296 GiB of SSM and 51.5814 GiB of LINEAR storage. These are
layout-derived sizes, not total process memory measurements.

Decode TP4 has 24 local heads and 32 LINEAR blocks: 1572864 SSM bytes plus
55296 conv bytes per block, or 3.3481 GiB of LINEAR storage per rank across
69 layers. Its SSM remains FP32. Checkpoint workspace and internal current
states remain FP32 on both roles.

Two BF16 64K serving batches each completed one materialization, ten warmups
and five measured requests (32 requests total, reuse disabled, 32 generated
tokens, batch one). Prefill `nvidia-smi` resident memory was unchanged at
every completed-request sample: 263559–264547 MiB across the eight ranks.
This is coarse process-residency evidence, not a transient staging peak or
proof against all failure-path leaks.

With the same `kv_cache_mem_mb=42000` Prefill budget, the FP32 reference logs
612 blocks (`[612, 407040]` base words) versus 986 with BF16. This configuration
therefore admits 61.1% more blocks, not twice as many: MLA, conv and other
layout costs remain. It is a capacity observation for this topology and
configuration, not a universal gain.

## Explicit boundary reference

A task-local diagnostic patch on Prefill stores
`value.to(tl.bfloat16).to(tl.float32)` at checkpoint scatter into an FP32 pool.
No current-state registry or cuLA workspace is rounded. Both services are
restarted with FP32 storage and the same model, namespace, inputs and
TP8→DP2/TP4 topology. This patch is excluded from the implementation commits.

The reference also passes all 172 request checks (including four RDMA prewarm
probes). The 168 formal requests have identical input-token hashes and reuse
lengths between BF16 storage and the reference. Complete generated token
sequences match in 60/168 cases; remaining generations differ despite passing
the same semantic checks. This is **not** full-model bitwise equivalence.
Concurrent batching, MTP and existing run-to-run generation variability prevent
assigning every generation difference to storage dtype. The low-level cache,
transfer and cuLA boundary tests independently establish exact conversion
semantics; they do not turn the model comparison into a logits-equivalence test.

A separate fixed 65536-token, batch-one, 32-output-token comparison uses input
SHA256 `21435f2c16f908c6e5b1ef05ce0e6f36791618cff6f367e145a2f608d43a8899`.
All 16 requests in the BF16 repeat batch (including materialization) produce
the same tokens. The reference's 16 warmups and five measured requests match those
tokens exactly; only its initial materialization request differs. This is a
representative sequential agreement check, not a substitute for full-model
logits comparison. The reference required 16 warmups to meet the last-three
wall-time convergence check.

## Implementation entry points

- [Cache gather/scatter](../../rtp_llm/models_py/triton_kernels/kimi_kda/cache_store.py): BF16 pool with FP32 computation buffers.
- [Terminal segment publication](../../rtp_llm/models_py/bindings/core/ExecOps.cc): per-segment widening, private ready event and tensor ownership.
- [Wire compatibility](../../rtp_llm/cpp/model_rpc/K3StateTransfer.h) and [protobuf](../../rtp_llm/cpp/model_rpc/proto/model_rpc_service.proto): storage/wire dtype separation and legacy fallback.
- [Transfer tests](../../rtp_llm/models_py/bindings/core/test/ExecOpsTest.cc): actual writer execution and ownership lifetime.
- [Deployment and rollback](BF16_PREFILL_STATE.md): role-local dtype configuration and upgrade order.

## Unrounded FP32 path regression

After restoring the diagnostic source file, the same implementation branch
with P FP32 / D FP32 also passes all 172 request checks. This tests the
preserved FP32 path; it is not a separately rebuilt historical binary.
All 168 formal input hashes and reuse lengths match across all three runs.

| Generated-token sequence comparison | Exact matches |
| --- | ---: |
| BF16 pool vs explicitly rounded FP32 reference | 60 / 168 |
| BF16 pool vs unrounded FP32 pool | 28 / 168 |
| Explicitly rounded FP32 reference vs unrounded FP32 pool | 33 / 168 |

All three runs pass the original semantic assertions. These sequence counts
include model generation variability and BF16 numerical effects, and are not
logit-error measurements. They must not be presented as an accuracy percentage
or as lossless FP32 Decode. The BF16 rounding happens before Decode receives
its FP32 state.

## Warmed serving samples

Same 65536 input tokens, 32 output tokens, batch one, reuse disabled,
TP8→DP2/TP4 and MTP3. Each group starts with one excluded materialization
request, then at least ten representative warmups; the last three warmup
wall times must be within 5% of their median. Five completed requests are
measured per group. All times below are medians.

| P storage / path | Warmups | TTFT, ms | Service total, ms | After first token, ms | HTTP wall, s (range) |
| --- | ---: | ---: | ---: | ---: | --- |
| BF16 | 10 | 1561.214 | 2041.312 | 480.621 | 2.128 (2.124–3.119) |
| BF16 repeat | 10 | 1560.092 | 2017.911 | 458.243 | 2.115 (2.101–3.109) |
| FP32, explicit BF16 boundary reference | 16 | 1560.656 | 2018.498 | 459.539 | 3.097 (2.104–3.134) |
| FP32, unrounded | 11 | 1560.353 | 2040.370 | 480.684 | 3.119 (2.122–3.123) |

TTFT includes Prefill and service/handoff work; it is not an isolated cuLA or
handoff timer. “After first token” is request duration minus TTFT, including
MTP execution and service overhead, not a standalone Decode kernel latency.
HTTP wall time sometimes includes roughly one additional second not reflected
in the service timer; its exact source is unresolved. Both BF16 batches are
reported to expose variability. Five samples do not establish a speedup, and
these numbers do not support claiming one.

Remaining validation gaps: isolated handoff timing; separately instrumented
FP32 workspace and CUDA/pinned-host staging peaks; exhaustive transport-failure
injection and staging-specific leak attribution; projection-KTP GPU smoke; full-model teacher-forced logits comparison. Existing ownership
unit tests, observed MTP acceptance and semantic smoke do not replace those
checks. No tolerance was relaxed to obtain the passing results.


## Follow-up ownership audit

The original writer test only exercised a successful store callback. The added
`failedTerminalSsmRetryKeepsIndependentSnapshots` test executes the real writer
with an injected `StoreFailed` callback, retries the same request and terminal
key, overwrites the source SSM, and drops the mock store's request references.
Both old and new reader-held snapshots retain their respective exact values;
each staging owner expires only after its final reader releases it. This runs
for synchronous and background publication. It does not simulate actual RDMA
cancellation or prove the real store's registration cleanup.

L20D-dev-144 rebuilt `exec_ops_test` locally inside `lhc_GPU` with CUDA13/SM10x.
The original terminal test and this new test both passed (154 ms test-body
time; this is correctness execution time, not a performance benchmark).
Evidence: `retry-build.log` and `retry-transfer-test.log` under the remote
artifact root above. Production code did not change in this follow-up.

## Ordinary Decode follow-up

The full 93-layer P BF16 / D FP32 service also passes 172 request checks with
`SP_TYPE` and `SP_MODEL_TYPE` empty, using the same TP8→DP2/TP4 pair. Every
request has `output_len - iter_count` equal to zero or one. The P and D logs
independently verify all eight ranks use the expected SSM dtype and
FastSafetensors, with no MTP model loaded. RDMA device checks, P padding/chunk
checks, all-rank DCP checks and Decode CUDA Graph buckets 4/8 pass.

The first ordinary run exposed a smoke fixture that hardcoded MTP acceptance
in its multimodal case. It now uses the existing `--require-mtp` flag; the
answer, chunk and cache checks remain. Eighteen fixture tests pass. The full
suite then passed against the retained services using a fresh prompt namespace.
The task-local ordinary runtime verifier also uses the actual process
environment (`SP_TYPE=`), instead of requiring a log line this mode does not
print. Neither correction fabricates service evidence or relaxes numerical
tolerances.

Evidence: `plain-recheck/accuracy.json` under the remote root; local copies
`full-plain-accuracy.json`, `plain-final-prefill.log` and
`plain-final-decode.log` in the task artifact directory.

## Live cancellation and retry

Against the retained ordinary-Decode services, two concurrent clients target
the two D owners. Each stream uses an exact 8193-token input and is closed
after eight content chunks, before completion. Its immediate retry must return
the exact requested JSON; a subsequent repeat must also return it and reuse
8192 prefix tokens. One materialization round, ten warmup rounds and ten
verification rounds passed: 42 interrupted streams and 84 validated follow-up
requests. All 42 tags have matching server-side cancellation exceptions in
`runtime/logs/prefill/access_r0_s0.log`; no tag is unmatched.

During the ten verification rounds, each GPU's completed-round memory sample
was constant (257883–259355 MiB across ranks). Host RSS was not flat: rank zero
increased by 110.44 MiB and the other ranks by approximately 5.6 MiB each. The
probe introduces distinct cached prefixes and records request metadata; it
does not isolate these costs from pinned staging. This is evidence of working
cancellation/retry and stable sampled GPU residency, **not** a proof of no
host-memory leak. Staging-specific allocation and release instrumentation is
still needed to close that part of acceptance.

Evidence: `cancel-retry-result.json` and `cancel-retry-summary.json`, retained
on Prefill and in the local task artifacts. The summary contains request IDs
and cancellation tracebacks without copying prompts. These tests do not inject
a transport outage. The separate C++ writer test covers failed store callbacks.
