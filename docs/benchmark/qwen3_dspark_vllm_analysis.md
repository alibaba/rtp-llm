# Qwen3 DSpARK backport: RTP-LLM and vLLM analysis

## Scope and result

This report covers the Qwen3 DSpARK backport on `origin/main` at
`10d1aba882`, using the same target and draft artifacts on two H20 GPUs. The
implementation retains DSpARK's two phases:

1. `COMMIT` writes accepted target auxiliary features into the draft feature
   KV cache.
2. `PROPOSE` consumes the newest committed anchor plus masked query slots and
   generates the Markov chain in one draft-model forward.

The target verify/reject/commit loop, tagged hybrid attention inputs, cache
group selection, sampler, TP broadcast, and RPC transport reuse the generic
mainline infrastructure. There is no second DSV4 cache, RPC, or CUDA-graph
implementation in this branch.

The K=7 implementation is functional and stable for the measured workload,
but it is not yet performance-equivalent to vLLM. The measured RTP-LLM gap is
1.66x at concurrency 1 and 4.09x at concurrency 8. Target verification, rather
than host bookkeeping, dominates the RTP-LLM decode timeline.

## Benchmark configuration

| Item | RTP-LLM | vLLM |
|---|---|---|
| Target / draft | identical local artifacts | identical local artifacts |
| GPUs | H20, devices 0-1 | H20, devices 6-7 |
| TP / dtype | TP=2, BF16 | TP=2, BF16 |
| Context limit | 20,480 | 20,480 |
| DSpARK width | `GEN_NUM_PER_CIRCLE=7` | `num_speculative_tokens=7` |
| Custom all-reduce | disabled | disabled |
| CUDA graph | requested, automatically disabled for DSpARK | `--enforce-eager` |
| Requests | 16; two production-shaped prompts repeated | same JSONL |
| Per request | 1,696 input tokens, 128 output tokens, greedy, ignore EOS | same |

RTP-LLM also disabled the three experimental FlashInfer TRT-LLM FMHA paths.
With the current dependency, the non-paged v2 prefill path allocates only a
1 KiB workspace and fails warmup for this prompt with a 5,242,368-byte request.

`GEN_NUM_PER_CIRCLE` is intentionally spelled as the existing RTP-LLM API
requires. Omitting it silently selects the default K=1. Every benchmark must
record the resolved value; a trace was used here to verify `mtp_step=7`.

## End-to-end performance

| Runtime | Concurrency | Success | Throughput (output tok/s) | Mean latency | p50 | p95 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| RTP-LLM K=7 | 1 | 16/16 | 28.01 | 4.569 s | 4.163 s | 4.225 s | 12.122 s* |
| vLLM K=7 | 1 | 16/16 | 46.60 | 2.747 s | 2.759 s | 2.890 s | 2.893 s |
| RTP-LLM K=7 | 8 | 16/16 | 92.26 | 10.133 s | 11.006 s | 11.109 s | 11.109 s |
| vLLM K=7 | 8 | 16/16 | 377.10 | 2.673 s | 2.654 s | 2.720 s | 2.720 s |

`*` The first request included cold-path work. Excluding it, RTP-LLM c1
requests were approximately 3.87-4.23 seconds.

The RTP-LLM c8 result is a second warm run. Enabling
`RTP_LLM_STREAM_ASYNC=1`, `RTP_LLM_DEVICE_INPUT=1`, and
`RTP_LLM_MTP_ASYNC_DEVICE_STATE=1` did not improve the stable macro result
(about 92.3 tok/s versus the earlier 92.5 tok/s baseline).

Setting `FORCE_SCORE_CONTEXT_ATTENTION=0` produced 31.38 tok/s at c1 but
89.96-92.09 tok/s at c8. A follow-up trace showed only a small target-verify
change (60.74 to 59.43 ms median). The c1 difference was therefore dominated
by cold-run variance and this switch is not claimed as a stable optimization.

Matching the internal target-only limits (`MAX_CONTEXT_BATCH_SIZE=8`,
`MAX_BATCH_TOKENS_SIZE=65536`, `CONCURRENCY_LIMIT=32`) also did not improve
DSpARK c8: the warm result was 90.47 tok/s. This indicates that the 4.09x c8
gap is not simply the default single-context admission limit; steady decode
execution remains dominant.

RTP requests required an average 83.0 engine iterations for 128 tokens at c1
and 83.375 at c8. Treating one token per round as the mandatory target token,
the approximate accepted-draft fraction is `(128 - rounds)/(rounds * 7)`, or
about 7.7-7.9%. vLLM's exported counters measured 832 accepted draft tokens
out of 8,512 proposed, or 9.77%. These definitions should not be compared to
an acceptance metric that includes the mandatory/bonus target token.

## RTP-LLM execution timeline

A request-scoped Kineto trace captured one prefill and eleven K=7 decode
rounds. Decode medians on rank 0 were:

| Scope | Median per round | Share / interpretation |
|---|---:|---|
| Whole decode round | 76.57 ms | 100% |
| Target verify forward | 60.74 ms | about 79%; primary bottleneck |
| DSpARK round-head/propose preparation | 10.89 ms | includes preparation and dependent work |
| DSpARK propose forward | 6.70 ms | one fixed-width draft forward |
| DSpARK Markov sampling | 3.97 ms | seven sequential Markov/sampler steps |
| Draft commit forward | 2.86 ms | accepted feature commit |
| Rejection sampling | 0.80 ms | generic speculative sampler |
| TP sync after rejection | 0.24 ms | small in this trace |
| Wait for previous bookkeeping | 0.006 ms p50 | not a material bottleneck |

Scopes are nested and therefore must not be summed as independent wall time.
The trace shows that moving bookkeeping tensors to CUDA does not address the
dominant cost. The high-value performance work is target verify attention /
backbone execution and higher-throughput batching, followed by reducing the
proposal preparation and seven-step Markov launch sequence.

The first prompt costs approximately 111 ms in the target prefill and 337 ms
in the DSpARK feature-cache seed/commit path. This is excluded from steady
decode-round comparisons but included in request latency.

## vLLM execution timeline

A matching rank-0 Kineto capture used TP=2, BF16, K=7, FA3 attention and eager
execution. Its stable c1 decode scope was 39.32 ms per round (15 rounds), versus
76.57 ms for RTP-LLM: RTP-LLM is about 1.95x slower at the round level. In a c8
capture, vLLM's eight-request decode scope remained 39.23 ms per round (13
rounds). Thus vLLM both executes a single speculative round faster and batches
eight active streams with almost no increase in the GPU scope. The latter is
the clearest timeline explanation for the larger 4.09x end-to-end c8 gap.

The c1 capture, including one 1,696-token prefill and 15 decode rounds, reported
685.94 ms of self CUDA time. GEMMs accounted for 591.39 ms (86.2%), FA3 forward
32.53 ms (4.7%), and TP all-reduce 26.52 ms (3.9%). The c8 capture shifted the
mix toward attention and communication: GEMMs were 75.3%, FA3 9.0%, and
all-reduce 8.0% of self CUDA time. These are whole-capture operator shares;
they must not be interpreted as disjoint DSpARK phase durations. vLLM exposes
the combined `execute_context` GPU annotation but not separate target-verify,
draft-propose, and commit annotations, whereas RTP-LLM's request scopes permit
that finer phase split.

Profiling forced vLLM workers from `fork` to `spawn`. Since this host image has
no `nvcc`, FlashInfer's top-k/top-p sampler could not JIT in the spawned worker;
the profiling runs therefore set `VLLM_USE_FLASHINFER_SAMPLER=0`. Requests were
greedy, while the target/draft models, K=7, TP=2, FA3 attention, and eager mode
were unchanged. Profiling throughput (35.30 tok/s c1 and 69.12 tok/s c8) is not
used in the end-to-end table because Kineto overhead is large; only device
scope timings and operator composition are used from those runs.

## Correctness and semantic alignment

The backport fixes and tests the following DSpARK-specific invariants:

- Speculators artifacts use a bonus anchor and a K+1 query layout.
- Speculators auxiliary layer IDs are one-based layer-output IDs; RTP captures
  zero-based layers `[7, 20, 33, 46, 59]` for artifact IDs
  `[8, 21, 34, 47, 60]`.
- Prefill seeds the draft feature KV from target auxiliary hidden states before
  the first proposal.
- The proposal prefix begins at the newest committed absolute position, not
  at that position plus one.
- Tagged attention inputs select the generic feature-cache group for DSpARK
  and the ordinary target KV group for verification. Kernel-local block IDs,
  not cache-store physical IDs, are passed to attention kernels.
- Target and draft vocabulary sizes are independent; mapping occurs at the
  sampler boundary and padded TP vocabulary columns are excluded.
- DSpARK rejects beam/tiled requests, preserves per-request sampling config,
  and maintains one Markov history row per request.
- Async multi-stream state may advance by K+1 tokens beyond a scheduler group
  snapshot. Both draft and verify sampler histories are sized from the current
  stream length; the draft sampler additionally reserves K append positions
  and its common output column.
- PD RPC uses a commit-only DSpARK handoff with no fabricated proposal. Other
  MTP/Eagle RPC contracts remain unchanged.

Greedy text is not byte-identical between the two runtimes. On the first
production prompt RTP-LLM stated an August 2022 date while vLLM stated August
2019, and ignore-EOS makes both outputs continue beyond a natural endpoint.
Consequently, 16/16 transport success and shape tests are not evidence of
token-level correctness. The tensor golden generated from the reference
implementation covers anchor layout, K=7 mask `151707`, auxiliary concatenation,
fused features, context K/V, Markov hidden state, and logits. A complete release
gate should load those tensors in an RTP model-level integration test and
compare intermediate tensors with dtype-appropriate tolerances.

## CUDA graph status

CUDA graph is deliberately disabled for DSpARK in both target and draft model
wrappers, even when the server-wide flag is on. Capturing the target verify
path left auxiliary hidden-state outputs stale; proposal/commit capture also
regressed acceptance and failed larger batches. All backport-specific graph
runner forks were removed, leaving the mainline graph implementation intact.

Re-enabling graph requires an explicit output-ownership contract for tagged
auxiliary hidden states, graph-stable feature-KV mutation, and acceptance plus
multi-batch regression tests. Until then, silently running eager is safer than
publishing stale features.

## Online configuration comparison

The only internal configuration found for this model family is
`internal_source/rtp_llm/test/smoke/BUILD` plus
`local_qwen35_dense_dspark.json`. Despite its filename it starts the dense
target only; it is not an authoritative online DSpARK manifest. Its useful
resource baseline is H20, TP=2, max sequence length 20,480, max batch tokens
65,536, concurrency 32, cache reuse enabled, block size 2,048, FP8-per-block,
warmup disabled, and CUDA graph disabled.

Recommended initial DSpARK deployment settings are therefore conservative:

- Start with TP=2, BF16, K=7, graph disabled, and exactly the validated
  attention backend. Do not infer K from the draft artifact.
- Leave `FORCE_SCORE_CONTEXT_ATTENTION` at its validated default until a
  broader prompt-length A/B demonstrates a repeatable paged-verify advantage;
  the current c8 test did not.
- The online batch-token and concurrency limits are safe for this workload,
  but do not claim a DSpARK speedup from them; the measured warm c8 result was
  slightly lower than the default run.
- Keep custom all-reduce disabled for the first correctness comparison; A/B it
  separately on the production topology.
- Do not enable stream-async/device-state by default yet. It is now stable on
  this workload but gave no throughput gain and previously exposed stale
  scheduler-snapshot assumptions.
- Enable cache reuse only after a prefix-reuse test verifies feature-KV and
  target-KV lifetimes together. The benchmark intentionally measured zero
  reuse.
- Validate FP8 target weights independently. The comparison in this report is
  BF16 and cannot predict the internal smoke configuration's FP8 accuracy or
  throughput.
- Record resolved model type, artifact config, K, auxiliary layers, target and
  draft vocabulary sizes, attention backend, graph state, TP/CP topology, and
  acceptance counters in startup telemetry.

## Remaining performance and release gates

1. Add the reference tensor golden as a hermetic RTP model-level test; include
   a different target/draft vocabulary case.
2. Add live TP=2 tests for K=7 at c1/c8, mixed stream ages, request admission,
   cancellation, and prefix reuse. Unit tests alone did not expose the stale
   group snapshot bugs.
3. Add CP coverage for tagged feature attention and reject unsupported
   topology combinations at startup rather than inside a request.
4. Add phase annotations to the vLLM DSpARK path (or capture it under Nsight
   Systems) to split its combined execution scope into target verify, proposal,
   Markov transitions, and commit. The current Kineto result provides the
   round-level and operator-level comparison but not a phase-for-phase split.
5. Optimize target verify first. Only after that, fuse/batch the seven Markov
   transitions and reduce proposal input materialization launches.
6. Design graph-safe auxiliary output buffers before re-enabling CUDA graph.

## Diff disposition

The final staged diff is 66 files, +3,635/-438 lines (4,073 lines of churn).
This count includes 1,501 added test/report lines and should not be used alone
as a complexity measure. The file-level disposition is:

| Disposition | Files | Reason |
|---|---|---|
| Required Qwen/DSpARK model code | `models/qwen_3_dspark.py`, `models_py/model_desc/qwen3_dspark_model.py`, `models_py/speculative/dspark_proposer_mixin.py`, their `__init__`/BUILD/registration files, `qwen_v2.py`, `qwen_v3.py`, `model_config.py`, `model_weight.py`, weight descriptors/converter | Artifact parsing, one-based auxiliary-layer conversion, fused feature/KV projections, COMMIT/PROPOSE model behavior, and weight mapping |
| Mainline speculative API adaptation | `ConfigModules.*`, `MTPModelConfigHelper.h`, `EngineBase.h`, `ProposeModelEngineInitParams.h`, `GenerateConfig.h`, `ModelTypes.*`, `OpDefs.*`, `OpData.*`, `DSparkCallPhase.h`, `PyWrappedModel.*`, `RtpLLMOp.cc` | Add one type/phase and carry tagged inputs/auxiliary row shape through existing APIs; no parallel executor abstraction |
| Generic runtime reuse with DSpARK branches | `MtpExecutor.*`, `MtpBatchStreamProcessor.*`, `SpeculativeSampler.cc`, `NormalSamplerInputGatherer.cc`, `NormalEngine.*`, `GenerateStream.*`, `StreamCacheResource.cc` | Reuse propose/verify/reject/commit scheduling, sampler, TP synchronization, stream state and cache lifecycle; DSpARK branches only supply its geometry and feature state |
| Necessary cache adaptation | `CacheConfigCreator.cc` and allocator test | One multi-layer DSpARK cache module; K is proposal width, not K independent cache modules |
| Necessary PD RPC adaptation | `DecodeRpcServer.cc`, `PrefillRpcServer.cc`, speculative handoff helpers in `GenerateStream.*` | Permit commit-only DSpARK handoff while preserving the proposal-bearing MTP/Eagle contract |
| Sampling backend parity | CUDA and ROCm speculative sampling kernels/headers plus CUDA tests | Point-mass deterministic draft probabilities and different target/draft vocabulary mappings must have identical backend semantics |
| Temporary dependency compatibility | `ops/fused_rope_kvcache_op.py` | Supports both old CUDA-13 artifact parameter `cp_position_ids` and current `position_ids`; remove after the wheel is upgraded |
| Tests and report | speculative C++/CUDA/Python tests, stream/cache tests, this document | Shape, phase, cache ID, vocabulary, sampling and multi-stream regressions |

Removed during convergence: every backport-specific `cpp/cuda_graph` change,
the local Bazel server launcher, a broken warmup helper/test, unused model-data
test includes, and formatting-only changes. There are no staged DSV4 files and
`git diff origin/main -- rtp_llm/cpp/cuda_graph` is empty.

## Verification status

Passing in the CUDA-13 build environment:

- `mtp_executor_test`
- `mtp_batch_stream_processor_test`
- `cuda_speculative_sampling_test`
- `test_dspark_proposer`
- `test_qwen3_dspark`
- `single_type_kv_cache_allocator_test`
- `generate_stream_test`
- `//rtp_llm:rtp_llm_package_libs`
- live TP=2 BF16 K=7 server and 16/16 c1 plus 16/16 c8 requests

`model_data_test` could not link in this environment because the test rule did
not place the existing `libtorch_nvshmem.so` beside `libtorch_cuda.so`; no
DSpARK code was compiled for that test after its unused backport-only includes
were removed. This is recorded as an environment/test-rule failure rather than
claimed as a pass.

## Reproduction artifacts

The benchmark driver and raw JSON results were kept outside the repository in
`/data0/caihaowen.chw/rtp-backport-run` during validation. Large Kineto traces
are temporary and should be deleted after their aggregate timing table above
has been reviewed.
