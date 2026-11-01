# Full-model HTTP decode A/B protocol

This is a measurement tool and an operator protocol, **not a new performance
result**. It starts no model and touches no remote host. CPU protocol fixtures
are unit tests only. Existing HTTP smoke and component results do not establish
this A/B comparison or the user's <=100 us additional decode-time requirement.

**Release rebase, 2026-10-21:** this development branch now targets
`feature/dsv41-release-20261019` at `cb42b084a8a6350daa70897555f18d5f1b47f58e`.
`full_model_http.py` explicitly fixes `DSV41_CPU_ENGRAM_LOOKUP=0` and
`DSV41_ASYNC_ENGRAM_LOOKUP=0` in both modes, including when inherited flags
enable them. This measures the same synchronous GPU lookup strategy on both
storage backends. The release's default local CPU-prefill optimization is a
separate comparison: external EGM has no CPU-readable table. Decode prefetch
is not implemented by the release's ordinary-prefill optimization. Rebuild
the matching native runtime and rerun model/Graph validation before measuring
this new baseline; historical main results do not establish its performance.

## What changes between A and B

| Setting | A: `local-pinned` | B: `external-egm` |
|---|---|---|
| Model/checkpoint identity | Complete real V4.1 checkpoint, including exact Engram tables | Same model config/tokenizer/non-Engram tensors; Engram table entries/files removed by verified staging |
| Engram loading | Ordinary main `EngramModule.from_checkpoint`: `load(embed.weight/scale)` → `_PinnedSharedTable` → `HostEngramEmbedding` | Existing fail-closed external bundle attachment and HOST_NUMA/FABRIC mapping |
| Online lookup | Existing raw FP8/E8M0 host-pinned lookup | Same raw layout and lookup kernel over imported EGM |
| Graph | Ordinary main decode Graph path | Existing opt-in process-lifetime external Graph path |
| Fixed shared settings | TP4/EP4/DP1, four workers on the same four consumer GPUs; PDFUSION, BF16, no speculative draft; same fastsafetensors NOGDS loader; same batch-1 Graph flag, context limit, cache policy and runtime | Same |

`full_model_http.py --storage-mode` defaults to `external-egm`. The local branch
requires a complete checkpoint: no fallback to a table-free consumer checkpoint,
no table copied out of the remote mapping to construct a baseline. Both branches
read bounded safetensors headers, check file sizes/ranges, every index/header
entry, and reject unindexed shards. Local requires exactly one weight and scale
per configured Engram layer, with exact row/head dimensions, FP8 E4M3 weight and
U8/E8M0 scale types. External continues to reject every full table and verify the
private Owner publication, hash ABI, revision and capacities.

Header checks do not hash the entire backbone or prove tensor numerical quality.
Record staging/source provenance and content digests separately. The same config
or operator revision string alone cannot prove identical weight bytes. Assets
must remain immutable from preflight through startup. The full real loader and
actual forward/HTTP output still have to pass.

Local launch removes all four external settings, even when inherited:
`RTP_ENGRAM_EXTERNAL_BUNDLE`, `RTP_ENGRAM_GRAPH_MODE`,
`RTP_ENGRAM_MAX_ALLOCATION_BYTES`, `RTP_ENGRAM_MAX_TOTAL_ALLOCATION_BYTES`.
It does not install a fake storage, override a model method or bypass Graph
capture. The normal main process is still `python -m rtp_llm.start_server`.
Both modes require the same complete runtime environment, including the native
EGM extension being installed; only B invokes external attachment.

## Exact timing boundaries from main source

| Recorded metric | Boundary and unit | Limits |
|---|---|---|
| `headers_received_ms` | Client request submission to parsed HTTP response headers | Transport/frontend, not TTFT |
| `first_sse_event_ms` | Submission to first complete SSE `data` event received by this client | May be only the assistant role |
| `client_visible_text_ttft_ms` | Submission to first nonempty `content` or `reasoning_content` delta | Client-visible text TTFT, not first engine token; `thinking` is disabled in this protocol |
| `client_eof_ms` | Submission to HTTP body EOF | Includes generation, network/frontend and final serialization |
| `text_event_intervals_ms` | Receive-time intervals between nonempty text SSE events | **Not token intervals, GPU time or per-step TPOT** |
| `client_text_window_per_final_token_ms` | `(last text receive - first text receive)/(final completion_tokens - 1)` | Descriptive aggregate only; first event may contain multiple tokens, last tokens may be buffered |
| `engine_stream_average_tpot_ms` | `(final aux.cost_time - final aux.first_token_cost_time)/(final aux.output_len - 1)` | Engine stream wall-time mean over one request; not kernel time or distribution of decode steps |

Source evidence, in the main-derived checkout used by these tools:

- [`NormalGenerateStream.cc`](../../../rtp_llm/cpp/normal_engine/NormalGenerateStream.cc)
  sets `cost_time_us = currentTimeInMicroSeconds() - begin_time_us_`, copies
  `firstTokenLatencyUs()`, `wait_time_us_`, `iter_count`, cumulative `output_len`
  and that backend emission's `step_output_len`.
- [`CompleteTokenIds.cc`](../../../rtp_llm/cpp/engine_base/stream/CompleteTokenIds.cc)
  records first-token latency from the engine stream begin time. This is not
  a CUDA event boundary.
- [`model_rpc_client.py`](../../../rtp_llm/cpp/model_rpc/model_rpc_client.py)
  converts `cost_time_us`, `first_token_cost_time_us`, `wait_time_us` to Python
  **milliseconds by division by 1000.0**. Its tracing helper uses the same
  mean TPOT formula for single-sequence streams, with positive timing and token
  count checks. The client stores raw aux fields unchanged in those units.
- [`custom_renderer.py`](../../../rtp_llm/openai/renderers/custom_renderer.py)
  emits cumulative `UsageInfo` counts; the terminal response carries the final
  real `aux_info`. The renderer can buffer text. Backend token updates, text
  events and SSE chunks are not one-to-one.
- [`frontend_server.py`](../../../rtp_llm/frontend/frontend_server.py)
  emits `data: <JSON>\r\n\r\n`. OpenAI streaming can end with EOF after a
  `finish_reason`, without `[DONE]`; the special trailing lowercase `[done]`
  branch applies to non-OpenAI streaming. A HTTP 200 stream can contain an
  inference error later in its body.

The parser supports CRLF/LF, comments, multiple `data:` lines per event, UTF-8,
optional `[DONE]`, and normal main finish+EOF. It rejects truncated frames,
malformed JSON, regressions of cumulative token counts, unexpected tools or
multiple choices, missing terminal usage, and inference errors. Completion token
counts are taken from the **final cumulative usage, never summed across events**.
Final engine aux must be terminal, with matching `output_len`; old interim aux
cannot stand in for missing final timing. Missing/invalid aux leaves engine mean
unavailable and invalidates A/B acceptance instead of inventing a zero.
`step_output_len > 1` is preserved; it does not split one chunk into fabricated
per-token timestamps. `iter_count` is recorded but does not imply one independently
timed decode step per HTTP event.

## Operator sequence: serial A-B-B-A, then independent cycles

1. Record source revision, native `.so` hashes, physical consumer GPUs, NUMA
   placement, CPU affinity/governor, power/clocks, Owner host/NUMA/IMEX topology,
   container memory/shm limits, competing workloads and observed Graph startup
   logs. Use the **same consumer GPUs** in A and B. Admit sufficient RAM/shm for
   the full local tables and loading/cache overhead; do not start A and B full
   models concurrently. Original full local loading can have a substantial
   checkpoint page-cache plus pinned-table peak. This HTTP tool does not bypass
   the loader or enforce the platform memory budget.
2. Stage the full local and external consumer checkpoints from the same verified
   source, keeping all non-Engram tensor content identical. Preserve the existing
   remote Owner while B consumers use it; do not free backing until every B
   worker has exited. No Owner restart or cleanup is done by the client.
3. Choose one immutable prompt file, a target **rendered prompt token length**,
   and a fixed long output length, initially 256 (or 512 if capacity permits).
   Obtain the rendered token count from a preliminary real request; character
   count or offline bare tokenizer length is not a substitute for chat-template
   token count. The client rejects a different observed count. Prompt+output
   must fit the fixed launch maximum of 4096.
4. For one cycle run **A1, B1, B2, A2**, one full server at a time. Each phase is a
   fresh startup with the same warmup/cache preparation, then the same sequence
   of requests. Restart B for B2 as well; merely splitting one uninterrupted run
   into two files is not an independent restart repeat. Wait for all old workers
   to exit before changing mode. Confirm `/health`, actual model/Graph startup
   and expected storage logs before collecting the phase.
5. Send requests sequentially, concurrency=1. Same seed appears at the same
   position in all four phases. Use several distinct seeds per cycle and repeat
   complete cycles with new seeds. Temperature 0.7 makes seed variation useful;
   temperature 0 is supported for deterministic correctness checks but changing
   its seed does not necessarily change work. `min_new_tokens=max_tokens` and
   `ignore_eos=true` intentionally force long benchmark generation. These are
   workload controls, not representative production stopping behavior. Any early
   finish is retained and marked incompatible with fixed-length comparison.
6. The launcher keeps `REUSE_CACHE=1` in both modes. Warmup count/order/prompt must
   match. Compare actual final `reuse_len`, input/output counts, and final text;
   unequal or missing cache state invalidates the quartet. Also inspect the
   detailed local/remote/memory reuse fields stored in raw aux. A different cache
   policy needs an explicitly identical launch change and a separate cycle.
7. Save every request, error, warmup and outlier. Do not rerun only slow samples,
   trim tails or subtract an empty-Graph number. Start with short screening runs,
   then multiple independent cycles and enough long requests for the intended
   uncertainty. Requests from one startup share state; their tokens are not
   independent trial replicates.

Example local preflight/launch (commands are examples, not executed results):

```bash
python tools/engram_sidecar/framework/full_model_http.py preflight \
  --storage-mode local-pinned --checkpoint /private-task/checkpoints/full \
  --model-revision VERIFIED_SOURCE_REVISION --cuda-graph > local-preflight.json

python tools/engram_sidecar/framework/full_model_http.py launch \
  --storage-mode local-pinned --checkpoint /private-task/checkpoints/full \
  --model-revision VERIFIED_SOURCE_REVISION --cuda-graph
```

For B use the documented external command and same `--cuda-graph`; add
`--storage-mode external-egm`, absolute private `--bundle`, verified revision,
and actual large allocation budgets. Do not pass `--bundle` to local mode.
Run plain `preflight` without `--load-native` to obtain a single JSON launch
report; `launch` still performs the real native import probe. Keep probe/startup
logs separately and verify their same-build provenance. The public report is an
operator-supplied intended configuration, **not runtime Graph replay attestation**.

After each phase's server becomes ready:

```bash
python tools/engram_sidecar/framework/http_decode_benchmark.py run \
  --launch-report local-preflight.json --label A1 --cycle graph-cycle-001 \
  --prompt-file prompt.txt --expected-prompt-tokens 128 --tokens 256 \
  --seeds 11,23,37,51,67 --warmups 2 --output ./graph-cycle-001-A1
```

Use B1/B2 with the external preflight report and A2 with the local report. Output
directories must not exist; their parent must exist. The command preserves all
request records on errors and continues the sequence, returning nonzero if any
request failed. `--timeout` is urllib's socket-operation timeout; it is not a
whole-request deadline. Responses are bounded to 64 MiB, individual events to
2 MiB. Raw successful/error SSE `data` payloads and their observation timestamps
are recorded. A malformed UTF-8 or truncated frame is rejected; the already
completed events are retained, not invented into a complete event.

Run directories are mode `0700` and new JSON files mode `0600`; preserve those
permissions. Raw requests/SSE/error bodies can contain model text, server paths
and IPs. Keep them private and extract sanitized statistics for sharing.
`public_identity` whitelists only the launch identity; it does not make the
entire raw result directory suitable for publication.

```bash
python tools/engram_sidecar/framework/http_decode_benchmark.py compare \
  ./graph-cycle-001-A1 ./graph-cycle-001-B1 ./graph-cycle-001-B2 ./graph-cycle-001-A2 \
  --output ./graph-cycle-001-comparison.json
```

The comparator checks chronology, complete phase/request records, seeds and exact
request bodies, config/tokenizer/native hashes, common launch settings, table
presence, storage labels, final token count, cache reuse and generated text. It
also requires the index and shard-header hashes to match between same-mode
repeats (A1/A2 and B1/B2); the two modes have intentionally different indexes. It
reports both paired request-mean differences `B1-A1`, `B2-A2` and their block
mean; it does not conceal a slow side with only the block mean. Warmups remain
explicit rows. Noncomparable rows remain present with reasons. Text equality is
supporting evidence, not a proof of identical token IDs or model numerical
quality. Record a separate token-level correctness check when needed.

## Acceptance and report boundary

Keep eager and Graph cycles separate. A Graph flag or successful HTTP response
alone does not prove capture/replay occurred; require actual backend capture/
replay evidence and stable model/storage addresses. No service modification or
Graph guard bypass is part of this client.

A report must include, per phase and request: final usage/finish reason, complete
aux, token/cache compatibility, TTFT, total elapsed, engine request-mean TPOT,
all raw events and errors, phase order/seed, runtime and checkpoint provenance,
Graph evidence, environment/load/memory observations, and artifact hashes. Report
mean/p50/p95/p99/max across **request means** only with that label and sample size;
these are not percentiles of all individual decode steps. At low request counts,
a p99 is effectively near the maximum and has weak tail precision. Retain every
outlier with corresponding server/device traces before attributing it to EGM,
network, CPU scheduling or other traffic.

The user's target is <=100 us **extra consumer decode time**, not <=100 us total
request latency and not a per-Engram-layer allowance. A >100 us difference of
matched request-mean TPOT is adverse full-service evidence, but a <=100 us mean
cannot establish an every-step or p99 bound. The comparator therefore always
reports `full_decode_step_100us_acceptance=not_established_by_request_mean_or_http_chunks`.
Actual step-level acceptance additionally needs correlated main backend decode
iteration boundaries/trace or CUDA events, their queue/scheduling boundary,
prefill exclusion, and distributions from matched same-work A/B runs. Existing
full-table joint lookup timing remains useful component evidence; it does not
replace the full model's hash/WKV/gate/attention/MoE/scheduler work.

## CPU validation

```bash
python3 -B -m unittest \
  tools.engram_sidecar.framework.test_full_model_http \
  tools.engram_sidecar.framework.test_http_decode_benchmark
```

These fixtures cover local/external selection and inherited settings, exact table
headers, normal Graph flags, main-style SSE without DONE, error-in-200, HTTP
errors, Unicode/CRLF/multiline framing, terminal/cumulative usage, multi-token
emissions, missing timing, cache/order/identity mismatch, all-outlier retention
and exclusive private artifact files. They are not live model, transport timing,
CUDA Graph correctness or performance tests.
