# Evaluator `/awi` protocol instrumentation

This document is the implementation contract for the runtime instrumentation
behind `POST /tokenize`, `POST /awi/generate`, `POST /awi/barrier` and
`GET /awi/observations`.

## Build provenance

`build.source.internal` / `build.source.public` are resolved, in order, from:

1. `RTP_INTERNAL_COMMIT` / `RTP_PUBLIC_COMMIT` environment variables. Both must
   be exactly 40 hexadecimal characters or they are ignored.
2. `RTP_AWI_INTERNAL_COMMIT` / `RTP_AWI_PUBLIC_COMMIT` compile-time macros.
   A launcher or Bazel `--copt`/generated header can define these.
3. `git rev-parse HEAD` from the process working directory (best effort; images
   normally have no `.git` and use the environment contract above).

The source pair used to build the image must therefore export both variables in
the candidate `launch.env`, for example:

```
RTP_INTERNAL_COMMIT=<40 hex internal RTP-LLM SHA>
RTP_PUBLIC_COMMIT=<40 hex public github-opensource SHA>
```

No SHA is synthesized from a requested configuration. An unresolved value is
returned as an empty string so a reviewer can see the missing build contract
instead of a fabricated placeholder.

## Event provenance

| Event | Produced by | Real source |
|---|---|---|
| `capture` | `CudaGraphRunner::captureOneGraphInstance` | emitted only after `graph.instantiate()` completes; backend is the compiled accelerator |
| `prefill` | `NormalEngine::step` | monotonic window around `executor_->process`; `input_tokens` from `GenerateStream::inputLength()`; chunked prefills aggregate to one event |
| `decode` | `NormalEngine::step` | monotonic window around the synchronized scheduler step; `step = outputTokenLen-1` so the first sampled token remains part of prefill |
| `resident` | `NormalEngine::step` + `FIFOScheduler::awiResidentSnapshot` | one scheduler snapshot after the barrier becomes resident; `kv_tokens`/`reserved_tokens` come from allocated block counts |
| `reuse` | `NormalEngine::step` | `GenerateStream::initialReuseLength()` from the allocator's actual reused KV prefix |
| `capacity_failure` | `AwiService::barrier` / `NormalEngine::step` | allocator admission check (`availableBlocksNum`) or a real stream allocation error |

`graph_replay` is observed from `CudaGraphRunner::replayGraph` and
`PyWrappedModel::forward`; an eager fallback in the same step clears it. Decode
timing synchronizes the accelerator after `executor_->process` so host and
collective overhead is included. Barrier wait happens in the scheduler before
the timed step and is excluded.

## Barrier semantics

`POST /awi/barrier` evicts prior benchmark prefix-cache entries (the memory KV
reuse configuration remains enabled), checks the real available KV block count
against the requested context, and arms an operation-scoped gate. The FIFO
scheduler withholds decode for those request ids while each request reserves its
full input+output horizon (`GenerateStream::setReserveStep(max_tokens)`); the
gate releases only when every operation request has a resident allocator-backed
reservation. If admission cannot satisfy the request the endpoint returns HTTP
507 with `failure_kind=kv_capacity_exceeded` and records the same event.

## Weights census

`rtp_llm/model_loader/awi_census.py` records every checkpoint index key after
`ModelLoader.load_weights()` from the actual `ModelWeights` tensors. Each rank
writes `rank_<tp>_<dp>_<ep>.json` under `RTP_LLM_AWI_CENSUS_DIR` (default
`/tmp/rtp_llm_awi_census`); the observations endpoint merges all rank files.
Engram checkpoint keys are cpu/pinned with `persistent_gpu_bytes=0`; all other
materialized tensors report their actual `cuda:N`/`hip:N` device and size.

## Known limitations

- Cross-rank graph replay is observed on the reporting rank; events carry
  `rank`/`world_size`, but the HTTP server is rank-local.
- Barrier prefills are admitted sequentially by the existing FIFO prefill lane
  (decode is concurrent once all reservations coexist).