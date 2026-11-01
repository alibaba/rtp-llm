# Experimental external Engram / decode CUDA Graph

This change adds an explicit `--cuda-graph` switch to `full_model_http.py`.
Without it, the previously validated eager behavior and graph rejection remain
unchanged. The switch selects **decode only, batch size 1, TP4/EP4, no draft**.
Prefill remains eager. Existing eager HTTP results are not evidence that this
new Graph path has passed GPU capture, replay, numerical or latency acceptance.

The implementation is deliberately scoped to one immutable generation per
worker process. **An Owner must remain alive until every consumer worker has
exited.** It does not support a running model detaching or replacing Engram,
reclaiming an Engram allocation when a model is destroyed, Graph failover, or a
production lifecycle protocol.

## Why captured pointers remain valid

The real engine path is:

1. `PyWrappedModel` calls Python model `initialize`. The V4.1 extra-weight hook
   validates the bundle/hash, imports one mapping per Owner allocation, verifies
   GPU sample reads, and constructs every local Engram module.
2. With `RTP_ENGRAM_GRAPH_MODE=process_lifetime`, model attachment pins the
   storage **before publishing a READY binding**. It then retains the storage in
   the existing process registry. Storage owns the mapped weight/scale views as
   well as the native imports. A repeated initialization reuses that binding;
   changing the environment does not change addresses, mode or generation.
3. The engine prepares decode metadata, warms up, captures, and immediately
   replays/synchronizes each graph. Engram hashing and lookup operate on device
   inputs; they do not import handles or access the checkpoint during capture.
4. Replay is C++ `CudaGraphRunner::replayGraph`; it bypasses Python model hooks.
   Therefore the storage pin is permanent after READY. `Storage.close()` rejects
   it before clearing views or changing READY, even if model-local references
   have been removed. Checking Python `forward` alone would not protect replay.
5. Model or graph destruction does **not** remove this pin. The process registry
   remains the owner of the original mapping until worker shutdown. Final native
   cleanup drains the selected CUDA primary context before unmapping; if drain
   cannot be proven, it leaves cleanup deferred until process exit.

The only unpin path added here is rollback **before READY**, when no model
forward/capture was admitted. If rollback fails, the resource is retained and
initialization fails; it is not treated as a successful eager fallback.

## Engine evidence and the explicit lifetime limit

The inspected sources are:

- `rtp_llm/cpp/models/PyWrappedModel.h`: Python initialization precedes runner
  capture; the runner receives the model's Engram window size.
- `rtp_llm/cpp/cuda_graph/cuda_graph_runner.cc`: a persistent
  `engram_token_windows` buffer is initialized to `-1`, copied for each request,
  and sliced for each captured shape. `captureOneGraphInstance` warms up twice;
  `replayAndSyncCheck` launches the captured graph and synchronizes.
- `rtp_llm/cpp/normal_engine/NormalEngine.cc`: `stop()` stops the scheduler, joins
  the loop thread and releases the dispatch executor.
- `rtp_llm/cpp/models/PyWrappedModel.cc`: graph runners are reset before the
  Python model reference is released.
- `rtp_llm/cpp/cuda_graph/cuda_graph_runner.h`: the destructor's explicit drain
  is conditional for generation-prefill; it is not an unconditional decode
  drain guarantee. Dirty capture is deliberately retained until process exit.
- `tools/engram_sidecar/native/engram_native.cc`: final `Mapping` destruction
  drains through `cuCtxSynchronize` before `cuMemUnmap`; errors defer cleanup.

Consequently this change does **not** infer that model teardown permits an
external allocation to be unmapped. It uses the longer, existing process
lifetime and does not add an atexit close, graph reset callback or weak-reference
cleanup. Live detach needs a future engine contract that proves all enqueuers
stopped, all streams drained and every possible replay destroyed.

## How to request the bounded experiment

Append `--cuda-graph` to the existing full-checkpoint `launch` command. The
launcher explicitly sets:

```text
ENABLE_CUDA_GRAPH=1
RTP_ENGRAM_GRAPH_MODE=process_lifetime
DECODE_CAPTURE_CONFIG=1
ENABLE_NATIVE_CUDA_GRAPH=0
SP_TYPE=
```

It clears inherited generation-prefill and prefill capture configuration. It
retains concurrency 1 and the complete checkpoint. `--cuda-graph` is a boolean
opt-in; there is no arbitrary capture-bucket or prefill Graph argument in this
slice. Without the flag the launcher sets Graph off and resets external mode to
`eager`, including when the parent shell has Graph environment variables.

The model rejects prefill/target-verify capture in the opted-in mode. It also
rejects a changed storage/hash/layer binding and an incomplete lifetime anchor.
Regular eager prefill still works. The ordinary decode Graph forward does not
add a CUDA capture query; actual replay never enters this Python guard.

## Verification performed and still required

CPU unit tests exercise model attachment with injected factories and the real
`FabricEngramStorage` orchestration with a resource-counting native fake. They
cover explicit opt-in, default rejection, full-layer transaction rollback,
pin/retention failures, no in-process rebinding, prefill/verify rejection, READY
close rejection without releasing mappings, and persistence after model GC.
Launcher tests cover the bounded configuration and removal of inherited prefill
capture state. These tests do not allocate CUDA memory or validate GPU graphs.

For real acceptance on authorized machines:

1. Run the unchanged eager baseline against the same immutable Owner generation.
2. Launch with `--cuda-graph` and require the real engine's batch-1 capture and
   replay completion logs; absence of errors alone is not evidence of Graph use.
3. Send multiple prompts with several generated tokens and verify that changing
   token histories is consumed on successive replays. Compare outputs/logits to
   the eager baseline with stated tolerances and check no table copies appear in
   consumer HBM. Include prompt boundaries, padding/masked rows and reuse.
4. Record latency distributions separately from correctness. Require actual
   replay evidence rather than comparing two eager runs under different flags.
5. Stop and wait for all consumer workers to exit before stopping the Owner.
   Capture/drain failure terminates the failed worker; do not unpin or swap its
   mapping and attempt to continue serving in that process.

Production unload/failover, generation-prefill Graph, speculative decoding and
larger batch capture are outside this result.
