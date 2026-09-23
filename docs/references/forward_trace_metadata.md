# Per-forward timeline metadata

The RTP `TorchProfile` export path automatically includes `rtp_forward_metadata`
(schema version 1). Exported events include phase, request count, sequence count
and total Q in their name, for example
`RTP::model_forward(id=1,phase=prefill_target,requests=2,sequences=2,tokens=8320)`.
Each event exposes the full record under `args.rtp_forward`. No model-input
logging flag is required. Names are expanded in the save worker.

`q_lens[i]` is the number of new tokens executed for sequence row `i`;
`prefix_lens[i]` is the initialized prefix before that execution; `kv_lens[i]`
includes the current Q (`prefix + Q`). `total_q_tokens` sums Q across real rows.
`logical_sequences`, source `request_count`, and physical/padding rows are distinct.
A missing `request_count` means the producer did not provide a trustworthy request
grouping; do not infer it from physical rows. K3 internal chunks have their own
forward IDs and `original_batch_indices` referencing the enclosing forward.

Normal decode uses Q=1 and its decode position as prefix; `input_lengths` contains
original prompt lengths there. MTP verification and draft update use the actual
current model input lengths. Device-state MTP must not substitute a host KV
allocation bound for the GPU's accepted sequence length.

## Synchronization contract

The collector does not call `.cpu()`, `.item()`, a CUDA synchronization API,
`cudaStreamWaitEvent`, or any bookkeeping join during a forward. It does not
allocate CUDA memory or create/record CUDA events in `snapshot()`.

At profiler start, before the model forward, it reserves a 16 MiB device arena.
Ready CPU arrays are copied into owned vectors. When precise lengths exist only
on CUDA, the collector submits `cudaMemcpyAsync(..., cudaMemcpyDeviceToDevice)`
on the producer's current stream into that arena. Source storage remains alive;
subsequent mutation of the source cannot replace the captured values. This does
add small D2D copies and CPU bookkeeping: absence of synchronization is not a
claim of zero overhead.

At profiler stop, completion events are recorded on every stream used for
snapshots. Only the export path waits for these events and reads the arena back.
The legacy synchronous `stop()` uses the same export function after collection;
normal step-window exports use `ProfilerSaveWorker`. These waits are not inserted
into a model forward. Existing Kineto stop/start behavior is unchanged.

If the fixed arena or 65,536-record limit is exhausted, records are marked
incomplete; the collector never grows device storage or blocks inference to
recover missing values. Unsupported dtype/layout is also explicit. The exporter
checks recorded scopes against records and sets `complete=false` for dropped,
invalid, failed or unmatched records. It also checks that existing CPU forward
and K3 target-chunk scopes have enclosing metadata scopes; omissions increment
`unannotated_forward_events`. An export failure leaves a `.partial` file;
only an enriched trace is renamed to the final filename.

The exporter currently parses the trace JSON in the save worker. Large traces
therefore require extra host memory during enrichment. CPU/IO interference and
D2D overhead still require performance measurement.

## Export a shape-replay case

```bash
python rtp_llm/tools/forward_trace_replay.py trace.json
python rtp_llm/tools/forward_trace_replay.py trace.json --forward 456 --output case.json
```

The tool refuses incomplete metadata. It emits the ordered Q/prefix/KV pairs,
parallel/layout settings, raw execution metadata, and inherited configuration
for a selected chunk. Feed this case to a model-boundary harness; sending HTTP
requests does not guarantee identical serving batch assembly.

Warm up the exact execution path for at least 10 representative completed runs,
restore each request's intended prefix state outside timing, then measure.
The checkpoint identity, deployed software revision and hardware must be supplied
by the reproduction environment. This feature does not record token contents,
KV values, KDA state, MoE routing or sparse indices; it is shape replay, not
numerically exact replay or a guarantee of identical end-to-end latency.

## Validation required before deployment

CPU checks cover immutable host snapshots, nested IDs, exceptions, capacity
failure and replay export validation. The source contract test prevents explicit
synchronization/D2H/device allocation in the snapshot function. These checks do
not prove CUDA runtime behavior or replace the production build.

On the approved GPU build/test host, build the complete affected target and test
ordinary prefill/decode, MTP target/draft/update, chunked prefill, graph replay,
CP/KTP, fake rows, TP and micro-batching. Select different Q/KV values on successive
replays to catch stale fixed-buffer snapshots. Exercise multiple producer streams
and a deliberately small arena. Confirm that every exported value matches the
actual inputs and every applicable trace reports complete metadata.

For each warmed case, compare baseline vs patched builds both with profiling off
and on. Inspect CUDA runtime calls and device transfers attributable to the new
collector within each model-forward interval: no added device/stream/event
synchronize, stream waits, blocking copies or D2H are allowed. Separately measure
forward span, throughput, CPU launch gaps and background export interference.
Do not call the no-sync/performance acceptance complete until those checks pass.
