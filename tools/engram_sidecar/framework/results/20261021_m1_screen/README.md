# Real complete-table latency screen — 2026-10-21

**The ≤100 µs additional consumer decode-time goal is not established.** This initial M=1 screen
measures one real Engram layer's lookup only. Both actual guarded Graph outputs
passed independent FP8/E8M0 → BF16 bitwise checks; all eight native counters
returned to zero. Eager and CPU wall-clock observations include >100 µs paired
overhead. The finite Graph GPU-event sample has no such violation.

The baseline is main's unchanged `_PinnedSharedTable`, populated from a separate
original checkpoint copy, and all 1,546,900 resident pages were GPU-local NUMA0.
The candidate maps an independent Owner on another host. Both execute unchanged
main Triton lookup, M=1 × 24 heads × 256 dimensions, over the full 94.415 GiB
layer. Owner holds both actual model layers (188.834 GiB physical, two allocations).
Attach/import/map and source loading are outside timing. No full-table HBM copy.

Source commit: `60e0dd54936d5b154102bc700b8396a46f02e642`; RTP main base:
`492ecf45b3ef55b22fd4882622abfdb2e2f664f2`. Two GB200 development hosts, separate
CUDA13 containers, SM100, driver580.95.05, Torch2.11.0+cu130. CPU compilation and
checkpoint copy had stopped before measurement. User-busy dev2 was excluded.

Each row contains 100 adjacent balanced-order local/remote pairs; values are
**paired remote minus local**, in µs, retaining negatives and all outliers.
Percentiles use nearest rank. Hot repeats fixed IDs; random samples uniformly
from the complete table. Neither is a production workload trace.

| Mode | IDs | Measurement | P50 Δ | P99 Δ | Max Δ | >100 µs |
|---|---|---|---:|---:|---:|---:|
| eager | hot | cpu_submit_to_event_complete | 14.112 | 33.727 | 173.184 | 1/100 |
| eager | hot | gpu_eager_event_interval | 13.888 | 32.576 | 173.664 | 1/100 |
| eager | random | cpu_submit_to_event_complete | 12.480 | 28.993 | 97.921 | 0/100 |
| eager | random | gpu_eager_event_interval | 11.488 | 29.600 | 95.744 | 0/100 |
| graph | hot | cpu_submit_to_event_complete | 1.664 | 11.648 | 14.784 | 0/100 |
| graph | hot | gpu_graph_event_interval | 2.944 | 4.736 | 4.864 | 0/100 |
| graph | random | cpu_submit_to_event_complete | 0.352 | 9.408 | 102.751 | 1/100 |
| graph | random | gpu_graph_event_interval | 1.184 | 3.808 | 3.936 | 0/100 |

Eager GPU events surround CPU Python dispatch and therefore include dispatch
stall gaps. Graph events are captured external nodes inside the Graph; they do
not measure only remote memory or exclusive kernel time. CPU wall runs from
submit/replay entry until event completion synchronization returns. Timing
calibrations are recorded separately; nothing is subtracted.

Two outliers deserve preservation:

- Eager/hot pair90: GPU delta173.664 µs and wall delta173.184 µs; remote submit
 235.008 µs and one involuntary thread context switch. This correlates with
 host scheduling, but does not establish exclusive cause.
- Graph/random pair25: wall delta102.751 µs, remote submit131.488 µs, no counted
 context switch. Its Graph GPU interval does not have the same spike; cause
 remains unresolved. Absence of a switch does not prove a fabric fault.

Before this screen, real cross-host lookup validated both layers, tokenizer
prefill and decode history, boundary/4GiB-address rows, int32/int64 IDs and exact
BF16 bits, then released all imports. Consumer checkpoint has40 full model layers
and96,081 non-table tensors. This does **not** mean the full model was used in
this latency screen.

Excluded: n-gram hashing, WKV/projection, gated residual, inter-layer model work,
attention/MoE, scheduling, HTTP and complete decode steps. Layer14 and the joint
lookup chain were not measured in this screen. A whole-decode P99 cannot be
obtained by summing layer percentiles. RTP engine-owned Graph validation and
full-model local/remote A/B remain separate gates. The earlier millisecond
outliers remain documented in [the latency work](../../CHECKPOINT_LATENCY.md).

## Reproduce and inspect

Follow [CHECKPOINT_LATENCY.md](../../CHECKPOINT_LATENCY.md) with the recorded source,
real private checkpoint/bundle/oracle, `--samples 100 --batches 1 --layers first`
and seed20261021; retain Owner until consumers exit. An initial invocation failed
before baseline allocation due to a missing output parent; its local operator log
was retained and a fresh directory was used for this successful screen.

`measurements.tar.gz` contains all raw samples, run/source identities, summaries,
actual Graph correctness records, calibration and successful cleanup. `archive.json`
records each file's digest and the compressed archive digest. `screen.json` embeds
absolute local/remote statistics as well as paired deltas. No Fabric capability
payload or endpoint credential is included. A 10,000-pair matrix and fresh-process
repeats are the next measurement steps; they must not replace these raw outliers.
