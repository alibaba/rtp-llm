# Partial real-table matrix — layer 1, 2026-10-21

**The matrix has not passed the ≤100 µs additional-latency requirement. The run
failed while preparing layer 14; it has no `complete.json` or final native-zero
resource proof.** Layer 1 completed all 14 batch/mode cases, each with 10,000 hot
and 10,000 random local/remote pairs. Those completed samples are retained in
full. A later explicit operator cleanup is separate evidence, not a successful
end to the original benchmark.

The Graph GPU-event comparison has one >100 µs positive delta:
**M=2, random, pair 1629: 139.360 µs**. M=1 Graph GPU-event P99 is 4.736 µs hot
and 5.376 µs random, with zero observed >100 µs deltas in those two finite samples.
That does not imply a pass for other batches, future requests, the whole matrix
or a complete decode step. Eager and CPU wall measurements contain additional
outliers up to approximately 114–118 ms. Their cause remains unresolved.

## Scope and reproducibility

- Actual complete layer-1 table: **101,377,628,352 bytes (94.415 GiB)**, 384,006,168
  rows, native FP8 weight plus E8M0/U8 scales, 256 dimensions and 24 lookup heads.
  M=1,2,4,8,16,32,64. This uses the real full table, not a small synthetic fixture.
- Baseline: unchanged main `_PinnedSharedTable`, populated from independently
  staged original checkpoint shards. The candidate reads the same logical table
  from a separate Owner through cross-host HOST_NUMA/FABRIC mappings. Both call
  the same main raw lookup kernel and produce local BF16 selected rows.
- Layer-1 baseline metadata records all **1,546,900 pages on GPU-local NUMA0**.
  The failed layer-14 baseline is a different table and is not measured here.
- Source commit `60e0dd54936d5b154102bc700b8396a46f02e642`; main base
  `492ecf45b3ef55b22fd4882622abfdb2e2f664f2`. All seven recorded source SHA256 values
  were independently checked against that commit. The runtime reports SM100,
  CUDA 13.0 and Torch 2.11.0+cu130; its raw device-name string is retained.
- One process/epoch, seed 20261021, 20 warmup iterations per case, one GPU at a
  time, no L2 disturbance. Same batch has the same recorded trace SHA between
  eager and Graph. Per-path execution is serial, with seeded balanced order.
- Hot repeats fixed IDs. Random samples uniformly over the complete table,
  using a preallocated trace. These are lookup workloads, not production token
  or n-gram distributions. ID preparation, allocation/import, source loading,
  correctness checks and calibration lie outside timed intervals.
- Four correctness checkpoints per case are recorded: two changed random inputs
  before measurement, then actual outputs after hot/random distributions. Both
  paths, including actual Graph replay outputs, are compared to the independent
  source FP8/E8M0-to-BF16 oracle. This is not an oracle check on every timed output.

Excluded: n-gram hash/history, WKV/projection, gate/residual, layer-to-layer model
work, attention/MoE/EP traffic, scheduler, HTTP and complete decode steps. Neither
layer 14 nor a joint two-layer chain has measurements from this run. Per-layer
P99 values cannot be added to obtain whole-step P99. The subsequent preparation
change and new runs must not erase, replace or filter these earlier anomalies.

## Recomputed paired statistics

Every value below is **remote minus its adjacent local sample**, in µs. Negative
values remain in the distribution. P99 is the nearest-rank P99 of paired deltas,
not remote P99 minus local P99. Each row has 10,000 pairs; counts use strict >100
µs. All absolute and paired statistics in the raw summaries were independently
recomputed and matched, including extrema and exceedance counts.

Eager CUDA-event intervals include Python/CPU dispatch gaps between records.
Graph CUDA-event intervals use captured external event nodes; they are not an
exclusive kernel, memory-fabric or bandwidth measurement. CPU wall spans
submission through event-completion synchronization. Its partition is submit
call plus completion-sync call; sync is not exclusively GPU waiting.

| Mode | M | IDs | GPU P50 Δ | GPU P99 Δ | GPU max Δ | GPU >100 | Wall P99 Δ | Wall max Δ | Wall >100 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| eager | 1 | hot | 13.440 | 38.400 | 95424.895 | 19/10,000 | 40.768 | 95467.422 | 22/10,000 |
| eager | 1 | random | 12.736 | 35.776 | 1090.016 | 12/10,000 | 37.985 | 1096.356 | 13/10,000 |
| eager | 2 | hot | 13.120 | 49.344 | 1549.120 | 23/10,000 | 55.457 | 1549.284 | 27/10,000 |
| eager | 2 | random | 12.736 | 35.456 | 1571.520 | 16/10,000 | 39.168 | 1572.709 | 22/10,000 |
| eager | 4 | hot | 12.544 | 35.584 | 1419.648 | 13/10,000 | 36.735 | 1420.900 | 13/10,000 |
| eager | 4 | random | 12.352 | 34.912 | 895.936 | 11/10,000 | 37.505 | 903.107 | 14/10,000 |
| eager | 8 | hot | 13.216 | 36.192 | 917.856 | 16/10,000 | 39.744 | 920.194 | 18/10,000 |
| eager | 8 | random | 12.608 | 33.440 | 941.824 | 14/10,000 | 35.137 | 943.075 | 15/10,000 |
| eager | 16 | hot | 13.952 | 36.384 | 1976.608 | 16/10,000 | 38.208 | 1980.774 | 17/10,000 |
| eager | 16 | random | 12.864 | 33.600 | 998.464 | 14/10,000 | 36.895 | 1003.875 | 15/10,000 |
| eager | 32 | hot | 13.664 | 35.328 | 2366.560 | 17/10,000 | 37.280 | 2369.062 | 21/10,000 |
| eager | 32 | random | 13.536 | 37.920 | 114382.049 | 16/10,000 | 39.424 | 114423.095 | 17/10,000 |
| eager | 64 | hot | 14.592 | 35.872 | 99924.387 | 19/10,000 | 38.144 | 99966.857 | 19/10,000 |
| eager | 64 | random | 14.080 | 34.848 | 1309.632 | 14/10,000 | 37.121 | 1309.380 | 16/10,000 |
| graph | 1 | hot | 2.944 | 4.736 | 15.776 | 0/10,000 | 12.640 | 929.219 | 8/10,000 |
| graph | 1 | random | 2.432 | 5.376 | 14.944 | 0/10,000 | 12.352 | 885.634 | 9/10,000 |
| graph | 2 | hot | 2.976 | 4.736 | 13.280 | 0/10,000 | 12.960 | 1494.373 | 6/10,000 |
| graph | 2 | random | 2.496 | 5.408 | 139.360 | 1/10,000 | 12.352 | 851.971 | 4/10,000 |
| graph | 4 | hot | 2.432 | 4.544 | 10.976 | 0/10,000 | 12.512 | 496.578 | 5/10,000 |
| graph | 4 | random | 2.752 | 5.088 | 26.720 | 0/10,000 | 12.544 | 1139.939 | 10/10,000 |
| graph | 8 | hot | 2.496 | 4.800 | 17.280 | 0/10,000 | 12.064 | 952.803 | 7/10,000 |
| graph | 8 | random | 2.976 | 4.992 | 15.616 | 0/10,000 | 12.736 | 396.737 | 5/10,000 |
| graph | 16 | hot | 3.296 | 5.504 | 22.528 | 0/10,000 | 12.896 | 1069.379 | 5/10,000 |
| graph | 16 | random | 3.456 | 5.216 | 27.456 | 0/10,000 | 13.280 | 118431.490 | 4/10,000 |
| graph | 32 | hot | 3.616 | 5.216 | 85.536 | 0/10,000 | 13.152 | 979.491 | 10/10,000 |
| graph | 32 | random | 3.808 | 5.184 | 24.352 | 0/10,000 | 13.729 | 1514.212 | 6/10,000 |
| graph | 64 | hot | 3.712 | 5.600 | 28.064 | 0/10,000 | 13.472 | 1059.011 | 2/10,000 |
| graph | 64 | random | 3.552 | 5.344 | 25.440 | 0/10,000 | 13.473 | 1064.260 | 7/10,000 |

Aggregating counts for inventory, without treating the mixed workloads as one
latency distribution: eager GPU 220/140,000 positive deltas exceed 100 µs; eager
wall 249/140,000; Graph GPU 1/140,000; Graph wall 88/140,000. There are **339 unique
pairs** with a GPU or wall positive exceedance; all are in
[all_positive_threshold_pairs.jsonl](all_positive_threshold_pairs.jsonl).
The complete raw archive also retains every non-exceedance and negative delta.

Only one independent process run exists. Epoch 0 is not an independent restart
replicate. No independent-run CI is computed; finite zero exceedances do not
bound future tails. No samples, percentiles or empty-Graph times are subtracted.

## Outlier evidence, including competing interpretations

The full paired records, top positive and negative extremes, and neighboring
samples are in [outlier_examples.json](outlier_examples.json).

| Case / IDs / pair | Path | GPU interval µs | Wall µs | Submit call µs | Completion-sync call µs | Thread voluntary / involuntary switches |
|---|---|---:|---:|---:|---:|---:|
| Graph M2 / random / 1629 | local | 27.712 | 69.985 | 50.336 | 19.649 | 0 / 0 |
| Graph M2 / random / 1629 | remote | 167.072 | 285.761 | 242.209 | 43.552 | 0 / 0 |
| Eager M32 / random / 2447 | local | 51.392 | 66.592 | 55.776 | 10.816 | 0 / 0 |
| Eager M32 / random / 2447 | remote | 114433.441 | 114489.687 | 114468.599 | 21.088 | 0 / 1 |
| Graph M16 / random / 7175 | local | 9.984 | 37.856 | 25.504 | 12.352 | 0 / 0 |
| Graph M16 / random / 7175 | remote | 23.264 | 118469.346 | 118440.930 | 28.416 | 0 / 1 |
| Graph M32 / hot / 8141 | local | 103.360 | 108396.548 | 108336.995 | 59.553 | 0 / 1 |
| Graph M32 / hot / 8141 | remote | 13.984 | 43.776 | 28.928 | 14.848 | 0 / 0 |

- **Graph M2 random pair 1629** runs remote then local. The GPU delta is 139.360
  µs and wall delta 215.776 µs. Both sides report zero counted thread switches.
  Nearby GPU deltas for pairs 1626–1632 are 4.448, 2.240, 3.424, **139.360**,
  -5.440, -3.360, -3.520 µs. The local interval also rises at the anomalous pair
  and immediately afterward. This establishes a measured exceedance, not its
  mechanism; absence of a counted switch does not establish a fabric fault.
- **Eager M32 random pair 2447** has GPU delta 114382.049 µs and wall delta
  114423.095 µs. The large value coincides with the remote CPU submit call and
  one involuntary switch. The eager event window includes CPU dispatch gaps.
  This is correlation with host-side delay, not proof that the whole interval
  was GPU execution, network access, or a single scheduling pause.
- **Graph M16 random pair 7175** has wall delta 118431.490 µs but GPU delta only
  13.280 µs. Its largest observed stall is in the host submit call. Conversely,
  **Graph M32 hot pair 8141** has a large local submit stall, giving wall delta
  -108352.772 µs. Slow host observations occur on both paths; the negative
  example must not be dropped or used to erase positive exceedances.
- A local Graph interval also spikes: M8 hot pair 4284, local GPU 243.936 µs
  versus remote 16.128 µs, GPU delta -227.808 µs, both thread-switch counts zero.
  A remote read is not necessary for every observed GPU-event outlier. This does
  not exonerate the separate positive remote anomaly.
- Empty-event calibration itself has a wall outlier: M4 Graph control sample 49,
  GPU event interval 3.264 µs, wall 391.841 µs, replay call 7.104 µs and completion
  sync 384.737 µs, thread switches 0/0. It contains no lookup kernel or remote
  table read. It demonstrates that this instrumentation/runtime can produce
  wall tails without Engram traffic; it cannot quantify or explain each real
  outlier, and its tail is not deducted from the result.

Attribution requires correlated CPU scheduling, CUDA/driver and device/fabric
traces on a reproducible case. The recorded interval partitions and current-thread
context-switch counts alone cannot assign blame to the fabric, OS or model code.

## Layer-14 failure and separate cleanup

The run requested layers [1,14]. After 14 layer-1 `CASE_COMPLETE` records, it logged
layer-14 `BASELINE_LOADING`, then `FAILED_RETAINED_UNTIL_PROCESS_EXIT` and `FAILED`
with `RuntimeError`. There is no completed layer-14 case or top-level complete
record. The generic failure log has no exact throw-site stack.

The separate read-only diagnostic of the retained 101,380,404,048-byte table
counted all **1,546,943 64-KiB pages**: NUMA0 1,287,941 pages (78.610 GiB), NUMA1
259,002 pages (15.808 GiB), **16.7428% away from NUMA0**. It proves that this
retained baseline does not satisfy exclusively GPU-local placement. It is
post-failure evidence, not proof that the remote Owner or FABRIC mapping failed.

Source identity matters: commit `60e0dd549` calls its original
`make_local_table` then `parse_numa_pages`; it does not contain the subsequently
added `after_load` memory-budget checks. Later preparation-code checks therefore
cannot be cited as the throw site of this run. The mixed placement is consistent
with rejection by the placement check, but the exact raising call remains
unconfirmed without the missing stack.

A later explicitly scoped operator action confirmed the original process absent
and no GPU compute PIDs, closed its diagnostic mapping and removed exactly the
failed table plus its lease. Private-container shm usage went from 101,380,456,448
bytes to zero. Host shm and checkpoint source files were not modified. This
cleanup is recorded separately in [failure_evidence.json](failure_evidence.json);
it does **not** create an original-run PASS or final all-zero native counter proof.
Only sanitized fields and original evidence hashes are published; process/IPC
identifiers, UID-bearing filenames and private paths are omitted.

## Artifacts and independent audit

- [measurements.tar.gz](measurements.tar.gz): all 29 original member payloads,
  byte-for-byte; 1,120,000 raw rows (280,000 pairs, two paths and two clocks).
- [archive.json](archive.json): original and normalized archive digests and every
  unchanged member SHA256. Only container tar/gzip owner/time/mode metadata was
  normalized; no raw JSON, sample, calibration or summary was edited.
- [matrix.json](matrix.json): independently recomputed absolute/paired statistics,
  source/trace checks, partial identity and explicit non-pass status.
- [all_positive_threshold_pairs.jsonl](all_positive_threshold_pairs.jsonl): every
  unique pair with a GPU or wall positive delta >100 µs, including paired local
  values, host partitions and switch counts.
- [outlier_examples.json](outlier_examples.json): top eight positive examples and
  four negative extremes for each clock/mode, the seven Graph M2 neighbors and
  largest wall calibration observations.
- [failure_evidence.json](failure_evidence.json): selected diagnostic/failure and
  separate cleanup evidence, preserving their scope limits.
- [audit.py](audit.py): stdlib-only reconstruction/audit against the original
  private input archive, original checksums, source checkout and diagnostic
  documents. It starts no model/GPU process and writes a new output directory.

The audit checked compressed archive size/SHA and every member payload; all raw
absolute/paired summary fields; seeded path order, IDs/pair completeness and
non-overlap; wall=submit+completion-sync; identical per-path paired clock metadata;
all source hashes against the recorded commit; and matching per-batch trace
hash/seed between eager and Graph. The raw trace tensor was not exported or
regenerated, so its exact bytes are not independently recovered here. Verified
source establishes that the paired paths use the same preallocated IDs.

Raw member contents were scanned for IP/SN/private home/workspace paths and
credential/capability patterns. The original tar's Unix owner metadata was
normalized before publication. The previous [M1 screen](../20261021_m1_screen/README.md)
is preserved unchanged. Future fixed-preparation runs and full HTTP A/B results
must be reported as separate runs with their own identities and all outliers.

Original compressed archive SHA256: `59082690b22d261c28a52e323fe89dacf93bdc44f6a524bd60206301f207dad4`.

Published normalized archive SHA256: `fcdf9f1d8907a976427f8d08eb617c67eb15f3ae9ad6c9d8a5298c8b5dcadf5c`.
