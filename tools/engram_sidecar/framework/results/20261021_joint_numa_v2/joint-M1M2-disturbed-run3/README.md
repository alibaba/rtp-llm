# Joint M=1/M=2 with scratch disturbance — 2026-10-21

**The priority M=1 Graph lookup chain has observed increments above 100 µs.**
Its random-ID GPU P99 increment is only **5.088 µs**, yet pairs 1508–1510 have
increments **1,199.776 / 1,329.984 / 901.888 µs**. All three are retained.
Across M=1 and M=2, there are **7/40,000 Graph GPU-event exceedances** and
**40/40,000 Graph wall exceedances**. This component threshold is
`FAIL_OBSERVED`; full decode is `NOT_MEASURED`.

The independently restarted process completed four cases, each hot/random with
10,000 paired calls: **320,000 raw records / 80,000 pairs**, 16 paired statistical
rows, and all eight native resource counters zero. This is separate from the
[normal matrix](../joint-matrix-numa-run2/README.md), with seed **20261023** and
its own trace identity. Within each run local and remote receive the same trace;
across these runs seeds and process state differ, so their difference is not a
controlled quantitative causal estimate of the scratch effect.

## What was changed and what was measured

The source remains `415ef274e19e45f71e705802500ed5488a057289`, all seven recorded
source hashes verified. Both actual layer tables still total **188.833 GiB**;
main's `_PinnedSharedTable` and the original FP8/E8M0 lookup kernel are unchanged.
The source CPU/NUMA preparation, external EGM Owner identity and format are the
same as the normal matrix.

`--l2-disturb` allocates **271,056,896 scratch bytes**, exactly twice runtime L2
(**135,528,448 bytes**). Verified code issues `scratch.add_(1)` before **each**
local or remote path, then synchronizes the stream before starting either timing
clock. Scratch execution and input preparation are excluded from the measured
interval. This is a cache perturbation experiment, not proposed per-request
production work or an estimate that includes the perturbation cost.

The only valid cache description is
**`scratch_disturbed_not_guaranteed_cold`**. The operation does not prove every
Engram cache line was evicted, control all hardware cache levels or recreate
whole-model traffic. Hot still means fixed IDs; random means uniformly selected
rows across the complete tables. Neither is a production token distribution,
and full allocated capacity is not the active working set. Two serial independent
lookup calls are measured in one guarded Graph/event window; real intervening
model computation, hash/history, WKV/projection, gating, MoE/EP, scheduling and
HTTP remain excluded.

## Placement, correctness and cleanup evidence

| Layer | Full pinned 64 KiB pages, all NUMA0 | Full source 64 KiB pages, all NUMA1 | Actual CUDA weight/scale aliases | PyTorch allocated delta |
|---|---:|---:|---|---:|
| 1 | 1,546,900 | 1,549,304 | both true | 0 bytes |
| 14 | 1,546,943 | 1,549,346 | both true | 0 bytes |

All complete source pages, including rounded tails, are present on NUMA1 both
before and after each baseline copy; pinned pages are wholly GPU-local NUMA0.
The three admission phases per layer were independently recomputed for complete
source/table futures, 16 GiB headroom and an 8 MiB temporary buffer, across
global available memory, every visible finite cgroup, shm and both NUMA nodes.
The minimum remaining slack is **43.775 GiB** on local NUMA after layer-14
prefault; the corresponding cgroup surplus is **75.046 GiB**. These are
snapshots, not reservations or peak bounds. Source mmap is not mlock, DONTNEED
success is not a reclamation guarantee, and no loading-duration timestamps exist.
Alias checks and zero PyTorch allocator delta do not measure all driver HBM.

For each case, both actual local/remote outputs of both layers match the original
checkpoint CPU BF16 oracle at random trace indices 0 and 1 before timing, hot
index 0 afterward, and random index 9,999 afterward. Graph checks use actual
replay output; they are not checks of every timed sample. Actual eager/Graph
trace hashes match for each layer and batch. Complete native cleanup applies to
this consumer only; it is not proof that the separate Owner or system file cache
was destroyed. Detailed source/memory evidence is retained in [report.json](report.json).

## Paired latency statistics

Values are remote minus local in µs, nearest-rank quantiles, retaining negative
samples. Every row has 10,000 pairs in one epoch/process; no tail is removed,
no calibration is subtracted and no independent-restart confidence bound is
claimed. Eager events include CPU dispatch gaps. Captured external Graph events
exclude the pre-submission gap but do not isolate exclusive kernel or fabric
service time. CPU wall is submit plus completion-call time, with both retained.

| Mode | M | IDs | GPU P50 Δ | GPU P99 Δ | GPU max Δ | GPU >100 | Wall P99 Δ | Wall max Δ | Wall >100 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| graph | 1 | hot | 5.824 | 7.520 | 14.304 | 0 | 17.023 | 2389.512 | 12 |
| graph | 1 | random | 1.568 | 5.088 | 1329.984 | 3 | 12.928 | 1335.172 | 8 |
| graph | 2 | hot | 6.144 | 8.224 | 206.944 | 1 | 16.609 | 898.211 | 8 |
| graph | 2 | random | 1.632 | 5.632 | 165.600 | 3 | 12.608 | 2145.414 | 12 |
| eager | 1 | hot | 25.440 | 91.232 | 109856.611 | 87 | 96.384 | 109892.328 | 92 |
| eager | 1 | random | 22.944 | 85.248 | 1231.616 | 73 | 90.816 | 1237.220 | 83 |
| eager | 2 | hot | 25.536 | 95.168 | 1172.800 | 89 | 95.872 | 1179.268 | 95 |
| eager | 2 | random | 23.264 | 83.936 | 1089.568 | 79 | 87.648 | 1096.483 | 86 |

Eager totals are 328/40,000 GPU-event and 356/40,000 wall exceedances. There are
**397 distinct pairs** with either positive delta >100 µs. Every one is retained
in [all_positive_threshold_pairs.jsonl](all_positive_threshold_pairs.jsonl),
along with the complete raw archive containing negative tails and non-outliers.
Small P99 does not establish the requested per-call upper bound.

## All seven Graph GPU-event exceedances

| M | IDs | Pair | Local GPU | Remote GPU | GPU Δ | Remote submit | Remote completion | Remote involuntary switches |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | random | 1508 | 21.280 | 1221.056 | 1199.776 | 27.393 | 1225.123 | 0 |
| 1 | random | 1509 | 21.888 | 1351.872 | 1329.984 | 35.264 | 1352.004 | 0 |
| 1 | random | 1510 | 19.808 | 921.696 | 901.888 | 31.936 | 923.555 | 0 |
| 2 | hot | 2744 | 13.728 | 220.672 | 206.944 | 222.368 | 28.737 | 0 |
| 2 | random | 4374 | 37.568 | 174.848 | 137.280 | 203.041 | 7.488 | 0 |
| 2 | random | 4744 | 19.488 | 185.088 | 165.600 | 191.328 | 20.128 | 1 |
| 2 | random | 8803 | 17.536 | 155.712 | 138.176 | 189.729 | 41.088 | 0 |

For the three consecutive M1 random anomalies, remote submission is only
27–35 µs while GPU event and completion-call intervals are roughly 0.9–1.35 ms.
Both local and remote have zero counted thread switches for all three. The slow
interval is therefore not just a long host replay submission. This still does
not establish a fabric, GPU, driver or specific kernel cause; profiling of the
long intervals is needed. M2 random pair 4744 records one remote involuntary
switch, but that correlation does not attribute its whole duration to the OS.

Large wall-only observations also remain visible. M1 Graph hot pair 1281 has
remote wall 2,482.920 µs (submit 2,464.424, completion 18.496) versus local wall
93.408 µs, giving a **2,389.512 µs wall increment**. Its GPU increment is
**-8.192 µs**, so that wall spike is not a corresponding GPU-service spike;
both sides have zero counted switches. In the opposite direction, M2 Graph
random pair 6543 has local wall 107,561.664 µs, mainly a 107,529.664 µs submit
call, while remote wall is 114.688 µs; wall delta is -107,446.976 µs. There is
one local involuntary switch, but causal duration remains unmeasured.

The local GPU baseline also has tails: M2 Graph random pair 8997 is 829.856 µs
locally versus 21.696 µs remotely, delta **-808.160 µs**, with no thread switches.
Eager M1 hot pair 6859 has GPU delta 109,856.611 µs and wall delta
109,892.328 µs, mostly in remote dispatch, with zero counted switches; eager
GPU events include CPU dispatch gaps. These samples are preserved, not used to
cancel positive violations or assigned automatically to OS/network behavior.

[anomaly_neighborhoods.json](anomaly_neighborhoods.json) retains ±2 same-epoch
pairs around every positive Graph GPU exceedance and selected extremes.
The M1 neighborhood is 0.320, **1,199.776**, **1,329.984**, **901.888**, 2.048 µs
for pairs 1507–1511; it is a three-pair burst with normal-size flanking intervals.
[Outlier examples](outlier_examples.json) retain additional positive/negative
samples and calibration. All raw records and controls remain available. This
run adds evidence that the M1 threshold is not yet met under this experiment;
it neither erases previous anomalies nor proves scratch caused them.

Separately captured, unperturbed timing controls comprise 200 event-only and
200 empty-Graph samples. Event-only GPU intervals span 3.296–3.360 µs, with
wall maximum 21.504 µs; empty-Graph wall maximum is 9.472 µs. Their short finite
run does not rule out workload-time instrumentation/system tails and is not
subtracted from any measured latency or outlier.

## Audit and archive

All **18 archive members** match original byte SHA values; 16 paired statistical
rows and their absolute/epoch distributions were independently recomputed from
all raw records. The audit also checked source commit, balanced paired order and
non-overlap, wall partitions, trace parity, actual full page placement, alias
flags, memory budgets, and all eight zero cleanup counters. Only tar/gzip
ownership, mode and timestamp metadata is normalized.

- [measurements.tar.gz](measurements.tar.gz): all original member bytes.
- [archive.json](archive.json): every member SHA and original/published archive identities.
- [report.json](report.json): full independent statistics and placement/budget audit.
- [all_positive_threshold_pairs.jsonl](all_positive_threshold_pairs.jsonl): all 397 threshold pairs.
- [outlier_examples.json](outlier_examples.json), [anomaly_neighborhoods.json](anomaly_neighborhoods.json): paired evidence.

Original archive SHA256: `98134abb0abe79f3d33176518d832efef51121b6a0deb89b0587378dd08ca57b`.

Normalized archive SHA256: `3443e9182659a6749ae985f43c11468838399d0d1de091f4042d9b9d63fed153`.

Complete-model local-pinned versus EGM HTTP A/B remains a separate experiment.
These joint real-table lookup results must not be presented as full decode
latency or converted to a model-step P99 by adding component quantiles.
