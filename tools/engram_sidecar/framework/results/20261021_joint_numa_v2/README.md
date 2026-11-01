# Real two-layer lookup — source NUMA staging v2

**The complete 188.833 GiB two-layer baseline was loaded with verified GPU-local
placement, actual joint Graph output checks passed, and the consumer native
resources closed with all eight counters zero. The full decode latency SLO is
still not measured, and the component threshold has observed failures.** With
scratch disturbance, the priority M=1 Graph random case has three increments
above 100 µs, reaching **1,329.984 µs**, even though its P99 increment is only
5.088 µs. The separate normal matrix has no M=1 GPU exceedance in 20,000 pairs,
but has 11/140,000 across M=1..64, with maximum 1,490.176 µs. Wall time has
additional spikes. Correct setup/cleanup, a small P99 and successful finite
subcases do not establish an all-request performance guarantee.

Source commit: `415ef274e19e45f71e705802500ed5488a057289`; main base:
`492ecf45b3ef55b22fd4882622abfdb2e2f664f2`. The change places untimed source file
cache on a second CPU NUMA node, then calls main's unchanged `_PinnedSharedTable`
and unchanged lookup kernel. It does not replace the baseline with remote memory,
change the table format, or add a full-table HBM copy.

## Run ledger

| Run | Coverage | Status and limits |
|---|---|---|
| [joint-M1-numa-screen-v2](joint-M1-numa-screen-v2/README.md) | Actual layers 1+14 together; M=1; eager/Graph; hot/random; 100 pairs each | Completed, native counters zero; eager exceedance preserved; Graph finite screen only |
| [joint-matrix-numa-run2](joint-matrix-numa-run2/README.md) | Actual layers 1+14 together; M=1..64; eager/Graph; hot/random; 10,000 pairs each | Completed, native counters zero; 11 Graph GPU positive exceedances; all 1,183 paired exceedances retained |
| [joint-M1M2-disturbed-run3](joint-M1M2-disturbed-run3/README.md) | Actual layers 1+14 together; M=1/M=2; eager/Graph; hot/random; 10,000 pairs each; 2×L2 scratch | Completed, native counters zero; 7 Graph GPU exceedances including 3 at M=1; all 397 paired exceedances retained |

Each process run has its own seed, trace identity and complete raw data. No
cross-run sample pairing or causal scratch-effect estimate is implied. This
phase does not add a layer-14 single-layer matrix; both actual layers are covered
by the joint run. The
[earlier failed layer-1 matrix](../20261021_layer1_matrix_partial/README.md)
remains unchanged: its measured Graph anomaly and mixed-NUMA layer-14 failure
must not be removed or treated as explained by this preparation change.

## What the joint interval means

One local or remote call executes both actual Engram lookups serially on the
same CUDA stream and returns both outputs. The GPU event/Graph window surrounds
the **whole two-lookup chain**; the result is not a sum of separately estimated
P99 values. Each layer has its own preallocated ID trace, shared between local
and remote paths. Both actual outputs are checked independently against the
original checkpoint oracle at the recorded correctness checkpoints.

The two inputs are independent: there is no intervening model-layer computation
or real model data dependency between the lookups. Hash/history, WKV/projection,
gate/residual, attention/MoE/EP, scheduling and HTTP are excluded. Covering all
Engram tables is different from timing an entire Engram model stage or a complete
decode step. Full-model [HTTP A/B](../../HTTP_DECODE_BENCHMARK.md) remains a
separate validation path, and its request means also do not establish per-step
GPU tail latency.

Complete table capacity is also different from the active lookup working set.
This M=1 screen visits only about 1.21 MiB of logical weight/scale data per
backend across its 100 random pairs, before duplicate rows or cache-line
overfetch. Eager and Graph reuse that trace; runtime L2 is about 129.25 MiB and
no disturbance was used. The reported random results therefore have
`cache_claim=uncontrolled`, not a cold-memory guarantee. The separate
scratch-disturbed run retains `scratch_disturbed_not_guaranteed_cold`; scratch traffic alone
does not prove cold lines or reproduce full-model cache pressure.

## Audit and publication

[audit_run.py](audit_run.py) performs CPU-only archive/member SHA checks, exact
source-commit checks, complete source/pinned-page placement and memory-admission
checks, same-trace and balanced-order checks, and independent recomputation of
absolute/paired statistics. Every original raw member is preserved byte-for-byte;
only tar/gzip owner, mode and timestamp metadata is normalized. Each run retains
its original and published archive SHA values, all positive exceedances, negative
outliers and calibration. No calibration or tail is subtracted.

The source-prefault checks are observations, not a reservation or a guarantee
against eviction. `DONTNEED` is not assumed to free pages. `memory_allocated`
reports PyTorch allocator bytes; it does not measure all driver/NVML HBM use.
Pointer-alias checks separately establish the actual host-backed tensor views.
Preparation phases have memory snapshots but no elapsed-time timestamps, so no
source-loading speed claim is made.
