# Full-model DPEP8 Graph A/B, 2026-10-29

Functional validation passed. **The additional 100 us decode-time target is not
established, and the long-input request-mean comparison exceeds it.** This is
one chronological A1–B1–B2–A2 cycle, not a production SLO qualification.

## Configuration and identity

- Release baseline `feature/dsv41-release-20261019`, commit
  `cb42b084a8a6350daa70897555f18d5f1b47f58e`; experiment source `f6b880c25`.
- Real complete V4.1 Flash checkpoint, 40 layers, two Engram layers (1, 14),
  approximately 188.83 GiB of logical raw FP8/E8M0 table data. No layer hack,
  fixture substitution or speculative model.
- Two independent single-host CP4 prefill instances; one two-host decode instance,
  **TP1/DP8/EP8**, four GPUs per host. MegaMoE, RDMA P→D, BF16 activations, FP8 KV.
- Decode Graph batch=1 on all eight ranks; prefill eager. Capture and synchronized
  replay checks were verified against the current worker PIDs for each phase.
- A uses the ordinary `_PinnedSharedTable` + `cudaHostRegister` path. B imports
  a complete virtual table from two HOST_NUMA/FABRIC Owner shards on distinct
  hosts. Same synchronous GPU lookup kernel in both modes; CPU lookup and async
  lookup are disabled. This does not measure the default CPU-prefill path.
- P and D use separate Owner generations. Bootstrap is the experimental private
  file bundle; **this run does not validate production RPC/service discovery**.
- The same native-library hashes and common launch settings were checked between
  phases. Each phase is a fresh two-host D startup; P stays unchanged.

## Protocol and observed result

All requests enter the same P frontend. Inputs contain 12, 652 or 2572 rendered
prompt tokens; output is forced to 256 tokens, temperature=0, seeds=11/23,
`reuse_cache=false`. Each phase/length has one warmup and two measured requests.
All **36 requests succeeded**, matched quartets have exactly equal generated
text and usage, and every request has `reuse_len=0`. No parser warnings or
outlier removal. Warmups are retained in the companion JSON but excluded from
the following measured-request averages.

Timing is `(aux.cost_time - aux.first_token_cost_time)/(output_len - 1)`:
**engine stream wall-time mean per output token**, not kernel duration or
individual decode-step latency. Each length has four A requests and four B
requests. Paired differences are B1−A1 and B2−A2 at matching seed positions.

| Input tokens | Pinned mean ms/token | EGM mean ms/token | Mean delta us/token | Paired delta range us/token | Pairs above 100 us |
|---:|---:|---:|---:|---:|---:|
| 12 | 19.197825 | 19.232502 | 34.676 | -70.788 to 231.886 | 1/4 |
| 652 | 19.213561 | 19.232538 | 18.977 | -61.557 to 101.882 | 1/4 |
| 2572 | 19.276229 | 19.663886 | 387.657 | -283.780 to 707.918 | 3/4 |

The short-input block means are small, but individual paired request means still
exceed 100 us. The 2572-token block has a +387.657 us/token mean difference.
Negative deltas and substantial within-phase variation are also retained. These
observations do not isolate fabric memory latency from scheduling, communication,
cache state or system drift. No per-step p99 or confidence bound is claimed.
See [sanitized samples and archive hashes](DPEP8_GRAPH_ABBA_20261029.json) and the
[measurement protocol](HTTP_DECODE_BENCHMARK.md). Complete private raw SSE/launch
artifacts are retained separately; no handles, endpoints or checkpoint bytes are
included in this public result.

## Placement and memory qualifications

Both local tables are shared by four workers per host, not copied per worker.
The default pinned allocator produced mixed NUMA placement (64 KiB pages):

| Host | Layer | NUMA0 pages | NUMA1 pages |
|---|---:|---:|---:|
| D0 | 1 | 682996 | 863904 |
| D0 | 14 | 376704 | 1170239 |
| D1 | 1 | 727128 | 819772 |
| D1 | 14 | 231907 | 1315036 |

A2 had exactly the same page counts as A1. This is a local-host pinned baseline,
**not a per-GPU local-NUMA optimum**. D0's Owner shard uses GPU1/NUMA0 for both
layers; D1's shard places layer1 on GPU1/NUMA0 and layer14 on GPU3/NUMA1.
Requests therefore combine local/cross-NUMA and cross-host row reads.

The complete local tmpfs tables were retained, without A consumer processes,
between phases to preserve A1/A2 placement. Consequently B's total host RAM
footprint in this cycle is not a measurement of production memory savings.

An earlier local load attempt was stopped by a resource watchdog before ready:
D0 reached its 384 GiB cgroup limit while loading tables and source pages; no OOM
kill occurred. With ample host memory, only that experiment container's limit
was raised to 512 GiB. D1 remained at 320 GiB. The successful first A phase
reused completed tables and is not a cold-load benchmark. No other workload or
GPU was reset, and no global cache drop was used.

## Remaining acceptance work

- Correlate full decode iteration timing with Graph/kernel/communication traces;
  diagnostic profiling is separate from these unprofiled timing samples.
- Repeat independent ABBA cycles with more requests, controlled workload and
  recorded device clocks/CPU scheduling before tail claims.
- Compare explicitly controlled NUMA placement and additional concurrency/batch
  sizes. This four-host experiment does not establish NVL72-wide scaling.
- Qualify the production RPC path, owner lifecycle/failover and memory footprint
  independently. Functional chat and Graph checks do not establish those claims.

## Separate eight-step trace check

After timing, one diagnostic request per backend used the 2572-token prompt and
8 forced output tokens with Kineto enabled. Both active decode rank0 traces
contain eight `cudaGraphLaunch` calls and sixteen `_lookup_host_kernel` calls,
correlated two lookups per Graph launch. Thus online Graph replay and the actual
lookup kernel are observed, not inferred only from startup flags.

In these **profiled short diagnostics**, the sum of the two lookup kernels per
Graph was 20.800–26.176 us for pinned and 17.984–19.424 us for EGM. This is a
different, small sample from the unprofiled 256-token ABBA requests. It neither
proves EGM is generally faster nor attributes the long-input TPOT regression.
The full Graph kernel spans and leading cumulative-duration kernels are retained
in the JSON; overlapping kernel durations must not be summed as wall time.
Further correlated, representative profiling is required before assigning a
root cause. Profiler windows were completed before final service verification.
