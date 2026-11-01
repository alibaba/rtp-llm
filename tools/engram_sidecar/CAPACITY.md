# Explicit capacity policy for shared Engram

The capacity extension preserves the small checkpoint/bootstrap defaults.
Increasing a ceiling authorizes a size; it does not reserve resources or prove
that an Owner's NUMA node, container, driver or fabric can support that size.
The rebuilt bridge passed nine native GPU tests on each of two hosts, followed
by a real cross-host 4.125 GiB Triton lookup probe. The complete 188.83 GiB real
Engram generation subsequently passed [cross-host lookup](framework/REAL_CHECKPOINT_LOOKUP.md).
A subsequent [complete-model eager HTTP smoke](framework/FULL_MODEL_HTTP.md) passed;
performance and production-service acceptance remain separate gates.

| Layer | Default | Maximum accepted by this development implementation |
| --- | ---: | ---: |
| Checkpoint plan, one padded allocation | 1 GiB | 128 GiB, explicitly supplied |
| Consumer bootstrap, one padded allocation | 1 GiB | 128 GiB, explicitly configured locally |
| Consumer bootstrap, unique allocations total | 8 GiB | 256 GiB, explicitly configured locally |
| Fixture Owner loading budget | 64 MiB | 256 GiB, explicitly supplied |
| Native allocation/import hard ceiling | 128 GiB | 128 GiB |
| One native load chunk | 64 MiB maximum | Unchanged; checkpoint loader defaults to 8 MiB |

The low-level native API receives an explicit allocation size; Python enforces
the small default policy before allocation/import. The native hard ceiling
still applies to direct callers. Rounded allocation sizes and generation totals
are checked before creating any Owner. Loading rechecks the immutable checkpoint
plan, granularity and the actual loaded native module's limit. Consumer bootstrap
checks all declared allocations against its local policy and loaded native
module before importing the first one.

## Configuration

Only the trusted local launcher sets these consumer environment variables:

```bash
export RTP_ENGRAM_MAX_ALLOCATION_BYTES=137438953472  # 128 GiB
export RTP_ENGRAM_MAX_TOTAL_ALLOCATION_BYTES=274877906944  # 256 GiB
```

Unset variables retain the defaults. Values must be positive canonical decimal
byte counts within the hard ceilings; signs, units, whitespace and leading zeroes
are rejected. A bundle/manifest cannot raise the local budget. These settings
do not change checkpoint identity, CUDA visibility or IMEX permissions.

Owner tooling passes `max_allocation_bytes` and `budget_bytes` explicitly to
`inspect_checkpoint`; the framework Owner exposes `--max-allocation-bytes` and
`--budget-bytes`. The defaults remain 1 GiB and 64 MiB respectively. Setting only
the consumer environment does not raise the Owner's budget. Rebuild and deploy
the native bridge on both sides before attempting larger mappings; an older
1 GiB bridge must reject them.

## Layout and resource accounting

The layout is unchanged: one allocation per layer, with contiguous FP8 weight
bytes followed by group-32 E8M0 scale bytes. Each tensor occupies one complete
manifest shard and receives a contiguous ATen view. Checkpoint files may be
sharded across tensor names, but a tensor split across files, multiple backing
shards per tensor and multiple Owners are unsupported. No new per-request
lookup branch, RPC, table copy or indirection is introduced by the larger caps.

For example, 384,006,168 and 384,016,682 rows, each with 256 weight bytes and
8 scale bytes, require respectively 101,377,628,352 and 101,380,404,048 bytes.
Their sum is approximately **188.8331 GiB** before VMM padding. If the runtime
granularity is 2 MiB, the total padded backing is 202,758,946,816 bytes. The
actual driver granularity must be queried, not assumed.

All of this backing is allocated on the Owner's selected local NUMA node.
NVL72 aggregate memory does not establish this local capacity. Check NUMA
availability, allowed memory nodes, container memory limits, existing workloads
and source-checkpoint disk space before a large run. A container's ordinary
memory quota alone does not establish either availability or enforcement for
this approximately 189 GiB EGM generation. In the successful real-table run,
the two EGM mappings had exactly 202,758,946,816 resident bytes while the verified
Owner container cgroup reported about 10.23 GiB current and 14.11 GiB peak.
This is a measured driver/kernel-specific observation, not a claim about all
ancestor cgroups or platform versions; see the [accounting evidence and limits](framework/REAL_CHECKPOINT_LOOKUP.md#platform-observation-resident-egm-and-container-accounting).
Do not assume every EGM byte is charged to this container's memory.current.

Budget measures unique padded EGM backing, counting shared weight/scale storage
once. It excludes source-file page-cache residency, bounded staging chunks,
other model weights and runtime/driver metadata. The current Owner zeroes the
complete allocation before chunked loading; each tensor is fully hashed before
publication. Bounded chunks do not guarantee bounded total RSS or file cache.
The loader now reads exact bytes through a checked file descriptor without
keeping full tensor mmap views. The opt-in `discard_file_cache=True` (Owner CLI
`--discard-private-checkpoint-cache`) advises release of private immutable
source files before allocation and already consumed complete pages during
streaming. It never drops global cache, and the advice does not guarantee
reclamation. Defaults leave checkpoint cache untouched.

## Acceptance gates

CPU execution in a separate development-container directory passed 68 tests
with no skips: capacity policy/arithmetic 9, bootstrap 36 and checkpoint loading
23. The checkpoint tests use real torch/safetensors with an accounting fake
native backend. The official-size metadata test creates only small JSON files.
These are direct Python test runs, not a new Bazel or native/GPU result.
The subsequent raw-reader revision passed 38 real Torch/safetensors CPU tests
without skips, including byte identity, >4 GiB seeks, partial reads, file
mutation, cache-advice ordering/bounds/failure and allocation rollback.

The subsequent enlarged-native GPU runs passed nine tests per host, including
100 attach/view/drain/close cycles and constructor rollback. Both ended with all
native resource counters zero. The [large-offset probe](framework/LARGE_OFFSET_PROBE.md)
then passed across two hosts/containers: exact BF16 bytes for int64 and int32
indices around the 4 GiB boundary and table tail, zero PyTorch HBM allocation on
attach, 7,680 bytes peak PyTorch allocation and 2 MiB driver free-memory change.
Consumer and Owner cleanup both returned every native resource counter to zero.

1. CPU policy, byte arithmetic, rounding/overflow and old-default rejection;
   explicit large-policy metadata accepted without allocating a large table.
   Malformed local settings, excessive total budget and bundle-supplied budget
   fields fail before native imports.
2. Rebuilt native tests and explicit capacity stepping across 1 GiB and 4 GiB.
   Validate reads around the 32-bit byte-offset boundary, at the tail, and for
   high-row IDs through both raw GPU reads and the unchanged Triton lookup.
3. Real checkpoint inspection and resource preflight, then full Owner loading
   with trusted tensor hashes. Cross-host consumers verify sampled rows and
   real Engram layer output without owning full embedding checkpoint tables.
4. Stop consumers before Owner teardown; require native counters to return to
   zero. Separately account for driver memory, container RSS and file cache.

The CPU gates do not establish driver allocation feasibility, performance,
full-model HTTP readiness or the additional 100-microsecond latency target.
