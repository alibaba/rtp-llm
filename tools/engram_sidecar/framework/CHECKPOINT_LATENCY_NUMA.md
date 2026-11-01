# Real-table lookup comparisons

`checkpoint_latency.py` compares the same `HostEngramEmbedding` kernel and row
IDs against a complete local `_PinnedSharedTable` and an imported EGM table.
It verifies BF16 results against independently read checkpoint rows, including
Graph replay with changing inputs. This is a component test, **not a full
decode-step or HTTP latency acceptance test**.

## Placement and admission

CPU affinity alone does not override an inherited Linux memory policy. For
example, a container with `prefer-many:0-1` can allocate a table on both NUMA
nodes despite binding its thread to a CPU on node 1. Select an authorized GPU,
read its `CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID`, and use an explicit matching policy:

```bash
numactl --membind=1 python -m tools.engram_sidecar.framework.checkpoint_latency \
  --device 2 \
  --checkpoint /absolute/consumer-checkpoint \
  --source-checkpoint /absolute/owner-checkpoint \
  --bundle /absolute/decode-bundle/bundle.json \
  --private-inputs /absolute/decode-bundle/oracle.private.json \
  --output /absolute/new-result-directory \
  --layers 1,14 --layer-mode single --batches 1,8 \
  --modes eager,graph --distributions hot,random --samples 10000 --warmup 100
```

GPU 2 / NUMA 1 above are examples, not portable topology assumptions. Actual
resident pages are checked in `/proc/self/numa_maps`; a mixed-node baseline is
rejected before timing. Record the Owner shards' placement separately: sharing
a host with one shard does not mean that shard is GPU-local.

Loading needs both the full pinned table and source file-cache headroom. The
default admission check counts only NUMA free memory. On a dedicated experiment,
`--allow-numa-cache-reclaim` may additionally count clean inactive file cache,
excluding dirty/writeback pages, while preserving the configured headroom
margin. This is an estimate, **not a memory reservation or guaranteed reclaim**.
Cgroup/host budgets, post-load free memory and actual NUMA checks still apply.

`--discard-private-source-cache` uses per-file `POSIX_FADV_DONTNEED` before and
after copying. It requires current-user, mode-0400 files in a mode-0700 source
directory. It does not invoke global `drop_caches` or change checkpoint bytes.
If an experiment constructed its pinned checkpoint with hardlinks, add
`--allow-source-hardlinks` only after verifying every alias belongs to the
experiment and no active model loader depends on it. Advice affects every
alias of that inode; it does not guarantee eviction. The default rejects
hardlinked inputs.

## Interpretation and cleanup

- Record kernel-event time, CPU submission time and synchronized completion
  separately. Eager event intervals can include launch gaps; HTTP chunks are
  not GPU step timestamps.
- Use the same batch/IDs/kernel and preserve all paired samples and outliers.
  Hot and uniform-random rows are synthetic distributions, not production
  request distributions. Report both.
- Component Graph success does not establish RTP engine Graph support. Test
  the full model separately, with the same storage lifetime and parallelism.
- Successful cleanup must leave native resource counters at zero. Failure
  retains references until process exit. Before removing a stale experiment
  tmpfs table, verify its exact lease PID has exited, ownership and size match,
  and no other live lease exists. Never bulk-delete shared tables.

For full-model comparisons, keep P fixed and vary only D storage, use matched
warmups/requests and A-B-B-A ordering. `http_decode_benchmark.py` disables prefix
reuse per request, records raw SSE events and engine timing, and does not claim
that request-average TPOT proves a per-step 100-microsecond bound.
