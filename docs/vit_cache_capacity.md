# ViT embedding cache: GPU and CPU tiers

Each `MMProcessEngine` has a shared embedding cache for synchronous and asynchronous
requests. Completed results use two independent byte budgets. Pending computation is
still deduplicated by media key.

| Environment variable | CLI argument | Per-process default |
| --- | --- | --- |
| `MM_CACHE_GPU_MAX_BYTES` | `--mm_cache_gpu_max_bytes` | 21474836480 (20 GiB) |
| `MM_CACHE_CPU_MAX_BYTES` | `--mm_cache_cpu_max_bytes` | 214748364800 (200 GiB) |
| `MM_HASH_KEY_CACHE_MAX_BYTES` | `--mm_hash_key_cache_max_bytes` | 268435456 (256 MiB, CPU) |

```bash
export MM_CACHE_GPU_MAX_BYTES=21474836480
export MM_CACHE_CPU_MAX_BYTES=214748364800
```

These are limits, not startup reservations. Each ViT process has its own limits;
multiple processes on one host/GPU multiply the possible aggregate cache residency.
Negative limits are rejected when constructing the cache.

## Lookup and eviction

1. New GPU results enter the GPU tier if they fit.
2. GPU capacity pressure moves the least recently used GPU result to CPU.
3. CPU capacity pressure evicts the least recently used CPU result completely.
4. A CPU hit restores each tensor's original device and promotes the result to GPU.
   Restoring a result does not recompute ViT or change its cache generation/hash IDs.
5. A result larger than the GPU budget goes directly to CPU if it fits there. On a
   hit it is restored for the current request but remains resident only in CPU.
6. A result too large for either tier bypasses caching without flushing unrelated
   entries. The computing request and existing waiters still receive their result.

The GPU and CPU tiers have separate LRU orders. Metadata/directory probes do not
update recency or perform tensor transfers. A CPU-resident result is a valid
metadata hit while its feature hashes and generation are available in the hash cache.

`MM_CACHE_GPU_MAX_BYTES=0` disables GPU residency while allowing CPU caching; a hit
still restores GPU tensors for consumption. `MM_CACHE_CPU_MAX_BYTES=0` disables CPU
residency, so GPU victims are dropped. Setting both to zero disables the embedding
cache. Hash caching can be disabled independently with `MM_HASH_KEY_CACHE_MAX_BYTES=0`.

## Accounting and transfer behavior

Embedding charges include embeddings, position IDs, and tensor-valued extra outputs,
recursing through tuples, lists, and dictionaries. GPU and CPU storage are charged
separately according to the actual device, including CPU metadata attached to a hot
GPU entry. If such metadata exhausts the CPU budget after all cold entries are gone,
the cache also drops hot entries that retain CPU storage.

A tensor view retains its backing storage; that allocation is charged in full.
Aliased storage is counted once within an entry, while sharing between entries is
conservatively charged to each entry. Device transfers compact views and recompute
the destination charge. Shapes, values, dtype, nesting, and original device placement
are preserved; strides and aliases between distinct views need not be preserved.

Transfers use ordinary pageable CPU memory and blocking copies. Producer CUDA events
ensure completed writes are visible across threads and streams. Residency changes and
transfers are serialized per cache to avoid unbounded concurrent staging allocations.
The index lock is released during copies, so metadata probes and unrelated GPU hits
can continue. CPU hit wait time includes acquiring the transfer lock; an already
started CUDA copy cannot be interrupted by the request timeout.

Eviction swaps the cached result instead of modifying tensors held by active requests.
Those requests may retain GPU storage until they finish. A failed CPU spill drops only
the affected cache entry and preserves the successful computation for current waiters.
A failed GPU restoration raises to that requester but retains the CPU value for retry.

The limits bound cache-owned tensor residency, not total GPU allocation or process RSS.
Model weights, workspaces, pending results, active request references, one in-progress
transfer, RPC/RDMA buffers, Python embedding-entry metadata, and allocator reservations
are outside these limits. No pin/lease or memory admission for pending computation is
introduced by this change.

`stats()` exposes `gpu_resident_bytes`, `cpu_resident_bytes`, per-tier entry counts and
limits, `gpu_hit`, `cpu_hit`, `promotion`, `demotion`, and `transfer_error`. Existing
aggregate resident-byte/token gauges continue to report both tiers combined.

## Hash cache and migration

The hash cache remains a separate CPU LRU. It owns CPU int32 hash tensors and charges
hash payload plus an estimate of key/value Python metadata. It does not retain GPU
embedding storage. Evicting an embedding does not immediately discard hash history,
but a metadata query requires the matching embedding generation in either tier.

`/mm_cache/keys` publishes up to the 100,000 most recent hash-cache keys. This bounds
the directory response, not the number of locally retained entries.

`MM_HASH_KEY_CACHE_ITEM_NUM` / `--mm_hash_key_cache_item_num` is replaced by
`MM_HASH_KEY_CACHE_MAX_BYTES` / `--mm_hash_key_cache_max_bytes`.

The shared embedding cache now uses explicit GPU and CPU byte limits instead of
estimating a budget from `MM_CACHE_ITEM_NUM` and the model shape. `MM_CACHE_ITEM_NUM`
controls only the legacy model-internal cache; it does not size or disable the shared
embedding cache.
