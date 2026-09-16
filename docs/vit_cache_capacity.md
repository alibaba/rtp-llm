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

These values size fixed per-process arenas allocated when `MMProcessEngine` is
constructed. The GPU arena is a single CUDA allocation, and the embedding and hash
CPU arenas are single host allocations. Multiple ViT processes on one host/GPU
therefore multiply the startup reservation. Negative limits are rejected. Configure
the GPU value below the memory left after model/runtime initialization or startup can
fail with CUDA OOM.

The URL data cache is unchanged and remains controlled by `URL_CACHE_ITEM_NUM`.

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

Tensor views are compacted while being copied into an arena, so a small embedding view
does not pin the complete backing batch allocation. Repeated references to the same
tensor object are stored once within an entry. Shapes, values, dtype, nesting, and
original device placement are preserved; strides and aliases between distinct views
need not be preserved.

Pool insertion and CPU spill use blocking copies where host memory is involved.
Producer CUDA events ensure completed writes are visible across threads and streams.
Residency changes and transfers are serialized per cache. The index lock is released
during copies, so metadata probes and unrelated GPU hits can continue. CPU hit wait
time includes acquiring the transfer lock; an already started CUDA copy cannot be
interrupted by the request timeout.

Pool views never escape the cache. Every hit returns an independent tensor copy, so an
evicted slot can be reused without changing data held by another request. CUDA events
make the returned copy ready before it crosses a thread/stream boundary and delay
internal slot reuse while residency-transfer copies are still in flight.

Eviction swaps the cached result instead of modifying tensors held by active requests.
Those requests may retain GPU storage until they finish. A failed CPU spill drops only
the affected cache entry and preserves the successful computation for current waiters.
A failed GPU restoration raises to that requester but retains the CPU value for retry.

The configured limits are the fixed arena allocations, while resident-byte metrics
report the used payload inside those arenas. Model weights, workspaces, pending results,
request-owned copies returned from cache hits, RPC/RDMA buffers, and Python metadata are
outside these limits. The fixed GPU arena prevents long-lived cache entries from
interleaving with variable ViT workspace allocations in the CUDA allocator.

`stats()` exposes `gpu_resident_bytes`, `cpu_resident_bytes`, per-tier entry counts and
limits, `gpu_hit`, `cpu_hit`, `promotion`, `demotion`, and `transfer_error`. It also
reports each pool's capacity, used, pending, free, and largest allocatable block bytes.
Existing aggregate resident-byte/token gauges continue to report both tiers combined.

## Hash cache and migration

The hash cache remains a separate CPU LRU, but its int32 tensors now reside in a fixed
CPU arena. Hash reads return copies so eviction and slot reuse cannot mutate an active
request. Its byte accounting includes hash payload plus an estimate of key/value Python
metadata. It does not retain GPU embedding storage. Evicting an embedding does not
immediately discard hash history, but a metadata query requires the matching embedding
generation in either tier.

`/mm_cache/keys` publishes up to the 100,000 most recent hash-cache keys. This bounds
the directory response, not the number of locally retained entries.

`MM_HASH_KEY_CACHE_ITEM_NUM` / `--mm_hash_key_cache_item_num` is replaced by
`MM_HASH_KEY_CACHE_MAX_BYTES` / `--mm_hash_key_cache_max_bytes`.

The shared embedding cache now uses explicit GPU and CPU byte limits instead of
estimating a budget from `MM_CACHE_ITEM_NUM` and the model shape. `MM_CACHE_ITEM_NUM`
controls only the legacy model-internal cache; it does not size or disable the shared
embedding cache.
