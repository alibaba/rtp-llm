# Optional prefill memory cache

The mock keeps GPU execution capacity and host prefix retention separate. Enable
`prefill.memory_cache` in the performance JSON; it is disabled when absent and
never creates a decode memory cache.

```json
{"prefill":{"memory_cache":{"enabled":true,"capacity_blocks":32768,"read_ms_per_block":0.1}}}
```

`capacity_blocks` counts complete logical cache-key blocks, using the engine's
prefill block size. For CP-sharded engines, convert physical block tokens through
CP size before comparing capacities. No KV tensors or corresponding host RAM are
allocated: this is a bounded metadata model.

At selection, match the GPU prefix first, then extend it through memory until the
first missing key. Only this combined prefix reduces `computeTokens`. A memory
hit still requires GPU allocation; it cannot increase GPU capacity or bypass its
reserve watermark. FIFO candidates re-match when selected. Successful prefill
writes complete prefix blocks to memory and retains computed GPU blocks through
the normal connector lifecycle. Cancellation or injected prefill failure does not
populate memory. Master cache status includes the union of GPU and memory keys,
while its total/available execution capacity remains GPU-only.

Memory uses access-order LRU. Reads and duplicate writes refresh recency without
resetting insertion time. Eviction lifetime measures insertion-to-eviction age,
not time since last access and not an eviction timeout. Capacity is not adjusted
to force this metric to a target. Engine crash clears both caches.

`read_ms_per_block` adds a modeled host-to-device copy delay for memory-only hits,
outside `rtp_llm_model_forward_us`. Zero means idealized instantaneous copies.
This first model snapshots metadata atomically: it does not simulate host buffer
pins, overlapping DMA, write queues, partial blocks, or a disk tier. These limits
must be considered when comparing a production memory-plus-disk deployment.

## Metrics

- `rtp_llm_stream_cache_device_reuse_length` and
  `rtp_llm_stream_cache_memory_reuse_length`: non-overlapping reused token counts.
- `rtp_llm_kv_cache_hit_rate`: combined reused tokens / input tokens, percent.
- `rtp_llm_kv_cache_memory_cache_status_total_block_num` and
  `...allocated_block_num`: configured capacity and retained key count.
- `...available_block_num` includes reclaimable cached blocks; `...used_ratio`
  measures pinned capacity, not cache occupancy. Atomic metadata copies have no
  persistent host pins. Use `mock_memory_cache_occupancy_ratio` for occupancy.
- `mock_memory_cache_total_tokens`, `mock_memory_cache_read_blocks_total`,
  `mock_memory_cache_evicted_blocks_total`: capacity and cumulative counters.
- `rtp_llm_kv_cache_evicted_block_lifetime_ms`: filter `scope=gpu,backing=device`
  or `scope=memory,backing=memory`; never compare an unfiltered mixture.

Validate a deployment using a stable traffic window after warm-up. Record batch
size, waiting, forward, GPU/memory reuse separately, and eviction counter deltas.
A cache with no evictions has no eviction-lifetime observation, not a zero lifetime.
