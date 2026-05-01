# DSV4 Memory-Cache Reuse — SWA Boundary State Analysis

Working notes for the `v4_flash_native_fp4_fp8_tp1_memory_reuse_cache_sm100`
smoke. Captures the diagnosis that motivated the SWA-snapshot fix
(Approach B) and the alternatives we ruled out.

## Setup

The smoke runs three queries against DSV4-Flash (TP=1, SM100_ARM):

| #   | tokens | extra_configs                              | expectation                       |
|-----|--------|--------------------------------------------|-----------------------------------|
| Q0  | 2765   | `enable_device_cache=false, memory=true`   | cold prefill                      |
| Q1  | larger | same                                       | cold prefill (different prompt)   |
| Q2  | 2765   | same (identical prompt to Q0)              | memory cache reuse, output == Q0  |

Server flags: `--reuse_cache 1 --enable_memory_cache 1
--memory_cache_size_mb 1024`. Backend env: `DSV4_USE_FRAMEWORK_KV=1
DSV4_DEBUG_REUSE=1`.

`enable_device_cache=false` is honored end-to-end —
`StreamCacheResource::enableDeviceCache()` AND-gates the per-request flag
with the server flag, so device cache stays off for every query in this
smoke even though `--reuse_cache 1` is set.

## Observed behavior

- Q2 reports `cached_tokens=2560`, and
  `local_reuse_len == memory_reuse_len == 2560`. So the framework is
  routing reuse strictly through the memory connector, not through the
  device-side `reuseCache()` path. Good — that exercises the path under
  test.
- But Q2's first token diverges from Q0's by token 3, and the rest of
  the response disagrees. Q0 and Q2 have byte-identical prompts (sha1
  c4acf5c1b6a9). The same divergence appears on the existing
  SM100_ARM golden, so it is not a hardware-precision artifact.

## Why the prefix data on disk is wrong

DSV4 has 7 KV pools. The relevant ones for this smoke are:

| pool id | name           | type   | tokens/block | per-request blocks |
|---------|----------------|--------|--------------|--------------------|
| 0,1,2   | CSA/HCA/IDX_KV | FULL   | 256          | variable (paged)   |
| 3,4,5   | *_STATE        | SWA    | 4 / 4 / 8    | fixed 2            |
| 6       | SWA_KV         | SWA    | 256          | fixed 2            |

The model itself stores the live SWA window as a circular buffer of
`win = SLIDING_WINDOW = 128` entries inside
`Attention.kv_cache[:B, :win]`. It is indexed by `pos % win`, NOT by
`pos // 256`.

For Q0 (2765 tokens → 11 logical 256-blocks), `SWAKVCacheGroup::malloc`
allocates `[NULL × 9, real, real]` for the SWA-typed pools, i.e. only
slot 9 and slot 10 are real blocks. The Python `_scatter_kv_pool` for
`attn_type == 7` then copies
`attn.kv_cache[:B, k*256 : k*256 + 256]` into slot k. With `T=win=128`
the slot-9 copy is `python_buf[:, 2304:2560]` which slices to length 0,
and the inner loop bails on `n <= 0`. **Net result: nothing is ever
written to the SWA pool blocks** even though the framework treats them
as "complete" (all GPU block ids are non-NULL for the last two slots).

`KVCacheMemoryConnector::buildCopyPlanForWrite` therefore marks the
last aligned key (index 9) as `is_complete=true` and writes garbage
data for it to memory cache. On Q2 the matcher returns `matched_num=10`
because the GPU blocks at slot 9 are non-NULL and the memory-cache
key 9 is "complete" — but the underlying SWA bytes are uninitialized.

## Why approach A (skip SWA on reuse gather) does not fix it

`_layer_reuse_gather_pools` already drops pool 7 from the prefill-reuse
gather list, with the comment "holds data at the wrong position". That
keeps Q2 from blindly *restoring* garbage SWA bytes on top of its own
freshly-zeroed buffer, but it doesn't help Q2 produce the right output:
the SWA buffer that Q2 actually needs at `start_pos = matched_num*256
= 2560` is the live circular state Q0 had right after it processed
token 2559 — and that state was never persisted anywhere. Q2 starts
its continuation with a zero SWA buffer and immediately diverges.

So the fix has to put the actual snapshot somewhere — A alone is not
enough.

## Why approach B is needed

The memory cache is keyed per cache-key (per 256-token block). For Q2
to be able to resume at any aligned boundary K it has matched against,
the SWA pool for that key must hold a *snapshot* of `attn.kv_cache[:,:win]`
as it stood right after processing token `K*256 - 1`.

Concretely:
1. The SWA pool block at slot K should contain the full 128-entry
   `swa_buf` snapshot for that boundary. Each 256-entry × 1024-byte
   block has plenty of room (128 × 1024 = 128 KiB out of 256 KiB), so
   we don't need to grow the per-block size.
2. During Q0's prefill we have to take the snapshot at every 256-token
   boundary, not just the last. The current single-shot scatter at the
   end of prefill is what produces the empty-buffer bug.
3. `SWAKVCacheGroup` (or the new full-style pool that replaces it for
   SWA_KV) must allocate one real block per cache-key, not just two
   tail blocks. Otherwise the per-boundary snapshot has nowhere to
   live.
4. On Q2's reuse path the gather must restore `swa_buf` from the SWA
   pool block at slot `matched_num - 1`, before the continuation
   forward runs.

Costs:
- Memory: every prefill writes one extra 128-KiB SWA snapshot per
  cache-key per layer (≈ 7.5 MiB per cache-key for 60 layers). This
  inflates per-request KV by `<n_blocks> × 7.5 MiB`. For 11-block
  prompts that is ~80 MiB — well within the 1 GiB memory cache.
- Compute: snapshot is a single `copy_` per layer per boundary. Run
  inline with the prefill — it's a chunked-prefill restructuring, not
  a fused-kernel change.

The CSA/HCA/Indexer state pools (3, 4, 5) have the same general issue
(they hold a fused state, not per-token KV) but they are bookkeeping
state for the compressor and are recomputed correctly when the
compressor's `kv_state`/`score_state` are zeroed before forward. The
Python prefill-reuse path (`_gather_all_layers(reuse_gather=True)`)
already restores them from the framework cache, and `_reset_compressor_state`
zeros only the partial-window portion. So the immediate fix is scoped
to pool 6 (SWA_KV).

## Plan B — what changes

1. **C++**
   - Treat pool 6 as a FULL group instead of SWA-tail.
     (`DSV4ConfigCreator.cc` pool 6 spec, `populateCacheConfig`'s
     group_type bookkeeping, `swa_group_num` decrement.)
   - Add an SWA snapshot region name and update
     `KVCacheRegionName` ↔ group_id mapping.

2. **Python (`deepseek_v4_model.py`)**
   - Restructure the prefill loop so the per-request forward runs in
     256-token chunks. Between chunks, snapshot
     `attn.kv_cache[:, :win]` into the SWA pool block for that
     boundary (writing all 60 layers in one go via `_scatter_kv_pool`
     with `T=win` and `entries_per_block=win`).
   - On reuse: restore `swa_buf` from the SWA pool block at slot
     `matched_num - 1` BEFORE continuing the forward at
     `start_pos = matched_num*256`.
   - Drop SWA from `_layer_reuse_gather_pools` skip list once the
     snapshot semantics are correct.

3. **Tests**
   - Update the C++ DSV4 cache tests
     (`PrefixCacheReuseRequiresTwoSWATailHits` etc.) to reflect
     full-pool SWA semantics — single-key tail hit instead of two,
     `is_complete` per cache-key, no NULL prefix.
   - Re-run
     `//internal_source/rtp_llm/test/smoke:v4_flash_native_fp4_fp8_tp1_memory_reuse_cache_sm100`
     and confirm Q0's text == Q2's text.
