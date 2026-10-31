# Request-aware sparse KV placement

Sparse KV offload routes new requests between the existing HBM and pinned-host
block ranges. A request whose complete block count exceeds one third of the
currently free HBM blocks prefers host storage, leaving HBM available for smaller
requests. Equality prefers HBM. Block zero is excluded from free capacity.

`RTP_LLM_SPARSE_KV_OFFLOAD_HBM_SHARE_DENOMINATOR` sets the denominator. The default
is `3`; `0` disables request-aware placement and preserves the ordinary allocation
order. Other valid values are integers at least `2`. The setting is read when an
offload pool with host blocks is constructed. HBM-only pools ignore it.

Initial admission uses the complete request length, including reused prefixes,
even when allocation is staged. Existing prefix blocks retain their addresses.
Later growth prefers the tier of the last allocated block. If the preferred tier
cannot supply all requested blocks, allocation falls back to the other tier; this
is a placement preference, not a hard HBM quota or a reservation. In particular,
a request that starts small can keep growing in HBM, and host exhaustion can
send a long request's remaining blocks to HBM.

The two ranges have independent free queues under the existing pool lock.
Admission reads free HBM capacity without scanning block IDs. Allocation failure
does not consume blocks or advance generations. Existing request and tree
references, eviction, transfer buffers and generation publication retain their
ordinary lifetimes. GPU hot-cache capacity and eviction are still shared across
the runtime's selection group; they are independent of this placement policy.

Tests cover long-first admission, the exact threshold, current free HBM capacity,
growth, repeated release/reuse, cache and in-flight load holds, cross-tier fallback,
atomic failure, disabled/invalid settings, staged allocation and reused prefixes
with load placeholders. The coordinator test exercises admission and growth
through the request allocator.
