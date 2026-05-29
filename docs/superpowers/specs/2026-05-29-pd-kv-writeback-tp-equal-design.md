# PD KV Writeback TP Equal Design

Date: 2026-05-29
Status: draft for review

## Goal

Enable best-effort KV cache writeback from decode back to the source prefill after a
PD-separated request finishes, so a later request entering through prefill can reuse
the decoded prefix through existing `REUSE_CACHE` local-device-cache logic.

Phase 1 supports only `prefill TP == decode TP` with one-to-one rank mapping. The
design must keep the request entrance on prefill, must not enable PD invert, and must
not affect the normal PD path when `enable_pd_kv_cache_writeback` is false.

## Non-Goals

- Do not use KVCM or remote cache as the writeback storage target.
- Do not implement heterogeneous prefill/decode parallelism in phase 1.
- Do not make writeback a correctness dependency of request completion.
- Do not guarantee a later cache hit after writeback; normal `REUSE_CACHE` eviction
  and matching rules still apply.
- Do not report service-startup init failure as a runtime monitoring metric. Startup
  failures should fail fast and exit.

## Current Code Facts

- Prefill already sends `peer_addrs` and `peer_grpc_addrs` to decode during remote
  allocate. Decode stores these into `GenerateConfig` as
  `pd_writeback_prefill_worker_addrs` and `pd_writeback_prefill_grpc_addrs`.
- Forward PD transfer handles topology at request time in
  `DecodeRpcServer::constructRemoteLoadRequest`. Equal TP is the simple case:
  decode rank `i` reads from prefill peer `i`, with `partition_count = 1` and
  `partition_id = 0`.
- The existing writeback prototype only calls the first prefill gRPC address. That is
  enough for TP1 but not enough for 4TP.
- The current P2P data plane already has the building blocks:
  `P2PConnectorWorkerPrefill::sendDecodeToPrefillWriteback` sends local decode
  blocks to the configured prefill transfer servers, and
  `P2PConnectorWorkerDecode::receiveDecodeToPrefillWriteback` receives into prefill
  destination blocks. `AsymmetricTpUtil` already maps equal TP to rank-to-same-rank
  transfer.
- `RemoteStore` exists in proto and lower cache-store code, but the gRPC service
  implementation is currently unimplemented. Using it in phase 1 would expand scope
  significantly.

## Approach Options

### Option A: Extend the Current P2P Writeback Path

Use the existing writeback RPC and P2P transfer client, but add an explicit topology
helper and TP-equal fanout. The decode side launches writeback after the request
finishes, starts receive RPCs on all source prefill ranks, then starts send RPCs on
all decode ranks. Each rank sends or receives its local TP shard.

This is the recommended approach because it reuses the current branch's manager,
manifest, allocator writeback APIs, and P2P data plane. It also mirrors forward PD
transfer's request-time topology model.

### Option B: Finish and Use CacheStore `RemoteStore`

Implement the unimplemented `RemoteStore` service and write decoded blocks back using
cache-store partition metadata.

This has a clean long-term shape for heterogeneous topology, but it revives a larger
inactive path, adds a new data-plane dependency, and is not needed for phase 1.

### Option C: Keep Single-Rank Writeback

Only support rank0 and document TP1 as the supported mode.

This is too weak for the current requirement because phase 1 must support 4TP to 4TP.

## Selected Design

Use Option A. Build a writeback topology layer modeled after forward PD transfer, and
make TP equal the only enabled topology in phase 1.

### Topology Contract

Add a small topology helper near writeback code, for example
`PdKvWritebackTopology`.

Inputs:

- local role and local TP size/rank
- decode gRPC worker addresses
- decode transfer worker addresses when available
- source prefill gRPC worker addresses
- source prefill transfer worker addresses
- source and destination compatibility metadata
- whether prefill CP mode is enabled

Output:

- topology mode: `tp_equal` in phase 1
- per-rank mapping entries:
  - decode rank
  - prefill rank
  - decode gRPC target
  - prefill gRPC target
  - prefill transfer target
  - local and remote partition count/id, both `1/0` for TP equal

Phase 1 validation:

- `prefill_grpc_addrs.size() == prefill_worker_addrs.size()`
- `prefill_grpc_addrs.size() == local_decode_tp_size`
- `decode_grpc_workers.size() == local_decode_tp_size`
- source and destination compatibility match, including layer/group layout and
  `partition_count`
- `source.partition_count == destination.partition_count == local_decode_tp_size`
- unsupported CP/heterogeneous modes return `Skipped("unsupported_topology")`

Startup can validate local invariants only: env gate, role, P2P connector init,
writeback manager init, local TP size, and local worker address availability. Cross-side
TP equality cannot be fully validated at startup because source prefill peer addresses
arrive with each PD request.

### Control Plane

Keep the user request entrance at prefill. During forward PD allocate, prefill already
passes source prefill transfer and gRPC addresses to decode. Decode records those in
`GenerateConfig` and uses them at request finish.

For TP equal, the decode master writeback launch should:

1. Build the writeback manifest from final decode cache keys and block ids.
2. Validate topology as `tp_equal`.
3. Start receive RPCs to every source prefill gRPC worker.
4. Start send RPCs to every decode gRPC worker, including local rank if applicable.
5. Wait in a detached writeback task and report per-stage metrics. The finished
   request itself remains complete regardless of writeback result.

The existing `PdKvWriteback` RPC currently routes only to prefill. For phase 1, keep
the same proto message and gRPC method, but make `RemoteRpcServiceImpl::PdKvWriteback`
dispatch by local role:

- on prefill, call a prefill receive handler
- on decode, call a decode send handler

Use distinct C++ handler names and metrics stages even though the wire RPC is shared,
so receive and send failures do not collapse into one bucket. A new proto RPC is not
required for phase 1.

The decode launch path also needs the local decode gRPC worker list. Today
`StreamCacheResource` can read the source prefill gRPC and transfer addresses from
`GenerateConfig`, but it does not own `resource_.grpc_workers`. Phase 1 should inject
`runtime_config_.worker_grpc_addrs` into the writeback RPC client during
`KVCacheConnectorCoordinator::initPdKvWriteback()`. This keeps address ownership close
to the existing connector coordinator and avoids mutating `ResourceContext` after
streams have copied it. Without this injected list, TP4 would still only launch from
the local process and would not protect or send nonzero-rank decode blocks.

For phase 1, keep the wire proto minimal: the existing `PdKvWritebackRequestPB` carries
the manifest and prefill transfer addresses needed by both the prefill receive handler
and decode send handler. The topology helper and `PdKvWritebackLaunchRequest` carry
the richer local topology information in-process. Future heterogeneous support can add
a repeated mapping message mirroring `RemoteStorePartition`, with source/destination
rank and partition count/id fields.

### Data Plane

The data plane remains P2P:

- Decode rank `i` sends the local decode shard for each reusable block.
- Prefill rank `i` allocates writeback blocks, receives the shard, commits cache keys
  into the local device cache, then releases the temporary writeback resource.
- The commit uses `commitWritebackBlocks(..., is_resident = false)` so normal
  `REUSE_CACHE` ownership and eviction apply.

For equal TP, `AsymmetricTpUtil` should produce exactly one destination per sender,
with `local_partition_count = 1`, `local_partition_id = 0`,
`remote_partition_count = 1`, and `remote_partition_id = 0`.

### Lifecycle

Writeback starts only after the decode request reaches `FINISHED`, but it must retain
the source KV blocks before normal release can free or reuse them.

Requirements:

- The initiating stream still builds the manifest before release.
- Every decode worker send handler must create a local source resource from the
  request's block ids and call the same ref-hold mechanism before transfer starts.
- The held resource lives until that rank's send returns.
- If a rank cannot hold the source blocks, that rank fails writeback and reports
  `hold_failed`; the main request remains successful.
- Prefill commits only after receive transfer succeeds. Failed receive frees temporary
  writeback blocks and does not install cache keys.

This avoids relying on rank0's local hold for all ranks, which would not protect
nonzero-rank decode blocks.

### Final Block Fork

Prefill initially cached the prompt-side final partial block. Decode may extend that
same logical block, creating a fork between the old prefill version and the final
decode version.

Phase 1 behavior:

- The writeback manifest must use the final decode cache key sequence and reusable
  block count.
- Prefill writeback commits the decode version under those final cache keys.
- Existing prefill cache entries are not globally deleted during writeback.
- Later matching follows existing prefix match and eviction behavior. If the decode
  version survives and matches, it can be reused; if it is evicted or shadowed by a
  shorter prefix, the system falls back normally.

This matches the intended `REUSE_CACHE` behavior: writeback increases reuse
opportunity but does not provide a persistent-cache guarantee.

### Environment Gate

The feature remains controlled by `enable_pd_kv_cache_writeback` and its environment
variable. When disabled:

- no manifest is built
- no writeback RPC is sent
- no P2P writeback transfer is attempted
- normal PD prefill/decode behavior and metrics are unchanged except for existing
  generic metrics

When enabled and initialization fails, service startup fails fast. Runtime failures are
best effort and reported through metrics/logs.

## Monitoring

Add dedicated writeback metrics rather than relying only on generic P2P metrics. The
generic P2P metrics do not distinguish forward PD transfer from decode-to-prefill
writeback.

Recommended metric group: `RtpLLMPdKvWritebackMetrics` or
`PdKvWritebackMetrics`, registered through the existing kmonitor `MetricsGroup`
pattern.

Required metrics:

- `rtp_llm_pd_kv_writeback_launch_qps`
- `rtp_llm_pd_kv_writeback_launch_failed_qps`
- `rtp_llm_pd_kv_writeback_launch_skipped_qps`
- `rtp_llm_pd_kv_writeback_launch_latency_us`
- `rtp_llm_pd_kv_writeback_rpc_qps`
- `rtp_llm_pd_kv_writeback_rpc_failed_qps`
- `rtp_llm_pd_kv_writeback_rpc_latency_us`
- `rtp_llm_pd_kv_writeback_transfer_qps`
- `rtp_llm_pd_kv_writeback_transfer_failed_qps`
- `rtp_llm_pd_kv_writeback_transfer_latency_us`
- `rtp_llm_pd_kv_writeback_receive_qps`
- `rtp_llm_pd_kv_writeback_receive_failed_qps`
- `rtp_llm_pd_kv_writeback_receive_latency_us`
- `rtp_llm_pd_kv_writeback_malloc_latency_us`
- `rtp_llm_pd_kv_writeback_commit_latency_us`
- `rtp_llm_pd_kv_writeback_block_count`
- `rtp_llm_pd_kv_writeback_token_count`

Recommended tags:

- `stage`: `launch`, `prefill_receive_rpc`, `decode_send_rpc`, `decode_send`,
  `prefill_receive`, `malloc`, `transfer`, `commit`
- `status`: `started`, `success`, `failed`, `skipped`
- `reason`: bounded enum such as `disabled`, `empty_manifest`,
  `missing_prefill_addrs`, `topology_mismatch`, `unsupported_topology`,
  `compatibility_mismatch`, `hold_failed`, `rpc_failed`, `transfer_failed`,
  `malloc_failed`, `commit_failed`
- `topology`: `tp_equal`, `unsupported`
- `role`: `decode`, `prefill`
- `tp_size`: numeric string

Do not add runtime metrics for service-startup init failure. That failure should be a
startup error with logs and process exit.

## Tests

### Unit and Static Tests

- Topology helper:
  - TP4 equal maps decode rank `i` to prefill rank `i`
  - mismatched address counts return `unsupported_topology` or
    `topology_mismatch`
  - CP/heterogeneous cases are rejected in phase 1 but represented in the API
- Coordinator/RPC fanout:
  - no `.front()`-only prefill receive path remains for enabled TP equal
  - all prefill gRPC targets are used for receive
  - all decode gRPC workers are used for send
- Decode send lifecycle:
  - each send handler holds source blocks before transfer
  - hold failure does not commit on prefill and reports a failed status
- Prefill receive:
  - malloc, transfer, commit, free ordering is preserved
  - transfer failure frees writeback blocks without committing cache keys
- P2P mapping:
  - TP4 equal produces partition count/id `1/0`
- Metrics:
  - launch skip/fail reasons report bounded labels
  - receive and transfer success/failure paths report dedicated writeback metrics

### Smoke Integration

Keep the existing TP1 qwen3 writeback reuse smoke. Add a TP4 qwen3 smoke case:

- prefill TP4/world4 and decode TP4/world4
- `enable_pd_kv_cache_writeback` enabled
- two related requests enter through prefill
- second request asserts through `aux_info_assertions`:
  - `aux_info.pd_sep == true`
  - `aux_info.prefill_local_reuse_len >= one reusable block`

The aux-info assertion is the primary integration proof because it verifies that the
later request reused local prefill cache, not just that a writeback RPC was attempted.
qwen35/qwen35-vl smoke can be added after TP4 qwen3 is stable because those cases are
heavier and less suitable as the first regression gate.

## Future Compatibility

The topology helper should be shaped to support the same families already handled by
forward PD transfer:

- `P TP -> D TP` equal and unequal
- `P TP -> D EP`
- `P CP -> D EP`

For future heterogeneous modes, the per-rank mapping can represent:

- D >= P: multiple decode ranks write partitions back to one prefill rank
- P >= D: one decode rank writes to multiple prefill ranks
- CP: prefill rank has full cache but receives partitioned data according to decode
  rank

The hard part is not whether decode can find prefill: forward PD allocate already
delivers the source prefill peer list to decode per request. The hard part is defining
the inverse partition mapping and making commit semantics correct when one side has
multiple physical shards. Phase 1 keeps this out of scope while preserving the data
structures needed to extend it.

## Rollout

1. Land topology and metrics behind the existing environment gate.
2. Verify TP1 regression still passes.
3. Verify TP4 qwen3 smoke passes and aux info proves prefill local reuse.
4. Keep heterogeneous topology returning explicit skip/failure reason.
5. Enable only on services configured with equal prefill/decode TP until later phases
   implement heterogeneous mapping.

## Open Decisions Resolved for Phase 1

- Unsupported topology skips writeback rather than failing user requests.
- Init failure remains service startup failure and does not need runtime metrics.
- Writeback commit is non-resident and follows normal `REUSE_CACHE` eviction.
- The first integration assertion is aux-info reuse, not log scanning.
