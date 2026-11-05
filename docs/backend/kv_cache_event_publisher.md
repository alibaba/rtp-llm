# KV cache event publisher

RTP-LLM can publish reusable local HBM, HOST (L2), and DISK (L3) prefix-cache keys directly to KVCM.
Cache event publishing is disabled by default: `none` disables cache events; `kvcm` enables them.
A configured KVCM storage backend may still use control-only registration and heartbeats when cache events are disabled.

## Semantics

Events describe logical cache keys, not physical block indices. Reusable components are normally reported independently;
`csa_state`, `swa_kv`, and reuse-disabled groups are excluded, without filtering by cache group type.
Each component requires a complete, IDLE, non-detached GroupSet at the reported tier. DEVICE requires all member
blocks; HOST and DISK each hold one packed block for the GroupSet. An allocated transfer target is not reported
until its copy commits. Starting a load or demotion withdraws the source; successful completion publishes the
destination, and failure/cancellation restores an attached source. Detached resources remain excluded.
Components are identified by `(block key, medium, spec name)`: deleting one preserves other components and media
for the same key. Duplicate puts and LRU touches do not produce events.

For DeepSeek V4.1, `hca_kv`, `csa_kv`, and `indexer_kv` share one HBM spec `F` only when all three are eligible
for publication and belong to the same FULL GroupSet. The same spec is used across local tiers.
`hca_kv` represents that GroupSet for registration, ADD/DELETE,
and snapshots; BTC still checks every member, so an incomplete or evicted GroupSet withdraws `F`. Otherwise these tags
retain their individual `F0`/`F1`/`F2` specs. The legacy `global_kv_1`/`global_kv_2` pool tags remain supported with the
same mapping. Other components keep their existing publication behavior.

Cache mutations only attempt a bounded non-blocking enqueue; network I/O runs on publisher workers. Queue overflow
and ordinary request failures do not trigger snapshots. Publishing is fail-open and never gates readiness or inference.

Only `tp_rank=0` publishes when `pp_size=1`. CP-sharded Prefill reports canonical keys at
`seq_size_per_block * cp_size` tokens per key; Decode keeps its local block granularity. Each DP replica needs a distinct
`KV_CACHE_EVENT_HOST_IP_PORT`; sharing one lets an authoritative snapshot from one replica replace another replica's
host state. When the value is empty, RTP-LLM derives a per-rank endpoint as
`server_ip:(start_port + rank_id * worker_info_port_num)`.

Publishing supports reusable groups whose primary memory placement is DEVICE, including sparse tails and their
local HOST/DISK copies. Groups whose primary placement is HOST/HOST_PINNED remain excluded; those topology groups
are distinct from DEVICE groups offloaded to the local HOST tier.

## KVCM lifecycle

The publisher registers the instance in prefix mode and registers the node, then sends `EVENT_BLOCK_SNAPSHOT` reports
independently of batched `EVENT_BLOCK_ADD`/`EVENT_BLOCK_DELETE` changes and `EVENT_HEARTBEAT` requests. It sends best-effort
`EVENT_HOST_DOWN` during engine shutdown. Each snapshot contains HBM, HOST and DISK together in one event and
replaces all medium state for that host identity. Empty snapshots clear all three tiers;
failed or rate-limited snapshot rounds are logged and discarded. Failed incremental batches are dropped and counted,
without an additional send delay. Reporting is best effort; KVCM's same-reporter snapshot/delta write
barrier still applies. The endpoint must support snapshot fencing and crash-safe commit semantics.

Snapshots are captured once after registration, then every 10 minutes by default, timed from capture/handoff rather
than ACK. Successful heartbeat or incremental responses with `snapshot_required=true` can request an early capture;
pending signals coalesce. Snapshot failures and `retry_after_ms` do not block incremental sending.
Registration is preserved during throttling; missing instances/nodes and leader errors trigger re-registration.

All local-tier events use `ST_EVENT_REPORT_L1P5`. This identifies the KVCM event-report storage backend, independently
of the local cache tier. The medium and URI distinguish local tiers:

| Local tier | KVCM `medium` | Location URI | Enable setting |
|---|---|---|---|
| DEVICE / HBM | `hbm` | `event_report://host/hbm` | `enable_device_cache` |
| HOST / L2 | `mem` | `event_report://host/mem` | `enable_memory_cache` |
| DISK / L3 | `disk` | `event_report://host/disk` | `enable_disk_cache` |

`host` above is the reporter's `host_ip_port`. Node registration declares the enabled tiers; HOST-only and DISK-only
publication are supported when device caching is disabled. With `reuse_cache` and the `kvcm` publisher enabled,
the existing tier settings select the reported media; no additional per-tier publisher setting is required.
Disabling all local tiers disables cache events. Reports contain keys and locations, not KV payloads or physical
addresses. A client querying only `hbm` will not see `mem`/`disk`; its medium filter must include the required tiers.

URIs have no size parameters, and spec registrations omit `size`.
Spec names are `F` for the shared DeepSeek V4.1 FULL GroupSet, `F0`/`F1`/`F2` for its
unmerged tags, and `rtp_llm_hbm_<block_tokens>_<tag>` for other eligible groups. Configure the L1P5 event storage on the
server and add it to the Instance Group's `event_report_storage_candidates`. Each media namespace reuses the same
registered spec names, including the historical `rtp_llm_hbm_` prefix. Adding HOST/DISK does not itself change the
instance's registered spec schema. The server must support multi-medium reports and all-medium authoritative
snapshots as defined by the [KVCM ReportEvent API](https://github.com/alibaba/tair-kvcache/blob/533be48be62b9bbc3a407d8558b084ab09cd3809/docs/api/report_event.md).

Instance registrations must match the published schema and token stride. The `F0`/`F1`/`F2` to `F` change is a schema
change: rollout requires a new HBM Instance ID or an explicit migration of the existing registration and metadata.
Runtime pool names retain the attention-type naming: `hca_kv`/`csa_kv`. Remote group names use `F` for all
FULL KV, `FL` for FULL KV plus CSA state and SWA KV, and `L3`/`L4` for the individual state groups.
Unmerged FULL groups use `F0`/`F1`/`F2` for HCA/CSA/indexer KV. This mapping also accepts the legacy
`global_kv_1`/`global_kv_2` pool tags and changes neither spec names nor writable group combinations.
Automatic `DeepSeekV41` remote configuration
with a single PACE DRAM backend merges the three compatible FULL groups into one `F_TP<n>` spec and
remote allocation per rank, using the original GPU buffers directly. State specs remain
`S3_TP<n>`/`S4_TP<n>`. Custom configurations, other backends, and `DeepSeekV41DSpark` retain
`F0_TP<n>`/`F1_TP<n>`/`F2_TP<n>`.
Merged FULL transfers use the SDK's existing `PACE_ENABLE_GATHERSCATTER` setting, cached at initialization;
RTP does not override it. `OFF` selects normal batch I/O. `ON` supports compact direct RDMA for non-tiering,
remote DRAM/OFFSET GAs when the Consumer supports foreign-GA bounds from Manager/SDK URIs and compact
external GPU memory is negotiated. Both paths send the original GPU IOVs to offsets in the single GA,
without payload packing. Other PACE configurations require separate validation before enabling `ON`.
The automatic remote-cache Instance ID includes both specs and groups, so shortening the group names changes
that ID and starts a new cache namespace. Custom `KVCM_CLIENT_CONFIG` registrations must use the same short
group names; an existing explicit Instance ID requires a matching registration migration or a new ID.
This pool-name compatibility mapping applies to the `DeepSeekV41` and `DeepSeekV41DSpark` model names supplied by
ModelFactory. Other models, including DeepSeek V4, retain their existing mappings.
Consumers must check FULL history and SWA windows before reuse.

## Configuration

Arguments have equivalent upper-case environment variables.

| Argument | Default | Meaning |
|---|---:|---|
| `--kv_cache_event_publisher_type` | `none` | `none` or `kvcm` |
| `--kv_cache_event_snapshot_interval_ms` | `600000` | positive capture interval in milliseconds |
| `--kv_cache_event_manager_endpoint` | empty | explicit KVCM Meta HTTP endpoint; overrides discovery |
| `--kv_cache_event_instance_group` | empty | group; falls back to `reco_instance_group` |
| `--kv_cache_event_instance_id` | empty | stable deployment-level instance ID |
| `--kv_cache_event_host_ip_port` | empty (auto-derived) | stable endpoint; must be unique for every DP replica when `dp_size>1` |

When the explicit endpoint is empty, the publisher reuses `RECO_ENABLE_VIPSERVER` and `RECO_VIPSERVER_DOMAIN`, or
`RECO_SERVER_ADDRESS` when VIPServer is disabled. The existing subscriber supplies Meta RPC bootstrap addresses;
`GetClusterInfo` supplies the current leader's host and Meta HTTP port. Reporting still uses HTTP, with no KVCM SDK
upgrade. A failed report invalidates the cached address so the publisher can rediscover the leader and re-register
through its existing retry loop. Discovery runs on the publisher workers and never gates readiness.

Invalid publisher identity disables publishing without disabling inference. VIPServer requires an internal build, as it
does for remote-cache storage. Direct RPC discovery requires the server's `GetClusterInfo` API and a leader endpoint;
standalone servers without leader election can continue to use the explicit HTTP endpoint.

## FlexLB worker metadata

`GetWorkerStatus` reports `kv_cache_group_mode` at protobuf field 25, matching FlexLB's protocol. It uses the same
DEVICE, prefix-reuse, and exact tag filters as HBM publishing. Eligible LINEAR groups report `WITH_MAMBA` (2);
other eligible groups report `FULL_ATTENTION_ONLY` (1). An unavailable cache topology or no eligible groups reports
`UNSPECIFIED` (0). DeepSeek V4.1's excluded `csa_state` and `swa_kv` do not change its FULL-only reporting mode.

FlexLB maps these modes to `QT_PREFIX_MATCH_WITH_MAMBA` and `QT_PREFIX_MATCH`, respectively, without an explicit
query-type override. Its query Instance ID and cache-key block size must still match the publisher's registration.

See [KVCM remote cache configuration](reuse_kv_cache.md#kvcm-remote-cache) for the shared backend settings.
