# KV cache event publisher

RTP-LLM can publish reusable HBM prefix-cache keys directly to KVCM. It is disabled by default: `none` creates no
queue, worker, or connection; `kvcm` enables publishing.

## Semantics

Events describe logical cache keys, not physical block indices. Reusable components are normally reported independently;
`csa_state`, `swa_kv`, and reuse-disabled groups are excluded, without filtering by cache group type.
Each component requires a complete, IDLE, non-detached DEVICE GroupSet.
Deleting a component preserves the others. Duplicate puts and LRU touches do not produce events.

For DeepSeek V4.1, `hca_kv`, `csa_kv`, and `indexer_kv` share one HBM spec `F` only when all three are eligible
for publication and belong to the same FULL GroupSet. `hca_kv` represents that GroupSet for registration, ADD/DELETE,
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

Publishing supports reusable DEVICE caches, including sparse tails. HOST/HOST_PINNED groups are skipped.

## KVCM lifecycle

The publisher registers the instance in prefix mode and registers the node, then sends `EVENT_BLOCK_SNAPSHOT` reports
independently of batched `EVENT_BLOCK_ADD`/`EVENT_BLOCK_DELETE` changes and `EVENT_HEARTBEAT` requests. It sends best-effort
`EVENT_HOST_DOWN` during engine shutdown. Snapshots replace all medium state for that host identity;
failed or rate-limited snapshot rounds are logged and discarded. Failed incremental batches are dropped and counted,
without an additional send delay. Reporting is best effort; KVCM's same-reporter snapshot/delta write
barrier still applies. The endpoint must support snapshot fencing and crash-safe commit semantics.

Snapshots are captured once after registration, then every 10 minutes by default, timed from capture/handoff rather
than ACK. Successful heartbeat or incremental responses with `snapshot_required=true` can request an early capture;
pending signals coalesce. Snapshot failures and `retry_after_ms` do not block incremental sending.
Registration is preserved during throttling; missing instances/nodes and leader errors trigger re-registration.

Events use `ST_EVENT_REPORT_L1P5`, medium `hbm`, and `event_report://host/hbm` URIs without size parameters; HBM spec
registrations omit `size`. Spec names are `F` for the shared DeepSeek V4.1 FULL GroupSet, `F0`/`F1`/`F2` for its
unmerged tags, and `rtp_llm_hbm_<block_tokens>_<tag>` for other eligible groups. Configure the L1P5 event storage on the
server and add it to the Instance Group's `event_report_storage_candidates`.

Instance registrations must match the published schema and token stride. The `F0`/`F1`/`F2` to `F` change is a schema
change: rollout requires a new HBM Instance ID or an explicit migration of the existing registration and metadata.
Runtime pool names and Remote group names retain the attention-type naming: `hca_kv`/`csa_kv` and
`Fhca_kv`/`Fcsa_kv`, including their aggregate group names. These groups use the original specs
`F0_TP<n>`/`F1_TP<n>`; indexer and state specs remain `F2_TP<n>`/`S3_TP<n>`/`S4_TP<n>`.
The automatic remote-cache Instance ID includes both specs and groups, so retaining the renamed groups can change
that ID relative to an instance registered with `Fglobal_kv_1`/`Fglobal_kv_2` groups.
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
