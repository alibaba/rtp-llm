# KV cache event publisher

RTP-LLM can publish reusable HBM prefix-cache keys directly to KVCM. It is disabled by default: `none` creates no
queue, worker, or connection; `kvcm` enables publishing.

## Semantics

Events describe logical cache keys, not physical block indices. Reusable FULL/SWA components are reported independently;
LINEAR and reuse-disabled groups are excluded. Each component requires a complete, IDLE, non-detached DEVICE GroupSet.
Deleting a component preserves the others. Duplicate puts and LRU touches do not produce events.

Cache mutations only attempt a bounded non-blocking enqueue; network I/O runs on the publisher worker. Queue overflow,
request failure, heartbeat failure, periodic reconciliation, and the server's `snapshot_required` feedback trigger an
authoritative snapshot. Publishing is fail-open and never affects allocation, eviction, readiness, or inference responses.

Only `tp_rank=0` publishes when `pp_size=1`. CP-sharded Prefill reports canonical keys at
`seq_size_per_block * cp_size` tokens per key; Decode keeps its local block granularity. Each DP replica needs a distinct
`KV_CACHE_EVENT_HOST_IP_PORT`; sharing one lets an authoritative snapshot from one replica replace another replica's
host state. When the value is empty, RTP-LLM derives a per-rank endpoint as
`server_ip:(start_port + rank_id * worker_info_port_num)`.

Publishing supports DEVICE FULL/SWA caches, including sparse tails. HOST/HOST_PINNED groups are skipped.

## KVCM lifecycle

The publisher registers the instance in prefix mode and registers the node, sends an `EVENT_BLOCK_SNAPSHOT`, then sends
batched `EVENT_BLOCK_ADD`/`EVENT_BLOCK_DELETE` changes and independent `EVENT_HEARTBEAT` requests. It sends best-effort
`EVENT_HOST_DOWN` during engine shutdown. Snapshots replace all medium state for that host identity;
failed snapshot payloads are retained for retry. The endpoint must support snapshot fencing and crash-safe
commit semantics.

`snapshot_required` requests a fresh snapshot. `retry_after_ms` delays data-event reports while heartbeats continue.
Registration is preserved during throttling; missing instances/nodes and leader errors trigger re-registration.

Events use `ST_EVENT_REPORT_L1P5`, medium `hbm`, and one spec `rtp_llm_hbm_<block_tokens>_<tag>` per eligible group.
Its `event_report://host/hbm?size=...` URI records that group's byte size summed over TP ranks. Configure the L1P5
event storage on the server and add it to the Instance Group's `event_report_storage_candidates`.

Instance registrations must match the per-tag schema and token stride. Consumers must check FULL history and SWA windows before reuse.

## Configuration

Arguments have equivalent upper-case environment variables.

| Argument | Default | Meaning |
|---|---:|---|
| `--kv_cache_event_publisher_type` | `none` | `none` or `kvcm` |
| `--kv_cache_event_manager_endpoint` | empty | KVCM Meta HTTP endpoint |
| `--kv_cache_event_instance_group` | empty | group; falls back to `reco_instance_group` |
| `--kv_cache_event_instance_id` | empty | stable deployment-level instance ID |
| `--kv_cache_event_host_ip_port` | empty (auto-derived) | stable endpoint; must be unique for every DP replica when `dp_size>1` |

Invalid configuration disables publishing without disabling inference. The KVCM manager endpoint must already be
resolved; service discovery and leader switching are outside this version.

See [KVCM remote cache configuration](reuse_kv_cache.md#kvcm-remote-cache) for the shared backend settings.
