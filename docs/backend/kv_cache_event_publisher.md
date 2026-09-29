# KV cache event publisher

RTP-LLM can publish reusable HBM prefix-cache keys directly to KVCM. It is disabled by default: `none` creates no
queue, worker, or connection; `kvcm` enables publishing.

## Semantics

Events describe logical reusable keys, not physical block indices. A key is added only after every DEVICE cache group
that participates in prefix reuse is complete and matchable, with no pending transfers. Removing any required group
deletes the key. Duplicate puts and LRU touches do not produce events.

Cache mutations only attempt a bounded non-blocking enqueue; network I/O runs on the publisher worker. Queue overflow,
request failure, heartbeat failure, periodic reconciliation, and the server's `snapshot_required` feedback trigger an
authoritative snapshot. Publishing is fail-open and never affects allocation, eviction, readiness, or inference responses.

Only `tp_rank=0` publishes when `pp_size=1`. Pipeline parallelism and CP-sharded KV cache are unsupported because the
runtime cannot assign an unambiguous external owner and block granularity. Each DP replica needs a distinct
`KV_CACHE_EVENT_HOST_IP_PORT`; sharing one lets an authoritative snapshot from one replica replace another replica's
host state. When the value is empty, RTP-LLM derives a per-rank endpoint as
`server_ip:(start_port + rank_id * worker_info_port_num)`.

The emitter tracks complete DEVICE prefix chains only. Tail-sparse and non-DEVICE reusable groups remain disabled
because snapshots do not include complete tail-state, HOST, or DISK state. Multiple FULL DEVICE groups are aggregated
into one HBM spec.

## KVCM lifecycle

The publisher registers the instance in prefix mode and registers the node, sends an `EVENT_BLOCK_SNAPSHOT`, then sends
batched `EVENT_BLOCK_ADD`/`EVENT_BLOCK_DELETE` changes and independent `EVENT_HEARTBEAT` requests. It joins the worker
before sending best-effort `EVENT_HOST_DOWN` during engine shutdown. Snapshots replace all medium state for that host
identity; failed snapshot payloads are retained for retry. The endpoint must support snapshot fencing and crash-safe
commit semantics.

`snapshot_required` triggers reconciliation on both successful and failed responses. `retry_after_ms` delays subsequent
data-event requests, capped at five minutes by default, while heartbeats continue independently. Snapshot throttling
preserves registration; missing instances/nodes and leader errors trigger re-registration.

Events use `ST_EVENT_REPORT_L1P5`, medium `hbm`, and an `event_report://host/hbm?size=...` URI containing the aggregate
byte size. Configure the corresponding L1P5 event storage on the server and add it to the Instance Group's
`event_report_storage_candidates`.

## Configuration

Arguments have equivalent upper-case environment variables.

| Argument | Default | Meaning |
|---|---:|---|
| `--kv_cache_event_publisher_type` | `none` | `none` or `kvcm` |
| `--kv_cache_event_manager_endpoint` | empty | KVCM Meta HTTP endpoint |
| `--kv_cache_event_instance_group` | empty | group; falls back to `kvcm_instance_group` |
| `--kv_cache_event_instance_id` | empty | stable deployment-level instance ID matching the registration configuration |
| `--kv_cache_event_host_ip_port` | empty (auto-derived) | stable endpoint; must be unique for every DP replica when `dp_size>1` |

Invalid configuration disables publishing without disabling inference. The KVCM manager endpoint must already be
resolved; service discovery and leader switching are outside this version.

See [KVCM remote cache](../kvcm_remote_cache.md) for paired artifacts, pickle compatibility, and supported configurations.
