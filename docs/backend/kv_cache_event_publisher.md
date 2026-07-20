# KV cache event publisher

RTP-LLM can publish its local HBM prefix-cache state from the engine process. The feature is disabled by default and
supports three modes:

- `none`: no worker thread, queue, network connection, or event construction.
- `log`: asynchronously logs bounded event batches for rollout validation.
- `kvcm`: registers the instance and node with KVCM, then reports incremental changes and authoritative snapshots.

```mermaid
flowchart LR
    Cache["BlockCache put / remove / eviction"] -->|"after state commit"| Logical["logical key completeness"]
    Logical -->|"non-blocking tryPublish"| Queue["bounded queue"]
    Queue --> Log["LogPublisher"]
    Queue --> KVCM["KVCMPublisher"]
    KVCM --> API["KVCM registerInstance / reportEvent"]
    Cache -->|"authoritative HBM snapshot"| KVCM
```

## Event semantics

Events describe reusable logical cache keys, not physical block indices.

- A single-group cache emits `BLOCK_ADD` after a new key is committed.
- A hybrid cache emits `BLOCK_ADD` only after all required groups for the key exist.
- Removing or evicting one group from a complete hybrid key emits one `BLOCK_DELETE`.
- Duplicate inserts and LRU touches do not emit events.
- `put`, `pop`, `remove`, and `selectAndEvict` call the non-blocking publisher while holding the `BlockCache` mutex so
  event order matches cache state transitions. Network I/O remains exclusively on the publisher worker thread.

Only `tp_rank=0` is an event owner. Each DP replica has an independent owner and must use a distinct
`KV_CACHE_EVENT_HOST_IP_PORT`. Other TP ranks use `NullPublisher`.
The HBM location spec is named `rtp_llm_hbm_<block_size_tokens>` and uses an
`rtp-llm://<host_ip_port>/hbm` URI.

## KVCM synchronization

`KVCMPublisher` uses the KVCM Meta HTTP APIs:

1. `POST /api/registerInstance`
2. `EVENT_HOST_DOWN` to remove stale state for the same reporter identity
3. `EVENT_NODE_REGISTER` for the `hbm` medium
4. `EVENT_BLOCK_SNAPSHOT` from the current `BlockCache`
5. batched `EVENT_BLOCK_ADD` and `EVENT_BLOCK_DELETE`
6. periodic `EVENT_HEARTBEAT`

The queue is bounded and producers use `try_lock`; inference threads never wait for queue space or network I/O. A
queue overflow, request failure, heartbeat failure, or periodic reconciliation marks the publisher dirty. After KVCM
is reachable, it resets the reporter scope and sends a new complete snapshot. Reporting is fail-open and never changes
cache allocation, reuse, eviction, engine readiness, or inference responses.

The first implementation publishes the device `BlockCache` as the `hbm` medium. DRAM cache events are not included.

## Configuration

The following server arguments also have equivalent upper-case environment variables.

| Argument | Default | Meaning |
|---|---:|---|
| `--kv_cache_event_publisher_type` | `none` | `none`, `log`, or `kvcm` |
| `--kv_cache_event_manager_endpoint` | empty | KVCM Meta HTTP endpoint, for example `http://127.0.0.1:56020` |
| `--kv_cache_event_instance_group` | empty | KVCM instance group; falls back to `reco_instance_group` |
| `--kv_cache_event_instance_id` | empty | Stable deployment-level instance ID |
| `--kv_cache_event_host_ip_port` | empty | Stable cache endpoint for this DP replica's TP owner |
| `--kv_cache_event_queue_capacity` | `100000` | Maximum queued mutations |
| `--kv_cache_event_report_batch_size` | `1000` | Maximum mutations per `ReportEvent` request |
| `--kv_cache_event_flush_interval_ms` | `20` | Maximum batch wait |
| `--kv_cache_event_heartbeat_interval_ms` | `1000` | Node heartbeat period |
| `--kv_cache_event_request_timeout_ms` | `1500` | HTTP connect and request timeout |
| `--kv_cache_event_retry_interval_ms` | `500` | Retry interval after registration or report failure |
| `--kv_cache_event_snapshot_interval_ms` | `300000` | Periodic authoritative reconciliation interval |
| `--kv_cache_event_log_max_keys` | `8` | Maximum key samples in each log batch |

`kvcm` mode requires the manager endpoint, instance group, instance ID, and host endpoint. Invalid configuration
disables the publisher while leaving inference available. `log` mode does not require KVCM settings.
The manager endpoint must be a resolved HTTP endpoint; this version does not perform KVCM service discovery or
leader switching inside RTP-LLM.

Example validation rollout:

```bash
KV_CACHE_EVENT_PUBLISHER_TYPE=log \
KV_CACHE_EVENT_INSTANCE_ID=my-model-v1 \
KV_CACHE_EVENT_HOST_IP_PORT=10.0.0.8:18000 \
python -m rtp_llm.start_server ...
```

Example KVCM rollout:

```bash
KV_CACHE_EVENT_PUBLISHER_TYPE=kvcm \
KV_CACHE_EVENT_MANAGER_ENDPOINT=http://kvcm-meta:56020 \
KV_CACHE_EVENT_INSTANCE_GROUP=production \
KV_CACHE_EVENT_INSTANCE_ID=my-model-v1 \
KV_CACHE_EVENT_HOST_IP_PORT=10.0.0.8:18000 \
python -m rtp_llm.start_server ...
```
