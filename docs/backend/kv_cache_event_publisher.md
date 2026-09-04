# KV cache event publisher

RTP-LLM can publish reusable HBM prefix-cache keys directly to KVCM. It is disabled by default: `none` creates no
queue, worker, or connection; `kvcm` enables publishing.

## Semantics

Events describe logical reusable keys, not physical block indices. A key is added only after every cache group that
participates in prefix reuse is complete and matchable. Removing any required group deletes the key. Duplicate puts
and LRU touches do not produce events.

Cache mutations only attempt a bounded non-blocking enqueue; network I/O runs on the publisher worker. Queue overflow,
request failure, heartbeat failure, and periodic reconciliation trigger an authoritative snapshot. Publishing is
fail-open and never affects allocation, eviction, readiness, or inference responses.

Only `tp_rank=0` publishes when `pp_size=1`. Pipeline parallelism and CP-sharded KV cache are unsupported because the
runtime cannot assign an unambiguous external owner and block granularity. Each DP replica needs a distinct
`KV_CACHE_EVENT_HOST_IP_PORT`; sharing one lets an authoritative snapshot from one replica replace another replica's
host state. When the value is empty, RTP-LLM derives a per-rank endpoint as
`server_ip:(start_port + rank_id * worker_info_port_num)`.

## KVCM lifecycle

The publisher registers the instance and node, sends an `EVENT_BLOCK_SNAPSHOT`, then sends batched
`EVENT_BLOCK_ADD`/`EVENT_BLOCK_DELETE` changes and `EVENT_HEARTBEAT`. It sends best-effort `EVENT_HOST_DOWN` during
engine shutdown. Snapshots atomically replace the complete host/medium state; failed payloads retry with exponential
backoff. The endpoint must support snapshot fencing and crash-safe commit semantics.

The initial version publishes device `SharedBlockCache` state as medium `hbm`; DRAM events are excluded.

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

`KVCacheConfig` pickle state supports legacy 43-, 54-, and 57-element layouts plus the current 62-element layout.
Processes exchanging this state must be upgraded or rolled back together.
