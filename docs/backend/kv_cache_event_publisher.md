# KV cache event publisher

RTP-LLM can publish reusable HBM prefix-cache keys directly to KVCM. `none` disables HBM events by default, and `kvcm`
enables them. A remote backend can still register and send control heartbeats with `none`.

## Semantics

Events describe logical reusable keys, not physical block indices. A key is added only after every cache group that
participates in prefix reuse is complete and matchable. Removing any required group deletes the key. Duplicate puts
and LRU touches do not produce events.

Cache mutations only attempt a bounded non-blocking enqueue; network I/O runs on the publisher worker. Queue overflow,
request failure, heartbeat failure, and periodic reconciliation trigger an authoritative snapshot. Publishing is
fail-open and never affects allocation, eviction, readiness, or inference responses.

Only `tp_rank=0` publishes when `pp_size=1`. RTP-LLM derives the host address for each DP replica as
`server_ip:(start_port + rank_id * worker_info_port_num)`. These addresses must be distinct because an authoritative
snapshot replaces the state for its host address.

## KVCM lifecycle

The publisher registers the instance and node, sends an `EVENT_BLOCK_SNAPSHOT`, then sends batched
`EVENT_BLOCK_ADD`/`EVENT_BLOCK_DELETE` changes and `EVENT_HEARTBEAT`. It sends best-effort `EVENT_HOST_DOWN` during
engine shutdown. Snapshots atomically replace the complete host/medium state; failed payloads retry with exponential
backoff. The endpoint must support snapshot fencing and crash-safe commit semantics.

The initial version publishes device `SharedBlockCache` state as medium `hbm`; DRAM events are excluded.

## Configuration

`RECO_SERVER_ADDRESS` accepts an IP or hostname with the KVCM Meta RPC port, such as `192.0.2.10:12345`,
an explicit VIPServer address such as `vipserver://kvcm.example.vipserver`, or a bare `.vipserver` domain such as
`kvcm.example.vipserver`. Both domain forms enable VIPServer discovery. The publisher calls `GetClusterInfo`
over RPC to find the leader's HTTP address.

When remote cache is enabled, the publisher uses the backend client's discovery settings and instance group.
`RECO_CLIENT_CONFIG` can supply the full remote client configuration. HBM-only deployments use
`RECO_SERVER_ADDRESS` and `RECO_INSTANCE_GROUP`.

| Environment variable | Default | When to set it |
|---|---|---|
| `RECO_SERVER_ADDRESS` | Empty | Set the KVCM Meta address for remote cache or HBM publishing. A full remote client configuration can supply its own address. |
| `RECO_INSTANCE_GROUP` | `default` | Set the group when the deployment uses a different KVCM group. |
| `ENABLE_REMOTE_CACHE` | `0` | Set to `1` for remote cache reads and writes. |
| `KV_CACHE_EVENT_PUBLISHER_TYPE` (`--kv_cache_event_publisher_type`) | `none` | Set to `kvcm` to publish HBM events. The remote backend can send control heartbeats in either mode. |
| `KV_CACHE_EVENT_INSTANCE_ID` (`--kv_cache_event_instance_id`) | `SPECTRUM_DEPLOYMENT_NAME`, then `DEPLOYMENT_NAME` | Set an explicit ID if the platform does not provide a deployment name or to keep an existing event identity. |
| `KV_CACHE_EVENT_SNAPSHOT_INTERVAL_MS` (`--kv_cache_event_snapshot_interval_ms`) | `300000` | Leave at the default for a five-minute snapshot interval. When remote cache is enabled, a nonpositive value disables HBM data events and retains control heartbeats. With remote cache disabled, the interval must be positive. |

To publish HBM events without remote cache, use `ENABLE_REMOTE_CACHE=0` and `KV_CACHE_EVENT_PUBLISHER_TYPE=kvcm`
with the KVCM address and group. Local device prefix reuse must be enabled for HBM data events.

HBM publishing requires an instance ID that is unique within the Meta and stable across replicas with the same
registration settings. Invalid configuration disables publishing without disabling inference.

### Upgrading existing deployments

Remove these variables from the RTP startup configuration.

| Removed variable | Replacement |
|---|---|
| `RECO_ENABLE_VIPSERVER`, `RECO_VIPSERVER_DOMAIN` | Move the VIPServer domain to `RECO_SERVER_ADDRESS`. The address format selects discovery. |
| `KV_CACHE_EVENT_MANAGER_ENDPOINT` | Set the KVCM Meta RPC address through `RECO_SERVER_ADDRESS`. The publisher discovers the leader's HTTP address. |
| `KV_CACHE_EVENT_INSTANCE_GROUP` | Use `RECO_INSTANCE_GROUP` or the group in the remote client configuration. |
| `KV_CACHE_EVENT_HOST_IP_PORT` | RTP-LLM derives the host address from the TP0 server address of each DP replica. |
| `RECO_VIP_JMENV`, `KVCM_VIP_JMENV` | RTP-LLM initializes the VIPServer client with `jmenv.tbsite.net`. |
| `RECO_SDK_CHECK`, `KVCM_SDK_CHECK` | RTP-LLM no longer attaches block IDs to SDK diagnostic traces. |

`KVCacheConfig` pickle state supports legacy 43-, 54-, and 57-element layouts plus the current 62-element layout.
Processes exchanging this state must be upgraded or rolled back together.
