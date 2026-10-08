# KVCM remote cache

## Dependencies and artifacts

`deps/kvcm.bzl` pins the internal KVCM, public SDK/Manager, and PACE revisions. Build the client RPM and server archive from the same source combination, with matching SDK headers and shared libraries. The updated virtual interfaces and StartWrite arguments change the ABI; legacy RPMs and the public PACE stub are incompatible.

The built-in artifact records pin the published SDK RPMs and x86 Manager archive from build `77672840`, including their URLs, SHA256 hashes, and matching `KVCM_SOURCE_ID`. The source tuple is internal `1c24aeac35c819c544316e9753eec0186ea53bd3`, public SDK/Manager `a71117d9745d7228f92aaf64c5414737878bd153`, and PACE `770bd4df361f86cd937f9144e910d202e1a7401f`. No external manifest is required for these artifacts.

Client selection follows the BUILD configuration. The standalone Manager is shared across x86 CUDA configurations:

| Build configuration | Client manifest variant | Server manifest variant |
|---|---|---|
| CUDA 12 x86 | `cuda` | `server` (x86) |
| CUDA 12.9 x86 | `cuda129_x86` | `server` (x86) |
| CUDA 13 x86 | `cuda130_x86` | `server` (x86) |
| CUDA 13 ARM | `cuda130_arm` | No local ARM Manager package; use an external Manager |

CUDA 12.9 and CUDA 13 variants are selected by Bazel and are not overridden by `KVCM_CLIENT_VARIANT`. Other configurations default to the CUDA 12 x86 SDK. No CPU-only SDK is included in this release; `--repo_env=KVCM_CLIENT_VARIANT=cpu` requires an explicit paired CPU artifact in an override manifest. Targets that launch the packaged Manager are restricted to x86; the ARM SDK can communicate with an external Manager.

To override the published artifacts, pass `--repo_env=KVCM_ARTIFACT_MANIFEST=/absolute/path/MANIFEST.json`. The manifest requires exactly one entry for the selected client variant and the `server` variant. All source IDs must match `internal_commit:opensource_commit:pace_commit` from the source lock:

```json
{
  "source_id": "<internal commit>:<opensource commit>:<pace commit>",
  "artifacts": [
    {"variant": "cuda", "source_id": "<same source_id>", "url": "<CUDA 12 x86 RPM URL>", "sha256": "<SHA256>"},
    {"variant": "cuda129_x86", "source_id": "<same source_id>", "url": "<CUDA 12.9 x86 RPM URL>", "sha256": "<SHA256>"},
    {"variant": "cuda130_x86", "source_id": "<same source_id>", "url": "<CUDA 13 x86 RPM URL>", "sha256": "<SHA256>"},
    {"variant": "cuda130_arm", "source_id": "<same source_id>", "url": "<CUDA 13 ARM RPM URL>", "sha256": "<SHA256>"},
    {"variant": "server", "source_id": "<same source_id>", "url": "<x86 Manager archive URL>", "sha256": "<SHA256>"}
  ]
}
```

Bazel validates the source IDs and download hashes. The server archive must also contain the matching `KVCM_SOURCE_ID` marker. Override manifests are tracked as Bazel file inputs, so in-place edits invalidate the affected artifact repositories, including on Bazel 6.4.

The `remote_cache_pace_contract` and `remote_cache_pace_ssd_contract` targets exercise CPU buffers using the selected SDK; their `smoke_kvcm_p1_cpu*` suite names describe the buffer type, not a CPU-only SDK requirement. With the published SDKs, the matching CUDA runtime must be available. Model smoke tests still require a CUDA SDK. CUDA 13 keeps remote cache opt-in: place `--config=remote_kv_cache` after `--config=cuda13` or `--config=cuda13_arm`. An external `KVCM_PACE_FIXTURE` must carry the updated source ID; the PACE provider and consumer revision remains unchanged.

SDK packaging must isolate its internal autil/gRPC symbols from RTP to avoid symbol interposition and duplicate destruction. Link with `-Wl,-Bsymbolic` and a version script exporting only the KVCM API (`_ZN16kv_cache_manager*`, `_ZNK16kv_cache_manager*`, `_ZTVN16kv_cache_manager*`, `_ZTIN16kv_cache_manager*`, and `_ZTSN16kv_cache_manager*`), with all other symbols local. Update the RPM hash in the manifest after relinking.

## Configuration

| Argument / environment variable | Default | Meaning |
|---|---:|---|
| `kvcm_default_query_type` / `KVCM_DEFAULT_QUERY_TYPE` | 2 | Instance default: 1=batch, 2=prefix, 3=SWA, 4=Mamba |
| `kvcm_query_type` / `KVCM_QUERY_TYPE` | 0 | Request mode; 0 uses the Instance default |
| `kvcm_sw_size` / `KVCM_SW_SIZE` | 0 | SWA window in cache keys/blocks; must be positive for SWA |
| `kvcm_read_backend_type` / `KVCM_READ_BACKEND_TYPE` | 0 | 0=regular query; 1=3fs, 2=mooncake, 3=PACE DRAM, 4=NFS, 5=VCNS 3fs, 9=PACE SSD |
| `kvcm_min_replica_count` / `KVCM_MIN_REPLICA_COUNT` | 0 | Minimum readable replicas for StartWrite; the server treats 0 as 1 |

Explicit `KVCM_CLIENT_CONFIG` JSON takes precedence over the generated Instance configuration; an omitted `default_query_type` defaults to 2. Requests can override it with `kvcm_query_type`. Each backend binds to one default Instance. Backend-specific queries use batch mode and accept only `kvcm_query_type=0` or `1`; use `kvcm_read_backend_type=0` for regular Mamba/SWA queries.

Set `--kvcm_model_sdk_config` (environment variable `RECO_MODEL_SDK_CONFIG`) for one data backend:

- DRAM: `[{"type":"pace","sdk_log_level":"INFO"}]`.
- SSD: `[{"type":"pace_ssd","sdk_log_level":"INFO"}]`. Reads can select `kvcm_read_backend_type=9`; the server must use `ST_TAIRMEMPOOL_SSD` and `media_type=5`.

The server storage configuration supplies addresses and media. Regular write candidates should contain only the selected data backend. Configure event storage separately in `event_report_storage_candidates`.

The pinned PACE revision (`770bd4df`) supports TENT TCP transfers, but defaults to AFT. To use TCP, both provider and consumer sidecars must use that revision and receive:

```sh
export TAIR_MEMPOOL_ENABLE_TENT=1
export MC_TENT_CONF='{"transports":{"tcp":{"enable":true},"aft":{"enable":false},"rdma":{"enable":false},"barex":{"enable":false},"shm":{"enable":false}},"policy":[{"name":"tcp_default","segment_type":"memory","transports":["tcp"]}]}'
```

TENT uses an RDMA device slot, so `--no_rdma` disables it. Updating the SDK dependency alone does not change the transport.

Batch/SWA misses preserve their original key positions. Reuse requires a complete FULL prefix, final LINEAR state, and complete SWA window across every TP rank; missing URIs do not count as hits. Mixed LINEAR+SWA writes may store the FULL+LINEAR portion first, but reads still require the complete SWA window. Existing IOV, pool/group, FULL+LINEAR, and same-layout TP support is retained.

Generated Instance identities include the default query mode and registered group configuration, so an upgrade may select a new cache namespace. Custom IDs must match the existing server configuration. `KVCacheConfig` uses pickle version 8 with 74 items and reads versions 1-7; communicating processes must use the same build.

## RPC interface

Metadata requests follow `RpcService/ExecuteFunction` -> `KVCacheManager::executeFunction` -> `KVCMStorageBackend::execute`. They must target TP0 and use this backend's Instance. Payload I/O runs on the corresponding TP rank. SDK errors return a non-OK RPC status.

| `RemoteOperationRequestPB.op` | SDK method | Result |
|---|---|---|
| `REMOTE_OPERATION_MATCH_LOCATION_LEN` | `MatchLocationLen` | `matched_blocks`, in blocks |
| `REMOTE_OPERATION_MATCH_META` | `MatchMeta` | `locations` and the original `metas` string |
| `REMOTE_OPERATION_REMOVE_CACHE` | `RemoveCache` | Success or failure |
| `REMOTE_OPERATION_GET_LOCATIONS_BY_BACKEND` | `GetCacheLocationsByBackend` | Key-aligned `backend_locations`, including empty entries, type, spec size, and URI |
| `REMOTE_OPERATION_GET_HOST_CACHE_STATE` | `GetHostCacheState` | Hosts, local length, P2P fetch length, and final length |
| `REMOTE_OPERATION_MATCH_LOCATION` | `MatchLocation` | Original locations, preserving empty batch/SWA entries |

Example protobuf JSON:

```json
{"remote_request":{"op":"REMOTE_OPERATION_MATCH_LOCATION_LEN","trace_id":"cache-length","metadata":{"query_type":2,"block_keys":[101,102,103]}}}
```

`metadata` accepts tokens, offset/bool masks, window size, detail level, backend, spec names, medium, and P2P host count. Backend queries support batch mode only; nonempty spec-name lists must align with the keys. Host-state queries support prefix and Mamba modes and return metadata.

With `kvcm_read_backend_type` set, locations are mapped by group/rank into TP payload requests, and their URIs are passed to `TransferClient::LoadKvCaches`. Event URIs describe locations and are not used as payload backends by this integration.

## I/O lifetime

RTP requires `sdk_config.drain_on_timeout=true`: the deadline prevents queued I/O from starting, while submitted work finishes before caller references are released. The SDK, TP broadcast, and PACE internal synchronous fallback use default budgets of 12s, 15s, and 10s, respectively, with different start times. Queuing and draining can exceed the outer budget. If a backend never returns, the call, its block references, and shutdown continue waiting.

The controller rank retains allocation pins until every peer payload RPC completes. Followers do not maintain the controller's allocation bookkeeping; they validate physical block ranges and retain the backing pool until local SDK I/O returns. Broadcast timeouts remain failures, but references are released only after peer completion. Cancellation remains subject to the backend contract, especially for Mooncake soft timeouts: SDK future completion alone does not guarantee that a backend has stopped accessing buffers.

Writes map offset/bool masks back to the original keys and fill actual URIs into specs in the same order. Failed writes abort through FinishWrite; empty sessions are closed on a best-effort basis, with server expiry as a fallback. PACE fallback preserves the hostname. DRAM uses `PREFER_LOCAL` (0) to avoid colliding with the legacy `ONLY_REMOTE` value 2; SSD uses `LOC_DEFAULT | MEDIA_TYPE_LOCALSSD` (5).

## Server event configuration

See [KV cache event publisher](backend/kv_cache_event_publisher.md) for publisher lifecycle and topology limits. Add the event storage to the Instance Group's `event_report_storage_candidates`:

```json
{"global_unique_name":"rtp_hbm_events","storage_type":"ST_EVENT_REPORT_L1P5","event_report":{"heartbeat_timeout_ms":30000,"cleanup_grace_ms":300000,"liveness_check_interval_ms":5000,"snapshot_min_interval_ms":1000},"check_storage_available_when_open":false}
```

## Supported scope

The integration supports SDK query/management interfaces, backend-specific reads, replica controls, same-layout TP payload routing, and cache event reporting. It uses the existing DEVICE block IOV/group layout. RTP CPU/HOST-source writes, zero-copy, asymmetric TP/CP, and GDR are outside this integration.

Models require matching client/server artifacts and an attention backend and page size supported by the target GPU. Publisher topology limits are documented in [KV cache event publisher](backend/kv_cache_event_publisher.md).
