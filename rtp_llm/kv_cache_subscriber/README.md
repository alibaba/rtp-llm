# RTP-LLM KV Cache Subscriber

This sidecar converts RTP-LLM's pull-based `GetCacheStatus` API into KVCM
`ReportEvent` calls. It is intentionally part of RTP-LLM because it depends on
the generated `model_rpc_service_pb2` module and the exact cache-status wire
contract.

## Data flow

1. Poll every configured RTP DP gRPC endpoint with
   `latest_cache_version=-1` and `need_cache_keys=true`.
2. Reject the whole poll when any DP fails or reports a different block size.
3. Merge all `cache_keys`, then diff the full set against the last
   KVCM-acknowledged set.
4. Report `EVENT_BLOCK_ADD` and `EVENT_BLOCK_DELETE` in bounded batches.
5. Advance the local baseline only after every KVCM request succeeds.

The RTP cache `version` is logged but is never used to skip a full pull. The
current value comes from `SharedBlockCache` and does not cover every memory
cache-key change.

Removals are debounced for two successful snapshots by default. An RPC failure
is not a successful empty snapshot and therefore can never trigger deletions.
Every five minutes the subscriber replays all observed adds so a restarted KVCM
can recover even when the RTP key set did not change.

## Start

Build RTP-LLM first so the existing model RPC protobuf Python files are
available, then run:

```bash
python -m rtp_llm.kv_cache_subscriber \
  --rtp-endpoints 127.0.0.1:8089 \
  --kvcm-url http://127.0.0.1:6382 \
  --instance-id my-deployment
```

The Bazel entry point is:

```bash
bazel run //rtp_llm/kv_cache_subscriber:subscriber -- \
  --rtp-endpoints 127.0.0.1:8089 \
  --kvcm-url static://10.0.0.1:6382,10.0.0.2:6382 \
  --instance-id my-deployment
```

`--kvcm-url` accepts `http(s)://`, `static://`, and
`spectrum://<virtual-service-id>:<port>`. When the argument is omitted,
`KVCM_URL` is used first, followed by `KVCM_VSERVICE_ID`.

For multiple RTP DP workers, list every endpoint separated by commas. Their
keys are reported as one KVCM host location:

```bash
--rtp-endpoints 10.0.0.8:8089,10.0.0.8:8098
```

## Consistency and recovery

- Cold start: after the first valid full snapshot, the subscriber sends
  `HOST_DOWN`, registers the node again, and adds the complete observed set.
  Disable this with `--no-reset-on-start` only when another subscriber owns the
  same `host_ip_port`.
- RTP outage: after `--engine-failure-threshold` consecutive failed full pulls,
  the subscriber reports `HOST_DOWN`. Recovery registers the node and replays
  the complete snapshot.
- KVCM outage: failed add/delete batches do not change the acknowledged local
  baseline, so the same diff is retried on the next poll. Adds that may have
  succeeded before a later chunk failed are tracked as uncertain and are
  explicitly deleted if they disappear from subsequent RTP snapshots. KVCM
  operations must remain idempotent because successful chunks may be replayed.
- Subscriber restart: the cold-start reset removes stale KVCM locations that
  cannot be inferred from an empty in-memory baseline.

The default `storage_type` is `ST_EVENT_REPORT`, matching the KVCacheManager2
vLLM Subscriber branch. The deployed KVCM protocol and instance backend must
support that storage type; otherwise set `KVCM_STORAGE_TYPE` to the backend's
event-reporting enum value.

## Important options

| Option | Default | Meaning |
| --- | ---: | --- |
| `--poll-interval-s` | `1` | Full-cache polling period |
| `--deletion-confirmations` | `2` | Consecutive full snapshots required before delete |
| `--engine-failure-threshold` | `3` | Consecutive pull failures before `HOST_DOWN` |
| `--full-refresh-interval-s` | `300` | Periodic idempotent replay of all current adds |
| `--kvcm-report-batch-size` | `1000` | Maximum events per `ReportEvent` request |
| `--host-ip-port` | local IP plus `8088` | Stable KVCM location identity |
| `--medium` | `hbm` | KVCM location medium |

## Tests

```bash
bazel test //rtp_llm/kv_cache_subscriber:tests
```

The suite covers full-snapshot diffing, deletion debounce, uncertain partial
reports, multi-DP aggregation and failure isolation, KVCM wire mapping and
batching, node re-registration, heartbeat gating, periodic full replay,
configuration validation, and manager response handling.
