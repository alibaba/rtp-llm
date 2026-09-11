# Dispatcher batch fanout

The dispatcher splits an HTTP batch into chunks, assigns frontends (FEs), and merges their
responses in input order. `/rtp_llm/batch_schedule` can also assign backend (BE) addresses for
callers that already split batches, including the BE-side `WhaleBertScoreOp` path.

## Configuration

FlexLB uses the mainline **schema version 2** configuration. Batch selection is round-robin;
there is no separate strategy factory or `BATCH_LOAD_BALANCE_STRATEGY` setting.

For an embedding worker fleet:

```sh
export FLEXLB_CONFIG='{
  "schemaVersion": 2,
  "workerRegistry": {"engineType": "EMBEDDING"},
  "router": {"batchScheduleMaxCount": 1000}
}'
```

`workerRegistry.engineType` defaults to `LLM`. LLM workers must have a published, alive endpoint
from the normal gRPC status synchronization. Embedding workers expose ARPC instead of that gRPC
API, so their availability comes from service discovery. Discovery returning no workers makes
embedding batch selection unavailable. Register the HTTP base port in an `http` endpoint;
embedding targets return `arpc_port = http_port + 1`, while LLM targets return `grpc_port`.
The existing `MODEL_SERVICE_CONFIG` supplies the model, role and discovery address.
Startup also requires the deployment's `HIPPO_ROLE`, as on main, even when consistency is disabled.

`router.batchScheduleMaxCount` defaults to 1000 and must be positive. Legacy top-level JSON fields
and the old `ENGINE_TYPE`, `BATCH_SCHEDULE_MAX_COUNT`, and `BATCH_LOAD_BALANCE_STRATEGY` environment
variables are rejected at startup; move the first two into the document above and remove the
strategy setting.

To enable HTTP dispatcher routes, configure an FE pool:

```sh
export DISPATCH_CONFIG='{
  "fePoolServiceId": "your-fe-discovery-service",
  "subBatch": "count:5",
  "feAllocation": "master",
  "preAssignBe": false
}'
```

`subBatch` accepts `count:N` (N chunks) or `size:N` (at most N items per chunk). FE allocation is
`master` by default; `local` assigns from each dispatcher's own healthy FE pool. Spring
`dispatch.*` properties and `DISPATCH_*` environment overrides take precedence over the JSON
(e.g. `DISPATCH_FE_ALLOCATION=local`). Invalid overrides fail at startup instead of being ignored.
`DISPATCH_CONFIG.discoveryFailureGraceMs` controls how long an empty FE discovery result may
retain the previous pool; the default is 300000 ms. FE health probes still filter that pool.

BE pre-assignment is optional and defaults to false. To enable it, set `preAssignBe=true` and
supply the same nonempty `DISPATCH_ROUTING_TOKEN` to the dispatcher and receiving RTP FEs. The
FE validates that token before accepting HTTP `role_addrs`. Keep the token outside JSON config
and obtain its value through the deployment's secret configuration.

For callers that split batches in BE and only need backend addresses, `DISPATCH_CONFIG` is
unnecessary; send `assign_fe=false` to `/rtp_llm/batch_schedule`.

## Leader election

Leader election uses mainline `LBStatusConsistencyService` unchanged. It is disabled by default.
A deployment that needs one shared allocation cursor must explicitly supply its existing
ZooKeeper configuration, for example:

```sh
export FLEXLB_SYNC_CONSISTENCY_CONFIG='{
  "needConsistency": true,
  "masterElectType": "ZOOKEEPER",
  "zookeeperConfig": {"zkHost": "your-zookeeper:2181", "zkTimeoutMs": 10000}
}'
```

With consistency enabled, followers forward batch scheduling to the elected master. The HTTP
path rejects self-forwarding and a second forwarding hop, matching the mainline gRPC guard.
An unknown leader, failed forwarding request or missing batch endpoint fails the request; a
follower does not allocate locally in these cases. Without consistency, each instance allocates
locally and the cursors are independent.

## Allocation contract

```json
{"batch_count": 5, "assign_be": true, "assign_fe": false}
```

Both assignment flags default to true. `batch_count` must be between 1 and the configured maximum;
a request with both flags false is invalid. Successful responses contain exactly that many
`server_status` entries, each with its requested BE fields and/or optional `fe_url`.

| FE mode | BE pre-assignment used | Master request | FE source |
| --- | --- | --- | --- |
| `master` | yes | `assign_be=true, assign_fe=true` | master response |
| `master` | no | `assign_be=false, assign_fe=true` | master response |
| `local` | yes | `assign_be=true, assign_fe=false` | local FE pool |
| `local` | no | no scheduling call | local FE pool |

BE assignment supports a single configured role. It rejects active `router.groupSelector`
rules or default targets because batch-count-only requests lack per-item routing information.
The dispatcher automatically defers BE placement to the FEs when group routing is active.
FE-only assignment works with multi-role deployments and during BE warm-up. Missing FE
assignments produce visible chunk failures; master mode does not fall back to a local FE cursor.

## LLM execution follows mainline scheduling

Unassigned LLM items use the normal per-request master scheduler, including its existing batching,
admission and completion handling. The dispatcher does not add aggregate-demand protobuf fields,
synthetic batch reservations or a second request lifecycle.

A chunk whose items all have the same preassigned PDFUSION backend can use `BatchGenerateCall`.
A local/static PDFUSION deployment can also use it without master routing. Other topologies use
per-item inference. Direct batch RPC validation rejects mixed destinations and non-PDFUSION
assignments, preserves result order and item deadlines, and does not replay a failed RPC.

This BE pre-assignment path is stateless placement: it does not reserve LLM capacity. Keep
`preAssignBe=false` when master-side LLM admission and accounting are required.
