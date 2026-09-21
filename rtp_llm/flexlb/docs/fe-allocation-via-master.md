# Dispatcher batch fanout

Dispatcher splits HTTP batches and merges FE responses in input order. Splittable text batches from `/` and `/batch_infer` are sent to FE `/batch_infer`, preserving top-level and nested generation settings without injecting a batch flag. Both FE routes share request parsing, access logging, metrics, disconnect checks and concurrency accounting; the HTTP route selects batch execution internally. Streaming, multimodal and adapter requests keep their original passthrough route. Callers that already split batches can use `POST /rtp_llm/batch_schedule` without enabling dispatcher routes.

## BE configuration

Keep the deployment's `MODEL_SERVICE_CONFIG` and `HIPPO_ROLE`; use mainline schema version 3:
```sh
export FLEXLB_CONFIG='{"schemaVersion":3,"requestLifecycle":{"request":{"timeoutMs":60000}},"workerRegistry":{"engineType":"EMBEDDING"},"router":{"batchScheduleMaxCount":1000}}'
```
Set the required `requestLifecycle.request.timeoutMs` for the workload; 60000 above is an example, not a default.

`LLM` (default) selects alive published gRPC endpoints. `EMBEDDING` selects discovered workers and returns `arpc_port = http_port + 1`; register the HTTP base port. Embedding master-info `alive` means discovered, not independently probed. Legacy `ENGINE_TYPE`, `BATCH_SCHEDULE_MAX_COUNT` and `BATCH_LOAD_BALANCE_STRATEGY` environment variables (including their `FLEXLB_` variants) are rejected, even when empty or dispatcher is disabled. Migrate engine/count settings to the schema above and remove strategy flags; batch placement uses round-robin.

Send `{"batch_count":5,"assign_be":true,"assign_fe":false}` for BE-only placement. Both flags default to true; at least one is required. Success has exactly `batch_count` targets; count must be 1–`batchScheduleMaxCount` (default 1000). BE assignment requires one configured role without active group routing. FE-only allocation is independent of BE topology/readiness.

## HTTP dispatcher configuration

```sh
export DISPATCH_FE_POOL_SERVICE_ID=your-fe-discovery-service
export DISPATCH_ROUTING_TOKEN=your-shared-token
java -jar flexlb-api.jar
```
Set these two variables in the platform's advanced environment settings to use the defaults. Dispatcher exposes only the following settings:

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `DISPATCH_FE_POOL_SERVICE_ID` | Unset: disabled | FE service discovery name; enables `/dispatcher` |
| `DISPATCH_ROUTING_TOKEN` | Empty | Shared credential on Master and receiving FEs; required for BE preassignment |
| `DISPATCH_SUB_BATCH` | `count:5` | `count:N` balances at most N nonempty chunks; `size:N` or bare N caps items per chunk |
| `DISPATCH_PRE_ASSIGN_BE` | `true` | Preassign a BE per eligible chunk |
| `DISPATCH_BATCH_TIMEOUT_MS` | `30000` | FE response read timeout in milliseconds |
| `DISPATCH_PROBE_PATH` | `/frontend_health` | Health endpoint; use `/health` for FEs exposing that route |

The five ordinary settings also support native Spring `dispatch.*` command-line or configuration-file properties; for example, `dispatch.sub-batch` and `dispatch.pre-assign-be`. Command-line properties override environment values, which override configuration-file values; all use the same binding and validation. This support is limited to Dispatcher and does not restore removed Master variables or `DISPATCH_CONFIG`. The credential `DISPATCH_ROUTING_TOKEN` is read directly from the environment on both Master and FE; keep it out of command-line arguments, which Master logs at startup.

Batch FE allocation always uses the Master batch-schedule coordinator, including when BE preassignment is off. There is no local allocation mode or fallback. Invalid allocation returns 400; unavailable master assignments return 503. Streaming and other passthrough requests retain their existing local FE routing.

Internal limits retain their previous defaults and are no longer configuration options: the whole FE sub-call is capped at `batch-timeout-ms + 30000` milliseconds; empty discovery retains the old pool for five minutes while health probes continue; aggregate requests and responses each have an independent 128 MiB limit, and each FE response has a 16 MiB limit. Fanout concurrency remains eight. Request accounting includes repeated envelopes. Exceeding either an aggregate budget or the per-FE response limit fails the whole batch with 413; response excess uses `batch_response_too_large`. Remove the old FE allocation mode, body-read margin, discovery grace and aggregate-byte settings when upgrading. HTTP input defaults to 5MB via the server's existing `spring.codec.max-in-memory-size` property in [application.yml](../flexlb-api/src/main/resources/application.yml). Discovered FE addresses are deduplicated before publication.

BE preassignment defaults on for single-stage PDFUSION batch inference with round-robin placement per chunk. Configure a matching nonempty `DISPATCH_ROUTING_TOKEN` on dispatcher and receiving FEs; dispatcher startup fails without it unless `--dispatch.pre-assign-be=false`. Set that property to false for multi-role deployments, FEs without preassignment support, or workloads requiring master admission/accounting: placement reserves no LLM capacity. Setting the token on an FE enables credential checks for preassigned `role_addrs` on both raw inference routes; FEs without the token retain mainline direct-routing behavior. Group routing and EMBEDDING engines disable BE preassignment; FE allocation remains enabled. Unassigned LLM items use mainline scheduling. For single-stage PDFUSION, configure its normal scheduling with `scheduler.type=DIRECT` and `dispatcher.type=NON_BATCH` as shown in the [Master configuration guide](../README.md#scheduler-ordering-decision-and-dispatcher). The schema-v3 `dispatcher.type=BATCH` setting controls Master-to-Prefill `EnqueueBatch` delivery; it is separate from HTTP `/dispatcher` fanout and FE-to-BE `BatchGenerateCall`. Batch RPC requires a shared PDFUSION target or static local PDFUSION; item deadlines/order are preserved and failed RPCs are not replayed.

`POST /dispatcher/_dryrun` (root batch) or `/dispatcher/_dryrun{endpoint}` for `/batch_infer`, `/v1/batch/chat/completions`, `/v1/embeddings` and `/v1/reranker` accepts the normal request body. It reuses validation, splitting and request budgets, returning `{"mode":"split|passthrough","chunk_count":N,"chunks":[...]}` before routing addresses are attached; passthrough returns the parsed JSON body as one chunk, an empty batch returns zero chunks. It never calls master/FE or advances allocation cursors, even with BE preassignment enabled, and is excluded from serving metrics and graceful-drain accounting. The serialized preview, including its `mode`/`chunk_count`/`chunks` wrapper and passthrough previews, uses the aggregate request byte limit (413 on excess); actual passthrough forwards raw bytes under the HTTP input limit. Use it to inspect splitting, not backend health or placement; use existing FE health metrics/logs for those. Unknown dry-run endpoints/methods return 400 locally. `_snapshot` is not provided.

## Leader election

Mainline election remains disabled by default, giving independent cursors. For a shared cursor, explicitly configure ZooKeeper:
```sh
export FLEXLB_SYNC_CONSISTENCY_CONFIG='{"needConsistency":true,"masterElectType":"ZOOKEEPER","zookeeperConfig":{"zkHost":"your-zookeeper:2181","zkTimeoutMs":10000}}'
```
Followers forward once; unknown leaders, self-forwarding, repeated hops and transport failures fail without local allocation. Upgrade every potential master to support `/rtp_llm/batch_schedule` before enabling Dispatcher.

## Shutdown

Pre-stop rejects new Dispatcher and batch-schedule HTTP requests with 503, waits for accepted HTTP exchanges (including streaming passthrough), then drains the existing gRPC transport before destroying serving resources. Health hooks and dry-run do not hold this drain open. The platform owns the forced-kill deadline.
