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
export DISPATCH_SUB_BATCH=count:5
export DISPATCH_FE_ALLOCATION=master
export DISPATCH_ROUTING_TOKEN=your-shared-token
java -jar flexlb-api.jar
```
Set these variables in the platform's advanced environment settings. Ordinary Dispatcher settings also support native Spring `dispatch.*` command-line or configuration-file properties. Environment names use `DISPATCH_` followed by the uppercase property name with hyphens replaced by underscores; for example, `DISPATCH_PRE_ASSIGN_BE` and `DISPATCH_BATCH_TIMEOUT_MS`. Command-line properties override environment values, which override configuration-file values; all use the same binding and validation. This support is limited to Dispatcher and does not restore removed Master variables or `DISPATCH_CONFIG`. The FE discovery name enables `/dispatcher` on the existing listener. The shared credential `DISPATCH_ROUTING_TOKEN` is read directly from the environment on both Master and FE; keep it out of command-line arguments, which Master logs at startup.

`count:N` balances at most N nonempty chunks; `size:N` caps items per chunk. `master` uses master FE allocation; `local` uses the local cursor. Discovered FE addresses are deduplicated before publication. Empty discovery can retain the old pool within `discoveryFailureGraceMs` (default 300000; zero disables retention); health probes continue. HTTP input defaults to 5MB via the `spring.codec.max-in-memory-size` property in [application.yml](../flexlb-api/src/main/resources/application.yml); aggregate requests/responses default to 128 MiB each, and each FE response to 16 MiB. Request accounting includes repeated envelopes; aggregate excess returns 413. Invalid allocation returns 400; unavailable master assignments return 503 without local fallback.

BE preassignment defaults on for single-stage PDFUSION batch inference with round-robin placement per chunk. Configure a matching nonempty `DISPATCH_ROUTING_TOKEN` on dispatcher and receiving FEs; dispatcher startup fails without it unless `--dispatch.pre-assign-be=false`. Set that property to false for multi-role deployments, FEs without preassignment support, or workloads requiring master admission/accounting: placement reserves no LLM capacity. Setting the token on an FE enables credential checks for preassigned `role_addrs` on both raw inference routes; FEs without the token retain mainline direct-routing behavior. Group routing disables preassignment. Unassigned LLM items use mainline scheduling. For single-stage PDFUSION, configure its normal scheduling with `scheduler.type=DIRECT` and `dispatcher.type=NON_BATCH` as shown in the [Master configuration guide](../README.md#scheduler-ordering-decision-and-dispatcher). The schema-v3 `dispatcher.type=BATCH` setting controls Master-to-Prefill `EnqueueBatch` delivery; it is separate from HTTP `/dispatcher` fanout and FE-to-BE `BatchGenerateCall`. Batch RPC requires a shared PDFUSION target or static local PDFUSION; item deadlines/order are preserved and failed RPCs are not replayed.

`POST /dispatcher/_dryrun` (root batch) or `/dispatcher/_dryrun{endpoint}` for `/batch_infer`, `/v1/batch/chat/completions`, `/v1/embeddings` and `/v1/reranker` accepts the normal request body. It reuses validation, splitting and request budgets, returning `{"mode":"split|passthrough","chunk_count":N,"chunks":[...]}` before routing addresses are attached; passthrough returns the parsed JSON body as one chunk, an empty batch returns zero chunks. It never calls master/FE or advances allocation cursors, even with BE preassignment enabled, and is excluded from serving metrics and graceful-drain accounting. The serialized preview, including its `mode`/`chunk_count`/`chunks` wrapper and passthrough previews, uses the aggregate request byte limit (413 on excess); actual passthrough forwards raw bytes under the HTTP input limit. Use it to inspect splitting, not backend health or placement; use existing FE health metrics/logs for those. Unknown dry-run endpoints/methods return 400 locally. `_snapshot` is not provided.

## Leader election

Mainline election remains disabled by default, giving independent cursors. For a shared cursor, explicitly configure ZooKeeper:
```sh
export FLEXLB_SYNC_CONSISTENCY_CONFIG='{"needConsistency":true,"masterElectType":"ZOOKEEPER","zookeeperConfig":{"zkHost":"your-zookeeper:2181","zkTimeoutMs":10000}}'
```
Followers forward once; unknown leaders, self-forwarding, repeated hops and transport failures fail without local allocation. Upgrade every potential master to support `/rtp_llm/batch_schedule` before enabling master FE allocation.

## Shutdown

Pre-stop rejects new Dispatcher and batch-schedule HTTP requests with 503, waits for accepted HTTP exchanges (including streaming passthrough), then drains the existing gRPC transport before destroying serving resources. Health hooks and dry-run do not hold this drain open. The platform owns the forced-kill deadline.
