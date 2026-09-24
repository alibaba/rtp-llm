# Dispatcher batch fanout

Dispatcher splits HTTP batches and merges FE responses in input order. Splittable text batches from `/` and `/batch_infer` are sent to FE `/batch_infer`, preserving top-level and nested generation settings without injecting a batch flag. Both FE routes share request parsing, access logging, metrics, disconnect checks and concurrency accounting; the HTTP route selects batch execution internally. Streaming, multimodal and adapter requests keep their original passthrough route. Callers that already split batches can use `POST /rtp_llm/batch_schedule` without enabling dispatcher routes.

## BE configuration

Keep the deployment's `MODEL_SERVICE_CONFIG` and `HIPPO_ROLE`; use mainline schema version 3:
```sh
export FLEXLB_CONFIG='{"schemaVersion":3,"requestLifecycle":{"request":{"timeoutMs":60000}},"workerRegistry":{"engineType":"EMBEDDING"},"router":{"batchScheduleMaxCount":1000}}'
```
Set the required `requestLifecycle.request.timeoutMs` for the workload; 60000 above is an example, not a default.

`LLM` (default) selects alive published gRPC endpoints. `EMBEDDING` selects discovered workers and returns `arpc_port = http_port + 1`; register the HTTP base port. Embedding reuses the existing worker discovery generations in `WorkerDirectory`; it does not publish LLM routing endpoints or run gRPC status/cache probes. Missing ARPC workers retire through discovery reconciliation, not the LLM status timeout. Embedding master-info `alive` means discovered, not independently probed. Configure `workerRegistry.engineType` and `router.batchScheduleMaxCount` only through `FLEXLB_CONFIG`; the former individual environment variables are not read. Batch placement always uses round-robin and has no strategy setting. BE batch placement reuses the mainline `EndpointRoundRobin` with its own cursor instance. Each batch selects consecutive addresses and advances the cursor as one operation; it does not interleave individual selections from concurrent batches. Addresses are ordered lexicographically, and subsequent batches continue after the last selected address even when membership changes. This can change the exact assignment sequence from the earlier index-based cursor. Placement still reserves no capacity.

Existing RTP callers send `{"batch_count":5}` for worker placement without configuring an FE pool. `allocation_type` is `BE` (default) or `FE`; Dispatcher chooses according to whether BE preassignment applies. Both use one `server_status` list with exactly `batch_count` entries in chunk order. BE targets retain worker IP, HTTP/RPC ports and role; FE targets contain `server_ip`, `http_port` and `role: FRONTEND`, without RPC ports. Batch scheduling metrics use the same `allocation_type` tag. Count must be 1–`batchScheduleMaxCount` (default 1000). BE assignment requires one configured role without active group routing. FE-only allocation bypasses worker scheduling and is independent of BE topology/readiness.

## HTTP dispatcher configuration

```sh
# Keep the complete FLEXLB_CONFIG already supplied by the deployment platform.
export DISPATCH_ENABLED=true
export DISPATCH_PRE_ASSIGN_BE=false
java -jar flexlb-api.jar
```
`DISPATCH_ENABLED` optionally overrides only `FLEXLB_CONFIG.httpDispatcher.enabled` after the complete Master configuration is parsed. Set it to `true` to enable ingress or `false` to disable it. When absent, the JSON setting remains authoritative (default false). The value must be `true` or `false` (case-insensitive, surrounding whitespace allowed); blank or invalid values fail startup. The override preserves all other platform-injected settings and does not replace or relax validation of the required `FLEXLB_CONFIG`. Do not overwrite that variable with a partial JSON document. This switch is independent of Master-to-BE `dispatcher.type`. Disabled ingress does not create Dispatcher clients or discovery tasks.

In Whale, set these environment variables in the Master role's advanced settings; no edit to the platform-generated JSON is needed. The example disables BE preassignment, so receiving FEs keep their existing BE routing. For colocated preassignment, use `DISPATCH_PRE_ASSIGN_BE=true`. See the [Whale user guide](dispatcher-user-guide.zh-CN.md) for deployment steps and client examples.

Without an FE service override, Dispatcher reuses `WorkerAddressService` and the existing model's worker HTTP endpoints, including its HTTP/gRPC port conversion. This requires exactly one non-VIT worker role and an FE serving on that worker's HTTP port. For independent FEs, workers without an HTTP FE, or ambiguous P/D ingress, set `DISPATCH_FE_POOL_SERVICE_ID` to the receiving FE HTTP service and `DISPATCH_PRE_ASSIGN_BE=false`. An FE override with BE preassignment enabled fails configuration validation. No new cluster or discovery implementation is needed. Ordinary Dispatcher settings remain:

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `DISPATCH_ENABLED` | Unset: use `FLEXLB_CONFIG.httpDispatcher.enabled` | Override only HTTP Dispatcher enablement; accepts `true` or `false` |
| `DISPATCH_FE_POOL_SERVICE_ID` | Empty: reuse worker HTTP endpoints | Optional FE HTTP service override; requires preassignment off |
| `DISPATCH_SUB_BATCH` | `count:5` | `count:N` balances at most N nonempty chunks; `size:N` or bare N caps items per chunk |
| `DISPATCH_PRE_ASSIGN_BE` | `true` | Select a colocated PDFUSION worker per eligible chunk |
| `DISPATCH_BATCH_TIMEOUT_MS` | `30000` | FE response read timeout in milliseconds |
| `DISPATCH_PROBE_PATH` | `/frontend_health` | Health endpoint; use `/health` for FEs exposing that route |

The five ordinary settings (FE service, sub-batch, preassignment, timeout and probe path) also support native Spring `dispatch.*` command-line or configuration-file properties; for example, `dispatch.sub-batch` and `dispatch.pre-assign-be`. Command-line properties override environment values, which override configuration-file values; all use the same binding and validation. This support is limited to Dispatcher and does not restore removed Master variables or `DISPATCH_CONFIG`. `DISPATCH_ENABLED` is read directly by the Master config loader; it has no `dispatch.enabled` Spring property.

Each split batch uses one Master batch-schedule coordinator call. With BE preassignment, worker round-robin selects one colocated PDFUSION worker per chunk: its HTTP port receives the FE request, and its RPC address is written into `role_addrs`. This path does not also select from the FE pool. Without preassignment, FE round-robin selects HTTP destinations from the default worker endpoints or the explicit FE service; each receiving FE schedules its own BE. There is no local allocation mode or fallback. Invalid allocation returns 400; unavailable master assignments return 503. Streaming and other passthrough requests retain their existing local FE routing. In P/D deployments, disable preassignment and specify the receiving FE HTTP service; this can be an existing colocated FE service, without deploying a separate FE cluster. Splitting distributes FE work, while existing per-request scheduling determines P/D placement; chunks are not pinned to distinct P/D workers.

Internal limits retain their previous defaults and are no longer configuration options: the whole FE sub-call is capped at `batch-timeout-ms + 30000` milliseconds; empty discovery retains the old pool for five minutes while health probes continue; aggregate requests and responses each have an independent 128 MiB limit, and each retained successful FE response has a 16 MiB limit. Error bodies are discarded without using the merge budget; their HTTP status keeps the normal chunk-failure semantics. Each request runs at most 64 sub-batches concurrently, independently of the aggregate byte budgets. Request accounting includes repeated envelopes. Exceeding either an aggregate budget or the per-FE response limit fails the whole batch with 413; response excess uses `batch_response_too_large`. Remove the old FE allocation mode, body-read margin, discovery grace and aggregate-byte settings when upgrading. HTTP input defaults to 5MB via the server's existing `spring.codec.max-in-memory-size` property in [application.yml](../flexlb-api/src/main/resources/application.yml). Discovered FE addresses are deduplicated before publication.

BE preassignment defaults on for batch inference where each worker colocates an FE and a single-stage PDFUSION BE, with round-robin placement per chunk. The cluster can contain multiple such workers; a non-PDFUSION target is rejected before fanout. Set `dispatch.pre-assign-be=false` for independent FEs, multi-role deployments, FEs without preassignment support, or workloads requiring master admission/accounting: placement reserves no LLM capacity. FE retains its existing direct-routing behavior for preassigned `role_addrs`. Group routing and EMBEDDING engines disable BE preassignment; FE allocation remains enabled. Unassigned LLM items use mainline scheduling. For single-stage PDFUSION, configure its normal scheduling with `scheduler.type=DIRECT` and `dispatcher.type=NON_BATCH` as shown in the [Master configuration guide](../README.md#scheduler-ordering-decision-and-dispatcher). The schema-v3 `dispatcher.type=BATCH` setting controls Master-to-Prefill `EnqueueBatch` delivery; it is separate from HTTP `/dispatcher` fanout and FE-to-BE `BatchGenerateCall`. Batch RPC requires a shared PDFUSION target or static local PDFUSION; item deadlines/order are preserved and failed RPCs are not replayed.

`POST /dispatcher/_dryrun` (root batch) or `/dispatcher/_dryrun{endpoint}` for `/batch_infer`, `/v1/batch/chat/completions`, `/v1/embeddings` and `/v1/reranker` accepts the normal request body. It reuses validation, splitting and request budgets, returning `{"mode":"split|passthrough","chunk_count":N,"chunks":[...]}` before routing addresses are attached; passthrough returns the parsed JSON body as one chunk, an empty batch returns zero chunks. It never calls master/FE or advances allocation cursors, even with BE preassignment enabled, and is excluded from serving metrics and graceful-drain accounting. The serialized preview, including its `mode`/`chunk_count`/`chunks` wrapper and passthrough previews, uses the aggregate request byte limit (413 on excess); actual passthrough forwards raw bytes under the HTTP input limit. Use it to inspect splitting, not backend health or placement; use existing FE health metrics/logs for those. Unknown dry-run endpoints/methods return 400 locally. `_snapshot` is not provided.

## Leader election

Mainline election remains disabled by default, giving independent cursors. For a shared cursor, explicitly configure ZooKeeper:
```sh
export FLEXLB_SYNC_CONSISTENCY_CONFIG='{"needConsistency":true,"masterElectType":"ZOOKEEPER","zookeeperConfig":{"zkHost":"your-zookeeper:2181","zkTimeoutMs":10000}}'
```
Followers forward once; unknown leaders, self-forwarding, repeated hops and transport failures fail without local allocation. Upgrade every potential master to the same batch-allocation protocol, using `allocation_type` (`BE`/`FE`) and unified `server_status`, before enabling Dispatcher. Responses lacking the requested target count or role fail validation; worker-only RTP callers retain their existing address schema.

## Shutdown

Pre-stop rejects new Dispatcher and batch-schedule HTTP requests with 503, waits for accepted HTTP exchanges (including streaming passthrough), then drains the existing gRPC transport before destroying serving resources. Health hooks and dry-run do not hold this drain open. The platform owns the forced-kill deadline.
