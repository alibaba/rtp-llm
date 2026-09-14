# Dispatcher batch fanout

Dispatcher splits HTTP batches and merges FE responses in input order. Callers that already split batches can use `POST /rtp_llm/batch_schedule` without enabling dispatcher routes.

## BE configuration

Keep the deployment's `MODEL_SERVICE_CONFIG` and `HIPPO_ROLE`; use mainline schema version 2:
```sh
export FLEXLB_CONFIG='{"schemaVersion":2,"workerRegistry":{"engineType":"EMBEDDING"},"router":{"batchScheduleMaxCount":1000}}'
```
`LLM` (default) selects alive published gRPC endpoints. `EMBEDDING` selects discovered workers and returns `arpc_port = http_port + 1`; register the HTTP base port. Embedding master-info `alive` means discovered, not independently probed. Legacy engine/count environment flags are rejected.

Send `{"batch_count":5,"assign_be":true,"assign_fe":false}` for BE-only placement. Both flags default to true; at least one is required. Success has exactly `batch_count` targets; count must be 1–`batchScheduleMaxCount` (default 1000). BE assignment requires one configured role without active group routing. FE-only allocation is independent of BE topology/readiness.

## HTTP dispatcher configuration

```sh
export DISPATCH_FE_POOL_SERVICE_ID=your-fe-discovery-service
export DISPATCH_SUB_BATCH=count:5
export DISPATCH_FE_ALLOCATION=master
```
Use native Spring `dispatch.*` properties / `DISPATCH_*` environment variables. **`DISPATCH_CONFIG` is no longer read**: migrate its fields to these properties; that JSON variable alone will leave dispatcher routes disabled. The FE discovery name enables `/dispatcher` on the existing listener.

`count:N` balances at most N nonempty chunks; `size:N` caps items per chunk. `master` uses master FE allocation; `local` uses the local cursor. Empty discovery can retain the old pool within `discoveryFailureGraceMs` (default 300000); health probes continue. HTTP input `MAX_IN_MEMORY_SIZE` defaults to 5MB, aggregate requests/responses to 128 MiB each, and each FE response to 16 MiB. Request accounting includes repeated envelopes; aggregate excess returns 413. Invalid allocation returns 400; unavailable master assignments return 503 without local fallback.

BE preassignment defaults off. Enabling it requires `DISPATCH_PRE_ASSIGN_BE=true` and a matching nonempty `DISPATCH_ROUTING_TOKEN` on dispatcher and receiving FEs. Group routing disables it. Placement reserves no LLM capacity: keep it off when master admission/accounting is required. Unassigned LLM items use mainline scheduling. Batch RPC requires a shared PDFUSION target or static local PDFUSION; item deadlines/order are preserved and failed RPCs are not replayed.

`POST /dispatcher/_dryrun` (root batch) or `/dispatcher/_dryrun{endpoint}` for `/batch_infer`, `/v1/batch/chat/completions`, `/v1/embeddings` and `/v1/reranker` accepts the normal request body. It reuses validation, splitting and request budgets, returning `{"mode":"split|passthrough","chunk_count":N,"chunks":[...]}` before routing addresses are attached; passthrough returns the original body as one chunk, an empty batch returns zero chunks. It never calls master/FE or advances allocation cursors, even with BE preassignment enabled, and is excluded from serving metrics and graceful-drain accounting. Preview output also uses the aggregate request byte limit (413 on excess). Use it to inspect splitting, not backend health or placement; use existing FE health metrics/logs for those. Unknown dry-run endpoints/methods return 400 locally. `_snapshot` is not provided.

## Leader election

Mainline election remains disabled by default, giving independent cursors. For a shared cursor, explicitly configure ZooKeeper:
```sh
export FLEXLB_SYNC_CONSISTENCY_CONFIG='{"needConsistency":true,"masterElectType":"ZOOKEEPER","zookeeperConfig":{"zkHost":"your-zookeeper:2181","zkTimeoutMs":10000}}'
```
Followers forward once; unknown leaders, self-forwarding, repeated hops and transport failures fail without local allocation. Upgrade every potential master to support `/rtp_llm/batch_schedule` before enabling master FE allocation.
