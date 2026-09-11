# Dispatcher batch fanout

Dispatcher splits HTTP batches and merges FE responses in input order. BE callers that already
split batches can use `POST /rtp_llm/batch_schedule` without enabling dispatcher routes.

## BE configuration

Keep the deployment's `MODEL_SERVICE_CONFIG` and `HIPPO_ROLE`; use mainline schema version 2:

```sh
export FLEXLB_CONFIG='{
  "schemaVersion": 2,
  "workerRegistry": {"engineType": "EMBEDDING"},
  "router": {"batchScheduleMaxCount": 1000}
}'
```

`LLM` (default) selects alive published gRPC endpoints. `EMBEDDING` selects discovered workers
and returns `arpc_port = http_port + 1`; register the HTTP base port. Embedding master-info
`alive` means discovered, not independently probed. Legacy engine/count environment flags are rejected.
Send `{"batch_count":5,"assign_be":true,"assign_fe":false}` for BE-only placement.
Both flags default to true; at least one is required. Success has exactly `batch_count` targets.
Count must be 1–`batchScheduleMaxCount` (default 1000). BE assignment requires one configured role
without active group routing; FE-only allocation is independent of BE topology and readiness.

## HTTP dispatcher configuration

```sh
export DISPATCH_FE_POOL_SERVICE_ID=your-fe-discovery-service
export DISPATCH_SUB_BATCH=count:5
export DISPATCH_FE_ALLOCATION=master
```

Use native Spring `dispatch.*` properties / `DISPATCH_*` environment variables. The FE discovery
name enables `/dispatcher` on the existing listener. `count:N` balances at most N nonempty chunks;
`size:N` caps items per chunk. `master` shares master FE allocation; `local` uses the local cursor.
Empty discovery retains the old pool for `discoveryFailureGraceMs` (default 300000); probes filter it.
Limits: input `MAX_IN_MEMORY_SIZE` defaults to 5MB, aggregate requests/responses to 128 MiB each,
and each FE response to 16 MiB. Request accounting includes repeated envelopes. Excess returns 413.
Invalid allocation returns 400; unavailable master assignments return 503 without local fallback.
BE preassignment defaults off; enabling it requires `DISPATCH_PRE_ASSIGN_BE=true` and a matching nonempty `DISPATCH_ROUTING_TOKEN`
on dispatcher and receiving FEs. Group routing disables it. This placement reserves no LLM capacity;
keep it off when master admission/accounting is required. Unassigned LLM items use mainline scheduling.
Atomic batch RPC requires a shared PDFUSION target or static local PDFUSION; item deadlines and order
are preserved, and failed RPCs are not replayed.

## Leader election

Mainline election remains disabled by default. For a shared cursor, explicitly configure ZooKeeper:

```sh
export FLEXLB_SYNC_CONSISTENCY_CONFIG='{"needConsistency":true,"masterElectType":"ZOOKEEPER","zookeeperConfig":{"zkHost":"your-zookeeper:2181","zkTimeoutMs":10000}}'
```

Followers forward once; unknown leaders, self-forwarding, repeated hops and transport failures fail
without local allocation. Disabled consistency gives independent cursors. Upgrade every potential
master to support `/rtp_llm/batch_schedule` before enabling master FE allocation.
