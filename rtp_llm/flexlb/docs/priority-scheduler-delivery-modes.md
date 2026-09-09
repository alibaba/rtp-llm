# FlexLB scheduling and configuration

`FLEXLB_CONFIG` is one strict JSON document. Only schema 3 is accepted; unknown
fields and fields belonging to inactive modes fail startup. There is no legacy
configuration converter. `MODEL_SERVICE_CONFIG` separately describes worker discovery.

## Supported modes

| Scheduler | Ordering | Decision | Dispatcher | Delivery |
|---|---|---|---|---|
| DIRECT | Omitted | Omitted | NON_BATCH | Return the selected route to the frontend |
| QUEUE | FIFO or PRIORITY | SINGLE or FIXED_WINDOW | NON_BATCH | Order and place requests, then return routes to the frontend |
| QUEUE | FIFO or PRIORITY | SINGLE or FIXED_WINDOW | BATCH | Order and place requests, then call Engine EnqueueBatch |

`DIRECT + BATCH` is rejected. DIRECT must omit `queueTimeoutMs`, ordering and decision.
FIFO must not contain priority settings. SINGLE must not contain window settings.
All modes use `dispatcher.maxInflightPerPrefillWorker`, the same request registry,
per-Prefill concurrency admission and request/decision lifetimes.

## Required values

The request inactivity timeout has no default. Other values below are fixture choices or documented defaults.

| Field | Requirement | Meaning |
|---|---|---|
| `requestLifecycle.request.timeoutMs` | Positive milliseconds | Maximum silence since registration or the latest matching Engine request status |
| `requestLifecycle.decision.lifetime` | Finite number ≥ 1; default 2.0 | Deadline: predicted remaining Prefill time (including waiting) × lifetime + 10000 ms |

```json
{
  "schemaVersion": 3,
  "scheduler": {
    "type": "QUEUE",
    "ordering": {"type": "FIFO"},
    "decision": {"type": "FIXED_WINDOW", "maxRequests": 8, "maxCollectionWaitMs": 300}
  },
  "dispatcher": {"type": "BATCH", "maxInflightPerPrefillWorker": 2},
  "requestLifecycle": {
    "request": {"timeoutMs": 60000},
    "decision": {"lifetime": 2.0}
  }
}
```

For DIRECT use `"scheduler":{"type":"DIRECT"}` and
`"dispatcher":{"type":"NON_BATCH"}`; keep the required request timeout.

## Retained settings and defaults

| Field | Default | Constraint / behavior |
|---|---|---|
| `schemaVersion` | `3` | Only `3` is accepted |
| `scheduler.type` | `QUEUE` | `QUEUE` / `DIRECT` |
| `scheduler.queueTimeoutMs` | `3600000` ms | Positive; QUEUE residence TTL only. DIRECT rejects this field |
| `dispatcher.maxInflightPerPrefillWorker` | `2` | Positive integer in all modes; counts batches in BATCH and requests in NON_BATCH / DIRECT |
| `scheduler.ordering.type` | `FIFO` | `FIFO` / `PRIORITY`; larger numeric priority runs first |
| `scheduler.ordering.defaultPriority` | `50` | PRIORITY only; used when a request omits priority |
| `scheduler.ordering.preemption` | All three stages enabled in PRIORITY | Omitting the object or using `{}` enables all stages. An explicit `allowedVictimStages` list replaces the defaults and must be non-empty. FIFO and DIRECT do not enable preemption. |
| `scheduler.ordering.preemption.timeoutMs` | `1000` ms | Positive; explicit configuration requires `DECODE_ENGINE_OWNED`; wait for Engine terminal after the Cancel ACK phase |
| `scheduler.decision.type` | `FIXED_WINDOW` | `SINGLE` / `FIXED_WINDOW` |
| `scheduler.decision.maxRequests` | `8` | Positive, FIXED_WINDOW only; group size is independent of the dispatcher concurrency limit |
| `scheduler.decision.maxCollectionWaitMs` | `300` ms | Non-negative; `0` can still combine waiting requests |
| `scheduler.decision.maxPredictedExecutionMs` | Omitted | Positive when set; FIXED_WINDOW only |
| `dispatcher.type` | `BATCH` | `BATCH` / `NON_BATCH` |
| `router.roles.prefill.executionTimeEstimator.type` | `FORMULA` | `FORMULA` / `LEARNING` |
| `router.roles.prefill.executionTimeEstimator.expression` | `sum(computeTokens) + 0.3*sum(hitCacheTokens)` | FORMULA only; valid non-empty expression, output milliseconds |
| `router.roles.prefill.cacheAffinity` | Omitted | Enables cache-leader preference within the configured TTFT penalty |
| `router.roles.prefill.cacheAffinity.maxExtraTtftMs` | `0` ms | Non-negative; only when cacheAffinity exists |
| `router.roles.prefill.cacheAffinity.minPrefixHitPercent` | `5` | Percentage in `[0,100]`; only when cacheAffinity exists |
| `router.roles.decode.availability.maxKvUsagePercent` | `90` | Percentage in `[0,100]`; `0` means zero usage is allowed, not disabled admission |
| `router.roles.decode.availability.maxEngineRequests` | Omitted | Optional positive Decode request cap; dispatch counts Engine ownership and permits, while preemptive placement also counts queued reservations |
| `router.groupSelector` | Omitted | First matching rule wins; no match uses `defaultTargets` |
| `workerRegistry.health.statusPollIntervalMs` | `20` ms | Positive |
| `workerRegistry.health.statusRpcTimeoutMs` | `5000` ms | Positive |
| `workerRegistry.health.statusStaleAfterMs` | `10000` ms | At least twice status RPC timeout |
| `workerRegistry.health.cleanupIntervalMs` | `3000` ms | Positive integer; fixed-rate scan for expired workers, independent of the stale timeout |
| `workerRegistry.cacheStatus.targetDiffSize` | `30` | Positive |
| `workerRegistry.cacheStatus.minRefreshIntervalMs` | `50` ms | Positive |
| `workerRegistry.cacheStatus.maxRefreshIntervalMs` | `3000` ms | At least the minimum refresh interval |
| `workerRegistry.cacheStatus.fullSnapshotDebugMode` | `false` | Cache full-snapshot debugging |
| `observability.cacheHit.recentKeyWindow.writeEnabled` | `true` | Record recent cache keys |
| `observability.cacheHit.recentKeyWindow.durationMs` | `1800000` ms | Positive |
| `observability.cacheHit.recentKeyWindow.maxKeyOccurrences` | `10000000` | Positive |
| `observability.cacheHit.metricsEnabled` | `true` | Cache-hit metrics |
| `observability.cacheHit.requestTraceLogEnabled` | `false` | Per-request cache trace |
| `observability.cacheHit.theoryLog` | Omitted | Object presence enables theory log |
| `observability.cacheHit.theoryLog.path` | `/home/admin/ai-whale/logs/master_theory_hit.log` | Non-empty path when theory log is enabled |

`groupSelector.defaultTargets` and `rules` default to `[]`. Each rule requires a
unique name, a non-empty match and non-empty targets. `match.apiKeys` defaults to
`[]`; `match.inputTokens.min/max` are optional inclusive bounds, at least one must
be set when inputTokens exists. Each target requires a group; its positive weight
defaults to `1`. Group weights and VIT random selection remain supported.

## Capacity and lifetime behavior

| Owner | Admission / expiry | Release |
|---|---|---|
| Global QUEUE | No request-count limit; queue TTL bounds residence | Successful placement, cancellation or expiry removes the waiting entry |
| Prefill generation | NON_BATCH / DIRECT count local and unmatched Engine requests against the configured limit | Completion, request inactivity expiry, proven rollback or retirement releases exact request ownership and wakes waiters |
| BATCH dispatcher | At most configured batch credits per Prefill generation | The last settled member closes its batch credit |
| Decode generation | KV threshold, full requested output-token reservation and optional Engine-request cap | Engine terminal/retirement, proven non-delivery or request inactivity expiry releases exact local ownership |
| Request inactivity | Latest matching Engine request status + `timeoutMs`; registration time is used before the first status | Silence expires the exact local request and its Prefill/Decode reservations without sending Engine Cancel |
| Post-decision evidence | Remaining Prefill prediction × lifetime + 10000 ms, independent of request age; default `2 × T + 10s` | Engine evidence advances ownership; missing evidence marks `SUSPECTED_LOST` while the request waits for confirmation until its inactivity deadline |

A zero prediction is valid; unknown prediction remains unknown. Prediction does not
replace request-count admission. Engine-reported shape/KV limits and Decode
admission still apply. QUEUE retains backpressured work until capacity changes or
its original TTL expires; DIRECT does not enqueue it.

Decode resolves one admission mode from the scheduler and its configured reclamation
policy before selecting workers. These modes are derived behavior, not JSON fields:

| Mode | Derived from | Selection and capacity scope |
|---|---|---|
| `IMMEDIATE` | DIRECT | Select only workers with available Engine-facing dispatch capacity |
| `WAIT_AT_DISPATCH` | QUEUE without Decode reclamation | Prefer currently dispatchable workers; when all are busy, retain a physically feasible route and wait for its Decode permit at delivery |
| `PREEMPT_AT_PLACEMENT` | PRIORITY QUEUE with Decode reclamation | Evaluate placement inventory, including queued reservations; an exact capacity miss can reclaim allowed lower-priority Decode owners |

`ScheduledRequest.DecodeBinding` carries the frozen request identity, normalized priority,
prompt-plus-output demand, capacity limits and mode. `RouteAdmission` owns the selected
generation pins and reservation until publication; delivery consumes the same binding. Worker inventory is not frozen with it:
selection previews a worker observation, and each reservation or dispatch permit
rechecks current inventory under the selected generation's admission lock. Every mode
requires a dispatch permit before external delivery; a successful preview grants no capacity.

Delivery predictions retain the Engine phases of preceding work and the complete
estimate of work that has not started. At delivery, only preceding work observed
as `ENGINE_RUNNING` consumes elapsed time. Queued work and the request or batch
being delivered retain their full estimates before applying the lifetime multiplier.

`maxUncachedTokens` is removed and rejected. Uncached tokens remain prediction inputs,
not a manually configured worker quota. PDFUSION ownership still requires whole-request
terminal evidence because WorkerStatus has no distinct Prefill-complete signal.

Prefill selection is fixed BEST_ONLY. Cache affinity can prefer the worker with
the best reusable prefix when its additional TTFT stays within `maxExtraTtftMs`
and its predictor-effective reusable prefix meets `minPrefixHitPercent`. The final
cache block remains compute work; equal cache hits preserve the best-TTFT candidate.
Decode prefers workers with current KV/request capacity. If all are busy, QUEUE
can still select a physically feasible worker: without preemption it registers
the route on the selected Prefill and waits for an exact Decode permit at delivery;
with Decode preemption it checks capacity or reclaims victims before Prefill
publication. DIRECT requires current capacity and atomically checks and registers
Prefill ownership under the endpoint lock. Concurrent ownership changes alone do
not reject a DIRECT request. There are no mean-relative outlier filters or
configurable load/KV selection weights.

## Preemption and RPC settings

Only priority preemption may actively send an Engine Cancel RPC. Request inactivity
expiry, ambiguous delivery outcomes, and client-cancellation bookkeeping are local
Master lifecycle operations and do not send Engine Cancel.

| Setting | Source | Meaning |
|---|---|---|
| `allowedVictimStages` | PRIORITY policy | Defaults to `PREFILL_QUEUED`, `DECODE_RESERVED`, `DECODE_ENGINE_OWNED`; an explicit list restricts reclamation to the listed stages |
| Queued Prefill reclamation | `PREFILL_QUEUED` | QUEUE + NON_BATCH: when `maxInflightPerPrefillWorker` is full, atomically replace strictly lower-priority requests still waiting in the Master's endpoint queue. Victims return retryable `NO_AVAILABLE_WORKER` and release their reservations; no Engine Cancel or automatic requeue. Equal priority never yields. BATCH keeps priority ordering in its waiting queue and the configured batch limit; committed batches cannot be reclaimed by this stage. |
| Cancel ACK | Internal `50` ms | Bounds acknowledgement; ACK alone is not execution completion |
| Preemption timeout | `scheduler.ordering.preemption.timeoutMs`, default `1000` ms | After the ACK phase, wait at most this long for the Engine terminal; ACK may have succeeded, failed or remained unknown |
| EnqueueBatch timeout | Runtime property `flexlb.engine-grpc.enqueue-timeout-ms`, default `5000` ms | RPC access timeout; ambiguous dispatch waits for confirmation until request inactivity expiry |

| Cancellation outcome | Incoming request | Victim and capacity |
|---|---|---|
| Engine terminal confirms release | Continue admission when the planned capacity is available | Settle exact ownership and release capacity once |
| No Engine terminal before timeout | Fail with `RESOURCE_EXHAUSTED` / `cancel_terminal_unknown`; do not requeue | Keep the victim's `CANCEL_REQUESTED`/`UNKNOWN` ownership until Engine terminal evidence or its request inactivity expiry; the preemption timeout alone does not release capacity |
| Matching terminal arrives after timeout | The failed incoming request remains terminal | Reconcile the victim and release its capacity; do not restart the incoming request |

See the [PRIORITY BATCH example](config-examples/flexlb-queue-priority-batch.json)
and [PRIORITY NON_BATCH example](config-examples/flexlb-queue-priority-non-batch.json)
for explicit workload configurations. Their non-default values are examples.
