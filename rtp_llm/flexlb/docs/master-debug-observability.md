# Master debug snapshots

The opt-in debug adapter reads existing ownership tables. It does not schedule,
cancel, reconcile, release, or install request history. Enable it with
`FLEXLB_DEBUG_ENABLED=true` (Spring property `flexlb.debug.enabled=true`).
Routes run on the **application HTTP port**, not the management port. Existing
`queue_snapshot` and `inflight_status` response contracts remain unchanged.

## Requests

```text
GET /rtp_llm/debug/snapshot?include=scheduler,queues,prefill,decode,engine
GET /rtp_llm/debug/requests/9007199254740993?include=scheduler
```

Parameters:

| Parameter | Default | Range / meaning |
|---|---|---|
| `include` | all five components | Nonempty comma-separated subset; unknown names reject with 400 |
| `limit` | 500 | 1..5000 rows across the entire response, including individual batch members |
| `scan_limit` | 2000 | `limit`..20000 visited entries across the entire response |
| `endpoint_limit` | 64 | 1..256 advisory endpoint generations |

The query budget is consumed in scheduler, queues, then endpoint-directory order.
An exhausted later component is explicit `budget_exhausted`, never empty success.
Use a narrower `include` to investigate a particular owner; there is no pagination
or claim that a bounded unordered sample is the oldest or highest-priority sample.
The global queue alone is traversed in its real priority/FIFO order.

The response has `schemaVersion=1`, `instanceId`, `snapshotId`, capture start/end
epoch milliseconds, `status`, `endpointDirectoryTruncated`, `endpointScope`, and
`components`. Component keys are `scheduler`, `queues`, `prefill/<generation>`,
`decode/<generation>`, and `engine/<generation>`.

Each component includes its own capture interval, `consistency`, `status`,
`scannedCount`, `truncated`, immutable `rows`, and scalar `metadata`. Successful
bounded pages include `row_limit`, `scan_limit`, and (when truncated)
`truncation_reason`. HTTP 200 can carry an explicit partial observation;
HTTP 400 rejects invalid queries and HTTP 503 means the capture could not be
served, including executor rejection and the 8 MiB serialized-response bound.
All responses set `Cache-Control: no-store`.

Component statuses are `ok`, `partial`, `busy`, `unavailable`, `budget_exhausted`,
`generation_changed`, and `not_applicable`. `not_applicable` means global queue
mode is disabled. None of these except `ok` is usable as complete evidence.

## Ownership and identity

* Scheduler rows expose public lifecycle phase separately from storage phase
  (`ACTIVE`, `TERMINALIZING`, `TOMBSTONE`), delivery claim, batch identity, future
  completion, and local resource/protection presence. Retained tombstones are
  intentionally queryable. A private diagnostic sequence gives each canonical
  slot a distinct generation without changing admission or lifecycle decisions.
  Scope that generation to `instanceId`; do not use creation time as an identity.
* Prefill rows distinguish active queued requests, individual/direct committed
  requests, and committed batch members. `route_leases_in_use`,
  `batch_leases_in_use`, retained request count, and unknown engine request count
  are different quantities. Open provisional leases need not have request rows.
* Decode rows distinguish reserved, confirmed, and retained non-owning entries.
  Reservation token, protocol protection, dispatch permit, queued flag, KV
  accounting, and settled timestamp remain separate. Metadata retains the
  existing `engine_load + active_dispatch_permits = engine_capacity_used` caliber;
  reserved requests are not by themselves an engine concurrency count.
* Queue rows contain sequence, effective scheduling priority, routing group and
  the existing blocker/claimant leaves. Observing a claimant does not consume a
  capacity edge. A blocked queue does not prove an engine is currently full.
* Engine rows represent the last **committed status report**, read together with
  its status/finished cursors from one atomic publication. Capture time is not
  report time and cannot establish freshness. An old report can remain after a
  newer local state transition.

All 64-bit identities (rid, batch, generation, reservation token, cursors) are
JSON strings. Endpoint generations are scoped to the Master instance. Scheduler
generation is not a wire attempt ID and is not propagated to Engine reports;
cross-owner joins by rid alone are advisory. The client must keep its own
attempt/cohort and environment epoch when it needs stronger correlation.

The endpoint directory is advisory and excludes detached retiring generations.
Even an untruncated response is not evidence that every retiring owner or every
remote Engine resource has disappeared. Request lookup filters each selected
owner, not a complete historical trace. A no-row result is absence at that
source's read point, not global request completion. Scheduler lookup reports
`generation_changed` if its selected slot is replaced while waiting for its lock.

## Concurrency and cost

Prefill, Decode and global queue copies use the existing owner lock with
`tryLock`; contention returns `busy`. Slot reads use their existing monitors and
the scheduler table traversal is weakly consistent per entry. No global lock is
introduced, and queue capture never nests slot/endpoint locks. Debug code copies
only scalar leaves and immutable small maps, then serializes on one dedicated
thread with at most one queued capture. An HTTP cancellation does not guarantee
that a thread waiting for a slot monitor is interrupted; this first version
bounds concurrent work and traversal, not worst-case monitor latency.

Map iteration does not guarantee a wall-clock bound (for example sparse table
capacity); budgets bound visited entries and emitted rows. No scheduling-path
global scan or full snapshot sort is added. Capture failure conservatively
exhausts the remaining scan budget because partial work cannot be accounted for.

## Functional adapter

`flexlb_ft.debug_client.DebugClient` records monotonic request intervals and
preserves source clocks, identity and coverage. `Capture.component` rejects
missing, busy, truncated or failed evidence with `DebugUnavailable`. A future
stage adapter should map this exception to ERROR, not PASS or a zero metric.
`check_scheduler_tombstone` verifies only the scheduler's local storage
invariant; it does not claim Prefill/Decode/Engine cleanup succeeded.

The independent `cases/debug_snapshot_readonly.py` exports `CASE_DEF`:
`master_debug_snapshot`, category `status`, profile `batch-window`, capability
`enqueue_batch`, `expected_fail=False`. It enables debug in its own environment,
issues one request, then polls that rid until its retained scheduler tombstone
can be checked. It saves `master_debug_snapshots.ndjson`, rejects Master instance
changes and never treats an empty cohort or missing request as success.
The case-framework owner registers this definition once in the status registry;
the standalone module does not modify the existing registration order.

The old fence-residue and drain assertions remain in place. Periodic G7
collection, complete transition events, full historical traces, and performance
overhead certification are follow-up work rather than implied by this adapter.

For a no-Fetch investigation these endpoints can show the retained Scheduler
slot, Prefill request/batch-lease accounting, Decode reservation/protection, and
last Engine report independently. They cannot establish C++ deferred-slot
lifetime, actual GPU execution-slot release, connector KV reference release, or
an exact remote TTL. Client/future completion is not any of those release proofs.
