# Dispatcher FE allocation

Status: implemented (updated 2026-09-01). Scope: dispatcher batch fanout
(`flexlb-api/org.flexlb.dispatcher`) and the master `/batch_schedule` contract
(`flexlb-sync` / `flexlb-common`).

## Why allocation is explicit

A dispatcher splits one client batch into N chunks and sends one chunk to each frontend (FE).
Independent per-dispatcher round-robin cursors can collide under load, while making every batch
depend on the elected master reduces availability. The dispatcher therefore exposes the choice as
an operator contract:

- `dispatch.fe-allocation=master` (default) uses the elected master's single FE cursor. Assignment
  is fleet-wide and attributable; an absent master assignment fails visibly with no local fallback.
- `dispatch.fe-allocation=local` uses this dispatcher's health-filtered `FePool`. It removes the
  master from FE allocation, at the cost of independent cursors on different dispatcher instances.

`DISPATCH_FE_ALLOCATION` overrides the JSON field. Configuration is validated at startup and the
mode is logged; changing it requires restarting/redeploying the dispatcher.

Backend (BE) pre-assignment is a separate optimization. `dispatch.pre-assign-be` defaults to
`false` for rolling-upgrade safety and should be enabled only after all FEs can deserialize the
HTTP `role_addrs` payload. It never changes the configured FE source.

## Allocation matrix

The dispatcher asks `/batch_schedule` only for values it will consume:

| FE mode | Endpoint consumes BE assignment | `preAssignBe` | Master request | FE source |
| --- | --- | --- | --- | --- |
| `master` | yes | `true` | `assign_be=true, assign_fe=true` | master response |
| `master` | no, or toggle off | any / `false` | `assign_be=false, assign_fe=true` | master response |
| `local` | yes | `true` | `assign_be=true, assign_fe=false` | local `FePool` |
| `local` | no, or toggle off | any / `false` | no master call | local `FePool` |

This prevents an endpoint that ignores `role_addrs` from advancing the BE strategy cursor and
prevents local FE mode from advancing the master's FE cursor.

## Wire contract

`BatchScheduleRequest` adds two boolean fields:

- `assign_be`: return worker address/role/port fields.
- `assign_fe`: stamp `fe_url` from the elected master's FE pool.

Both default to `true` when omitted, preserving the original `{\"batch_count\": N}` behavior.
They are additive snake_case JSON fields, and request/response DTOs ignore unknown properties for
mixed-version compatibility.

When `assign_be=false`, `DefaultRouter.batchSchedule` returns N index-preserving placeholder
targets without consulting worker topology, role validation, or a BE strategy. The outer master
handler can then stamp `fe_url` onto those placeholders. Consequently FE-only fanout remains
available while the BE table is warming and in multi-role deployments.

A request with both flags false is invalid; the dispatcher avoids issuing it.

## Master mode flow

1. `BatchHandler` computes `assign_be` and `assign_fe` from the endpoint and configuration.
2. `BatchScheduleClient` calls `BatchScheduleCoordinator` in-process on the master or forwards to
   the elected master from a slave.
3. If requested, `DefaultRouter` reserves BE targets once.
4. If requested, `MasterFeAssigner` calls `FePool.nextBatch(N)` once and stamps the returned URLs
   1:1 onto targets.
5. `FanoutService` sends each chunk only to its stamped URL.

`MasterFeAssigner` stamps only when the node resolved locally
(`!isNeedConsistency() || isMaster()`). A slave therefore preserves the URLs already stamped by
the master instead of consuming its own cursor. The in-process master path stamps in
`BatchScheduleClient`; a forwarded request stamps in `HttpLoadBalanceServer`. Both use the same
bean and cursor.

Assignment is all-or-nothing per pool reservation. If the master has no FE view, a URL is blank,
or assignment throws, affected chunks fail with `CHUNK_NO_FE`; master mode deliberately does not
fall back to a local cursor.

## Local mode flow

`BatchHandler` does not request master FE URLs. Immediately before constructing chunk plans,
`FanoutService` calls the local `FePool.nextBatch(N)` once. One health snapshot and one contiguous
cursor reservation keep URL-to-chunk mapping deterministic even though chunk calls run
concurrently.

If local discovery is empty or unhealthy, affected chunks fail with `CHUNK_NO_FE`. Local mode does
not fall back to the master: each mode has one explicit source, so failures and load attribution
remain understandable.

## BE pre-assignment compatibility

When `preAssignBe=true` and an endpoint is pre-assignable, each selected target becomes a Python
`RoleAddr` in that chunk's copied `generate_config.role_addrs`. The FE then skips its own master
routing round-trip.

The default is `false` because older FE builds may leave JSON role addresses as dictionaries and
fail when model RPC code reads `addr.role`. Enable the optimization only after all FEs include the
`RoleAddr.validate_role` conversion. Disabling it still permits master FE allocation through an
FE-only request and does not move the BE cursor.

BE batch selection currently uses its own round-robin batch strategy rather than the normal
per-request pending-load ledger. Until the next worker-status synchronization, ordinary
load-aware `/schedule` traffic can therefore underestimate a worker that just received a batch.
This is an explicit, opt-in limitation: prefer a load-independent ordinary strategy when the two
paths are mixed heavily, and observe the allocation-dimension tags and dispatcher PV records before
enabling BE pre-assignment broadly.

## Deployment behavior

- Master FE mode requires the elected master to have a dispatcher `FePool`
  (`dispatch.fe-pool-service-id`). Without one, master FE assignment is empty and chunks fail
  visibly.
- Multi-role deployments can use master FE-only mode (`preAssignBe=false`) or local FE mode.
  BE pre-assignment still requires a topology supported by `DefaultRouter.batchSchedule`.
- During a master outage, switching to `DISPATCH_FE_ALLOCATION=local` and restarting removes the
  master dependency for FE selection. If BE pre-assignment is enabled, disable it as well to remove
  the remaining master call.
- Rolling upgrade is safe by default: BE stamping is off, old `batch_count` callers retain both
  assignments, and unknown additive fields are ignored. A new dispatcher talking to an old master
  may cause that old master to compute an unused BE target for an FE-only request, but service
  correctness is preserved until the master upgrade completes.

## Observability

- `dispatcher.preassign.rt` is tagged with `result`, `assign_be`, and `assign_fe`, so an empty
  FE-only result is distinguishable from a BE-only failure.
- `dispatcher.fanout.rt` is tagged with `fe_allocation=master|local`.
- Batch PV logs record `assignBe` and `assignFe`.
- Master batch-schedule metrics carry the same allocation-dimension tags.
- `CHUNK_NO_FE` is the primary failed-chunk signal for either source; use the allocation-mode tag
  and startup config log to identify which pool to inspect.

## Code map

- `BatchScheduleRequest` — additive allocation flags with legacy defaults.
- `DefaultRouter.batchSchedule` — BE selection or FE-only placeholders.
- `MasterFeAssigner` — guarded master FE stamping.
- `BatchScheduleClient` / `HttpLoadBalanceServer` — local and forwarded master entry points.
- `BatchHandler` — computes requested dimensions and avoids unnecessary master calls.
- `FanoutService` — reserves the configured FE vector and performs bounded-concurrency fanout.
- `DispatchConfig` / `FeAllocationMode` — validated operator controls.

## Contract tests

- Legacy JSON defaults and explicit snake_case flags.
- FE-only placeholders without role checks or BE strategy invocation.
- Master stamping guards, one contiguous reservation, and `assign_fe=false` no-op.
- Local allocation reserves once and ignores stale master URLs.
- Non-pre-assignable endpoints request FE only; local/no-BE mode skips the master entirely.
- Allocation dimensions appear in metrics and PV logs.
- End-to-end fanout covers successful assignment, missing assignment, partial failure, timeouts,
  malformed responses, cancellation, and header/query propagation.
