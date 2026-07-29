# FE allocation via the master

Status: implemented (2026-07-22). Scope: dispatcher batch-fanout path (`flexlb-api`
`org.flexlb.dispatcher`) plus the master `/batch_schedule` response (`flexlb-sync` /
`flexlb-common`).

## Problem

The dispatcher fans a batch request out to N frontend (FE) instances, one chunk per FE. FE
selection used a per-instance `FePool` round-robin cursor. With multiple dispatcher instances,
each cursor is private and uncoordinated, so instances collide on the same FE under load and FE
load splits across per-instance cursors — which also makes FE load impossible to attribute to a
single decision when debugging.

## Design

FE selection is sourced **solely** from the elected master's single `FePool` cursor — the one
place that is already a global singleton (active/standby election). The master stamps a per-chunk
`fe_url` onto each `/batch_schedule` target; the dispatcher fans each chunk out to that URL. There
is **no local fallback**: a chunk the master did not assign fails visibly. FE load is therefore
attributable to exactly one cursor, and load debugging never has to reason about a second,
per-instance distribution.

Control-plane decision is single-pointed (the master picks the FE); the data-plane (chunk bodies,
N-way fanout, response aggregation) stays multi-instance. This mirrors how BE selection already
works (`RoundRobinLoadBalancer.selectBatch` advances one master cursor), reusing the
`/batch_schedule` round-trip the dispatcher already makes rather than adding a link or protocol.

## How the master discovers FE

The master performs **no new** FE discovery. A node that runs the dispatcher — which the elected
master does, since dispatcher and master share the JVM, the 7001 listener, and the Spring beans
(`BatchScheduleClient` javadoc) — already runs the dispatcher's `FePool` machinery:

```
DISPATCH_FE_POOL_SERVICE_ID  (env: dispatch.fe-pool-service-id)
  -> DispatcherFePoolRefresher            (@ConditionalOnProperty on that env)
       |- serviceDiscovery.listen(serviceId, cb)         push, real-time
       |- @Scheduled(30s) serviceDiscovery.getHosts(id)  poll, freshness fallback
       both write AtomicReference<List<String>> fePoolUrls; url = "http://" + ipPort
  -> FePool(refresher.source(), FeHealthChecker)
       next()/nextBatch(n) read a fresh snapshot, skipping FeHealthChecker-dead hosts
  -> MasterFeAssigner.assign() calls fePool.nextBatch(targets.size()) once and zips the
     result 1:1 onto target.fe_url (all-or-nothing — see below)
```

`ServiceDiscovery` is the same VipServer-backed infrastructure the master already uses for BE
workers (`flexlb-common`, consumed by `flexlb-sync`); only the serviceId differs
(`dispatch.fe-pool-service-id` for FE vs the BE worker service). No new module dependency, no new
link.

Every dispatcher node (including the elected master) runs its own refresher against the same
serviceId, so all nodes see an eventually-consistent FE set. The master's list is not "more
authoritative" — the value of the design is a **single cursor**, not a more-accurate list. FE
liveness is the master's own `FeHealthChecker` view: `fePool.next()`/`nextBatch()` skip hosts the
master marks dead, and the dispatcher trusts the master's pick without re-probing.

## Data flow

```
dispatcher receives batch
  -> BatchScheduleClient.requestTargets(count)          in-process on the master, HTTP-forward on a slave
       -> BatchScheduleCoordinator.schedule()
            master:  RouteService.batchSchedule -> DefaultRouter -> RoundRobinLoadBalancer.selectBatch
            slave:   forwardToMaster (HTTP) to the elected master
       -> MasterFeAssigner.assign(targets): fe_url = fePool.nextBatch(targets.size()), zipped 1:1
            one snapshot + one contiguous cursor reservation for the whole batch, and
            all-or-nothing: an empty snapshot stamps no target at all (every fe_url stays
            null -> every chunk fails CHUNK_NO_FE downstream) rather than leaving a
            stamped prefix and a null tail
            (only when this node resolved locally AND the FePool bean exists — see Guards)
            master-local resolution: BatchScheduleClient stamps here
            slave forward:           the master already stamped in its HttpLoadBalanceServer, the
                                     slave's own assign() is a guarded no-op
  -> BatchHandler: BE role_addrs stamped only when preAssignBe && spec.isPreAssignable();
                   per-chunk fe_url extracted from targets and passed to fanout
  -> FanoutService.dispatchOne: send chunk to target.fe_url; null fe_url -> fail with CHUNK_NO_FE
```

## Guards (correctness)

`MasterFeAssigner.assign` is the single stamping point. It stamps `fe_url` only when both hold:

1. **This node resolved the batch locally** — it is the elected master, or consistency is off
   (`!isNeedConsistency() || isMaster()`). A slave that merely forwarded to the master already
   holds the master's assignment in the response; re-stamping it with the slave's own cursor would
   reintroduce the collision the feature removes.
2. **The `FePool` bean exists** (injected via `ObjectProvider`) — i.e. this node also runs the
   dispatcher. Absent it, targets keep `fe_url == null`.

There are two local-resolution entry points, and **both** route through the same `MasterFeAssigner`
bean (hence the same single `FePool` cursor):

- **The master's own in-process dispatcher** resolves via `BatchScheduleClient` and does not pass
  through `HttpLoadBalanceServer` — so `BatchScheduleClient` invokes `MasterFeAssigner.assign`
  itself. (An earlier cut wired stamping into the HTTP handler only, which left this path unstamped
  and failed every chunk the master resolved locally.)
- **A slave's forwarded request** is answered by the master's `HttpLoadBalanceServer`, which invokes
  `MasterFeAssigner.assign` before replying. The slave then calls `assign` again on the returned
  targets, but guard #1 (`isMaster()==false`) makes it a no-op, preserving the master's stamp.

Each batch request therefore advances the one global cursor exactly once, whether resolved on the
master in-process or forwarded from a slave.

## Decisions

- **No fallback.** Determinism of load attribution is prioritized over availability. A chunk with
  no master `fe_url` fails visibly with a distinct reason (`CHUNK_NO_FE`, so "the master isn't
  assigning FEs" reads straight off the metric) rather than silently rerouting to a
  per-instance pick.
- **FE always from the master, decoupled from `DISPATCH_PRE_ASSIGN_BE`.** The `/batch_schedule`
  round-trip is now unconditional for every splittable batch, because it carries `fe_url` even for
  endpoints that ignore BE `role_addrs` stamping. BE stamping stays gated on
  `preAssignBe && spec.isPreAssignable()`.
- **`fe_url` is additive on the wire.** `BatchScheduleTarget` is `@JsonInclude(NON_NULL)` +
  `@JsonIgnoreProperties(ignoreUnknown=true)`; an unset target is byte-identical to the pre-feature
  schema and an older peer that never sends it still parses.

## Precondition (deployment)

The elected master node **must** run the dispatcher (`dispatch.fe-pool-service-id` set) for
coordination to engage — that is what gives it the `FePool` bean. Normal deployments satisfy this
(dispatcher and master co-located in one JVM). If the elected master does not run the dispatcher,
it has no FE view, stamps no `fe_url`, and — because there is no fallback — the batch chunks fail.

**Single-role only.** The `/batch_schedule` resolve (`DefaultRouter.batchSchedule`) supports a
single-role deployment only — it rejects a multi-role fleet (configured, or detected, `roleTypes`
size > 1) with `INVALID_REQUEST`. Because FE is now sourced solely from that resolve with no
fallback, a multi-role deployment fails **every** splittable batch with `CHUNK_NO_FE`. (Before this
change, non-preAssignable endpoints skipped the master entirely and fanned out off the per-instance
local pool, so FE selection still worked under multi-role.) Dispatcher batch-fanout is therefore
supported only on single-role deployments — do not enable it on a multi-role (e.g. PD/VL-split)
fleet.

## Consequences

- **Availability:** a slave that transiently cannot reach the master, or a master with no FE view,
  yields chunks with no `fe_url` → those chunks fail. This is the accepted trade for a single,
  fully-attributable FE source.
- **Extra round-trip:** endpoints that previously skipped `/batch_schedule` (OpenAI-batch,
  embeddings — FE ignores their stamped `role_addrs`) now always make the call for `fe_url`.
  In-process on the master, one HTTP hop on a slave.
- **BE cursor advance:** `selectBatch` advances the master's BE round-robin cursor whenever it is
  called for `fe_url`, including for requests that do not stamp BE. This slightly perturbs BE
  round-robin fairness for those requests. A future FE-only master mode could avoid it if it
  matters.
- **Master re-election mid-request:** `MasterFeAssigner.assign` re-reads `isMaster()`/
  `isNeedConsistency()` live — the same predicate `BatchScheduleCoordinator.schedule` routed on. If
  election flips between routing and stamping, both outcomes are benign: a node demoted after
  resolving locally no-ops the stamp → those chunks fail visibly with `CHUNK_NO_FE` (self-healing on
  retry); a slave promoted after receiving a master-stamped response re-stamps with its now-master
  cursor — still one valid cursor advancing once, no collision.

## Operations

- **Rollback.** There is **no runtime switch** for this behavior — `FanoutService` no longer holds a
  local `FePool`, so FE selection cannot be toggled back to per-instance at runtime. Rolling back the
  "FE-from-master, no-fallback" behavior means deploying a build from before this change; treat it as a
  code-level, not config-level, rollback when planning a release.
- **Primary alert = `CHUNK_NO_FE`.** Alert on the `no_fe_assignment` chunk-failure rate/ratio, not on
  the log. It is the single authoritative signal that "the master is not assigning FEs". The empty-pool
  WARN is rate-limited (`suppressed=N` carries magnitude); the unexpected-exception ERROR is
  deliberately unthrottled (per-occurrence, with stack, so a real bug cannot hide behind the throttled
  WARN). Both are diagnostic only.
- **`preassign.rt` semantics changed under the same name.** The metric name is unchanged for dashboard
  continuity, but it now fires for **every** splittable batch (not only when `preAssignBe` is on), so
  its sample volume rises; and its `RESULT_EMPTY` tag now means "the master returned no targets → those
  chunks will fail with `CHUNK_NO_FE`" — a failure precursor, not the benign "no pre-assignment" it
  meant before. Update any dashboard/alert that keyed off the old volume or the old `RESULT_EMPTY`
  meaning.
- **`CHUNK_NO_FE` triage order.** (1) Is the elected master running the dispatcher
  (`dispatch.fe-pool-service-id` set)? (2) Is the FE `FePool` snapshot non-empty (FE discovery healthy)?
  (3) Can slaves reach the master? — transport errors/timeouts collapse to empty targets, the same
  symptom. A master re-election in flight self-heals on retry.

## Code

- `flexlb-common` `dao/loadbalance/BatchScheduleTarget.java` — `fe_url` field.
- `flexlb-api` `dispatcher/MasterFeAssigner.java` — the single stamping point; `ObjectProvider<FePool>`
  + consistency guards.
- `flexlb-api` `httpserver/HttpLoadBalanceServer.java` — delegates a slave-forwarded response to
  `MasterFeAssigner.assign`.
- `flexlb-api` `dispatcher/BatchScheduleClient.java` — invokes `MasterFeAssigner.assign` on the
  master's own in-process resolution (the path that bypasses the HTTP handler).
- `flexlb-api` `dispatcher/FanoutService.java` — no `FePool`; uses `plan.feUrl()`, null → fail.
- `flexlb-api` `dispatcher/BatchHandler.java` — `resolveTargets` (always calls master), conditional
  BE stamp, per-chunk `fe_url` to fanout.

## Tests

- `BatchScheduleTargetJsonTest` — `fe_url` round-trip; null omitted; legacy JSON parses.
- `HttpLoadBalanceServerTest` — master stamps from local pool; a forwarding slave does not restamp.
- `BatchScheduleClientTest` — master-local in-process resolution stamps `fe_url` from the master
  cursor; a slave's forwarded (already-stamped) response is not restamped.
- `MasterFeAssignerTest` — the single stamp point's guard/exception branches directly: null/empty
  targets no-op; an absent `FePool` bean (the deployment precondition failure) leaves `fe_url` null;
  a slave does not restamp; the empty-pool and unexpected-exception paths are swallowed leaving
  `fe_url` null (never aborting the schedule).
- `FanoutServiceTest` — uses master `fe_url`; null/short/whole-batch-null assignment fails the chunk
  with the FE client never invoked (proves no fallback).
- `BatchHandlerContractTest` — a non-preAssignable endpoint still calls the master for FE but skips
  BE stamping.
- `DispatcherE2ETest` — end-to-end fanout over three FEs, assigned per chunk index by the mock
  master; and a master-assigns-no-FE case that fails the whole batch while contacting no FE
  (end-to-end no-fallback lock).

All verified with the mutation discipline (reintroducing a fallback, dropping the stamp, dropping a
guard, or re-adding the BE-toggle gate each turns a test red).
