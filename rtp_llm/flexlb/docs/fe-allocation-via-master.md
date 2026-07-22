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
       next() reads a fresh snapshot each call, skipping FeHealthChecker-dead hosts
  -> HttpLoadBalanceServer.assignFeUrls() calls fePool.next() to stamp each target.fe_url
```

`ServiceDiscovery` is the same VipServer-backed infrastructure the master already uses for BE
workers (`flexlb-common`, consumed by `flexlb-sync`); only the serviceId differs
(`dispatch.fe-pool-service-id` for FE vs the BE worker service). No new module dependency, no new
link.

Every dispatcher node (including the elected master) runs its own refresher against the same
serviceId, so all nodes see an eventually-consistent FE set. The master's list is not "more
authoritative" — the value of the design is a **single cursor**, not a more-accurate list. FE
liveness is the master's own `FeHealthChecker` view: `fePool.next()` skips hosts the master marks
dead, and the dispatcher trusts the master's pick without re-probing.

## Data flow

```
dispatcher receives batch
  -> BatchScheduleClient.requestTargets(count)          in-process on the master, HTTP-forward on a slave
       -> BatchScheduleCoordinator.schedule()
            master:  RouteService.batchSchedule -> DefaultRouter -> RoundRobinLoadBalancer.selectBatch
            slave:   forwardToMaster (HTTP) to the elected master
  -> master's HttpLoadBalanceServer.assignFeUrls(response): for each target, fe_url = fePool.next()
       (only when this node resolved locally AND the FePool bean exists — see Guards)
  -> BatchHandler: BE role_addrs stamped only when preAssignBe && spec.isPreAssignable();
                   per-chunk fe_url extracted from targets and passed to fanout
  -> FanoutService.dispatchOne: send chunk to target.fe_url; null fe_url -> throw -> CHUNK_PICK_FAILED
```

## Guards (correctness)

`HttpLoadBalanceServer.assignFeUrls` stamps `fe_url` only when both hold:

1. **This node resolved the batch locally** — it is the elected master, or consistency is off
   (`!isNeedConsistency() || isMaster()`). A slave that merely forwarded to the master already
   holds the master's assignment in the response; re-stamping it with the slave's own cursor would
   reintroduce the collision the feature removes.
2. **The `FePool` bean exists** (injected via `ObjectProvider`) — i.e. this master node also runs
   the dispatcher. Absent it, targets keep `fe_url == null`.

The master's own in-process dispatcher does not pass through `HttpLoadBalanceServer`; it keeps
using `fePool.next()` at fanout time — the **same** `FePool` bean, hence the same cursor — so the
master node and every slave advance one global cursor.

## Decisions

- **No fallback.** Determinism of load attribution is prioritized over availability. A chunk with
  no master `fe_url` fails visibly (`CHUNK_PICK_FAILED`) rather than silently rerouting to a
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

## Code

- `flexlb-common` `dao/loadbalance/BatchScheduleTarget.java` — `fe_url` field.
- `flexlb-api` `httpserver/HttpLoadBalanceServer.java` — `assignFeUrls`, `ObjectProvider<FePool>`.
- `flexlb-api` `dispatcher/FanoutService.java` — no `FePool`; uses `plan.feUrl()`, null → fail.
- `flexlb-api` `dispatcher/BatchHandler.java` — `resolveTargets` (always calls master), conditional
  BE stamp, per-chunk `fe_url` to fanout.

## Tests

- `BatchScheduleTargetJsonTest` — `fe_url` round-trip; null omitted; legacy JSON parses.
- `HttpLoadBalanceServerTest` — master stamps from local pool; a forwarding slave does not restamp.
- `FanoutServiceTest` — uses master `fe_url`; null/short assignment fails the chunk with the FE
  client never invoked (proves no fallback).
- `BatchHandlerContractTest` — a non-preAssignable endpoint still calls the master for FE but skips
  BE stamping.
- `DispatcherE2ETest` — end-to-end fanout over three FEs, now assigned per chunk index by the
  mock master.

All verified with the mutation discipline (reintroducing a fallback, dropping the stamp, dropping a
guard, or re-adding the BE-toggle gate each turns a test red).
