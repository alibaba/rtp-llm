# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

The approved refactor target is [the proposed component design](../docs/scheduler-design/proposed-component-classes.html). Follow its complete responsibilities and budgets; [the current implementation diagram](../docs/current-class-diagram.md) records partial progress, not completion. Historical proposals that delete RequestLifecycle or add SchedulerLoop do not supersede this target.

## Module Overview

flexlb-sync is the core load balancing module of FlexLB. It handles:

- Full-fleet Prefill and Decode cost selection and VIT random selection
- Worker node status synchronization via gRPC
- Master election using ZooKeeper
- Request routing across different role types (PREFILL, DECODE, PDFUSION, VIT)
- Integration with flexlb-cache for KV cache management

## Key Architecture Concepts

### Routing and scheduling

- `RequestWorkerSelector` performs one multi-role selection and returns exact endpoint-generation capabilities
- `RequestScheduler` defines scheduling timing, request/endpoint change notifications, and stopping scheduling. `RequestLifecycle` owns admission and cancellation; `ApplicationLifecycle` orders service shutdown
- `AbstractRequestScheduler` binds RequestLifecycle, shares its repository and validates the default scheduling mode. `RequestLifecycle` coordinates resource handoff, request effects and settlement; `RequestContext` owns atomic request facts and result precedence
- `DirectRequestScheduler` owns immediate scheduling and selection; DeliveryDispatcher performs resource preparation and commit
- `QueuedRequestScheduler` owns triggering, planning and capacity waits; queue interfaces own ordering and GroupPlanner owns batching rules
- `PlacementConfiguration` constructs the fixed-mode scheduler at startup
- `RequestRepository` owns shared registration, identity lookup, and archival; each request retains its original Scheduler
- `PrefillEndpoint` directly owns the Prefill generation, resource records, status reconciliation and `PrefillRequestQueue` index. `PrefillState` and `PrefillObservation` have been deleted. Resource release and worker-selection cleanup no longer use a separate State-to-Endpoint release report; notifications run after unlocking. `GroupPlanner` is shared by scheduling and prediction. The merged class is still above budget.
- `DecodeEndpoint` directly owns the Decode resource records and mutation lock; `DecodeState` has been deleted. Queue resource preparation is in `DeliveryDispatcher.enqueueRoute`, with unpublished preparation released by `finishPlacement`. The default scheduler retains retry and preemption timing. These changes do not yet satisfy all class budgets.
- `PrefillEndpoint` receives the actual dispatch mode as a value and has no Dispatcher dependency. Direct and queued dispatch share Decode reservation preparation; an attachment failure releases the newly acquired exact handle before propagating the original error.
- `DeliveryDispatcher` chooses NON_BATCH or BATCH delivery without reselecting endpoints, using the exact request owner and the same `Delivery.Member` identity from preparation through delivery

### Role-Based Routing

The system routes requests through multiple worker types based on model requirements:

- **PREFILL**: Initial token processing
- **DECODE**: Token generation
- **PDFUSION**: Prefill-Decode fusion workers
- **VIT**: Vision-language model processing

Key classes:

- `RoleType` (in flexlb-common): Enum defining worker roles
- `RequestWorkerSelector`: Combines role selections and closes selected Workers when selection fails
- `RequestContext`: Owns request facts, exact identities, event acceptance, result precedence, and atomic lifecycle changes; it does not call Endpoint operations or construct execution tasks
- `RequestRepository`: Owns shared request identities and exact archival only
- `RequestLifecycle`: Owns admission handles, cancellation, deadlines and request settlement; `ExpirationTimer` schedules exact registrations. `ApplicationLifecycle` in flexlb-api orders shutdown; execution components drain and close their own work

### Worker Status Synchronization

- `WorkerStatusSynchronizer`: Owns periodic work and retirement retries using existing executors
- `GrpcWorkerStatusRunner`: Fetches worker status via gRPC
- `EndpointRegistry`: Owns discovered generations, endpoint publication/retirement and immutable routing snapshots
- `GrpcCacheStatusCheckRunner`: Syncs KV cache status with flexlb-cache module

### Master Election

- `ZookeeperMasterElectService`: ZooKeeper-based leader election
- `MasterStatusService`: Exposes local node identity and elected master status using the identity captured once by the ZooKeeper election service

## Development Commands

### Build this module only

```bash
# From flexlb-sync directory
mvn clean package -DskipTests

# From parent directory
mvn clean package -pl flexlb-sync -DskipTests
```

### Run tests

```bash
# Run all tests in this module
mvn test

# Run a specific test class
mvn test -Dtest=RequestWorkerSelectorTest

# Run a specific test method
mvn test -Dtest=RequestWorkerSelectorTest#testRouteSuccess
```

### Code formatting check

```bash
# From parent directory
mvn spotless:check -Pspotless-check

# Auto-format code
mvn spotless:apply -Pspotless-check
```

## Project Structure

```
flexlb-sync/
├── balance/
│   ├── endpoint/                  # exact generation ownership and capacity
│   ├── eviction/                  # priority-only exact-route preemption
│   ├── planner/                   # endpoint-local decision-group formation
│   ├── prediction/                # Prefill execution-time estimators
│   ├── projection/                # frozen queue/TTFT projections
│   ├── scheduler/
│   │   ├── RequestWorkerSelector.java       # one-pass multi-role routing
│   │   ├── DirectRequestScheduler.java # immediate selection/commit
│   │   ├── QueuedRequestScheduler.java # model-wide ordering/commit
│   │   ├── RequestContext.java      # request lifecycle protocol
│   │   ├── RequestScheduler.java    # public scheduling and cancellation contract
│   │   ├── RequestRepository.java   # exact shared registration and archival
│   │   ├── RequestLifecycle.java    # admission, cancellation, deadlines and settlement
│   └── strategy/
│       ├── VitWorkerSelector.java             # VIT selection
│       ├── CostBasedPrefillStrategy.java   # Predicted Prefill cost strategy
│       └── DecodeSelector.java    # Capacity-filtered Decode cost selection
├── consistency/
│   ├── MasterStatusView.java      # Master election interface
│   └── ZookeeperMasterElectService.java  # ZK implementation
├── sync/
│   ├── runner/
│   │   ├── GrpcWorkerStatusRunner.java    # Worker status sync
│   │   └── GrpcCacheStatusCheckRunner.java # Cache status sync
└── service/
    └── grpc/
        └── WorkerStatusRpcClient.java   # gRPC client for workers
```

## Key Dependencies

This module depends on:

- **flexlb-common**: Shared data models (`ServerStatus`, `RoleType`, `WorkerStatus`); `RequestContext` belongs to flexlb-sync
- **flexlb-cache**: KV cache management (`CacheAwareService`, `KvCacheManager`)
- **flexlb-grpc**: gRPC protocol definitions and clients

External dependencies:

- Spring Boot 2.7.18 (WebFlux for reactive programming)
- Apache Curator 5.4.0 (ZooKeeper client)
- gRPC 1.65.1
- Caffeine (local caching)
- OpenTelemetry (distributed tracing)

## Important Implementation Notes

### Rollback Mechanism

When routing fails for a later role type (e.g., PREFILL succeeds but DECODE fails),
`RequestRouteSelector` discards incomplete selections without closing shared Worker endpoints.
The request scheduler acquires resource reservations after selection and owns
rollback until queue publication or direct delivery handoff.

### Load Balancer Selection

The router calls explicit role selectors. Prefill and Decode selectors must
evaluate the complete live fleet before reducing to the configured policy
winner. Capacity commit, delivery, and priority preemption must consume that
winner rather than invoke another selector.

### Worker Status Updates

Worker status is updated asynchronously by scheduled runners. `EndpointRegistry`
owns discovered worker identities and generation-fenced endpoint publication,
and exposes immutable routing snapshots and exact captures to strategies.
RPC completion records validated heartbeat evidence independently of status
application. Each generation permits one status RPC and one pending application;
application backlog must not delay subsequent heartbeats or acknowledge unapplied
finished-task cursors. Discovery errors retain the last known workers. Successful
discovery removal or three validated `alive=false` responses close admission and
request retirement; transport errors and elapsed local time do not prove death.

### Cache Integration

The module calls `CacheAwareService` from flexlb-cache to update cache information
received from workers. See `GrpcCacheStatusCheckRunner`.
Cache mutation and retirement cleanup share the generation's cache lock, separate
from status application. Replacement waits for exact endpoint drain and successful
cache cleanup. Same-version replies reuse indexed keys while refreshing capacity;
failed indexing leaves the acknowledged cursor unchanged so the payload is retried.

## Configuration

This module reads configuration from:

- `FLEXLB_CONFIG`: Load balance strategy, timeouts, batch settings
- `FLEXLB_SYNC_CONSISTENCY_CONFIG`: ZooKeeper connection, master election
- `MODEL_SERVICE_CONFIG`: Backend worker endpoints

Configuration is injected via `ConfigService` interface (implementation in flexlb-common).

## Testing Strategy

- Unit tests use Mockito 5.20.0 (no PowerMock needed with Java 21)
- Test classes mirror source structure (e.g., `RequestWorkerSelectorTest` for `RequestWorkerSelector`)
- Mock external dependencies (gRPC clients, cache managers, config service)
- Focus on routing logic, strategy selection, and error handling

## Important Reminders

Do what is asked; no more, no less.
Always prefer editing existing files over creating new ones.
Do not proactively create documentation files (\*.md) or README files unless explicitly requested.

Spring creates one scheduler at startup. The scheduler and request contexts use the ConfigService startup configuration directly; changes to scheduling and shared PrefillEndpoint settings require restart. BATCH abandonment requires sender exit and exact remote cleanup proof before archival; response completion does not release these obligations. Ordinary cancellation capability is checked before using the Engine cleanup protocol.

Prefill policy owns capacity failure classification through read-only CandidateInput methods; RequestWorkerSelector preserves policy results. Endpoint commitDirect validates the request record, retirement and capacity in its event loop; RequestRoute holds a direct Endpoint reference. Synchronous queued preemption owns incoming-reservation rollback after withdrawal failures; AdmissionHandle has no placement-failure callback protocol.

DeliveryDispatcher receives TaskTimer directly for transport/preemption timeouts. RequestLifecycle has no TaskTimer dependency; it owns request deadlines through ExpirationTimer.

RequestLifecycle.submit/cancel is the generation request entry; RequestSchedulerEntry is deleted. VIT-only and stale VIT preselection are handled by FlexlbServiceImpl through RequestWorkerSelector before generation admission. RequestLifecycle must not depend on the selector or choose queue rules. DeliveryDispatcher owns the shared startup Decode admission capacity; lifecycle and scheduler do not carry or forward it.

PlacementFacts holds the exact operation and publication status; acquired route and pending terminal duplicates are removed. RequestContext.route and pendingEnd are authoritative. Pending ends must not suppress deadline reinstallation while placement is unfinished. Registry aliases and endpoint transport addresses remain distinct identities; do not replace routing-view directory keys with endpoint.ipPort(). Monitoring uses tryGetLoadMetric and skips a busy endpoint; accounting assertions use exact resource snapshots.

ExpirationTimer borrows TaskTimer and never creates or closes a pool. ScheduledTaskTimer owns the existing separate deadline and cleanup timer pools. ApplicationLifecycle closes deadline registrations, then their timer pool before draining request tasks; the cleanup timer stays available through delivery cleanup. TaskTimer returns the original JDK Future; cancel(false) must not interrupt an already running callback.

PlacementConfiguration creates an unstarted scheduler. ApplicationLifecycle binds and starts both default and custom schedulers. Tests explicitly start their scheduler when background progress is needed; configuration factory methods do not start threads.

QueuedRequestScheduler, its PrefillQueueScheduler, and Delivery do not own DeliveryExecution. Scheduler constructs the selected delivery input; DeliveryDispatcher uses its injected execution component for capacity checks/subscriptions and resource preparation. Scheduling still chooses candidates and prefix policy. Route-submit telemetry is emitted at Dispatcher publication, not through a scheduler forwarding method.

Request observations use one immutable ContextResult.Snapshot; the SnapshotFacts delegate wrapper was removed. No-change observations retain snapshot identity, and later transitions cannot mutate historical snapshots.

Scheduler.awaitStopped includes accepted selection work, not just task submission. Direct tracks accepted selection tasks on the existing RequestTaskPool; Queue identifies its planner threads and rejects self-wait. Settlement and shared executor shutdown stay with their existing owners.

Transport confirmation uses RequestContext.confirmDelivery directly; it never selects a frontend response. ACK response selection remains a separate cancellation boundary. ResponseCandidate no longer copies End, and exceptional Future completion uses the existing terminal candidate with the original Throwable.

Prefill status reduction directly returns WorkerEndpoint.StatusChange.Prefill. The redundant StatusReconciliation wrapper was removed; resource changes remain under the endpoint lock and notifications remain outside it.

Direct and queued placement use `DeliveryDispatcher.finishPlacement(route, admission)`: it releases unpublished resources and finishes admission. Both paths store the exact Prefill reservation on Route before publication callbacks. ROLLBACK_COMPLETED requires successful release of both endpoints; published routes remain owned by RequestLifecycle. PlacementPlan calls this same complete operation and does not implement rollback. The partial close and close(tasks) protocols are removed.

CloseProgress.notificationsDone includes successful deadline registration updates and owner notifications. Context does not duplicate ExpirationTimer state in a timersDone field. Failed or blocked timer updates must keep the request unarchived; RequestTerminalSettlementTest and RequestNotificationOwnershipRaceTest cover those boundaries.

Immediate custom schedulers call `DeliveryDispatcher.submitRoute(route, admission)`. The dispatcher finishes admission before publishing the existing RouteDelivered evidence; RouteDelivery and commitSelectedRoute are removed. Queued Prefill commits accept only the selected plan, never a caller-supplied selected Worker. Decode permit resolution and its request mutation run in one Endpoint event; transferDispatch and synchronized(permit) are removed. Reservation occupancy remains visible to prediction before commit. These protocol removals do not imply that the five core class budgets are met.

QueuedCommit and CommittedHandoff are deleted. Endpoint validates the prepared Delivery itself; that same object retains its Endpoint reference, preceding-work snapshot and commit fact. Closing a committed Delivery releases temporary dispatch permits, never committed request rows. Batch features are built outside the Endpoint event loop and shared by prediction and the batch record. RequestLifecycle writes release evidence directly to Context.CloseFacts; the intermediate Settlement record is deleted. Delivery errors use the same advanceRequest path as other request events.

Direct commit returns only the preceding WorkSnapshot. RequestRoute holds its Endpoint reference; no separate selection object or Delivery wrapper is allocated. Dispatcher reads the existing DeliveryPermit state to decide whether the executor owns cleanup; Delivery.transferred is deleted. Failure before executor acceptance remains the submitting caller’s cleanup responsibility. finishPlacement returns resource cleanup errors for reporting; admission completion errors propagate.

Confirmation eligibility is shared through ContextResult.Snapshot.canConfirmDelivery; Context still arbitrates inactivity and preemption. AdmissionHandle delegates request-state eligibility to Context.beginPlacement under the same monitor. Closed request-task admission is not retried; its registration failure propagates. Unaccepted task submission, response registration, physical settlement and deadline recovery remain; DeliveryClaimTest covers recovery without another event. Prefill status application rereads ledger facts under lock and reuses out-of-lock prediction only for identical surviving members; in-lock failure retires before unlocking.

Blocked-head bypass belongs to PrefillRequestQueue: custom queues default to strict order; OrderedPrefillRequestQueue retains the existing same-priority, different-Decode, once-per-head rule. PrefillEndpoint records successful dispatch under its existing lock. QueuedRequestScheduler never replaces GroupPlanner.dispatchReason for bypass candidates and retains the planner deadline alongside the blocked head capacity subscription. Queue decisions never settle requests.

PlacementResult.ENDPOINT_CHANGED distinguishes successful Prefill/Decode selection from failure to capture its exact generation. Its selection field retains the original SUCCESS result; RequestWorkerSelector no longer reselects Decode internally. Default schedulers retain their own retry/rejection behavior. Strategy failures are forwarded unchanged.
