package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.projection.WorkSnapshot;
import static org.flexlb.balance.scheduler.ContextResult.*;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.BooleanSupplier;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Shared lifecycle primitives for scheduler contract tests.
 */
final class RequestProtocolTestSupport {
    private static final Map<QueuedRequestScheduler.Plan, RequestRoute> observedPlans =
            java.util.Collections.synchronizedMap(new java.util.WeakHashMap<>());

    static QueuedRequestScheduler.Plan plan(RequestContext context, RequestRoute routing) {
        QueuedRequestScheduler.Plan plan;
        try {
            var constructor = QueuedRequestScheduler.Plan.class.getDeclaredConstructor(GlobalQueueEntry.class, long.class);
            constructor.setAccessible(true);
            plan = constructor.newInstance(new GlobalQueueEntry(context, null), 0L);
        } catch (ReflectiveOperationException failure) {
            throw new AssertionError(failure);
        }
        ReflectionTestUtils.setField(plan, "result", PlacementResult.success(routing));
        return plan;
    }

    static RequestContext planContext(QueuedRequestScheduler.Plan plan) {
        planRouting(plan);
        return ((GlobalQueueEntry) ReflectionTestUtils.getField(plan, "entry")).context;
    }

    static RequestRoute planRouting(QueuedRequestScheduler.Plan plan) {
        PlacementResult<?, ?> result = (PlacementResult<?, ?>) ReflectionTestUtils.getField(plan, "result");
        if (result != null && result.value() != null) {
            observedPlans.put(plan, (RequestRoute) result.value());
        }
        return observedPlans.get(plan);
    }

    static QueuedRequestScheduler.Plan matchingPlan(RequestContext context, RequestRoute routing) {
        return org.mockito.ArgumentMatchers.argThat(plan -> plan != null && planContext(plan) == context && planRouting(plan) == routing);
    }

    static PlacementResult<RequestRoute, PlacementKey> published(QueuedRequestScheduler.Plan plan, RequestRoute route) {
        ReflectionTestUtils.setField(plan, "pendingDecodeRollback", null);
        return PlacementResult.success(route);
    }

    static boolean adopt(QueuedRequestScheduler scheduler, QueuedRequestScheduler.Plan plan,
                         org.flexlb.balance.eviction.DecodeCapacityAcquirer.PreemptionResult result) {
        ReflectionTestUtils.setField(plan, "preemptionResult", result);
        return Boolean.TRUE.equals(ReflectionTestUtils.invokeMethod(scheduler, "settlePreemptionResult", plan, true));
    }

    static DecodeEndpoint decodeEndpoint() {
        DecodeEndpoint endpoint = org.mockito.Mockito.mock(DecodeEndpoint.class);
        org.mockito.Mockito.lenient().when(endpoint.ownership(org.mockito.ArgumentMatchers.any()))
                .thenReturn(DecodeResources.Ownership.NONE);
        org.mockito.Mockito.lenient().when(endpoint.markQueued(org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any()))
                .thenReturn(DecodeResources.ResourceChange.of(DecodeResources.MutationStatus.STALE, false));
        org.mockito.Mockito.lenient().when(endpoint.updatePreemption(org.mockito.ArgumentMatchers.anyLong(), org.mockito.ArgumentMatchers.any()))
                .thenReturn(DecodeResources.ResourceChange.of(DecodeResources.MutationStatus.STALE, false));
        org.mockito.Mockito.lenient().when(endpoint.release(org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any()))
                .thenReturn(org.flexlb.balance.endpoint.DecodeResources.ResourceChange.of(DecodeResources.ReservationReleaseResult.RELEASED, false));
        return endpoint;
    }

    interface AdmissionCompletion extends AutoCloseable { @Override void close(); }

    static AdmissionCompletion finishOnExit(AdmissionHandle handle) {
        return () -> { if (handle != null) { handle.finish(); } };
    }

    static AbstractRequestScheduler.Publication publication(java.util.function.BooleanSupplier action) {
        return new AbstractRequestScheduler.Publication() {
            private boolean published;
            @Override public void publish() { published = action.getAsBoolean(); }
            @Override public boolean published() { return published; }
        };
    }

    static boolean publish(AbstractRequestScheduler.Publication action) { action.publish(); return action.published(); }

    static org.flexlb.balance.endpoint.DecodeEndpoint.EngineDispatchPermit handoff(java.util.function.BooleanSupplier action) {
        var permit = org.mockito.Mockito.mock(org.flexlb.balance.endpoint.DecodeEndpoint.EngineDispatchPermit.class);
        org.mockito.Mockito.when(permit.belongsTo(org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any())).thenReturn(true);
        org.mockito.Mockito.when(permit.dispatch()).thenAnswer(unused -> DecodeResources.ResourceChange.of(action.getAsBoolean()
                ? DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED
                : DecodeResources.EngineDispatchPermitTransferStatus.OWNERSHIP_LOST, false));
        return permit;
    }

    static boolean closeAdmissionAndAwaitMutations(AbstractRequestScheduler scheduler) {
        if (!SchedulerTestSupport.repository(scheduler).closeRegistration()) { return false; }
        scheduler.runtime.awaitAdmissions();
        return true;
    }

    static void expireInactiveRequest(AbstractRequestScheduler scheduler, RequestContext context, long nowMs) {
        if (context == null) { return; }
        var deadline = context.snapshot().deadlines().inactivity();
        if (deadline != null) { scheduler.onDeadline(deadline, nowMs); }
        SchedulerTestSupport.awaitRequestWork(scheduler.runtime);
    }

    static AdmissionHandle beginAdmission(AbstractRequestScheduler scheduler, RequestContext context) {
        return scheduler.claimAdmissionHandle(context.data().getRequestId(), context.data().getFuture());
    }

    @SuppressWarnings("unchecked")
    static <T> T field(RequestContext context, String name) {
        var state = context.snapshot();
        return (T) switch (name) {
            case "requestDeadline" -> state.deadlines().request();
            case "decisionDeadline" -> state.deadlines().visibility();
            case "inactivityDeadline" -> state.deadlines().inactivity();
            case "decisionVisibility" -> state.engine();
            case "lastWorkerStatusAtMs" -> state.lastWorkerAtMs();
            case "cancellationReason" -> state.cancellation() == null ? null : state.cancellation().reason();
            default -> ReflectionTestUtils.getField(context, name);
        };
    }

    /**
     * Seed cancellation in state-only fixtures without exposing an internal production operation.
     */
    static boolean recordCancellation(AbstractRequestScheduler scheduler, RequestContext context, CancelReason reason, String detail) {
        var claim = context.requestCancel(new Cancellation(0L, reason, false, System.currentTimeMillis(), null, detail));
        return claim instanceof Granted<?>;
    }

    /**
     * Inspect a private decision in state-only fixtures without widening the production API.
     */
    static <T> T inspect(AbstractRequestScheduler scheduler, RequestContext requestContext, String decision, Object... arguments) {
        synchronized (requestContext) {
            if (decision.equals("inactivityExpiresAtMsLocked")) { return (T) Long.valueOf(requestContext.snapshot().inactivityExpiresAtMs()); }
            boolean requestDecision = java.util.Arrays.stream(RequestContext.class.getDeclaredMethods())
                    .anyMatch(method -> method.getName().equals(decision));
            return requestDecision ? ReflectionTestUtils.invokeMethod(requestContext, decision, arguments)
                    : ReflectionTestUtils.invokeMethod(scheduler, decision, prepend(requestContext, arguments));
        }
    }

    static QueuedRequestScheduler publication(AbstractRequestScheduler requests) { return (QueuedRequestScheduler) requests; }

    static QueuedRequestScheduler queue(org.flexlb.config.ConfigService config, RequestWorkerSelector router,
            org.flexlb.service.monitor.DeliveryMetricsReporter reporter,
            org.flexlb.balance.eviction.DecodeCapacityAcquirer eviction, AbstractRequestScheduler owner, PlacementAvailability availability) {
        return (QueuedRequestScheduler) SchedulerTestSupport.configure(owner, config.loadBalanceConfig(), router, reporter, eviction, availability);
    }
    static RequestScheduler configure(AbstractRequestScheduler owner, org.flexlb.config.ConfigService config,
            RequestWorkerSelector router, org.flexlb.service.monitor.DeliveryMetricsReporter reporter,
            org.flexlb.balance.eviction.DecodeCapacityAcquirer eviction, PlacementAvailability availability) {
        return SchedulerTestSupport.configure(owner, config.loadBalanceConfig(), router, reporter, eviction, availability);
    }

    static void close(RequestScheduler scheduler) {
        SchedulerTestSupport.runtime(scheduler).requests().stopAccepting();
        if (scheduler instanceof QueuedRequestScheduler queue) { queue.close(); }
    }

    static int queuedCount(RequestScheduler scheduler) {
        if (!(scheduler instanceof QueuedRequestScheduler queue)) { return 0; }
        var lock = (ReentrantLock) ReflectionTestUtils.getField(queue, "lock");
        lock.lock();
        try {
            var ordered = (OrderedRequestQueue) ReflectionTestUtils.getField(queue, "orderedQueue");
            var events = (java.util.Deque<?>) ReflectionTestUtils.getField(queue, "events");
            return ordered.size() + (int) events.stream().filter(event -> {
                var kind = (Enum<?>) ReflectionTestUtils.getField(event, "kind");
                return kind.name().equals("SUBMIT") || kind.name().equals("REQUEUE");
            }).count();
        } finally {
            lock.unlock();
        }
    }

    static CompletableFuture<Response> register(AbstractRequestScheduler owner, RequestContext context) {
        return owner.register(context, () -> { }, context.data().getConfig().isQueue()
                ? StrategyErrorType.RESOURCE_EXHAUSTED : StrategyErrorType.BATCH_SLO_EXPIRED);
    }

    /** Isolates queue ordering from the common request protocol. */
    static AbstractRequestScheduler schedulerMock() {
        var config = SchedulingTestConfig.batchConfig();
        var service = org.mockito.Mockito.mock(org.flexlb.config.ConfigService.class);
        org.mockito.Mockito.when(service.loadBalanceConfig()).thenReturn(config);
        return SchedulerTestSupport.create(service, org.mockito.Mockito.mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class),
                org.mockito.Mockito.mock(org.flexlb.service.monitor.RequestSchedulerReporter.class),
                org.mockito.Mockito.mock(org.flexlb.service.RecentCacheKeyTraceReporter.class));
    }

    static AbstractRequestScheduler initialize(ResponseCompletionExecutor publisher, RequestContext context, ExpirationTimer deadlines) {
        var config = org.mockito.Mockito.mock(org.flexlb.config.ConfigService.class);
        org.mockito.Mockito.when(config.loadBalanceConfig()).thenReturn(context.data().getConfig());
        var owner = SchedulerTestSupport.create(config, org.mockito.Mockito.mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class),
                org.mockito.Mockito.mock(org.flexlb.service.monitor.RequestSchedulerReporter.class),
                org.mockito.Mockito.mock(org.flexlb.service.RecentCacheKeyTraceReporter.class));
        ReflectionTestUtils.setField(owner.runtime, "responseCompletions", publisher);
        ReflectionTestUtils.setField(owner.runtime, "timer", deadlines);
        register(owner, context);
        return owner;
    }

    static void applyPrefillStatus(AbstractRequestScheduler scheduler, PrefillEndpoint source,
            RoleType role, PrefillState.PrefillRequestStatus status) {
        applyPrefillStatus(scheduler, scheduler.requests.findActive(status.route().requestId()), source, role, status);
    }

    static void applyPrefillStatus(AbstractRequestScheduler scheduler, RequestContext context, PrefillEndpoint source,
            RoleType role, PrefillState.PrefillRequestStatus status) {
        if (context == null || status.route().prefillEp() != source) { return; }
        var kind = switch (status.kind()) {
            case ACTIVE -> EngineKind.ACTIVE;
            case COMPLETED -> EngineKind.COMPLETED;
            case FAILED -> EngineKind.FAILED;
            case PRIORITY_CANCELED -> EngineKind.PRIORITY_CANCELLED;
        };
        scheduler.onEngineEvidence(context, new EngineEvidence(status.route(), status.route().decodeReservation(),
                role, kind, status.errorCode(), null, System.currentTimeMillis()));
        SchedulerTestSupport.awaitRequestWork(scheduler.runtime);
    }

    static void applyDecodeStatus(AbstractRequestScheduler scheduler, DecodeEndpoint source, DecodeResources.DecodeRequestStatus status) {
        applyDecodeStatus(scheduler, scheduler.requests.findActive(status.reservation().requestId()), source, status);
    }

    static void applyDecodeStatus(AbstractRequestScheduler scheduler, RequestContext context, DecodeEndpoint source,
            DecodeResources.DecodeRequestStatus status) {
        if (context == null) { return; }
        var route = context.snapshot().route();
        if (route == null || route.decodeEp() != source) { return; }
        var kind = status.kind() == DecodeResources.DecodeRequestStatus.Kind.TERMINAL
                ? status.errorCode() == 0L ? EngineKind.COMPLETED : EngineKind.FAILED
                : status.allocationObserved() ? EngineKind.ALLOCATED : EngineKind.ACTIVE;
        scheduler.onEngineEvidence(context, new EngineEvidence(route, status.reservation(), RoleType.DECODE,
                kind, status.errorCode(), null, System.currentTimeMillis()));
        SchedulerTestSupport.awaitRequestWork(scheduler.runtime);
    }

    static void expireInactivity(AbstractRequestScheduler scheduler, RequestContext context, ExpirationTimer.Token exact, long nowMs) {
        scheduler.onDeadline(exact, nowMs);
        SchedulerTestSupport.awaitRequestWork(scheduler.runtime);
    }

    private static void run(Runnable continuation) {
        if (continuation != null) { continuation.run(); }
    }

    private static Object[] prepend(RequestContext context, Object[] arguments) {
        Object[] all = new Object[arguments.length + 1];
        all[0] = context;
        System.arraycopy(arguments, 0, all, 1, arguments.length);
        return all;
    }

    static EndToken claimTerminal(AbstractRequestScheduler scheduler, RequestContext context, TerminalOutcome outcome,
            Response response, boolean publish) {
        EndKind kind = outcome.phase() == RequestState.Phase.TIMED_OUT ? EndKind.INACTIVITY : EndKind.FRONTEND;
        if (outcome.phase() == RequestState.Phase.CANCELLED && context.snapshot().cancellation() == null) {
            context.requestCancel(new Cancellation(0L, CancelReason.CLIENT_CANCELLED, false,
                    System.currentTimeMillis(), StrategyErrorType.REQUEST_CANCELLED, outcome.detail()));
        }
        var claim = context.beginClose(new EndEvidence(context.snapshot().route(), kind, null,
                outcome.phase() == RequestState.Phase.COMPLETED, 0L,
                response == null ? StrategyErrorType.DISPATCH_FAILED : java.util.Arrays.stream(StrategyErrorType.values()).filter(value -> value.getErrorCode() == response.getCode()).findFirst().orElse(StrategyErrorType.DISPATCH_FAILED), outcome.detail(), false, false),
                System.currentTimeMillis());
        return claim instanceof Granted<EndToken> granted ? granted.value() : null;
    }

    private RequestProtocolTestSupport() {
    }

    static RequestContext context(FlexlbConfig config, long requestId) {
        Request request = Request.builder()
                .requestId(requestId)
                .seqLen(16L).build();
        SchedulingTestConfig.configureRequiredValues(config);
        RequestContext context = new RequestContext(config);
        context.data().setRequest(request);
        context.data().setGenerateInputPb(com.google.protobuf.ByteString.copyFromUtf8("test-input"));
        context.data().setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(1)));
        return context;
    }

    static void bind(AbstractRequestScheduler lifecycle, Registered registered) {
        try (AdmissionHandle admission = lifecycle.claimAdmissionHandle(registered.item().requestId(), registered.future()); var admissionCompletion1 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertTrue((lifecycle.commitRoute(registered.item(), RequestProtocolTestSupport.publication(() -> true)) == org.flexlb.balance.PlacementResult.Status.SUCCESS));
        }
    }

    static void beginRoutePublication(AbstractRequestScheduler lifecycle, Registered registered) {
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
    }

    static PlacementResult.Status commitRoute(AbstractRequestScheduler lifecycle, Registered registered) {
        try (AdmissionHandle admission = lifecycle.claimAdmissionHandle(registered.item().requestId(), registered.future()); var admissionCompletion2 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            return lifecycle.commitRoute(registered.item(), RequestProtocolTestSupport.publication(() -> true));
        }
    }

    static DeliveryClaim claimRoute(AbstractRequestScheduler registry, RequestRoute item, BooleanSupplier handoff) {
        DeliveryClaim claim = claimRouteWithoutPrediction(registry, item, handoff);
        if (claim != null) {
            registry.onDeliveryEvidence(claim, new Prediction(claim, System.currentTimeMillis() + 30_000L, System.currentTimeMillis()));
        }
        return claim;
    }

    static DeliveryClaim claimBatch(AbstractRequestScheduler registry, RequestRoute item, long batchId, BooleanSupplier handoff) {
        DeliveryClaim claim = claimBatchWithoutPrediction(registry, item, batchId, handoff);
        if (claim != null) {
            registry.onDeliveryEvidence(claim, new Prediction(claim, System.currentTimeMillis() + 30_000L, System.currentTimeMillis()));
        }
        return claim;
    }

    static DeliveryClaim claimRouteWithoutPrediction(AbstractRequestScheduler registry, RequestRoute item, BooleanSupplier handoff) {
        return registry.claimDelivery(item, DeliveryClaimKind.ROUTE_DECISION, 0L, RequestProtocolTestSupport.handoff(handoff));
    }

    static DeliveryClaim claimBatchWithoutPrediction(AbstractRequestScheduler registry, RequestRoute item, long batchId, BooleanSupplier handoff) {
        return registry.claimDelivery(item, DeliveryClaimKind.BATCH_ENQUEUE, batchId, RequestProtocolTestSupport.handoff(handoff));
    }

    private static WorkSnapshot emptyWork() {
        return new WorkSnapshot(System.currentTimeMillis(), java.util.List.of(), java.util.List.of(), 0L);
    }

    // State-only fixtures deliberately seed a phase without executing publication or timers.
    static void startRouteDelivery(AbstractRequestScheduler scheduler, RequestContext context) {
        assertNotNull(scheduler.claimDelivery(context.snapshot().route(), DeliveryClaimKind.ROUTE_DECISION,
                0L, handoff(() -> true)));
    }

    static void startBatchDelivery(AbstractRequestScheduler scheduler, RequestContext context, long batchId) {
        assertNotNull(scheduler.claimDelivery(context.snapshot().route(), DeliveryClaimKind.BATCH_ENQUEUE,
                batchId, handoff(() -> true)));
    }

    static ExpirationTimer.Token markDecodeAccepted(RequestContext context) {
        var previous = context.snapshot().deadlines().visibility();
        var route = context.snapshot().route();
        context.recordEngineEvidence(new EngineEvidence(route, route.decodeReservation(), RoleType.DECODE,
                EngineKind.ACTIVE, 0L, null, System.currentTimeMillis()));
        return previous;
    }

    static void markAcknowledged(RequestContext context) {
        assertTrue(Thread.holdsLock(context));
        var claim = context.snapshot().delivery().identity();
        if (context.snapshot().delivery().send() == SendState.NOT_STARTED) {
            if (claim.kind() == DeliveryClaimKind.BATCH_ENQUEUE) {
                context.recordDeliveryResult(new TransportResult(claim, org.flexlb.balance.delivery.DeliveryResult.delivered(), System.currentTimeMillis()));
            } else {
                context.recordDeliveryResult(new RouteDelivered(claim, System.currentTimeMillis() + 30_000L, System.currentTimeMillis()));
            }
        }
        var selected = context.selectResponse(new ResponseCandidate(context, ResponseKind.CONFIRMATION, null, System.currentTimeMillis()));
        org.junit.jupiter.api.Assertions.assertInstanceOf(Granted.class, selected);
    }

    static Runnable acknowledge(AbstractRequestScheduler scheduler, RequestContext context) {
        var snapshot = context.snapshot();
        var claim = snapshot.delivery().identity();
        return () -> {
            if (claim.kind() == DeliveryClaimKind.BATCH_ENQUEUE) {
                scheduler.onDeliveryEvidence(claim, new TransportResult(claim,
                        org.flexlb.balance.delivery.DeliveryResult.delivered(), System.currentTimeMillis()));
            } else {
                scheduler.onDeliveryEvidence(claim, new RouteDelivered(claim, System.currentTimeMillis() + 30_000L,
                        System.currentTimeMillis()));
            }
        };
    }

    static boolean prepareMember(AbstractRequestScheduler registry, RequestRoute item) {
        var state = item.ctx().snapshot();
        return registry.requests.isCurrent(item.ctx()) && state.route() == item && state.preparedDelivery();
    }

    static void awaitGlobalCapacityWaiters(RequestScheduler scheduler, int expected) throws InterruptedException {
        var queue = (QueuedRequestScheduler) scheduler;
        var lock = (ReentrantLock) ReflectionTestUtils.getField(queue, "lock");
        var waiting = (PlacementCapacityWaitQueue) ReflectionTestUtils.getField(queue, "waitingRequests");
        var membership = (Map<?, ?>) ReflectionTestUtils.getField(waiting, "membership");
        awaitCondition(() -> {
            lock.lock();
            try { return membership.size() == expected; }
            finally { lock.unlock(); }
        });
    }

    static void awaitGlobalCapacityWaiters(AbstractRequestScheduler scheduler, int expected)
            throws InterruptedException {
        Object coordinator = scheduler;
        var lock = (ReentrantLock)
                ReflectionTestUtils.getField(coordinator, "lock");
        var waitQueue = (PlacementCapacityWaitQueue) ReflectionTestUtils.getField(coordinator, "waitingRequests");
        var waiting = (Map<?, ?>)
                ReflectionTestUtils.getField(waitQueue, "membership");
        // A route's close callback runs before park. Observe actual wait registration
        // under the coordinator lock, rather than treating that callback as a barrier.
        awaitCondition(() -> {
            lock.lock();
            try {
                return waiting.keySet().stream().filter(entry -> waitQueue.isWaiting((GlobalQueueEntry) entry)).count() == expected;
            } finally {
                lock.unlock();
            }
        });
    }

    static void await(CountDownLatch latch) {
        try {
            if (!latch.await(5, TimeUnit.SECONDS)) {
                throw new AssertionError("latch was not released");
            }
        } catch (InterruptedException interrupted) {
            Thread.currentThread().interrupt();
            throw new AssertionError(
                    "interrupted while awaiting latch", interrupted);
        }
    }

    static void awaitCondition(BooleanSupplier condition)
            throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (System.nanoTime() < deadline) {
            if (condition.getAsBoolean()) {
                return;
            }
            Thread.sleep(1L);
        }
        assertTrue(condition.getAsBoolean(), "condition did not become true");
    }

    record Registered(RequestRoute item, CompletableFuture<Response> future) {
    }
    static void transport(AbstractRequestScheduler scheduler, DeliveryClaim claim, org.flexlb.balance.delivery.DeliveryResult result) {
        scheduler.onDeliveryEvidence(claim, new TransportResult(claim, result, System.currentTimeMillis()));
    }

    static void prediction(AbstractRequestScheduler scheduler, DeliveryClaim claim, WorkSnapshot work, long predictedMs) {
        long now = System.currentTimeMillis();
        long deadline = org.flexlb.balance.prediction.PrefillPredictionBoundary.predictedVisibilityDeadlineMs(
                work, predictedMs, claim.owner().data().getConfig().getRequestLifecycle().getDecision().getLifetime(), now);
        scheduler.onDeliveryEvidence(claim, new Prediction(claim, deadline, now));
    }

    static void routeDelivered(AbstractRequestScheduler scheduler, DeliveryClaim claim, WorkSnapshot work, long predictedMs) {
        long now = System.currentTimeMillis();
        long deadline = org.flexlb.balance.prediction.PrefillPredictionBoundary.predictedVisibilityDeadlineMs(
                work, predictedMs, claim.owner().data().getConfig().getRequestLifecycle().getDecision().getLifetime(), now);
        scheduler.onDeliveryEvidence(claim, new RouteDelivered(claim, deadline, now));
    }

    static boolean preemptionPhase(AbstractRequestScheduler scheduler, PreemptionRegistration claim,
                                   org.flexlb.balance.preemption.PreemptionCancelPhase phase) {
        return scheduler.onPreemption(claim, new AbstractRequestScheduler.PreemptionEvent(phase, false, null, null, null));
    }

    static boolean releasePreemption(AbstractRequestScheduler scheduler, PreemptionRegistration claim) {
        return scheduler.onPreemption(claim, new AbstractRequestScheduler.PreemptionEvent(null, true, null, null, null));
    }

    static boolean finishPreemption(AbstractRequestScheduler scheduler, PreemptionRegistration claim, String detail) {
        var route = claim.owner().snapshot().route();
        return route != null && scheduler.onPreemption(claim, new AbstractRequestScheduler.PreemptionEvent(
                null, false, route.decodeEp(), claim.reservation(), detail));
    }

    static void decodeRetired(AbstractRequestScheduler scheduler, RequestContext context, DecodeEndpoint source,
                              DecodeResources.ReservationHandle reservation) {
        var route = context.snapshot().route();
        if (route == null || route.decodeEp() != source) { return; }
        scheduler.onEngineEvidence(context, new EngineEvidence(route, reservation, RoleType.DECODE,
                EngineKind.RETIRED, 0L, "Decode endpoint generation retired", System.currentTimeMillis()));
    }

    static PreemptionRegistration activePreemption(RequestContext context) {
        var facts = context.snapshot().preemption();
        return facts == null || facts.finished() ? null : facts.identity();
    }

    static boolean preemptionFinished(PreemptionRegistration claim) {
        var facts = claim.owner().snapshot().preemption();
        var result = resolved(claim);
        return facts != null && facts.identity() == claim && facts.finished()
                || result != null && result.outcome() == org.flexlb.balance.preemption.VictimResolution.Outcome.REQUEST_END;
    }

    static org.flexlb.balance.preemption.VictimResolution resolved(PreemptionRegistration claim) {
        return claim.owner().data().getScheduler().runtime.signals().resolved(claim);
    }

    static <T> T awaitResult(java.util.concurrent.CompletableFuture<T> future) {
        try { return future.get(5L, TimeUnit.SECONDS); }
        catch (Exception failure) { throw new AssertionError("request result did not complete", failure); }
    }

}
