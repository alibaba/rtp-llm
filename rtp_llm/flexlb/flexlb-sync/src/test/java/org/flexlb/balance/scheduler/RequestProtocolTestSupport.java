package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.RequestContext.AdmissionHandle;
import org.flexlb.balance.scheduler.RequestContext.DeliveryClaim;
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
        org.mockito.Mockito.lenient().when(endpoint.release(org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any()))
                .thenReturn(DecodeResources.ReservationReleaseResult.RELEASED);
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
        org.mockito.Mockito.when(permit.dispatch()).thenAnswer(unused -> action.getAsBoolean()
                ? DecodeResources.EngineDispatchPermitTransferStatus.TRANSFERRED
                : DecodeResources.EngineDispatchPermitTransferStatus.OWNERSHIP_LOST);
        return permit;
    }

    static boolean closeAdmissionAndAwaitMutations(AbstractRequestScheduler scheduler) {
        if (!org.flexlb.balance.scheduler.SchedulerTestSupport.repository(scheduler).closeRegistration()) { return false; }
        scheduler.awaitAdmissionMutations();
        return true;
    }

    static void expireInactiveRequest(AbstractRequestScheduler scheduler, RequestContext context, long nowMs) {
        if (context == null) { return; }
        Runnable effect = inspect(scheduler, context, "decideInactivityLocked", nowMs, null);
        ReflectionTestUtils.invokeMethod(scheduler, "execute", context, effect);
    }

    static AdmissionHandle beginAdmission(AbstractRequestScheduler scheduler, RequestContext context) {
        return scheduler.claimAdmissionHandle(context.getRequestId(), context.getFuture());
    }

    @SuppressWarnings("unchecked")
    static <T> T field(RequestContext context, String name) {
        return (T) ReflectionTestUtils.getField(context, name);
    }

    /**
     * Seed cancellation in state-only fixtures without exposing an internal production operation.
     */
    static boolean recordCancellation(AbstractRequestScheduler scheduler, RequestContext requestContext, CancelReason reason, String message) {
        return Boolean.TRUE.equals(ReflectionTestUtils.invokeMethod(scheduler, "recordCancellationLocked", requestContext, reason, message));
    }

    /**
     * Inspect a private decision in state-only fixtures without widening the production API.
     */
    static <T> T inspect(AbstractRequestScheduler scheduler, RequestContext requestContext, String decision, Object... arguments) {
        synchronized (requestContext) {
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
        SchedulerTestSupport.runtime(scheduler).stopAccepting();
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
        var future = owner.register(context, context.getConfig().isQueue()
                ? StrategyErrorType.RESOURCE_EXHAUSTED : StrategyErrorType.BATCH_SLO_EXPIRED);
        if (context.scheduler() == owner && !future.isDone()) {
            if (context.getConfig().isQueue()) { owner.expirationTimer().scheduleRequestDeadline(context, context.getRequestExpiresAtMs()); }
            owner.expirationTimer().scheduleInactivityDeadline(context);
        }
        return future;
    }

    /** Isolates queue ordering from the common request protocol. */
    static AbstractRequestScheduler schedulerMock() {
        var config = SchedulingTestConfig.batchConfig();
        var service = org.mockito.Mockito.mock(org.flexlb.config.ConfigService.class);
        org.mockito.Mockito.when(service.loadBalanceConfig()).thenReturn(config);
        var runtime = new SchedulerRuntime(new RequestRepository(), org.mockito.Mockito.mock(org.flexlb.balance.endpoint.EndpointRegistry.class),
                org.mockito.Mockito.mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class),
                org.mockito.Mockito.mock(org.flexlb.service.monitor.RequestSchedulerReporter.class),
                org.mockito.Mockito.mock(DefaultBatchDispatcher.class), service,
                org.mockito.Mockito.mock(org.flexlb.service.RecentCacheKeyTraceReporter.class),
                org.mockito.Mockito.mock(org.flexlb.balance.eviction.EngineCancelChannel.class));
        var timer = org.mockito.Mockito.mock(ExpirationTimer.class);
        return org.mockito.Mockito.mock(QueuedRequestScheduler.class, org.mockito.Mockito.withSettings()
                .useConstructor(config, org.mockito.Mockito.mock(RequestWorkerSelector.class), runtime.deliveryReporter(),
                        org.mockito.Mockito.mock(org.flexlb.balance.eviction.DecodeCapacityAcquirer.class), runtime, new PlacementAvailability())
                .defaultAnswer(invocation -> {
                    String name = invocation.getMethod().getName();
                    if (name.equals("expirationTimer")) { return timer; }
                    if (invocation.getMethod().getDeclaringClass() == QueuedRequestScheduler.class) {
                        return invocation.callRealMethod();
                    }
                    return org.mockito.Answers.RETURNS_DEFAULTS.answer(invocation);
                }));
    }

    static AbstractRequestScheduler initialize(ResponseCompletionExecutor publisher, RequestContext context, ExpirationTimer timer) {
        var config = org.mockito.Mockito.mock(org.flexlb.config.ConfigService.class);
        org.mockito.Mockito.when(config.loadBalanceConfig()).thenReturn(context.getConfig());
        var owner = SchedulerTestSupport.create(config, org.mockito.Mockito.mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class),
                org.mockito.Mockito.mock(org.flexlb.service.monitor.RequestSchedulerReporter.class),
                org.mockito.Mockito.mock(org.flexlb.service.RecentCacheKeyTraceReporter.class));
        ReflectionTestUtils.setField(owner, "responseCompletions", publisher);
        ReflectionTestUtils.setField(owner, "expirationTimer", timer);
        register(owner, context);
        return owner;
    }

    static void applyPrefillStatus(AbstractRequestScheduler scheduler, PrefillEndpoint source,
            RoleType role, PrefillState.PrefillRequestStatus requestStatus) {
        applyPrefillStatus(scheduler, scheduler.findRequestContext(requestStatus.route().requestId()), source, role, requestStatus);
    }

    static void applyPrefillStatus(AbstractRequestScheduler scheduler, RequestContext context,
            PrefillEndpoint source, RoleType role,
            PrefillState.PrefillRequestStatus requestStatus) {
        if (context != null) { run(context.scheduler().acceptPrefillStatus(context, source, role, requestStatus, System.currentTimeMillis())); }
    }

    static void applyDecodeStatus(AbstractRequestScheduler scheduler, DecodeEndpoint source,
            DecodeResources.DecodeRequestStatus requestStatus) {
        applyDecodeStatus(scheduler, scheduler.findRequestContext(requestStatus.reservation().requestId()), source, requestStatus);
    }

    static void applyDecodeStatus(AbstractRequestScheduler scheduler, RequestContext context,
            DecodeEndpoint source, DecodeResources.DecodeRequestStatus requestStatus) {
        if (context != null) { run(context.scheduler().acceptDecodeStatus(context, source, requestStatus, System.currentTimeMillis())); }
    }

    static void expireInactivity(AbstractRequestScheduler scheduler, RequestContext context,
            ExpirationTimer.InactivityDeadline deadline, long nowMs) {
        scheduler.enqueueInactivityDeadline(context, deadline, nowMs, () -> { });
        scheduler.runtime.continuations().awaitIdle();
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

    static TerminalAction claimTerminal(AbstractRequestScheduler scheduler, RequestContext requestContext,
            TerminalOutcome outcome, Response response, boolean publish) {
        return scheduler.claimFinalizationLocked(requestContext,
                requestContext.decideFinalizationLocked(null, outcome, response, publish));
    }

    private RequestProtocolTestSupport() {
    }

    static RequestContext context(FlexlbConfig config, long requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(16L);
        SchedulingTestConfig.configureRequiredValues(config);
        RequestContext context = new RequestContext(config);
        context.setRequest(request);
        context.setGenerateInputPb(com.google.protobuf.ByteString.copyFromUtf8("test-input"));
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(1)));
        return context;
    }

    static void bind(AbstractRequestScheduler lifecycle, Registered registered) {
        try (AdmissionHandle admission = lifecycle.claimAdmissionHandle(registered.item().requestId(), registered.future()); var admissionCompletion1 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertTrue((lifecycle.commitRoute(registered.item(), RequestProtocolTestSupport.publication(() -> true)) == org.flexlb.balance.PlacementResult.Status.SUCCESS));
        }
    }

    static void bindRoute(AbstractRequestScheduler lifecycle, Registered registered) {
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
            registry.setDeliveryPrediction(claim, emptyWork(), 30_000L);
        }
        return claim;
    }

    static DeliveryClaim claimBatch(AbstractRequestScheduler registry, RequestRoute item, long batchId, BooleanSupplier handoff) {
        DeliveryClaim claim = claimBatchWithoutPrediction(registry, item, batchId, handoff);
        if (claim != null) {
            registry.setDeliveryPrediction(claim, emptyWork(), 30_000L);
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
    static void startRouteDelivery(AbstractRequestScheduler scheduler, RequestContext requestContext) {
        assertNotNull(scheduler.claimDelivery(requestContext.activeRoute(), DeliveryClaimKind.ROUTE_DECISION, 0L, RequestProtocolTestSupport.handoff(() -> true)));
    }

    static void startBatchDelivery(AbstractRequestScheduler scheduler, RequestContext requestContext, long batchId) {
        assertNotNull(scheduler.claimDelivery(requestContext.activeRoute(), DeliveryClaimKind.BATCH_ENQUEUE, batchId, RequestProtocolTestSupport.handoff(() -> true)));
    }

    static void markAcknowledged(RequestContext requestContext) {
        assertTrue(Thread.holdsLock(requestContext));
        org.springframework.test.util.ReflectionTestUtils.setField(requestContext, "deliveryAcknowledged", true);
        org.springframework.test.util.ReflectionTestUtils.setField(requestContext, "stage", RequestContext.RequestStage.RESULT_PENDING);
    }

    static Runnable acknowledge(AbstractRequestScheduler scheduler, RequestContext requestContext) {
        return scheduler.acknowledgeDeliveryLocked(requestContext, null);
    }

    static boolean prepareMember(AbstractRequestScheduler registry, RequestRoute item) {
        synchronized (item.ctx()) {
            return registry.ownsPreparedDeliveryLocked(item.ctx(), item);
        }
    }

    static void awaitGlobalCapacityWaiters(RequestScheduler scheduler, int expected) throws InterruptedException {
        var queue = (QueuedRequestScheduler) scheduler;
        var lock = (ReentrantLock) ReflectionTestUtils.getField(queue, "lock");
        var waiting = (PlacementWaitQueue) ReflectionTestUtils.getField(queue, "waitingRequests");
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
        var waitQueue = (PlacementWaitQueue) ReflectionTestUtils.getField(coordinator, "waitingRequests");
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
}
