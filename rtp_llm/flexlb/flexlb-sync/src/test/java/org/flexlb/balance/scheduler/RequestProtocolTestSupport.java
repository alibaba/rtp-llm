package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;
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

import static org.flexlb.balance.scheduler.ContextResult.Cancellation;
import static org.flexlb.balance.scheduler.ContextResult.End;
import static org.flexlb.balance.scheduler.ContextResult.EndEvidence;
import static org.flexlb.balance.scheduler.ContextResult.EndKind;
import static org.flexlb.balance.scheduler.ContextResult.Prediction;
import static org.flexlb.balance.scheduler.ContextResult.RouteDelivered;
import static org.flexlb.balance.scheduler.ContextResult.SendState;
import static org.flexlb.balance.scheduler.ContextResult.TransportResult;
import static org.flexlb.balance.scheduler.ContextResult.WorkerRequestEvent;
import static org.flexlb.balance.scheduler.ContextResult.WorkerRequestEventType;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Shared lifecycle primitives for scheduler contract tests.
 */
final class RequestProtocolTestSupport {
    private static final Map<QueuedRequestScheduler.PlacementPlan, RouteExecution> observedPlans =
            java.util.Collections.synchronizedMap(new java.util.WeakHashMap<>());

    static QueuedRequestScheduler.PlacementPlan plan(
            RequestContext context, RouteExecution routing) {
        QueuedRequestScheduler.PlacementPlan plan;
        try {
            var scheduler = org.mockito.Mockito.mock(QueuedRequestScheduler.class);
            var owner = context.data().getRequestScheduler();
            ReflectionTestUtils.setField(
                    scheduler,
                    "tasks",
                    owner == null ? org.mockito.Mockito.mock(RequestTaskPool.class) : owner.tasks);
            ReflectionTestUtils.setField(
                    scheduler, "closed", new java.util.concurrent.atomic.AtomicBoolean());
            var constructor =
                    QueuedRequestScheduler.PlacementPlan.class.getDeclaredConstructor(
                            QueuedRequestScheduler.class, GlobalQueueEntry.class, long.class);
            constructor.setAccessible(true);
            plan = constructor.newInstance(scheduler, new GlobalQueueEntry(context, null), 0L);
        } catch (ReflectiveOperationException failure) {
            throw new AssertionError(failure);
        }
        ReflectionTestUtils.setField(plan, "result", PlacementResult.success(routing));
        ReflectionTestUtils.setField(plan, "handle", admissionHandle(context));
        return plan;
    }

    static AbstractRequestScheduler placementScheduler(PlacementOperation operation) {
        var data = ReflectionTestUtils.getField(operation.owner(), "data");
        return (AbstractRequestScheduler) ReflectionTestUtils.getField(data, "requestScheduler");
    }

    static PlacementOperation admissionHandle(RequestContext context) {
        var scheduler = context.data().getRequestScheduler();
        if (scheduler != null
                && scheduler.requests != null
                && scheduler.requests.isCurrent(context)
                && context.snapshot().placement() != null) {
            return context.snapshot().placement().operation();
        }
        if (scheduler == null) {
            scheduler = schedulerMock();
            SchedulerTestSupport.bindOwner(context, scheduler);
        }
        if (context.data().getFuture() == null) {
            context.data().setFuture(new CompletableFuture<>());
        }
        context.register(
                context.data().getRequestId(), System.currentTimeMillis(), Long.MAX_VALUE, 60_000L);
        if (context.snapshot().placement() == null) context.beginPlacement(null);
        var operation = context.snapshot().placement().operation();
        org.mockito.Mockito.doAnswer(
                        call -> {
                            if (context.beginPlacementCompletion(operation)) {
                                context.completePlacement(
                                        operation, call.getArgument(1), System.currentTimeMillis());
                            }
                            return null;
                        })
                .when(scheduler)
                .completePlacement(
                        org.mockito.ArgumentMatchers.same(operation),
                        org.mockito.ArgumentMatchers.any(),
                        org.mockito.ArgumentMatchers.nullable(Response.class));
        return operation;
    }

    static RequestContext planContext(QueuedRequestScheduler.PlacementPlan plan) {
        planRouting(plan);
        return ((GlobalQueueEntry) ReflectionTestUtils.getField(plan, "entry")).context;
    }

    static RouteExecution planRouting(QueuedRequestScheduler.PlacementPlan plan) {
        PlacementResult<?, ?> result =
                (PlacementResult<?, ?>) ReflectionTestUtils.getField(plan, "result");
        if (result != null && result.value() != null) {
            observedPlans.put(plan, (RouteExecution) result.value());
        }
        return observedPlans.get(plan);
    }

    private static final java.util.Map<RouteResult, RouteExecution> selectedExecutions =
            java.util.Collections.synchronizedMap(new java.util.WeakHashMap<>());

    static RouteResult selectionOf(RouteExecution route) {
        synchronized (selectedExecutions) {
            return selectedExecutions.entrySet().stream()
                    .filter(e -> e.getValue() == route)
                    .findFirst()
                    .orElseThrow()
                    .getKey();
        }
    }

    static RouteExecution mockSelectedRoute() {
        var route = org.mockito.Mockito.mock(RouteExecution.class);
        org.mockito.Mockito.when(route.validateEndpoints(org.mockito.ArgumentMatchers.any()))
                .thenAnswer(call -> call.getArgument(0));
        var result = org.mockito.Mockito.mock(RouteResult.class);
        org.mockito.Mockito.when(route.result()).thenReturn(result);
        selectedExecutions.put(result, route);
        org.mockito.Mockito.when(route.prefillEndpoint())
                .thenReturn(org.mockito.Mockito.mock(PrefillEndpoint.class));
        var reservation =
                new java.util.concurrent.atomic.AtomicReference<
                        DecodeResources.RequestReservation>();
        org.mockito.Mockito.when(route.decodeReservation()).thenAnswer(call -> reservation.get());
        org.mockito.Mockito.doAnswer(
                        call -> {
                            reservation.set(call.getArgument(0));
                            return route;
                        })
                .when(route)
                .attachReservation(org.mockito.ArgumentMatchers.any());
        return route;
    }

    static boolean recordPrefillReservation(RouteExecution route) {
        var reservation = org.mockito.Mockito.mock(PrefillEndpoint.RequestRecord.class);
        org.mockito.Mockito.when(reservation.belongsTo(route)).thenReturn(true);
        route.attachPrefillReservation(reservation);
        if (org.mockito.Mockito.mockingDetails(route).isMock()) {
            org.mockito.Mockito.when(route.prefillReservation()).thenReturn(reservation);
        }
        return true;
    }

    static PlacementResult<RouteExecution, PlacementKey> published(
            RouteExecution selected, RouteExecution route) {
        recordPrefillReservation(selected);
        return PlacementResult.success(route);
    }

    static boolean adopt(
            QueuedRequestScheduler scheduler,
            QueuedRequestScheduler.PlacementPlan plan,
            org.flexlb.balance.eviction.DecodePreemption.PreemptionResult result) {
        var endpoint = planRouting(plan).decodeEndpoint();
        boolean accepted = false;
        Throwable failure = null;
        try {
            accepted = plan.accept(endpoint, result.reservation());
            return accepted;
        } catch (RuntimeException | Error failed) {
            failure = failed;
            throw failed;
        } finally {
            if (!accepted && endpoint != null) {
                try {
                    endpoint.release(
                                    result.reservation(),
                                    DecodeResources.ReleaseReason.LOCAL_ROLLBACK)
                            .status();
                } catch (RuntimeException | Error failed) {
                    if (failure == null) {
                        throw failed;
                    }
                    failure.addSuppressed(failed);
                }
            }
        }
    }

    static DecodeEndpoint decodeEndpoint() {
        DecodeEndpoint endpoint = org.mockito.Mockito.mock(DecodeEndpoint.class);
        org.mockito.Mockito.when(endpoint.acceptsNewRequests())
                .thenAnswer(
                        call ->
                                !endpoint.isRetired()
                                        && (endpoint.getStatus() == null
                                                || endpoint.getStatus()
                                                        .isAvailableForScheduling()));
        org.mockito.Mockito.lenient()
                .when(endpoint.ownership(org.mockito.ArgumentMatchers.any()))
                .thenReturn(DecodeResources.Ownership.NONE);
        org.mockito.Mockito.lenient()
                .when(endpoint.markQueued(org.mockito.ArgumentMatchers.any()))
                .thenReturn(
                        DecodeResources.ResourceChange.of(
                                DecodeResources.MutationStatus.STALE, false));
        org.mockito.Mockito.lenient()
                .when(
                        endpoint.updatePreemption(
                                org.mockito.ArgumentMatchers.anyLong(),
                                org.mockito.ArgumentMatchers.any()))
                .thenReturn(
                        DecodeResources.ResourceChange.of(
                                DecodeResources.MutationStatus.STALE, false));
        org.mockito.Mockito.lenient()
                .when(
                        endpoint.release(
                                org.mockito.ArgumentMatchers.any(),
                                org.mockito.ArgumentMatchers.any()))
                .thenReturn(
                        org.flexlb.balance.endpoint.DecodeResources.ResourceChange.of(
                                DecodeResources.ReservationReleaseResult.RELEASED, false));
        return endpoint;
    }

    interface AdmissionCompletion extends AutoCloseable {
        @Override
        void close();
    }

    static AdmissionCompletion finishOnExit(PlacementOperation handle) {
        return () -> {
            if (handle != null) {
                AbstractRequestScheduler.finishPlacement(handle);
            }
        };
    }

    interface Publication {
        void publish();

        boolean published();
    }

    static PlacementResult.Status commitRoute(
            AbstractRequestScheduler lifecycle, RouteExecution route, Publication action) {
        var admission = lifecycle.beginRoute(route);
        if (admission == null) return PlacementResult.Status.CLOSED;
        Throwable failure = org.flexlb.util.Failures.run(null, action::publish);
        boolean published = action.published();
        boolean recorded =
                AbstractRequestScheduler.recordRoute(admission, route, published, failure);
        return !recorded
                ? PlacementResult.Status.CLOSED
                : published ? PlacementResult.Status.SUCCESS : PlacementResult.Status.BLOCKED;
    }

    static RequestProtocolTestSupport.Publication publication(
            java.util.function.BooleanSupplier action) {
        return new RequestProtocolTestSupport.Publication() {
            private boolean published;

            @Override
            public void publish() {
                published = action.getAsBoolean();
            }

            @Override
            public boolean published() {
                return published;
            }
        };
    }

    static boolean publish(RequestProtocolTestSupport.Publication action) {
        action.publish();
        return action.published();
    }

    static org.flexlb.balance.endpoint.DecodeEndpoint.EngineDispatchPermit handoff(
            java.util.function.BooleanSupplier action) {
        var permit =
                org.mockito.Mockito.mock(
                        org.flexlb.balance.endpoint.DecodeEndpoint.EngineDispatchPermit.class);
        org.mockito.Mockito.when(
                        permit.belongsTo(
                                org.mockito.ArgumentMatchers.any(),
                                org.mockito.ArgumentMatchers.any()))
                .thenReturn(true);
        org.mockito.Mockito.when(permit.dispatch())
                .thenAnswer(
                        unused ->
                                DecodeResources.ResourceChange.of(
                                        action.getAsBoolean()
                                                ? DecodeResources.EngineDispatchPermitTransferStatus
                                                        .TRANSFERRED
                                                : DecodeResources.EngineDispatchPermitTransferStatus
                                                        .OWNERSHIP_LOST,
                                        false));
        return permit;
    }

    static boolean closeAdmissionAndAwaitMutations(AbstractRequestScheduler scheduler) {
        if (!SchedulerTestSupport.repository(scheduler).closeRegistration()) {
            return false;
        }
        scheduler.stopAdmissionsAndAwaitCompletion();
        return true;
    }

    static void expireInactiveRequest(
            AbstractRequestScheduler scheduler, RequestContext context, long nowMs) {
        if (context == null) {
            return;
        }
        var deadline = context.snapshot().deadlines().inactivity();
        if (deadline != null) {
            scheduler.onDeadline(deadline, nowMs);
        }
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(scheduler));
    }

    static PlacementOperation beginAdmission(
            AbstractRequestScheduler scheduler, RequestContext context) {
        return SchedulerTestSupport.claim(
                scheduler, context.data().getRequestId(), context.data().getFuture());
    }

    @SuppressWarnings("unchecked")
    static <T> T field(RequestContext context, String name) {
        var state = context.snapshot();
        return (T)
                switch (name) {
                    case "requestDeadline" -> state.deadlines().request();
                    case "decisionDeadline" -> state.deadlines().visibility();
                    case "inactivityDeadline" -> state.deadlines().inactivity();
                    case "decisionVisibility" -> state.engine();
                    case "lastWorkerStatusAtMs" -> state.lastWorkerAtMs();
                    case "cancellationReason" ->
                            state.cancellation() == null ? null : state.cancellation().reason();
                    default -> ReflectionTestUtils.getField(context, name);
                };
    }

    /**
     * Seed cancellation in state-only fixtures without exposing an internal production operation.
     */
    static boolean recordCancellation(
            AbstractRequestScheduler scheduler,
            RequestContext context,
            CancelReason reason,
            String detail) {
        return context.requestCancel(
                new Cancellation(0L, reason, false, System.currentTimeMillis(), null, detail));
    }

    /**
     * Inspect a private decision in state-only fixtures without widening the production API.
     */
    static <T> T inspect(
            AbstractRequestScheduler scheduler,
            RequestContext requestContext,
            String decision,
            Object... arguments) {
        synchronized (requestContext) {
            if (decision.equals("inactivityExpiresAtMsLocked")) {
                return (T) Long.valueOf(requestContext.snapshot().inactivityExpiresAtMs());
            }
            boolean requestDecision =
                    java.util.Arrays.stream(RequestContext.class.getDeclaredMethods())
                            .anyMatch(method -> method.getName().equals(decision));
            return requestDecision
                    ? ReflectionTestUtils.invokeMethod(requestContext, decision, arguments)
                    : ReflectionTestUtils.invokeMethod(
                            scheduler, decision, prepend(requestContext, arguments));
        }
    }

    static AbstractRequestScheduler publication(AbstractRequestScheduler requests) {
        return requests;
    }

    /**
     * Rewire an already-bound owner only for protocol/fault-injection fixtures.
     */
    static AbstractRequestScheduler injectPlacementDependencies(
            AbstractRequestScheduler owner,
            org.flexlb.config.ConfigService service,
            RequestRouteSelector router,
            org.flexlb.service.monitor.DeliveryMetricsReporter reporter,
            org.flexlb.balance.eviction.DecodePreemption eviction,
            PlacementAvailability availability) {
        org.mockito.Mockito.doAnswer(
                        call -> {
                            PlacementResult<RouteResult, PlacementKey> selected =
                                    call.getArgument(1);
                            if (selected.status() != PlacementResult.Status.SUCCESS)
                                return selected.asFailure();
                            var execution = selectedExecutions.get(selected.value());
                            if (execution != null && execution.ctx() == null)
                                org.mockito.Mockito.doReturn(call.getArgument(0))
                                        .when(execution)
                                        .ctx();
                            return execution == null
                                    ? RouteExecution.fromSelection(call.getArgument(0), selected)
                                    : PlacementResult.success(execution);
                        })
                .when(owner)
                .prepareSelectedRoute(
                        org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any());
        var config = service.loadBalanceConfig();
        ReflectionTestUtils.setField(
                org.flexlb.balance.scheduler.SchedulerTestServices.scheduler(owner),
                "router",
                router);
        if (org.flexlb.balance.scheduler.SchedulerTestServices.scheduler(owner)
                instanceof QueuedRequestScheduler queue) {
            ReflectionTestUtils.setField(queue, "decodePreemption", eviction);
            ReflectionTestUtils.setField(owner.dispatcher, "telemetry", reporter);
            ReflectionTestUtils.setField(
                    queue,
                    "plannerCount",
                    Math.max(2, config.getInternalRuntime().getQueuePlannerThreads()));
            ReflectionTestUtils.setField(
                    queue,
                    "queueSettings",
                    QueueExecutionSettings.capture(
                            config, org.flexlb.balance.planner.BuiltinGroupPlanner.from(config)));
            var placementQueue = new OrderedRequestQueue(config.isPriorityOrdering());
            ReflectionTestUtils.setField(queue, "orderedQueue", placementQueue);
            ReflectionTestUtils.setField(
                    queue,
                    "scanBudgetMultiplier",
                    config.queueScheduler().getScanBudgetMultiplier());
            var oldAvailability =
                    (PlacementAvailability) ReflectionTestUtils.getField(queue, "availability");
            var listener =
                    (PlacementAvailability.Listener)
                            ReflectionTestUtils.getField(queue, "availabilityListener");
            oldAvailability.removeListener(listener);
            ReflectionTestUtils.setField(queue, "availability", availability);
            ReflectionTestUtils.setField(
                    queue,
                    "waitingRequests",
                    new PlacementCapacityWaitQueue(placementQueue.order(), availability));
            availability.addListener(listener);
            org.mockito.Mockito.doCallRealMethod()
                    .when(queue)
                    .removeRequest(org.mockito.ArgumentMatchers.any());
            SchedulerTestSupport.startScheduling(queue);
        }
        var scheduler = (AbstractRequestScheduler) SchedulerTestServices.scheduler(owner);
        scheduler.start();
        return scheduler;
    }

    static void close(AbstractRequestScheduler lifecycle) {
        close(org.flexlb.balance.scheduler.SchedulerTestServices.scheduler(lifecycle));
    }

    static void close(RequestScheduler scheduler) {
        SchedulerTestSupport.runtime(scheduler).requests().stopAccepting();
        if (scheduler instanceof QueuedRequestScheduler queue) {
            queue.close();
        }
    }

    static int queuedCount(RequestScheduler scheduler) {
        if (!(scheduler instanceof QueuedRequestScheduler queue)) {
            return 0;
        }
        var lock = (ReentrantLock) ReflectionTestUtils.getField(queue, "lock");
        lock.lock();
        try {
            var ordered = (OrderedRequestQueue) ReflectionTestUtils.getField(queue, "orderedQueue");
            return ordered.size();
        } finally {
            lock.unlock();
        }
    }

    static CompletableFuture<Response> submitForManualPlacement(
            RequestScheduler owner, RequestContext context) {
        return submitForManualPlacement(SchedulerTestSupport.lifecycle(owner), context);
    }

    static CompletableFuture<Response> submitForManualPlacement(
            AbstractRequestScheduler owner, RequestContext context) {
        // Protocol fixtures provide the initial route themselves. Later retries use the policy.
        if (org.mockito.Mockito.mockingDetails(owner).isMock()) {
            org.mockito.Mockito.doNothing().doCallRealMethod().when(owner).onRequest(context);
        }
        return owner.submit(context, () -> {});
    }

    /**
     * Isolates queue ordering from the common request protocol.
     */
    static AbstractRequestScheduler schedulerMock() {
        var config = SchedulingTestConfig.batchConfig();
        var service = org.mockito.Mockito.mock(org.flexlb.config.ConfigService.class);
        org.mockito.Mockito.when(service.loadBalanceConfig()).thenReturn(config);
        var scheduler =
                SchedulerTestSupport.create(
                        service,
                        org.mockito.Mockito.mock(
                                org.flexlb.service.monitor.DeliveryMetricsReporter.class),
                        org.mockito.Mockito.mock(
                                org.flexlb.service.monitor.RequestSchedulerReporter.class),
                        org.mockito.Mockito.mock(org.flexlb.service.TheoryCacheHitReporter.class));
        org.mockito.Mockito.doReturn(null)
                .when(scheduler)
                .submit(org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any());
        return scheduler;
    }

    static AbstractRequestScheduler initialize(
            ResponseCompletionExecutor publisher,
            RequestContext context,
            ExpirationTimer deadlines) {
        var config = org.mockito.Mockito.mock(org.flexlb.config.ConfigService.class);
        org.mockito.Mockito.when(config.loadBalanceConfig()).thenReturn(context.data().getConfig());
        var owner =
                SchedulerTestSupport.create(
                        config,
                        org.mockito.Mockito.mock(
                                org.flexlb.service.monitor.DeliveryMetricsReporter.class),
                        org.mockito.Mockito.mock(
                                org.flexlb.service.monitor.RequestSchedulerReporter.class),
                        org.mockito.Mockito.mock(org.flexlb.service.TheoryCacheHitReporter.class));
        org.flexlb.balance.scheduler.SchedulerTestServices.setDependency(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(owner),
                "responseCompletions",
                publisher);
        org.flexlb.balance.scheduler.SchedulerTestServices.setDependency(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(owner),
                "timer",
                deadlines);
        submitForManualPlacement(owner, context);
        return owner;
    }

    static void applyPrefillStatus(
            AbstractRequestScheduler scheduler,
            PrefillEndpoint source,
            RoleType role,
            PrefillEndpoint.PrefillRequestStatus status) {
        applyPrefillStatus(
                scheduler,
                scheduler.requests.findActive(status.route().requestId()),
                source,
                role,
                status);
    }

    static void applyPrefillStatus(
            AbstractRequestScheduler scheduler,
            RequestContext context,
            PrefillEndpoint source,
            RoleType role,
            PrefillEndpoint.PrefillRequestStatus status) {
        if (context == null || status.route().prefillEndpoint() != source) {
            return;
        }
        var kind =
                switch (status.kind()) {
                    case ACTIVE -> WorkerRequestEventType.ACTIVE;
                    case COMPLETED -> WorkerRequestEventType.COMPLETED;
                    case FAILED -> WorkerRequestEventType.FAILED;
                    case PRIORITY_CANCELED -> WorkerRequestEventType.PRIORITY_CANCELLED;
                };
        scheduler.onWorkerRequestEvent(
                context,
                new WorkerRequestEvent(
                        status.route(),
                        status.route().decodeReservation(),
                        role,
                        kind,
                        status.errorCode(),
                        null,
                        System.currentTimeMillis()));
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(scheduler));
    }

    static void applyDecodeStatus(
            AbstractRequestScheduler scheduler,
            DecodeEndpoint source,
            DecodeResources.DecodeRequestStatus status) {
        applyDecodeStatus(
                scheduler,
                scheduler.requests.findActive(status.reservation().requestId()),
                source,
                status);
    }

    static void applyDecodeStatus(
            AbstractRequestScheduler scheduler,
            RequestContext context,
            DecodeEndpoint source,
            DecodeResources.DecodeRequestStatus status) {
        if (context == null) {
            return;
        }
        var route = context.snapshot().route();
        if (route == null || route.decodeEndpoint() != source) {
            return;
        }
        var kind =
                status.kind() == DecodeResources.DecodeRequestStatus.Kind.TERMINAL
                        ? status.errorCode() == 0L
                                ? WorkerRequestEventType.COMPLETED
                                : WorkerRequestEventType.FAILED
                        : status.allocationObserved()
                                ? WorkerRequestEventType.ALLOCATED
                                : WorkerRequestEventType.ACTIVE;
        scheduler.onWorkerRequestEvent(
                context,
                new WorkerRequestEvent(
                        route,
                        status.reservation(),
                        RoleType.DECODE,
                        kind,
                        status.errorCode(),
                        null,
                        System.currentTimeMillis()));
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(scheduler));
    }

    static void expireInactivity(
            AbstractRequestScheduler scheduler,
            RequestContext context,
            ExpirationTimer.Token exact,
            long nowMs) {
        scheduler.onDeadline(exact, nowMs);
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.scheduler.SchedulerTestSupport.runtime(scheduler));
    }

    private static void run(Runnable continuation) {
        if (continuation != null) {
            continuation.run();
        }
    }

    private static Object[] prepend(RequestContext context, Object[] arguments) {
        Object[] all = new Object[arguments.length + 1];
        all[0] = context;
        System.arraycopy(arguments, 0, all, 1, arguments.length);
        return all;
    }

    static End endFixture(RequestContext context, EndKind kind, String detail) {
        return context.beginClose(
                        EndEvidence.failure(
                                context.snapshot().route(),
                                kind,
                                StrategyErrorType.DISPATCH_FAILED,
                                detail),
                        System.currentTimeMillis())
                .outcome();
    }

    private RequestProtocolTestSupport() {}

    static void closing(RouteExecution route) {
        var ctx = route.ctx();
        long now = System.currentTimeMillis();
        ctx.register(route.requestId(), now, now + 60_000L, 60_000L);
        var admission = ctx.beginPlacement(null);
        org.junit.jupiter.api.Assertions.assertNotNull(admission);
        org.junit.jupiter.api.Assertions.assertTrue(
                ctx.bindPlacement(admission, route, ContextResult.PublicationResult.PREPARED));
        org.junit.jupiter.api.Assertions.assertTrue(
                ctx.bindPlacement(admission, route, ContextResult.PublicationResult.PUBLISHED));
        ctx.completePlacement(admission, ContextResult.PublicationResult.PUBLISHED, now);
        ctx.beginClose(
                ContextResult.EndEvidence.failure(
                        route,
                        ContextResult.EndKind.DELIVERY_FAILURE,
                        org.flexlb.dao.loadbalance.StrategyErrorType.DISPATCH_FAILED,
                        "test failure"),
                now);
    }

    static RequestContext context(FlexlbConfig config, long requestId) {
        Request request =
                Request.builder().requestId(String.valueOf(requestId)).seqLen(16L).build();
        SchedulingTestConfig.configureRequiredValues(config);
        RequestContext context = new RequestContext(config);
        context.data().setRequest(request);
        context.data()
                .setGenerateInputPb(com.google.protobuf.ByteString.copyFromUtf8("test-input"));
        context.data()
                .setSchedulingMetadata(
                        SchedulingMetadata.explicit(
                                50, System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(1)));
        return context;
    }

    static void bind(AbstractRequestScheduler lifecycle, Registered registered) {
        try (PlacementOperation admission =
                        SchedulerTestSupport.claim(
                                lifecycle, registered.item().requestId(), registered.future());
                var admissionCompletion1 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertTrue(
                    (RequestProtocolTestSupport.commitRoute(
                                    lifecycle,
                                    registered.item(),
                                    RequestProtocolTestSupport.publication(() -> true))
                            == org.flexlb.balance.PlacementResult.Status.SUCCESS));
        }
    }

    static void beginRoutePublication(AbstractRequestScheduler lifecycle, Registered registered) {
        assertEquals(PlacementResult.Status.SUCCESS, commitRoute(lifecycle, registered));
    }

    static PlacementResult.Status commitRoute(
            AbstractRequestScheduler lifecycle, Registered registered) {
        try (PlacementOperation admission =
                        SchedulerTestSupport.claim(
                                lifecycle, registered.item().requestId(), registered.future());
                var admissionCompletion2 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            return RequestProtocolTestSupport.commitRoute(
                    lifecycle,
                    registered.item(),
                    RequestProtocolTestSupport.publication(() -> true));
        }
    }

    static Delivery.Member claimRoute(
            AbstractRequestScheduler registry, RouteExecution item, BooleanSupplier handoff) {
        Delivery.Member claim = claimRouteWithoutPrediction(registry, item, handoff);
        if (claim != null) {
            registry.onDeliveryResult(
                    new Prediction(
                            claim,
                            System.currentTimeMillis() + 30_000L,
                            System.currentTimeMillis()));
        }
        return claim;
    }

    static Delivery.Member claimBatch(
            AbstractRequestScheduler registry,
            RouteExecution item,
            long batchId,
            BooleanSupplier handoff) {
        Delivery.Member claim = claimBatchWithoutPrediction(registry, item, batchId, handoff);
        if (claim != null) {
            registry.onDeliveryResult(
                    new Prediction(
                            claim,
                            System.currentTimeMillis() + 30_000L,
                            System.currentTimeMillis()));
        }
        return claim;
    }

    static Delivery.Member claimRouteWithoutPrediction(
            AbstractRequestScheduler registry, RouteExecution item, BooleanSupplier handoff) {
        return DeliveryDispatcher.commitDelivery(
                registry,
                new org.flexlb.balance.scheduler.Delivery.Member(
                        item, RequestProtocolTestSupport.handoff(handoff)),
                Delivery.SendMode.ROUTE_DECISION,
                0L);
    }

    static Delivery.Member claimBatchWithoutPrediction(
            AbstractRequestScheduler registry,
            RouteExecution item,
            long batchId,
            BooleanSupplier handoff) {
        return DeliveryDispatcher.commitDelivery(
                registry,
                new org.flexlb.balance.scheduler.Delivery.Member(
                        item, RequestProtocolTestSupport.handoff(handoff)),
                Delivery.SendMode.BATCH_ENQUEUE,
                batchId);
    }

    private static WorkSnapshot emptyWork() {
        return new WorkSnapshot(
                System.currentTimeMillis(), java.util.List.of(), java.util.List.of(), 0L);
    }

    // State-only fixtures deliberately seed a phase without executing publication or timers.
    static void startRouteDelivery(AbstractRequestScheduler scheduler, RequestContext context) {
        assertNotNull(
                DeliveryDispatcher.commitDelivery(
                        scheduler,
                        new org.flexlb.balance.scheduler.Delivery.Member(
                                context.snapshot().route(), handoff(() -> true)),
                        Delivery.SendMode.ROUTE_DECISION,
                        0L));
    }

    static void startBatchDelivery(
            AbstractRequestScheduler scheduler, RequestContext context, long batchId) {
        assertNotNull(
                DeliveryDispatcher.commitDelivery(
                        scheduler,
                        new org.flexlb.balance.scheduler.Delivery.Member(
                                context.snapshot().route(), handoff(() -> true)),
                        Delivery.SendMode.BATCH_ENQUEUE,
                        batchId));
    }

    static ExpirationTimer.Token markDecodeAccepted(RequestContext context) {
        var previous = context.snapshot().deadlines().visibility();
        var route = context.snapshot().route();
        org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                context,
                observedRequest ->
                        observedRequest.recordWorkerRequestEvent(
                                new WorkerRequestEvent(
                                        route,
                                        route.decodeReservation(),
                                        RoleType.DECODE,
                                        WorkerRequestEventType.ACTIVE,
                                        0L,
                                        null,
                                        System.currentTimeMillis())));
        return previous;
    }

    static void markAcknowledged(RequestContext context) {
        assertTrue(Thread.holdsLock(context));
        var claim = context.snapshot().delivery().identity();
        if (context.snapshot().delivery().send() == SendState.NOT_STARTED) {
            if (claim.kind() == Delivery.SendMode.BATCH_ENQUEUE) {
                org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                        context,
                        observedRequest ->
                                observedRequest.recordDeliveryResult(
                                        new TransportResult(
                                                claim,
                                                org.flexlb.balance.scheduler.Delivery.Result
                                                        .delivered(),
                                                System.currentTimeMillis())));
            } else {
                org.flexlb.balance.scheduler.RequestChangeAssertions.record(
                        context,
                        observedRequest ->
                                observedRequest.recordDeliveryResult(
                                        new RouteDelivered(
                                                claim,
                                                System.currentTimeMillis() + 30_000L,
                                                System.currentTimeMillis())));
            }
        }
        var selected = context.confirmDelivery(System.currentTimeMillis());
        org.junit.jupiter.api.Assertions.assertTrue(selected);
    }

    static Runnable acknowledge(AbstractRequestScheduler scheduler, RequestContext context) {
        var snapshot = context.snapshot();
        var claim = snapshot.delivery().identity();
        return () -> {
            if (claim.kind() == Delivery.SendMode.BATCH_ENQUEUE) {
                scheduler.onDeliveryResult(
                        new TransportResult(
                                claim,
                                org.flexlb.balance.scheduler.Delivery.Result.delivered(),
                                System.currentTimeMillis()));
            } else {
                scheduler.onDeliveryResult(
                        new RouteDelivered(
                                claim,
                                System.currentTimeMillis() + 30_000L,
                                System.currentTimeMillis()));
            }
        };
    }

    static boolean prepareMember(AbstractRequestScheduler registry, RouteExecution item) {
        var state = item.ctx().snapshot();
        return registry.requests.isCurrent(item.ctx())
                && state.route() == item
                && state.preparedDelivery();
    }

    static void awaitGlobalCapacityWaiters(RequestScheduler scheduler, int expected)
            throws InterruptedException {
        var queue = (QueuedRequestScheduler) scheduler;
        var lock = (ReentrantLock) ReflectionTestUtils.getField(queue, "lock");
        var waiting =
                (PlacementCapacityWaitQueue) ReflectionTestUtils.getField(queue, "waitingRequests");
        var membership = (Map<?, ?>) ReflectionTestUtils.getField(waiting, "membership");
        awaitCondition(
                () -> {
                    lock.lock();
                    try {
                        return membership.size() == expected;
                    } finally {
                        lock.unlock();
                    }
                });
    }

    static void awaitGlobalCapacityWaiters(AbstractRequestScheduler scheduler, int expected)
            throws InterruptedException {
        Object coordinator =
                org.flexlb.balance.scheduler.SchedulerTestServices.scheduler(scheduler);
        var lock = (ReentrantLock) ReflectionTestUtils.getField(coordinator, "lock");
        var waitQueue =
                (PlacementCapacityWaitQueue)
                        ReflectionTestUtils.getField(coordinator, "waitingRequests");
        var waiting = (Map<?, ?>) ReflectionTestUtils.getField(waitQueue, "membership");
        // A route's close callback runs before park. Observe actual wait registration
        // under the coordinator lock, rather than treating that callback as a barrier.
        awaitCondition(
                () -> {
                    lock.lock();
                    try {
                        return waiting.keySet().stream()
                                        .filter(
                                                entry ->
                                                        waitQueue.isWaiting(
                                                                (GlobalQueueEntry) entry))
                                        .count()
                                == expected;
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
            throw new AssertionError("interrupted while awaiting latch", interrupted);
        }
    }

    static void awaitCondition(BooleanSupplier condition) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (System.nanoTime() < deadline) {
            if (condition.getAsBoolean()) {
                return;
            }
            Thread.sleep(1L);
        }
        assertTrue(condition.getAsBoolean(), "condition did not become true");
    }

    record Registered(RouteExecution item, CompletableFuture<Response> future) {}

    static void transport(
            AbstractRequestScheduler scheduler,
            Delivery.Member claim,
            org.flexlb.balance.scheduler.Delivery.Result result) {
        scheduler.onDeliveryResult(new TransportResult(claim, result, System.currentTimeMillis()));
    }

    static void prediction(
            AbstractRequestScheduler scheduler,
            Delivery.Member claim,
            WorkSnapshot work,
            long predictedMs) {
        long now = System.currentTimeMillis();
        long deadline =
                org.flexlb.balance.prediction.PrefillTimePredictor.predictedVisibilityDeadlineMs(
                        work,
                        predictedMs,
                        claim.owner()
                                .data()
                                .getConfig()
                                .getRequestLifecycle()
                                .getDecision()
                                .getLifetime(),
                        now);
        scheduler.onDeliveryResult(new Prediction(claim, deadline, now));
    }

    static void routeDelivered(
            AbstractRequestScheduler scheduler,
            Delivery.Member claim,
            WorkSnapshot work,
            long predictedMs) {
        long now = System.currentTimeMillis();
        long deadline =
                org.flexlb.balance.prediction.PrefillTimePredictor.predictedVisibilityDeadlineMs(
                        work,
                        predictedMs,
                        claim.owner()
                                .data()
                                .getConfig()
                                .getRequestLifecycle()
                                .getDecision()
                                .getLifetime(),
                        now);
        scheduler.onDeliveryResult(new RouteDelivered(claim, deadline, now));
    }

    static boolean preemptionPhase(
            AbstractRequestScheduler scheduler,
            PreemptionRegistration claim,
            org.flexlb.balance.preemption.PreemptionCancelPhase phase) {
        return scheduler.onPreemption(
                claim,
                new AbstractRequestScheduler.PreemptionEvent(phase, false, null, null, null));
    }

    static boolean releasePreemption(
            AbstractRequestScheduler scheduler, PreemptionRegistration claim) {
        return scheduler.onPreemption(
                claim, new AbstractRequestScheduler.PreemptionEvent(null, true, null, null, null));
    }

    static boolean finishPreemption(
            AbstractRequestScheduler scheduler, PreemptionRegistration claim, String detail) {
        var route = claim.owner().snapshot().route();
        return route != null
                && scheduler.onPreemption(
                        claim,
                        new AbstractRequestScheduler.PreemptionEvent(
                                null, false, route.decodeEndpoint(), claim.reservation(), detail));
    }

    static void decodeRetired(
            AbstractRequestScheduler scheduler,
            RequestContext context,
            DecodeEndpoint source,
            DecodeResources.RequestReservation reservation) {
        var route = context.snapshot().route();
        if (route == null || route.decodeEndpoint() != source) {
            return;
        }
        scheduler.onEndpointRetired(
                context,
                new ContextResult.EndpointRetirement(
                        route,
                        reservation,
                        RoleType.DECODE,
                        route.decodeEndpoint(),
                        "Decode endpoint generation retired",
                        System.currentTimeMillis()));
    }

    static PreemptionRegistration activePreemption(RequestContext context) {
        var facts = context.snapshot().preemption();
        return facts == null || facts.finished() ? null : facts.identity();
    }

    static boolean preemptionFinished(PreemptionRegistration claim) {
        var facts = claim.owner().snapshot().preemption();
        var result = resolved(claim);
        return facts != null && facts.identity() == claim && facts.finished()
                || result != null
                        && result.outcome()
                                == org.flexlb.balance.preemption.VictimResolution.Outcome
                                        .REQUEST_END;
    }

    static org.flexlb.balance.preemption.VictimResolution resolved(PreemptionRegistration claim) {
        return claim.result();
    }

    static <T> T awaitResult(java.util.concurrent.CompletableFuture<T> future) {
        try {
            return future.get(5L, TimeUnit.SECONDS);
        } catch (Exception failure) {
            throw new AssertionError("request result did not complete", failure);
        }
    }
}
