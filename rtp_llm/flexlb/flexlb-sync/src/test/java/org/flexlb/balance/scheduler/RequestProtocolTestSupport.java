package org.flexlb.balance.scheduler;

import org.flexlb.balance.RoutingResult;
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
    private RequestProtocolTestSupport() {}

    static SelectedRoute plan(RequestContext context, RouteResult routing) {
        return new SelectedRoute(context, routing, admissionHandle(context));
    }

    static RoutingContext admissionHandle(RequestContext context) {
        var scheduler = context.data().getRequestScheduler();
        if (scheduler != null
                && scheduler.repository != null
                && scheduler.repository.isCurrent(context)
                && context.snapshot().routingContext() != null) {
            return context.snapshot().routingContext();
        }
        if (scheduler == null) {
            scheduler = schedulerMock();
            SchedulerTestSupport.bindOwner(context, scheduler);
        }
        if (context.data().getFuture() == null) {
            context.data().setFuture(new CompletableFuture<>());
        }
        context.register(
                context.data().getRequestId(),
                System.currentTimeMillis(),
                Long.MAX_VALUE,
                60_000L,
                org.flexlb.dao.loadbalance.StrategyErrorType.BATCH_SLO_EXPIRED);
        if (context.snapshot().routingContext() == null) context.beginRouteSelection();
        var operation = context.snapshot().routingContext();
        org.mockito.Mockito.doAnswer(
                        call -> {
                            if (context.beginRoutingCompletion(operation)) {
                                context.completeRouting(
                                        operation, call.getArgument(1), System.currentTimeMillis());
                            }
                            return null;
                        })
                .when(scheduler)
                .completeRouting(
                        org.mockito.ArgumentMatchers.same(operation),
                        org.mockito.ArgumentMatchers.any(),
                        org.mockito.ArgumentMatchers.nullable(Response.class));
        return operation;
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

    static AbstractRequestScheduler placementScheduler(RoutingContext operation) {
        var data = ReflectionTestUtils.getField(operation.requestContext(), "data");
        return (AbstractRequestScheduler) ReflectionTestUtils.getField(data, "requestScheduler");
    }

    static RouteResult planRouting(SelectedRoute plan) {
        return plan.route;
    }

    static RouteResult selectionOf(RouteResult route) {
        return route;
    }

    static RouteResult mockSelectedRoute(RequestContext context) {
        var prefill = org.mockito.Mockito.mock(PrefillEndpoint.class);
        var status = new org.flexlb.dao.loadbalance.ServerStatus();
        status.setRequestId(context.data().getRequestId());
        status.setRole(RoleType.PREFILL);
        status.setSuccess(true);
        return mockSelectedRoute(
                context, java.util.List.of(SelectionTestSupport.metadata(prefill, status, 0L)));
    }

    /**
     * Scheduler control-flow fixture with an explicit mock resource owner, constructed before routing.
     */
    static RouteResult mockSelectedRoute(
            RequestContext context, java.util.List<RoleSelection> roles) {
        var registered = new java.util.ArrayList<RoleSelection>();
        for (var role : roles) {
            if (role instanceof RoleSelection.Prefill prefill && prefill.registration == null) {
                var record = org.mockito.Mockito.mock(PrefillEndpoint.RequestRecord.class);
                registered.add(
                        new RoleSelection.Prefill(
                                (PrefillEndpoint) prefill.endpoint(),
                                prefill.toServerStatus(context.data().getRequestId()),
                                prefill.workMs(),
                                record));
            } else registered.add(role);
        }
        var route =
                org.mockito.Mockito.mock(
                        RouteResult.class,
                        org.mockito.Mockito.withSettings()
                                .useConstructor(context, registered)
                                .defaultAnswer(org.mockito.Mockito.CALLS_REAL_METHODS));
        org.mockito.Mockito.doAnswer(call -> call.getArgument(0))
                .when(route)
                .validateEndpoints(org.mockito.ArgumentMatchers.any());
        return route;
    }

    static RoutingResult<RouteResult, CapacityKey> published(
            RouteResult selected, RouteResult route) {
        recordPrefillReservation(selected);
        var context = selected.ctx();
        var operation = context.snapshot().routingContext();
        context.publishRoute(operation, selected);
        return RoutingResult.success(route);
    }

    static boolean recordPrefillReservation(RouteResult route) {
        org.junit.jupiter.api.Assertions.assertNotNull(route.prefillReservation());
        return true;
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
                .when(endpoint.ownership(org.mockito.ArgumentMatchers.any()))
                .thenReturn(DecodeResources.Ownership.NONE);
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
        org.mockito.Mockito.lenient()
                .when(
                        endpoint.tryRegisterSelection(
                                org.mockito.ArgumentMatchers.any(),
                                org.mockito.ArgumentMatchers.anyString(),
                                org.mockito.ArgumentMatchers.anyLong(),
                                org.mockito.ArgumentMatchers.anyLong(),
                                org.mockito.ArgumentMatchers.anyInt(),
                                org.mockito.ArgumentMatchers.any()))
                .thenAnswer(
                        call -> {
                            return new DecodeResources.Acquisition<>(
                                    DecodeResources.ReserveStatus.ACQUIRED,
                                    new DecodeResources.RequestReservation(
                                            1L, call.getArgument(1), 1L));
                        });
        return endpoint;
    }

    static AdmissionCompletion finishOnExit(RoutingContext handle) {
        return () -> {
            if (handle != null) {
                AbstractRequestScheduler.finishRouting(handle);
            }
        };
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

    static boolean closeRegistrationAndAwaitScheduling(AbstractRequestScheduler scheduler) {
        if (!SchedulerTestSupport.repository(scheduler).closeRegistration()) {
            return false;
        }
        scheduler.close();
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

    static RoutingContext beginAdmission(
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
                new Cancellation(
                        0L,
                        reason,
                        false,
                        System.currentTimeMillis(),
                        reason == CancelReason.DEADLINE_EXCEEDED
                                ? org.flexlb.dao.loadbalance.StrategyErrorType.BATCH_SLO_EXPIRED
                                : org.flexlb.dao.loadbalance.StrategyErrorType.REQUEST_CANCELLED,
                        detail));
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

    static QueuedRequestScheduler publication(AbstractRequestScheduler requests) {
        return (QueuedRequestScheduler) requests;
    }

    /**
     * Rewire an already-bound owner only for protocol/fault-injection fixtures.
     */
    static AbstractRequestScheduler injectPlacementDependencies(
            AbstractRequestScheduler owner,
            org.flexlb.config.ConfigService service,
            RequestRouteSelector router,
            org.flexlb.service.monitor.DeliveryMetricsReporter reporter,
            org.flexlb.balance.preemption.DecodePreemption eviction,
            CapacityNotifier availability) {
        var config = service.loadBalanceConfig();
        ReflectionTestUtils.setField(
                org.flexlb.balance.scheduler.SchedulerTestServices.scheduler(owner),
                "router",
                router);
        if (org.flexlb.balance.scheduler.SchedulerTestServices.scheduler(owner)
                instanceof QueuedRequestScheduler queue) {
            ReflectionTestUtils.setField(queue, "decodePreemption", eviction);
            ReflectionTestUtils.setField(queue, "priorityOrdering", config.isPriorityOrdering());
            org.flexlb.balance.scheduler.DeliveryTestSupport.setField(
                    owner.dispatcher, "reporter", reporter);
            ReflectionTestUtils.setField(
                    queue,
                    "plannerCount",
                    Math.max(2, config.getInternalRuntime().getQueuePlannerThreads()));
            ReflectionTestUtils.setField(
                    queue,
                    "queueSettings",
                    QueueExecutionSettings.capture(
                            config, org.flexlb.balance.planner.BuiltinGroupPlanner.from(config)));
            var schedulingQueue = new OrderedRequestQueue(config.isPriorityOrdering());
            ReflectionTestUtils.setField(queue, "orderedQueue", schedulingQueue);
            ReflectionTestUtils.setField(
                    queue,
                    "scanBudgetMultiplier",
                    config.queueScheduler().getScanBudgetMultiplier());
            var oldAvailability =
                    (CapacityNotifier) ReflectionTestUtils.getField(queue, "capacityNotifier");
            var listener =
                    (CapacityNotifier.Listener)
                            ReflectionTestUtils.getField(queue, "capacityListener");
            oldAvailability.removeListener(listener);
            ReflectionTestUtils.setField(queue, "capacityNotifier", availability);
            ReflectionTestUtils.setField(
                    queue,
                    "waitingRequests",
                    new CapacityWaitQueue(schedulingQueue.order(), availability));
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

    static CompletableFuture<Response> submitForManualPlacement(
            RequestScheduler owner, RequestContext context) {
        return submitForManualPlacement(SchedulerTestSupport.lifecycle(owner), context);
    }

    static CompletableFuture<Response> submitForManualPlacement(
            AbstractRequestScheduler owner, RequestContext context) {
        // Protocol fixtures provide the initial route themselves. Later retries use the policy.
        if (org.mockito.Mockito.mockingDetails(owner).isMock()) {
            org.mockito.Mockito.doNothing().doCallRealMethod().when(owner).scheduleRequest(context);
        }
        return owner.submit(context, () -> {});
    }

    static void applyPrefillStatus(
            AbstractRequestScheduler scheduler,
            PrefillEndpoint source,
            RoleType role,
            PrefillEndpoint.PrefillRequestStatus status) {
        applyPrefillStatus(
                scheduler,
                scheduler.repository.findActive(status.route().requestId()),
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
                scheduler.repository.findActive(status.reservation().requestId()),
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

    static void closing(RouteResult route) {
        var ctx = route.ctx();
        long now = System.currentTimeMillis();
        ctx.register(
                route.requestId(),
                now,
                now + 60_000L,
                60_000L,
                org.flexlb.dao.loadbalance.StrategyErrorType.BATCH_SLO_EXPIRED);
        var admission = ctx.beginRouteSelection();
        org.junit.jupiter.api.Assertions.assertNotNull(admission);
        org.junit.jupiter.api.Assertions.assertTrue(ctx.publishRoute(admission, route));
        ctx.completeRouting(admission, ContextResult.RoutingCompletion.FINISHED, now);
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
        RoutingContext admission =
                SchedulerTestSupport.claim(
                        lifecycle, registered.item().requestId(), registered.future());
        try (var admissionCompletion1 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertTrue(
                    (RequestProtocolTestSupport.commitRoute(
                                    lifecycle,
                                    registered.item(),
                                    RequestProtocolTestSupport.publication(() -> true))
                            == org.flexlb.balance.RoutingResult.Status.SUCCESS));
        }
    }

    static void beginRoutePublication(AbstractRequestScheduler lifecycle, Registered registered) {
        assertEquals(RoutingResult.Status.SUCCESS, commitRoute(lifecycle, registered));
    }

    static RoutingResult.Status commitRoute(
            AbstractRequestScheduler lifecycle, RouteResult route, Publication action) {
        var admission = lifecycle.currentRoutingContext(route);
        if (admission == null) return RoutingResult.Status.CLOSED;
        Throwable failure = org.flexlb.util.Failures.run(null, action::publish);
        boolean published = action.published();
        boolean recorded = published && route.ctx().publishRoute(admission, route);
        if (recorded && route.prefillEndpoint() != null)
            failure =
                    org.flexlb.util.Failures.run(
                            failure,
                            () -> route.prefillEndpoint().signal(PrefillEndpoint.Signal.INPUT));
        if (!recorded)
            failure =
                    org.flexlb.util.Failures.append(
                            failure, AbstractRequestScheduler.finishUnpublishedRoute(route, null));
        org.flexlb.util.Failures.rethrow(failure, "test publication failed");
        return recorded
                ? RoutingResult.Status.SUCCESS
                : route.ctx().canPublishRoute(admission)
                        ? RoutingResult.Status.BLOCKED
                        : RoutingResult.Status.CLOSED;
    }

    static RoutingResult.Status commitRoute(
            AbstractRequestScheduler lifecycle, Registered registered) {
        RoutingContext admission =
                SchedulerTestSupport.claim(
                        lifecycle, registered.item().requestId(), registered.future());
        try (var admissionCompletion2 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            return RequestProtocolTestSupport.commitRoute(
                    lifecycle,
                    registered.item(),
                    RequestProtocolTestSupport.publication(() -> true));
        }
    }

    static Delivery.Member claimRoute(
            AbstractRequestScheduler registry, RouteResult item, BooleanSupplier handoff) {
        Delivery.Member claim = claimRouteWithoutPrediction(registry, item, handoff);
        if (claim != null) {
            registry.recordDeliveryPrediction(
                    new Prediction(
                            claim,
                            System.currentTimeMillis() + 30_000L,
                            System.currentTimeMillis()));
        }
        return claim;
    }

    static Delivery.Member claimRouteWithoutPrediction(
            AbstractRequestScheduler registry, RouteResult item, BooleanSupplier handoff) {
        return org.flexlb.balance.scheduler.DeliveryTestSupport.transport(registry.dispatcher)
                .commit(
                        new org.flexlb.balance.scheduler.Delivery.Member(
                                item, RequestProtocolTestSupport.handoff(handoff)),
                        Delivery.SendMode.ROUTE_DECISION,
                        0L,
                        null);
    }

    static Delivery.Member claimBatch(
            AbstractRequestScheduler registry,
            RouteResult item,
            long batchId,
            BooleanSupplier handoff) {
        Delivery.Member claim = claimBatchWithoutPrediction(registry, item, batchId, handoff);
        if (claim != null) {
            registry.recordDeliveryPrediction(
                    new Prediction(
                            claim,
                            System.currentTimeMillis() + 30_000L,
                            System.currentTimeMillis()));
        }
        return claim;
    }

    static Delivery.Member claimBatchWithoutPrediction(
            AbstractRequestScheduler registry,
            RouteResult item,
            long batchId,
            BooleanSupplier handoff) {
        return org.flexlb.balance.scheduler.DeliveryTestSupport.transport(registry.dispatcher)
                .commit(
                        new org.flexlb.balance.scheduler.Delivery.Member(
                                item, RequestProtocolTestSupport.handoff(handoff)),
                        Delivery.SendMode.BATCH_ENQUEUE,
                        batchId,
                        null);
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

    static boolean prepareMember(AbstractRequestScheduler registry, RouteResult item) {
        var state = item.ctx().snapshot();
        return registry.repository.isCurrent(item.ctx())
                && state.route() == item
                && state.preparedDelivery();
    }

    static void awaitGlobalCapacityWaiters(RequestScheduler scheduler, int expected)
            throws InterruptedException {
        var queue = (QueuedRequestScheduler) scheduler;
        var lock = (ReentrantLock) ReflectionTestUtils.getField(queue, "lock");
        var waiting = (CapacityWaitQueue) ReflectionTestUtils.getField(queue, "waitingRequests");
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

    static void awaitGlobalCapacityWaiters(AbstractRequestScheduler scheduler, int expected)
            throws InterruptedException {
        Object coordinator =
                org.flexlb.balance.scheduler.SchedulerTestServices.scheduler(scheduler);
        var lock = (ReentrantLock) ReflectionTestUtils.getField(coordinator, "lock");
        var waitQueue =
                (CapacityWaitQueue) ReflectionTestUtils.getField(coordinator, "waitingRequests");
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
        scheduler.recordDeliveryPrediction(new Prediction(claim, deadline, now));
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

    static java.util.concurrent.CompletableFuture<PreemptionRegistration.Reply> cancelForPreemption(
            DeliveryCoordinator dispatcher,
            PreemptionRegistration registration,
            org.flexlb.balance.endpoint.DecodeEndpoint endpoint,
            String detail,
            long timeoutMs) {
        return SchedulerTestSupport.preemption(registration.owner().data().getRequestScheduler())
                .cancelRequest(
                        dispatcher,
                        SchedulerTestSupport.runtime(
                                        registration.owner().data().getRequestScheduler())
                                .taskTimer(),
                        registration,
                        endpoint,
                        detail,
                        timeoutMs);
    }

    static boolean preemptionPhase(
            AbstractRequestScheduler scheduler,
            PreemptionRegistration claim,
            org.flexlb.balance.preemption.PreemptionCancelPhase phase) {
        return org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(scheduler)
                .applyEvent(
                        claim,
                        new org.flexlb.balance.preemption.PreemptionCompletion.PreemptionEvent(
                                phase, false, null, null, null));
    }

    static boolean releasePreemption(
            AbstractRequestScheduler scheduler, PreemptionRegistration claim) {
        return org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(scheduler)
                .applyEvent(
                        claim,
                        new org.flexlb.balance.preemption.PreemptionCompletion.PreemptionEvent(
                                null, true, null, null, null));
    }

    static boolean finishPreemption(
            AbstractRequestScheduler scheduler, PreemptionRegistration claim, String detail) {
        var route = claim.owner().snapshot().route();
        return route != null
                && org.flexlb.balance.scheduler.SchedulerTestSupport.preemption(scheduler)
                        .applyEvent(
                                claim,
                                new org.flexlb.balance.preemption.PreemptionCompletion
                                        .PreemptionEvent(
                                        null,
                                        false,
                                        route.decodeEndpoint(),
                                        claim.reservation(),
                                        detail));
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
                                == org.flexlb.balance.preemption.PreemptionRequestResult.Outcome
                                        .REQUEST_END;
    }

    static org.flexlb.balance.preemption.PreemptionRequestResult resolved(
            PreemptionRegistration claim) {
        return claim.result();
    }

    static <T> T awaitResult(java.util.concurrent.CompletableFuture<T> future) {
        try {
            return future.get(5L, TimeUnit.SECONDS);
        } catch (Exception failure) {
            throw new AssertionError("request result did not complete", failure);
        }
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

    private static WorkSnapshot emptyWork() {
        return new WorkSnapshot(
                System.currentTimeMillis(), java.util.List.of(), java.util.List.of(), 0L);
    }

    /**
     * Keeps a selected route with its placement operation and closes any unfinished attempt when the fixture exits.
     */
    static final class SelectedRoute implements AutoCloseable {
        final RequestContext context;
        final RouteResult route;
        private RoutingContext operation;

        SelectedRoute(RequestContext context, RouteResult route, RoutingContext operation) {
            this.context = context;
            this.route = route;
            this.operation = operation;
        }

        @Override
        public void close() {
            var finishing = operation;
            operation = null;
            if (finishing == null) return;
            var failure =
                    AbstractRequestScheduler.finishUnpublishedRoute(
                            context.snapshot().route() == route ? null : route, finishing);
            if (failure != null) placementScheduler(finishing).recordRequestFailure(failure);
        }
    }

    interface AdmissionCompletion extends AutoCloseable {
        @Override
        void close();
    }

    interface Publication {
        void publish();

        boolean published();
    }

    record Registered(RouteResult item, CompletableFuture<Response> future) {}
}
