package org.flexlb.balance.schedule;

import org.flexlb.balance.queue.CapacityWaitQueue;
import org.flexlb.balance.queue.GlobalQueueEntry;
import org.flexlb.balance.queue.OrderedRequestQueue;
import org.flexlb.balance.schedule.RequestContext.SelectionAttempt;
import org.flexlb.balance.strategy.RequestRouteSelector;
import org.flexlb.balance.strategy.RoleSelection;
import org.flexlb.balance.strategy.RoutingResult;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.service.RoleType;
import org.flexlb.worker.endpoint.CapacityNotifier;
import org.flexlb.worker.endpoint.DecodeEndpoint;
import org.flexlb.worker.endpoint.DecodeResources;
import org.flexlb.worker.endpoint.PrefillEndpoint;
import org.flexlb.worker.endpoint.Request;
import org.flexlb.worker.endpoint.RequestEventType;
import org.flexlb.worker.endpoint.Response;
import org.flexlb.worker.endpoint.StrategyErrorType;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.BooleanSupplier;

import static org.flexlb.balance.schedule.RequestContext.EndKind;
import static org.flexlb.balance.schedule.RequestContext.RequestEndEvent;
import static org.flexlb.balance.schedule.RequestContext.RequestEvent;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Shared lifecycle primitives for scheduler contract tests.
 */
final class RequestProtocolTestSupport {
    private RequestProtocolTestSupport() {}

    static void register(
            RequestContext context,
            String requestId,
            long nowMs,
            long expiresAtMs,
            long inactivityTimeoutMs,
            StrategyErrorType timeoutError) {
        if (context.snapshot().requestId() != null) return;
        if (context.getRequest() == null) {
            context.setRequest(
                    org.flexlb.worker.endpoint.Request.builder().requestId(requestId).build());
        }
        context.setSchedulingMetadata(
                org.flexlb.balance.schedule.SchedulingMetadata.explicit(
                        context.getPriority(), expiresAtMs));
        var future = context.getFuture();
        context.register(
                context.getRequestScheduler(),
                future instanceof ResponseCompletionExecutor.RequestFuture response
                        ? response
                        : null,
                nowMs,
                inactivityTimeoutMs,
                timeoutError);
        if (!(future instanceof ResponseCompletionExecutor.RequestFuture))
            context.setFuture(future);
    }

    static SelectedRoute plan(RequestContext context, RequestResult routing) {
        return new SelectedRoute(context, routing, admissionHandle(context));
    }

    static SelectionAttempt admissionHandle(RequestContext context) {
        var scheduler = context.getRequestScheduler();
        if (scheduler != null
                && scheduler.repository != null
                && scheduler.repository.isCurrent(context)
                && context.currentSelectionAttempt() != null) {
            return context.currentSelectionAttempt();
        }
        if (scheduler == null) {
            scheduler = schedulerMock();
            SchedulerTestSupport.bindOwner(context, scheduler);
        }
        if (context.getFuture() == null) {
            context.setFuture(new CompletableFuture<>());
        }
        RequestProtocolTestSupport.register(
                context,
                context.getRequestId(),
                System.currentTimeMillis(),
                Long.MAX_VALUE,
                60_000L,
                org.flexlb.worker.endpoint.StrategyErrorType.SCHEDULING_DEADLINE_EXCEEDED);
        if (context.currentSelectionAttempt() == null) context.beginRouteSelection();
        var operation = context.currentSelectionAttempt();
        org.mockito.Mockito.doAnswer(
                        call -> {
                            if (context.currentSelectionAttempt() != null) {
                                context.finishSelectionWithoutCommit(
                                        operation, call.getArgument(1), System.currentTimeMillis());
                            }
                            return null;
                        })
                .when(scheduler)
                .finishSelectionWithoutCommit(
                        org.mockito.ArgumentMatchers.same(operation),
                        org.mockito.ArgumentMatchers.nullable(Response.class));
        return operation;
    }

    /**
     * Isolates queue ordering from the common request protocol.
     */
    static AbstractRequestScheduler schedulerMock() {
        var config = SchedulingTestConfig.newConfig();
        var service = org.mockito.Mockito.mock(org.flexlb.config.ConfigService.class);
        org.mockito.Mockito.when(service.loadBalanceConfig()).thenReturn(config);
        var scheduler =
                SchedulerTestSupport.create(
                        service,
                        org.mockito.Mockito.mock(
                                org.flexlb.balance.metrics.DeliveryMetricsReporter.class),
                        org.mockito.Mockito.mock(
                                org.flexlb.balance.metrics.RequestSchedulerReporter.class),
                        org.mockito.Mockito.mock(
                                org.flexlb.balance.requestlog.TheoryCacheHitReporter.class));
        org.mockito.Mockito.doReturn(null)
                .when(scheduler)
                .submit(org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.any());
        return scheduler;
    }

    static AbstractRequestScheduler placementScheduler(SelectionAttempt operation) {
        return (AbstractRequestScheduler)
                ReflectionTestUtils.getField(operation.requestContext(), "requestScheduler");
    }

    static RequestResult planRouting(SelectedRoute plan) {
        return plan.route;
    }

    static RequestResult selectionOf(RequestResult route) {
        return route;
    }

    static RequestResult mockSelectedRoute(RequestContext context) {
        var prefill = org.mockito.Mockito.mock(PrefillEndpoint.class);
        var status = new org.flexlb.worker.status.ServerStatus();
        status.setRequestId(context.getRequestId());
        status.setRole(RoleType.PREFILL);
        status.setSuccess(true);
        return mockSelectedRoute(
                context, java.util.List.of(SelectionTestSupport.metadata(prefill, status, 0L)));
    }

    /**
     * Scheduler control-flow fixture with an explicit mock resource owner, constructed before routing.
     */
    static RequestResult mockSelectedRoute(
            RequestContext context, java.util.List<RoleSelection> roles) {
        var registered = new java.util.ArrayList<RoleSelection>();
        for (var role : roles) {
            if (role.role().supportsPrefill() && role.registration() == null) {
                var record = org.mockito.Mockito.mock(PrefillEndpoint.RequestRegistration.class);
                registered.add(
                        RoleSelection.prefill(
                                (PrefillEndpoint) role.endpoint(),
                                role.toServerStatus(context.getRequestId()),
                                role.computeTokens(),
                                record));
            } else registered.add(role);
        }
        var route =
                org.mockito.Mockito.mock(
                        RequestResult.class,
                        org.mockito.Mockito.withSettings()
                                .useConstructor(context, registered)
                                .defaultAnswer(org.mockito.Mockito.CALLS_REAL_METHODS));
        org.mockito.Mockito.doAnswer(call -> call.getArgument(0))
                .when(route)
                .validateEndpoints(org.mockito.ArgumentMatchers.any(RequestResult.class));
        return route;
    }

    static RequestResult published(RequestResult selected, RequestResult route) {
        recordPrefillReservation(selected);
        var context = selected.ctx();
        var operation = context.currentSelectionAttempt();
        var publication = selected;
        long now = System.currentTimeMillis();
        synchronized (context) {
            context.registerSelectedRoute(operation, publication);
            context.tryCommitRoute(
                    operation,
                    publication,
                    new RequestResult(context, RequestContext.ResponseKind.ACK, now),
                    now);
        }
        return publication;
    }

    static boolean recordPrefillReservation(RequestResult route) {
        org.junit.jupiter.api.Assertions.assertNotNull(route.prefillRegistration());
        return true;
    }

    static DecodeEndpoint decodeEndpoint() {
        DecodeEndpoint endpoint = org.mockito.Mockito.mock(DecodeEndpoint.class);
        org.mockito.Mockito.when(endpoint.acceptsNewRequests())
                .thenAnswer(
                        call ->
                                !endpoint.hasStoppedAcceptingRequests()
                                        && (endpoint.getStatus() == null
                                                || endpoint.getStatus()
                                                        .isAvailableForScheduling()));
        org.mockito.Mockito.lenient()
                .when(endpoint.registrationState(org.mockito.ArgumentMatchers.any()))
                .thenReturn(DecodeResources.RegistrationState.NOT_REGISTERED);
        org.mockito.Mockito.lenient()
                .when(endpoint.releaseRequest(org.mockito.ArgumentMatchers.any()))
                .thenReturn(true);
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
                            return new WorkerEndpoint.Registration<>(
                                    WorkerEndpoint.RegistrationStatus.REGISTERED,
                                    new DecodeEndpoint.RequestRegistration(
                                            1L, call.getArgument(1)));
                        });
        return endpoint;
    }

    static AdmissionCompletion finishOnExit(SelectionAttempt handle) {
        return () -> {
            if (handle != null) {
                handle.requestContext()
                        .getRequestScheduler()
                        .finishSelectionWithoutCommit(handle, null);
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
                org.flexlb.balance.schedule.SchedulerTestSupport.runtime(scheduler));
    }

    static SelectionAttempt beginAdmission(
            AbstractRequestScheduler scheduler, RequestContext context) {
        return SchedulerTestSupport.claim(scheduler, context.getRequestId(), context.getFuture());
    }

    @SuppressWarnings("unchecked")
    static <T> T field(RequestContext context, String name) {
        var state = context.snapshot();
        return (T)
                switch (name) {
                    case "requestDeadline" -> state.deadlines().request();
                    case "decisionDeadline" -> state.deadlines().visibility();
                    case "inactivityDeadline" -> state.deadlines().inactivity();
                    case "lastWorkerStatusAtMs" ->
                            ReflectionTestUtils.getField(context, "lastActivityAtMs");
                    case "cancellationReason" ->
                            state.cancellationReason() == null ? null : state.cancellationReason();
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
                reason,
                false,
                System.currentTimeMillis(),
                reason == CancelReason.DEADLINE_EXCEEDED
                        ? org.flexlb.worker.endpoint.StrategyErrorType.SCHEDULING_DEADLINE_EXCEEDED
                        : org.flexlb.worker.endpoint.StrategyErrorType.REQUEST_CANCELLED,
                detail,
                null);
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
            org.flexlb.balance.metrics.DeliveryMetricsReporter reporter,
            org.flexlb.balance.schedule.DecodePreemption eviction,
            CapacityNotifier availability) {
        var config = service.loadBalanceConfig();
        ReflectionTestUtils.setField(
                org.flexlb.balance.schedule.SchedulerTestServices.scheduler(owner),
                "router",
                router);
        if (org.flexlb.balance.schedule.SchedulerTestServices.scheduler(owner)
                instanceof QueuedRequestScheduler queue) {
            ReflectionTestUtils.setField(queue, "decodePreemption", eviction);
            ReflectionTestUtils.setField(queue, "priorityOrdering", config.isPriorityOrdering());
            ReflectionTestUtils.setField(
                    queue,
                    "plannerCount",
                    Math.max(2, config.getInternalRuntime().getQueuePlannerThreads()));

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
        close(org.flexlb.balance.schedule.SchedulerTestServices.scheduler(lifecycle));
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
        org.mockito.Mockito.when(config.loadBalanceConfig()).thenReturn(context.getConfig());
        var owner =
                SchedulerTestSupport.create(
                        config,
                        org.mockito.Mockito.mock(
                                org.flexlb.balance.metrics.DeliveryMetricsReporter.class),
                        org.mockito.Mockito.mock(
                                org.flexlb.balance.metrics.RequestSchedulerReporter.class),
                        org.mockito.Mockito.mock(
                                org.flexlb.balance.requestlog.TheoryCacheHitReporter.class));
        org.flexlb.balance.schedule.SchedulerTestServices.setDependency(
                org.flexlb.balance.schedule.SchedulerTestSupport.runtime(owner),
                "responseCompletions",
                publisher);
        org.flexlb.balance.schedule.SchedulerTestServices.setDependency(
                org.flexlb.balance.schedule.SchedulerTestSupport.runtime(owner),
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
        var kind = status.eventType();
        scheduler.onRequestEvent(
                context,
                new RequestEvent(
                        (RequestResult) status.route(),
                        status.route().decodeRegistration(),
                        role,
                        null,
                        kind,
                        status.errorCode(),
                        null,
                        System.currentTimeMillis()));
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.schedule.SchedulerTestSupport.runtime(scheduler));
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
                                ? RequestEventType.COMPLETED
                                : RequestEventType.FAILED
                        : RequestEventType.ALLOCATED;
        scheduler.onRequestEvent(
                context,
                new RequestEvent(
                        route,
                        status.reservation(),
                        RoleType.DECODE,
                        null,
                        kind,
                        status.errorCode(),
                        null,
                        System.currentTimeMillis()));
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.schedule.SchedulerTestSupport.runtime(scheduler));
    }

    static void expireInactivity(
            AbstractRequestScheduler scheduler,
            RequestContext context,
            ExpirationTimer.Token exact,
            long nowMs) {
        scheduler.onDeadline(exact, nowMs);
        SchedulerTestSupport.awaitRequestWork(
                org.flexlb.balance.schedule.SchedulerTestSupport.runtime(scheduler));
    }

    static RequestResult endFixture(RequestContext context, EndKind kind, String detail) {
        return context.endRequest(
                        RequestEndEvent.failure(
                                context.snapshot().route(),
                                kind,
                                StrategyErrorType.DISPATCH_FAILED,
                                detail),
                        System.currentTimeMillis(),
                        null)
                .requestEndResult();
    }

    static void closing(RequestResult route) {
        var ctx = route.ctx();
        long now = System.currentTimeMillis();
        RequestProtocolTestSupport.register(
                ctx,
                route.requestId(),
                now,
                now + 60_000L,
                60_000L,
                org.flexlb.worker.endpoint.StrategyErrorType.SCHEDULING_DEADLINE_EXCEEDED);
        var admission = ctx.beginRouteSelection();
        org.junit.jupiter.api.Assertions.assertNotNull(admission);
        synchronized (ctx) {
            ctx.registerSelectedRoute(admission, route);
            ctx.tryCommitRoute(
                    admission,
                    route,
                    new RequestResult(ctx, RequestContext.ResponseKind.ACK, now),
                    now);
        }
        ctx.finishSelectionWithoutCommit(admission, null, now);
        ctx.endRequest(
                RequestContext.RequestEndEvent.failure(
                        route,
                        RequestContext.EndKind.FAILURE,
                        org.flexlb.worker.endpoint.StrategyErrorType.DISPATCH_FAILED,
                        "test failure"),
                now,
                null);
    }

    static RequestContext context(FlexlbConfig config, long requestId) {
        Request request =
                Request.builder().requestId(String.valueOf(requestId)).seqLen(16L).build();
        SchedulingTestConfig.configureRequiredValues(config);
        RequestContext context = new RequestContext(config);
        context.setRequest(request);

        context.setSchedulingMetadata(
                SchedulingMetadata.explicit(
                        50, System.currentTimeMillis() + TimeUnit.MINUTES.toMillis(1)));
        return context;
    }

    static void bind(AbstractRequestScheduler lifecycle, Registered registered) {
        SelectionAttempt admission =
                SchedulerTestSupport.claim(
                        lifecycle, registered.item().requestId(), registered.future());
        try (var admissionCompletion1 = RequestProtocolTestSupport.finishOnExit(admission)) {
            assertNotNull(admission);
            assertTrue(
                    (RequestProtocolTestSupport.commitRoute(
                                    lifecycle,
                                    registered.item(),
                                    RequestProtocolTestSupport.publication(() -> true))
                            == org.flexlb.balance.strategy.RoutingResult.Status.SUCCESS));
        }
    }

    static void beginRouteResult(AbstractRequestScheduler lifecycle, Registered registered) {
        assertEquals(RoutingResult.Status.SUCCESS, commitRoute(lifecycle, registered));
    }

    static RoutingResult.Status commitRoute(
            AbstractRequestScheduler lifecycle, RequestResult route, Publication action) {
        var admission = lifecycle.currentSelectionAttempt(route);
        if (admission == null) return RoutingResult.Status.CLOSED;
        Throwable failure = org.flexlb.util.Failures.run(null, action::publish);
        boolean published = action.published();
        boolean recorded = published && route.ctx().canPublishRoute(admission);
        if (recorded) {
            var context = route.ctx();
            long nowMs = System.currentTimeMillis();
            var candidate = new RequestResult(context, RequestContext.ResponseKind.ACK, nowMs);
            try (var completion =
                    new ResponseCompletionExecutor.Registration(
                            SchedulerTestSupport.runtime(lifecycle).responseCompletions(),
                            context,
                            (ResponseCompletionExecutor.RequestFuture) context.getFuture(),
                            () -> context.snapshot().responseForFuture() == candidate)) {
                synchronized (context) {
                    context.registerSelectedRoute(admission, route);
                    recorded = context.tryCommitRoute(admission, route, candidate, nowMs);
                    if (recorded)
                        completion.prepare(
                                ResponseCompletionExecutor.RequestFuture.Completion.RESPONSE,
                                route.successResponse(),
                                null);
                }
                if (recorded) completion.publish(false);
            }
        }
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
        SelectionAttempt admission =
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

    static RequestResult claimRouteWithoutPrediction(
            AbstractRequestScheduler scheduler, RequestResult route, BooleanSupplier handoff) {
        return claimRoute(scheduler, route, handoff);
    }

    static RequestResult claimRoute(
            AbstractRequestScheduler scheduler, RequestResult route, BooleanSupplier handoff) {
        if (route.ctx().snapshot().committedRoute() != null)
            return handoff.getAsBoolean() ? route : null;
        if (!handoff.getAsBoolean()) return null;
        Boolean committed =
                org.springframework.test.util.ReflectionTestUtils.invokeMethod(
                        scheduler,
                        "tryCommitDelivery",
                        route,
                        route.ctx().currentSelectionAttempt());
        return Boolean.TRUE.equals(committed) ? route : null;
    }

    static Runnable acknowledge(AbstractRequestScheduler scheduler, RequestContext context) {
        var claim = context.snapshot().committedRoute();
        return () -> scheduler.processPendingRequestActions(claim.ctx());
    }

    static void routeDelivered(AbstractRequestScheduler scheduler, RequestResult claim) {
        scheduler.processPendingRequestActions(claim.ctx());
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
        Object coordinator = org.flexlb.balance.schedule.SchedulerTestServices.scheduler(scheduler);
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

    static void decodeRetired(
            AbstractRequestScheduler scheduler,
            RequestContext context,
            DecodeEndpoint source,
            DecodeEndpoint.RequestRegistration reservation) {
        var route = context.snapshot().route();
        if (route == null || route.decodeEndpoint() != source) {
            return;
        }
        scheduler.onRequestEvent(
                context,
                new RequestContext.RequestEvent(
                        route,
                        reservation,
                        RoleType.DECODE,
                        route.decodeEndpoint(),
                        org.flexlb.worker.endpoint.RequestEventType.ENDPOINT_CLOSED,
                        0L,
                        "Decode endpoint generation retired",
                        System.currentTimeMillis()));
    }

    static <T> T awaitResult(java.util.concurrent.CompletableFuture<T> future) {
        try {
            return future.get(5L, TimeUnit.SECONDS);
        } catch (Exception failure) {
            throw new AssertionError("request result did not complete", failure);
        }
    }

    static boolean commitSelectedRoute(
            RequestContext context, SelectionAttempt selecting, RequestResult route) {
        synchronized (context) {
            if (!context.canPublishRoute(selecting)) return false;
            context.registerSelectedRoute(selecting, route);
            long nowMs = context.snapshot().updatedAtMs();
            return context.tryCommitRoute(
                    selecting,
                    route,
                    new RequestResult(context, RequestContext.ResponseKind.ACK, nowMs),
                    nowMs);
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

    /**
     * Keeps a selected route with its placement operation and closes any unfinished attempt when
     * the fixture exits.
     */
    static final class SelectedRoute implements AutoCloseable {
        final RequestContext context;
        final RequestResult route;
        private SelectionAttempt operation;

        SelectedRoute(RequestContext context, RequestResult route, SelectionAttempt operation) {
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

    /**
     * Releases the test admission scope after its controlled operation finishes.
     */
    interface AdmissionCompletion extends AutoCloseable {
        @Override
        void close();
    }

    /**
     * Runs a test route publication and retains whether the publication succeeded.
     */
    interface Publication {
        boolean published();

        void publish();
    }

    record Registered(RequestResult item, CompletableFuture<Response> future) {}
}
