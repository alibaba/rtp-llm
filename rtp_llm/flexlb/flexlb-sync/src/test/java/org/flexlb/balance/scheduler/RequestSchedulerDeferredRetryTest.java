package org.flexlb.balance.scheduler;

import org.flexlb.balance.admission.AdmissionFallback;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillGenerationRuntime;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.metric.NoOpFlexMonitor;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Black-box contracts for scheduler-owned placement retry.
 *
 * <p>The router exposes only its existing queue result. These tests therefore
 * do not prescribe a Deferred DTO, wait-key type, or retry implementation.
 * They specify the externally observable ownership and concurrency semantics.
 */
class RequestSchedulerDeferredRetryTest {

    private final List<SchedulerFixture> fixtures = new ArrayList<>();

    @AfterEach
    void tearDown() {
        for (SchedulerFixture fixture : fixtures) {
            fixture.close();
        }
    }

    @Test
    void nanoDeadlineSaturationSupportsNegativeNanoTimeOrigins() {
        assertEquals(-5L, RequestScheduler.saturatingAddNanos(-10L, 5L));
        assertEquals(Long.MAX_VALUE,
                RequestScheduler.saturatingAddNanos(Long.MAX_VALUE - 2L, 5L));
    }

    @Test
    @Timeout(30)
    void deferredSelectionKeepsOriginalFutureAndRetriesFreshRouting()
            throws Exception {
        RetryThenFatalRouter router = new RetryThenFatalRouter();
        SchedulerFixture fixture = fixture(router);

        CompletableFuture<Response> original =
                fixture.scheduler.submit(context(101L));

        Response terminal = original.get(3, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                terminal.getCode(),
                "temporary capacity loss must not publish its 8402 response");
        assertTrue(router.attempts.get() >= 2,
                "the scheduler must make a fresh placement decision after backoff");
        assertSame(original, fixture.lifecycle.requestSlot(101L).future(),
                "retry must retain the exact registered request future");
    }

    @Test
    @Timeout(30)
    void terminalCapacityRejectionDoesNotCreateALaneOrBlockALaterRequest()
            throws Exception {
        StaticRejectingRouter router = new StaticRejectingRouter();
        SchedulerFixture fixture = fixture(router);
        BalanceContext oversized = context(151L);
        oversized.getRequest().setSeqLen(257L);
        BalanceContext small = context(152L);
        small.getRequest().setSeqLen(128L);

        Response first = fixture.scheduler.submit(oversized)
                .get(3, TimeUnit.SECONDS);
        Response second = fixture.scheduler.submit(small)
                .get(3, TimeUnit.SECONDS);

        assertEquals(StrategyErrorType.RESOURCE_EXHAUSTED.getErrorCode(),
                first.getCode());
        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                second.getCode(),
                "a later request for the same role/group must be routed immediately");
        assertEquals(2, router.attempts.get());
        assertEquals(0, fixture.scheduler.getQueuedRequestCount(),
                "a static rejection must not occupy a placement wait lane");
    }

    @Test
    @Timeout(30)
    void healthyFirstAttemptsRemainConcurrentWithoutDeferredBacklog()
            throws Exception {
        int requests = 4;
        ConcurrentFatalRouter router = new ConcurrentFatalRouter(requests);
        SchedulerFixture fixture = fixture(router);
        CyclicBarrier submitTogether = new CyclicBarrier(requests);

        try (ExecutorService executor =
                     Executors.newFixedThreadPool(requests)) {
            List<Future<CompletableFuture<Response>>> submissions =
                    new ArrayList<>();
            for (int index = 0; index < requests; index++) {
                long requestId = 200L + index;
                submissions.add(executor.submit(() -> {
                    await(submitTogether);
                    return fixture.scheduler.submit(context(requestId));
                }));
            }

            List<CompletableFuture<Response>> responses = new ArrayList<>();
            for (Future<CompletableFuture<Response>> submission : submissions) {
                responses.add(submission.get(3, TimeUnit.SECONDS));
            }
            for (CompletableFuture<Response> response : responses) {
                assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                        response.get(3, TimeUnit.SECONDS).getCode());
            }
        }

        assertEquals(requests, router.attempts.get());
        assertTrue(router.maxConcurrent.get() > 1,
                "an empty deferred backlog must not serialize independent "
                        + "first placement attempts");
    }

    @Test
    @Timeout(30)
    void laterSamePriorityRequestCannotCommitBeforeFullResultReachesLane()
            throws Exception {
        FullHandoffRaceRouter router = new FullHandoffRaceRouter();
        SchedulerFixture fixture = fixture(router);

        try (ExecutorService executor = Executors.newFixedThreadPool(2)) {
            Future<CompletableFuture<Response>> older = executor.submit(
                    () -> fixture.scheduler.submit(context(
                            FullHandoffRaceRouter.OLDER_REQUEST_ID,
                            50,
                            Long.MAX_VALUE,
                            fixture.config)));
            assertTrue(router.olderCommitEntered.await(3, TimeUnit.SECONDS),
                    "the older request never entered its final Prefill admission");

            Future<CompletableFuture<Response>> later = executor.submit(
                    () -> fixture.scheduler.submit(context(
                            FullHandoffRaceRouter.LATER_REQUEST_ID,
                            50,
                            Long.MAX_VALUE,
                            fixture.config)));
            assertTrue(router.laterRouteReturned.await(3, TimeUnit.SECONDS),
                    "the later request never selected the same exact Prefill");

            try {
                assertFalse(router.laterCommitted.await(1, TimeUnit.SECONDS),
                        "a later same-priority request stole the exact Prefill "
                                + "before the older FULL result was published "
                                + "to its wait lane");
            } finally {
                router.releaseOlderCommit.countDown();
            }

            older.get(3, TimeUnit.SECONDS);
            later.get(3, TimeUnit.SECONDS);
        }
    }

    @Test
    @Timeout(30)
    void displacedSourceLaneAttemptCanMigrateToAFreshNonOverlappingEndpoint()
            throws Exception {
        MigratingRouter router = new MigratingRouter();
        SchedulerFixture fixture = fixture(router);

        CompletableFuture<Response> first = fixture.scheduler.submit(
                context(301L, 50, Long.MAX_VALUE, fixture.config));
        assertTrue(router.firstRetryEntered.await(3, TimeUnit.SECONDS),
                "the H1 lane head never entered its fresh retry");

        CompletableFuture<Response> higherPriority = fixture.scheduler.submit(
                context(302L, 100, Long.MAX_VALUE, fixture.config));
        assertEquals(1, router.attemptsFor(302L),
                "the later request must first classify into the same H1 lane");

        router.releaseFirstRetry.countDown();
        awaitCondition(() -> fixture.lifecycle.requestSlot(301L) != null
                && activeItem(fixture.lifecycle.requestSlot(301L)) != null);

        BatchItem migrated = activeItem(fixture.lifecycle.requestSlot(301L));
        assertNotNull(migrated);
        assertEquals("h2", migrated.prefill().getServerIp(),
                "the displaced H1 attempt must commit its fresh H2 selection");
        assertFalse(first.isDone(),
                "a placement commit is not an engine-dispatch terminal");
        assertFalse(higherPriority.isDone(),
                "the higher-priority H1 waiter must remain independently queued");
    }

    @Test
    @Timeout(30)
    void deadlineInWaitLanePublishesOnlyBatchSloExpiredAndNeverDispatches()
            throws Exception {
        AlwaysDeferredRouter router = new AlwaysDeferredRouter();
        SchedulerFixture fixture = fixture(router);
        AtomicInteger terminalPublications = new AtomicInteger();

        CompletableFuture<Response> future = fixture.scheduler.submit(
                context(401L, 50, System.currentTimeMillis() + 100L,
                        fixture.config));
        future.whenComplete((response, failure) ->
                terminalPublications.incrementAndGet());

        Response terminal = future.get(3, TimeUnit.SECONDS);
        assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(),
                terminal.getCode());
        assertEquals(RequestLifecycleState.TIMED_OUT,
                fixture.scheduler.getRequestState(401L, 0L).state());
        RequestSlot slot = fixture.lifecycle.requestSlot(401L);
        assertNotNull(slot);
        assertNull(activeItem(slot),
                "a request which expires in a wait lane must never dispatch");
        TimeUnit.MILLISECONDS.sleep(100L);
        assertEquals(1, terminalPublications.get(),
                "deadline ownership must publish exactly one terminal result");
    }

    @Test
    @Timeout(30)
    void policyReselectDeadlineBypassesAFullSevenHundredFiftyLaneBudget()
            throws Exception {
        int backgroundLanes = 750;
        long targetId = 900_000L;
        long reselectAtMs = System.currentTimeMillis() + 120L;
        PolicyDeadlineRouter router = new PolicyDeadlineRouter(
                targetId, reselectAtMs);
        SchedulerFixture fixture = fixture(router, backgroundLanes + 16);
        for (int lane = 0; lane < backgroundLanes; lane++) {
            fixture.scheduler.submit(context(
                    targetId + 1L + lane,
                    50,
                    Long.MAX_VALUE,
                    fixture.config));
        }

        long startedNanos = System.nanoTime();
        Response terminal = fixture.scheduler.submit(context(
                        targetId, 50, Long.MAX_VALUE, fixture.config))
                .get(1, TimeUnit.SECONDS);
        long elapsedMs = TimeUnit.NANOSECONDS.toMillis(
                System.nanoTime() - startedNanos);

        assertEquals(StrategyErrorType.INVALID_REQUEST.getErrorCode(),
                terminal.getCode());
        assertTrue(router.targetAttempts.get() >= 2);
        assertTrue(elapsedMs < 500L,
                () -> "policy refresh waited " + elapsedMs
                        + "ms behind the global fallback stream");
    }

    @Test
    @Timeout(30)
    void cancellingEarlierLanesDoesNotLeaveAGhostFallbackDelay()
            throws Exception {
        int laneCount = 300;
        ManyLaneDeferredRouter router = new ManyLaneDeferredRouter(laneCount);
        SchedulerFixture fixture = fixture(router, laneCount + 16);
        for (int lane = 0; lane < laneCount; lane++) {
            fixture.scheduler.submit(context(
                    ManyLaneDeferredRouter.FIRST_REQUEST_ID + lane,
                    50,
                    Long.MAX_VALUE,
                    fixture.config));
        }
        int survivor = laneCount - 1;
        assertEquals(1, router.attemptsFor(survivor),
                "the tail lane unexpectedly consumed its fallback turn");
        for (int lane = 0; lane < survivor; lane++) {
            fixture.scheduler.cancelRequest(
                    ManyLaneDeferredRouter.FIRST_REQUEST_ID + lane,
                    0L,
                    CancelReason.CLIENT_CANCELLED);
        }
        awaitCondition(() -> fixture.scheduler.getQueuedRequestCount() == 1);

        int before = router.attemptsFor(survivor);
        long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(300L);
        while (router.attemptsFor(survivor) == before
                && System.nanoTime() < deadline) {
            TimeUnit.MILLISECONDS.sleep(1L);
        }
        assertTrue(router.attemptsFor(survivor) > before,
                "cancelled lanes left their old global probe reservations behind");
    }

    @Test
    @Tag("performance-regression")
    @Timeout(value = 120, unit = TimeUnit.SECONDS)
    void hundredThousandSameLaneWaitersShareOneBoundedSteadyProbeStream()
            throws Exception {
        int requests = Integer.getInteger(
                "flexlb.perf.placement-lane.waiters", 100_000);
        int steadyWindowMs = Integer.getInteger(
                "flexlb.perf.placement-lane.steady-window-ms", 250);
        int maximumSteadyRoutes = Integer.getInteger(
                "flexlb.perf.placement-lane.max-steady-routes", 12);
        AlwaysDeferredRouter router = new AlwaysDeferredRouter();
        SchedulerFixture fixture = fixture(router, requests + 16);

        long started = System.nanoTime();
        for (int index = 0; index < requests; index++) {
            fixture.scheduler.submit(context(
                    1_000_000L + index,
                    50,
                    Long.MAX_VALUE,
                    fixture.config));
        }
        long classifiedNanos = System.nanoTime() - started;
        assertTrue(router.attempts.get() >= requests,
                "every request must complete its caller-thread classification");

        TimeUnit.MILLISECONDS.sleep(100L);
        int routesBefore = router.attempts.get();
        TimeUnit.MILLISECONDS.sleep(steadyWindowMs);
        int steadyRoutes = router.attempts.get() - routesBefore;
        System.out.printf(
                "FlexLB placement-lane performance: waiters=%d "
                        + "classification_ms=%.3f steady_window_ms=%d "
                        + "steady_route_calls=%d%n",
                requests,
                classifiedNanos / 1_000_000.0,
                steadyWindowMs,
                steadyRoutes);

        assertTrue(steadyRoutes <= maximumSteadyRoutes,
                () -> "one blocked lane made " + steadyRoutes
                        + " fresh routes in " + steadyWindowMs
                        + " ms; expected a shared O(1) probe stream capped at "
                        + maximumSteadyRoutes);
    }

    @Test
    @Tag("performance-regression")
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void sevenHundredFiftyLanesShareOneFairBoundedFallbackBudget()
            throws Exception {
        int laneCount = Integer.getInteger(
                "flexlb.perf.placement-lane.count", 750);
        int steadyWindowMs = Integer.getInteger(
                "flexlb.perf.placement-lane.global-window-ms", 1_000);
        int maximumSteadyRoutes = Integer.getInteger(
                "flexlb.perf.placement-lane.max-global-routes", 600);
        ManyLaneDeferredRouter router = new ManyLaneDeferredRouter(laneCount);
        SchedulerFixture fixture = fixture(router, laneCount + 16);

        long started = System.nanoTime();
        for (int index = 0; index < laneCount; index++) {
            fixture.scheduler.submit(context(
                    ManyLaneDeferredRouter.FIRST_REQUEST_ID + index,
                    50,
                    Long.MAX_VALUE,
                    fixture.config));
        }
        awaitCondition(router::everyLaneRetried);
        long firstSweepNanos = System.nanoTime() - started;

        int routesBefore = router.attempts.get();
        TimeUnit.MILLISECONDS.sleep(steadyWindowMs);
        int steadyRoutes = router.attempts.get() - routesBefore;
        System.out.printf(
                "FlexLB global placement budget: lanes=%d first_sweep_ms=%.3f "
                        + "steady_window_ms=%d steady_route_calls=%d%n",
                laneCount,
                firstSweepNanos / 1_000_000.0,
                steadyWindowMs,
                steadyRoutes);

        assertTrue(router.everyLaneRetried(),
                "round-robin fallback must not permanently starve any lane");
        assertTrue(steadyRoutes <= maximumSteadyRoutes,
                () -> laneCount + " blocked lanes made " + steadyRoutes
                        + " fresh routes in " + steadyWindowMs
                        + " ms; expected one scheduler-wide stream capped at "
                        + maximumSteadyRoutes);
    }

    private SchedulerFixture fixture(Router router) {
        return fixture(router, 64);
    }

    private SchedulerFixture fixture(Router router, int maxOutstanding) {
        SchedulerFixture fixture = new SchedulerFixture(router, maxOutstanding);
        fixtures.add(fixture);
        return fixture;
    }

    private static BalanceContext context(long requestId) {
        return context(requestId, 50, Long.MAX_VALUE, null);
    }

    private static BalanceContext context(
            long requestId,
            int priority,
            long expiresAtMs,
            FlexlbConfig config) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(128L);
        request.setMaxNewTokens(8);
        request.setPriority(priority);
        request.setModel("deferred-retry-contract");
        BalanceContext context = new BalanceContext();
        context.setConfig(config);
        context.setRequest(request);
        context.setSchedulingMetadata(
                SchedulingMetadata.explicit(priority, expiresAtMs));
        return context;
    }

    private static BatchItem activeItem(RequestSlot slot) {
        synchronized (slot) {
            return slot.activeItem();
        }
    }

    private static void awaitCondition(java.util.function.BooleanSupplier condition)
            throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (!condition.getAsBoolean() && System.nanoTime() < deadline) {
            TimeUnit.MILLISECONDS.sleep(1L);
        }
        assertTrue(condition.getAsBoolean(), "condition did not become true");
    }

    private static Response rejected(StrategyErrorType type) {
        return RequestLifecycleCoordinator.buildErrorResponse(type, null);
    }

    private static void await(CyclicBarrier barrier) {
        try {
            barrier.await(2, TimeUnit.SECONDS);
        } catch (Exception failure) {
            throw new AssertionError("concurrent test barrier failed", failure);
        }
    }

    private static final class SchedulerFixture implements AutoCloseable {
        private final EndpointRegistry endpointRegistry;
        private final RequestLifecycleCoordinator lifecycle;
        private final RequestScheduler scheduler;
        private final FlexlbConfig config;

        private SchedulerFixture(Router router, int maxOutstanding) {
            config = SchedulingTestConfig.batchConfig();
            SchedulingTestConfig.usePriorityQueue(config);
            SchedulingTestConfig.useQueueCapacity(config)
                    .setMaxOutstandingRequestsGlobal(maxOutstanding);
            ConfigService configService = new ConfigService() {
                @Override
                public FlexlbConfig loadBalanceConfig() {
                    return config;
                }
            };
            BatchSchedulerReporter batchReporter =
                    new BatchSchedulerReporter(new NoOpFlexMonitor());
            lifecycle = new RequestLifecycleCoordinator(
                    configService,
                    batchReporter,
                    new RequestSchedulerReporter(new NoOpFlexMonitor()),
                    new NoCancelChannel());
            endpointRegistry = new EndpointRegistry(
                    configService,
                    lifecycle,
                    batchReporter,
                    () -> {
                        throw new AssertionError(
                                "no endpoint is constructed in this fixture");
                    },
                    (endpointId, endpoint, activeConfig, delivery, port) -> {
                        throw new AssertionError(
                                "no endpoint is constructed in this fixture");
                    });
            AdmissionFallback noFallback = (context, future) -> false;
            scheduler = new RequestScheduler(
                    configService,
                    router,
                    endpointRegistry,
                    batchReporter,
                    noFallback,
                    lifecycle);
        }

        @Override
        public void close() {
            new RequestShutdownOrchestrator(
                    lifecycle, endpointRegistry, scheduler)
                    .shutdown();
        }
    }

    private static final class NoCancelChannel
            implements EngineCancelChannel {

        @Override
        public boolean isSupported(
                org.flexlb.balance.endpoint.DecodeEndpoint endpoint) {
            return false;
        }

        @Override
        public CompletableFuture<CancelOutcome> cancel(
                org.flexlb.balance.preemption.CancelTarget target,
                long requestId,
                long timeoutMs) {
            return CompletableFuture.completedFuture(
                    CancelOutcome.unsupported());
        }
    }

    private static final class RetryThenFatalRouter implements Router {
        private final AtomicInteger attempts = new AtomicInteger();

        @Override
        public Response routeDirect(BalanceContext context) {
            throw new AssertionError("QUEUE contract must not route DIRECT");
        }

        @Override
        public QueueRoutingResult routeForQueue(BalanceContext context) {
            if (attempts.getAndIncrement() == 0) {
                return new QueueRoutingResult.Deferred(
                        org.flexlb.dao.route.RoleType.PREFILL, null);
            }
            return new QueueRoutingResult.Rejected(
                    rejected(StrategyErrorType.INVALID_REQUEST));
        }
    }

    private static final class StaticRejectingRouter implements Router {
        private final AtomicInteger attempts = new AtomicInteger();

        @Override
        public Response routeDirect(BalanceContext context) {
            throw new AssertionError("QUEUE contract must not route DIRECT");
        }

        @Override
        public QueueRoutingResult routeForQueue(BalanceContext context) {
            attempts.incrementAndGet();
            StrategyErrorType type = context.getRequest().getSeqLen() > 256L
                    ? StrategyErrorType.RESOURCE_EXHAUSTED
                    : StrategyErrorType.INVALID_REQUEST;
            return new QueueRoutingResult.Rejected(rejected(type));
        }
    }

    private static final class ConcurrentFatalRouter implements Router {
        private final CyclicBarrier routeTogether;
        private final AtomicInteger attempts = new AtomicInteger();
        private final AtomicInteger active = new AtomicInteger();
        private final AtomicInteger maxConcurrent = new AtomicInteger();

        private ConcurrentFatalRouter(int requests) {
            routeTogether = new CyclicBarrier(requests);
        }

        @Override
        public Response routeDirect(BalanceContext context) {
            throw new AssertionError("QUEUE contract must not route DIRECT");
        }

        @Override
        public QueueRoutingResult routeForQueue(BalanceContext context) {
            attempts.incrementAndGet();
            int nowActive = active.incrementAndGet();
            maxConcurrent.accumulateAndGet(nowActive, Math::max);
            try {
                await(routeTogether);
                return new QueueRoutingResult.Rejected(
                        rejected(StrategyErrorType.INVALID_REQUEST));
            } finally {
                active.decrementAndGet();
            }
        }
    }

    private static final class AlwaysDeferredRouter implements Router {
        private final AtomicInteger attempts = new AtomicInteger();

        @Override
        public Response routeDirect(BalanceContext context) {
            throw new AssertionError("QUEUE contract must not route DIRECT");
        }

        @Override
        public QueueRoutingResult routeForQueue(BalanceContext context) {
            attempts.incrementAndGet();
            return new QueueRoutingResult.Deferred(
                    RoleType.PREFILL, "test");
        }
    }

    private static final class ManyLaneDeferredRouter implements Router {
        private static final long FIRST_REQUEST_ID = 2_000_000L;

        private final AtomicInteger attempts = new AtomicInteger();
        private final AtomicIntegerArray attemptsByLane;

        private ManyLaneDeferredRouter(int lanes) {
            attemptsByLane = new AtomicIntegerArray(lanes);
        }

        @Override
        public Response routeDirect(BalanceContext context) {
            throw new AssertionError("QUEUE contract must not route DIRECT");
        }

        @Override
        public QueueRoutingResult routeForQueue(BalanceContext context) {
            int lane = Math.toIntExact(
                    context.getRequestId() - FIRST_REQUEST_ID);
            attempts.incrementAndGet();
            attemptsByLane.incrementAndGet(lane);
            return new QueueRoutingResult.Deferred(
                    RoleType.PREFILL, "budget-lane-" + lane);
        }

        private boolean everyLaneRetried() {
            for (int lane = 0; lane < attemptsByLane.length(); lane++) {
                if (attemptsByLane.get(lane) < 2) { return false; }
            }
            return true;
        }

        private int attemptsFor(int lane) {
            return attemptsByLane.get(lane);
        }
    }

    private static final class PolicyDeadlineRouter implements Router {
        private final long targetId;
        private final long reselectAtMs;
        private final AtomicInteger targetAttempts = new AtomicInteger();
        private final PrefillEndpoint full = mock(PrefillEndpoint.class);

        private PolicyDeadlineRouter(long targetId, long reselectAtMs) {
            this.targetId = targetId;
            this.reselectAtMs = reselectAtMs;
            when(full.getIp()).thenReturn("policy-hot");
            when(full.prepareOfferPinned(
                    org.mockito.ArgumentMatchers.any(),
                    org.mockito.ArgumentMatchers.anyLong(),
                    org.mockito.ArgumentMatchers.anyInt()))
                    .thenReturn(null);
        }

        @Override
        public Response routeDirect(BalanceContext context) {
            throw new AssertionError("QUEUE contract must not route DIRECT");
        }

        @Override
        public QueueRoutingResult routeForQueue(BalanceContext context) {
            if (context.getRequestId() != targetId) {
                return new QueueRoutingResult.Deferred(
                        RoleType.PREFILL,
                        "policy-background-" + context.getRequestId());
            }
            targetAttempts.incrementAndGet();
            if (System.currentTimeMillis() >= reselectAtMs) {
                return new QueueRoutingResult.Rejected(
                        rejected(StrategyErrorType.INVALID_REQUEST));
            }
            WorkerEndpoint.GenerationPin pin =
                    mock(WorkerEndpoint.GenerationPin.class);
            when(pin.endpoint()).thenReturn(full);
            SelectedRole selected = mock(SelectedRole.class);
            ServerStatus status = new ServerStatus();
            status.setSuccess(true);
            status.setRole(RoleType.PREFILL);
            status.setRequestId(targetId);
            status.setServerIp("policy-hot");
            status.setHttpPort(8010);
            status.setGroup("policy-target");
            when(selected.serverStatus()).thenReturn(status);
            when(selected.takeGenerationPin()).thenReturn(pin);
            when(selected.reselectNotAfterMs()).thenReturn(reselectAtMs);
            Response response = new Response();
            response.setSuccess(true);
            response.setServerStatus(List.of(status));
            return QueueRouteAdmission.prepare(
                    context, List.of(selected), response);
        }
    }

    private static final class FullHandoffRaceRouter implements Router {
        private static final long OLDER_REQUEST_ID = 281L;
        private static final long LATER_REQUEST_ID = 282L;

        private final CountDownLatch olderCommitEntered = new CountDownLatch(1);
        private final CountDownLatch releaseOlderCommit = new CountDownLatch(1);
        private final CountDownLatch laterRouteReturned = new CountDownLatch(1);
        private final CountDownLatch laterCommitted = new CountDownLatch(1);
        private final PrefillEndpoint endpoint = mock(PrefillEndpoint.class);

        private FullHandoffRaceRouter() {
            when(endpoint.getIp()).thenReturn("h1");
            when(endpoint.prepareOfferPinned(
                    org.mockito.ArgumentMatchers.any(),
                    org.mockito.ArgumentMatchers.anyLong(),
                    org.mockito.ArgumentMatchers.anyInt()))
                    .thenAnswer(invocation -> {
                        long requestId = invocation.getArgument(1);
                        if (requestId == OLDER_REQUEST_ID) {
                            olderCommitEntered.countDown();
                            if (!releaseOlderCommit.await(3, TimeUnit.SECONDS)) {
                                throw new AssertionError(
                                        "older Prefill admission was not released");
                            }
                            return null;
                        }
                        PrefillGenerationRuntime.PreparedOffer offer =
                                mock(PrefillGenerationRuntime.PreparedOffer.class);
                        when(offer.seal()).thenReturn(true);
                        org.mockito.Mockito.doAnswer(ignored -> {
                            laterCommitted.countDown();
                            return null;
                        }).when(offer).commit(
                                org.mockito.ArgumentMatchers.any());
                        return offer;
                    });
        }

        @Override
        public Response routeDirect(BalanceContext context) {
            throw new AssertionError("QUEUE contract must not route DIRECT");
        }

        @Override
        public QueueRoutingResult routeForQueue(BalanceContext context) {
            QueueRoutingResult result = admittedOn(context);
            if (context.getRequestId() == LATER_REQUEST_ID) {
                laterRouteReturned.countDown();
            }
            return result;
        }

        private QueueRoutingResult admittedOn(BalanceContext context) {
            WorkerEndpoint.GenerationPin pin =
                    mock(WorkerEndpoint.GenerationPin.class);
            when(pin.endpoint()).thenReturn(endpoint);
            SelectedRole selected = mock(SelectedRole.class);
            ServerStatus status = new ServerStatus();
            status.setSuccess(true);
            status.setRole(RoleType.PREFILL);
            status.setRequestId(context.getRequestId());
            status.setServerIp("h1");
            status.setHttpPort(8001);
            status.setGroup("test");
            when(selected.serverStatus()).thenReturn(status);
            when(selected.takeGenerationPin()).thenReturn(pin);
            Response response = new Response();
            response.setSuccess(true);
            response.setServerStatus(List.of(status));
            return QueueRouteAdmission.prepare(
                    context, List.of(selected), response);
        }
    }

    private static final class MigratingRouter implements Router {
        private final CountDownLatch firstRetryEntered = new CountDownLatch(1);
        private final CountDownLatch releaseFirstRetry = new CountDownLatch(1);
        private final java.util.concurrent.ConcurrentMap<Long, AtomicInteger>
                attempts = new java.util.concurrent.ConcurrentHashMap<>();
        private final PrefillEndpoint fullH1 = fullEndpoint("h1");

        @Override
        public Response routeDirect(BalanceContext context) {
            throw new AssertionError("QUEUE contract must not route DIRECT");
        }

        @Override
        public QueueRoutingResult routeForQueue(BalanceContext context) {
            long requestId = context.getRequestId();
            int attempt = attempts.computeIfAbsent(
                    requestId, ignored -> new AtomicInteger())
                    .incrementAndGet();
            if (requestId == 301L && attempt == 2) {
                firstRetryEntered.countDown();
                await(releaseFirstRetry);
                return admittedOn(context, availableEndpoint("h2"), "h2", 8002);
            }
            return admittedOn(context, fullH1, "h1", 8001);
        }

        private int attemptsFor(long requestId) {
            AtomicInteger count = attempts.get(requestId);
            return count == null ? 0 : count.get();
        }

        private static PrefillEndpoint fullEndpoint(String ip) {
            PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
            when(endpoint.getIp()).thenReturn(ip);
            return endpoint;
        }

        private static PrefillEndpoint availableEndpoint(String ip) {
            PrefillEndpoint endpoint = fullEndpoint(ip);
            PrefillGenerationRuntime.PreparedOffer offer =
                    mock(PrefillGenerationRuntime.PreparedOffer.class);
            when(offer.seal()).thenReturn(true);
            when(endpoint.prepareOfferPinned(
                    org.mockito.ArgumentMatchers.any(),
                    org.mockito.ArgumentMatchers.anyLong(),
                    org.mockito.ArgumentMatchers.anyInt()))
                    .thenReturn(offer);
            return endpoint;
        }

        private static QueueRoutingResult admittedOn(
                BalanceContext context,
                PrefillEndpoint endpoint,
                String ip,
                int port) {
            WorkerEndpoint.GenerationPin pin =
                    mock(WorkerEndpoint.GenerationPin.class);
            when(pin.endpoint()).thenReturn(endpoint);
            SelectedRole selected = mock(SelectedRole.class);
            ServerStatus status = new ServerStatus();
            status.setSuccess(true);
            status.setRole(RoleType.PREFILL);
            status.setRequestId(context.getRequestId());
            status.setServerIp(ip);
            status.setHttpPort(port);
            status.setGroup("test");
            when(selected.serverStatus()).thenReturn(status);
            when(selected.takeGenerationPin()).thenReturn(pin);
            Response response = new Response();
            response.setSuccess(true);
            response.setServerStatus(List.of(status));
            return QueueRouteAdmission.prepare(
                    context, List.of(selected), response);
        }

        private static void await(CountDownLatch latch) {
            try {
                assertTrue(latch.await(3, TimeUnit.SECONDS),
                        "migration test latch timed out");
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                throw new AssertionError(
                        "migration test latch interrupted", interrupted);
            }
        }
    }
}
