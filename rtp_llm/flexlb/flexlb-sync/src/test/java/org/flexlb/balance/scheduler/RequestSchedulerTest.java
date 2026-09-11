package org.flexlb.balance.scheduler;


import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.eviction.EvictionManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.NullSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.after;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class RequestSchedulerTest {

    private static DefaultRouter mockRouter() {
        DefaultRouter router = mock(DefaultRouter.class);
        when(router.queueAdmissionRole()).thenReturn(RoleType.PREFILL);
        return router;
    }

    @ParameterizedTest
    @NullSource
    @ValueSource(strings = {"healthy-group", "unavailable-group"})
    void unavailableRequestMustNotStopAHealthyFollowingRequest(String healthyGroup) {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        DefaultRouter router = mockRouter();
        when(router.queueAdmissionRole()).thenReturn(RoleType.PREFILL);
        EndpointRegistry endpoints = mock(EndpointRegistry.class);
        RequestRegistry lifecycle = mock(RequestRegistry.class);
        BalanceContext unavailable = RequestLifecycleTestSupport.context(config, 910001L);
        BalanceContext healthy = RequestLifecycleTestSupport.context(config, 910002L);
        CompletableFuture<Response> first = new CompletableFuture<>();
        CompletableFuture<Response> second = new CompletableFuture<>();
        when(lifecycle.register(unavailable)).thenReturn(first);
        when(lifecycle.register(healthy)).thenReturn(second);
        when(lifecycle.claimAdmissionMutation(910001L, first)).thenReturn(mock(AdmissionMutation.class));
        when(lifecycle.claimAdmissionMutation(910002L, second)).thenReturn(mock(AdmissionMutation.class));
        when(router.resolvePolicyGroup(unavailable)).thenReturn("unavailable-group");
        when(router.resolvePolicyGroup(healthy)).thenReturn(healthyGroup);
        when(router.select(unavailable, "unavailable-group")).thenReturn(PlacementResult.blocked(new PlacementKey(RoleType.DECODE, "unavailable-group")));
        RouteAdmission healthyRoute = mock(RouteAdmission.class);
        when(router.select(healthy, healthyGroup)).thenReturn(PlacementResult.success(healthyRoute));
        when(healthyRoute.tryEnqueue(healthy, second, lifecycle))
                .thenReturn(PlacementResult.success(mock(ScheduledRequest.class)));
        RequestScheduler scheduler = new RequestScheduler(service, router, endpoints, mock(BatchSchedulerReporter.class), mock(EvictionManager.class), lifecycle, new PlacementAvailability());
        try {
            scheduler.submit(unavailable);
            verify(router, timeout(500)).select(unavailable, "unavailable-group");
            scheduler.submit(healthy);
            verify(router, timeout(500)).select(healthy, healthyGroup);
            verify(healthyRoute, timeout(500)).tryEnqueue(healthy, second, lifecycle);
        } finally {
            first.complete(new Response());
            second.complete(new Response());
            scheduler.closePlacement();
        }
    }

    @Test
    void priorityRescueConsumesTheOriginalExactRoute() {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.allowVictim(
                config, VictimStage.DECODE_RESERVED);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        DefaultRouter router = mockRouter();
        when(router.queueAdmissionRole()).thenReturn(RoleType.PREFILL);
        EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
        RequestRegistry lifecycle = mock(RequestRegistry.class);
        EvictionManager eviction = mock(EvictionManager.class);
        PlacementAvailability availability = new PlacementAvailability();

        BalanceContext context = context(897L, 90);
        CompletableFuture<Response> future = new CompletableFuture<>();
        when(lifecycle.register(context)).thenReturn(future);
        when(lifecycle.claimAdmissionMutation(897L, future)).thenReturn(
                mock(AdmissionMutation.class));

        DecodeEndpoint selectedEndpoint = mock(DecodeEndpoint.class);
        when(selectedEndpoint.ipPort()).thenReturn("selected-decode:8080");
        RouteAdmission selectedRoute = mock(RouteAdmission.class);
        PlacementKey blocker = PlacementKey.exact(
                RoleType.DECODE, "g1", "selected-decode:8080");
        when(router.select(context, null)).thenReturn(
                PlacementResult.success(selectedRoute));
        when(selectedRoute.tryEnqueue(context, future, lifecycle))
                .thenReturn(PlacementResult.blocked(blocker));
        when(selectedRoute.blockedEndpoint()).thenReturn(selectedEndpoint);
        when(eviction.tryAdmit(
                context, future, selectedRoute, selectedEndpoint))
                .thenReturn(true);

        RequestScheduler scheduler = new RequestScheduler(
                configService,
                router,
                endpointRegistry,
                mock(BatchSchedulerReporter.class),
                eviction,
                lifecycle,
                availability);
        try {
            scheduler.submit(context);

            verify(eviction, timeout(1_000)).tryAdmit(
                    context, future, selectedRoute, selectedEndpoint);
            verify(router, times(1)).select(context, null);
        } finally {
            future.complete(new Response());
            scheduler.closePlacement();
        }
    }

    @Test
    void nonBatchWaitsWhenEveryEngineRequestSlotIsOccupied() {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useFixedWindowDecision(config);
        SchedulingTestConfig.useNonBatchDispatcher(config);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        DefaultRouter router = mockRouter();
        when(router.queueAdmissionRole()).thenReturn(RoleType.PREFILL);
        EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
        when(endpointRegistry.getEndpointCount(RoleType.PREFILL)).thenReturn(1);
        AtomicInteger availableSlots = new AtomicInteger();
        RequestRegistry lifecycle = mock(RequestRegistry.class);
        BalanceContext context = context(899L);
        CompletableFuture<Response> future = new CompletableFuture<>();
        when(lifecycle.register(context)).thenReturn(future);
        when(lifecycle.claimAdmissionMutation(899L, future)).thenReturn(
                mock(AdmissionMutation.class));
        RouteAdmission route = mock(RouteAdmission.class);
        PrefillEndpoint endpoint = mockPrefillEndpoint("127.0.0.1", 8000);
        when(route.blockedEndpoint()).thenReturn(endpoint);
        when(route.tryEnqueue(context, future, lifecycle)).thenReturn(
                PlacementResult.blocked(PlacementKey.exact(
                        RoleType.PREFILL, "g1", "127.0.0.1:8000")));
        when(router.select(context, null)).thenReturn(
                PlacementResult.success(route),
                PlacementResult.rejected(RequestRegistry.buildErrorResponse(
                        StrategyErrorType.NO_PREFILL_WORKER, null)));
        PlacementAvailability availability = new PlacementAvailability();
        RequestScheduler scheduler = new RequestScheduler(
                configService,
                router,
                endpointRegistry,
                mock(BatchSchedulerReporter.class),
                mock(EvictionManager.class),
                lifecycle,
                availability);
        try {
            scheduler.submit(context);
            verify(route, timeout(1_000)).tryEnqueue(context, future, lifecycle);
            verify(router, after(100).times(1)).select(context, null);

            availableSlots.set(1);
            availability.capacityChanged(PlacementKey.exact(
                    RoleType.PREFILL, "g1", "127.0.0.1:8000"));
            verify(router, timeout(1_000).times(2)).select(context, null);
        } finally {
            scheduler.closePlacement();
        }
    }

    @Test
    void planningContinuesWithoutAnAvailablePrefillEndpoint() {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useFixedWindowDecision(config)
                .setMaxRequests(2);
        SchedulingTestConfig.useNonBatchDispatcher(config);
        config.queueScheduler().getDecision().setMaxCollectionWaitMs(0L);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        DefaultRouter router = mockRouter();
        when(router.queueAdmissionRole()).thenReturn(RoleType.PREFILL);
        EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
        RequestRegistry lifecycle = mock(RequestRegistry.class);
        BalanceContext context = context(900L);
        CompletableFuture<Response> future = new CompletableFuture<>();
        when(lifecycle.register(context)).thenReturn(future);
        when(lifecycle.claimAdmissionMutation(900L, future)).thenReturn(
                mock(AdmissionMutation.class));
        when(router.select(context, null)).thenReturn(
                PlacementResult.rejected(RequestRegistry.buildErrorResponse(
                        StrategyErrorType.NO_PREFILL_WORKER, null)));

        RequestScheduler scheduler = new RequestScheduler(
                configService,
                router,
                endpointRegistry,
                mock(BatchSchedulerReporter.class),
                mock(EvictionManager.class),
                lifecycle,
                new PlacementAvailability());
        scheduler.submit(context);

        verify(router, timeout(1_000)).select(context, null);
        scheduler.closePlacement();
    }

    @Test
    void singleDecisionDoesNotSerializeTheGlobalPlanningFrontier()
            throws Exception {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useSingleDecision(config);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        DefaultRouter router = mockRouter();
        EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
        RequestRegistry lifecycle = mock(RequestRegistry.class);
        BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
        PlacementAvailability availability = new PlacementAvailability();
        when(router.queueAdmissionRole()).thenReturn(RoleType.PREFILL);

        BalanceContext gate = context(900L);
        CompletableFuture<Response> gateFuture = new CompletableFuture<>();
        CountDownLatch gatePlanningStarted = new CountDownLatch(1);
        CountDownLatch releaseGatePlanning = new CountDownLatch(1);
        when(lifecycle.register(gate)).thenReturn(gateFuture);
        when(lifecycle.claimAdmissionMutation(900L, gateFuture)).thenReturn(
                mock(AdmissionMutation.class));
        when(router.select(gate, null)).thenAnswer(invocation -> {
            gatePlanningStarted.countDown();
            releaseGatePlanning.await(5, TimeUnit.SECONDS);
            return PlacementResult.rejected(
                    RequestRegistry.buildErrorResponse(
                            StrategyErrorType.NO_PREFILL_WORKER, null));
        });

        CountDownLatch aggregatePlansStarted = new CountDownLatch(6);
        List<BalanceContext> contexts = new ArrayList<>();
        for (long requestId = 901L; requestId < 907L; requestId++) {
            BalanceContext context = context(requestId);
            CompletableFuture<Response> future = new CompletableFuture<>();
            RouteAdmission route = mock(RouteAdmission.class);
            when(lifecycle.register(context)).thenReturn(future);
            when(lifecycle.claimAdmissionMutation(requestId, future)).thenReturn(
                    mock(AdmissionMutation.class));
            when(router.select(context, null)).thenAnswer(invocation -> {
                aggregatePlansStarted.countDown();
                return PlacementResult.success(route);
            });
            ScheduledRequest published = mock(ScheduledRequest.class);
            when(published.prefillEp()).thenReturn(mock(PrefillEndpoint.class));
            when(route.tryEnqueue(context, future, lifecycle)).thenReturn(
                    PlacementResult.success(published));
            contexts.add(context);
        }

        RequestScheduler scheduler = new RequestScheduler(
                configService,
                router,
                endpointRegistry,
                reporter,
                mock(EvictionManager.class),
                lifecycle,
                availability);
        try {
            scheduler.submit(gate);
            assertTrue(gatePlanningStarted.await(5, TimeUnit.SECONDS));
            for (BalanceContext context : contexts) {
                scheduler.submit(context);
            }
            releaseGatePlanning.countDown();
            assertTrue(aggregatePlansStarted.await(5, TimeUnit.SECONDS),
                    "all aggregate slots must be submitted from one captured frontier");
            } finally {
            releaseGatePlanning.countDown();
            scheduler.closePlacement();
        }
    }

    @Test
    void higherPriorityRequestRunsAheadOfBlockedLowerPriorityFrontier() {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useSingleDecision(config);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        DefaultRouter router = mockRouter();
        RequestRegistry lifecycle = mock(RequestRegistry.class);
        EvictionManager evictionManager = mock(EvictionManager.class);

        BalanceContext lowPriority = context(910L, 10);
        BalanceContext highPriority = context(911L, 90);
        CompletableFuture<Response> lowFuture = new CompletableFuture<>();
        CompletableFuture<Response> highFuture = new CompletableFuture<>();
        when(lifecycle.register(lowPriority)).thenReturn(lowFuture);
        when(lifecycle.register(highPriority)).thenReturn(highFuture);
        when(lifecycle.claimAdmissionMutation(910L, lowFuture)).thenReturn(
                mock(AdmissionMutation.class));
        when(lifecycle.claimAdmissionMutation(911L, highFuture)).thenReturn(
                mock(AdmissionMutation.class));
        when(router.select(lowPriority, null)).thenReturn(
                PlacementResult.blocked(
                        PlacementKey.anyGroup(RoleType.PREFILL)));
        when(router.select(highPriority, null)).thenReturn(
                PlacementResult.rejected(RequestRegistry.buildErrorResponse(
                        StrategyErrorType.NO_PREFILL_WORKER, null)));

        RequestScheduler scheduler = new RequestScheduler(
                configService,
                router,
                mock(EndpointRegistry.class),
                mock(BatchSchedulerReporter.class),
                evictionManager,
                lifecycle,
                new PlacementAvailability());
        try {
            scheduler.submit(lowPriority);
            verify(router, timeout(1_000)).select(lowPriority, null);

            scheduler.submit(highPriority);
            verify(router, timeout(1_000)).select(highPriority, null);
            assertFalse(lowFuture.isDone());
        } finally {
            scheduler.closePlacement();
        }
    }

    @Test
    void expiredHeadDoesNotConsumeTheCapacityOpportunity() {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.useFifoQueue(config);
        // This contract isolates the ordered-head retry semantics, so it asks
        // for a one-request planning frontier explicitly. A wider frontier may
        // speculatively prepare a bounded suffix, which is covered by the
        // planning-frontier tests instead.
        SchedulingTestConfig.useFixedWindowDecision(config).setMaxRequests(1);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        DefaultRouter router = mockRouter();
        RequestRegistry lifecycle =
                mock(RequestRegistry.class);
        PlacementAvailability availability = new PlacementAvailability();
        PlacementKey blocker = PlacementKey.anyGroup(RoleType.PREFILL);

        BalanceContext expired = context(801L);
        BalanceContext follower = context(802L);
        CompletableFuture<Response> expiredFuture = new CompletableFuture<>();
        CompletableFuture<Response> followerFuture = new CompletableFuture<>();
        when(lifecycle.register(expired)).thenReturn(expiredFuture);
        when(lifecycle.register(follower)).thenReturn(followerFuture);
        when(lifecycle.claimAdmissionMutation(801L, expiredFuture)).thenReturn(
                mock(AdmissionMutation.class));
        when(lifecycle.claimAdmissionMutation(802L, followerFuture)).thenReturn(
                mock(AdmissionMutation.class));
        when(router.select(any())).thenReturn(
                PlacementResult.blocked(blocker));

        RequestScheduler scheduler = new RequestScheduler(
                configService,
                router,
                mock(EndpointRegistry.class),
                mock(BatchSchedulerReporter.class),
                mock(EvictionManager.class),
                lifecycle,
                availability);
        scheduler.submit(expired);
        scheduler.submit(follower);

        // Let the ordered head establish its blocked state before publishing
        // the capacity edge. This mirrors the immutable deadline metadata
        // used by production ingress and removes a test-only race between
        // enqueue and the decision thread.
        verify(router, timeout(1_000)).select(expired, null);
        expired.setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() - 1L));
        availability.capacityChanged(blocker);

        // The global queue makes one ordered decision at a time.  The
        // follower is not speculatively routed behind a blocked head; once
        // the expired head is removed it receives its single fresh attempt.
        verify(router, timeout(1_000).times(1)).select(follower, null);
        verify(router, times(1)).select(expired, null);
        scheduler.closePlacement();
    }

    @Test
    void endpointConflictAllowsIndependentSuffixCommit() throws Exception {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.useFifoQueue(config);
        SchedulingTestConfig.useFixedWindowDecision(config)
                .setMaxRequests(2);
        config.queueScheduler().getDecision().setMaxCollectionWaitMs(0L);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        DefaultRouter router = mockRouter();
        RequestRegistry lifecycle = mock(RequestRegistry.class);
        PlacementAvailability availability = new PlacementAvailability();

        BalanceContext blocked = context(803L);
        BalanceContext independent = context(804L);
        CompletableFuture<Response> blockedFuture = new CompletableFuture<>();
        CompletableFuture<Response> independentFuture = new CompletableFuture<>();
        when(lifecycle.register(blocked)).thenReturn(blockedFuture);
        when(lifecycle.register(independent)).thenReturn(independentFuture);
        when(lifecycle.claimAdmissionMutation(803L, blockedFuture)).thenReturn(
                mock(AdmissionMutation.class));
        when(lifecycle.claimAdmissionMutation(804L, independentFuture)).thenReturn(
                mock(AdmissionMutation.class));

        PrefillEndpoint fullEndpoint = mockPrefillEndpoint("full-prefill", 8080);
        PrefillEndpoint availableEndpoint = mock(PrefillEndpoint.class);
        RouteAdmission blockedRoute = mock(RouteAdmission.class);
        RouteAdmission independentRoute = mock(RouteAdmission.class);
        ScheduledRequest independentItem = mock(ScheduledRequest.class);
        PlacementKey exactBlocker = PlacementKey.exact(
                RoleType.PREFILL, "g1", "full-prefill:8080");
        CountDownLatch blockedRouteAttempts = new CountDownLatch(2);
        when(independentItem.prefillEp()).thenReturn(availableEndpoint);
        when(router.select(blocked, null)).thenAnswer(invocation -> {
            blockedRouteAttempts.countDown();
            return PlacementResult.success(blockedRoute);
        });
        when(router.select(independent, null)).thenReturn(
                PlacementResult.success(independentRoute));
        when(blockedRoute.tryEnqueue(blocked, blockedFuture, lifecycle))
                .thenReturn(PlacementResult.blocked(exactBlocker));
        when(blockedRoute.blockedEndpoint()).thenReturn(fullEndpoint);
        when(independentRoute.prefillEndpoint())
                .thenReturn(availableEndpoint);
        when(independentRoute.tryEnqueue(
                independent, independentFuture, lifecycle)).thenReturn(
                        PlacementResult.success(independentItem));

        RequestScheduler scheduler = new RequestScheduler(
                configService,
                router,
                mock(EndpointRegistry.class),
                mock(BatchSchedulerReporter.class),
                mock(EvictionManager.class),
                lifecycle,
                availability);
        try {
            scheduler.submit(blocked);
            scheduler.submit(independent);

            verify(blockedRoute, timeout(1_000).times(1))
                    .tryEnqueue(blocked, blockedFuture, lifecycle);
            verify(independentRoute, timeout(1_000))
                    .tryEnqueue(independent, independentFuture, lifecycle);
            assertFalse(blockedFuture.isDone());

            availability.capacityChanged(
                    new PlacementKey(RoleType.PREFILL, "g1"));
            assertFalse(blockedRouteAttempts.await(100, TimeUnit.MILLISECONDS),
                    "a group-wide edge must not release an exact endpoint blocker");
            availability.capacityChanged(exactBlocker);
            assertTrue(blockedRouteAttempts.await(1, TimeUnit.SECONDS));
        } finally {
            scheduler.closePlacement();
        }
    }

    @Test
    void endpointConflictNeverLetsSameEndpointSuffixOvertake() throws Exception {
        try (CapacityFixture fixture = new CapacityFixture(0, false)) {
            fixture.submitBlockedRequests();
            assertEquals(List.of(), fixture.admitted);
            assertEquals(1, fixture.requests.get(0).attempts.get());
            assertEquals(0, fixture.requests.get(1).attempts.get());
            assertEquals(0, fixture.requests.get(2).attempts.get());

            fixture.releaseSlots(1);
            fixture.requests.get(0).future.get(5, TimeUnit.SECONDS);
            assertTrue(fixture.requests.get(1).blockedAttempt.await(5, TimeUnit.SECONDS),
                    "the next waiter must confirm that the single released slot was consumed");
            fixture.awaitIndependentCommit();
            assertEquals(List.of(820L), fixture.admitted);
            assertEquals(0, fixture.availableSlots.get());
            assertEquals(1, fixture.requests.get(1).attempts.get());
            assertEquals(0, fixture.requests.get(2).attempts.get(),
                    "the suffix must remain parked behind the first confirming miss");

            // Another ingress wakeup does not grant capacity or retry the blocked head.
            fixture.awaitIndependentCommit();
            assertEquals(1, fixture.requests.get(1).attempts.get());
            assertEquals(0, fixture.requests.get(2).attempts.get());

            fixture.releaseSlots(2);
            fixture.awaitAllPublished();
            assertEquals(List.of(820L, 821L, 822L), fixture.admitted);
            assertEquals(0, fixture.availableSlots.get());
        }
    }

    @Test
    void oneCapacityEventAdmitsAllReleasedSlotsInFifoOrder() throws Exception {
        assertReleasedCapacityAdmitsWaiters(3, 0);
    }

    @Test
    void capacityReleasedDuringPublicationAdmitsWaitersWithoutAnotherEvent() throws Exception {
        assertReleasedCapacityAdmitsWaiters(1, 2);
    }

    @Test
    void capacityReleasedBeforeBlockedReturnRetriesActiveRequestWithoutLosingWakeup() throws Exception {
        try (CapacityFixture fixture = new CapacityFixture(0, false)) {
            fixture.submitBlockedRequests();
            CapacityRequest head = fixture.requests.get(0);
            head.beforeBlockedReturn = () -> fixture.releaseSlots(1);

            // Activate the head while the endpoint is still full. Its failed
            // admission publishes a release after reading capacity but before park.
            fixture.availability.capacityChanged(fixture.key);
            head.future.get(5, TimeUnit.SECONDS);
            assertTrue(fixture.requests.get(1).blockedAttempt.await(5, TimeUnit.SECONDS));
            fixture.awaitIndependentCommit();

            assertEquals(List.of(820L), fixture.admitted);
            assertEquals(0, fixture.availableSlots.get());
            assertEquals(3, head.attempts.get(),
                    "the stale full observation must trigger a fresh successful admission");
            assertEquals(1, fixture.requests.get(1).attempts.get());
            assertEquals(0, fixture.requests.get(2).attempts.get());
            fixture.awaitIndependentCommit();
            assertEquals(1, fixture.requests.get(1).attempts.get(),
                    "the confirming miss must stop the chain until another capacity edge");
        }
    }

    @Test
    void activeRetryRoutesAcrossFleetWithoutCascadingWhenSourceCapacityIsConsumed() throws Exception {
        try (CapacityFixture fixture = new CapacityFixture(0, false,
                new CompletableFuture<>(), 1)) {
            WorkerEndpoint.GenerationPin pin = mock(WorkerEndpoint.GenerationPin.class);
            when(pin.endpoint()).thenReturn(fixture.endpoint);
            when(fixture.endpoint.tryPinGeneration()).thenReturn(pin);
            when(fixture.endpoint.canAcceptRequest()).thenAnswer(
                    invocation -> fixture.availableSlots.get() > 0);
            fixture.submitBlockedRequests();

            PrefillEndpoint otherFullEndpoint = mockPrefillEndpoint("other-full-prefill", 8080);
            PlacementKey otherKey = PlacementKey.exact(
                    RoleType.PREFILL, "g1", "other-full-prefill:8080");
            List<RouteAdmission> otherRoutes = new ArrayList<>();
            CountDownLatch confirmingMiss = new CountDownLatch(1);
            for (CapacityRequest request : fixture.requests.subList(1, 3)) {
                RouteAdmission otherRoute = mock(RouteAdmission.class);
                otherRoutes.add(otherRoute);
                when(otherRoute.prefillEndpoint()).thenReturn(otherFullEndpoint);
                when(otherRoute.blockedEndpoint()).thenReturn(otherFullEndpoint);
                when(otherRoute.tryEnqueue(request.context, request.future, fixture.lifecycle))
                        .thenAnswer(invocation -> {
                            confirmingMiss.countDown();
                            return PlacementResult.blocked(otherKey);
                        });
                doAnswer(invocation -> PlacementResult.success(
                        fixture.availableSlots.get() > 0 ? request.route : otherRoute))
                        .when(fixture.router).select(request.context, null);
            }

            CapacityRequest head = fixture.requests.get(0);
            CapacityRequest second = fixture.requests.get(1);
            CapacityRequest third = fixture.requests.get(2);
            fixture.releaseSlots(1);
            head.future.get(5, TimeUnit.SECONDS);
            assertTrue(confirmingMiss.await(5, TimeUnit.SECONDS),
                    "the next active waiter must route again even after the source becomes full");
            fixture.awaitIndependentCommit();
            assertEquals(List.of(820L), fixture.admitted);
            assertEquals(0, fixture.availableSlots.get());
            verify(fixture.router, times(2)).select(head.context, null);
            verify(fixture.router, times(2)).select(second.context, null);
            verify(otherRoutes.get(0), times(1))
                    .tryEnqueue(second.context, second.future, fixture.lifecycle);
            verify(fixture.router, times(1)).select(third.context, null);
            verify(otherRoutes.get(1), never())
                    .tryEnqueue(third.context, third.future, fixture.lifecycle);
            assertEquals(0, second.attempts.get(),
                    "the confirming retry selected another endpoint without publishing to the full source");
            assertEquals(0, third.attempts.get(),
                    "the source's remaining suffix must stay parked after one confirming retry");
            assertFalse(second.future.isDone());
            assertFalse(third.future.isDone());

            // Unrelated ingress cannot restart either exact endpoint's retry chain.
            fixture.awaitIndependentCommit();
            verify(fixture.router, times(2)).select(second.context, null);
            verify(fixture.router, times(1)).select(third.context, null);

            // The second waiter now owns the other endpoint's blocker. Its exact
            // edge must wake it before the source's suffix consumes the new slots.
            fixture.availableSlots.addAndGet(2);
            fixture.availability.capacityChanged(otherKey);
            second.future.get(5, TimeUnit.SECONDS);
            fixture.awaitIndependentCommit();
            assertEquals(List.of(820L, 821L), fixture.admitted);
            assertEquals(1, fixture.availableSlots.get());
            verify(fixture.router, times(3)).select(second.context, null);
            verify(fixture.router, times(1)).select(third.context, null);
            assertFalse(third.future.isDone());

            fixture.availability.capacityChanged(fixture.key);
            fixture.awaitAllPublished();
            assertEquals(List.of(820L, 821L, 822L), fixture.admitted);
            assertEquals(0, fixture.availableSlots.get());
            verify(fixture.router, times(3)).select(second.context, null);
            verify(fixture.router, times(2)).select(third.context, null);
        }
    }

    @Test
    void cancellingActiveRetryHandsUnusedCapacityToNextWaiter() throws Exception {
        try (CapacityFixture fixture = new CapacityFixture(0, true)) {
            fixture.submitBlockedRequests();
            fixture.releaseSlots(1);
            assertTrue(fixture.headRetryStarted.await(5, TimeUnit.SECONDS));

            assertTrue(fixture.requests.get(0).future.cancel(false));
            fixture.allowHeadRetry.countDown();
            fixture.requests.get(1).future.get(5, TimeUnit.SECONDS);
            assertTrue(fixture.requests.get(2).blockedAttempt.await(5, TimeUnit.SECONDS));
            fixture.awaitIndependentCommit();

            assertEquals(List.of(821L), fixture.admitted);
            assertEquals(0, fixture.availableSlots.get());
            assertEquals(1, fixture.requests.get(0).attempts.get(),
                    "the cancelled active request must not consume the released slot");
            assertEquals(1, fixture.requests.get(2).attempts.get());
        }
    }

    @Test
    void completedActiveRetryHandsOffWithoutItsDelayedCompletionCallback() throws Exception {
        ConcurrentLinkedQueue<Runnable> callbacks = new ConcurrentLinkedQueue<>();
        CompletableFuture<Response> headFuture = new CompletableFuture<>() {
            @Override
            public CompletableFuture<Response> whenComplete(
                    BiConsumer<? super Response, ? super Throwable> action) {
                return super.whenCompleteAsync(action, callbacks::add);
            }
        };
        try (CapacityFixture fixture = new CapacityFixture(0, false, headFuture, 1)) {
            fixture.submitBlockedRequests();
            PrefillEndpoint independent = mockPrefillEndpoint("independent", 8080);
            CapacityRequest first = fixture.createRequest(840L, independent);
            CountDownLatch publicationStarted = new CountDownLatch(1);
            CountDownLatch allowPublication = new CountDownLatch(1);
            first.beforePublication = () -> {
                publicationStarted.countDown();
                assertTrue(allowPublication.await(5, TimeUnit.SECONDS));
                return null;
            };
            try {
                fixture.scheduler.submit(first.context);
                assertTrue(publicationStarted.await(5, TimeUnit.SECONDS));
                fixture.releaseSlots(1);
                assertTrue(headFuture.complete(new Response()));
                assertEquals(1, callbacks.size(), "the completed active request's cleanup must still be deferred");

                CapacityRequest second = fixture.createRequest(841L, independent);
                fixture.scheduler.submit(second.context);
                allowPublication.countDown();
                second.future.get(5, TimeUnit.SECONDS);
                // The independent endpoint and the retry chain have separate
                // scan turns; verify the handoff without another event or callback.
                fixture.requests.get(1).future.get(5, TimeUnit.SECONDS);
                assertEquals(1, callbacks.size(), "handoff must not rely on running completion callbacks");
                Runnable callback;
                while ((callback = callbacks.poll()) != null) {
                    callback.run();
                }

                // Even a subsequent edge must not be swallowed by an active request
                // that was removed from the ordered queue during pruning.
                fixture.availability.capacityChanged(fixture.key);
                fixture.requests.get(1).future.get(5, TimeUnit.SECONDS);
                assertEquals(List.of(821L), fixture.admitted);
                assertEquals(0, fixture.availableSlots.get());
                assertEquals(1, fixture.requests.get(0).attempts.get());
            } finally {
                allowPublication.countDown();
            }
        }
    }

    private void assertReleasedCapacityAdmitsWaiters(
            int initiallyReleasedSlots,
            int slotsReleasedDuringPublication) throws Exception {
        try (CapacityFixture fixture = new CapacityFixture(slotsReleasedDuringPublication, false)) {
            fixture.submitBlockedRequests();
            fixture.releaseSlots(initiallyReleasedSlots);
            fixture.awaitAllPublished();

            assertEquals(List.of(820L, 821L, 822L), fixture.admitted);
            assertEquals(0, fixture.availableSlots.get());
        }
    }

    @Test
    void staleEndpointConflictReplansWithoutFixedRetryBudget() {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.useFifoQueue(config);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        DefaultRouter router = mockRouter();
        RequestRegistry lifecycle = mock(RequestRegistry.class);
        PlacementAvailability availability = new PlacementAvailability();

        BalanceContext context = context(807L);
        CompletableFuture<Response> future = new CompletableFuture<>();
        when(lifecycle.register(context)).thenReturn(future);
        when(lifecycle.claimAdmissionMutation(807L, future)).thenReturn(
                mock(AdmissionMutation.class),
                mock(AdmissionMutation.class));

        PrefillEndpoint staleEndpoint = mockPrefillEndpoint("stale-prefill", 8080);
        RouteAdmission staleRoute = mock(RouteAdmission.class);
        RouteAdmission freshRoute = mock(RouteAdmission.class);
        ScheduledRequest committed = mock(ScheduledRequest.class);
        when(committed.prefillEp()).thenReturn(mock(PrefillEndpoint.class));
        PlacementKey exactBlocker = PlacementKey.exact(
                RoleType.PREFILL, "g1", "stale-prefill:8080");
        when(router.select(context, null)).thenReturn(
                PlacementResult.success(staleRoute),
                PlacementResult.success(freshRoute));
        when(staleRoute.tryEnqueue(context, future, lifecycle))
                .thenReturn(PlacementResult.blocked(exactBlocker));
        when(staleRoute.blockedEndpoint()).thenReturn(staleEndpoint);
        when(staleRoute.blockedEndpointChanged()).thenReturn(true);
        when(freshRoute.tryEnqueue(context, future, lifecycle))
                .thenReturn(PlacementResult.success(committed));

        RequestScheduler scheduler = new RequestScheduler(
                configService,
                router,
                mock(EndpointRegistry.class),
                mock(BatchSchedulerReporter.class),
                mock(EvictionManager.class),
                lifecycle,
                availability);
        try {
            scheduler.submit(context);

            verify(freshRoute, timeout(1_000))
                    .tryEnqueue(context, future, lifecycle);
            verify(router, times(2)).select(context, null);
        } finally {
            scheduler.closePlacement();
        }
    }

    @Test
    void terminalRouteRejectionDoesNotAcquireDecodeAcceptance() {
        Fixture fixture = new Fixture(true);

        Response response = fixture.scheduler.submit(fixture.context).join();

        assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(),
                response.getCode());
        verify(fixture.router, timeout(1_000))
                .select(fixture.context, null);
        verify(fixture.lifecycle, never())
                .commitRoute(
                        any(), any());
        fixture.scheduler.closePlacement();
    }

    @Test
    void prioritySelectorMissWaitsWithoutInventingAnEvictionRoute() {
        Fixture fixture = new Fixture(true);
        when(fixture.router.select(fixture.context, null)).thenReturn(
                PlacementResult.blocked(
                        PlacementKey.anyGroup(RoleType.PREFILL)));
        CompletableFuture<Response> waiting =
                fixture.scheduler.submit(fixture.context);

        verify(fixture.router, timeout(1_000))
                .select(fixture.context, null);
        verify(fixture.evictionManager, never())
                .tryAdmit(any(), any(), any(), any());
        assertFalse(waiting.isDone());
        verify(fixture.lifecycle, never())
                .commitRoute(
                        any(), any());
        fixture.scheduler.closePlacement();
    }

    @Test
    void fifoTemporaryDecodeMissRemainsQueued() {
        Fixture fixture = new Fixture(false);
        when(fixture.router.select(fixture.context, null)).thenReturn(
                PlacementResult.blocked(
                        PlacementKey.anyGroup(RoleType.DECODE)));
        CompletableFuture<Response> waiting =
                fixture.scheduler.submit(fixture.context);

        verify(fixture.router, timeout(1_000))
                .select(fixture.context, null);
        assertFalse(waiting.isDone());
        verify(fixture.evictionManager, never())
                .tryAdmit(any(), any(), any(), any());
        verify(fixture.lifecycle, never())
                .commitRoute(
                        any(), any());
        fixture.scheduler.closePlacement();
    }

    @Test
    void initialPlacementFailureCompletesTheRegisteredGeneration() {
        Fixture fixture = new Fixture(false);
        when(fixture.router.select(fixture.context, null))
                .thenThrow(new IllegalStateException("selector failed"));

        CompletableFuture<Response> returned =
                fixture.scheduler.submit(fixture.context);

        assertEquals(fixture.future, returned);
        assertEquals(StrategyErrorType.DISPATCH_FAILED.getErrorCode(),
                returned.join().getCode());
        fixture.scheduler.closePlacement();
    }

    private static PrefillEndpoint mockPrefillEndpoint(String ip, int port) {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        WorkerStatus status = WorkerStatus.createDiscovered(
                RoleType.PREFILL, "g1", ip, port, port, "test");
        when(endpoint.getStatus()).thenReturn(status);
        when(endpoint.ipPort()).thenReturn(status.getIpPort());
        when(endpoint.getIp()).thenReturn(ip);
        return endpoint;
    }

    private static BalanceContext context(long requestId) {
        return context(requestId, 50);
    }

    private static BalanceContext context(long requestId, int priority) {
        BalanceContext context = new BalanceContext();
        Request request = new Request();
        request.setRequestId(requestId);
        request.setPriority(priority);
        context.setRequest(request);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                priority, System.currentTimeMillis() + 60_000L));
        return context;
    }

    /** Real ordered scheduler with an exact endpoint whose free slots are explicitly controlled. */
    private static final class CapacityFixture implements AutoCloseable {
        private final DefaultRouter router = mockRouter();
        private final RequestRegistry lifecycle = mock(RequestRegistry.class);
        private final PlacementAvailability availability = new PlacementAvailability();
        private final PrefillEndpoint endpoint = mockPrefillEndpoint("capacity-prefill", 8080);
        private final PlacementKey key = PlacementKey.exact(
                RoleType.PREFILL, "g1", "capacity-prefill:8080");
        private final AtomicInteger availableSlots = new AtomicInteger();
        private final List<Long> admitted = new CopyOnWriteArrayList<>();
        private final List<CapacityRequest> requests = new ArrayList<>();
        private final ConcurrentLinkedQueue<CapacityRequest> pendingReports = new ConcurrentLinkedQueue<>();
        private final CountDownLatch followersParked = new CountDownLatch(2);
        private final CountDownLatch allPublished = new CountDownLatch(3);
        private final CountDownLatch headRetryStarted = new CountDownLatch(1);
        private final CountDownLatch allowHeadRetry = new CountDownLatch(1);
        private final int slotsReleasedDuringPublication;
        private final boolean pauseHeadRetry;
        private final CompletableFuture<Response> headFuture;
        private final RequestScheduler scheduler;
        private long nextIndependentId = 830L;

        private CapacityFixture(int slotsReleasedDuringPublication, boolean pauseHeadRetry) {
            this(slotsReleasedDuringPublication, pauseHeadRetry, new CompletableFuture<>(), 0);
        }

        private CapacityFixture(int slotsReleasedDuringPublication, boolean pauseHeadRetry,
                CompletableFuture<Response> headFuture, int plannerThreads) {
            this.slotsReleasedDuringPublication = slotsReleasedDuringPublication;
            this.pauseHeadRetry = pauseHeadRetry;
            this.headFuture = headFuture;
            FlexlbConfig config = SchedulingTestConfig.batchConfig();
            if (plannerThreads > 0) {
                config = spy(config);
                var runtime = spy(config.getInternalRuntime());
                when(runtime.getQueuePlannerThreads()).thenReturn(plannerThreads);
                when(config.getInternalRuntime()).thenReturn(runtime);
            }
            SchedulingTestConfig.useFifoQueue(config);
            SchedulingTestConfig.useSingleDecision(config);
            SchedulingTestConfig.useNonBatchDispatcher(config).setMaxInflightPerPrefillWorker(3);
            ConfigService configService = mock(ConfigService.class);
            when(configService.loadBalanceConfig()).thenReturn(config);

            BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
            doAnswer(invocation -> {
                CapacityRequest request = pendingReports.remove();
                // The coordinator reports only after retiring the queue entry. Completing
                // here cannot hide a lost handoff with an earlier completion callback.
                request.future.complete(new Response());
                if (request.primaryEndpoint) {
                    allPublished.countDown();
                }
                return null;
            }).when(reporter).reportRouteSubmitTimeMs(any(), any(), anyLong());
            for (long requestId = 820L; requestId < 823L; requestId++) {
                requests.add(createRequest(requestId, endpoint));
            }
            scheduler = new RequestScheduler(
                    configService,
                    router,
                    mock(EndpointRegistry.class),
                    reporter,
                    mock(EvictionManager.class),
                    lifecycle,
                    availability);
        }

        private CapacityRequest createRequest(long requestId, PrefillEndpoint target) {
            CapacityRequest request = new CapacityRequest(requestId, target == endpoint,
                    requestId == 820L ? headFuture : new CompletableFuture<>());
            RouteAdmission route = request.route;
            ScheduledRequest item = mock(ScheduledRequest.class);
            ServerStatus prefill = mock(ServerStatus.class);
            when(prefill.getRole()).thenReturn(RoleType.PREFILL);
            when(item.prefill()).thenReturn(prefill);
            when(item.prefillEp()).thenReturn(target);
            when(lifecycle.register(request.context)).thenReturn(request.future);
            when(lifecycle.claimAdmissionMutation(requestId, request.future))
                    .thenReturn(mock(AdmissionMutation.class));
            when(route.prefillEndpoint()).thenReturn(target);
            when(route.blockedEndpoint()).thenReturn(target);
            AtomicInteger plans = new AtomicInteger();
            when(router.select(request.context, null)).thenAnswer(invocation -> {
                if (plans.incrementAndGet() == 2 && requestId == 820L && pauseHeadRetry) {
                    headRetryStarted.countDown();
                    assertTrue(allowHeadRetry.await(5, TimeUnit.SECONDS),
                            "the cancelled active request's planning gate must be released");
                }
                return PlacementResult.success(route);
            });
            when(route.tryEnqueue(request.context, request.future, lifecycle)).thenAnswer(invocation -> {
                request.attempts.incrementAndGet();
                request.beforePublication.call();
                if (request.primaryEndpoint) {
                    int slots = availableSlots.getAndUpdate(value -> value > 0 ? value - 1 : value);
                    if (slots == 0) {
                        request.blockedAttempt.countDown();
                        Runnable hook = request.beforeBlockedReturn;
                        request.beforeBlockedReturn = null;
                        if (hook != null) {
                            hook.run();
                        }
                        return PlacementResult.blocked(key);
                    }
                    admitted.add(requestId);
                    if (requestId == 820L && slotsReleasedDuringPublication > 0) {
                        // Status reconciliation releases capacity before publication returns
                        // and before the global queue can retire this active request.
                        releaseSlots(slotsReleasedDuringPublication);
                    }
                }
                pendingReports.add(request);
                return PlacementResult.success(item);
            });
            AtomicInteger closes = new AtomicInteger();
            doAnswer(invocation -> {
                if (closes.incrementAndGet() == 1
                        && request.primaryEndpoint && requestId != 820L) {
                    // Followers close their first plan only after the endpoint
                    // conflict has parked them behind the initially full head.
                    followersParked.countDown();
                }
                return null;
            }).when(route).close();
            return request;
        }

        private void submitBlockedRequests() throws InterruptedException {
            for (CapacityRequest request : requests) {
                scheduler.submit(request.context);
            }
            assertTrue(followersParked.await(5, TimeUnit.SECONDS),
                    "all three requests must be parked before the first capacity event");
        }

        private void releaseSlots(int count) {
            availableSlots.addAndGet(count);
            availability.capacityChanged(key);
        }

        private void awaitAllPublished() throws InterruptedException {
            assertTrue(allPublished.await(5, TimeUnit.SECONDS),
                    () -> "released slots must be consumed without another event; admitted=" + admitted
                            + ", freeSlots=" + availableSlots.get());
        }

        private void awaitIndependentCommit() throws Exception {
            long requestId = nextIndependentId++;
            PrefillEndpoint independent = mockPrefillEndpoint("independent-" + requestId, 8080);
            CapacityRequest request = createRequest(requestId, independent);
            scheduler.submit(request.context);
            request.future.get(5, TimeUnit.SECONDS);
        }

        @Override
        public void close() {
            allowHeadRetry.countDown();
            scheduler.closePlacement();
        }
    }

    private static final class CapacityRequest {
        private final BalanceContext context;
        private final CompletableFuture<Response> future;
        private final AtomicInteger attempts = new AtomicInteger();
        private final RouteAdmission route = mock(RouteAdmission.class);
        private final CountDownLatch blockedAttempt = new CountDownLatch(1);
        private final boolean primaryEndpoint;
        private Callable<Void> beforePublication = () -> null;
        private Runnable beforeBlockedReturn;

        private CapacityRequest(long requestId, boolean primaryEndpoint, CompletableFuture<Response> future) {
            context = context(requestId);
            this.primaryEndpoint = primaryEndpoint;
            this.future = future;
        }
    }

    private static final class Fixture {
        private final long requestId = 701L;
        private final FlexlbConfig config = SchedulingTestConfig.batchConfig();
        private final ConfigService configService = mock(ConfigService.class);
        private final DefaultRouter router = mockRouter();
        private final EvictionManager evictionManager = mock(EvictionManager.class);
        private final RequestRegistry lifecycle =
                mock(RequestRegistry.class);
        private final PlacementAvailability availability =
                new PlacementAvailability();
        private final BalanceContext context = mock(BalanceContext.class);
        private final CompletableFuture<Response> future =
                new CompletableFuture<>();
        private final RequestScheduler scheduler;

        private Fixture(boolean priority) {
            if (priority) {
                SchedulingTestConfig.usePriorityQueue(config);
            } else {
                SchedulingTestConfig.useFifoQueue(config);
            }
            when(configService.loadBalanceConfig()).thenReturn(config);
            when(context.getRequest()).thenReturn(new Request());
            when(context.getConfig()).thenReturn(config);
            when(router.queueAdmissionRole()).thenReturn(RoleType.PREFILL);
            when(context.getRequestId()).thenReturn(requestId);
            when(lifecycle.register(context))
                    .thenReturn(future);
            when(lifecycle.claimAdmissionMutation(requestId, future)).thenReturn(
                    mock(AdmissionMutation.class));
            when(lifecycle.publishDecisionResponseAsync(
                    anyLong(), any(), any())).thenAnswer(invocation -> {
                        @SuppressWarnings("unchecked")
                        CompletableFuture<Response> responseFuture =
                                (CompletableFuture<Response>) invocation.getArgument(1);
                        responseFuture.complete(invocation.getArgument(2));
                        return true;
                    });
            when(router.select(context, null)).thenReturn(
                    PlacementResult.rejected(
                            RequestRegistry.buildErrorResponse(
                                    StrategyErrorType.NO_PREFILL_WORKER,
                                    null)));
            scheduler = new RequestScheduler(
                    configService,
                    router,
                    mock(EndpointRegistry.class),
                    mock(BatchSchedulerReporter.class),
                    evictionManager,
                    lifecycle,
                    availability);
        }
    }
}
