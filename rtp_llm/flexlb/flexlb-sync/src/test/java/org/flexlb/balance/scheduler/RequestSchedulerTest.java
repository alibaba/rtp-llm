package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.EvictionManager;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.after;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class RequestSchedulerTest {

    @Test
    void unavailableGroupMustNotStopAHealthyIndependentGroup() {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        DefaultRouter router = mock(DefaultRouter.class);
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
        when(router.resolvePolicyGroup(healthy)).thenReturn("healthy-group");
        when(router.select(unavailable, "unavailable-group")).thenReturn(PlacementResult.blocked(new PlacementKey(RoleType.DECODE, "unavailable-group")));
        RouteAdmission healthyRoute = mock(RouteAdmission.class);
        when(router.select(healthy, "healthy-group")).thenReturn(PlacementResult.success(healthyRoute));
        when(healthyRoute.tryEnqueue(healthy, second, lifecycle))
                .thenReturn(PlacementResult.success(mock(ScheduledRequest.class)));
        RequestScheduler scheduler = new RequestScheduler(service, router, endpoints, mock(BatchSchedulerReporter.class), mock(EvictionManager.class), lifecycle, new PlacementAvailability());
        try {
            scheduler.submit(unavailable);
            verify(router, timeout(500)).select(unavailable, "unavailable-group");
            scheduler.submit(healthy);
            verify(router, timeout(500)).select(healthy, "healthy-group");
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
        DefaultRouter router = mock(DefaultRouter.class);
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
        DefaultRouter router = mock(DefaultRouter.class);
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
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.ipPort()).thenReturn("127.0.0.1:8000");
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
        DefaultRouter router = mock(DefaultRouter.class);
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
        DefaultRouter router = mock(DefaultRouter.class);
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
        DefaultRouter router = mock(DefaultRouter.class);
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
        DefaultRouter router = mock(DefaultRouter.class);
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
        DefaultRouter router = mock(DefaultRouter.class);
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

        PrefillEndpoint fullEndpoint = mock(PrefillEndpoint.class);
        PrefillEndpoint availableEndpoint = mock(PrefillEndpoint.class);
        RouteAdmission blockedRoute = mock(RouteAdmission.class);
        RouteAdmission independentRoute = mock(RouteAdmission.class);
        ScheduledRequest independentItem = mock(ScheduledRequest.class);
        PlacementKey exactBlocker = PlacementKey.exact(
                RoleType.PREFILL, "g1", "full-prefill:8080");
        CountDownLatch blockedRouteAttempts = new CountDownLatch(2);
        when(independentItem.prefillEp()).thenReturn(availableEndpoint);
        when(fullEndpoint.ipPort()).thenReturn("full-prefill:8080");
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
    void endpointConflictNeverLetsSameEndpointSuffixOvertake() {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.useFifoQueue(config);
        config.queueScheduler().getDecision().setMaxCollectionWaitMs(0L);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        DefaultRouter router = mock(DefaultRouter.class);
        when(router.queueAdmissionRole()).thenReturn(RoleType.PREFILL);
        EndpointRegistry endpointRegistry = mock(EndpointRegistry.class);
        when(endpointRegistry.getEndpointCount(RoleType.PREFILL)).thenReturn(1);
        RequestRegistry lifecycle = mock(RequestRegistry.class);
        PlacementAvailability availability = new PlacementAvailability();

        BalanceContext older = context(805L);
        BalanceContext younger = context(806L);
        CompletableFuture<Response> olderFuture = new CompletableFuture<>();
        CompletableFuture<Response> youngerFuture = new CompletableFuture<>();
        when(lifecycle.register(older)).thenReturn(olderFuture);
        when(lifecycle.register(younger)).thenReturn(youngerFuture);
        when(lifecycle.claimAdmissionMutation(805L, olderFuture)).thenReturn(
                mock(AdmissionMutation.class));
        when(lifecycle.claimAdmissionMutation(806L, youngerFuture)).thenReturn(
                mock(AdmissionMutation.class));

        PrefillEndpoint fullEndpoint = mock(PrefillEndpoint.class);
        RouteAdmission olderRoute = mock(RouteAdmission.class);
        RouteAdmission youngerRoute = mock(RouteAdmission.class);
        ScheduledRequest olderItem = mock(ScheduledRequest.class);
        PlacementKey exactBlocker = PlacementKey.exact(
                RoleType.PREFILL, "g1", "full-prefill:8080");
        when(router.select(older, null)).thenReturn(
                PlacementResult.success(olderRoute));
        when(router.select(younger, null)).thenReturn(
                PlacementResult.success(youngerRoute));
        when(olderRoute.tryEnqueue(older, olderFuture, lifecycle))
                .thenReturn(
                        PlacementResult.blocked(exactBlocker),
                        PlacementResult.success(olderItem));
        when(olderRoute.blockedEndpoint()).thenReturn(fullEndpoint);
        when(fullEndpoint.ipPort()).thenReturn("full-prefill:8080");
        when(olderRoute.prefillEndpoint()).thenReturn(fullEndpoint);
        when(youngerRoute.prefillEndpoint()).thenReturn(fullEndpoint);
        when(olderItem.prefillEp()).thenReturn(fullEndpoint);

        RequestScheduler scheduler = new RequestScheduler(
                configService,
                router,
                endpointRegistry,
                mock(BatchSchedulerReporter.class),
                mock(EvictionManager.class),
                lifecycle,
                availability);
        try {
            scheduler.submit(older);
            scheduler.submit(younger);

            verify(olderRoute, timeout(1_000).times(1))
                    .tryEnqueue(older, olderFuture, lifecycle);
            verify(youngerRoute, after(100).never())
                    .tryEnqueue(younger, youngerFuture, lifecycle);
            assertFalse(olderFuture.isDone());
            assertFalse(youngerFuture.isDone());

            availability.capacityChanged(exactBlocker);
            verify(olderRoute, timeout(1_000).times(2))
                    .tryEnqueue(older, olderFuture, lifecycle);
            verify(youngerRoute, after(100).never())
                    .tryEnqueue(younger, youngerFuture, lifecycle);
        } finally {
            scheduler.closePlacement();
        }
    }

    @Test
    void staleEndpointConflictReplansWithoutFixedRetryBudget() {
        FlexlbConfig config = SchedulingTestConfig.batchConfig();
        SchedulingTestConfig.useFifoQueue(config);
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        DefaultRouter router = mock(DefaultRouter.class);
        RequestRegistry lifecycle = mock(RequestRegistry.class);
        PlacementAvailability availability = new PlacementAvailability();

        BalanceContext context = context(807L);
        CompletableFuture<Response> future = new CompletableFuture<>();
        when(lifecycle.register(context)).thenReturn(future);
        when(lifecycle.claimAdmissionMutation(807L, future)).thenReturn(
                mock(AdmissionMutation.class),
                mock(AdmissionMutation.class));

        PrefillEndpoint staleEndpoint = mock(PrefillEndpoint.class);
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

    private static final class Fixture {
        private final long requestId = 701L;
        private final FlexlbConfig config = SchedulingTestConfig.batchConfig();
        private final ConfigService configService = mock(ConfigService.class);
        private final DefaultRouter router = mock(DefaultRouter.class);
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
