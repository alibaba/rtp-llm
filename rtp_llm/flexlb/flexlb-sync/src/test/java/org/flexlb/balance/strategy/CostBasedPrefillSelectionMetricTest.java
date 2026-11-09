package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.QueueOrderingConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusTestSupport;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.ArgumentCaptor;
import org.mockito.Mockito;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class CostBasedPrefillSelectionMetricTest {
    private static final int FULL_FLEET_SIZE = 750;

    private FlexlbConfig config;
    private EndpointRegistry registry;
    private CacheAwareService cache;
    private EngineHealthReporter reporter;
    private CacheMetricsReporter cacheReporter;
    private CostBasedPrefillStrategy strategy;
    private RequestContext context;

    @BeforeEach
    void setUp() {
        config = StrategyTestSupport.config();
        config.setDispatcher(ConfigTestSupport.nonBatch());
        config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator()
                .setExpression("sum(computeTokens)");
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        registry = StrategyTestSupport.endpointRegistry(configService);
        publish("10.0.0.1", 8080);

        cache = mock(CacheAwareService.class);
        when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of());
        reporter = mock(EngineHealthReporter.class);
        cacheReporter = mock(CacheMetricsReporter.class);
        strategy = new CostBasedPrefillStrategy(
                registry, cache, reporter, cacheReporter, SchedulingTestConfig.configService(config));

        Request request = Request.builder()
                .requestId(10_001L)
                .seqLen(1_000L)
                .rawPriority(50)
                .blockCacheKeys(List.of()).build();
        context = new RequestContext(config);
        context.data().setRequest(request);
        context.data().setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() + 60_000L));
    }

    @AfterEach
    void tearDown() {
        registry.close();
    }

    @Test
    void selectionDoesNotOwnTheRequestDeadline() {
        context.data().setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() - 1L));

        try (RequestRoute ignored = select()) {
            assertTrue(ignored.prefill().isSuccess());
        }
    }

    @ParameterizedTest
    @CsvSource({"5,1,false,BLOCKED", "50,1,false,BLOCKED", "5,2,false,SUCCESS",
            "50,1,true,SUCCESS", "10,1,true,BLOCKED", "5,1,true,BLOCKED"})
    void fullRequestCapacityIsSelectableOnlyWithEligibleQueuedPreemption(
            int incomingPriority, int requestLimit, boolean preemptQueued, PlacementResult.Status expectedStatus) {
        registry.close();
        config.queueScheduler().getDecision().setMaxRequests(2);
        config.getDispatcher().setMaxInflightPerPrefillWorker(requestLimit);
        config.queueScheduler().getDecision().setMaxCollectionWaitMs(60_000L);
        QueueOrderingConfig ordering = ConfigTestSupport.priority();
        ordering.setPreemption(null);
        if (preemptQueued) {
            PreemptionConfig preemption = new PreemptionConfig();
            preemption.setAllowedVictimStages(Set.of(VictimStage.PREFILL_QUEUED));
            ordering.setPreemption(preemption);
        }
        config.queueScheduler().setOrdering(ordering);
        refreshContext();
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        registry = StrategyTestSupport.endpointRegistry(service);
        publish("10.0.0.1", 8080);
        strategy = new CostBasedPrefillStrategy(registry, cache, reporter, cacheReporter, SchedulingTestConfig.configService(config));
        PrefillEndpoint endpoint = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.1:8080");
        Request queuedRequest = Request.builder()
                .requestId(9_001L)
                .seqLen(1_000L)
                .rawPriority(10).build();
        RequestContext queuedContext = new RequestContext(config);
        queuedContext.data().setRequest(queuedRequest);
        queuedContext.data().setSchedulingMetadata(SchedulingMetadata.explicit(
                10, System.currentTimeMillis() + 120_000L));
        queuedContext.data().setFuture(new CompletableFuture<>());
        RequestRoute queued = SchedulingTestConfig.createRoute(queuedContext,
                null, null, null, endpoint, null, null, System.currentTimeMillis());
        assertTrue(StrategyTestSupport.offer(endpoint, queued));
        context.data().setSchedulingMetadata(SchedulingMetadata.explicit(
                incomingPriority, System.currentTimeMillis() + 120_000L));

        PlacementResult<RequestRoute, RoleType> result = StrategyTestSupport.select(strategy, context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, null);

        assertEquals(expectedStatus, result.status());
        if (expectedStatus == PlacementResult.Status.BLOCKED) {
            verify(cache, Mockito.never()).findMatchingEngines(any(), any(), any());
        }
        if (result.status() == PlacementResult.Status.SUCCESS) {
            try (RequestRoute selected = result.value()) {
                assertEquals("10.0.0.1", selected.prefill().getServerIp());
                assertTrue(selected.prefillWorkMs() > 0L);
            }
        }
        assertEquals(1, endpoint.captureRouteProjectionInputs().queue().activeItems().size(),
                "selection must not evict a lower-priority queued request");
    }

    @ParameterizedTest
    @CsvSource({"false,NON_BATCH", "true,BATCH"})
    void everyDeliveryModeReportsItsSelectionEstimates(
            boolean batchDelivery, String deliveryMode) {
        config.getDispatcher().setType(DispatcherConfig.Type.valueOf(deliveryMode));
        refreshContext();

        try (RequestRoute selected = select()) {
            ArgumentCaptor<Long> ttft = ArgumentCaptor.forClass(Long.class);
            ArgumentCaptor<Long> execution = ArgumentCaptor.forClass(Long.class);
            verify(reporter).reportPrefillSelectedEstimates(
                    Mockito.eq(RoleType.PREFILL), Mockito.eq("10.0.0.1"),
                    Mockito.eq(deliveryMode), ttft.capture(), execution.capture());
            assertEquals(selected.prefill().getPrefillTime(), ttft.getValue());
            assertEquals(selected.prefillWorkMs(), execution.getValue());
        }
    }

    @Test
    void telemetryFailureNeverChangesAValidSelection() {
        doThrow(new IllegalStateException("metrics unavailable"))
                .when(reporter).reportPrefillSelectedEstimates(
                        any(), any(), any(), Mockito.anyLong(), Mockito.anyLong());

        try (RequestRoute selected = select()) {
            assertTrue(selected.prefill().isSuccess());
            assertEquals("10.0.0.1", selected.prefill().getServerIp());
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"cacheHit", "selectedCache", "candidateCache"})
    void failedSelectionReportingClosesTheSelectedGeneration(String metric) {
        IllegalStateException failure = new IllegalStateException("metrics unavailable");
        switch (metric) {
            case "cacheHit" -> doThrow(failure).when(cacheReporter)
                    .reportCacheHitMetrics(any(), Mockito.anyLong(), Mockito.anyDouble());
            case "selectedCache" -> doThrow(failure).when(cacheReporter)
                    .reportRoutingSelectedCacheMatchMetrics(any(), Mockito.anyLong(), Mockito.anyLong());
            case "candidateCache" -> doThrow(failure).when(cacheReporter)
                    .reportRoutingCandidateMaxCacheMatchMetrics(any(), Mockito.anyLong());
            default -> throw new AssertionError(metric);
        }
        var endpoint = registry.get(RoleType.PREFILL, "10.0.0.1:8080");
        var pin = Mockito.spy(endpoint.tryPinGeneration());
        var directory = Mockito.spy(registry);
        Mockito.doReturn(pin).when(directory).capture(RoleType.PREFILL, "10.0.0.1:8080");
        strategy = new CostBasedPrefillStrategy(directory, cache, reporter, cacheReporter, SchedulingTestConfig.configService(config));
        try {
            org.junit.jupiter.api.Assertions.assertSame(failure,
                    org.junit.jupiter.api.Assertions.assertThrows(IllegalStateException.class, this::select));
            verify(pin).close();
        } finally {
            pin.close();
        }
    }

    @Test
    void cacheLeaderInsideTtftCapOverridesTheBaselineCandidate() {
        configureAffinity(600L, 5.0);

        try (RequestRoute selected = select()) {
            assertEquals("10.0.0.2", selected.prefill().getServerIp());
            verify(cacheReporter).reportCacheAffinityDecision(
                    RoleType.PREFILL, "10.0.0.2", "CACHE_LEADER");
        }
    }

    @Test
    void cacheLeaderAtEndOfFullFleetRemainsEligible() {
        configureAffinity(600L, 5.0);
        String cacheLeader = null;
        for (int index = 2; index < FULL_FLEET_SIZE; index++) {
            String ip = "10.2." + index / 250 + '.' + index % 250;
            publish(ip, 8080);
            cacheLeader = ip;
        }
        when(cache.findMatchingEngines(any(), any(), any()))
                .thenReturn(Map.of(cacheLeader + ":8080", 5));

        try (RequestRoute selected = select()) {
            assertEquals(cacheLeader, selected.prefill().getServerIp(),
                    "cache-first must inspect the complete 750-node fleet");
        }
    }

    @Test
    void exactCostTiesRotateWithoutShrinkingTheCandidateFleet() {
        publish("10.0.0.2", 8080);
        publish("10.0.0.3", 8080);

        Set<String> selectedIps = new HashSet<>();
        for (int index = 0; index < 3; index++) {
            context.data().setRequest(context.data().getRequest().toBuilder().requestId(30_000L + index).build());
            try (RequestRoute selected = select()) {
                selectedIps.add(selected.prefill().getServerIp());
            }
        }

        assertEquals(Set.of("10.0.0.1", "10.0.0.2", "10.0.0.3"), selectedIps);
    }

    @ParameterizedTest
    @CsvSource({"400,5.0,OVER_CAP", "600,60.0,LOW_CACHE_HIT"})
    void cacheAffinityGateUsesTheConfiguredBaselineCandidate(
            long maxExtraTtftMs, double minPrefixHitPercent, String reason) {
        configureAffinity(maxExtraTtftMs, minPrefixHitPercent);

        try (RequestRoute selected = select()) {
            assertEquals("10.0.0.1", selected.prefill().getServerIp());
            verify(cacheReporter).reportCacheAffinityDecision(
                    RoleType.PREFILL, "10.0.0.1", reason);
        }
    }

    @Test
    void equalMaximumCacheHitPreservesBestOnlySelection() {
        configureFocusedAffinity();
        PrefillCandidateSet candidates = candidates(
                new long[] {100L, 105L, 1_000L},
                new long[] {500L, 500L, 0L});

        int selectedIndex = strategy.selectBestCandidate(candidates, 100L, RoleType.PREFILL, null, 1_000L);

        assertEquals(0, selectedIndex,
                "equal cache hit must preserve the minimum-TTFT baseline");
        verify(cacheReporter).reportCacheAffinityDecision(
                RoleType.PREFILL,
                "10.1.0." + (selectedIndex + 1),
                "NO_CACHE_LEAD");
    }

    @Test
    void cacheAffinityBudgetUpdateAndRestoreReachTheExistingSelector() {
        configureFocusedAffinity();
        PrefillCandidateSet candidates = candidates(
                new long[] {100L, 105L}, new long[] {500L, 600L});
        var affinity = config.getRouter().getRoles().getPrefill().getCacheAffinity();
        assertEquals(1, strategy.selectBestCandidate(candidates, 100L, RoleType.PREFILL, null, 1_000L));
        affinity.setMaxExtraTtftMs(0L);
        assertEquals(0, strategy.selectBestCandidate(candidates, 100L, RoleType.PREFILL, null, 1_000L));
        affinity.setMaxExtraTtftMs(10L);
        assertEquals(1, strategy.selectBestCandidate(candidates, 100L, RoleType.PREFILL, null, 1_000L));
    }

    @Test
    void strictlyGreaterCacheHitOverridesTheBaselineCandidate() {
        configureFocusedAffinity();
        PrefillCandidateSet candidates = candidates(
                new long[] {100L, 105L, 1_000L},
                new long[] {500L, 600L, 0L});

        int selectedIndex = strategy.selectBestCandidate(candidates, 100L, RoleType.PREFILL, null, 1_000L);

        assertEquals(1, selectedIndex);
        verify(cacheReporter).reportCacheAffinityDecision(
                RoleType.PREFILL, "10.1.0.2", "CACHE_LEADER");
    }

    @Test
    void cacheLeaderRemainsPreferredAcrossRepeatedSelections() {
        configureAffinity(600L, 5.0);

        try (RequestRoute selected = select()) {
            assertEquals("10.0.0.2", selected.prefill().getServerIp());
        }
        context.data().setRequest(context.data().getRequest().toBuilder().requestId(20_002L).build());

        try (RequestRoute selected = select()) {
            assertEquals("10.0.0.2", selected.prefill().getServerIp());
        }
        verify(cacheReporter, times(2))
                .reportCacheAffinityDecision(
                        RoleType.PREFILL, "10.0.0.2", "CACHE_LEADER");
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void unknownEngineTokenDemandDoesNotPreventCountBasedAdmission(boolean missingTokenDemand) {
        var endpoint = (PrefillEndpoint)
                registry.get(RoleType.PREFILL, "10.0.0.1:8080");
        var status = endpoint.getStatus();
        var response = StrategyTestSupport.response(RoleType.PREFILL, true,
                1_000_000L, 1_000_000L, status.appliedStatusCursor().statusVersion() + 1L);
        var task = new TaskInfo();
        task.setRequestId(1L);
        task.setPhase(TaskPhase.RUNNING);
        task.setInputLength(missingTokenDemand ? 0L : 100L);
        response.setRunningQueryLen(1L);
        response.setRunningTaskInfo(Map.of("1", task));
        status.lock.lock();
        try {
            endpoint.publishStatus(endpoint.applyPreparedStatus(status, status.prepareNewStatus(WorkerStatusTestSupport.freezeStatusResponse(status, response))));
        } finally {
            status.lock.unlock();
        }
        assertTrue(endpoint.captureRouteProjectionInputs().work().hasUnknownWork());

        context.data().setRequest(context.data().getRequest().toBuilder().requestId(40_001L).build());
        var result = StrategyTestSupport.select(strategy, context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, null);
        assertEquals(PlacementResult.Status.SUCCESS, result.status());
        try (RequestRoute selected = result.value()) {
            assertEquals("10.0.0.1", selected.prefill().getServerIp());
        }
    }

    @Test
    void sparseCacheLeadersAcrossTheFleetRotateOnlyAmongExactTies() {
        configureFocusedAffinity();
        long[] ttft = new long[130];
        long[] hits = new long[130];
        java.util.Arrays.fill(ttft, 105L);
        ttft[0] = 100L;
        hits[12] = 550L;
        hits[63] = 700L;
        ttft[63] = 111L;
        hits[64] = 600L;
        hits[129] = 600L;
        PrefillCandidateSet candidates = candidates(ttft, hits);

        for (int expected : new int[] {129, 64, 129, 64}) {
            assertEquals(expected, strategy.selectBestCandidate(candidates, 100L, RoleType.PREFILL, null, 1_000L));
        }
    }

    @Test
    void saturatedAffinityTiesExcludeUnknownWorkWithTheSameRank() {
        configureFocusedAffinity();
        PrefillCandidateSet candidates = candidates(
                new long[] {Long.MAX_VALUE - 5L, Long.MAX_VALUE}, new long[] {100L, 600L});
        PrefillEndpoint unknown = mock(PrefillEndpoint.class);
        candidates.addCandidate("unknown:8080", unknown,
                new RouteProjection.Candidate(RouteProjection.Candidate.State.UNMODELED_ENGINE_WORK,
                        RouteProjection.Candidate.UNKNOWN, 1L,
                        RouteProjection.Candidate.InitialHeadDisposition.NONE, "", null, 600L, 600L), 3L);

        for (int i = 0; i < 4; i++) {
            assertEquals(1, strategy.selectBestCandidate(candidates, Long.MAX_VALUE - 5L,
                    RoleType.PREFILL, null, 1_000L));
        }
        verify(cacheReporter, times(4)).reportCacheAffinityDecision(
                RoleType.PREFILL, "10.1.0.2", "CACHE_LEADER");
    }

    @Test
    void tokenBalanceSkipsCountFullWorkerEvenWithLowEngineQueue() {
        registry.close();
        config.getDispatcher().setMaxInflightPerPrefillWorker(2);
        config.getRouter().getRoles().getPrefill().setTokenBalance(new RoutingConfig.TokenBalanceConfig());
        registry = StrategyTestSupport.endpointRegistry(SchedulingTestConfig.configService(config));
        publish("10.0.0.1", 8080); publish("10.0.0.2", 8080);
        strategy = new CostBasedPrefillStrategy(registry, cache, reporter, cacheReporter,
                SchedulingTestConfig.configService(config));
        var a = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.1:8080");
        var b = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.2:8080");
        var leases = new java.util.ArrayList<org.flexlb.balance.endpoint.PrefillState.RouteReservation>();
        for (int i = 0; i < 2; i++) {
            var route = mock(RequestRoute.class);
            when(route.requestId()).thenReturn(8000L + i);
            when(route.seqLen()).thenReturn(10L);
            var lease = a.ownership.reserveUnqueuedRoute(route, 1, 2).reservation();
            assertTrue(lease != null);
            leases.add(lease);
        }
        var busy = mock(RequestRoute.class);
        when(busy.requestId()).thenReturn(9000L);
        when(busy.seqLen()).thenReturn(10_000L);
        var busyLease = b.ownership.reserveUnqueuedRoute(busy, 1, 2).reservation();
        assertTrue(busyLease != null);
        try (var selected = select()) {
            assertEquals(b, selected.prefillEp(), "low Engine queue cannot bypass the hard inflight limit");
            var accepted = selected.prefillEp().ownership.reserveUnqueuedRoute(selected, 1, 2).reservation();
            assertTrue(accepted != null, "selected worker must accept without waiting for an unrelated completion");
            selected.prefillEp().ownership.release(accepted, null);
        } finally {
            for (var lease : leases) { a.ownership.release(lease, null); }
            b.ownership.release(busyLease, null);
        }
    }

    @ParameterizedTest
    @CsvSource({"9,0,BLOCKED", "9,8,BLOCKED", "9,9,SUCCESS", "4,0,SUCCESS", "4,2,SUCCESS", "0,0,SUCCESS"})
    void congestedWorkerStillDefinesCacheEligibility(int leaderHit, int availableHit, PlacementResult.Status expected) {
        configureEngineQueueAdmission();
        publish("10.0.0.2", 8080);
        var a = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.1:8080");
        var b = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.2:8080");
        var leases = new java.util.ArrayList<org.flexlb.balance.endpoint.PrefillState.RouteReservation>();
        var tasks = new java.util.HashMap<String, org.flexlb.dao.master.TaskInfo>();
        try {
            for (long id = 0; id < a.requestLimit; id++) {
                var busy = mock(RequestRoute.class);
                when(busy.requestId()).thenReturn(8000L + id);
                when(busy.seqLen()).thenReturn(10L);
                var lease = a.ownership.reserveUnqueuedRoute(busy, 1, a.requestLimit).reservation();
                assertTrue(lease != null); leases.add(lease);
                tasks.put(Long.toString(8000L + id), runningTask(8000L + id, 10L));
                updateEngineQueue(a, 0, tasks.size(), tasks);
            }
            updateEngineQueue(a, 1, 0);
            context.data().setRequest(context.data().getRequest().toBuilder()
                    .blockCacheKeys(List.of(1L,2L,3L,4L,5L,6L,7L,8L,9L)).cacheKeyBlockSize(100L).build());
            when(cache.findMatchingEngines(any(), any(), any())).thenAnswer(call -> {
                List<String> addresses = call.getArgument(2);
                return java.util.stream.Stream.of(Map.entry(a.ipPort(), leaderHit), Map.entry(b.ipPort(), availableHit))
                        .filter(entry -> addresses.contains(entry.getKey()))
                        .collect(java.util.stream.Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
            });
            var result = StrategyTestSupport.select(strategy, context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, null);
            try (var selected = result.value()) {
                assertEquals(expected, result.status());
                if (expected == PlacementResult.Status.SUCCESS) { assertEquals(b, selected.prefillEp()); }
            }
            for (var lease : leases) { a.ownership.release(lease, null); }
            leases.clear();
            updateEngineQueue(a, 0, 0);
            org.mockito.Mockito.doReturn(Map.of(a.ipPort(), leaderHit)).when(cache).findMatchingEngines(any(), any(), any());
            try (var selected = select()) {
                if (leaderHit > 5) { assertEquals(a, selected.prefillEp(), "released hot cache capacity resumes selection"); }
            }
        } finally { for (var lease : leases) { a.ownership.release(lease, null); } }
    }

    @Test
    void engineWaitingThresholdBypassesTokenCapAndWakesOnQueueDrainWithUnchangedTotal() {
        configureEngineQueueAdmission();
        var a = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.1:8080");
        var busy = mock(RequestRoute.class);
        when(busy.requestId()).thenReturn(9000L);
        when(busy.seqLen()).thenReturn(2_100_000L);
        var lease = a.ownership.reserveUnqueuedRoute(busy, 1, 0).reservation();
        try {
            updateEngineQueue(a, 1, 0);
            assertTrue(!a.engineQueueAllowsAdmission());
            var blocked = StrategyTestSupport.select(strategy, context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, null);
            assertEquals(PlacementResult.Status.BLOCKED, blocked.status());
            // No completion, no decrease in total Engine count: waiting -> running must wake routing.
            var change = updateEngineQueue(a, 0, 1, Map.of("9000", runningTask(9000L, 2_100_000L)));
            assertTrue(change.reconciliation().capacityReleased());
            assertTrue(change.reconciliation().schedulingInputsChanged());
            assertTrue(a.engineQueueAllowsAdmission());
            try (var selected = select()) {
                assertEquals(a, selected.prefillEp());
                var admitted = a.ownership.reserveUnqueuedRoute(selected, 1, 10).reservation();
                assertTrue(admitted != null);
                a.ownership.release(admitted, null);
            }
            updateEngineQueue(a, -1, 1);
            assertTrue(!a.engineQueueAllowsAdmission(), "unknown waiting count must not bypass admission");
        } finally { a.ownership.release(lease, null); }
    }

    @ParameterizedTest
    @ValueSource(ints = {0, 5})
    void lowEngineQueueCannotDowngradeCacheWhenLeaderIsSaturated(int fallbackHit) {
        configureEngineQueueAdmission();
        publish("10.0.0.2", 8080);
        var a = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.1:8080");
        var b = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.2:8080");
        var busy = mock(RequestRoute.class);
        when(busy.requestId()).thenReturn(9000L);
        when(busy.seqLen()).thenReturn(2_100_000L);
        var lease = a.ownership.reserveUnqueuedRoute(busy, 1, 0).reservation();
        try {
            updateEngineQueue(a, 1, 0);
            updateEngineQueue(b, 0, 0);
            context.data().setRequest(context.data().getRequest().toBuilder()
                    .blockCacheKeys(List.of(1L,2L,3L,4L,5L,6L,7L,8L,9L)).cacheKeyBlockSize(100L).build());
            when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of(a.ipPort(), 9, b.ipPort(), fallbackHit));
            var blocked = StrategyTestSupport.select(strategy, context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, null);
            if (blocked.value() != null) { blocked.value().close(); }
            assertEquals(PlacementResult.Status.BLOCKED, blocked.status());
            updateEngineQueue(a, 0, 1, Map.of("9000", runningTask(9000L, 2_100_000L)));
            try (var selected = select()) {
                assertEquals(a, selected.prefillEp(), "low queue may bypass tokens only on a cache-eligible worker");
                assertEquals(900L, selected.hitCache());
            }
            assertTrue(strategy.hasEngineQueueAdmission(RoleType.PREFILL, null));
            assertTrue(!strategy.hasEngineQueueAdmission(RoleType.PREFILL, "other"), "another pool cannot grant relief");
        } finally { a.ownership.release(lease, null); }
    }

    @Test
    void equalHotCacheMatchesRotateWithoutTokenLoadRanking() {
        configureEngineQueueAdmission();
        publish("10.0.0.2", 8080);
        var a = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.1:8080");
        var b = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.2:8080");
        var load = a.ownership.planTokenWork(1_000_000L);
        context.data().setRequest(context.data().getRequest().toBuilder()
                .blockCacheKeys(List.of(1L,2L,3L,4L,5L,6L,7L,8L,9L)).cacheKeyBlockSize(100L).build());
        when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of(a.ipPort(), 9, b.ipPort(), 9));
        try (load; var first = select(); var second = select()) {
            assertEquals(900, first.hitCache());
            assertEquals(900, second.hitCache());
            assertEquals(Set.of(a, b), Set.of(first.prefillEp(), second.prefillEp()));
        }
    }

    @ParameterizedTest
    @CsvSource({"32768,true", "20000,true", "19999,false"})
    void coldRequestsIgnoreWeakHitsInSelectionEstimateAndReservation(long input, boolean cold) {
        configureEngineQueueAdmission();
        config.getRouter().getRoles().getPrefill().getTokenBalance().setMinPrefixHitPercent(5);
        publish("10.0.0.2", 8080);
        var a = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.1:8080");
        var b = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.2:8080");
        context.data().setRequest(context.data().getRequest().toBuilder()
                .seqLen(input).blockCacheKeys(List.of(1L)).cacheKeyBlockSize(1000L).build());
        when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of(a.ipPort(), 1));
        // Just one token of extra work must outweigh any weak cache match on A.
        try (var load = a.ownership.planTokenWork(1L); var selected = select()) {
            assertEquals(cold ? b : a, selected.prefillEp());
            assertEquals(cold ? 0L : 1000L, selected.hitCache());
            var estimate = strategy.estimateQueuedWork(context.data().getRequest(), RoleType.PREFILL, null);
            assertEquals(cold ? 0L : 1000L, estimate.hitTokens());
            assertEquals(cold ? input : input - 1000L, estimate.workMs());
            var endpoint = selected.prefillEp();
            var lease = endpoint.ownership.reserveUnqueuedRoute(selected, 1L, 0L).reservation();
            assertTrue(lease != null);
            assertEquals(cold ? input : input - 1000L, endpoint.ownership.tokenWorkload().pendingTokens());
            endpoint.ownership.release(lease, null);
            assertEquals(0L, endpoint.ownership.tokenWorkload().pendingTokens());
        }
        assertEquals(0L, a.ownership.tokenWorkload().plannedTokens());
        assertEquals(0L, b.ownership.tokenWorkload().plannedTokens());
    }

    @Test
    void weakHitFullWorkerCannotBlockAnIdleZeroHitWorker() {
        configureEngineQueueAdmission();
        config.getRouter().getRoles().getPrefill().getTokenBalance().setMinPrefixHitPercent(5);
        publish("10.0.0.2", 8080);
        var a = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.1:8080");
        var b = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.2:8080");
        var busy = mock(RequestRoute.class);
        when(busy.requestId()).thenReturn(9000L);
        when(busy.seqLen()).thenReturn(2_100_000L);
        var lease = a.ownership.reserveUnqueuedRoute(busy, 1, 0).reservation();
        updateEngineQueue(a, 1, 0);
        try {
            context.data().setRequest(context.data().getRequest().toBuilder()
                    .seqLen(32768L).blockCacheKeys(List.of(1L)).cacheKeyBlockSize(1024L).build());
            when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of(a.ipPort(), 1));
            try (var selected = select()) {
                assertEquals(b, selected.prefillEp());
                assertEquals(0L, selected.hitCache());
            }
        } finally { a.ownership.release(lease, null); }
    }

    @Test
    void coldAdmissionUsesCurrentCapacityWithoutCacheAffinityOrReservations() {
        configureEngineQueueAdmission(false);
        config.getRouter().getRoles().getPrefill().getTokenBalance().setEngineQueueAdmissionThreshold(null);
        publish("10.0.0.2", 8080);
        var a = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.1:8080");
        var b = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.2:8080");
        var leases = new java.util.ArrayList<org.flexlb.balance.endpoint.PrefillState.RouteReservation>();
        try {
            for (var endpoint : List.of(a, b)) {
                updateEngineQueue(endpoint, 1, 0);
                assertEquals(CostBasedPrefillStrategy.ColdAdmission.AVAILABLE, coldAdmission(null));
                var busy = mock(RequestRoute.class);
                when(busy.requestId()).thenReturn(endpoint == a ? 21001L : 21002L);
                when(busy.seqLen()).thenReturn(2_100_000L);
                leases.add(endpoint.ownership.reserveUnqueuedRoute(busy, 1, 0).reservation());
            }
            assertEquals(CostBasedPrefillStrategy.ColdAdmission.INSUFFICIENT_CAPACITY, coldAdmission(null));
            b.ownership.release(leases.removeLast(), null);
            assertEquals(CostBasedPrefillStrategy.ColdAdmission.AVAILABLE, coldAdmission(null));
            assertEquals(0L, b.ownership.tokenWorkload().pendingTokens());
            assertEquals(0L, b.ownership.tokenWorkload().plannedTokens());
            try (var selected = select()) { assertEquals(b, selected.prefillEp()); }
        } finally {
            a.ownership.release(leases.getFirst(), null);
        }
    }

    @Test
    void idleWorkerInAnotherGroupCannotAuthorizeTheRequest() {
        configureEngineQueueAdmission();
        var a = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.1:8080");
        updateEngineQueue(a, 0, 0);
        var status = StrategyTestSupport.workerStatus(RoleType.PREFILL, "other", "10.0.0.2", 8080, 8081,
                true, 1_000_000L, 1_000_000L);
        StrategyTestSupport.publishEndpoint(registry, RoleType.PREFILL, "10.0.0.2:8080", status);
        var busy = mock(RequestRoute.class);
        when(busy.requestId()).thenReturn(22001L); when(busy.seqLen()).thenReturn(2_100_000L);
        var b = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.2:8080");
        var lease = b.ownership.reserveUnqueuedRoute(busy, 1, 0).reservation();
        updateEngineQueue(b, 1, 0);
        try {
            assertEquals(CostBasedPrefillStrategy.ColdAdmission.AVAILABLE, coldAdmission(null));
            assertEquals(CostBasedPrefillStrategy.ColdAdmission.INSUFFICIENT_CAPACITY, coldAdmission("other"));
            assertEquals(CostBasedPrefillStrategy.ColdAdmission.UNKNOWN, coldAdmission("missing"));
        } finally { b.ownership.release(lease, null); }
    }

    @Test
    void coldAdmissionRetainsInflightCapWhileBypassingTokenWaterline() {
        configureEngineQueueAdmission();
        var a = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.1:8080");
        updateEngineQueue(a, 0, 0);
        var leases = new java.util.ArrayList<org.flexlb.balance.endpoint.PrefillState.RouteReservation>();
        var tasks = new java.util.HashMap<String, org.flexlb.dao.master.TaskInfo>();
        try {
            for (int i = 0; i < 10; i++) {
                var busy = mock(RequestRoute.class);
                when(busy.requestId()).thenReturn(23000L + i); when(busy.seqLen()).thenReturn(i == 0 ? 2_100_000L : 1L);
                var lease = a.ownership.reserveUnqueuedRoute(busy, 1, 0).reservation();
                assertTrue(lease != null);
                leases.add(lease);
                tasks.put(Long.toString(23000L + i), runningTask(23000L + i, i == 0 ? 2_100_000L : 1L));
                updateEngineQueue(a, 0, tasks.size(), tasks);
            }
            assertEquals(CostBasedPrefillStrategy.ColdAdmission.INSUFFICIENT_CAPACITY, coldAdmission(null));
            a.ownership.release(leases.removeLast(), null);
            tasks.remove("23009");
            updateEngineQueue(a, 0, tasks.size(), tasks);
            assertEquals(CostBasedPrefillStrategy.ColdAdmission.AVAILABLE, coldAdmission(null), "Engine queue override admits full token backlog");
            updateEngineQueue(a, 1, 0);
            assertEquals(CostBasedPrefillStrategy.ColdAdmission.INSUFFICIENT_CAPACITY, coldAdmission(null));
        } finally { for (var lease : leases) { a.ownership.release(lease, null); } }
    }

    @ParameterizedTest
    @CsvSource({
            "100, 9, 10, false", "100, 10, 10, true", "100, 11, 10, true",
            "11, 1, 10, false", "11, 2, 10, true",
            "1, 0, 10, false", "1, 1, 10, true",
            "2, 1, 50, true", "2, 1, 50.1, false",
            "20, 1, 7.5, false", "20, 2, 7.5, true",
            "2, 1, 100, false", "2, 2, 100, true"
    })
    void coldSheddingVetoRequiresConfiguredFractionOfWholePool(
            int workers, int accepting, double percent, boolean available) {
        var policy = new org.flexlb.config.PrefillQueueSheddingConfig();
        policy.setMinAcceptingWorkerPercent(percent);
        config.queueScheduler().setPrefillQueueShedding(policy);
        configureEngineQueueAdmission(false);
        var leases = new java.util.LinkedHashMap<PrefillEndpoint, org.flexlb.balance.endpoint.PrefillState.RouteReservation>();
        try {
            for (int i = 1; i <= workers; i++) {
                if (i > 1) { publish("10.0.0." + i, 8080); }
                var endpoint = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0." + i + ":8080");
                updateEngineQueue(endpoint, 1, 0);
                if (i > accepting) {
                    var busy = mock(RequestRoute.class);
                    when(busy.requestId()).thenReturn(30000L + i);
                    when(busy.seqLen()).thenReturn(2_100_000L);
                    leases.put(endpoint, endpoint.ownership.reserveUnqueuedRoute(busy, 1, 0).reservation());
                }
            }
            assertEquals(available ? CostBasedPrefillStrategy.ColdAdmission.AVAILABLE
                    : CostBasedPrefillStrategy.ColdAdmission.INSUFFICIENT_CAPACITY, coldAdmission(null));
            for (var entry : registry.prefillRoutingSnapshot(RoleType.PREFILL)) {
                assertEquals(leases.containsKey(entry.endpoint()) ? 2_100_000L : 0L,
                        entry.endpoint().ownership.tokenWorkload().pendingTokens());
                assertEquals(0L, entry.endpoint().ownership.tokenWorkload().plannedTokens());
            }
            Mockito.verifyNoInteractions(cache);
        } finally { leases.forEach((endpoint, lease) -> endpoint.ownership.release(lease, null)); }
    }

    @Test
    void requestCountBlockedWorkersRemainInPercentageDenominator() {
        configureEngineQueueAdmission(false);
        var leases = new java.util.LinkedHashMap<PrefillEndpoint, java.util.List<org.flexlb.balance.endpoint.PrefillState.RouteReservation>>();
        try {
            for (int i = 2; i <= 11; i++) {
                publish("10.0.0." + i, 8080);
                var endpoint = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0." + i + ":8080");
                var owned = new java.util.ArrayList<org.flexlb.balance.endpoint.PrefillState.RouteReservation>();
                leases.put(endpoint, owned);
                for (int j = 0; j < 10; j++) {
                    var busy = mock(RequestRoute.class);
                    when(busy.requestId()).thenReturn(40000L + i * 10 + j);
                    when(busy.seqLen()).thenReturn(1L);
                    owned.add(endpoint.ownership.reserveUnqueuedRoute(busy, 1, 0).reservation());
                }
            }
            // One accepting worker out of eleven is below the default 10%, despite spare tokens.
            assertEquals(CostBasedPrefillStrategy.ColdAdmission.INSUFFICIENT_CAPACITY, coldAdmission(null));
            var freed = leases.entrySet().iterator().next();
            freed.getKey().ownership.release(freed.getValue().removeLast(), null);
            assertEquals(CostBasedPrefillStrategy.ColdAdmission.AVAILABLE, coldAdmission(null));
        } finally { leases.forEach((endpoint, owned) -> owned.forEach(lease -> endpoint.ownership.release(lease, null))); }
    }

    @Test
    void percentageDenominatorExcludesWorkersOutsideTheRequestGroup() {
        configureEngineQueueAdmission();
        for (int i = 2; i <= 21; i++) { publish("10.0.0." + i, 8080); }
        var status = StrategyTestSupport.workerStatus(RoleType.PREFILL, "isolated", "10.0.1.1", 8080, 8081,
                true, 1_000_000L, 1_000_000L);
        StrategyTestSupport.publishEndpoint(registry, RoleType.PREFILL, "10.0.1.1:8080", status);
        // The sole accepting worker in this request's pool is 100%, not 1/22.
        assertEquals(CostBasedPrefillStrategy.ColdAdmission.AVAILABLE, coldAdmission("isolated"));
        assertEquals(CostBasedPrefillStrategy.ColdAdmission.UNKNOWN, coldAdmission("missing"));
    }

    @ParameterizedTest
    @CsvSource({"10, 2", "50, 6", "100, 11"})
    void engineQueueReliefCannotBypassTheConfiguredWorkerFraction(double percent, int required) {
        var policy = new org.flexlb.config.PrefillQueueSheddingConfig();
        policy.setMinAcceptingWorkerPercent(percent);
        config.queueScheduler().setPrefillQueueShedding(policy);
        configureEngineQueueAdmission();
        for (int i = 1; i <= 11; i++) {
            if (i > 1) { publish("10.0.0." + i, 8080); }
            updateEngineQueue((PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0." + i + ":8080"), 1, 0);
        }
        for (int i = 1; i <= required; i++) {
            assertEquals(false, strategy.hasEngineQueueAdmission(RoleType.PREFILL, null));
            updateEngineQueue((PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0." + i + ":8080"), 0, 0);
        }
        assertTrue(strategy.hasEngineQueueAdmission(RoleType.PREFILL, null));
        assertEquals(false, strategy.hasEngineQueueAdmission(RoleType.PREFILL, "missing"));
        // Group isolation applies to the early Engine-queue relief check as well.
        var status = StrategyTestSupport.workerStatus(RoleType.PREFILL, "isolated", "10.0.1.1", 8080, 8081,
                true, 1_000_000L, 1_000_000L);
        StrategyTestSupport.publishEndpoint(registry, RoleType.PREFILL, "10.0.1.1:8080", status);
        updateEngineQueue((PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.1.1:8080"), 0, 0);
        assertTrue(strategy.hasEngineQueueAdmission(RoleType.PREFILL, "isolated"));
    }

    @Test
    void engineQueueCannotBypassReportedStreamsAtTheInflightLimit() {
        configureEngineQueueAdmission();
        var worker = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.1:8080");
        updateEngineQueue(worker, 0, 256);
        var busy = mock(RequestRoute.class);
        when(busy.requestId()).thenReturn(20000L);
        when(busy.seqLen()).thenReturn(2_100_000L);
        assertNull(worker.ownership.reserveUnqueuedRoute(busy, 1, 10).reservation());
        assertEquals(CostBasedPrefillStrategy.ColdAdmission.INSUFFICIENT_CAPACITY, coldAdmission(null));
        assertTrue(!strategy.hasEngineQueueAdmission(RoleType.PREFILL, null));
        assertTrue(!strategy.isClusterOverloaded(RoleType.PREFILL, null), "inflight capacity is not proof of sustained queue congestion");
        updateEngineQueue(worker, 0, 9);
        var lease = worker.ownership.reserveUnqueuedRoute(busy, 1, 10).reservation();
        assertNotNull(lease);
        worker.ownership.release(lease, null);
    }

    @Test
    void fourWorkerPoolRequiresThreeCongestedEnginesAndColdTrafficUsesIdleWorkers() {
        configureEngineQueueAdmission();
        for (int i = 2; i <= 4; i++) { publish("10.0.0." + i, 8080); }
        var workers = new java.util.ArrayList<PrefillEndpoint>();
        for (int i = 1; i <= 4; i++) {
            var worker = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0." + i + ":8080");
            workers.add(worker);
            updateEngineQueue(worker, 0, 2);
        }
        updateEngineQueue(workers.get(0), 100, 0);
        assertTrue(!strategy.isClusterOverloaded(RoleType.PREFILL, null));
        var used = new java.util.HashSet<PrefillEndpoint>();
        for (int i = 0; i < 60; i++) {
            try (var route = select()) {
                assertTrue(route.prefillEp() != workers.get(0));
                used.add(route.prefillEp());
            }
        }
        assertEquals(3, used.size());
        updateEngineQueue(workers.get(1), 100, 0);
        assertTrue(!strategy.isClusterOverloaded(RoleType.PREFILL, null), "half the pool is not a majority");
        updateEngineQueue(workers.get(2), 100, 0);
        for (long ms = 0; ms < 10_000; ms += 100) {
            assertTrue(!strategy.isClusterOverloaded(RoleType.PREFILL, null, ms));
        }
        assertTrue(strategy.isClusterOverloaded(RoleType.PREFILL, null, 10_000));
        var request = context.data().getRequest().toBuilder().seqLen(1000L)
                .blockCacheKeys(List.of(1L,2L,3L,4L,5L,6L,7L,8L,9L)).cacheKeyBlockSize(100L).build();
        when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of(workers.get(3).ipPort(), 9));
        assertTrue(strategy.hotRequestHasEngineQueueAdmission(request, RoleType.PREFILL, null));
        when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of(workers.get(0).ipPort(), 9));
        assertTrue(!strategy.hotRequestHasEngineQueueAdmission(request, RoleType.PREFILL, null));
        assertTrue(!strategy.isClusterOverloaded(RoleType.PREFILL, "missing"));
        updateEngineQueue(workers.get(2), 0, 100);
        assertTrue(!strategy.isClusterOverloaded(RoleType.PREFILL, null));
        updateEngineQueue(workers.get(2), -1, 0);
        assertTrue(!strategy.isClusterOverloaded(RoleType.PREFILL, null), "unknown telemetry is not congestion");
    }

    @Test
    void globalPressureCountsWaitingBelowTheDispatchWatermarkAndRotatingMajorities() {
        var balance = new RoutingConfig.TokenBalanceConfig();
        balance.setEngineQueueAdmissionThreshold(30L);
        config.getRouter().getRoles().getPrefill().setTokenBalance(balance);
        registry.close();
        registry = StrategyTestSupport.endpointRegistry(SchedulingTestConfig.configService(config));
        var workers = new java.util.ArrayList<PrefillEndpoint>();
        for (int i = 1; i <= 3; i++) {
            publish("10.0.0." + i, 8080);
            workers.add((PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0." + i + ":8080"));
        }
        strategy = new CostBasedPrefillStrategy(registry, cache, reporter, cacheReporter,
                SchedulingTestConfig.configService(config));
        for (var worker : workers) { updateEngineQueue(worker, 10, 0); }
        for (long ms = 0; ms <= 10_000; ms += 100) {
            assertTrue(!strategy.isClusterOverloaded(RoleType.PREFILL, null, ms));
        }
        for (long ms = 10_000; ms <= 20_000; ms += 100) {
            int idle = (int) (ms / 100) % 3;
            for (int i = 0; i < 3; i++) {
                updateEngineQueue(workers.get(i), i == idle ? 10 : 11, 0);
                assertTrue(workers.get(i).engineQueueAllowsAdmission());
                assertTrue(!workers.get(i).engineQueueCongested());
            }
            assertEquals(ms >= 20_000, strategy.isClusterOverloaded(RoleType.PREFILL, null, ms));
        }
        updateEngineQueue(workers.get(0), 10, 0);
        updateEngineQueue(workers.get(1), 10, 0);
        assertTrue(!strategy.isClusterOverloaded(RoleType.PREFILL, null, 20_001));
        updateEngineQueue(workers.get(0), 11, 0);
        updateEngineQueue(workers.get(1), 11, 0);
        assertTrue(!strategy.isClusterOverloaded(RoleType.PREFILL, null, 20_002));
        assertTrue(!strategy.isClusterOverloaded(RoleType.PREFILL, null, 30_001));
        assertTrue(strategy.isClusterOverloaded(RoleType.PREFILL, null, 30_002));
        // Unknown telemetry contributes to the fleet size, but never to the waiting majority.
        updateEngineQueue(workers.get(0), -1, 0);
        updateEngineQueue(workers.get(1), -1, 0);
        assertTrue(!strategy.isClusterOverloaded(RoleType.PREFILL, null, 30_003));
        // Dispatch can oscillate at 30/31 without resetting the global pressure window.
        for (long ms = 40_000; ms <= 50_000; ms += 100) {
            long waiting = ms / 100 % 2 == 0 ? 30 : 31;
            for (var worker : workers) {
                updateEngineQueue(worker, waiting, 0);
                assertEquals(waiting > 30, worker.engineQueueDispatchBlocked());
                assertEquals(waiting <= 30, worker.engineQueueAllowsAdmission());
            }
            assertEquals(ms >= 50_000, strategy.isClusterOverloaded(RoleType.PREFILL, null, ms));
        }
    }

    private CostBasedPrefillStrategy.ColdAdmission coldAdmission(String group) {
        return strategy.coldAdmission(context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, group);
    }

    private void configureEngineQueueAdmission() { configureEngineQueueAdmission(true); }

    private void configureEngineQueueAdmission(boolean enabled) {
        registry.close();
        var balance = new RoutingConfig.TokenBalanceConfig();
        balance.setEngineQueueAdmissionThreshold(enabled ? 0L : null);
        balance.setEngineQueueOverloadDurationMs(0L);
        config.getRouter().getRoles().getPrefill().setTokenBalance(balance);
        config.getDispatcher().setMaxInflightPerPrefillWorker(10);
        registry = StrategyTestSupport.endpointRegistry(SchedulingTestConfig.configService(config));
        publish("10.0.0.1", 8080);
        strategy = new CostBasedPrefillStrategy(registry, cache, reporter, cacheReporter,
                SchedulingTestConfig.configService(config));
    }

    private org.flexlb.dao.master.TaskInfo runningTask(long id, long tokens) {
        var task = new org.flexlb.dao.master.TaskInfo();
        task.setRequestId(id);
        task.setBatchId(0L);
        task.setInputLength(tokens);
        task.setPhase(org.flexlb.enums.TaskPhase.RUNNING);
        return task;
    }

    private org.flexlb.balance.endpoint.WorkerEndpoint.StatusChange.Prefill updateEngineQueue(
            PrefillEndpoint endpoint, long waiting, long running) {
        return updateEngineQueue(endpoint, waiting, running, Map.of());
    }

    private org.flexlb.balance.endpoint.WorkerEndpoint.StatusChange.Prefill updateEngineQueue(
            PrefillEndpoint endpoint, long waiting, long running, Map<String, org.flexlb.dao.master.TaskInfo> tasks) {
        var status = endpoint.getStatus();
        var response = StrategyTestSupport.response(RoleType.PREFILL, true, 1_000_000L, 1_000_000L,
                status.appliedStatusCursor().statusVersion() + 1L);
        response.setWaitingQueryLen(waiting);
        response.setRunningQueryLen(running);
        response.setRunningTaskInfo(tasks);
        status.lock.lock();
        try {
            var change = endpoint.applyPreparedStatus(status,
                    status.prepareNewStatus(WorkerStatusTestSupport.freezeStatusResponse(status, response)));
            endpoint.publishStatus(change);
            return (org.flexlb.balance.endpoint.WorkerEndpoint.StatusChange.Prefill) change;
        } finally { status.lock.unlock(); }
    }

    @Test
    void tokenBalanceIgnoresTtftAndUsesAcceptedUncachedWork() {
        registry.close();
        config.getRouter().getRoles().getPrefill().setTokenBalance(new RoutingConfig.TokenBalanceConfig());
        registry = StrategyTestSupport.endpointRegistry(SchedulingTestConfig.configService(config));
        publish("10.0.0.1", 8080); publish("10.0.0.2", 8080);
        strategy = new CostBasedPrefillStrategy(registry, cache, reporter, cacheReporter,
                SchedulingTestConfig.configService(config));
        var a = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.1:8080");
        var b = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.2:8080");
        var set = new PrefillCandidateSet();
        set.reset(2);
        set.addCandidate(a.ipPort(), a, new RouteProjection.Candidate(RouteProjection.Candidate.State.MODELED,
                100_000, 100, RouteProjection.Candidate.InitialHeadDisposition.NONE, "", null, 900, 900), 0);
        set.addCandidate(b.ipPort(), b, new RouteProjection.Candidate(RouteProjection.Candidate.State.MODELED,
                1, 1, RouteProjection.Candidate.InitialHeadDisposition.NONE, "", null, 0, 0), 0);
        assertEquals(0, strategy.selectBestCandidate(set, 1, RoleType.PREFILL, null, 1000));
        var route = mock(RequestRoute.class);
        when(route.requestId()).thenReturn(123456L);
        when(route.seqLen()).thenReturn(2_100_000L);
        var lease = a.ownership.reserveUnqueuedRoute(route, 1, 0).reservation();
        // Token backlog alone cannot block a cache leader with an accepting Engine queue.
        assertEquals(0, strategy.selectBestCandidate(set, 1, RoleType.PREFILL, null, 1000));
        set.reset(2);
        set.addCandidate(a.ipPort(), a, new RouteProjection.Candidate(RouteProjection.Candidate.State.MODELED,
                1, 1, RouteProjection.Candidate.InitialHeadDisposition.NONE, "", null, 0, 0), 0);
        set.addCandidate(b.ipPort(), b, new RouteProjection.Candidate(RouteProjection.Candidate.State.MODELED,
                100_000, 100, RouteProjection.Candidate.InitialHeadDisposition.NONE, "", null, 0, 0), 0);
        assertEquals(1, strategy.selectBestCandidate(set, 1, RoleType.PREFILL, null, 1000));
        a.ownership.release(lease, null);
        try (var first = select(); var second = select()) {
            org.junit.jupiter.api.Assertions.assertNotEquals(first.prefillEp(), second.prefillEp(),
                    "parallel plans must balance before either admission commits");
            assertEquals(2000, a.ownership.tokenWorkload().plannedTokens()
                    + b.ownership.tokenWorkload().plannedTokens());
            var accepted = first.prefillEp().ownership.reserveUnqueuedRoute(first, 1, 0).reservation();
            assertEquals(0, first.prefillEp().ownership.tokenWorkload().plannedTokens());
            assertEquals(1000, first.prefillEp().ownership.tokenWorkload().pendingTokens());
            first.prefillEp().ownership.release(accepted, null);
        }
        assertEquals(0, a.ownership.tokenWorkload().score());
        assertEquals(0, b.ownership.tokenWorkload().score());
    }

    @Test
    void agedShortRequestsRescueOnlyToCacheEligibleWorkers() {
        registry.close();
        var balance = new RoutingConfig.TokenBalanceConfig();
        balance.setEngineQueueOverloadDurationMs(0L);
        balance.setHotspotRescue(new RoutingConfig.HotspotRescueConfig());
        config.getRouter().getRoles().getPrefill().setTokenBalance(balance);
        registry = StrategyTestSupport.endpointRegistry(SchedulingTestConfig.configService(config));
        publish("10.0.0.1", 8080);
        strategy = new CostBasedPrefillStrategy(registry, cache, reporter, cacheReporter,
                SchedulingTestConfig.configService(config));
        var source = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.1:8080");
        var running = mock(RequestRoute.class);
        when(running.requestId()).thenReturn(99999L);
        when(running.seqLen()).thenReturn(2_100_000L);
        var lease = source.ownership.reserveUnqueuedRoute(running, 10000, 0).reservation();
        updateEngineQueue(source, 31, 0);
        when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of("10.0.0.1:8080", 9));
        for (int phase = 0; phase < 5; phase++) {
            if (phase == 1) { publish("10.0.0.2", 8080); }
            if (phase == 4) { when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of("10.0.0.1:8080", 9, "10.0.0.2:8080", 9)); }
            long length = phase == 3 ? 40000L : 1000L;
            var ctx = new RequestContext(config);
            ctx.data().setRequest(Request.builder().requestId(10001L + phase).seqLen(length).rawPriority(50)
                    .blockCacheKeys(List.of(1L,2L,3L,4L,5L,6L,7L,8L,9L)).cacheKeyBlockSize(length / 10).build());
            long now = System.currentTimeMillis();
            ctx.data().setSchedulingMetadata(SchedulingMetadata.explicit(50, now + 60000));
            ctx.register(ctx.data().getRequestId(), phase == 2 ? now : now - 10000, now + 60000, 60000);
            try (var route = new RequestRoute(ctx)) {
                var result = strategy.select(ctx.data().getRequest(), ctx.data().getSchedulingMetadata(), RoleType.PREFILL, null, route);
                if (phase == 4) {
                    assertEquals(PlacementResult.Status.SUCCESS, result.status());
                    assertEquals("10.0.0.2", route.prefill().getServerIp());
                    assertEquals(900, route.hitCache(), "rescue must retain cache eligibility");
                } else {
                    assertEquals(PlacementResult.Status.BLOCKED, result.status());
                }
            }
        }
        source.ownership.release(lease, null);
    }

    @ParameterizedTest
    @CsvSource({"0,9,true", "50,9,true", "51,9,false", "0,8,false", "0,0,false"})
    void migratedHotRequestRechecksTargetLoadAndCacheWithoutReturningToSource(long targetPending,
            int targetHitBlocks, boolean allowed) {
        registry.close();
        var balance = new RoutingConfig.TokenBalanceConfig();
        balance.setEngineQueueAdmissionThreshold(0L);
        balance.setEngineQueueOverloadDurationMs(0L);
        var rescue = new RoutingConfig.HotspotRescueConfig();
        rescue.setMaxIdlePendingTokens(50);
        balance.setHotspotRescue(rescue);
        config.getRouter().getRoles().getPrefill().setTokenBalance(balance);
        registry = StrategyTestSupport.endpointRegistry(SchedulingTestConfig.configService(config));
        publish("10.0.0.1", 8080);
        publish("10.0.0.2", 8080);
        strategy = new CostBasedPrefillStrategy(registry, cache, reporter, cacheReporter,
                SchedulingTestConfig.configService(config));
        var source = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.1:8080");
        var target = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.2:8080");
        var running = mock(RequestRoute.class);
        when(running.requestId()).thenReturn(99999L);
        when(running.seqLen()).thenReturn(targetPending);
        var lease = target.ownership.reserveUnqueuedRoute(running, 1, 0).reservation();
        updateEngineQueue(target, 0, 1, Map.of("99999", runningTask(99999L, targetPending)));
        when(cache.findMatchingEngines(any(), any(), any())).thenReturn(
                Map.of("10.0.0.1:8080", 9, "10.0.0.2:8080", targetHitBlocks));
        var ctx = new RequestContext(config);
        ctx.data().setRequest(Request.builder().requestId(10001L).seqLen(1000L).rawPriority(50)
                .blockCacheKeys(List.of(1L,2L,3L,4L,5L,6L,7L,8L,9L)).cacheKeyBlockSize(100L).build());
        long now = System.currentTimeMillis();
        ctx.data().setSchedulingMetadata(SchedulingMetadata.explicit(50, now + 60000));
        ctx.register(10001L, now - 10000, now + 60000, 60000);
        org.springframework.test.util.ReflectionTestUtils.setField(ctx, "prefillMigrationSource", source);
        try (var route = new RequestRoute(ctx)) {
            var result = strategy.select(ctx.data().getRequest(), ctx.data().getSchedulingMetadata(), RoleType.PREFILL, null, route);
            assertEquals(allowed ? PlacementResult.Status.SUCCESS : PlacementResult.Status.BLOCKED, result.status());
            if (allowed) {
                assertEquals(target, route.prefillEp());
                assertEquals(900, route.hitCache());
            }
        } finally {
            target.ownership.release(lease, null);
        }
    }

    private RequestRoute select() {
        PlacementResult<RequestRoute, RoleType> result =
                StrategyTestSupport.select(strategy, context.data().getRequest(), context.data().getSchedulingMetadata(), RoleType.PREFILL, null);
        assertEquals(PlacementResult.Status.SUCCESS, result.status());
        return result.value();
    }

    private void configureFocusedAffinity() {
        RoutingConfig.CacheAffinityConfig affinity =
                new RoutingConfig.CacheAffinityConfig();
        affinity.setMaxExtraTtftMs(10L);
        affinity.setMinPrefixHitPercent(0.0);
        config.getRouter().getRoles().getPrefill().setCacheAffinity(affinity);
        refreshContext();
    }

    private PrefillCandidateSet candidates(long[] ttftMs, long[] hitTokens) {
        assertEquals(ttftMs.length, hitTokens.length);
        PrefillCandidateSet candidates = new PrefillCandidateSet();
        candidates.reset(ttftMs.length);
        for (int i = 0; i < ttftMs.length; i++) {
            String ip = "10.1.0." + (i + 1);
            PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
            when(endpoint.getIp()).thenReturn(ip);
            candidates.addCandidate(
                    ip + ":8080",
                    endpoint,
                    new RouteProjection.Candidate(
                            RouteProjection.Candidate.State.MODELED,
                            ttftMs[i],
                            ttftMs[i],
                            RouteProjection.Candidate.InitialHeadDisposition.NONE,
                            "",
                            null,
                            hitTokens[i],
                            hitTokens[i]),
                    i + 1L);
        }
        return candidates;
    }

    private void refreshContext() {
        var refreshed = new RequestContext(config);
        refreshed.data().setRequest(context.data().getRequest());
        refreshed.data().setSchedulingMetadata(context.data().getSchedulingMetadata());
        org.springframework.test.util.ReflectionTestUtils.setField(refreshed.data(), "startTime", context.data().getStartTime());
        context = refreshed;
        var currentRegistry = (EndpointRegistry) org.springframework.test.util.ReflectionTestUtils.getField(strategy, "endpointRegistry");
        strategy = new CostBasedPrefillStrategy(currentRegistry, cache, reporter, cacheReporter,
                SchedulingTestConfig.configService(config));
    }

    private void configureAffinity(
            long maxExtraTtftMs,
            double minPrefixHitPercent) {
        config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator()
                .setExpression("sum(computeTokens) + 2*sum(hitCacheTokens)");
        RoutingConfig.CacheAffinityConfig affinity =
                new RoutingConfig.CacheAffinityConfig();
        affinity.setMaxExtraTtftMs(maxExtraTtftMs);
        affinity.setMinPrefixHitPercent(minPrefixHitPercent);
        config.getRouter().getRoles().getPrefill().setCacheAffinity(affinity);
        refreshContext();

        publish("10.0.0.2", 8080);
        context.data().setRequest(context.data().getRequest().toBuilder().requestId(20_001L).build());
        context.data().setRequest(context.data().getRequest().toBuilder().blockCacheKeys(List.of(1L, 2L, 3L, 4L, 5L)).build());
        context.data().setRequest(context.data().getRequest().toBuilder().cacheKeyBlockSize(100L).build());
        when(cache.findMatchingEngines(any(), any(), any()))
                .thenReturn(Map.of("10.0.0.2:8080", 5));
    }

    private void publish(String ip, int port) {
        WorkerStatus status = StrategyTestSupport.workerStatus(
                RoleType.PREFILL, null, ip, port, port + 1,
                true, 1_000_000L, 1_000_000L);
        StrategyTestSupport.publishEndpoint(
                registry, RoleType.PREFILL, ip + ":" + port, status);
    }
}
