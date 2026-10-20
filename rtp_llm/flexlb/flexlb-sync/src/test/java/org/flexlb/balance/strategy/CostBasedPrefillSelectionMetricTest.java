package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.scheduler.PlacementKey;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.QueueOrderingConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.testfixture.TaskInfo;
import org.flexlb.testfixture.WorkerStatusTestFixtures;
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
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class CostBasedPrefillSelectionMetricTest {
    private static CacheMatchResult matches(Map<String, Integer> blocks) {
        return new CacheMatchResult(org.flexlb.dao.cache.HostCacheMatch.fromLocalMatches(blocks),
                CacheMatchSource.KVCM, 0L, 100L);
    }
    private static final int FULL_FLEET_SIZE = 750;

    private FlexlbConfig config;
    private EndpointRegistry registry;
    private CacheAwareService cache;
    private EngineHealthReporter reporter;
    private CostBasedPrefillStrategy strategy;
    private BalanceContext context;

    @BeforeEach
    void setUp() {
        config = StrategyTestSupport.config();
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator()
                .setExpression("sum(computeTokens)");
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        registry = StrategyTestSupport.endpointRegistry(configService);
        publish("10.0.0.1", 8080);

        cache = mock(CacheAwareService.class);
        when(cache.findMatchingEngines(any()))
                .thenReturn(CacheMatchResult.empty(CacheMatchSource.KVCM));
        reporter = mock(EngineHealthReporter.class);
        strategy = new CostBasedPrefillStrategy(
                new WorkerDirectory(registry), cache, reporter);

        Request request = new Request();
        request.setRequestId("10001");
        request.setSeqLen(1_000L);
        request.setPriority(50);
        request.setBlockCacheKeys(List.of());
        context = new BalanceContext(config);
        context.setRequest(request);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() + 60_000L));
    }

    @AfterEach
    void tearDown() {
        registry.close();
    }

    @Test
    void pvCapturesRealCacheQueryAndSelectionAcrossRetries() {
        when(cache.findMatchingEngines(any())).thenReturn(new CacheMatchResult(
                Map.of(), CacheMatchSource.KVCM, 37L, 100L));
        for (int attempt = 0; attempt < 2; attempt++) {
            try (SelectedRole selected = select()) {
                var decision = context.getRoutingTelemetry().routingDecisions().get(RoleType.PREFILL);
                assertEquals("CostBasedPrefill", decision.strategy());
                assertEquals("10.0.0.1:8080@0", decision.selectedEndpoint());
                assertEquals(1, decision.candidates().size());
                assertTrue(decision.candidates().getFirst().selected());
                assertEquals(selected.serverStatus().getPrefillTime(),
                        decision.candidates().getFirst().projectedTtftMs());
                assertEquals(decision.selectionReason(), context.getRoutingTelemetry().selectionReasons().get(RoleType.PREFILL));
            }
        }
        assertEquals("KVCM", context.getRoutingTelemetry().cacheSource());
        assertEquals(2, context.getRoutingTelemetry().cacheQueryCount());
        assertEquals(74L, context.getRoutingTelemetry().cacheQueryUs());
        assertEquals("10.0.0.1", context.getRoutingTelemetry().cacheSelections().get(RoleType.PREFILL).selectedIp());
        assertEquals(1, context.getRoutingTelemetry().routingDecisions().size());
        assertEquals(2, context.getRoutingTelemetry().routingDecisions().get(RoleType.PREFILL).routingAttempt());
    }

    @Test
    void selectionDoesNotOwnTheRequestDeadline() {
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() - 1L));

        try (SelectedRole ignored = select()) {
            assertTrue(ignored.serverStatus().isSuccess());
        }
    }

    @ParameterizedTest
    @CsvSource({"5,1,false,SUCCESS", "50,1,false,SUCCESS", "5,2,false,SUCCESS",
            "50,1,true,SUCCESS", "10,1,true,SUCCESS", "5,1,true,SUCCESS"})
    void queuedNonBatchRemainsSelectableRegardlessOfDispatchLimitOrPreemption(
            int incomingPriority, int requestLimit, boolean preemptQueued, PlacementResult.Status expectedStatus) {
        registry.close();
        config.queueScheduler().getDecision().setMaxRequests(2);
        config.getDispatcher().setMaxInflightPerPrefillWorker(requestLimit);
        config.queueScheduler().getDecision().setMaxCollectionWaitMs(60_000L);
        QueueOrderingConfig ordering = QueueOrderingConfig.priority();
        ordering.setPreemption(null);
        if (preemptQueued) {
            PreemptionConfig preemption = new PreemptionConfig();
            preemption.setAllowedVictimStages(Set.of(VictimStage.PREFILL_QUEUED));
            ordering.setPreemption(preemption);
        }
        config.queueScheduler().setOrdering(ordering);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        registry = StrategyTestSupport.endpointRegistry(service);
        publish("10.0.0.1", 8080);
        strategy = new CostBasedPrefillStrategy(new WorkerDirectory(registry), cache, reporter);
        PrefillEndpoint endpoint = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.1:8080@0");
        Request queuedRequest = new Request();
        queuedRequest.setRequestId("9001");
        queuedRequest.setSeqLen(1_000L);
        queuedRequest.setPriority(10);
        BalanceContext queuedContext = new BalanceContext(config);
        queuedContext.setRequest(queuedRequest);
        queuedContext.setSchedulingMetadata(SchedulingMetadata.explicit(
                10, System.currentTimeMillis() + 120_000L));
        ScheduledRequest queued = new ScheduledRequest(queuedContext, new CompletableFuture<>(),
                null, null, null, endpoint, null, null, System.currentTimeMillis());
        assertTrue(StrategyTestSupport.offer(endpoint, queued));
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                incomingPriority, System.currentTimeMillis() + 120_000L));

        PlacementResult<SelectedRole, PlacementKey> result = strategy.select(context, RoleType.PREFILL, null);

        assertEquals(expectedStatus, result.status());
        if (expectedStatus == PlacementResult.Status.BLOCKED) {
            verify(cache, Mockito.never()).findMatchingEngines(any());
        }
        if (result.status() == PlacementResult.Status.SUCCESS) {
            try (SelectedRole selected = result.value()) {
                assertEquals("10.0.0.1", selected.serverStatus().getServerIp());
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
        config.setDispatcher(batchDelivery
                ? new DispatcherConfig()
                : DispatcherConfig.nonBatch());

        try (SelectedRole selected = select()) {
            ArgumentCaptor<Long> ttft = ArgumentCaptor.forClass(Long.class);
            ArgumentCaptor<Long> execution = ArgumentCaptor.forClass(Long.class);
            verify(reporter).reportPrefillSelectedEstimates(
                    Mockito.eq(RoleType.PREFILL), Mockito.eq("10.0.0.1:8080"),
                    Mockito.eq(deliveryMode), ttft.capture(), execution.capture());
            assertEquals(selected.serverStatus().getPrefillTime(), ttft.getValue());
            assertEquals(selected.prefillWorkMs(), execution.getValue());
        }
    }

    @Test
    void telemetryFailureNeverChangesAValidSelection() {
        doThrow(new IllegalStateException("metrics unavailable"))
                .when(reporter).reportPrefillSelectedEstimates(
                        any(), any(), any(), Mockito.anyLong(), Mockito.anyLong());

        try (SelectedRole selected = select()) {
            assertTrue(selected.serverStatus().isSuccess());
            assertEquals("10.0.0.1", selected.serverStatus().getServerIp());
        }
    }

    @Test
    void cacheLeaderInsideTtftCapOverridesTheBaselineCandidate() {
        configureAffinity(600L, 5.0);

        try (SelectedRole selected = select()) {
            assertEquals("10.0.0.2", selected.serverStatus().getServerIp());
            verify(reporter).reportCacheAffinityDecision(
                    RoleType.PREFILL, "10.0.0.2:8080", "CACHE_LEADER");
        }
    }

    @Test
    void cacheLeaderWithAllDispatchSlotsOccupiedStillWins() {
        configureAffinity(600L, 5.0);
        PrefillEndpoint endpoint = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.2:8080@0");
        var reservations = new java.util.ArrayList<org.flexlb.balance.endpoint.PrefillState.RouteReservation>();
        var occupants = new java.util.ArrayList<ScheduledRequest>();
        try {
            for (long id = 31_001L; id <= 31_002L; id++) {
                Request request = new Request();
                request.setRequestId(Long.toString(id));
                request.setSeqLen(10L);
                BalanceContext ctx = new BalanceContext(config);
                ctx.setRequest(request);
                ScheduledRequest item = new ScheduledRequest(ctx, new CompletableFuture<>(),
                        null, null, null, endpoint, null, null, System.currentTimeMillis());
                try (var pin = endpoint.tryPinGeneration()) {
                    var result = endpoint.reserveUnqueuedRoute(pin, item, 10L);
                    assertEquals(org.flexlb.balance.endpoint.PrefillState.CapacityStatus.ACQUIRED, result.status());
                    reservations.add(result.reservation());
                    occupants.add(item);
                }
            }
            try (var commit = endpoint.tryBeginRouteCommitAdmission();
                 var handoff = commit.commit(occupants, reservations)) { }
            org.junit.jupiter.api.Assertions.assertFalse(endpoint.routeAdmissionAvailability().isAvailable());
            try (SelectedRole selected = select()) {
                assertEquals("10.0.0.2", selected.serverStatus().getServerIp());
            }
        } finally {
            occupants.forEach(item -> endpoint.releaseCommittedItem(item.execution()));
            reservations.forEach(org.flexlb.balance.endpoint.PrefillState.RouteReservation::close);
        }
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void cacheLookupCannotCommitARetiredOrReplacedPrefillGeneration(boolean replace) {
        configureAffinity(600L, 5.0);
        PrefillEndpoint old = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.2:8080@0");
        when(cache.findMatchingEngines(any())).thenAnswer(invocation -> {
            EndpointRegistry.DetachedGeneration detached;
            WorkerStatus status = old.getStatus();
            status.lock.lock();
            try {
                detached = registry.detachAndBeginRetirement(RoleType.PREFILL, old.ipPort(), status);
            } finally {
                status.lock.unlock();
            }
            detached.retireAndAwait();
            if (replace) { publish("10.0.0.2", 8080); }
            return matches(Map.of("10.0.0.2:8080", 5));
        });

        var stale = strategy.select(context, RoleType.PREFILL, null);
        if (stale.value() != null) { stale.value().close(); }
        assertEquals(PlacementResult.Status.BLOCKED, stale.status(),
                "a cache lookup must not transfer ownership to a different endpoint generation");
        // A stale cache entry must not revive a removed worker on the next snapshot.
        org.mockito.Mockito.doReturn(matches(Map.of("10.0.0.2:8080", 5))).when(cache)
                .findMatchingEngines(any());
        try (SelectedRole selected = select()) {
            assertEquals(replace ? "10.0.0.2" : "10.0.0.1", selected.serverStatus().getServerIp());
        }
    }

    @Test
    void cacheMatchingSeesTheLiveFleetBeforeApplyingRouteGroupCompatibility() {
        WorkerStatus target = StrategyTestSupport.workerStatus(RoleType.PREFILL, "target",
                "10.0.0.2", 8080, 8081, true, 1_000_000L, 1_000_000L);
        StrategyTestSupport.publishEndpoint(registry, RoleType.PREFILL, "10.0.0.2:8080", target);
        context.getRequest().setCacheKeyBlockSize(100L);
        context.getRequest().setBlockCacheKeys(List.of(1L));
        when(cache.findMatchingEngines(any()))
                .thenReturn(matches(Map.of("10.0.0.1:8080", 9, "10.0.0.2:8080", 1)));

        var result = strategy.select(context, RoleType.PREFILL, "target");
        assertEquals(PlacementResult.Status.SUCCESS, result.status());
        try (var selected = result.value()) {
            assertEquals("target", selected.serverStatus().getGroup());
        }
        @SuppressWarnings("unchecked")
        ArgumentCaptor<org.flexlb.cache.domain.CacheMatchQuery> candidates = ArgumentCaptor.forClass(org.flexlb.cache.domain.CacheMatchQuery.class);
        verify(cache).findMatchingEngines(candidates.capture());
        org.junit.jupiter.api.Assertions.assertNull(candidates.getValue().group());
    }

    @Test
    void directCacheMatchingIncludesBusyAlivePrefillAndRejectsItsFullWinner() {
        registry.close();
        config.getScheduler().setType(SchedulerConfig.Type.DIRECT);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        registry = StrategyTestSupport.endpointRegistry(service);
        publish("10.0.0.1", 8080);
        strategy = new CostBasedPrefillStrategy(new WorkerDirectory(registry), cache, reporter);
        configureAffinity(600L, 5.0);
        PrefillEndpoint endpoint = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.2:8080@0");
        var reservations = new java.util.ArrayList<org.flexlb.balance.endpoint.PrefillState.RouteReservation>();
        var occupants = new java.util.ArrayList<ScheduledRequest>();
        try {
            for (long id = 32_001L; id <= 32_002L; id++) {
                Request request = new Request();
                request.setRequestId(Long.toString(id));
                request.setSeqLen(10L);
                BalanceContext ctx = new BalanceContext(config);
                ctx.setRequest(request);
                ScheduledRequest item = new ScheduledRequest(ctx, new CompletableFuture<>(),
                        null, null, null, endpoint, null, null, System.currentTimeMillis());
                try (var pin = endpoint.tryPinGeneration()) {
                    var result = endpoint.reserveUnqueuedRoute(pin, item, 10L);
                    assertEquals(org.flexlb.balance.endpoint.PrefillState.CapacityStatus.ACQUIRED, result.status());
                    reservations.add(result.reservation());
                    occupants.add(item);
                }
            }
            try (var commit = endpoint.tryBeginRouteCommitAdmission();
                 var handoff = commit.commit(occupants, reservations)) { }
            org.junit.jupiter.api.Assertions.assertFalse(endpoint.canAcceptRequest());
            var result = strategy.select(context, RoleType.PREFILL, null);
            if (result.value() != null) { result.value().close(); }
            @SuppressWarnings("unchecked")
            ArgumentCaptor<org.flexlb.cache.domain.CacheMatchQuery> candidates = ArgumentCaptor.forClass(org.flexlb.cache.domain.CacheMatchQuery.class);
            verify(cache).findMatchingEngines(candidates.capture());
            org.junit.jupiter.api.Assertions.assertNull(candidates.getValue().group());
            assertEquals(PlacementResult.Status.BLOCKED, result.status(), "do not fall back to the colder idle P");
            assertEquals("10.0.0.2:8080@0", result.diagnostics().get("selectedEndpoint"));
        } finally {
            occupants.forEach(item -> endpoint.releaseCommittedItem(item.execution()));
            reservations.forEach(org.flexlb.balance.endpoint.PrefillState.RouteReservation::close);
        }
        try (SelectedRole recovered = select()) {
            assertEquals("10.0.0.2", recovered.serverStatus().getServerIp());
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
        when(cache.findMatchingEngines(any())).thenReturn(new CacheMatchResult(
                Map.of(cacheLeader + ":8080", HostCacheMatch.local(5)),
                CacheMatchSource.KVCM, 0L, 100L));

        try (SelectedRole selected = select()) {
            assertEquals(cacheLeader, selected.serverStatus().getServerIp(),
                    "cache-first must inspect the complete 750-node fleet");
            var decision = context.getRoutingTelemetry().routingDecisions().get(RoleType.PREFILL);
            assertTrue(decision.snapshotTruncated());
            assertEquals(5, decision.candidates().size());
            assertEquals(5, decision.candidates().stream().map(candidate -> candidate.endpoint()).distinct().count());
            assertEquals(cacheLeader + ":8080@0", decision.candidates().getFirst().endpoint());
            assertTrue(decision.candidates().getFirst().selected());
        }
    }

    @Test
    void exactCostTiesRotateWithoutShrinkingTheCandidateFleet() {
        publish("10.0.0.2", 8080);
        publish("10.0.0.3", 8080);

        Set<String> selectedIps = new HashSet<>();
        for (int index = 0; index < 3; index++) {
            context.getRequest().setRequestId(Long.toString(30_000L + index));
            try (SelectedRole selected = select()) {
                selectedIps.add(selected.serverStatus().getServerIp());
            }
        }

        assertEquals(Set.of("10.0.0.1", "10.0.0.2", "10.0.0.3"), selectedIps);
    }

    @ParameterizedTest
    @CsvSource({"400,5.0,OVER_CAP", "600,60.0,LOW_CACHE_HIT"})
    void cacheAffinityGateUsesTheConfiguredBaselineCandidate(
            long maxExtraTtftMs, double minPrefixHitPercent, String reason) {
        configureAffinity(maxExtraTtftMs, minPrefixHitPercent);

        try (SelectedRole selected = select()) {
            assertEquals("10.0.0.1", selected.serverStatus().getServerIp());
            verify(reporter).reportCacheAffinityDecision(
                    RoleType.PREFILL, "10.0.0.1:8080", reason);
        }
    }

    @Test
    void equalMaximumCacheHitPreservesBestOnlySelection() {
        configureFocusedAffinity();
        PrefillCandidateSet candidates = candidates(
                new long[] {100L, 105L, 1_000L},
                new long[] {500L, 500L, 0L});

        int selectedIndex = strategy.selectBestCandidate(
                candidates, 100L, RoleType.PREFILL, null, 1_000L, config);

        assertEquals(0, selectedIndex,
                "equal cache hit must preserve the minimum-TTFT baseline");
        verify(reporter).reportCacheAffinityDecision(
                RoleType.PREFILL,
                "10.1.0." + (selectedIndex + 1) + ":8080",
                "NO_CACHE_LEAD");
    }

    @Test
    void strictlyGreaterCacheHitOverridesTheBaselineCandidate() {
        configureFocusedAffinity();
        PrefillCandidateSet candidates = candidates(
                new long[] {100L, 105L, 1_000L},
                new long[] {500L, 600L, 0L});

        int selectedIndex = strategy.selectBestCandidate(
                candidates, 100L, RoleType.PREFILL, null, 1_000L, config);

        assertEquals(1, selectedIndex);
        verify(reporter).reportCacheAffinityDecision(
                RoleType.PREFILL, "10.1.0.2:8080", "CACHE_LEADER");
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void equalTtftPrefersCacheOnlyWhenAffinityEnabled(boolean affinityEnabled) {
        if (affinityEnabled) {
            configureFocusedAffinity();
        } else {
            config.getRouter().getRoles().getPrefill().setCacheAffinity(null);
        }
        PrefillCandidateSet candidates = candidates(
                new long[] {100L, 100L, 105L},
                new long[] {100L, 900L, 900L});

        Set<Integer> selected = new HashSet<>();
        for (int attempt = 0; attempt < 6; attempt++) {
            selected.add(strategy.selectBestCandidate(
                    candidates, 100L, RoleType.PREFILL, null, 1_000L, config));
        }

        assertEquals(affinityEnabled ? Set.of(1) : Set.of(0, 1), selected);
    }

    @Test
    void equalTtftAndCacheHitRotateAmongBestCandidates() {
        configureFocusedAffinity();
        PrefillCandidateSet candidates = candidates(
                new long[] {100L, 100L, 100L},
                new long[] {900L, 100L, 900L});

        Set<Integer> selected = new HashSet<>();
        for (int attempt = 0; attempt < 6; attempt++) {
            selected.add(strategy.selectBestCandidate(
                    candidates, 100L, RoleType.PREFILL, null, 1_000L, config));
        }

        assertEquals(Set.of(0, 2), selected);
    }

    @Test
    void cacheLeaderRemainsPreferredAcrossRepeatedSelections() {
        configureAffinity(600L, 5.0);

        try (SelectedRole selected = select()) {
            assertEquals("10.0.0.2", selected.serverStatus().getServerIp());
        }
        context.getRequest().setRequestId("20002");

        try (SelectedRole selected = select()) {
            assertEquals("10.0.0.2", selected.serverStatus().getServerIp());
        }
        verify(reporter, times(2))
                .reportCacheAffinityDecision(
                        RoleType.PREFILL, "10.0.0.2:8080", "CACHE_LEADER");
    }

    @ParameterizedTest
    @CsvSource({"false,false", "true,false", "false,true", "true,true"})
    void unknownEngineWorkDoesNotHideThePreferredPrefill(
            boolean missingTokenDemand, boolean preferCacheLeader) {
        if (preferCacheLeader) {
            configureAffinity(600L, 5.0);
        }
        String expectedIp = preferCacheLeader ? "10.0.0.2" : "10.0.0.1";
        var endpoint = (PrefillEndpoint)
                registry.get(RoleType.PREFILL, expectedIp + ":8080@0");
        var status = endpoint.getStatus();
        var response = StrategyTestSupport.response(RoleType.PREFILL, true,
                1_000_000L, 1_000_000L, status.appliedStatusCursor().statusVersion() + 1L);
        var task = new TaskInfo();
        task.setRequestId("1");
        task.setPhase(TaskPhase.RUNNING);
        task.setInputLength(missingTokenDemand ? 0L : 100L);
        response.setRunningQueryLen(1L);
        response.setRunningTaskInfo(Map.of("1", task));
        status.lock.lock();
        try {
            endpoint.applyPreparedStatus(status, WorkerStatusTestFixtures.prepare(status, WorkerStatusTestFixtures.observation(status, response))).run();
        } finally {
            status.lock.unlock();
        }
        assertTrue(endpoint.captureRouteProjectionInputs().work().hasUnknownWork());

        context.getRequest().setRequestId("40001");
        var result = strategy.select(context, RoleType.PREFILL, null);
        assertEquals(PlacementResult.Status.SUCCESS, result.status());
        try (SelectedRole selected = result.value()) {
            assertEquals(expectedIp, selected.serverStatus().getServerIp());
        }
    }

    private SelectedRole select() {
        PlacementResult<SelectedRole, PlacementKey> result =
                strategy.select(context, RoleType.PREFILL, null);
        assertEquals(PlacementResult.Status.SUCCESS, result.status());
        return result.value();
    }

    private void configureFocusedAffinity() {
        RoutingConfig.CacheAffinityConfig affinity =
                new RoutingConfig.CacheAffinityConfig();
        affinity.setMaxExtraTtftMs(10L);
        affinity.setMinPrefixHitPercent(0.0);
        config.getRouter().getRoles().getPrefill().setCacheAffinity(affinity);
    }

    private PrefillCandidateSet candidates(long[] ttftMs, long[] hitTokens) {
        assertEquals(ttftMs.length, hitTokens.length);
        PrefillCandidateSet candidates = new PrefillCandidateSet();
        candidates.reset(ttftMs.length);
        for (int i = 0; i < ttftMs.length; i++) {
            String ip = "10.1.0." + (i + 1);
            PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
            when(endpoint.getIp()).thenReturn(ip);
            WorkerStatus status = mock(WorkerStatus.class);
            when(status.getMetricIpPort()).thenReturn(ip + ":8080");
            when(endpoint.getStatus()).thenReturn(status);
            candidates.addCandidate(
                    ip + ":8080",
                    endpoint,
                    new RouteProjection.Candidate(
                            ttftMs[i],
                            ttftMs[i],
                            RouteProjection.Candidate.InitialHeadDisposition.NONE,
                            "",
                            hitTokens[i],
                            hitTokens[i]),
                    i + 1L);
        }
        return candidates;
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

        publish("10.0.0.2", 8080);
        context.getRequest().setRequestId("20001");
        context.getRequest().setBlockCacheKeys(List.of(1L, 2L, 3L, 4L, 5L));
        context.getRequest().setCacheKeyBlockSize(100L);
        when(cache.findMatchingEngines(any())).thenReturn(new CacheMatchResult(
                Map.of("10.0.0.2:8080", HostCacheMatch.local(5)),
                CacheMatchSource.KVCM, 0L, 100L));
    }

    private void publish(String ip, int port) {
        WorkerStatus status = StrategyTestSupport.workerStatus(
                RoleType.PREFILL, null, ip, port, port + 1,
                true, 1_000_000L, 1_000_000L);
        StrategyTestSupport.publishEndpoint(
                registry, RoleType.PREFILL, status.getLogicalIpPort(), status);
    }
}
