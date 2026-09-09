package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.QueueOrderingConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.config.VictimStage;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
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
        when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of());
        reporter = mock(EngineHealthReporter.class);
        strategy = new CostBasedPrefillStrategy(
                new WorkerDirectory(registry), cache, reporter);

        Request request = new Request();
        request.setRequestId(10_001L);
        request.setSeqLen(1_000L);
        request.setPriority(50);
        request.setBlockCacheKeys(List.of());
        context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() + 60_000L));
    }

    @AfterEach
    void tearDown() {
        registry.close();
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
    @CsvSource({"5,1,false,BLOCKED", "50,1,false,BLOCKED", "5,2,false,SUCCESS",
            "50,1,true,SUCCESS", "10,1,true,BLOCKED", "5,1,true,BLOCKED"})
    void fullRequestCapacityIsSelectableOnlyWithEligibleQueuedPreemption(
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
        PrefillEndpoint endpoint = (PrefillEndpoint) registry.get(RoleType.PREFILL, "10.0.0.1:8080");
        Request queuedRequest = new Request();
        queuedRequest.setRequestId(9_001L);
        queuedRequest.setSeqLen(1_000L);
        queuedRequest.setPriority(10);
        BalanceContext queuedContext = new BalanceContext();
        queuedContext.setRequest(queuedRequest);
        queuedContext.setConfig(config);
        queuedContext.setSchedulingMetadata(SchedulingMetadata.explicit(
                10, System.currentTimeMillis() + 120_000L));
        ScheduledRequest queued = new ScheduledRequest(queuedContext, new CompletableFuture<>(),
                null, null, null, endpoint, null, null, System.currentTimeMillis());
        assertTrue(StrategyTestSupport.offer(endpoint, queued));
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                incomingPriority, System.currentTimeMillis() + 120_000L));

        PlacementResult<SelectedRole, RoleType> result = strategy.select(context, RoleType.PREFILL, null);

        assertEquals(expectedStatus, result.status());
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
                    Mockito.eq(RoleType.PREFILL), Mockito.eq("10.0.0.1"),
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

        try (SelectedRole selected = select()) {
            assertEquals(cacheLeader, selected.serverStatus().getServerIp(),
                    "cache-first must inspect the complete 750-node fleet");
        }
    }

    @Test
    void exactCostTiesRotateWithoutShrinkingTheCandidateFleet() {
        publish("10.0.0.2", 8080);
        publish("10.0.0.3", 8080);

        Set<String> selectedIps = new HashSet<>();
        for (int index = 0; index < 3; index++) {
            context.getRequest().setRequestId(30_000L + index);
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
                    RoleType.PREFILL, "10.0.0.1", reason);
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
                "10.1.0." + (selectedIndex + 1),
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
                RoleType.PREFILL, "10.1.0.2", "CACHE_LEADER");
    }

    @Test
    void cacheLeaderRemainsPreferredAcrossRepeatedSelections() {
        configureAffinity(600L, 5.0);

        try (SelectedRole selected = select()) {
            assertEquals("10.0.0.2", selected.serverStatus().getServerIp());
        }
        context.getRequest().setRequestId(20_002L);

        try (SelectedRole selected = select()) {
            assertEquals("10.0.0.2", selected.serverStatus().getServerIp());
        }
        verify(reporter, times(2))
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
            endpoint.applyPreparedStatus(status, status.prepareNewStatus(status.freezeStatusResponse(response))).run();
        } finally {
            status.lock.unlock();
        }
        assertTrue(endpoint.captureRouteProjectionInputs().work().hasUnknownWork());

        context.getRequest().setRequestId(40_001L);
        var result = strategy.select(context, RoleType.PREFILL, null);
        assertEquals(PlacementResult.Status.SUCCESS, result.status());
        try (SelectedRole selected = result.value()) {
            assertEquals("10.0.0.1", selected.serverStatus().getServerIp());
        }
    }

    private SelectedRole select() {
        PlacementResult<SelectedRole, RoleType> result =
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
        context.getRequest().setRequestId(20_001L);
        context.getRequest().setBlockCacheKeys(List.of(1L, 2L, 3L, 4L, 5L));
        context.getRequest().setCacheKeyBlockSize(100L);
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
