package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.mockito.ArgumentCaptor;
import org.mockito.Mockito;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class CostBasedPrefillSelectionMetricTest {
    private FlexlbConfig config;
    private EndpointRegistry registry;
    private CacheAwareService cache;
    private EngineHealthReporter reporter;
    private CostBasedPrefillStrategy strategy;
    private BalanceContext context;

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
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
        configureAffinity(600L, 5.0,
                RoutingConfig.CandidateChoiceType.BEST_ONLY);

        try (SelectedRole selected = select()) {
            assertEquals("10.0.0.2", selected.serverStatus().getServerIp());
            verify(reporter).reportCacheAffinityDecision(
                    RoleType.PREFILL, "10.0.0.2", "CACHE_LEADER");
        }
    }

    @ParameterizedTest
    @CsvSource({"400,5.0,OVER_CAP", "600,60.0,LOW_CACHE_HIT"})
    void cacheAffinityGateFallsBackToTheBaselineCandidate(
            long maxExtraTtftMs, double minPrefixHitPercent, String reason) {
        configureAffinity(maxExtraTtftMs, minPrefixHitPercent,
                RoutingConfig.CandidateChoiceType.BEST_ONLY);

        try (SelectedRole selected = select()) {
            assertEquals("10.0.0.1", selected.serverStatus().getServerIp());
            verify(reporter).reportCacheAffinityDecision(
                    RoleType.PREFILL, "10.0.0.1", reason);
        }
    }

    @Test
    void lruUsesCachePreferenceThenFallsBackToItsBaselinePool() {
        configureAffinity(600L, 5.0,
                RoutingConfig.CandidateChoiceType.LEAST_RECENTLY_USED_IN_POOL);

        try (SelectedRole selected = select()) {
            assertEquals("10.0.0.2", selected.serverStatus().getServerIp());
        }
        PrefillEndpoint cacheEndpoint = (PrefillEndpoint)
                registry.get(RoleType.PREFILL, "10.0.0.2:8080");
        cacheEndpoint.getLastSelectedTime().set(Long.MAX_VALUE);
        context.getRequest().setRequestId(20_002L);

        try (SelectedRole selected = select()) {
            assertEquals("10.0.0.1", selected.serverStatus().getServerIp());
        }
        assertEquals(Long.MAX_VALUE, cacheEndpoint.getLastSelectedTime().get());
        verify(reporter).reportCacheAffinityDecision(
                RoleType.PREFILL, "10.0.0.1", "CACHE_AFFINITY_FALLBACK");
    }

    private SelectedRole select() {
        PlacementResult<SelectedRole, RoleType> result =
                strategy.selectForQueue(context, RoleType.PREFILL, null);
        assertEquals(PlacementResult.Status.SUCCESS, result.status());
        return result.value();
    }

    private void configureAffinity(
            long maxExtraTtftMs,
            double minPrefixHitPercent,
            RoutingConfig.CandidateChoiceType choiceType) {
        config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator()
                .setExpression("sum(computeTokens) + 2*sum(hitCacheTokens)");
        RoutingConfig.CandidateChoiceConfig candidateChoice = config.getRouter()
                .getRoles().getPrefill().getCandidateChoice();
        candidateChoice.setType(choiceType);
        candidateChoice.getOutlierRejection()
                .setMaxPendingVsAverageMultiplier(0.0);
        candidateChoice.getOutlierRejection()
                .setMaxProjectedDrainVsAverageMultiplier(0.0);
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
