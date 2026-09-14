package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.domain.CacheMatchResult;
import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.pv.RoutingDecision;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.junit.jupiter.api.Test;

import java.time.Duration;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class CostBasedPrefillBatchIntegrationTest {

    @Test
    void retainsEachRequestsCacheSnapshotAndMaterializesItsExactWorker() {
        FlexlbConfig config = new FlexlbConfig();
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getRouter().getRoles().getPrefill().getCandidateChoice()
                .setType(RoutingConfig.CandidateChoiceType.BEST_ONLY);
        config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator()
                .setExpression("sum(computeTokens)");
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        EndpointRegistry registry = StrategyTestSupport.endpointRegistry(configService);
        try {
            for (String ip : List.of("10.0.0.1", "10.0.0.2")) {
                WorkerStatus status = StrategyTestSupport.workerStatus(
                        RoleType.PREFILL, null, ip, 8080, 8081, true, 1_000_000L, 1_000_000L);
                StrategyTestSupport.publishEndpoint(registry, RoleType.PREFILL,
                        status.getLogicalIpPort(), status);
            }
            CacheAwareService cache = mock(CacheAwareService.class);
            when(cache.findMatchingEngines(any())).thenReturn(
                    CacheMatchResult.empty(CacheMatchSource.LOCAL_SYNC),
                    new CacheMatchResult(Map.of("10.0.0.1:8080@0", HostCacheMatch.local(9)),
                            CacheMatchSource.LOCAL_SYNC, 0L, 100L));
            CostBasedBatchedPrefillStrategy strategy = new CostBasedBatchedPrefillStrategy(
                    new WorkerDirectory(registry), cache, mock(EngineHealthReporter.class));
            var results = strategy.selectBatch(List.of(
                    new PrefillStrategy.BatchRequest(
                            context(config, "a"), RoleType.PREFILL, null),
                    new PrefillStrategy.BatchRequest(
                            context(config, "b"), RoleType.PREFILL, null)));
            try {
                assertEquals(List.of(PlacementResult.Status.SUCCESS, PlacementResult.Status.SUCCESS),
                        results.stream().map(PlacementResult::status).toList());
                assertEquals(List.of("10.0.0.2", "10.0.0.1"),
                        results.stream().map(r -> r.value().serverStatus().getServerIp()).toList());
                assertEquals(List.of(0L, 900L), results.stream()
                        .map(r -> r.value().serverStatus().getDebugInfo().getHitCacheLen()).toList());
            } finally {
                results.stream().filter(r -> r.value() != null).forEach(r -> r.value().close());
            }
        } finally {
            registry.close();
        }
    }

    @Test
    void productionBatchSharesWorkerKvCapacity() {
        FlexlbConfig config = config("0");
        EndpointRegistry registry = registryWithWorkers(config, "10.0.0.1");
        try {
            CostBasedBatchedPrefillStrategy strategy = new CostBasedBatchedPrefillStrategy(
                    new WorkerDirectory(registry),
                    emptyCache(),
                    mock(EngineHealthReporter.class));
            var results = strategy.selectBatch(List.of(
                    batchRequest(config, "kv-a"),
                    batchRequest(config, "kv-b")));
            try {
                assertEquals(PlacementResult.Status.SUCCESS, results.get(0).status());
                assertEquals(PlacementResult.Status.BLOCKED, results.get(1).status());
            } finally {
                results.stream()
                        .filter(result -> result.value() != null)
                        .forEach(result -> result.value().close());
            }
        } finally {
            registry.close();
        }
    }

    @Test
    void recordsJointGlobalPlanningSeparatelyFromBestOnlyPolicy() {
        FlexlbConfig config = config("0");
        EndpointRegistry registry = registryWithWorkers(config, "10.0.0.1", "10.0.0.2");
        try {
            CostBasedBatchedPrefillStrategy strategy = new CostBasedBatchedPrefillStrategy(
                    new WorkerDirectory(registry), emptyCache(), mock(EngineHealthReporter.class));
            BalanceContext first = context(config, "trace-a");
            BalanceContext second = context(config, "trace-b");
            first.getRequest().setSeqLen(60L);
            second.getRequest().setSeqLen(60L);
            var results = strategy.selectBatch(List.of(
                    new PrefillStrategy.BatchRequest(first, RoleType.PREFILL, null),
                    new PrefillStrategy.BatchRequest(second, RoleType.PREFILL, null)));
            try {
                RoutingDecision firstDecision = first.getRoutingTelemetry()
                        .routingDecisions().get(RoleType.PREFILL);
                RoutingDecision secondDecision = second.getRoutingTelemetry()
                        .routingDecisions().get(RoleType.PREFILL);
                assertEquals("CostBasedBatchedPrefill", firstDecision.strategy());
                assertEquals("GLOBAL_BATCH", firstDecision.selectionReason());
                assertEquals("BEST_ONLY", firstDecision.prefillPolicy().candidateChoice());
                assertNotNull(firstDecision.globalPlanning());
                assertEquals(2, firstDecision.globalPlanning().requestCount());
                assertEquals(firstDecision.globalPlanning().decisionId(),
                        secondDecision.globalPlanning().decisionId());
            } finally {
                results.stream()
                        .filter(result -> result.value() != null)
                        .forEach(result -> result.value().close());
            }
        } finally {
            registry.close();
        }
    }

    @Test
    void emptyBatchReturnsNoPlacements() {
        CostBasedBatchedPrefillStrategy strategy = new CostBasedBatchedPrefillStrategy(
                mock(WorkerDirectory.class),
                mock(CacheAwareService.class),
                mock(EngineHealthReporter.class));
        assertEquals(List.of(), strategy.selectBatch(List.of()));
    }

    @Test
    void telemetryFailureReleasesTheSelectedGenerationPin() {
        FlexlbConfig config = config("0");
        EndpointRegistry registry = registryWithWorkers(config, "10.0.0.1");
        try {
            EngineHealthReporter reporter = mock(EngineHealthReporter.class);
            doThrow(new IllegalStateException("metric failed"))
                    .when(reporter)
                    .reportCacheHitMetrics(
                            eq(RoleType.PREFILL), anyString(), anyLong(), anyDouble());
            CostBasedBatchedPrefillStrategy strategy = new CostBasedBatchedPrefillStrategy(
                    new WorkerDirectory(registry), emptyCache(), reporter);

            assertThrows(IllegalStateException.class, () -> strategy.selectBatch(
                    List.of(batchRequest(config, "metric-failure"))));
            assertTimeoutPreemptively(Duration.ofSeconds(1), registry::close);
        } finally {
            registry.close();
        }
    }

    private static FlexlbConfig config(String estimatorExpression) {
        FlexlbConfig config = new FlexlbConfig();
        config.setDispatcher(DispatcherConfig.nonBatch());
        config.getRouter().getRoles().getPrefill().getCandidateChoice()
                .setType(RoutingConfig.CandidateChoiceType.BEST_ONLY);
        config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator()
                .setExpression(estimatorExpression);
        return config;
    }

    private static EndpointRegistry registryWithWorkers(
            FlexlbConfig config, String... ips) {
        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        EndpointRegistry registry = StrategyTestSupport.endpointRegistry(configService);
        for (String ip : ips) {
            WorkerStatus status = StrategyTestSupport.workerStatus(
                    RoleType.PREFILL, null, ip, 8080, 8081, true, 100L, 100L);
            StrategyTestSupport.publishEndpoint(registry, RoleType.PREFILL,
                    status.getLogicalIpPort(), status);
        }
        return registry;
    }

    private static CacheAwareService emptyCache() {
        CacheAwareService cache = mock(CacheAwareService.class);
        when(cache.findMatchingEngines(any()))
                .thenReturn(CacheMatchResult.empty(CacheMatchSource.LOCAL_SYNC));
        return cache;
    }

    private static PrefillStrategy.BatchRequest batchRequest(
            FlexlbConfig config, String requestId) {
        BalanceContext context = context(config, requestId);
        context.getRequest().setSeqLen(60L);
        return new PrefillStrategy.BatchRequest(
                context, RoleType.PREFILL, null);
    }

    private static BalanceContext context(FlexlbConfig config, String id) {
        Request request = new Request();
        request.setRequestId(id);
        request.setSeqLen(1_000L);
        request.setPriority(50);
        request.setBlockCacheKeys(List.of());
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(50, System.currentTimeMillis() + 60_000L));
        return context;
    }
}
