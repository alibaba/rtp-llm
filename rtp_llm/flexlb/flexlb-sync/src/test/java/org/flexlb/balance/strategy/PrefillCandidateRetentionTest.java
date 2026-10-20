package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.cache.monitor.CacheMetricsReporter;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.route.RoleType;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.junit.jupiter.api.Test;

import java.lang.ref.WeakReference;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.anyDouble;
import static org.mockito.Mockito.anyLong;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.reset;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class PrefillCandidateRetentionTest {
    @Test
    void completedSelectionReleasesTheRetiredFleetOnAnIdlePlannerThread() throws Exception {
        WeakReference<WorkerEndpoint> retired = selectThenRetire();
        try {
            for (int attempt = 0; attempt < 30 && !retired.refersTo(null); attempt++) {
                System.gc();
                Thread.sleep(25L);
            }
            assertTrue(retired.refersTo(null), "planning must release retired endpoints and their cache keys");
        } finally {
            var field = CostBasedPrefillStrategy.class.getDeclaredField("CANDIDATES");
            field.setAccessible(true);
            ((ThreadLocal<?>) field.get(null)).remove();
        }
    }

    @Test
    void nestedSelectionCannotResetTheOuterCandidateMetrics() {
        var config = StrategyTestSupport.config();
        config.setDispatcher(ConfigTestSupport.nonBatch());
        config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator().setExpression("sum(computeTokens)");
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var registry = StrategyTestSupport.endpointRegistry(service);
        var status = StrategyTestSupport.workerStatus(RoleType.PREFILL, null, "127.0.0.1", 8080, 8081,
                true, 100_000L, 100_000L);
        StrategyTestSupport.publishEndpoint(registry, RoleType.PREFILL, status.getIpPort(), status);
        CacheAwareService cache = mock(CacheAwareService.class);
        when(cache.findMatchingEngines(any(), any(), any())).thenAnswer(invocation ->
                Map.of(status.getIpPort(), invocation.getArgument(0).equals(List.of(1L)) ? 1 : 2));
        var metrics = mock(CacheMetricsReporter.class);
        var strategy = new CostBasedPrefillStrategy(registry, cache, mock(EngineHealthReporter.class), metrics, service);
        var outer = Request.builder().requestId(90002L).seqLen(1000L)
                .blockCacheKeys(List.of(1L)).cacheKeyBlockSize(100L).build();
        var nested = outer.toBuilder().requestId(90003L).seqLen(2000L).blockCacheKeys(List.of(2L, 3L)).build();
        var metadata = SchedulingMetadata.explicit(50, System.currentTimeMillis() + 60_000L);
        var reentered = new AtomicBoolean();
        doAnswer(invocation -> {
            if (reentered.compareAndSet(false, true)) {
                try (var selected = StrategyTestSupport.select(strategy, nested, metadata, RoleType.PREFILL, null).value()) {
                    assertNotNull(selected);
                }
            }
            return null;
        }).when(metrics).reportCacheHitMetrics(eq(RoleType.PREFILL), anyLong(), anyDouble());
        try {
            try (var selected = StrategyTestSupport.select(strategy, outer, metadata, RoleType.PREFILL, null).value()) {
                assertNotNull(selected);
            }
            verify(metrics).reportRoutingSelectedCacheMatchMetrics(RoleType.PREFILL, 100L, 1000L);
            verify(metrics).reportRoutingSelectedCacheMatchMetrics(RoleType.PREFILL, 200L, 2000L);
            verify(metrics).reportRoutingCandidateMaxCacheMatchMetrics(RoleType.PREFILL, 100L);
            verify(metrics).reportRoutingCandidateMaxCacheMatchMetrics(RoleType.PREFILL, 200L);
        } finally {
            registry.close();
        }
    }

    private WeakReference<WorkerEndpoint> selectThenRetire() {
        var config = StrategyTestSupport.config();
        config.setDispatcher(ConfigTestSupport.nonBatch());
        config.getRouter().getRoles().getPrefill().getExecutionTimeEstimator().setExpression("sum(computeTokens)");
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var registry = StrategyTestSupport.endpointRegistry(service);
        var status = StrategyTestSupport.workerStatus(RoleType.PREFILL, null, "127.0.0.1", 8080, 8081,
                true, 100_000L, 100_000L);
        WorkerEndpoint endpoint = StrategyTestSupport.publishEndpoint(registry, RoleType.PREFILL, status.getIpPort(), status);
        CacheAwareService cache = mock(CacheAwareService.class);
        when(cache.findMatchingEngines(any(), any(), any())).thenReturn(Map.of());
        var strategy = new CostBasedPrefillStrategy(registry, cache, mock(EngineHealthReporter.class),
                mock(CacheMetricsReporter.class), service);
        var request = Request.builder().requestId(90001L).seqLen(1000L).blockCacheKeys(List.of()).build();
        try {
            try (var selected = StrategyTestSupport.select(strategy, request,
                    SchedulingMetadata.explicit(50, System.currentTimeMillis() + 60_000L), RoleType.PREFILL, null).value()) {
                assertNotNull(selected);
            }
            return new WeakReference<>(endpoint);
        } finally {
            registry.close();
            reset(cache); // Mockito records the lazy address view, which also owns its fleet.
        }
    }
}
