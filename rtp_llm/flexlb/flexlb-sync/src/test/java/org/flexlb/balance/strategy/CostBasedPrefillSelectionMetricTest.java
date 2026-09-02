package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.resource.PrefillResourceMeasure;
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
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class CostBasedPrefillSelectionMetricTest {

    @Test
    void selectionDoesNotOwnTheRequestDeadline() {
        try (Fixture fixture = fixture(false, false)) {
            fixture.context().setSchedulingMetadata(
                    SchedulingMetadata.explicit(
                            50, System.currentTimeMillis() - 1L));

            EndpointSelection selected = fixture.strategy().selectForQueue(
                    fixture.context(), RoleType.PREFILL, null);
            assertTrue(selected.selected());
            selected.endpoint().close();
        }
    }

    @ParameterizedTest
    @CsvSource({
            "false,false,NON_BATCH",
            "false,true,BATCH",
            "true,false,NON_BATCH",
            "true,true,BATCH"
    })
    void everyModeReportsTheEstimatesUsedByItsSharedSelectionExit(
            boolean shortestTtft,
            boolean batchDelivery,
            String deliveryMode) {
        try (Fixture fixture = fixture(shortestTtft, batchDelivery)) {
            EndpointSelection selected = fixture.strategy().selectForQueue(
                    fixture.context(), RoleType.PREFILL, null);
            assertTrue(selected.selected());
            try (SelectedRole role = selected.endpoint()) {
                ArgumentCaptor<Long> ttft = ArgumentCaptor.forClass(Long.class);
                ArgumentCaptor<Long> execution = ArgumentCaptor.forClass(Long.class);
                verify(fixture.reporter()).reportPrefillSelectedEstimates(
                        Mockito.eq(RoleType.PREFILL),
                        Mockito.eq("10.0.0.1"),
                        Mockito.eq(deliveryMode),
                        ttft.capture(),
                        execution.capture());
                assertEquals(role.serverStatus().getPrefillTime(), ttft.getValue());
                assertEquals(role.prefillWorkMs(), execution.getValue());
            }
        }
    }

    @ParameterizedTest
    @CsvSource({"false", "true"})
    void telemetryFailureNeverChangesAValidSelection(boolean shortestTtft) {
        try (Fixture fixture = fixture(shortestTtft, false)) {
            doThrow(new IllegalStateException("metrics unavailable"))
                    .when(fixture.reporter())
                    .reportPrefillSelectedEstimates(
                            any(), any(), any(),
                            Mockito.anyLong(), Mockito.anyLong());

            EndpointSelection selected = fixture.strategy().selectForQueue(
                    fixture.context(), RoleType.PREFILL, null);
            assertTrue(selected.selected());
            try (SelectedRole role = selected.endpoint()) {
                assertTrue(role.serverStatus().isSuccess());
                assertEquals("10.0.0.1", role.serverStatus().getServerIp());
            }
        }
    }

    private static Fixture fixture(
            boolean shortestTtft,
            boolean batchDelivery) {
        FlexlbConfig config = new FlexlbConfig();
        config.setDispatcher(batchDelivery
                ? new DispatcherConfig()
                : DispatcherConfig.nonBatch());
        RoutingConfig.FormulaEstimatorConfig estimator =
                (RoutingConfig.FormulaEstimatorConfig) config.getRouter()
                        .getRoles().getPrefill().getExecutionTimeEstimator();
        estimator.setExpression("sum(computeTokens)");
        RoutingConfig.EstimatedTtftSelectorConfig selector =
                (RoutingConfig.EstimatedTtftSelectorConfig) config.getRouter()
                        .getRoles().getPrefill().getSelector();
        if (shortestTtft) {
            selector.setCandidateChoice(
                    new RoutingConfig.LeastRecentlyUsedInPoolConfig());
        }

        ConfigService configService = mock(ConfigService.class);
        when(configService.loadBalanceConfig()).thenReturn(config);
        EndpointRegistry registry = StrategyTestSupport.endpointRegistry(configService);
        WorkerStatus status = StrategyTestSupport.workerStatus(
                RoleType.PREFILL, null, "10.0.0.1", 8080, 8081,
                true, 1_000_000L, 1_000_000L);
        registry.registerPreinitializedEndpoint(
                RoleType.PREFILL, "10.0.0.1:8080", status);

        PrefillResourceMeasure measure = mock(PrefillResourceMeasure.class);
        when(measure.isResourceAvailable(anyLong())).thenReturn(true);
        CacheAwareService cache = mock(CacheAwareService.class);
        when(cache.findMatchingEngines(any(), any(), any()))
                .thenReturn(Map.of());
        EngineHealthReporter reporter = mock(EngineHealthReporter.class);
        CostBasedPrefillStrategy strategy = new CostBasedPrefillStrategy(
                new WorkerDirectory(registry), cache, measure, reporter);

        Request request = new Request();
        request.setRequestId(10_001L);
        request.setSeqLen(1_000L);
        request.setPriority(50);
        request.setBlockCacheKeys(List.of());
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        context.setConfig(config);
        context.setSchedulingMetadata(SchedulingMetadata.explicit(
                50, System.currentTimeMillis() + 60_000L));
        return new Fixture(strategy, context, reporter, registry);
    }

    private record Fixture(
            CostBasedPrefillStrategy strategy,
            BalanceContext context,
            EngineHealthReporter reporter,
            EndpointRegistry registry) implements AutoCloseable {
        @Override
        public void close() {
            registry.close();
        }
    }
}
