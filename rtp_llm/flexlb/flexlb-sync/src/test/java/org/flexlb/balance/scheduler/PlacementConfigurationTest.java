package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.eviction.DecodeCapacityAcquirer;
import org.flexlb.config.ConfigService;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class PlacementConfigurationTest {
    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void springAssemblyInstallsStrategiesBeforeServingAndRuntimeClosesThem(boolean queued) throws Exception {
        var config = SchedulingTestConfig.batchConfig();
        if (!queued) {
            config.setScheduler(SchedulerConfig.direct());
            SchedulingTestConfig.useNonBatchDispatcher(config);
        }
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        RequestWorkerSelector router = mock(RequestWorkerSelector.class);
        var request = RequestProtocolTestSupport.context(config, 8972L);
        when(router.select(request, null)).thenReturn(
                PlacementResult.rejected(Response.error(StrategyErrorType.NO_PREFILL_WORKER)));
        try (var spring = new AnnotationConfigApplicationContext()) {
            spring.registerBean(ConfigService.class, () -> service);
            spring.registerBean(RequestWorkerSelector.class, () -> router);
            spring.registerBean(EndpointRegistry.class, () -> mock(EndpointRegistry.class));
            spring.registerBean(DecodeCapacityAcquirer.class, () -> mock(DecodeCapacityAcquirer.class));
            spring.registerBean(DeliveryMetricsReporter.class, () -> mock(DeliveryMetricsReporter.class));
            spring.registerBean(RequestSchedulerReporter.class, () -> mock(RequestSchedulerReporter.class));
            spring.registerBean(RecentCacheKeyTraceReporter.class, () -> mock(RecentCacheKeyTraceReporter.class));
            spring.registerBean(org.flexlb.balance.eviction.EngineCancelChannel.class, () -> mock(org.flexlb.balance.eviction.EngineCancelChannel.class));
            spring.registerBean(DefaultBatchDispatcher.class, () -> mock(DefaultBatchDispatcher.class));
            spring.register(PlacementAvailability.class, RequestRepository.class, PlacementConfiguration.class, SchedulerRuntime.class);
            spring.refresh();
            request.setGenerateInputPb(com.google.protobuf.ByteString.copyFromUtf8("input"));
            var future = SchedulerTestSupport.initializedScheduler(spring.getBean(SchedulerRuntime.class)).submit(request);
            assertSame(future, request.getFuture());
            assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(), future.get(3, TimeUnit.SECONDS).getCode());
        }
    }
}
