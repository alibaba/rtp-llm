package org.flexlb.balance.scheduler;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.eviction.DecodeCapacityAcquirer;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.mock.env.MockEnvironment;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class PlacementConfigurationTest {
    @ParameterizedTest
    @EnumSource(DispatcherConfig.Type.class)
    void springAssemblyUsesOneDispatcherBeforeServingAndRuntimeCloses(
            DispatcherConfig.Type dispatcherType) throws Exception {
        var config = SchedulingTestConfig.batchConfig();
        if (dispatcherType == DispatcherConfig.Type.NON_BATCH) {
            config.setScheduler(ConfigTestSupport.direct());
            SchedulingTestConfig.useNonBatchDispatcher(config);
        }
        var environment = spy(new MockEnvironment()
                .withProperty("server.port", "7001"));
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        RequestWorkerSelector router = mock(RequestWorkerSelector.class);
        var request = RequestProtocolTestSupport.context(config, 8972L);
        when(router.select(request, null)).thenReturn(
                PlacementResult.rejected(Response.error(StrategyErrorType.NO_PREFILL_WORKER)));
        var batchDispatcher = spy(new DeliveryDispatcher(null, service, null, mock(DeliveryMetricsReporter.class), environment));
        RequestRepository repository;
        try (var spring = new AnnotationConfigApplicationContext()) {
            spring.setEnvironment(environment);
            spring.registerBean(ConfigService.class, () -> service);
            spring.registerBean(RequestWorkerSelector.class, () -> router);
            spring.registerBean(org.flexlb.service.VitCacheSelector.class,
                    () -> mock(org.flexlb.service.VitCacheSelector.class));
            spring.registerBean(EndpointRegistry.class, () -> mock(EndpointRegistry.class));
            spring.registerBean(DecodeCapacityAcquirer.class, () -> mock(DecodeCapacityAcquirer.class));
            spring.registerBean(DeliveryMetricsReporter.class, () -> mock(DeliveryMetricsReporter.class));
            spring.registerBean(RequestSchedulerReporter.class, () -> mock(RequestSchedulerReporter.class));
            spring.registerBean(RecentCacheKeyTraceReporter.class, () -> mock(RecentCacheKeyTraceReporter.class));
            spring.registerBean(EngineGrpcClient.class, () -> mock(EngineGrpcClient.class));
            spring.registerBean(DeliveryDispatcher.class, () -> batchDispatcher);
            spring.register(PlacementAvailability.class, RequestRepository.class, PlacementConfiguration.class, SchedulerRuntime.class);
            spring.refresh();

            assertEquals(1, spring.getBeansOfType(DeliveryDispatcher.class).size());
            var deliveryStrategy = spring.getBean(DeliveryDispatcher.class);
            assertSame(batchDispatcher, deliveryStrategy);
            assertEquals(dispatcherType, deliveryStrategy.mode());
            var scheduler = spring.getBean(RequestScheduler.class);
            if (dispatcherType == DispatcherConfig.Type.BATCH) {
                assertInstanceOf(DeliveryDispatcher.class, deliveryStrategy);
                assertInstanceOf(QueuedRequestScheduler.class, scheduler);
            } else {
                assertInstanceOf(DeliveryDispatcher.class, deliveryStrategy);
                assertInstanceOf(DirectRequestScheduler.class, scheduler);
                verify(environment, never()).getProperty("server.port");
            }
            repository = spring.getBean(RequestRepository.class);
            request.data().setGenerateInputPb(com.google.protobuf.ByteString.copyFromUtf8("input"));
            var future = scheduler.submit(request);
            assertSame(future, request.data().getFuture());
            assertEquals(StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(), future.get(3, TimeUnit.SECONDS).getCode());
        }
        assertTrue(repository.isClosed());
        assertTrue(repository.isClosed());
    }
}
