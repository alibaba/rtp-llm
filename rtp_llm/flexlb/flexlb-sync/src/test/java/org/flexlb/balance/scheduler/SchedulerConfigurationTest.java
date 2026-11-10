package org.flexlb.balance.scheduler;

import org.flexlb.balance.RoutingResult;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.preemption.DecodePreemption;
import org.flexlb.config.ConfigService;
import org.flexlb.config.ConfigTestSupport;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.TheoryCacheHitReporter;
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

/**
 * Checks scheduler assembly and startup ordering, including ownership of the shared dispatcher at shutdown.
 */
class SchedulerConfigurationTest {
    @ParameterizedTest
    @EnumSource(DispatcherConfig.Type.class)
    void springAssemblyUsesOneDispatcherBeforeServingAndRuntimeCloses(
            DispatcherConfig.Type dispatcherType) throws Exception {
        var config = SchedulingTestConfig.batchConfig();
        if (dispatcherType == DispatcherConfig.Type.NON_BATCH) {
            config.setScheduler(ConfigTestSupport.direct());
            SchedulingTestConfig.useNonBatchDispatcher(config);
        }
        var environment = spy(new MockEnvironment().withProperty("server.port", "7001"));
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        RequestRouteSelector router = mock(RequestRouteSelector.class);
        var request = RequestProtocolTestSupport.context(config, 8972L);
        when(router.select(request, null))
                .thenReturn(
                        RoutingResult.rejected(
                                Response.error(StrategyErrorType.NO_PREFILL_WORKER)));
        var batchDispatcher =
                spy(
                        new DeliveryCoordinator(
                                new org.flexlb.balance.scheduler.RequestRepository(),
                                null,
                                service,
                                mock(DeliveryMetricsReporter.class),
                                new org.flexlb.balance.scheduler.SchedulerConfiguration()
                                        .decodeAdmissionCapacity(service),
                                org.mockito.Mockito.mock(
                                        org.flexlb.balance.scheduler.DeliveryExecution.class)));
        RequestRepository repository;
        ScheduledTaskTimer taskTimer;
        ScheduledTaskTimer deadlineTaskTimer;
        try (var spring = new AnnotationConfigApplicationContext()) {
            spring.setEnvironment(environment);
            spring.registerBean(
                    "nacosConfigSource",
                    org.flexlb.service.config.ConfigSource.class,
                    () -> mock(org.flexlb.service.config.ConfigSource.class));
            spring.registerBean(
                    "uniConfigConfigSource",
                    org.flexlb.service.config.ConfigSource.class,
                    () -> mock(org.flexlb.service.config.ConfigSource.class));
            spring.registerBean(
                    "environmentConfigSource",
                    org.flexlb.service.config.ConfigSource.class,
                    () -> mock(org.flexlb.service.config.ConfigSource.class));
            spring.registerBean(ConfigService.class, () -> service);
            spring.registerBean(RequestRouteSelector.class, () -> router);
            spring.registerBean(
                    org.flexlb.service.VitCacheSelector.class,
                    () -> mock(org.flexlb.service.VitCacheSelector.class));
            spring.registerBean(EndpointRegistry.class, () -> mock(EndpointRegistry.class));
            spring.registerBean(DecodePreemption.class, () -> mock(DecodePreemption.class));
            spring.registerBean(
                    DeliveryMetricsReporter.class, () -> mock(DeliveryMetricsReporter.class));
            spring.registerBean(
                    RequestSchedulerReporter.class, () -> mock(RequestSchedulerReporter.class));
            spring.registerBean(
                    TheoryCacheHitReporter.class, () -> mock(TheoryCacheHitReporter.class));
            spring.registerBean(EngineGrpcClient.class, () -> mock(EngineGrpcClient.class));
            spring.registerBean(DeliveryCoordinator.class, () -> batchDispatcher);
            spring.register(
                    RequestConsumer.class,
                    CapacityNotifier.class,
                    RequestRepository.class,
                    SchedulerConfiguration.class);
            spring.refresh();

            assertEquals(1, spring.getBeansOfType(DeliveryCoordinator.class).size());
            var deliveryStrategy = spring.getBean(DeliveryCoordinator.class);
            assertSame(batchDispatcher, deliveryStrategy);
            assertEquals(dispatcherType, deliveryStrategy.mode());
            var scheduler = spring.getBean(RequestScheduler.class);
            if (dispatcherType == DispatcherConfig.Type.BATCH) {
                assertInstanceOf(DeliveryCoordinator.class, deliveryStrategy);
                assertInstanceOf(QueuedRequestScheduler.class, scheduler);
            } else {
                assertInstanceOf(DeliveryCoordinator.class, deliveryStrategy);
                assertInstanceOf(DirectRequestScheduler.class, scheduler);
                verify(environment, never()).getProperty("server.port");
            }
            repository = spring.getBean(RequestRepository.class);
            request.data().setGenerateInputPb(com.google.protobuf.ByteString.copyFromUtf8("input"));
            var future = spring.getBean(RequestConsumer.class).entry.submit(request);
            assertSame(future, request.data().getFuture());
            assertEquals(
                    StrategyErrorType.NO_PREFILL_WORKER.getErrorCode(),
                    future.get(3, TimeUnit.SECONDS).getCode());
            taskTimer = spring.getBean("taskTimer", ScheduledTaskTimer.class);
            deadlineTaskTimer = spring.getBean("deadlineTaskTimer", ScheduledTaskTimer.class);
            spring.getBean(AbstractRequestScheduler.class).shutdown(System.nanoTime());
            taskTimer.schedule(() -> {}, 0L).get(2, TimeUnit.SECONDS);
            deadlineTaskTimer.schedule(() -> {}, 0L).get(2, TimeUnit.SECONDS);
        }
        assertTrue(repository.isClosed());
        org.junit.jupiter.api.Assertions.assertThrows(
                java.util.concurrent.RejectedExecutionException.class,
                () -> taskTimer.schedule(() -> {}, 0L));
        org.junit.jupiter.api.Assertions.assertThrows(
                java.util.concurrent.RejectedExecutionException.class,
                () -> deadlineTaskTimer.schedule(() -> {}, 0L));
    }

    /**
     * Receives the lazily assembled scheduler to verify Spring resolves the shared request entry point.
     */
    static class RequestConsumer {
        final AbstractRequestScheduler entry;

        RequestConsumer(
                @org.springframework.context.annotation.Lazy AbstractRequestScheduler entry) {
            this.entry = entry;
        }
    }
}
