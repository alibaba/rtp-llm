package org.flexlb.balance.scheduler;

import org.flexlb.config.ConfigService;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.monitor.DeliveryMetricsReporter;

/**
 * Creates a bounded dispatcher through its production configuration.
 */
public final class DeliveryDispatcherTestFactory {

    private DeliveryDispatcherTestFactory() {}

    public static DeliveryCoordinator create(
            EngineGrpcClient grpcClient,
            ConfigService configService,
            DeliveryMetricsReporter reporter,
            int poolSize,
            int queueSize) {
        var config = org.mockito.Mockito.spy(configService.loadBalanceConfig());
        var sizing = org.mockito.Mockito.spy(config.getInternalRuntime());
        org.mockito.Mockito.doReturn(poolSize).when(sizing).getBatchDispatchThreads();
        org.mockito.Mockito.doReturn(queueSize).when(sizing).getBatchDispatchQueueCapacity();
        org.mockito.Mockito.doReturn(sizing).when(config).getInternalRuntime();
        var dispatcherService =
                org.mockito.Mockito.mock(
                        ConfigService.class,
                        org.mockito.AdditionalAnswers.delegatesTo(configService));
        org.mockito.Mockito.doReturn(config).when(dispatcherService).loadBalanceConfig();
        return new RuntimeSizedDispatcher(grpcClient, dispatcherService, reporter);
    }

    /**
     * Executor sizing belongs to Runtime, so carry the same test configuration into composition.
     */
    static ConfigService runtimeConfigFor(DeliveryCoordinator dispatcher, ConfigService fallback) {
        return dispatcher instanceof RuntimeSizedDispatcher sized ? sized.config : fallback;
    }

    /**
     * Retains the executor sizing configuration used by the test dispatcher.
     */
    private static final class RuntimeSizedDispatcher extends DeliveryCoordinator {
        private final ConfigService config;

        private RuntimeSizedDispatcher(
                EngineGrpcClient client, ConfigService config, DeliveryMetricsReporter reporter) {
            super(
                    new RequestRepository(),
                    client,
                    config,
                    reporter,
                    new org.flexlb.balance.scheduler.SchedulerConfiguration()
                            .decodeAdmissionCapacity(config),
                    org.mockito.Mockito.mock(DeliveryExecution.class));
            this.config = config;
        }
    }
}
