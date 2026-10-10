package org.flexlb.balance.scheduler;

import org.flexlb.config.ConfigService;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.monitor.DeliveryMetricsReporter;

/** Creates a bounded dispatcher through its production configuration. */
public final class DeliveryDispatcherTestFactory {

    private DeliveryDispatcherTestFactory() {
    }

    public static DeliveryDispatcher create(EngineGrpcClient grpcClient,
                                            ConfigService configService,
                                            DeliveryMetricsReporter reporter,
                                            int poolSize,
                                            int queueSize) {
        var config = org.mockito.Mockito.spy(configService.loadBalanceConfig());
        var sizing = org.mockito.Mockito.spy(config.getInternalRuntime());
        org.mockito.Mockito.doReturn(poolSize).when(sizing).getBatchDispatchThreads();
        org.mockito.Mockito.doReturn(queueSize).when(sizing).getBatchDispatchQueueCapacity();
        org.mockito.Mockito.doReturn(sizing).when(config).getInternalRuntime();
        var dispatcherService = org.mockito.Mockito.mock(ConfigService.class,
                org.mockito.AdditionalAnswers.delegatesTo(configService));
        org.mockito.Mockito.doReturn(config).when(dispatcherService).loadBalanceConfig();
        return new RuntimeSizedDispatcher(grpcClient, dispatcherService, reporter);
    }

    /** Executor sizing belongs to Runtime, so carry the same test configuration into composition. */
    static ConfigService runtimeConfigFor(DeliveryDispatcher dispatcher, ConfigService fallback) {
        return dispatcher instanceof RuntimeSizedDispatcher sized ? sized.config : fallback;
    }

    private static final class RuntimeSizedDispatcher extends DeliveryDispatcher {
        private final ConfigService config;

        private RuntimeSizedDispatcher(EngineGrpcClient client, ConfigService config,
                                       DeliveryMetricsReporter reporter) {
            super(client, config, null, reporter, null);
            this.config = config;
        }
    }
}
