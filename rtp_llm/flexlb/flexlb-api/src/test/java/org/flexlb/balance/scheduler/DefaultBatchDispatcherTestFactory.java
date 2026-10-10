package org.flexlb.balance.scheduler;

import org.flexlb.config.ConfigService;
import org.flexlb.engine.grpc.EngineGrpcClient;

/** Creates a bounded dispatcher through its production configuration. */
public final class DefaultBatchDispatcherTestFactory {

    private DefaultBatchDispatcherTestFactory() {
    }

    public static DefaultBatchDispatcher create(EngineGrpcClient grpcClient,
                                                ConfigService configService,
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
        return new DefaultBatchDispatcher(grpcClient, dispatcherService, null);
    }
}
