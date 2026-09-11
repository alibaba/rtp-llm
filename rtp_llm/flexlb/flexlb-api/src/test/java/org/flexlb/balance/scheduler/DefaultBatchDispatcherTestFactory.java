package org.flexlb.balance.scheduler;

import org.flexlb.config.ConfigService;
import org.flexlb.engine.grpc.EngineGrpcClient;

/** Test-source bridge to the package-visible dispatcher sizing injection. */
public final class DefaultBatchDispatcherTestFactory {

    private DefaultBatchDispatcherTestFactory() {
    }

    public static DefaultBatchDispatcher create(EngineGrpcClient grpcClient,
                                                ConfigService configService,
                                                int poolSize,
                                                int queueSize) {
        return new DefaultBatchDispatcher(
                grpcClient, configService, null, poolSize, queueSize);
    }
}
