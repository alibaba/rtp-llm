package org.flexlb.engine.grpc.config;

import io.netty.util.concurrent.MultithreadEventExecutorGroup;
import org.flexlb.config.ConfigService;
import org.junit.jupiter.api.Test;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Checks that configured callback and event loop sizes are used when creating channel resources.
 */
class ChannelConfigurationSizingTest {
    @Test
    void createsConfiguredCallbackAndIoPools() {
        var config =
                ConfigService.parse(
                        """
                {"requestLifecycle":{"request":{"timeoutMs":1000}},
                 "engineGrpc":{"executorThreads":3,"executorQueueCapacity":7,"eventLoopThreads":2}}
                """);
        var service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        var factory = new ChannelConfiguration(service);
        var callbacks = factory.managedChannelThreadPoolExecutor();
        var clientIo = factory.managedChannelEventLoopGroup();
        var serverIo = factory.grpcServerEventLoopGroup();
        try {
            assertEquals(3, callbacks.getCorePoolSize());
            assertEquals(3, callbacks.getMaximumPoolSize());
            assertEquals(7, callbacks.getQueue().remainingCapacity());
            assertEquals(
                    GrpcCallbackCallerRunsPolicy.class,
                    callbacks.getRejectedExecutionHandler().getClass());
            assertEquals(2, ((MultithreadEventExecutorGroup) clientIo).executorCount());
            assertEquals(4, ((MultithreadEventExecutorGroup) serverIo).executorCount());
        } finally {
            callbacks.shutdownNow();
            clientIo.shutdownGracefully(0, 5, TimeUnit.SECONDS).syncUninterruptibly();
            serverIo.shutdownGracefully(0, 5, TimeUnit.SECONDS).syncUninterruptibly();
        }
    }
}
