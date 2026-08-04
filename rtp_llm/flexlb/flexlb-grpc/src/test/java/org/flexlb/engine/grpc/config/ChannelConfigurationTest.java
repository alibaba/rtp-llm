package org.flexlb.engine.grpc.config;

import io.netty.channel.EventLoopGroup;
import org.flexlb.config.FlexlbConfig;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ThreadPoolExecutor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ChannelConfigurationTest {

    @Test
    void createsDefaultClientResources() {
        ChannelConfiguration configuration = new ChannelConfiguration(new FlexlbConfig());
        ThreadPoolExecutor callbackExecutor = configuration.managedChannelThreadPoolExecutor();
        EventLoopGroup eventLoopGroup = configuration.managedChannelEventLoopGroup();
        try {
            assertEquals(0, callbackExecutor.getQueue().remainingCapacity());
        } finally {
            callbackExecutor.shutdownNow();
            eventLoopGroup.shutdownGracefully().syncUninterruptibly();
        }
    }

    @Test
    void rejectsCallbackExecutorMaximumBelowCoreSize() {
        FlexlbConfig config = new FlexlbConfig();
        config.setGrpcClientCallbackExecutorThreads(2);
        config.setGrpcClientCallbackExecutorMaxThreads(1);

        assertThrows(IllegalArgumentException.class,
                () -> new ChannelConfiguration(config).managedChannelThreadPoolExecutor());
    }

    @Test
    void rejectsNonPositiveEventLoopThreadCount() {
        FlexlbConfig config = new FlexlbConfig();
        config.setGrpcClientEventLoopThreads(0);

        assertThrows(IllegalArgumentException.class,
                () -> new ChannelConfiguration(config).managedChannelEventLoopGroup());
    }
}
