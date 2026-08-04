package org.flexlb.engine.grpc.config;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import io.netty.channel.DefaultSelectStrategyFactory;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.epoll.Epoll;
import io.netty.channel.epoll.EpollEventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.util.concurrent.DefaultEventExecutorChooserFactory;
import io.netty.util.concurrent.RejectedExecutionHandlers;
import io.netty.util.internal.PlatformDependent;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.nio.channels.spi.SelectorProvider;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

@Configuration
@Slf4j
public class ChannelConfiguration {

    private final FlexlbConfig config;

    @Autowired
    public ChannelConfiguration(ConfigService configService) {
        this(configService.loadBalanceConfig());
    }

    ChannelConfiguration(FlexlbConfig config) {
        // Only for test
        this.config = config;
    }

    @Bean
    public ThreadPoolExecutor managedChannelThreadPoolExecutor() {
        int coreThreads = positive("grpcClientCallbackExecutorThreads", config.getGrpcClientCallbackExecutorThreads());
        int maxThreads = positive("grpcClientCallbackExecutorMaxThreads", config.getGrpcClientCallbackExecutorMaxThreads());
        if (maxThreads < coreThreads) {
            throw new IllegalArgumentException("grpcClientCallbackExecutorMaxThreads must be at least core threads");
        }
        return new GrpcCallbackThreadPoolExecutor(
                coreThreads,
                maxThreads,
                60, TimeUnit.SECONDS,
                new SynchronousQueue<>(),
                new NamedThreadFactory("engine-grpc-client-executor")
        );
    }

    @Bean
    public EventLoopGroup managedChannelEventLoopGroup() {
        int threads = positive("grpcClientEventLoopThreads", config.getGrpcClientEventLoopThreads());
        if (Epoll.isAvailable()) {
            return new EpollEventLoopGroup(
                    threads,
                    null,
                    DefaultEventExecutorChooserFactory.INSTANCE,
                    DefaultSelectStrategyFactory.INSTANCE,
                    RejectedExecutionHandlers.reject(),
                    PlatformDependent::newMpscQueue
            );
        }
        return new NioEventLoopGroup(
                threads,
                null,
                DefaultEventExecutorChooserFactory.INSTANCE,
                SelectorProvider.provider(),
                DefaultSelectStrategyFactory.INSTANCE,
                RejectedExecutionHandlers.reject(),
                PlatformDependent::newMpscQueue
        );
    }

    private int positive(String name, int value) {
        if (value <= 0) {
            throw new IllegalArgumentException(name + " must be positive");
        }
        return value;
    }
}
