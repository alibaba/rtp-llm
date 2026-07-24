package org.flexlb.engine.grpc.config;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import io.netty.channel.DefaultSelectStrategyFactory;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.util.concurrent.DefaultEventExecutorChooserFactory;
import io.netty.util.concurrent.DefaultThreadFactory;
import io.netty.util.concurrent.RejectedExecutionHandlers;
import io.netty.util.internal.PlatformDependent;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.nio.channels.spi.SelectorProvider;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

@Configuration
@Slf4j
public class ChannelConfiguration {

    private final FlexlbConfig config;

    public ChannelConfiguration(ConfigService configService) {
        this.config = configService.loadBalanceConfig();
    }

    @Bean
    public ThreadPoolExecutor managedChannelThreadPoolExecutor() {
        return new ThreadPoolExecutor(
                config.getGrpcClientExecutorCoreSize(),
                config.getGrpcClientExecutorMaxSize(),
                5, TimeUnit.MINUTES,
                new LinkedBlockingQueue<>(config.getGrpcClientExecutorQueueSize()),
                new NamedThreadFactory("engine-grpc-client-executor")
        );
    }

    @Bean
    public ThreadPoolExecutor forwarderChannelExecutor() {
        return new ThreadPoolExecutor(
                config.getForwarderExecutorCoreSize(),
                config.getForwarderExecutorMaxSize(),
                5, TimeUnit.MINUTES,
                new LinkedBlockingQueue<>(config.getForwarderExecutorQueueSize()),
                new NamedThreadFactory("flexlb-forwarder-channel-executor"),
                new ThreadPoolExecutor.AbortPolicy()
        );
    }

    @Bean
    public EventLoopGroup managedChannelEventLoopGroup() {
        return new NioEventLoopGroup(
                config.getGrpcClientEventLoopThreads(),
                null,
                DefaultEventExecutorChooserFactory.INSTANCE,
                SelectorProvider.provider(),
                DefaultSelectStrategyFactory.INSTANCE,
                RejectedExecutionHandlers.reject(),
                PlatformDependent::newMpscQueue
        );
    }

    @Bean(destroyMethod = "")
    public EventLoopGroup grpcServerEventLoopGroup() {
        return new NioEventLoopGroup(
                config.getGrpcServerWorkerEventLoopThreads(),
                new DefaultThreadFactory("grpc-server-elg")
        );
    }
}
