package org.flexlb.engine.grpc.config;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import io.netty.channel.DefaultSelectStrategyFactory;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.util.concurrent.DefaultEventExecutorChooserFactory;
import io.netty.util.concurrent.RejectedExecutionHandlers;
import io.netty.util.internal.PlatformDependent;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.FlexlbConfig;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.nio.channels.spi.SelectorProvider;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

@Configuration
@Slf4j
public class ChannelConfiguration {

    private final FlexlbConfig flexlbConfig;

    public ChannelConfiguration(FlexlbConfig flexlbConfig) {
        this.flexlbConfig = flexlbConfig;
    }

    @Bean
    public ThreadPoolExecutor managedChannelThreadPoolExecutor() {
        int multiplier = flexlbConfig.getGrpcEventLoopMultiplier();
        int cpus = Runtime.getRuntime().availableProcessors();
        return new ThreadPoolExecutor(
                cpus * multiplier,
                cpus * multiplier * 2,
                5, TimeUnit.MINUTES,
                new SynchronousQueue<>(),
                new NamedThreadFactory("engine-grpc-client-executor")
        );
    }

    @Bean
    public EventLoopGroup managedChannelEventLoopGroup() {
        int threads = Runtime.getRuntime().availableProcessors() * flexlbConfig.getGrpcEventLoopMultiplier();
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
}
