package org.flexlb.engine.grpc.config;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import io.netty.channel.DefaultSelectStrategyFactory;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.util.concurrent.DefaultEventExecutorChooserFactory;
import io.netty.util.concurrent.RejectedExecutionHandlers;
import io.netty.util.internal.PlatformDependent;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.nio.channels.spi.SelectorProvider;
import java.util.concurrent.SynchronousQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

@Configuration
@Slf4j
public class ChannelConfiguration {

    @Bean
    public ThreadPoolExecutor managedChannelThreadPoolExecutor() {
        return new ThreadPoolExecutor(
                Runtime.getRuntime().availableProcessors() * 4,
                Runtime.getRuntime().availableProcessors() * 8,
                5, TimeUnit.MINUTES,
                new SynchronousQueue<>(),
                new NamedThreadFactory("engine-grpc-client-executor")
        );
    }

    @Bean
    public EventLoopGroup managedChannelEventLoopGroup() {
        return new NioEventLoopGroup(
                Runtime.getRuntime().availableProcessors() * 8,
                null,
                DefaultEventExecutorChooserFactory.INSTANCE,
                SelectorProvider.provider(),
                DefaultSelectStrategyFactory.INSTANCE,
                RejectedExecutionHandlers.reject(),
                PlatformDependent::newMpscQueue
        );
    }
}
