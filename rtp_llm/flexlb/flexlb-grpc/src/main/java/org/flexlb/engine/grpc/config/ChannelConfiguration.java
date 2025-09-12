package org.flexlb.engine.grpc.config;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

@Configuration
@Slf4j
public class ChannelConfiguration {

    @Bean
    public ThreadPoolExecutor managedChannelThreadPoolExecutor() {
        return new ThreadPoolExecutor(
                Runtime.getRuntime().availableProcessors() * 2,
                Runtime.getRuntime().availableProcessors() * 4,
                5, TimeUnit.MINUTES,
                new LinkedBlockingQueue<>(4096),
                new NamedThreadFactory("engine-grpc-client-executor")
        );
    }

    @Bean
    public EventLoopGroup managedChannelEventLoopGroup() {
        return new NioEventLoopGroup(Runtime.getRuntime().availableProcessors(), new NamedThreadFactory("engine-grpc-client-event-loop"));
    }
}
