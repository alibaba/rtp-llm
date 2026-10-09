package org.flexlb.engine.grpc.config;

import io.micrometer.core.instrument.util.NamedThreadFactory;
import io.netty.channel.DefaultSelectStrategyFactory;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.util.concurrent.DefaultEventExecutorChooserFactory;
import io.netty.util.concurrent.DefaultThreadFactory;
import io.netty.util.concurrent.RejectedExecutionHandlers;
import io.netty.util.internal.PlatformDependent;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.nio.channels.spi.SelectorProvider;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

@Configuration
public class ChannelConfiguration {

    private final FlexlbConfig config;

    public ChannelConfiguration(ConfigService configService) {
        this.config = configService.loadBalanceConfig();
    }

    @Bean
    public ThreadPoolExecutor managedChannelThreadPoolExecutor() {
        return new ThreadPoolExecutor(
                config.getInternalRuntime().getGrpcClientExecutorThreads(),
                config.getInternalRuntime().getGrpcClientExecutorThreads(),
                5, TimeUnit.MINUTES,
                new LinkedBlockingQueue<>(config.getInternalRuntime().getGrpcClientExecutorQueueCapacity()),
                new NamedThreadFactory("engine-grpc-client-executor")
        );
    }

    /**
     * Dedicated executor for {@link org.flexlb.httpserver.FlexlbGrpcForwarder} channels.
     * <p>
     * Kept separate from {@link #managedChannelThreadPoolExecutor()} so that load
     * from {@code EngineGrpcClient} (engine status queries) cannot saturate the
     * Forwarder's channel callback threads. Only a request for which no Master
     * was selected may fall back to local scheduling; after a Master is selected,
     * any forwarding failure is terminal to prevent duplicate dispatch.
     */
    @Bean
    public ThreadPoolExecutor forwarderChannelExecutor() {
        return new ThreadPoolExecutor(
                16,
                16,
                5, TimeUnit.MINUTES,
                new LinkedBlockingQueue<>(2000),
                new NamedThreadFactory("flexlb-forwarder-channel-executor"),
                new ThreadPoolExecutor.AbortPolicy()
        );
    }

    @Bean
    public EventLoopGroup managedChannelEventLoopGroup() {
        // No custom taskQueue: grpc-netty's WriteQueue.scheduleFlush marks the
        // flush scheduled before execute() and never resets that flag if the
        // event loop rejects the task. A bounded/rejecting queue here would
        // permanently wedge flush scheduling for the channel (writes and
        // cancels queue forever), which is a strictly worse retention mode
        // than the default unbounded MPSC queue. Load is bounded instead at
        // admission: the dispatch executor's admission permit bounds in-flight
        // EnqueueBatch payloads (see DefaultBatchDispatcher).
        return new NioEventLoopGroup(
                config.getInternalRuntime().getGrpcClientEventLoopThreads(),
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
                config.getInternalRuntime().getGrpcServerWorkerEventLoopThreads(),
                new DefaultThreadFactory("grpc-server-elg")
        );
    }
}
