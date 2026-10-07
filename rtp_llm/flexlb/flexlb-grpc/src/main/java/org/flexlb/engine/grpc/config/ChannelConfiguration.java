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

    /**
     * Bounded event-loop task queue capacity per client event-loop thread.
     *
     * <p>Root-cause fix (case55 diagnostic 92649, 2026-10-08): the previously
     * used {@code PlatformDependent::newMpscQueue} (unbounded) let the gRPC
     * client event-loop task queue grow without limit under high-concurrency
     * ramps (512-1024 threads); queued tasks retained CompositeByteBuf
     * payloads until the JVM ran out of Java heap (45.6 GB heap dump on the
     * faulted master) and deep CompositeByteBuf release recursion overflowed
     * the stack. A bounded LinkedBlockingQueue (available on the pinned Netty 4.1.101,
     * whose PlatformDependent.newMpscQueue has no capacity overload) makes
     * backpressure observable via the existing
     * {@code RejectedExecutionHandlers.reject()} rejection path instead of
     * exhausting the heap.
     */
    static final int GRPC_CLIENT_EVENT_LOOP_TASK_QUEUE_CAPACITY = 10_000;

    /**
     * Bounded task queue for one event loop. Package-private for the
     * regression test; bounds Netty's requested {@code maxPendingTasks} by
     * the configured cap.
     */
    static LinkedBlockingQueue<Runnable> boundedTaskQueue(int maxPendingTasks) {
        return new LinkedBlockingQueue<>(
                Math.min(Math.max(1, maxPendingTasks), GRPC_CLIENT_EVENT_LOOP_TASK_QUEUE_CAPACITY));
    }

    @Bean
    public EventLoopGroup managedChannelEventLoopGroup() {
        return new NioEventLoopGroup(
                config.getInternalRuntime().getGrpcClientEventLoopThreads(),
                null,
                DefaultEventExecutorChooserFactory.INSTANCE,
                SelectorProvider.provider(),
                DefaultSelectStrategyFactory.INSTANCE,
                RejectedExecutionHandlers.reject(),
                ChannelConfiguration::boundedTaskQueue
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
