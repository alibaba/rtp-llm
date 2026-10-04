package org.flexlb.httpserver;

import io.grpc.Server;
import io.grpc.ServerInterceptors;
import io.grpc.netty.NettyServerBuilder;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import io.netty.util.concurrent.DefaultThreadFactory;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.constant.MetricConstant;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.interceptor.GrpcQosHeaderInterceptor;
import org.flexlb.interceptor.GrpcServerTimingInterceptor;
import org.flexlb.interceptor.GrpcTraceInterceptor;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.core.env.Environment;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.io.IOException;
import java.util.Objects;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionHandler;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

@Component
public class FlexlbGrpcServer {

    /**
     * Offset from HTTP port to gRPC port for FlexLB's own servers.
     * This is separate from CommonConstants.GRPC_PORT_OFFSET which applies
     * to backend inference engine ports (HTTP+1→gRPC).
     */
    static final int FLEXLB_GRPC_PORT_OFFSET = 2;
    private static final int DEFAULT_HTTP_PORT = 7001;
    private final long quietPeriodNanos;

    private final FlexlbServiceImpl flexlbServiceImpl;
    private final ConfigService configService;
    private final Environment environment;
    private final EventLoopGroup grpcServerEventLoopGroup;
    private final FlexMonitor monitor;
    private final GrpcServerTimingInterceptor grpcServerTimingInterceptor;
    private final GrpcQosHeaderInterceptor grpcQosHeaderInterceptor;

    private Server server;
    private NioEventLoopGroup bossGroup;
    private volatile ThreadPoolExecutor grpcExecutor;
    private final CountingAbortHandler countingAbortHandler = new CountingAbortHandler();

    public FlexlbGrpcServer(FlexlbServiceImpl flexlbServiceImpl,
                            ConfigService configService,
                            Environment environment,
                            @Qualifier("grpcServerEventLoopGroup") EventLoopGroup grpcServerEventLoopGroup,
                            FlexMonitor monitor,
                            GrpcServerTimingInterceptor grpcServerTimingInterceptor,
                            GrpcQosHeaderInterceptor grpcQosHeaderInterceptor) {
        this.flexlbServiceImpl = flexlbServiceImpl;
        this.configService = configService;
        this.environment = environment;
        this.quietPeriodNanos = TimeUnit.MILLISECONDS.toNanos(
                configService.loadBalanceConfig().getGrpcServer().getShutdownQuietPeriodMs());
        this.grpcServerEventLoopGroup = grpcServerEventLoopGroup;
        this.monitor = Objects.requireNonNull(monitor, "monitor");
        this.grpcServerTimingInterceptor = grpcServerTimingInterceptor;
        this.grpcQosHeaderInterceptor = grpcQosHeaderInterceptor;
    }

    @PostConstruct
    public void start() throws IOException {
        // Always derive gRPC port from HTTP port.
        // server.port may come from --server.port CLI arg (Spring Environment only)
        // or from -Dserver.port JVM property; check both.
        String portStr = environment.getProperty("server.port");
        if (portStr == null) {
            portStr = System.getProperty("server.port", String.valueOf(DEFAULT_HTTP_PORT));
        }
        int httpPort = Integer.parseInt(portStr);
        int port = httpPort + FLEXLB_GRPC_PORT_OFFSET;

        FlexlbConfig.GrpcServerConfig executorConfig = configService.loadBalanceConfig().getGrpcServer();
        Logger.info("FlexLB gRPC executor config: coreSize={}, maxSize={}, queueSize={}",
                executorConfig.getExecutorCoreSize(), executorConfig.getExecutorMaxSize(),
                executorConfig.getExecutorQueueSize());

        this.bossGroup = new NioEventLoopGroup(1, new DefaultThreadFactory("flexlb-grpc-server-boss"));
        this.grpcExecutor = new ThreadPoolExecutor(
                executorConfig.getExecutorCoreSize(), executorConfig.getExecutorMaxSize(),
                60L, TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(executorConfig.getExecutorQueueSize()),
                new DefaultThreadFactory("flexlb-grpc-executor"),
                countingAbortHandler
        );

        // Register monitoring metrics for the gRPC server executor
        registerMetrics();
        reportExecutorMetrics();

        server = NettyServerBuilder.forPort(port)
                .channelType(NioServerSocketChannel.class)
                .bossEventLoopGroup(bossGroup)
                .workerEventLoopGroup(grpcServerEventLoopGroup)
                .executor(grpcExecutor)
                .addService(ServerInterceptors.intercept(flexlbServiceImpl,
                        new GrpcTraceInterceptor(), grpcServerTimingInterceptor, grpcQosHeaderInterceptor))
                .maxInboundMessageSize(16 * 1024 * 1024)
                .flowControlWindow(4 * 1024 * 1024)
                .build()
                .start();

        Logger.info("FlexLB gRPC server started on port {}", port);
    }

    private void registerMetrics() {
        monitor.register(MetricConstant.GRPC_SERVER_EXECUTOR_ACTIVE_THREADS,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.GRPC_SERVER_EXECUTOR_QUEUE_SIZE,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.GRPC_SERVER_EXECUTOR_POOL_SIZE,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.GRPC_SERVER_EXECUTOR_MAX_POOL_SIZE,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        // Report the rejection handler's cumulative count as a snapshot, without adding it again.
        monitor.register(MetricConstant.GRPC_SERVER_EXECUTOR_REJECTED_TASKS,
                FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
    }

    @Scheduled(fixedRate = 2000)
    private void reportExecutorMetrics() {
        ThreadPoolExecutor executor = grpcExecutor;
        if (executor == null) {
            return;
        }
        monitor.report(MetricConstant.GRPC_SERVER_EXECUTOR_ACTIVE_THREADS, executor.getActiveCount());
        monitor.report(MetricConstant.GRPC_SERVER_EXECUTOR_QUEUE_SIZE, executor.getQueue().size());
        monitor.report(MetricConstant.GRPC_SERVER_EXECUTOR_POOL_SIZE, executor.getPoolSize());
        monitor.report(MetricConstant.GRPC_SERVER_EXECUTOR_MAX_POOL_SIZE, executor.getMaximumPoolSize());
        monitor.report(MetricConstant.GRPC_SERVER_EXECUTOR_REJECTED_TASKS,
                countingAbortHandler.getRejectionCount());
    }

    /**
     * Called before Spring destroys any serving resources.
     */
    public synchronized void drain() {
        if (server == null || server.isTerminated()) {
            return;
        }
        boolean interrupted = false;
        long startedAt = System.nanoTime();
        Logger.info("keep serving until no new requests for {} ms",
                TimeUnit.NANOSECONDS.toMillis(quietPeriodNanos));
        try {
            while (!server.isShutdown()) {
                long lastArrival = Math.max(startedAt, grpcServerTimingInterceptor.getLastScheduleArrivalNanos());
                long remaining = quietPeriodNanos - (System.nanoTime() - lastArrival);
                if (remaining <= 0) {
                    Logger.info("Schedule quiet period elapsed; shutting down gRPC and waiting for accepted RPCs");
                    server.shutdown();
                    break;
                }
                try {
                    TimeUnit.NANOSECONDS.sleep(remaining);
                } catch (InterruptedException e) {
                    interrupted = true;
                }
            }
            while (!server.isTerminated()) {
                try {
                    server.awaitTermination();
                } catch (InterruptedException e) {
                    // An interrupt must not turn graceful drain into resource destruction.
                    interrupted = true;
                }
            }
            Logger.info("All accepted gRPC requests completed");
        } finally {
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
    }

    @PreDestroy
    public void shutdown() {
        drain();
        if (bossGroup != null) {
            bossGroup.shutdownGracefully();
        }
        if (grpcServerEventLoopGroup != null) {
            grpcServerEventLoopGroup.shutdownGracefully();
        }
        if (grpcExecutor != null) {
            grpcExecutor.shutdown();
            try {
                grpcExecutor.awaitTermination(5, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                grpcExecutor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }

    /**
     * Counts tasks rejected because the pool is saturated or shutting down.
     * AbortPolicy throws instead of running tasks on the calling Netty event loop.
     */
    static class CountingAbortHandler implements RejectedExecutionHandler {
        private final AtomicLong rejectionCount = new AtomicLong(0);
        private final ThreadPoolExecutor.AbortPolicy delegate =
                new ThreadPoolExecutor.AbortPolicy();

        @Override
        public void rejectedExecution(Runnable r, ThreadPoolExecutor executor) {
            rejectionCount.incrementAndGet();
            delegate.rejectedExecution(r, executor);
        }

        public long getRejectionCount() {
            return rejectionCount.get();
        }
    }
}
