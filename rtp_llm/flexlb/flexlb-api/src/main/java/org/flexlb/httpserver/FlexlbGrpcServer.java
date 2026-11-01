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
import org.flexlb.interceptor.GrpcQosHeaderInterceptor;
import org.flexlb.interceptor.GrpcServerTimingInterceptor;
import org.flexlb.interceptor.GrpcTraceInterceptor;
import org.flexlb.metric.ExecutorStats;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.core.env.Environment;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.io.IOException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionHandler;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

@Component
public class FlexlbGrpcServer implements ExecutorStats {

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
    private final GrpcReporter grpcReporter;
    private final GrpcServerTimingInterceptor grpcServerTimingInterceptor;
    private final GrpcQosHeaderInterceptor grpcQosHeaderInterceptor;

    private Server server;
    private NioEventLoopGroup bossGroup;
    private ThreadPoolExecutor grpcExecutor;
    private ThreadPoolExecutor grpcControlExecutor;
    private GrpcReceivingGuard receivingGuard;
    private CountingCallerRunsHandler countingCallerRunsHandler;

    public FlexlbGrpcServer(
            FlexlbServiceImpl flexlbServiceImpl,
            ConfigService configService,
            Environment environment,
            @Qualifier("grpcServerEventLoopGroup") EventLoopGroup grpcServerEventLoopGroup,
            GrpcReporter grpcReporter,
            GrpcServerTimingInterceptor grpcServerTimingInterceptor,
            GrpcQosHeaderInterceptor grpcQosHeaderInterceptor) {
        this.flexlbServiceImpl = flexlbServiceImpl;
        this.configService = configService;
        this.environment = environment;
        this.quietPeriodNanos =
                TimeUnit.MILLISECONDS.toNanos(
                        configService
                                .loadBalanceConfig()
                                .getGrpcServer()
                                .getShutdownQuietPeriodMs());
        this.grpcServerEventLoopGroup = grpcServerEventLoopGroup;
        this.grpcReporter = grpcReporter == null ? new GrpcReporter(null) : grpcReporter;
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

        FlexlbConfig.GrpcServerConfig executorConfig =
                configService.loadBalanceConfig().getGrpcServer();
        Logger.info(
                "FlexLB gRPC executor config: coreSize={}, maxSize={}, queueSize={}",
                executorConfig.getExecutorCoreSize(),
                executorConfig.getExecutorMaxSize(),
                executorConfig.getExecutorQueueSize());

        GrpcReceivingBudget.Limits receiving =
                GrpcReceivingBudget.resolve(
                        executorConfig, io.netty.util.internal.PlatformDependent.maxDirectMemory());
        Logger.info(
                "FlexLB receiving limits: schedule={}, control={}, timeoutMs={}, bodyBudgetBytes={}, scheduleBodyBudgetBytes={}, controlBodyBudgetBytes={}",
                receiving.scheduleCalls(),
                receiving.controlCalls(),
                receiving.timeoutMs(),
                receiving.budgetBytes(),
                receiving.scheduleBudgetBytes(),
                receiving.controlBudgetBytes());

        this.bossGroup =
                new NioEventLoopGroup(1, new DefaultThreadFactory("flexlb-grpc-server-boss"));
        this.countingCallerRunsHandler = new CountingCallerRunsHandler();
        this.grpcExecutor =
                new ThreadPoolExecutor(
                        executorConfig.getExecutorCoreSize(),
                        executorConfig.getExecutorMaxSize(),
                        60L,
                        TimeUnit.SECONDS,
                        new LinkedBlockingQueue<>(executorConfig.getExecutorQueueSize()),
                        new DefaultThreadFactory("flexlb-grpc-executor"),
                        countingCallerRunsHandler);

        // Keep small Cancel requests independent of Schedule receiving and callback saturation.
        this.grpcControlExecutor =
                new ThreadPoolExecutor(
                        Math.min(8, executorConfig.getExecutorMaxSize()),
                        Math.min(8, executorConfig.getExecutorMaxSize()),
                        60L,
                        TimeUnit.SECONDS,
                        new LinkedBlockingQueue<>(
                                (int) Math.min(1000, Math.max(16L, 2L * receiving.controlCalls()))),
                        new DefaultThreadFactory("flexlb-grpc-control"),
                        new CountingCallerRunsHandler());
        this.receivingGuard =
                new GrpcReceivingGuard(
                        receiving.scheduleCalls(),
                        receiving.controlCalls(),
                        receiving.timeoutMs(),
                        grpcExecutor,
                        grpcControlExecutor,
                        receiving.scheduleBudgetBytes(),
                        receiving.controlBudgetBytes());

        grpcReporter.registerServerExecutor(
                this,
                grpcExecutor.getMaximumPoolSize(),
                countingCallerRunsHandler.getCallerRunsCount()::get);
        grpcReporter.registerReceivingMetrics();

        server =
                NettyServerBuilder.forPort(port)
                        .channelType(NioServerSocketChannel.class)
                        .bossEventLoopGroup(bossGroup)
                        .workerEventLoopGroup(grpcServerEventLoopGroup)
                        // Method lookup and the call supplier run on the transport thread; the
                        // supplier
                        // switches all payload parsing and service callbacks to the appropriate
                        // worker pool.
                        .directExecutor()
                        .callExecutor(receivingGuard)
                        .addStreamTracerFactory(receivingGuard)
                        .intercept(new GrpcOverloadInterceptor(grpcExecutor, grpcControlExecutor))
                        .intercept(receivingGuard)
                        .addService(
                                ServerInterceptors.intercept(
                                        flexlbServiceImpl,
                                        new GrpcTraceInterceptor(),
                                        grpcServerTimingInterceptor,
                                        grpcQosHeaderInterceptor))
                        .maxInboundMessageSize(org.flexlb.constant.GrpcConstants.MAX_MESSAGE_SIZE)
                        .flowControlWindow(4 * 1024 * 1024)
                        .build()
                        .start();

        Logger.info("FlexLB gRPC server started on port {}", port);
    }

    @Override
    public int activeThreads() {
        return grpcExecutor.getActiveCount();
    }

    @Override
    public int queuedTasks() {
        return grpcExecutor.getQueue().size();
    }

    @Override
    public int poolSize() {
        return grpcExecutor.getPoolSize();
    }

    @Override
    public long completedTasks() {
        return grpcExecutor.getCompletedTaskCount();
    }

    /** Report cumulative totals as gauges: query deltas across samples, not gauge sums. */
    @Scheduled(fixedDelay = 1000)
    public void reportReceivingMetrics() {
        GrpcReceivingGuard guard = receivingGuard;
        if (guard == null) return;
        grpcReporter.reportReceivingMetrics(
                guard.receiving(),
                guard.controlReceiving(),
                guard.rejected(),
                guard.timedOut(),
                guard.pendingTimers(),
                grpcExecutor.getQueue().size(),
                grpcExecutor.getActiveCount(),
                grpcControlExecutor.getQueue().size(),
                grpcControlExecutor.getActiveCount(),
                guard.receivingBytes(),
                guard.controlReceivingBytes());
    }

    /** Called before Spring destroys any serving resources. */
    public synchronized void drain() {
        if (server == null || server.isTerminated()) {
            return;
        }
        boolean interrupted = false;
        long startedAt = System.nanoTime();
        Logger.info(
                "keep serving until no new requests for {} ms",
                TimeUnit.NANOSECONDS.toMillis(quietPeriodNanos));
        try {
            while (!server.isShutdown()) {
                long lastArrival =
                        Math.max(
                                startedAt,
                                grpcServerTimingInterceptor.getLastScheduleArrivalNanos());
                long remaining = quietPeriodNanos - (System.nanoTime() - lastArrival);
                if (remaining <= 0) {
                    Logger.info(
                            "Schedule quiet period elapsed; shutting down gRPC and waiting for accepted RPCs");
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
        if (receivingGuard != null) {
            receivingGuard.close();
        }
        if (grpcControlExecutor != null) {
            grpcControlExecutor.shutdown();
        }
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
                // Preserve already accepted lifecycle callbacks even during interrupted shutdown.
                Thread.currentThread().interrupt();
            }
        }
    }

    /** Accepted transport callbacks retain cleanup ownership even during executor shutdown. */
    static class CountingCallerRunsHandler implements RejectedExecutionHandler {
        private final AtomicLong callerRunsCount = new AtomicLong();

        @Override
        public void rejectedExecution(Runnable task, ThreadPoolExecutor executor) {
            callerRunsCount.incrementAndGet();
            task.run();
        }

        public AtomicLong getCallerRunsCount() {
            return callerRunsCount;
        }
    }
}
