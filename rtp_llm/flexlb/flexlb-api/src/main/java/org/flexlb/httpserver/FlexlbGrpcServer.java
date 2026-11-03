package org.flexlb.httpserver;

import io.grpc.Server;
import io.grpc.ServerInterceptors;
import io.grpc.netty.NettyServerBuilder;
import io.micrometer.core.instrument.FunctionCounter;
import io.micrometer.core.instrument.Gauge;
import io.micrometer.core.instrument.MeterRegistry;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioServerSocketChannel;
import io.netty.util.concurrent.DefaultThreadFactory;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.enums.FlexMetricType;
import org.springframework.scheduling.annotation.Scheduled;
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.constant.MetricConstant;
import org.flexlb.interceptor.GrpcQosHeaderInterceptor;
import org.flexlb.interceptor.GrpcServerTimingInterceptor;
import org.flexlb.interceptor.GrpcTraceInterceptor;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.core.env.Environment;
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
public class FlexlbGrpcServer {

    /**
     * Offset from HTTP port to gRPC port for FlexLB's own servers.
     * This is separate from CommonConstants.GRPC_PORT_OFFSET which applies
     * to backend inference engine ports (HTTP+1→gRPC).
     */
    static final int FLEXLB_GRPC_PORT_OFFSET = 2;
    private static final int DEFAULT_HTTP_PORT = 7001;
    private final long quietPeriodNanos;

    /**
     * Metric prefix — matches {@code MicrometerFlexMonitor.METRIC_PREFIX} so that
     * metrics registered directly via {@link MeterRegistry} follow the same
     * naming convention as those going through the FlexMonitor abstraction.
     */
    private static final String METRIC_PREFIX = "flexlb.";

    private final FlexlbServiceImpl flexlbServiceImpl;
    private final ConfigService configService;
    private final Environment environment;
    private final EventLoopGroup grpcServerEventLoopGroup;
    private final MeterRegistry meterRegistry;
    private final GrpcServerTimingInterceptor grpcServerTimingInterceptor;
    private final GrpcQosHeaderInterceptor grpcQosHeaderInterceptor;

    @Autowired(required = false)
    private FlexMonitor flexMonitor;

    private Server server;
    private NioEventLoopGroup bossGroup;
    private ThreadPoolExecutor grpcExecutor;
    private ThreadPoolExecutor grpcControlExecutor;
    private GrpcReceivingGuard receivingGuard;
    private CountingCallerRunsHandler countingCallerRunsHandler;

    public FlexlbGrpcServer(FlexlbServiceImpl flexlbServiceImpl,
                            ConfigService configService,
                            Environment environment,
                            @Qualifier("grpcServerEventLoopGroup") EventLoopGroup grpcServerEventLoopGroup,
                            @Autowired(required = false) MeterRegistry meterRegistry,
                            GrpcServerTimingInterceptor grpcServerTimingInterceptor,
                            GrpcQosHeaderInterceptor grpcQosHeaderInterceptor) {
        this.flexlbServiceImpl = flexlbServiceImpl;
        this.configService = configService;
        this.environment = environment;
        this.quietPeriodNanos = TimeUnit.MILLISECONDS.toNanos(
                configService.loadBalanceConfig().getGrpcServer().getShutdownQuietPeriodMs());
        this.grpcServerEventLoopGroup = grpcServerEventLoopGroup;
        this.meterRegistry = meterRegistry;
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
        Logger.info("FlexLB gRPC executor config: coreSize={}, maxSize={}, queueSize={}, "
                        + "permitKeepAliveTimeMs={}, permitKeepAliveWithoutCalls={}",
                executorConfig.getExecutorCoreSize(), executorConfig.getExecutorMaxSize(),
                executorConfig.getExecutorQueueSize(), executorConfig.getPermitKeepAliveTimeMs(),
                executorConfig.isPermitKeepAliveWithoutCalls());

        GrpcReceivingBudget.Limits receiving = GrpcReceivingBudget.resolve(executorConfig,
                io.netty.util.internal.PlatformDependent.maxDirectMemory());
        Logger.info("FlexLB receiving limits: schedule={}, control={}, timeoutMs={}, bodyBudgetBytes={}, scheduleBodyBudgetBytes={}, controlBodyBudgetBytes={}",
                receiving.scheduleCalls(), receiving.controlCalls(), receiving.timeoutMs(),
                receiving.budgetBytes(), receiving.scheduleBudgetBytes(), receiving.controlBudgetBytes());

        this.bossGroup = new NioEventLoopGroup(1, new DefaultThreadFactory("flexlb-grpc-server-boss"));
        this.countingCallerRunsHandler = new CountingCallerRunsHandler();
        this.grpcExecutor = new ThreadPoolExecutor(
                executorConfig.getExecutorCoreSize(), executorConfig.getExecutorMaxSize(),
                60L, TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(executorConfig.getExecutorQueueSize()),
                new DefaultThreadFactory("flexlb-grpc-executor"),
                countingCallerRunsHandler
        );

        // Keep small Cancel requests independent of Schedule receiving and callback saturation.
        this.grpcControlExecutor = new ThreadPoolExecutor(
                Math.min(8, executorConfig.getExecutorMaxSize()),
                Math.min(8, executorConfig.getExecutorMaxSize()), 60L, TimeUnit.SECONDS,
                new LinkedBlockingQueue<>((int) Math.min(1000, Math.max(16L, 2L * receiving.controlCalls()))),
                new DefaultThreadFactory("flexlb-grpc-control"), new CountingCallerRunsHandler());
        this.receivingGuard = new GrpcReceivingGuard(receiving.scheduleCalls(), receiving.controlCalls(),
                receiving.timeoutMs(), grpcExecutor, grpcControlExecutor, receiving.scheduleBudgetBytes(),
                receiving.controlBudgetBytes());

        // Register monitoring metrics for the gRPC server executor
        registerMetrics();
        registerReceivingMetrics();

        server = NettyServerBuilder.forPort(port)
                .channelType(NioServerSocketChannel.class)
                .bossEventLoopGroup(bossGroup)
                .workerEventLoopGroup(grpcServerEventLoopGroup)
                // Method lookup and the call supplier run on the transport thread; the supplier
                // switches all payload parsing and service callbacks to the appropriate worker pool.
                .directExecutor()
                .callExecutor(receivingGuard)
                .addStreamTracerFactory(receivingGuard)
                .intercept(new GrpcOverloadInterceptor(grpcExecutor, grpcControlExecutor))
                .intercept(receivingGuard)
                .addService(ServerInterceptors.intercept(flexlbServiceImpl,
                        new GrpcTraceInterceptor(), grpcServerTimingInterceptor, grpcQosHeaderInterceptor))
                .maxInboundMessageSize(org.flexlb.constant.GrpcConstants.MAX_MESSAGE_SIZE)
                .flowControlWindow(4 * 1024 * 1024)
                .permitKeepAliveTime(executorConfig.getPermitKeepAliveTimeMs(), TimeUnit.MILLISECONDS)
                .permitKeepAliveWithoutCalls(executorConfig.isPermitKeepAliveWithoutCalls())
                .build()
                .start();

        Logger.info("FlexLB gRPC server started on port {}", port);
    }

    /**
     * Register Micrometer gauges and function counters for the gRPC server executor.
     *
     * <p>Metrics exposed via the {@code /prometheus} endpoint:
     * <ul>
     *   <li>{@code flexlb_grpc_server_executor_active_threads} — gauge: active thread count</li>
     *   <li>{@code flexlb_grpc_server_executor_queue_size} — gauge: pending task queue length</li>
     *   <li>{@code flexlb_grpc_server_executor_pool_size} — gauge: current thread pool size</li>
     *   <li>{@code flexlb_grpc_server_executor_max_pool_size} — gauge: maximum thread pool size</li>
     *   <li>{@code flexlb_grpc_server_executor_completed_tasks_total} — counter: completed task count</li>
     *   <li>{@code flexlb_grpc_server_executor_caller_runs_total} — counter: saturated executor inline executions</li>
     * </ul>
     *
     * <p>When {@link MeterRegistry} is not available (e.g. actuator not on classpath),
     * metric registration is silently skipped.
     */
    private void registerMetrics() {
        if (meterRegistry == null) {
            Logger.info("MeterRegistry not available, skipping gRPC server executor metrics");
            return;
        }

        // Gauges — auto-read from ThreadPoolExecutor on each scrape
        Gauge.builder(METRIC_PREFIX + MetricConstant.GRPC_SERVER_EXECUTOR_ACTIVE_THREADS,
                        grpcExecutor, ThreadPoolExecutor::getActiveCount)
                .description("gRPC server executor active thread count")
                .register(meterRegistry);

        Gauge.builder(METRIC_PREFIX + MetricConstant.GRPC_SERVER_EXECUTOR_QUEUE_SIZE,
                        grpcExecutor, exec -> exec.getQueue().size())
                .description("gRPC server executor pending task queue size")
                .register(meterRegistry);

        Gauge.builder(METRIC_PREFIX + MetricConstant.GRPC_SERVER_EXECUTOR_POOL_SIZE,
                        grpcExecutor, ThreadPoolExecutor::getPoolSize)
                .description("gRPC server executor current pool size")
                .register(meterRegistry);

        Gauge.builder(METRIC_PREFIX + MetricConstant.GRPC_SERVER_EXECUTOR_MAX_POOL_SIZE,
                        grpcExecutor, ThreadPoolExecutor::getMaximumPoolSize)
                .description("gRPC server executor maximum pool size")
                .register(meterRegistry);

        // FunctionCounters — monotonically increasing values read on each scrape
        FunctionCounter.builder(METRIC_PREFIX + MetricConstant.GRPC_SERVER_EXECUTOR_COMPLETED_TASKS,
                        grpcExecutor, ThreadPoolExecutor::getCompletedTaskCount)
                .description("gRPC server executor total completed tasks")
                .register(meterRegistry);

        FunctionCounter.builder(METRIC_PREFIX + MetricConstant.GRPC_SERVER_EXECUTOR_CALLER_RUNS,
                        countingCallerRunsHandler, handler -> handler.getCallerRunsCount().doubleValue())
                .description("gRPC server executor inline executions on saturation")
                .register(meterRegistry);

        Logger.info("FlexLB gRPC server executor metrics registered with MeterRegistry");
    }

    private static final String[] RECEIVING_METRICS = {
        "app.grpc.server.receiving.active", "app.grpc.server.receiving.control.active",
        "app.grpc.server.receiving.rejected.total", "app.grpc.server.receiving.timed.out.total",
        "app.grpc.server.receiving.timers.active", "app.grpc.server.executor.queue.size",
        "app.grpc.server.executor.active.threads", "app.grpc.server.control.queue.size",
        "app.grpc.server.control.active.threads", "app.grpc.server.receiving.budget.used.bytes",
        "app.grpc.server.receiving.control.budget.used.bytes"
    };

    private void registerReceivingMetrics() {
        if (flexMonitor != null) {
            for (String name : RECEIVING_METRICS) flexMonitor.register(name, FlexMetricType.GAUGE);
        }
    }

    /** Report cumulative totals as gauges: query deltas across samples, not gauge sums. */
    @Scheduled(fixedDelay = 1000)
    public void reportReceivingMetrics() {
        GrpcReceivingGuard guard = receivingGuard;
        FlexMonitor monitor = flexMonitor;
        if (guard == null || monitor == null) return;
        double[] values = {guard.receiving(), guard.controlReceiving(), guard.rejected(), guard.timedOut(),
            guard.pendingTimers(), grpcExecutor.getQueue().size(), grpcExecutor.getActiveCount(),
            grpcControlExecutor.getQueue().size(), grpcControlExecutor.getActiveCount(),
            guard.receivingBytes(), guard.controlReceivingBytes()};
        for (int i = 0; i < RECEIVING_METRICS.length; i++) monitor.report(RECEIVING_METRICS[i], values[i]);
    }

    /** Called before Spring destroys any serving resources. */
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

    /**
     * gRPC executor tasks include message ownership transfers and terminal callbacks.
     * Throwing here can lose a MessageProducer without closing its retained buffers
     * (grpc-java 1.65.1 ServerImpl.messagesAvailable). Never discard these tasks.
     * New RPCs are shed separately by GrpcOverloadInterceptor; already accepted
     * callbacks apply synchronous backpressure when the bounded queue is full.
     */
    static class CountingCallerRunsHandler implements RejectedExecutionHandler {
        private final AtomicLong callerRunsCount = new AtomicLong();

        @Override
        public void rejectedExecution(Runnable task, ThreadPoolExecutor executor) {
            callerRunsCount.incrementAndGet();
            // Unlike JDK CallerRunsPolicy, do not silently drop terminal callbacks
            // racing executor shutdown. The server is drained before shutdown.
            task.run();
        }

        public AtomicLong getCallerRunsCount() {
            return callerRunsCount;
        }
    }
}
