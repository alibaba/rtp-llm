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
import org.flexlb.config.ConfigService;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.constant.MetricConstant;
import org.flexlb.interceptor.GrpcServerTimingInterceptor;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import javax.annotation.PreDestroy;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.RejectedExecutionHandler;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

@Component
public class FlexlbGrpcServer {

    /**
     * Offset from HTTP port to gRPC port for FlexLB's own servers.
     * This is separate from CommonConstants.GRPC_PORT_OFFSET which applies
     * to backend inference engine ports (HTTP+1→gRPC).
     */
    static final int FLEXLB_GRPC_PORT_OFFSET = 2;
    private static final int DEFAULT_HTTP_PORT = 7001;

    /**
     * Metric prefix — matches {@code MicrometerFlexMonitor.METRIC_PREFIX} so that
     * metrics registered directly via {@link MeterRegistry} follow the same
     * naming convention as those going through the FlexMonitor abstraction.
     */
    private static final String METRIC_PREFIX = "flexlb.";

    private final FlexlbServiceImpl flexlbServiceImpl;
    private final ConfigService configService;
    private final FlexlbConfig flexlbConfig;
    private final Environment environment;
    private final EventLoopGroup grpcServerEventLoopGroup;
    private final MeterRegistry meterRegistry;
    private final GrpcServerTimingInterceptor grpcServerTimingInterceptor;

    private Server server;
    private NioEventLoopGroup bossGroup;
    private ThreadPoolExecutor grpcExecutor;
    private CountingAbortHandler countingAbortHandler;

    public FlexlbGrpcServer(FlexlbServiceImpl flexlbServiceImpl,
                            ConfigService configService,
                            Environment environment,
                            @Qualifier("grpcServerEventLoopGroup") EventLoopGroup grpcServerEventLoopGroup,
                            @Autowired(required = false) MeterRegistry meterRegistry,
                            GrpcServerTimingInterceptor grpcServerTimingInterceptor) {
        this.flexlbServiceImpl = flexlbServiceImpl;
        this.configService = configService;
        this.flexlbConfig = configService.loadBalanceConfig();
        this.environment = environment;
        this.grpcServerEventLoopGroup = grpcServerEventLoopGroup;
        this.meterRegistry = meterRegistry;
        this.grpcServerTimingInterceptor = grpcServerTimingInterceptor;
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

        // Read executor sizes from FlexlbConfig (unified config system)
        int coreSize = flexlbConfig.getFlexlbGrpcExecutorCoreSize();
        int maxSize = flexlbConfig.getFlexlbGrpcExecutorMaxSize();
        int queueSize = flexlbConfig.getFlexlbGrpcExecutorQueueSize();

        Logger.info("FlexLB gRPC executor config: coreSize={}, maxSize={}, queueSize={}",
                coreSize, maxSize, queueSize);

        this.bossGroup = new NioEventLoopGroup(1, new DefaultThreadFactory("flexlb-grpc-server-boss"));
        this.countingAbortHandler = new CountingAbortHandler();
        this.grpcExecutor = new ThreadPoolExecutor(
                coreSize, maxSize,
                60L, TimeUnit.SECONDS,
                queueSize > 0 ? new LinkedBlockingQueue<Runnable>(queueSize) : new LinkedBlockingQueue<Runnable>(),
                new DefaultThreadFactory("flexlb-grpc-executor"),
                countingAbortHandler
        );

        // Register monitoring metrics for the gRPC server executor
        registerMetrics();

        server = NettyServerBuilder.forPort(port)
                .channelType(NioServerSocketChannel.class)
                .bossEventLoopGroup(bossGroup)
                .workerEventLoopGroup(grpcServerEventLoopGroup)
                .executor(grpcExecutor)
                .addService(ServerInterceptors.intercept(flexlbServiceImpl,
                        grpcServerTimingInterceptor))
                .maxInboundMessageSize(16 * 1024 * 1024)
                .flowControlWindow(4 * 1024 * 1024)
                .build()
                .start();

        Logger.info("FlexLB gRPC server started on port {}", port);
    }

    int getBoundPort() {
        return server == null ? -1 : server.getPort();
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
     *   <li>{@code flexlb_grpc_server_executor_caller_runs_total} — counter: AbortPolicy rejections</li>
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
                        countingAbortHandler, handler -> handler.getRejectionCount().doubleValue())
                .description("gRPC server executor rejection count (AbortPolicy)")
                .register(meterRegistry);

        Logger.info("FlexLB gRPC server executor metrics registered with MeterRegistry");
    }

    @PreDestroy
    public void shutdown() {
        if (server != null) {
            server.shutdown();
            try {
                server.awaitTermination(5, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                server.shutdownNow();
                Thread.currentThread().interrupt();
            }
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
                grpcExecutor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }

    /**
     * Custom {@link RejectedExecutionHandler} that delegates to
     * {@link ThreadPoolExecutor.AbortPolicy} and counts the number of
     * times the rejection policy is triggered.
     *
     * <p>The rejection count is exposed as a {@link FunctionCounter} metric
     * ({@code flexlb_grpc_server_executor_caller_runs_total} — metric name kept
     * for backward compat). When AbortPolicy fires, it means the executor thread
     * pool and queue are both saturated. A {@link java.util.concurrent.RejectedExecutionException}
     * is thrown which gRPC catches and converts into an {@code UNAVAILABLE}
     * status response to the client. This prevents the Netty EventLoop thread
     * from being forced to execute the task synchronously (which was the
     * behaviour with {@code CallerRunsPolicy} and caused ~990 ms blocking).
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

        public AtomicLong getRejectionCount() {
            return rejectionCount;
        }
    }
}
