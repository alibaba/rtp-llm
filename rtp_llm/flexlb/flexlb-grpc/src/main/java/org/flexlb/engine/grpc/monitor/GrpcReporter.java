package org.flexlb.engine.grpc.monitor;

import io.netty.channel.EventLoopGroup;
import io.netty.util.concurrent.EventExecutor;
import io.netty.util.concurrent.SingleThreadEventExecutor;
import org.flexlb.constant.MetricConstant;
import org.flexlb.engine.grpc.GrpcCallTrace;
import org.flexlb.engine.grpc.GrpcFailure;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.ExecutorStats;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.SafeFlexMonitor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import reactor.netty.resources.LoopResources;

import javax.annotation.PostConstruct;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.function.LongSupplier;

import static org.flexlb.constant.MetricConstant.GRPC_CALL_COUNT;
import static org.flexlb.constant.MetricConstant.GRPC_CALL_DURATION;
import static org.flexlb.constant.MetricConstant.GRPC_CALL_FAILURE;
import static org.flexlb.constant.MetricConstant.GRPC_CHANNEL_POOL_SIZE;
import static org.flexlb.constant.MetricConstant.GRPC_CONNECTION_DURATION;
import static org.flexlb.constant.MetricConstant.GRPC_DELIVERY_DELAY;

/**
 * Reporter for gRPC channel pool metrics.
 * Metric registration and reporting are best-effort and do not interrupt RPC processing.
 *
 * Note: All gRPC metrics use "engineIp" as the IP tag name (previously "ip"),
 * consistent with other FlexLB reporters (EngineHealthReporter, DeliveryMetricsReporter,
 * CacheMetricsReporter, etc.). If any Grafana dashboard queries reference the old "ip"
 * tag for these gRPC metrics, they must be updated to use "engineIp" instead.
 */
@Component
public class GrpcReporter {

    private final FlexMonitor monitor;

    private ThreadPoolExecutor clientExecutor;
    private Map<String, EventLoopGroup> eventLoops = Map.of();

    @Autowired(required = false)
    public void bindTransportStats(
            @Qualifier("managedChannelThreadPoolExecutor") ThreadPoolExecutor executor,
            @Qualifier("managedChannelEventLoopGroup") EventLoopGroup eventLoop,
            LoopResources serverLoops) {
        clientExecutor = executor;
        eventLoops =
                Map.of(
                        "serverWorker",
                        serverLoops.onServer(true),
                        "serverSelector",
                        serverLoops.onServerSelect(true),
                        "gRpcEventLoopGroup",
                        eventLoop);
    }

    @Scheduled(fixedRate = 2000)
    public void reportTransportStats() {
        var executor = clientExecutor;
        if (executor != null) {
            String name = MetricConstant.ENGINE_BALANCING_THREAD_POOL_INFO;
            monitor.report(
                    name,
                    FlexMetricTags.of(
                            "threadPool", "gRpcExecutor", "type", "executingTaskThreadSize"),
                    executor.getActiveCount());
            monitor.report(
                    name,
                    FlexMetricTags.of("threadPool", "gRpcExecutor", "type", "queueSize"),
                    executor.getQueue().size());
            monitor.report(
                    name,
                    FlexMetricTags.of("threadPool", "gRpcExecutor", "type", "corePoolSize"),
                    executor.getCorePoolSize());
            monitor.report(
                    name,
                    FlexMetricTags.of(
                            "threadPool", "gRpcExecutor", "type", "currentThreadSizeInPool"),
                    executor.getPoolSize());
        }
        eventLoops.forEach(this::reportEventLoopGroup);
    }

    private void reportEventLoopGroup(String eventLoopGroupName, EventLoopGroup eventLoopGroup) {
        int totalActiveExecutorCount = 0;
        int totalPendingTask = 0;
        for (EventExecutor executor : eventLoopGroup) {
            boolean isShutdown = executor.isShutdown();
            boolean isTerminated = executor.isTerminated();
            boolean isShuttingDown = executor.isShuttingDown();
            // Record active worker count
            if (!isShutdown && !isTerminated && !isShuttingDown) {
                totalActiveExecutorCount++;
            }
            if (executor instanceof SingleThreadEventExecutor singleThreadEventExecutor) {
                int pendingTasks = singleThreadEventExecutor.pendingTasks();
                totalPendingTask += pendingTasks;
            }
        }
        Map<String, String> metricMap = new HashMap<>();
        metricMap.put("name", eventLoopGroupName);
        metricMap.put("type", "active-executor-count");
        monitor.report(
                org.flexlb.constant.MetricConstant.ENGINE_BALANCING_EVENT_LOOP_GROUP_INFO,
                FlexMetricTags.of(metricMap),
                totalActiveExecutorCount);
        metricMap.put("type", "pending-task-total-count");
        monitor.report(
                org.flexlb.constant.MetricConstant.ENGINE_BALANCING_EVENT_LOOP_GROUP_INFO,
                FlexMetricTags.of(metricMap),
                totalPendingTask);
    }

    public GrpcReporter(FlexMonitor monitor) {
        this.monitor = SafeFlexMonitor.wrap(monitor);
    }

    public void registerServerExecutor(
            ExecutorStats stats, int maximumPoolSize, LongSupplier callerRuns) {
        monitor.observe(
                MetricConstant.GRPC_SERVER_EXECUTOR_ACTIVE_THREADS,
                FlexMetricType.GAUGE,
                stats,
                ExecutorStats::activeThreads);
        monitor.observe(
                MetricConstant.GRPC_SERVER_EXECUTOR_QUEUE_SIZE,
                FlexMetricType.GAUGE,
                stats,
                ExecutorStats::queuedTasks);
        monitor.observe(
                MetricConstant.GRPC_SERVER_EXECUTOR_POOL_SIZE,
                FlexMetricType.GAUGE,
                stats,
                ExecutorStats::poolSize);
        monitor.observe(
                MetricConstant.GRPC_SERVER_EXECUTOR_MAX_POOL_SIZE,
                FlexMetricType.GAUGE,
                stats,
                ignored -> maximumPoolSize);
        monitor.observe(
                MetricConstant.GRPC_SERVER_EXECUTOR_COMPLETED_TASKS,
                FlexMetricType.COUNTER,
                stats,
                ExecutorStats::completedTasks);
        monitor.observe(
                MetricConstant.GRPC_SERVER_EXECUTOR_CALLER_RUNS,
                FlexMetricType.COUNTER,
                stats,
                ignored -> callerRuns.getAsLong());
    }

    private static final String[] RECEIVING_METRICS = {
        "app.grpc.server.receiving.active", "app.grpc.server.receiving.control.active",
        "app.grpc.server.receiving.rejected.total", "app.grpc.server.receiving.timed.out.total",
        "app.grpc.server.receiving.timers.active", "app.grpc.server.executor.queue.size",
        "app.grpc.server.executor.active.threads", "app.grpc.server.control.queue.size",
        "app.grpc.server.control.active.threads", "app.grpc.server.receiving.budget.used.bytes",
        "app.grpc.server.receiving.control.budget.used.bytes"
    };

    public void registerReceivingMetrics() {
        for (String name : RECEIVING_METRICS) monitor.register(name, FlexMetricType.GAUGE);
    }

    /** Cumulative receiving totals remain gauges for existing dashboard queries. */
    public void reportReceivingMetrics(double... values) {
        if (values.length != RECEIVING_METRICS.length) {
            throw new IllegalArgumentException(
                    "Expected " + RECEIVING_METRICS.length + " receiving metrics");
        }
        for (int i = 0; i < values.length; i++) monitor.report(RECEIVING_METRICS[i], values[i]);
    }

    @PostConstruct
    public void init() {
        monitor.register(
                MetricConstant.ENGINE_BALANCING_THREAD_POOL_INFO,
                FlexMetricType.GAUGE,
                FlexPriorityType.PRECISE);
        monitor.register(
                MetricConstant.ENGINE_BALANCING_EVENT_LOOP_GROUP_INFO,
                FlexMetricType.GAUGE,
                FlexPriorityType.PRECISE);
        SafeFlexMonitor.run(
                GRPC_CALL_FAILURE,
                () ->
                        monitor.register(
                                GRPC_CALL_FAILURE, FlexMetricType.QPS, FlexPriorityType.PRECISE));
        SafeFlexMonitor.run(
                GRPC_DELIVERY_DELAY,
                () ->
                        monitor.register(
                                GRPC_DELIVERY_DELAY,
                                FlexMetricType.GAUGE,
                                FlexPriorityType.PRECISE));
        this.monitor.register(GRPC_CHANNEL_POOL_SIZE, FlexMetricType.GAUGE);
        this.monitor.register(GRPC_CALL_DURATION, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(GRPC_RESPONSE_SIZE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(GRPC_CALL_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        this.monitor.register(
                GRPC_CONNECTION_DURATION, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
    }

    /**
     * Report the current size of the gRPC channel pool
     *
     * @param channelPoolSize current number of channels in the pool
     */
    public void reportChannelPoolSize(int channelPoolSize) {
        monitor.report(GRPC_CHANNEL_POOL_SIZE, FlexMetricTags.of("type", "grpc"), channelPoolSize);
    }

    /**
     * Report gRPC call metrics
     *
     * @param serviceType Service type
     * @param duration Call duration in milliseconds
     * @param isRetry Whether this is a retry call
     */
    public void reportCallMetrics(String serviceType, long duration, boolean isRetry) {
        FlexMetricTags tags =
                FlexMetricTags.of("service", serviceType, "retry", String.valueOf(isRetry));

        // Report call duration
        monitor.report(GRPC_CALL_DURATION, tags, duration);

        // Report call count
        monitor.report(GRPC_CALL_COUNT, tags, 1);
    }

    /**
     * Report gRPC connection duration
     *
     * @param serviceType Service type
     * @param connectionDuration Connection duration in microseconds
     */
    public void reportConnectionDuration(String serviceType, long connectionDuration) {
        FlexMetricTags tags = FlexMetricTags.of("service", serviceType);

        // Report connection duration
        monitor.report(GRPC_CONNECTION_DURATION, tags, connectionDuration);
    }

    /** Reports one logical RPC only after its final failure, including any retry. */
    public void reportCallFailure(
            String serviceType,
            boolean retry,
            String phase,
            GrpcFailure.Analysis failure,
            GrpcCallTrace.Snapshot transport) {
        FlexMetricTags tags =
                FlexMetricTags.of(
                        "service", serviceType,
                        "retry", String.valueOf(retry),
                        "phase", phase,
                        "reason", failure.kind().name(),
                        "origin", transport.failureOrigin(failure),
                        "grpc_code", failure.grpcCode().name(),
                        "transport_stage", transport.stage().name(),
                        "transport_cause",
                                transport.transportCauseKind() == null
                                        ? "NONE"
                                        : transport.transportCauseKind().name(),
                        "transport_event",
                                "NONE".equals(transport.transportEvent())
                                                && failure.kind()
                                                        == GrpcFailure.Kind.CONNECT_REFUSED
                                        ? "CONNECT_REFUSED"
                                        : transport.transportEvent());
        monitor.report(GRPC_CALL_FAILURE, tags, 1.0);
    }

    /** Local transport-to-Future delivery time, including successful calls. Unknown samples are omitted. */
    public void reportDeliveryDelay(
            String serviceType, boolean retry, GrpcCallTrace.Snapshot transport) {
        if (transport.callbackDelayMs() >= 0) {
            monitor.report(
                    GRPC_DELIVERY_DELAY,
                    FlexMetricTags.of(
                            "service",
                            serviceType,
                            "retry",
                            String.valueOf(retry),
                            "stage",
                            "STREAM_CLOSED_TO_CALLBACK"),
                    transport.callbackDelayMs());
        }
        if (transport.responseDeliveryMs() >= 0) {
            monitor.report(
                    GRPC_DELIVERY_DELAY,
                    FlexMetricTags.of(
                            "service",
                            serviceType,
                            "retry",
                            String.valueOf(retry),
                            "stage",
                            "TRAILERS_TO_CALLBACK"),
                    transport.responseDeliveryMs());
        }
    }
}
