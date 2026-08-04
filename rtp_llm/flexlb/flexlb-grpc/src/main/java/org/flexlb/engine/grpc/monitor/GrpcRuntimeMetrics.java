package org.flexlb.engine.grpc.monitor;

import org.flexlb.engine.grpc.config.GrpcCallbackThreadPoolExecutor;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.flexlb.constant.MetricConstant.GRPC_CALLBACK_EXECUTOR_REJECTED_TOTAL;
import static org.flexlb.constant.MetricConstant.GRPC_CALL_INFLIGHT;

@Component
public class GrpcRuntimeMetrics {

    private final FlexMonitor monitor;
    private final ThreadPoolExecutor callbackExecutor;
    private final ConcurrentMap<GrpcCallKey, AtomicInteger> inFlightCalls = new ConcurrentHashMap<>();
    private final ConcurrentLinkedQueue<GrpcCallHandle> activeCalls = new ConcurrentLinkedQueue<>();

    public GrpcRuntimeMetrics(
            FlexMonitor monitor,
            @Qualifier("managedChannelThreadPoolExecutor") ThreadPoolExecutor callbackExecutor) {
        this.monitor = monitor;
        this.callbackExecutor = callbackExecutor;
    }

    @PostConstruct
    public void init() {
        monitor.register(GRPC_CALL_INFLIGHT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(GRPC_CALLBACK_EXECUTOR_REJECTED_TOTAL, FlexMetricType.COUNTER, FlexPriorityType.PRECISE);
    }

    /**
     * Records subscription to one outbound gRPC call.
     *
     * @param client logical client issuing the call
     * @param service RPC method name
     * @param requestTimeoutMs RPC deadline in milliseconds
     * @return a handle that must be completed when the RPC terminates
     */
    public GrpcCallHandle recordCallStarted(String client, String service, long requestTimeoutMs) {
        GrpcCallKey key = new GrpcCallKey(client, service);
        GrpcCallHandle handle = new GrpcCallHandle(
                key,
                System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(Math.max(0, requestTimeoutMs)));
        inFlightCalls.computeIfAbsent(key, ignored -> new AtomicInteger()).incrementAndGet();
        activeCalls.add(handle);
        return handle;
    }

    /**
     * Records termination of one outbound gRPC call.
     *
     * @param handle handle returned when the RPC started
     */
    public void recordCallCompleted(GrpcCallHandle handle) {
        completeCall(handle);
    }

    /**
     * Reports the current RPC in-flight gauges and callback-executor rejections.
     */
    @Scheduled(fixedRate = 2000)
    public void reportRuntimeMetrics() {
        expireIncompleteCalls();
        inFlightCalls.forEach((key, count) -> monitor.report(
                GRPC_CALL_INFLIGHT,
                FlexMetricTags.of("client", key.client(), "service", key.service()),
                count.get()));
        reportRejectedCallbackTasks();
    }

    private void reportRejectedCallbackTasks() {
        if (!(callbackExecutor instanceof GrpcCallbackThreadPoolExecutor grpcCallbackExecutor)) {
            return;
        }
        monitor.report(
                GRPC_CALLBACK_EXECUTOR_REJECTED_TOTAL,
                FlexMetricTags.of("executor", "gRpcExecutor"),
                grpcCallbackExecutor.getRejectedTaskCount());
    }

    private void expireIncompleteCalls() {
        long currentTimeNanos = System.nanoTime();
        activeCalls.removeIf(handle -> handle.isCompleted()
                || (handle.isExpired(currentTimeNanos) && completeCall(handle)));
    }

    private boolean completeCall(GrpcCallHandle handle) {
        if (!handle.markCompleted()) {
            return false;
        }
        AtomicInteger inFlightCount = inFlightCalls.get(handle.key());
        if (inFlightCount != null) {
            inFlightCount.updateAndGet(current -> Math.max(0, current - 1));
        }
        return true;
    }

    private record GrpcCallKey(String client, String service) {
    }

    public static final class GrpcCallHandle {

        private final GrpcCallKey key;
        private final long deadlineNanos;
        private final AtomicBoolean completed = new AtomicBoolean();

        private GrpcCallHandle(GrpcCallKey key, long deadlineNanos) {
            this.key = key;
            this.deadlineNanos = deadlineNanos;
        }

        private GrpcCallKey key() {
            return key;
        }

        private boolean markCompleted() {
            return completed.compareAndSet(false, true);
        }

        private boolean isCompleted() {
            return completed.get();
        }

        private boolean isExpired(long currentTimeNanos) {
            return currentTimeNanos - deadlineNanos >= 0;
        }
    }
}
