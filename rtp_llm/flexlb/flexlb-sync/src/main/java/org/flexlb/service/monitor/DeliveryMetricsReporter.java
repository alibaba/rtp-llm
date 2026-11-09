package org.flexlb.service.monitor;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.EndpointStats;
import org.flexlb.balance.scheduler.RequestRepository;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.ExecutorStats;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.SafeFlexMonitor;
import org.flexlb.util.Logger;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.Map;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Supplier;

import static org.flexlb.constant.MetricConstant.ACK_TO_RESPONSE_TIME_MS;
import static org.flexlb.constant.MetricConstant.BATCHER_QUEUE_SIZE;
import static org.flexlb.constant.MetricConstant.BATCH_ACTUAL_TIME_MS;
import static org.flexlb.constant.MetricConstant.BATCH_PREDICTED_TIME_MS;
import static org.flexlb.constant.MetricConstant.BATCH_PREDICT_GAP_MS;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_RATIO;
import static org.flexlb.constant.MetricConstant.CACHE_REQUEST_TOTAL;
import static org.flexlb.constant.MetricConstant.DECODE_INFLIGHT_HARD_KV_RESERVED_TOKENS;
import static org.flexlb.constant.MetricConstant.DECODE_INFLIGHT_KV_RESERVED_TOKENS;
import static org.flexlb.constant.MetricConstant.DECODE_TOTAL_LOAD;
import static org.flexlb.constant.MetricConstant.DISPATCH_ACK_TIME_MS;
import static org.flexlb.constant.MetricConstant.DISPATCH_EXECUTOR_ACTIVE_THREADS;
import static org.flexlb.constant.MetricConstant.DISPATCH_EXECUTOR_COMPLETED_TASKS;
import static org.flexlb.constant.MetricConstant.DISPATCH_EXECUTOR_POOL_SIZE;
import static org.flexlb.constant.MetricConstant.DISPATCH_EXECUTOR_QUEUE_SIZE;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_BATCH_SIZE;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_BATCH_TOTAL_TOKENS;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_DISPATCH_REASON;
import static org.flexlb.constant.MetricConstant.INFLIGHT_BATCH_COUNT;
import static org.flexlb.constant.MetricConstant.INFLIGHT_MAX_AGE_MS;
import static org.flexlb.constant.MetricConstant.INFLIGHT_REQUEST_COUNT;
import static org.flexlb.constant.MetricConstant.INFLIGHT_TTL_EXPIRED_QPS;
import static org.flexlb.constant.MetricConstant.ROUTE_SUBMIT_TIME_MS;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_LENGTH;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_WAIT_TIME_MS;
import static org.flexlb.constant.MetricConstant.SCHEDULER_INFLIGHT_SIZE;

/**
 * Queue, delivery, prediction and resource metrics for both delivery modes.
 */
@Slf4j
@Component
public class DeliveryMetricsReporter {
    private static final String[] FIXED_WINDOW_DISPATCH_REASONS = {
        "batch_full", "fixed_window_timeout", "predicted_execution_cap"
    };

    /** role tag value for scheduler-ledger series (vs PREFILL/DECODE endpoint ledgers). */
    public static final String SCHEDULER_ROLE = "SCHEDULER";

    /** engineIp tag value for scheduler-ledger series (no real engine behind them). */
    public static final String SCHEDULER_ENGINE_IP = "scheduler";

    @Autowired private ObjectProvider<RequestRepository> requestSources;
    @Autowired private ObjectProvider<EndpointStats> endpointSources;
    @Autowired private ObjectProvider<RequestSchedulerReporter> requestReporters;

    private final FlexMonitor monitor;

    @Autowired
    public DeliveryMetricsReporter(FlexMonitor monitor) {
        this.monitor = SafeFlexMonitor.wrap(monitor);
    }

    @Scheduled(fixedRateString = "${report.interval.ms:2000}")
    public void report() {
        var requests = requestSources.getObject();
        var endpoints = endpointSources.getObject();
        var requestReporter = requestReporters.getObject();
        if (requests.isClosed()) {
            return;
        }
        try {
            reportSchedulerInflight(requests.liveRequestCount(), requests.oldestLiveRequestAgeMs());
        } catch (RuntimeException failure) {
            warnIsolated("Failed to report scheduler inflight metrics", failure);
        }
        reportEndpoints(
                "Prefill",
                endpoints::prefillStats,
                this::reportPrefill,
                (address, endpoint) ->
                        requestReporter.reportPrefillQueueDepth(address, endpoint.queueDepth()));
        reportEndpoints(
                "Decode",
                endpoints::decodeStats,
                this::reportDecode,
                (address, endpoint) ->
                        requestReporter.reportDecodePreemption(
                                address, endpoint.resourceSnapshot()));
    }

    // ==================== Inflight metrics ====================

    /** Scheduler size retains role=PREFILL; oldest age uses the distinct SCHEDULER ledger role. */
    public void reportSchedulerInflight(int size, long oldestAgeMs) {
        monitor.report(
                SCHEDULER_INFLIGHT_SIZE,
                FlexMetricTags.of("role", RoleType.PREFILL.name(), "engineIp", SCHEDULER_ENGINE_IP),
                size);
        monitor.report(
                INFLIGHT_MAX_AGE_MS,
                FlexMetricTags.ofEngine(SCHEDULER_ENGINE_IP, "role", SCHEDULER_ROLE),
                oldestAgeMs);
    }

    public void reportPrefill(EndpointStats.Prefill endpoint) {
        String ip = endpoint.getIp();
        reportBatcherQueueSize(RoleType.PREFILL.name(), ip, endpoint.queueDepth());
        var counts = endpoint.queueCounters().byPriority();
        boolean empty = true;
        for (int priority = 0; priority < counts.length; priority++) {
            if (counts[priority] == 0) continue;
            empty = false;
            reportBatcherQueueDepthByPriority(
                    RoleType.PREFILL.name(), ip, priority, counts[priority]);
        }
        if (empty) reportBatcherQueueDepthByPriority(RoleType.PREFILL.name(), ip, 0, 0);
        var stats = endpoint.stats();
        reportPrefillInflight(
                ip, stats.batchCount(), stats.locallyOwnedRequests(), stats.maxObservedAgeMs());
    }

    /**
     * Report per-worker batcher queue size via {@code app.flexlb.batcher.queue.size}.
     * <p>Independent metric name to avoid tag schema conflict with {@code routing.queue.length}
     * (which uses type=batchQueue tag). Uses the same role + engineIp tag pattern as other
     * per-worker metrics.
     */
    public void reportBatcherQueueSize(String role, String engineIp, int depth) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, "role", role);
        monitor.report(BATCHER_QUEUE_SIZE, tags, depth);
    }

    // ==================== Queue metrics ====================

    /**
     * Report per-worker batcher queue depth bucketed by normalized Auto-TPM
     * priority via {@code routing.queue.length} (type=batchQueue series).
     * <p>Tagged by the raw 1-100 priority value, "0" for legacy items without
     * a budget — same convention as {@link #reportBatchWaitTimeMs} adding the
     * priority dimension to {@code routing.queue.wait.time.ms}. Only priorities
     * present in the queue are reported (no zero-fill), mirroring the
     * wait-time-by-priority empty-bucket behavior.
     */
    public void reportBatcherQueueDepthByPriority(
            String role, String engineIp, int priority, int depth) {
        FlexMetricTags tags =
                FlexMetricTags.ofEngine(
                        engineIp,
                        "type",
                        "batchQueue",
                        "role",
                        role,
                        "priority",
                        String.valueOf(priority));
        monitor.report(ROUTING_QUEUE_LENGTH, tags, depth);
    }

    /** Report one Prefill ownership snapshot using the existing per-worker series. */
    public void reportPrefillInflight(
            String engineIp, int batchCount, int locallyOwnedRequests, long maxObservedAgeMs) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, "role", RoleType.PREFILL.name());
        monitor.report(INFLIGHT_BATCH_COUNT, tags, batchCount);
        monitor.report(INFLIGHT_REQUEST_COUNT, tags, locallyOwnedRequests);
        monitor.report(INFLIGHT_MAX_AGE_MS, tags, maxObservedAgeMs);
    }

    public void reportDecode(EndpointStats.Decode endpoint) {
        var stats = endpoint.stats();
        reportDecodeInflight(
                endpoint.getIp(),
                stats.inflight(),
                stats.totalLoad(),
                stats.expectedKv(),
                stats.hardKv(),
                stats.oldestAgeMs());
    }

    /** Report one Decode snapshot, keeping expected KV separate from non-reclaimable hard KV. */
    public void reportDecodeInflight(
            String engineIp,
            int inflight,
            int totalLoad,
            long expectedKv,
            long hardKv,
            long oldestAgeMs) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, "role", RoleType.DECODE.name());
        monitor.report(INFLIGHT_REQUEST_COUNT, tags, inflight);
        monitor.report(DECODE_TOTAL_LOAD, tags, totalLoad);
        monitor.report(DECODE_INFLIGHT_KV_RESERVED_TOKENS, tags, expectedKv);
        monitor.report(DECODE_INFLIGHT_HARD_KV_RESERVED_TOKENS, tags, hardKv);
        monitor.report(INFLIGHT_MAX_AGE_MS, tags, oldestAgeMs);
    }

    public void registerExecutor(ExecutorStats stats) {
        monitor.observe(
                DISPATCH_EXECUTOR_ACTIVE_THREADS,
                FlexMetricType.GAUGE,
                stats,
                ExecutorStats::activeThreads);
        monitor.observe(
                DISPATCH_EXECUTOR_QUEUE_SIZE,
                FlexMetricType.GAUGE,
                stats,
                ExecutorStats::queuedTasks);
        monitor.observe(
                DISPATCH_EXECUTOR_POOL_SIZE, FlexMetricType.GAUGE, stats, ExecutorStats::poolSize);
        monitor.observe(
                DISPATCH_EXECUTOR_COMPLETED_TASKS,
                FlexMetricType.COUNTER,
                stats,
                ExecutorStats::completedTasks);
    }

    @PostConstruct
    public void init() {
        for (String metric :
                new String[] {
                    ROUTING_QUEUE_LENGTH,
                    ENGINE_BALANCING_MASTER_BATCH_SIZE,
                    ENGINE_BALANCING_MASTER_BATCH_TOTAL_TOKENS,
                    INFLIGHT_BATCH_COUNT,
                    INFLIGHT_REQUEST_COUNT,
                    SCHEDULER_INFLIGHT_SIZE,
                    BATCHER_QUEUE_SIZE,
                    DECODE_TOTAL_LOAD,
                    DECODE_INFLIGHT_KV_RESERVED_TOKENS,
                    DECODE_INFLIGHT_HARD_KV_RESERVED_TOKENS,
                    INFLIGHT_MAX_AGE_MS,
                }) {
            monitor.register(metric, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        }
        for (String metric :
                new String[] {
                    ROUTING_QUEUE_WAIT_TIME_MS,
                    BATCH_PREDICTED_TIME_MS,
                    BATCH_ACTUAL_TIME_MS,
                    BATCH_PREDICT_GAP_MS,
                    DISPATCH_ACK_TIME_MS,
                    ROUTE_SUBMIT_TIME_MS,
                    ACK_TO_RESPONSE_TIME_MS,
                }) {
            monitor.register(metric, FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        }
        for (String metric :
                new String[] {
                    ENGINE_BALANCING_MASTER_DISPATCH_REASON, INFLIGHT_TTL_EXPIRED_QPS,
                }) {
            monitor.register(metric, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        }
    }

    /** Report a committed batch from its frozen delivery facts. */
    public void reportBatchDelivery(
            String role,
            String engineIp,
            long batchId,
            String decisionReason,
            int remainingQueueDepth,
            int itemCount,
            long hitTokens,
            long totalTokens,
            long predictedMs) {
        try {
            if (itemCount == 0) {
                return;
            }
            reportDispatchReason(role, engineIp, decisionReason);
            reportBatcherQueueSize(role, engineIp, remainingQueueDepth);
            FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, "role", role);
            if (totalTokens > 0L) {
                monitor.report(CACHE_HIT_COUNT, tags, hitTokens);
                monitor.report(CACHE_HIT_RATIO, tags, hitTokens / (double) totalTokens);
                monitor.report(CACHE_REQUEST_TOTAL, tags, 1.0);
            }
            FlexMetricTags reasonTags =
                    FlexMetricTags.ofEngine(engineIp, "role", role, "reason", decisionReason);
            monitor.report(ENGINE_BALANCING_MASTER_BATCH_SIZE, reasonTags, itemCount);
            monitor.report(ENGINE_BALANCING_MASTER_BATCH_TOTAL_TOKENS, reasonTags, totalTokens);
            monitor.report(BATCH_PREDICTED_TIME_MS, tags, Math.max(0L, predictedMs));
        } catch (Throwable failure) {
            try {
                log.warn("Delivery telemetry failed: batchId={}", batchId, failure);
            } catch (Throwable ignored) {
                // A diagnostic failure must not change already committed delivery ownership.
            }
        }
    }

    // ==================== Dispatch reason metrics ====================

    /**
     * Report batch dispatch reason via {@code engine.balancing.master.dispatch.reason}.
     */
    public void reportDispatchReason(String role, String engineIp, String reason) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, "role", role, "reason", reason);
        monitor.report(ENGINE_BALANCING_MASTER_DISPATCH_REASON, tags, 1.0);
    }

    /**
     * Report batch wait time (enqueue to dispatch) via {@code routing.queue.wait.time.ms}.
     * <p>Tagged by the normalized Auto-TPM priority (raw 1-100 value, "0" for
     * legacy items without a budget — same convention as the auto_tpm.* family).
     */
    public void reportBatchWaitTimeMs(String role, String engineIp, long waitMs, int priority) {
        FlexMetricTags tags =
                FlexMetricTags.ofEngine(
                        engineIp, "role", role, "priority", String.valueOf(priority));
        monitor.report(ROUTING_QUEUE_WAIT_TIME_MS, tags, waitMs);
    }

    /**
     * Report inflight entries evicted from an endpoint ledger (prefill/decode
     * orphan sweeps via {@link EndpointRegistry}
     * and friends) via the same {@code app.flexlb.inflight.ttl.expired.qps}
     * series family.
     * <p>Endpoint-side evictions were previously log-only
     * (event=endpoint_inflight_ttl_eviction); this closes the gap with the
     * shared {role, engineIp, reason} tag schema. On this architecture the
     * endpoint ledgers have a single stale-unobserved exit, so the reason
     * bucket is {@code ttl}; only non-zero counts are reported, keeping the
     * series sparse.
     */
    public void reportEndpointInflightTtlExpired(
            String role, String engineIp, String reason, int count) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, "role", role, "reason", reason);
        monitor.report(INFLIGHT_TTL_EXPIRED_QPS, tags, count);
    }

    /** Completion observers are independent: a failed metric must not suppress the next one. */
    public void reportBatchCompletion(String engineIp, long predictedMs, long actualMs) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, "role", RoleType.PREFILL.name());
        monitor.report(BATCH_PREDICTED_TIME_MS, tags, predictedMs);
        monitor.report(BATCH_ACTUAL_TIME_MS, tags, actualMs);
        monitor.report(BATCH_PREDICT_GAP_MS, tags, actualMs - predictedMs);
    }

    /** Report an existing schedule-path latency series with its role and engine tags. */
    public void reportLatency(Latency latency, String role, String engineIp, long durationMs) {
        monitor.report(latency.metric, FlexMetricTags.ofEngine(engineIp, "role", role), durationMs);
    }

    /** Prepare schedule-path meters before an endpoint receives traffic. */
    public void prepareEndpointMetrics(String role, String engineIp) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, "role", role);
        if (RoleType.PREFILL.name().equals(role) || RoleType.PDFUSION.name().equals(role)) {
            monitor.prepare(DISPATCH_ACK_TIME_MS, tags);
            monitor.prepare(ROUTE_SUBMIT_TIME_MS, tags);
            monitor.prepare(ROUTING_QUEUE_WAIT_TIME_MS, tags);
            for (String reason : FIXED_WINDOW_DISPATCH_REASONS) {
                FlexMetricTags reasonTags =
                        FlexMetricTags.ofEngine(engineIp, "role", role, "reason", reason);
                monitor.prepare(ENGINE_BALANCING_MASTER_DISPATCH_REASON, reasonTags);
            }
        }
    }

    private static <T> void reportEndpoints(
            String role,
            Supplier<Map<String, T>> snapshot,
            Consumer<T> batchMetrics,
            BiConsumer<String, T> admissionMetrics) {
        Map<String, T> endpoints;
        try {
            endpoints = snapshot.get();
        } catch (RuntimeException failure) {
            warnIsolated("Failed to snapshot " + role + " endpoints for metrics", failure);
            return;
        }
        for (Map.Entry<String, T> entry : endpoints.entrySet()) {
            try {
                batchMetrics.accept(entry.getValue());
            } catch (RuntimeException failure) {
                warnIsolated(
                        "Failed to report "
                                + role
                                + " endpoint metrics: endpoint="
                                + entry.getKey(),
                        failure);
            }
            try {
                admissionMetrics.accept(entry.getKey(), entry.getValue());
            } catch (RuntimeException failure) {
                warnIsolated(
                        "Failed to report "
                                + role
                                + " admission metrics: endpoint="
                                + entry.getKey(),
                        failure);
            }
        }
    }

    private static void warnIsolated(String message, RuntimeException failure) {
        try {
            Logger.warn(message, failure);
        } catch (Throwable ignored) {
            // Telemetry must never couple otherwise independent metric leaves.
        }
    }

    /**
     * Maps delivery timing observations to their exported metric names.
     */
    public enum Latency {
        DISPATCH_ACK(DISPATCH_ACK_TIME_MS),
        ROUTE_SUBMIT(ROUTE_SUBMIT_TIME_MS),
        ACK_TO_RESPONSE(ACK_TO_RESPONSE_TIME_MS);

        private final String metric;

        Latency(String metric) {
            this.metric = metric;
        }
    }
}
