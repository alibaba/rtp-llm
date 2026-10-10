package org.flexlb.service.monitor;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.List;

import static com.google.common.math.LongMath.saturatedAdd;
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

/** Queue, delivery, prediction and resource metrics for both delivery modes. */
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

    private final FlexMonitor monitor;

    @Autowired
    public DeliveryMetricsReporter(FlexMonitor monitor) {
        this.monitor = monitor;
    }

    @PostConstruct
    public void init() {
        // Queue — same type as RoutingQueueReporter
        monitor.register(ROUTING_QUEUE_LENGTH, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(ROUTING_QUEUE_WAIT_TIME_MS, FlexMetricType.TIMER, FlexPriorityType.PRECISE);

        // Dispatch reason — independent metric for batch path
        monitor.register(ENGINE_BALANCING_MASTER_DISPATCH_REASON, FlexMetricType.QPS, FlexPriorityType.PRECISE);

        // Batch size — gauge, reported per dispatch
        monitor.register(ENGINE_BALANCING_MASTER_BATCH_SIZE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        // Batch total tokens — gauge, reported per dispatch
        monitor.register(ENGINE_BALANCING_MASTER_BATCH_TOTAL_TOKENS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        // Inflight — batch count and request count per worker (FlexLB scheduler view, tagged by role)
        monitor.register(INFLIGHT_BATCH_COUNT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(INFLIGHT_REQUEST_COUNT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        // Scheduler-level inflight size — uses scheduler-level tags (role=PREFILL, engineIp="scheduler")
        // Note: the former per-engine app.engine.health.check.local.inflight.size has been removed.
        monitor.register(SCHEDULER_INFLIGHT_SIZE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        // Batcher queue size — per-engine pending batch request count (FlexLB batcher queue depth)
        monitor.register(BATCHER_QUEUE_SIZE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        // Decode total load and inflight KV reserved — per decode worker (FlexLB scheduler view)
        monitor.register(DECODE_TOTAL_LOAD, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(DECODE_INFLIGHT_KV_RESERVED_TOKENS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(DECODE_INFLIGHT_HARD_KV_RESERVED_TOKENS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(INFLIGHT_MAX_AGE_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        // Inflight TTL expired — count of inflight requests cleaned up by the TTL task, QPS tagged by role
        monitor.register(INFLIGHT_TTL_EXPIRED_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);

        // Prediction accuracy — predicted vs actual engine execution time (timer for distribution)
        monitor.register(BATCH_PREDICTED_TIME_MS, FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        monitor.register(BATCH_ACTUAL_TIME_MS, FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        monitor.register(BATCH_PREDICT_GAP_MS, FlexMetricType.TIMER, FlexPriorityType.PRECISE);

        // Dispatch-to-ACK time — latency from gRPC dispatch to engine EnqueueBatch acknowledgment (timer for distribution)
        monitor.register(DISPATCH_ACK_TIME_MS, FlexMetricType.TIMER, FlexPriorityType.PRECISE);

        // Route+submit time — from schedule() entry to batcher offer completion (timer for distribution)
        monitor.register(ROUTE_SUBMIT_TIME_MS, FlexMetricType.TIMER, FlexPriorityType.PRECISE);

        // ACK-to-response time — from engine ACK to schedule response sent to client (timer for distribution)
        monitor.register(ACK_TO_RESPONSE_TIME_MS, FlexMetricType.TIMER, FlexPriorityType.PRECISE);

        log.info("DeliveryMetricsReporter initialized (20 metrics)");
    }

    /** Report a committed delivery; batchId is zero for individual route delivery. */
    public void reportDelivery(long batchId, String decisionReason, int remainingQueueDepth,
                               List<RequestRoute> items, long predictedMs) {
        try {
            if (items.isEmpty()) {
                return;
            }
            String role = RoleType.PREFILL.name();
            String engineIp = items.getFirst().prefillEp().getIp();
            if (batchId != 0L) {
                reportDispatchReason(role, engineIp, decisionReason);
            }
            reportBatcherQueueSize(role, engineIp, remainingQueueDepth);
            long nowMs = System.currentTimeMillis();
            long hitTokens = 0L;
            long totalTokens = 0L;
            for (RequestRoute item : items) {
                reportBatchWaitTimeMs(role, engineIp, Math.max(0L, nowMs - item.enqueuedAtMs()), item.priority());
                if (batchId != 0L) {
                    hitTokens = saturatedAdd(hitTokens, Math.max(0L, item.hitCache()));
                    totalTokens = saturatedAdd(totalTokens, Math.max(0L, item.seqLen()));
                }
            }
            if (batchId != 0L) {
                FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, "role", role);
                if (totalTokens > 0L) {
                    monitor.report(CACHE_HIT_COUNT, tags, hitTokens);
                    monitor.report(CACHE_HIT_RATIO, tags, hitTokens / (double) totalTokens);
                    monitor.report(CACHE_REQUEST_TOTAL, tags, 1.0);
                }
                FlexMetricTags reasonTags = FlexMetricTags.ofEngine(engineIp, "role", role, "reason", decisionReason);
                monitor.report(ENGINE_BALANCING_MASTER_BATCH_SIZE, reasonTags, items.size());
                monitor.report(ENGINE_BALANCING_MASTER_BATCH_TOTAL_TOKENS, reasonTags, totalTokens);
                monitor.report(BATCH_PREDICTED_TIME_MS, tags,
                        Math.max(0L, predictedMs));
            }
        } catch (Throwable failure) {
            try {
                log.warn("Delivery telemetry failed: batchId={}", batchId, failure);
            } catch (Throwable ignored) {
                // A diagnostic failure must not change already committed delivery ownership.
            }
        }
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
    public void reportBatcherQueueDepthByPriority(String role, String engineIp, int priority, int depth) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "type", "batchQueue",
                "role", role,
                "priority", String.valueOf(priority));
        monitor.report(ROUTING_QUEUE_LENGTH, tags, depth);
    }

    /**
     * Report per-worker batcher queue size via {@code app.flexlb.batcher.queue.size}.
     * <p>Independent metric name to avoid tag schema conflict with {@code routing.queue.length}
     * (which uses type=batchQueue tag). Uses the same role + engineIp tag pattern as other
     * per-worker metrics.
     */
    public void reportBatcherQueueSize(String role, String engineIp, int depth) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", role);
        monitor.report(BATCHER_QUEUE_SIZE, tags, depth);
    }

    /**
     * Report batch wait time (enqueue to dispatch) via {@code routing.queue.wait.time.ms}.
     * <p>Tagged by the normalized Auto-TPM priority (raw 1-100 value, "0" for
     * legacy items without a budget — same convention as the auto_tpm.* family).
     */
    public void reportBatchWaitTimeMs(String role, String engineIp, long waitMs, int priority) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", role,
                "priority", String.valueOf(priority));
        monitor.report(ROUTING_QUEUE_WAIT_TIME_MS, tags, waitMs);
    }

    // ==================== Dispatch reason metrics ====================

    /**
     * Report batch dispatch reason via {@code engine.balancing.master.dispatch.reason}.
     */
    public void reportDispatchReason(String role, String engineIp, String reason) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", role,
                "reason", reason);
        monitor.report(ENGINE_BALANCING_MASTER_DISPATCH_REASON, tags, 1.0);
    }

    // ==================== Inflight metrics ====================

    /** Scheduler size retains role=PREFILL; oldest age uses the distinct SCHEDULER ledger role. */
    public void reportSchedulerInflight(int size, long oldestAgeMs) {
        monitor.report(SCHEDULER_INFLIGHT_SIZE,
                FlexMetricTags.of("role", RoleType.PREFILL.name(), "engineIp", SCHEDULER_ENGINE_IP), size);
        monitor.report(INFLIGHT_MAX_AGE_MS,
                FlexMetricTags.ofEngine(SCHEDULER_ENGINE_IP, "role", SCHEDULER_ROLE), oldestAgeMs);
    }

    /** Report one Prefill ownership snapshot using the existing per-worker series. */
    public void reportPrefillInflight(String engineIp, PrefillState.Stats stats) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, "role", RoleType.PREFILL.name());
        monitor.report(INFLIGHT_BATCH_COUNT, tags, stats.batchCount());
        monitor.report(INFLIGHT_REQUEST_COUNT, tags, stats.locallyOwnedRequests());
        monitor.report(INFLIGHT_MAX_AGE_MS, tags, stats.maxObservedAgeMs());
    }

    /**
     * Report inflight entries evicted from an endpoint ledger (prefill/decode
     * orphan sweeps via {@link org.flexlb.balance.endpoint.EndpointRegistry}
     * and friends) via the same {@code app.flexlb.inflight.ttl.expired.qps}
     * series family.
     * <p>Endpoint-side evictions were previously log-only
     * (event=endpoint_inflight_ttl_eviction); this closes the gap with the
     * shared {role, engineIp, reason} tag schema. On this architecture the
     * endpoint ledgers have a single stale-unobserved exit, so the reason
     * bucket is {@code ttl}; only non-zero counts are reported, keeping the
     * series sparse.
     */
    public void reportEndpointInflightTtlExpired(String role, String engineIp,
                                                 String reason, int count) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", role,
                "reason", reason);
        monitor.report(INFLIGHT_TTL_EXPIRED_QPS, tags, count);
    }

    /** Report one Decode snapshot, keeping expected KV separate from non-reclaimable hard KV. */
    public void reportDecodeInflight(String engineIp, int inflight, int totalLoad,
                                     long expectedKv, long hardKv, long oldestAgeMs) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, "role", RoleType.DECODE.name());
        monitor.report(INFLIGHT_REQUEST_COUNT, tags, inflight);
        monitor.report(DECODE_TOTAL_LOAD, tags, totalLoad);
        monitor.report(DECODE_INFLIGHT_KV_RESERVED_TOKENS, tags, expectedKv);
        monitor.report(DECODE_INFLIGHT_HARD_KV_RESERVED_TOKENS, tags, hardKv);
        monitor.report(INFLIGHT_MAX_AGE_MS, tags, oldestAgeMs);
    }

    /** Completion observers are independent: a failed metric must not suppress the next one. */
    public void reportBatchCompletion(String engineIp, long batchId, long predictedMs, long actualMs) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, "role", RoleType.PREFILL.name());
        reportCompletionMetric(BATCH_PREDICTED_TIME_MS, tags, predictedMs, batchId);
        reportCompletionMetric(BATCH_ACTUAL_TIME_MS, tags, actualMs, batchId);
        reportCompletionMetric(BATCH_PREDICT_GAP_MS, tags, actualMs - predictedMs, batchId);
    }

    private void reportCompletionMetric(String metric, FlexMetricTags tags, long value, long batchId) {
        try {
            monitor.report(metric, tags, value);
        } catch (RuntimeException failure) {
            log.warn("Batch completion metric failed: batchId={} metric={}", batchId, metric, failure);
        }
    }

    public enum Latency {
        DISPATCH_ACK(DISPATCH_ACK_TIME_MS),
        ROUTE_SUBMIT(ROUTE_SUBMIT_TIME_MS),
        ACK_TO_RESPONSE(ACK_TO_RESPONSE_TIME_MS);

        private final String metric;

        Latency(String metric) {
            this.metric = metric;
        }
    }

    /** Report an existing schedule-path latency series with its role and engine tags. */
    public void reportLatency(Latency latency, String role, String engineIp, long durationMs) {
        monitor.report(latency.metric, FlexMetricTags.ofEngine(engineIp, "role", role), durationMs);
    }

    /** Prepare schedule-path meters before an endpoint receives traffic. */
    public void prepareEndpointMetrics(String role, String engineIp) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", role);
        if (RoleType.PREFILL.name().equals(role) || RoleType.PDFUSION.name().equals(role)) {
            monitor.prepare(DISPATCH_ACK_TIME_MS, tags);
            monitor.prepare(ROUTE_SUBMIT_TIME_MS, tags);
            monitor.prepare(ROUTING_QUEUE_WAIT_TIME_MS, tags);
            for (String reason : FIXED_WINDOW_DISPATCH_REASONS) {
                FlexMetricTags reasonTags = FlexMetricTags.ofEngine(engineIp,
                        "role", role,
                        "reason", reason);
                monitor.prepare(ENGINE_BALANCING_MASTER_DISPATCH_REASON, reasonTags);
            }
        }
    }

}
