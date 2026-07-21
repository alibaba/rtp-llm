package org.flexlb.service.monitor;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.dao.route.RoleType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;

import static org.flexlb.constant.MetricConstant.BATCHER_QUEUE_SIZE;
import static org.flexlb.constant.MetricConstant.BATCH_ACTUAL_TIME_MS;
import static org.flexlb.constant.MetricConstant.BATCH_PREDICTED_TIME_MS;
import static org.flexlb.constant.MetricConstant.BATCH_PREDICT_GAP_MS;
import static org.flexlb.constant.MetricConstant.DISPATCH_ACK_TIME_MS;
import static org.flexlb.constant.MetricConstant.ACK_TO_RESPONSE_TIME_MS;
import static org.flexlb.constant.MetricConstant.ROUTE_SUBMIT_TIME_MS;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_RATIO;
import static org.flexlb.constant.MetricConstant.CACHE_REQUEST_TOTAL;
import static org.flexlb.constant.MetricConstant.DECODE_INFLIGHT_KV_RESERVED_TOKENS;
import static org.flexlb.constant.MetricConstant.DECODE_TOTAL_LOAD;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_BATCH_SIZE;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_BATCH_TOTAL_TOKENS;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_DISPATCH_REASON;
import static org.flexlb.constant.MetricConstant.INFLIGHT_BATCH_COUNT;
import static org.flexlb.constant.MetricConstant.INFLIGHT_REQUEST_COUNT;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_LENGTH;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_WAIT_TIME_MS;
import static org.flexlb.constant.MetricConstant.SCHEDULER_INFLIGHT_SIZE;

/**
 * Batch scheduling metrics reporter for FlexLB batch dispatch path.
 *
 * <p>Batch-path metrics use independent metric names to avoid tag schema
 * conflicts with the non-batch path:
 * queue (routing.queue.length + routing.queue.wait.time.ms),
 * dispatch reason (engine.balancing.master.dispatch.reason),
 * inflight (flexlb.scheduler.inflight.size + health.check.running.task.info.size).
 */
@Slf4j
@Component
public class BatchSchedulerReporter {

    private static final String[] FIXED_WINDOW_DISPATCH_REASONS = {
            "batch_full", "fixed_window_timeout", "predict_threshold"
    };

    private final FlexMonitor monitor;

    @Autowired
    public BatchSchedulerReporter(FlexMonitor monitor) {
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

        log.info("BatchSchedulerReporter initialized (17 metrics)");
    }

    // ==================== Queue metrics ====================

    /**
     * Report per-worker batcher queue depth via {@code routing.queue.length}.
     */
    public void reportBatcherQueueDepth(String role, String engineIp, String engineIpPort, int depth) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, engineIpPort,
                "type", "batchQueue",
                "role", role);
        monitor.report(ROUTING_QUEUE_LENGTH, tags, depth);
    }

    /**
     * Report per-worker batcher queue size via {@code app.flexlb.batcher.queue.size}.
     * <p>Independent metric name to avoid tag schema conflict with {@code routing.queue.length}
     * (which uses type=batchQueue tag). Uses the same role + engineIp tag pattern as other
     * per-worker metrics.
     */
    public void reportBatcherQueueSize(String role, String engineIp, String engineIpPort, int depth) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, engineIpPort,
                "role", role);
        monitor.report(BATCHER_QUEUE_SIZE, tags, depth);
    }

    /**
     * Report batch wait time (enqueue to dispatch) via {@code routing.queue.wait.time.ms}.
     */
    public void reportBatchWaitTimeMs(String role, String engineIp, String engineIpPort, long waitMs) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, engineIpPort,
                "role", role);
        monitor.report(ROUTING_QUEUE_WAIT_TIME_MS, tags, waitMs);
    }

    // ==================== Dispatch reason metrics ====================

    /**
     * Report batch dispatch reason via {@code engine.balancing.master.dispatch.reason}.
     */
    public void reportDispatchReason(String role, String engineIp, String engineIpPort, String reason) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, engineIpPort,
                "role", role,
                "reason", reason);
        monitor.report(ENGINE_BALANCING_MASTER_DISPATCH_REASON, tags, 1.0);
    }

    // ==================== Inflight metrics ====================

    /**
     * Report batch-aggregated cache hit metrics via reuse of the existing
     * {@code cache.hit.count} / {@code cache.hit.ratio} / {@code cache.request.total}
     * keys registered by {@link CacheMetricsReporter}.
     *
     * @param role        prefill / decode
     * @param engineIp    the selected prefill endpoint IP
     * @param hitTokens   total cache-hit tokens across the batch
     * @param totalTokens total sequence length across the batch
     */
    public void reportBatchCacheHitMetrics(String role, String engineIp, String engineIpPort, long hitTokens, long totalTokens) {
        if (totalTokens <= 0L) {
            return;
        }
        double hitRatio = hitTokens / (double) totalTokens;
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, engineIpPort,
                "role", role);
        monitor.report(CACHE_HIT_COUNT, tags, hitTokens);
        monitor.report(CACHE_HIT_RATIO, tags, hitRatio);
        monitor.report(CACHE_REQUEST_TOTAL, tags, 1.0);
    }

    /**
     * Report batch size (number of requests dispatched together) via {@code engine.balancing.master.batch.size}.
     */
    public void reportBatchSize(String role, String engineIp, String engineIpPort, String reason, int batchSize) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, engineIpPort,
                "role", role,
                "reason", reason);
        monitor.report(ENGINE_BALANCING_MASTER_BATCH_SIZE, tags, batchSize);
    }

    /**
     * Report batch total token count (sum of seqLen across picked items) via
     * {@code engine.balancing.master.batch.total.tokens}.
     */
    public void reportBatchTotalTokens(String role, String engineIp, String engineIpPort, String reason, long totalTokens) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, engineIpPort,
                "role", role,
                "reason", reason);
        monitor.report(ENGINE_BALANCING_MASTER_BATCH_TOTAL_TOKENS, tags, totalTokens);
    }

    /**
     * Report scheduler inflight size via {@code flexlb.scheduler.inflight.size}.
     * <p>Uses an independent metric name (not {@code engine.health.check.local.inflight.size})
     * because this is a scheduler-level metric with tag schema (role=PREFILL, engineIp="scheduler"),
     * which differs from EngineHealthReporter's per-engine version tagged by
     * (model, code, engineIp=real-engine-IP, role). Sharing the same metric name would cause
     * tag schema conflicts in kmonitor grouping.
     * Uses role=PREFILL + engineIp=scheduler tags to match the Grafana panel filter.
     */
    public void reportSchedulerInflightSize(int size) {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", RoleType.PREFILL.name(),
                "engineIp", "scheduler",
                "engineIpPort", "scheduler");
        monitor.report(SCHEDULER_INFLIGHT_SIZE, tags, size);
    }

    /**
     * Report per-worker inflight batch count (number of dispatched-but-uncompleted batches)
     * via {@code flexlb.inflight.batch.count}.
     * <p>Unified for both prefill and decode workers, tagged by role and engineIp.
     */
    public void reportInflightBatchCount(String role, String engineIp, String engineIpPort, int count) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, engineIpPort,
                "role", role);
        monitor.report(INFLIGHT_BATCH_COUNT, tags, count);
    }

    /**
     * Report per-worker inflight request count (dispatched but not yet confirmed by engine)
     * via {@code flexlb.inflight.request.count}.
     * <p>Unified for both prefill and decode workers, tagged by role and engineIp.
     * Replaces the former separate reportPrefillInflightRequestCount and reportDecodeInflightCount.
     */
    public void reportInflightRequestCount(String role, String engineIp, String engineIpPort, int count) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, engineIpPort,
                "role", role);
        monitor.report(INFLIGHT_REQUEST_COUNT, tags, count);
    }

    // ==================== Decode inflight metrics ====================

    /**
     * Report per-decode-worker total load (confirmed running + scheduler inflight)
     * via {@code flexlb.decode.total.load}.
     */
    public void reportDecodeTotalLoad(String engineIp, String engineIpPort, int totalLoad) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, engineIpPort,
                "role", RoleType.DECODE.name());
        monitor.report(DECODE_TOTAL_LOAD, tags, totalLoad);
    }

    /**
     * Report per-decode-worker inflight KV cache reserved tokens (local inflight reservation not yet confirmed by the engine)
     * via {@code flexlb.decode.inflight.kv.reserved.tokens}.
     */
    public void reportDecodeInflightKvReserved(String engineIp, String engineIpPort, long kvReservedTokens) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, engineIpPort,
                "role", RoleType.DECODE.name());
        monitor.report(DECODE_INFLIGHT_KV_RESERVED_TOKENS, tags, kvReservedTokens);
    }

    // ==================== Prediction accuracy metrics ====================

    /**
     * Report formula-predicted batch execution time via {@code app.flexlb.batch.predicted.time.ms}.
     */
    public void reportBatchPredictedTimeMs(String role, String engineIp, String engineIpPort, long predictedMs) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, engineIpPort,
                "role", role);
        monitor.report(BATCH_PREDICTED_TIME_MS, tags, predictedMs);
    }

    /**
     * Report engine-reported actual batch execution time via {@code app.flexlb.batch.actual.time.ms}.
     */
    public void reportBatchActualTimeMs(String role, String engineIp, String engineIpPort, long actualMs) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, engineIpPort,
                "role", role);
        monitor.report(BATCH_ACTUAL_TIME_MS, tags, actualMs);
    }

    /**
     * Report the gap between actual and predicted batch execution time via {@code app.flexlb.batch.predict.gap.ms}.
     */
    public void reportBatchPredictGapMs(String role, String engineIp, String engineIpPort, long gapMs) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, engineIpPort,
                "role", role);
        monitor.report(BATCH_PREDICT_GAP_MS, tags, gapMs);
    }

    // ==================== Dispatch-to-ACK latency metrics ====================

    /**
     * Report dispatch-to-ACK latency (from gRPC dispatch to engine EnqueueBatch acknowledgment)
     * via {@code app.flexlb.dispatch.ack.time.ms}.
     *
     * @param role     prefill / decode
     * @param engineIp the prefill endpoint IP
     * @param ackTimeMs milliseconds from dispatch to ACK
     */
    public void reportDispatchAckTimeMs(String role, String engineIp, String engineIpPort, long ackTimeMs) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, engineIpPort,
                "role", role);
        monitor.report(DISPATCH_ACK_TIME_MS, tags, ackTimeMs);
    }

    /** Prepare schedule-path meters before an endpoint receives traffic. */
    public void prepareEndpointMetrics(String role, String engineIp, String engineIpPort) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, engineIpPort,
                "role", role);
        if (RoleType.PREFILL.name().equals(role) || RoleType.PDFUSION.name().equals(role)) {
            monitor.prepare(DISPATCH_ACK_TIME_MS, tags);
            monitor.prepare(ROUTE_SUBMIT_TIME_MS, tags);
            monitor.prepare(ROUTING_QUEUE_WAIT_TIME_MS, tags);
            for (String reason : FIXED_WINDOW_DISPATCH_REASONS) {
                FlexMetricTags reasonTags = FlexMetricTags.ofEngine(engineIp, engineIpPort,
                        "role", role,
                        "reason", reason);
                monitor.prepare(ENGINE_BALANCING_MASTER_DISPATCH_REASON, reasonTags);
            }
        }
    }

    // ==================== Route+submit latency metrics ====================

    /**
     * Report route+submit latency (from schedule() entry to batcher offer completion)
     * via {@code app.flexlb.route.submit.time.ms}.
     *
     * @param role      prefill / decode
     * @param engineIp  the prefill endpoint IP
     * @param submitMs  milliseconds from schedule entry to batcher offer completion
     */
    public void reportRouteSubmitTimeMs(String role, String engineIp, String engineIpPort, long submitMs) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, engineIpPort,
                "role", role);
        monitor.report(ROUTE_SUBMIT_TIME_MS, tags, submitMs);
    }

    // ==================== ACK-to-response latency metrics ====================

    /**
     * Report ACK-to-response latency (from engine EnqueueBatch acknowledgment to schedule
     * response sent to the client) via {@code app.flexlb.ack.to.response.time.ms}.
     *
     * @param role             prefill / decode
     * @param engineIp         the prefill endpoint IP
     * @param ackToResponseMs  milliseconds from engine ACK to response sent
     */
    public void reportAckToResponseTimeMs(String role, String engineIp, String engineIpPort, long ackToResponseMs) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp, engineIpPort,
                "role", role);
        monitor.report(ACK_TO_RESPONSE_TIME_MS, tags, ackToResponseMs);
    }
}
