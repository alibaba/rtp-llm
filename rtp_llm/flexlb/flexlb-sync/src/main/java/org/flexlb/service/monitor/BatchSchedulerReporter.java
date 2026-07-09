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

import static org.flexlb.constant.MetricConstant.BATCH_ACTUAL_TIME_MS;
import static org.flexlb.constant.MetricConstant.BATCH_INFLIGHT_COUNT;
import static org.flexlb.constant.MetricConstant.BATCH_INFLIGHT_REQUEST_COUNT;
import static org.flexlb.constant.MetricConstant.BATCH_PREDICTED_TIME_MS;
import static org.flexlb.constant.MetricConstant.BATCH_PREDICT_GAP_MS;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_RATIO;
import static org.flexlb.constant.MetricConstant.CACHE_REQUEST_TOTAL;
import static org.flexlb.constant.MetricConstant.DECODE_INFLIGHT_COUNT;
import static org.flexlb.constant.MetricConstant.DECODE_TOTAL_LOAD;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_BATCH_SIZE;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_BATCH_TOTAL_TOKENS;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_DISPATCH_REASON;
import static org.flexlb.constant.MetricConstant.ENGINE_LOCAL_TASK_MAP_SIZE;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_LENGTH;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_WAIT_TIME_MS;

/**
 * Batch scheduling metrics reporter for FlexLB batch dispatch path.
 *
 * <p>Batch-path metrics use independent metric names to avoid tag schema
 * conflicts with the non-batch path:
 * queue (routing.queue.length + routing.queue.wait.time.ms),
 * dispatch reason (engine.balancing.master.dispatch.reason),
 * inflight (health.check.local.task.map.size + health.check.running.task.info.size).
 */
@Slf4j
@Component
public class BatchSchedulerReporter {

    private final FlexMonitor monitor;

    @Autowired
    public BatchSchedulerReporter(FlexMonitor monitor) {
        this.monitor = monitor;
    }

    @PostConstruct
    public void init() {
        // Queue — same type as RoutingQueueReporter
        monitor.register(ROUTING_QUEUE_LENGTH, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(ROUTING_QUEUE_WAIT_TIME_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        // Dispatch reason — independent metric for batch path
        monitor.register(ENGINE_BALANCING_MASTER_DISPATCH_REASON, FlexMetricType.QPS, FlexPriorityType.PRECISE);

        // Batch size — gauge, reported per dispatch
        monitor.register(ENGINE_BALANCING_MASTER_BATCH_SIZE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        // Batch total tokens — gauge, reported per dispatch
        monitor.register(ENGINE_BALANCING_MASTER_BATCH_TOTAL_TOKENS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        // Inflight — batch count and request count per prefill worker (FlexLB scheduler view)
        monitor.register(BATCH_INFLIGHT_COUNT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(BATCH_INFLIGHT_REQUEST_COUNT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(ENGINE_LOCAL_TASK_MAP_SIZE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        // Decode inflight — per decode worker (FlexLB scheduler view)
        monitor.register(DECODE_INFLIGHT_COUNT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(DECODE_TOTAL_LOAD, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        // Prediction accuracy — predicted vs actual engine execution time
        monitor.register(BATCH_PREDICTED_TIME_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(BATCH_ACTUAL_TIME_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(BATCH_PREDICT_GAP_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        log.info("BatchSchedulerReporter initialized (13 metrics)");
    }

    // ==================== Queue metrics ====================

    /**
     * Report per-worker batcher queue depth via {@code routing.queue.length}.
     */
    public void reportBatcherQueueDepth(String role, String engineIp, int depth) {
        FlexMetricTags tags = FlexMetricTags.of(
                "type", "batchQueue",
                "role", role,
                "engineIp", engineIp);
        monitor.report(ROUTING_QUEUE_LENGTH, tags, depth);
    }

    /**
     * Report batch wait time (enqueue to dispatch) via {@code routing.queue.wait.time.ms}.
     */
    public void reportBatchWaitTimeMs(String role, String engineIp, long waitMs) {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", role,
                "engineIp", engineIp);
        monitor.report(ROUTING_QUEUE_WAIT_TIME_MS, tags, waitMs);
    }

    // ==================== Dispatch reason metrics ====================

    /**
     * Report batch dispatch reason via {@code engine.balancing.master.dispatch.reason}.
     */
    public void reportDispatchReason(String role, String engineIp, String reason) {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", role,
                "engineIp", engineIp,
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
    public void reportBatchCacheHitMetrics(String role, String engineIp, long hitTokens, long totalTokens) {
        if (totalTokens <= 0L) {
            return;
        }
        double hitRatio = hitTokens / (double) totalTokens;
        FlexMetricTags tags = FlexMetricTags.of(
                "role", role,
                "engineIp", engineIp);
        monitor.report(CACHE_HIT_COUNT, tags, hitTokens);
        monitor.report(CACHE_HIT_RATIO, tags, hitRatio);
        monitor.report(CACHE_REQUEST_TOTAL, tags, 1.0);
    }

    /**
     * Report batch size (number of requests dispatched together) via {@code engine.balancing.master.batch.size}.
     */
    public void reportBatchSize(String role, String engineIp, String reason, int batchSize) {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", role,
                "engineIp", engineIp,
                "reason", reason);
        monitor.report(ENGINE_BALANCING_MASTER_BATCH_SIZE, tags, batchSize);
    }

    /**
     * Report batch total token count (sum of seqLen across picked items) via
     * {@code engine.balancing.master.batch.total.tokens}.
     */
    public void reportBatchTotalTokens(String role, String engineIp, String reason, long totalTokens) {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", role,
                "engineIp", engineIp,
                "reason", reason);
        monitor.report(ENGINE_BALANCING_MASTER_BATCH_TOTAL_TOKENS, tags, totalTokens);
    }

    /**
     * Report scheduler inflight map size via {@code health.check.local.task.map.size}.
     * Uses role=PREFILL + engineIp=scheduler tags to match the Grafana panel filter.
     */
    public void reportSchedulerInflightSize(int size) {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", RoleType.PREFILL.name(),
                "engineIp", "scheduler");
        monitor.report(ENGINE_LOCAL_TASK_MAP_SIZE, tags, size);
    }

    /**
     * Report per-worker inflight batch count (number of dispatched-but-uncompleted batches)
     * via {@code flexlb.batch.inflight.count}.
     */
    public void reportPrefillInflightBatchCount(String role, String engineIp, int count) {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", role,
                "engineIp", engineIp);
        monitor.report(BATCH_INFLIGHT_COUNT, tags, count);
    }

    /**
     * Report per-worker inflight request count (sum of requests across all inflight batches)
     * via {@code flexlb.batch.inflight.request.count}.
     */
    public void reportPrefillInflightRequestCount(String role, String engineIp, int count) {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", role,
                "engineIp", engineIp);
        monitor.report(BATCH_INFLIGHT_REQUEST_COUNT, tags, count);
    }

    // ==================== Decode inflight metrics ====================

    /**
     * Report per-decode-worker inflight request count (dispatched but not yet confirmed by engine)
     * via {@code flexlb.decode.inflight.count}.
     */
    public void reportDecodeInflightCount(String engineIp, int count) {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", RoleType.DECODE.name(),
                "engineIp", engineIp);
        monitor.report(DECODE_INFLIGHT_COUNT, tags, count);
    }

    /**
     * Report per-decode-worker total load (confirmed running + scheduler inflight)
     * via {@code flexlb.decode.total.load}.
     */
    public void reportDecodeTotalLoad(String engineIp, int totalLoad) {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", RoleType.DECODE.name(),
                "engineIp", engineIp);
        monitor.report(DECODE_TOTAL_LOAD, tags, totalLoad);
    }

    // ==================== Prediction accuracy metrics ====================

    /**
     * Report formula-predicted batch execution time via {@code batch.predicted.time.ms}.
     */
    public void reportBatchPredictedTimeMs(String role, String engineIp, long predictedMs) {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", role,
                "engineIp", engineIp);
        monitor.report(BATCH_PREDICTED_TIME_MS, tags, predictedMs);
    }

    /**
     * Report engine-reported actual batch execution time via {@code batch.actual.time.ms}.
     */
    public void reportBatchActualTimeMs(String role, String engineIp, long actualMs) {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", role,
                "engineIp", engineIp);
        monitor.report(BATCH_ACTUAL_TIME_MS, tags, actualMs);
    }

    /**
     * Report the gap between actual and predicted batch execution time via {@code batch.predict.gap.ms}.
     */
    public void reportBatchPredictGapMs(String role, String engineIp, long gapMs) {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", role,
                "engineIp", engineIp);
        monitor.report(BATCH_PREDICT_GAP_MS, tags, gapMs);
    }
}
