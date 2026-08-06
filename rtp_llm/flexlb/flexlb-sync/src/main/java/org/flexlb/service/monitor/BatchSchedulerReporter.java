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
import java.util.concurrent.ConcurrentHashMap;

import static org.flexlb.constant.MetricConstant.BATCHER_QUEUE_SIZE;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_EXPIRED_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_QUEUE_DEPTH;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_QUEUE_WAIT_TIME_MS;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_REQUEST_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_SCHEDULE_TO_ACK_TIME_MS;
import static org.flexlb.constant.MetricConstant.BATCH_ACTUAL_TIME_MS;
import static org.flexlb.constant.MetricConstant.BATCH_PREDICTED_TIME_MS;
import static org.flexlb.constant.MetricConstant.BATCH_PREDICT_GAP_MS;
import static org.flexlb.constant.MetricConstant.DISPATCH_ACK_TIME_MS;
import static org.flexlb.constant.MetricConstant.ACK_TO_RESPONSE_TIME_MS;
import static org.flexlb.constant.MetricConstant.ROUTE_SUBMIT_TIME_MS;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_RATIO;
import static org.flexlb.constant.MetricConstant.CACHE_REQUEST_TOTAL;
import static org.flexlb.constant.MetricConstant.DECODE_ENGINE_LOADING_COUNT;
import static org.flexlb.constant.MetricConstant.DECODE_ENGINE_RUNNING_COUNT;
import static org.flexlb.constant.MetricConstant.DECODE_ENGINE_TASKS_COUNT;
import static org.flexlb.constant.MetricConstant.DECODE_ENGINE_WAITING_COUNT;
import static org.flexlb.constant.MetricConstant.DECODE_INFLIGHT_KV_RESERVED_HARD_TOKENS;
import static org.flexlb.constant.MetricConstant.DECODE_INFLIGHT_KV_RESERVED_TOKENS;
import static org.flexlb.constant.MetricConstant.DECODE_INFLIGHT_REQUESTS_COUNT;
import static org.flexlb.constant.MetricConstant.DECODE_TOTAL_LOAD;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_BATCH_SIZE;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_BATCH_TOTAL_TOKENS;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_DISPATCH_REASON;
import static org.flexlb.constant.MetricConstant.INFLIGHT_BATCH_COUNT;
import static org.flexlb.constant.MetricConstant.INFLIGHT_REQUEST_COUNT;
import static org.flexlb.constant.MetricConstant.INFLIGHT_TTL_EXPIRED_QPS;
import static org.flexlb.constant.MetricConstant.BATCH_QUEUE_WAIT_TIME_MS;
import static org.flexlb.constant.MetricConstant.PREFILL_ENGINE_TASKS_COUNT;
import static org.flexlb.constant.MetricConstant.PREFILL_INFLIGHT_ENTRIES_COUNT;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_LENGTH;
import static org.flexlb.constant.MetricConstant.SCHEDULER_INFLIGHT_SIZE;
import static org.flexlb.constant.MetricConstant.SCHEDULER_INFLIGHT_TOTAL_SIZE;

/**
 * Batch scheduling metrics reporter for FlexLB batch dispatch path.
 *
 * <p>Batch-path metrics use independent metric names to avoid tag schema
 * conflicts with the non-batch path:
 * queue (routing.queue.length + flexlb.batch.queue.wait.time.ms),
 * dispatch reason (engine.balancing.master.dispatch.reason),
 * inflight (flexlb.scheduler.inflight.size).
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
        // Batch queue wait time — batch-path-only TIMER, independent from the
        // non-batch routing.queue.wait.time.ms GAUGE owned by RoutingQueueReporter.
        // routing.queue.length itself is registered by RoutingQueueReporter;
        // this reporter only reports to it (type=batchQueue tag).
        monitor.register(BATCH_QUEUE_WAIT_TIME_MS, FlexMetricType.TIMER, FlexPriorityType.PRECISE);

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
        monitor.register(SCHEDULER_INFLIGHT_SIZE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        // Batcher queue size — per-engine pending batch request count (FlexLB batcher queue depth)
        monitor.register(BATCHER_QUEUE_SIZE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        // Decode total load and inflight KV reserved — per decode worker (FlexLB scheduler view)
        monitor.register(DECODE_TOTAL_LOAD, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(DECODE_INFLIGHT_KV_RESERVED_TOKENS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(DECODE_INFLIGHT_KV_RESERVED_HARD_TOKENS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        // Decode layer-2 phase-split counts (WAITING / LOADING / RUNNING)
        monitor.register(DECODE_ENGINE_WAITING_COUNT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(DECODE_ENGINE_LOADING_COUNT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(DECODE_ENGINE_RUNNING_COUNT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        // Two-layer breakdown — prefill and decode layer-1 / layer-2 counts
        monitor.register(PREFILL_INFLIGHT_ENTRIES_COUNT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(PREFILL_ENGINE_TASKS_COUNT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(DECODE_INFLIGHT_REQUESTS_COUNT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(DECODE_ENGINE_TASKS_COUNT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        // Inflight TTL eviction QPS — reported by InflightStore.evict()
        monitor.register(INFLIGHT_TTL_EXPIRED_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);

        // Scheduler-level inflight total size (includes tombstones)
        monitor.register(SCHEDULER_INFLIGHT_TOTAL_SIZE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

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

        // Auto-TPM Queue MVP metrics — priority-tagged, reported regardless of
        // whether the priority pick-order switch is on
        monitor.register(AUTO_TPM_REQUEST_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_QUEUE_WAIT_TIME_MS, FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_SCHEDULE_TO_ACK_TIME_MS, FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_EXPIRED_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_QUEUE_DEPTH, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        log.info("BatchSchedulerReporter initialized");
    }

    // ==================== Queue metrics ====================

    /**
     * Report per-worker batcher queue depth via {@code routing.queue.length}.
     */
    public void reportBatcherQueueDepth(String role, String engineIp, int depth) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
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
    public void reportBatcherQueueSize(String role, String engineIp, int depth) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", role);
        monitor.report(BATCHER_QUEUE_SIZE, tags, depth);
    }

    /**
     * Report batch wait time (enqueue to dispatch) via {@code app.flexlb.batch.queue.wait.time.ms}.
     */
    public void reportBatchWaitTimeMs(String role, String engineIp, long waitMs) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", role);
        monitor.report(BATCH_QUEUE_WAIT_TIME_MS, tags, waitMs);
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
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", role);
        monitor.report(CACHE_HIT_COUNT, tags, hitTokens);
        monitor.report(CACHE_HIT_RATIO, tags, hitRatio);
        monitor.report(CACHE_REQUEST_TOTAL, tags, 1.0);
    }

    /**
     * Report batch size (number of requests dispatched together) via {@code engine.balancing.master.batch.size}.
     */
    public void reportBatchSize(String role, String engineIp, String reason, int batchSize) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", role,
                "reason", reason);
        monitor.report(ENGINE_BALANCING_MASTER_BATCH_SIZE, tags, batchSize);
    }

    /**
     * Report batch total token count (sum of seqLen across picked items) via
     * {@code engine.balancing.master.batch.total.tokens}.
     */
    public void reportBatchTotalTokens(String role, String engineIp, String reason, long totalTokens) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
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
                "engineIp", "scheduler");
        monitor.report(SCHEDULER_INFLIGHT_SIZE, tags, size);
    }

    /**
     * Report scheduler inflight total size (including tombstones within TTL)
     * via {@code app.flexlb.scheduler.inflight.total.size}.
     * <p>Uses the same scheduler-level tag schema as {@link #reportSchedulerInflightSize}.
     */
    public void reportSchedulerInflightTotalSize(int size) {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", RoleType.PREFILL.name(),
                "engineIp", "scheduler");
        monitor.report(SCHEDULER_INFLIGHT_TOTAL_SIZE, tags, size);
    }

    /**
     * Report an inflight TTL eviction event (RUNNING item timed out via
     * {@code InflightItem#timeoutWithError()}) via {@code app.flexlb.inflight.ttl.expired.qps}.
     * <p>Called by {@link org.flexlb.balance.scheduler.InflightStore#evict()} on each
     * successful timeout. Uses scheduler-level tags (no engineIp — the TTL sweep is
     * a scheduler-level operation, not per-engine).
     */
    public void reportInflightTtlExpired() {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", RoleType.PREFILL.name(),
                "engineIp", "scheduler");
        monitor.report(INFLIGHT_TTL_EXPIRED_QPS, tags, 1.0);
    }

    /**
     * Report per-worker inflight batch count (number of dispatched-but-uncompleted batches)
     * via {@code flexlb.inflight.batch.count}.
     * <p>Unified for both prefill and decode workers, tagged by role and engineIp.
     */
    public void reportInflightBatchCount(String role, String engineIp, int count) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", role);
        monitor.report(INFLIGHT_BATCH_COUNT, tags, count);
    }

    /**
     * Report per-worker inflight request count (dispatched but not yet confirmed by engine)
     * via {@code flexlb.inflight.request.count}.
     * <p>Unified for both prefill and decode workers, tagged by role and engineIp.
     * Replaces the former separate reportPrefillInflightRequestCount and reportDecodeInflightCount.
     */
    public void reportInflightRequestCount(String role, String engineIp, int count) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", role);
        monitor.report(INFLIGHT_REQUEST_COUNT, tags, count);
    }

    // ==================== Decode inflight metrics ====================

    /**
     * Report per-decode-worker total load (confirmed running + scheduler inflight)
     * via {@code flexlb.decode.total.load}.
     */
    public void reportDecodeTotalLoad(String engineIp, int totalLoad) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", RoleType.DECODE.name());
        monitor.report(DECODE_TOTAL_LOAD, tags, totalLoad);
    }

    /**
     * Report per-decode-worker inflight KV cache reserved tokens (local inflight reservation not yet confirmed by the engine)
     * via {@code flexlb.decode.inflight.kv.reserved.tokens}.
     */
    public void reportDecodeInflightKvReserved(String engineIp, long kvReservedTokens) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", RoleType.DECODE.name());
        monitor.report(DECODE_INFLIGHT_KV_RESERVED_TOKENS, tags, kvReservedTokens);
    }

    /**
     * Report per-decode-worker inflight hard KV cache reserved tokens (layer-1 Σ kvTokens, seqLen-only
     * hard demand) via {@code app.flexlb.decode.inflight.kv.reserved.hard.tokens}.
     * <p>Complements {@link #reportDecodeInflightKvReserved} which carries the expected
     * (seqLen + maxNewTokens) view.
     */
    public void reportDecodeInflightKvReservedHard(String engineIp, long kvReservedHardTokens) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", RoleType.DECODE.name());
        monitor.report(DECODE_INFLIGHT_KV_RESERVED_HARD_TOKENS, tags, kvReservedHardTokens);
    }

    // ==================== Decode phase-split metrics ====================

    /**
     * Report per-decode-worker engine-accepted tasks in the WAITING phase
     * via {@code app.flexlb.decode.engine.waiting.count}.
     */
    public void reportDecodeEngineWaitingCount(String engineIp, int count) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", RoleType.DECODE.name());
        monitor.report(DECODE_ENGINE_WAITING_COUNT, tags, count);
    }

    /**
     * Report per-decode-worker engine-accepted tasks in the LOADING phase (remote KV loading)
     * via {@code app.flexlb.decode.engine.loading.count}.
     */
    public void reportDecodeEngineLoadingCount(String engineIp, int count) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", RoleType.DECODE.name());
        monitor.report(DECODE_ENGINE_LOADING_COUNT, tags, count);
    }

    /**
     * Report per-decode-worker engine-accepted tasks in the RUNNING phase
     * via {@code app.flexlb.decode.engine.running.count}.
     */
    public void reportDecodeEngineRunningCount(String engineIp, int count) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", RoleType.DECODE.name());
        monitor.report(DECODE_ENGINE_RUNNING_COUNT, tags, count);
    }

    // ==================== Two-layer breakdown metrics ====================

    /**
     * Report per-prefill-worker layer-1 inflight entry count (dispatched, not yet acknowledged)
     * via {@code app.flexlb.prefill.inflight.entries.count}.
     */
    public void reportPrefillInflightEntriesCount(String role, String engineIp, int count) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", role);
        monitor.report(PREFILL_INFLIGHT_ENTRIES_COUNT, tags, count);
    }

    /**
     * Report per-prefill-worker layer-2 engine-acknowledged task count
     * via {@code app.flexlb.prefill.engine.tasks.count}.
     */
    public void reportPrefillEngineTasksCount(String role, String engineIp, int count) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", role);
        monitor.report(PREFILL_ENGINE_TASKS_COUNT, tags, count);
    }

    /**
     * Report per-decode-worker layer-1 inflight request count (reserved locally, not yet accepted)
     * via {@code app.flexlb.decode.inflight.requests.count}.
     */
    public void reportDecodeInflightRequestsCount(String engineIp, int count) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", RoleType.DECODE.name());
        monitor.report(DECODE_INFLIGHT_REQUESTS_COUNT, tags, count);
    }

    /**
     * Report per-decode-worker layer-2 engine-accepted task count
     * via {@code app.flexlb.decode.engine.tasks.count}.
     */
    public void reportDecodeEngineTasksCount(String engineIp, int count) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", RoleType.DECODE.name());
        monitor.report(DECODE_ENGINE_TASKS_COUNT, tags, count);
    }

    // ==================== Prediction accuracy metrics ====================

    /**
     * Report formula-predicted batch execution time via {@code app.flexlb.batch.predicted.time.ms}.
     */
    public void reportBatchPredictedTimeMs(String role, String engineIp, long predictedMs) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", role);
        monitor.report(BATCH_PREDICTED_TIME_MS, tags, predictedMs);
    }

    /**
     * Report engine-reported actual batch execution time via {@code app.flexlb.batch.actual.time.ms}.
     */
    public void reportBatchActualTimeMs(String role, String engineIp, long actualMs) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", role);
        monitor.report(BATCH_ACTUAL_TIME_MS, tags, actualMs);
    }

    /**
     * Report the gap between actual and predicted batch execution time via {@code app.flexlb.batch.predict.gap.ms}.
     */
    public void reportBatchPredictGapMs(String role, String engineIp, long gapMs) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
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
    public void reportDispatchAckTimeMs(String role, String engineIp, long ackTimeMs) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", role);
        monitor.report(DISPATCH_ACK_TIME_MS, tags, ackTimeMs);
    }

    /** Prepare schedule-path meters before an endpoint receives traffic. */
    public void prepareEndpointMetrics(String role, String engineIp) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", role);
        if (RoleType.PREFILL.name().equals(role) || RoleType.PDFUSION.name().equals(role)) {
            monitor.prepare(DISPATCH_ACK_TIME_MS, tags);
            monitor.prepare(ROUTE_SUBMIT_TIME_MS, tags);
            monitor.prepare(BATCH_QUEUE_WAIT_TIME_MS, tags);
            for (String reason : FIXED_WINDOW_DISPATCH_REASONS) {
                FlexMetricTags reasonTags = FlexMetricTags.ofEngine(engineIp,
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
    public void reportRouteSubmitTimeMs(String role, String engineIp, long submitMs) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
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
    public void reportAckToResponseTimeMs(String role, String engineIp, long ackToResponseMs) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", role);
        monitor.report(ACK_TO_RESPONSE_TIME_MS, tags, ackToResponseMs);
    }

    // ==================== Auto-TPM Queue MVP metrics ====================

    /**
     * Auto-TPM tag caches: priorities are a small fixed set ({30..70} after
     * normalization) and engine IPs are one per endpoint, so tags are cached
     * to keep per-request report calls allocation-free on the hot path.
     */
    private final ConcurrentHashMap<Integer, FlexMetricTags> autoTpmPriorityTags = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Integer, ConcurrentHashMap<String, FlexMetricTags>> autoTpmEngineTags =
            new ConcurrentHashMap<>();

    private FlexMetricTags autoTpmTags(int priority) {
        return autoTpmPriorityTags.computeIfAbsent(priority,
                p -> FlexMetricTags.of("priority", String.valueOf(p)));
    }

    private FlexMetricTags autoTpmTags(int priority, String engineIp) {
        return autoTpmEngineTags
                .computeIfAbsent(priority, p -> new ConcurrentHashMap<>())
                .computeIfAbsent(engineIp,
                        ip -> FlexMetricTags.ofEngine(ip, "priority", String.valueOf(priority)));
    }

    /**
     * Report an Auto-TPM request arrival via {@code auto_tpm.request.count}.
     * Called by BatchScheduler.submit() after a successful batcher offer.
     */
    public void reportAutoTpmRequestCount(int priority) {
        monitor.report(AUTO_TPM_REQUEST_COUNT, autoTpmTags(priority), 1.0);
    }

    /**
     * Report per-item batcher queue wait by priority via
     * {@code auto_tpm.queue.wait.time.ms}.
     */
    public void reportAutoTpmQueueWaitTimeMs(int priority, String engineIp, long waitMs) {
        monitor.report(AUTO_TPM_QUEUE_WAIT_TIME_MS, autoTpmTags(priority, engineIp), waitMs);
    }

    /**
     * Report schedule-to-ack time (TTFT proxy) by priority via
     * {@code auto_tpm.schedule.to.ack.time.ms}.
     */
    public void reportAutoTpmScheduleToAckTimeMs(int priority, long scheduleToAckMs) {
        monitor.report(AUTO_TPM_SCHEDULE_TO_ACK_TIME_MS, autoTpmTags(priority), scheduleToAckMs);
    }

    /**
     * Report a queue-deadline expiry by priority via {@code auto_tpm.expired.count}.
     * Core starvation observation for low-priority requests.
     */
    public void reportAutoTpmExpiredCount(int priority) {
        monitor.report(AUTO_TPM_EXPIRED_COUNT, autoTpmTags(priority), 1.0);
    }

    /**
     * Report batcher queue depth by priority via {@code auto_tpm.queue.depth}.
     * Called on the existing periodic per-endpoint metric path.
     */
    public void reportAutoTpmQueueDepth(int priority, String engineIp, int depth) {
        monitor.report(AUTO_TPM_QUEUE_DEPTH, autoTpmTags(priority, engineIp), depth);
    }
}
