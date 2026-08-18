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

import static org.flexlb.constant.MetricConstant.BATCHER_PARK_QPS;
import static org.flexlb.constant.MetricConstant.BATCHER_QUEUE_ENTER_QPS;
import static org.flexlb.constant.MetricConstant.BATCHER_QUEUE_LEAVE_QPS;
import static org.flexlb.constant.MetricConstant.BATCHER_QUEUE_SIZE;
import static org.flexlb.constant.MetricConstant.ENGINE_WAIT_FILTERED_QPS;
import static org.flexlb.constant.MetricConstant.SELECTION_FALLBACK_QPS;
import static org.flexlb.constant.MetricConstant.BATCH_ACTUAL_TIME_MS;
import static org.flexlb.constant.MetricConstant.BATCH_PREDICTED_TIME_MS;
import static org.flexlb.constant.MetricConstant.BATCH_PREDICT_GAP_MS;
import static org.flexlb.constant.MetricConstant.DISPATCH_ACK_TIME_MS;
import static org.flexlb.constant.MetricConstant.DISPATCH_RECONCILIATION_EVENT_QPS;
import static org.flexlb.constant.MetricConstant.DISPATCH_RECONCILIATION_FENCE_SIZE;
import static org.flexlb.constant.MetricConstant.ACK_TO_RESPONSE_TIME_MS;
import static org.flexlb.constant.MetricConstant.ROUTE_SUBMIT_TIME_MS;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_COUNT;
import static org.flexlb.constant.MetricConstant.CACHE_HIT_RATIO;
import static org.flexlb.constant.MetricConstant.CACHE_REQUEST_TOTAL;
import static org.flexlb.constant.MetricConstant.DECODE_INFLIGHT_KV_RESERVED_TOKENS;
import static org.flexlb.constant.MetricConstant.DECODE_INFLIGHT_HARD_KV_RESERVED_TOKENS;
import static org.flexlb.constant.MetricConstant.DECODE_TOTAL_LOAD;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_BATCH_SIZE;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_BATCH_TOTAL_TOKENS;
import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_DISPATCH_REASON;
import static org.flexlb.constant.MetricConstant.INFLIGHT_BATCH_COUNT;
import static org.flexlb.constant.MetricConstant.INFLIGHT_CLEANUP_SKIPPED_FENCED_QPS;
import static org.flexlb.constant.MetricConstant.INFLIGHT_MAX_AGE_MS;
import static org.flexlb.constant.MetricConstant.INFLIGHT_REQUEST_COUNT;
import static org.flexlb.constant.MetricConstant.INFLIGHT_TTL_EXPIRED_QPS;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_LENGTH;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_WAIT_TIME_MS;
import static org.flexlb.constant.MetricConstant.SCHEDULER_INFLIGHT_AUDIT_RELEASE_QPS;
import static org.flexlb.constant.MetricConstant.SCHEDULER_INFLIGHT_FENCED_MAX_AGE_MS;
import static org.flexlb.constant.MetricConstant.SCHEDULER_INFLIGHT_MAX_AGE_MS;
import static org.flexlb.constant.MetricConstant.SCHEDULER_INFLIGHT_SIZE;
import static org.flexlb.constant.MetricConstant.SCHEDULER_RESTORE_PENDING_DISPATCH_QPS;

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

    /** role tag value for scheduler-ledger series (vs PREFILL/DECODE endpoint ledgers). */
    public static final String SCHEDULER_ROLE = "SCHEDULER";

    /** engineIp tag value for scheduler-ledger series (no real engine behind them). */
    public static final String SCHEDULER_ENGINE_IP = "scheduler";

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
        // Scheduler-level inflight max age — same scheduler-level tags as SCHEDULER_INFLIGHT_SIZE
        monitor.register(SCHEDULER_INFLIGHT_MAX_AGE_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        // Restore-pending-dispatch — Decode capacity full at flush, item returned to batcher queue (QPS)
        monitor.register(SCHEDULER_RESTORE_PENDING_DISPATCH_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);

        // Batcher queue size — per-engine pending batch request count (FlexLB batcher queue depth)
        monitor.register(BATCHER_QUEUE_SIZE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        // Batcher park — requests parked by the batcher instead of dispatched (inflight_full etc.), QPS tagged by reason
        monitor.register(BATCHER_PARK_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        // Batcher queue enter/leave — per-engine admission & departure rates (QPS); leave is
        // tagged by reason (dispatched / deadline_evicted / admission_timeout /
        // token_capacity_rejected / drained / dispatch_aborted / removed) so the
        // enter-vs-leave gap pinpoints where queued requests actually go (na130_4)
        monitor.register(BATCHER_QUEUE_ENTER_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(BATCHER_QUEUE_LEAVE_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        // Prefill selection telemetry — engine-wait hard-filter hits and all-filtered
        // least-loaded fallbacks (cluster-level, tagged by role only)
        monitor.register(ENGINE_WAIT_FILTERED_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(SELECTION_FALLBACK_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);

        // Decode total load and inflight KV reserved — per decode worker (FlexLB scheduler view)
        monitor.register(DECODE_TOTAL_LOAD, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(DECODE_INFLIGHT_KV_RESERVED_TOKENS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(DECODE_INFLIGHT_HARD_KV_RESERVED_TOKENS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(INFLIGHT_MAX_AGE_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        // Inflight TTL expired — count of inflight requests cleaned up by the TTL task, QPS tagged by role
        monitor.register(INFLIGHT_TTL_EXPIRED_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        // Inflight cleanup fence skips — TTL-due entries retained by a stronger fence, QPS tagged by role
        monitor.register(INFLIGHT_CLEANUP_SKIPPED_FENCED_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);

        // Dispatch reconciliation — fence lifecycle events (QPS) and live fence population (gauge)
        monitor.register(DISPATCH_RECONCILIATION_EVENT_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(DISPATCH_RECONCILIATION_FENCE_SIZE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        // Post-ACK invisible releases (QPS tagged by reason) and fenced inflight max age (gauge)
        monitor.register(SCHEDULER_INFLIGHT_AUDIT_RELEASE_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(SCHEDULER_INFLIGHT_FENCED_MAX_AGE_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

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

        log.info("BatchSchedulerReporter initialized (32 metrics)");
    }

    // ==================== Queue metrics ====================

    /**
     * Report per-worker batcher queue depth via {@code routing.queue.length}.
     *
     * @deprecated Replaced by {@link #reportBatcherQueueDepthByPriority} which
     *             carries the priority tag. Retained for backward compatibility.
     */
    @Deprecated
    public void reportBatcherQueueDepth(String role, String engineIp, int depth) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "type", "batchQueue",
                "role", role);
        monitor.report(ROUTING_QUEUE_LENGTH, tags, depth);
    }

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
     * Report batcher park decisions via {@code app.flexlb.batcher.park.qps},
     * tagged by park reason (inflight_full / outside_window /
     * wait_for_min_batch / wait_for_target_batch).
     * <p>Aggregated across endpoints (no engineIp tag): the park QPS by reason
     * is the cluster-level signal that requests are silently waiting on
     * backpressure instead of being dispatched or rejected. {@code count} is
     * the parks accumulated since the caller's previous flush (one report per
     * reason per 10s window — hotspot-free under backpressure storms).
     */
    public void reportBatcherPark(String reason, long count) {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", RoleType.PREFILL.name(),
                "reason", reason);
        monitor.report(BATCHER_PARK_QPS, tags, count);
    }

    /**
     * Report one batcher queue admission via
     * {@code app.flexlb.batcher.queue.enter.qps}, tagged by engineIp + role.
     * Counted at the single enqueue success point shared by offer / tryOffer
     * / versioned re-offer, so every path that puts an item into a per-engine
     * batcher queue is counted exactly once.
     */
    public void reportBatcherQueueEnter(String role, String engineIp) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", role);
        monitor.report(BATCHER_QUEUE_ENTER_QPS, tags, 1.0);
    }

    /**
     * Report batcher queue departures via
     * {@code app.flexlb.batcher.queue.leave.qps}, tagged by engineIp + role
     * + reason:
     * dispatched (flush-time dispatch to the engine),
     * deadline_evicted (SLO-budget dropHead expiry),
     * admission_timeout (queue admission timeout sweep),
     * token_capacity_rejected (strict padded batch-token capacity reject),
     * drained (batcher shutdown stopAndDrainTo, queued + staged),
     * dispatch_aborted (pre-send revalidation drop / claim-or-send failure),
     * removed (all other removals — cancel/replace paths).
     *
     * @param count number of items that left the queue with this reason
     */
    public void reportBatcherQueueLeave(String role, String engineIp, String reason, int count) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", role,
                "reason", reason);
        monitor.report(BATCHER_QUEUE_LEAVE_QPS, tags, count);
    }

    /**
     * Report engine-wait hard-filter hits via
     * {@code app.flexlb.engine.wait.filtered.qps}, tagged by role (PREFILL).
     * Aggregated per prefill selection round ({@code count} = candidate
     * endpoints excluded); the filter decision is cluster-level so there is
     * no single engineIp tag.
     *
     * @param count endpoints excluded by the engine-wait hard filter this round
     */
    public void reportEngineWaitFiltered(int count) {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", RoleType.PREFILL.name());
        monitor.report(ENGINE_WAIT_FILTERED_QPS, tags, count);
    }

    /**
     * Report one least-loaded fallback prefill selection via
     * {@code app.flexlb.selection.fallback.qps}, tagged by role (PREFILL) —
     * every feasible candidate was filtered out and the strategy kept
     * routing by falling back to the least-loaded endpoint.
     */
    public void reportSelectionFallback() {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", RoleType.PREFILL.name());
        monitor.report(SELECTION_FALLBACK_QPS, tags, 1.0);
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
     * Report the age (ms) of the oldest entry in the scheduler's own inflight
     * ledger via {@code app.flexlb.scheduler.inflight.max.age.ms}.
     * <p>Uses the same scheduler-level tags as {@link #reportSchedulerInflightSize}
     * (role=PREFILL, engineIp="scheduler") so the size and age gauges stay
     * joinable in the same panel. Unlike the per-worker
     * {@link #reportInflightMaxAgeMs}, this gauge is immune to per-endpoint
     * ledger releases — it exposes master-side leaks (fence-skipped or
     * post-ACK entries) that keep the scheduler ledger pinned.
     *
     * @param ageMs age of the oldest scheduler inflight entry, 0 when empty
     */
    public void reportSchedulerInflightMaxAgeMs(long ageMs) {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", RoleType.PREFILL.name(),
                "engineIp", "scheduler");
        monitor.report(SCHEDULER_INFLIGHT_MAX_AGE_MS, tags, ageMs);
        // Unified series: same metric name and tag schema ({engineIp, role})
        // as the per-worker reportInflightMaxAgeMs, so a single role='*'
        // grouping on inflight.max.age.ms compares the scheduler ledger
        // against the PREFILL/DECODE endpoint ledgers — the exact contrast
        // the post-ACK ghost incident lacked (endpoints ~10s, scheduler 300s+).
        monitor.report(INFLIGHT_MAX_AGE_MS,
                FlexMetricTags.ofEngine(SCHEDULER_ENGINE_IP, "role", SCHEDULER_ROLE), ageMs);
    }

    /**
     * Report one restore-pending-dispatch event — a flush-time item returned
     * to the batcher queue because the Decode concurrency gate reported
     * CAPACITY_FULL — via {@code app.flexlb.scheduler.restore.pending.dispatch.qps}.
     * <p>Scheduler-level metric tagged by role only.
     */
    public void reportSchedulerRestorePendingDispatch() {
        FlexMetricTags tags = FlexMetricTags.of("role", RoleType.PREFILL.name());
        monitor.report(SCHEDULER_RESTORE_PENDING_DISPATCH_QPS, tags, 1.0);
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

    /**
     * Report the age (ms) of the oldest inflight entry per worker
     * via {@code flexlb.inflight.max.age.ms}.
     */
    public void reportInflightMaxAgeMs(String role, String engineIp, long ageMs) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", role);
        monitor.report(INFLIGHT_MAX_AGE_MS, tags, ageMs);
    }

    /**
     * Report inflight entries evicted from the scheduler ledger by the TTL
     * cleanup task via {@code app.flexlb.inflight.ttl.expired.qps}.
     * <p>Tagged role=SCHEDULER + engineIp="scheduler" (the ledger that
     * evicted — the former hardcoded role=PREFILL tag mislabelled these
     * scheduler-level evictions as an endpoint series) + reason
     * (ttl / hard_age_cap), keeping one tag schema with the endpoint series.
     *
     * @param reason eviction reason bucket
     * @param count  number of entries evicted in this cleanup cycle
     */
    public void reportSchedulerInflightTtlExpired(String reason, int count) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(SCHEDULER_ENGINE_IP,
                "role", SCHEDULER_ROLE,
                "reason", reason);
        monitor.report(INFLIGHT_TTL_EXPIRED_QPS, tags, count);
    }

    /**
     * Report inflight entries evicted from an endpoint ledger
     * (PrefillEndpoint.evictExpiredBatches / DecodeEndpoint.evictExpiredRequests
     * / orphan decode reservation reclaims) via the same
     * {@code app.flexlb.inflight.ttl.expired.qps} series family.
     * <p>Endpoint-side evictions were previously log-only
     * (event=endpoint_inflight_ttl_eviction); this closes the gap with the
     * shared {role, engineIp, reason} tag schema.
     */
    public void reportEndpointInflightTtlExpired(String role, String engineIp, String reason, int count) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", role,
                "reason", reason);
        monitor.report(INFLIGHT_TTL_EXPIRED_QPS, tags, count);
    }

    /**
     * Report the count of inflight entries skipped by the TTL cleanup because
     * a stronger fence (preemption claim / dispatch reconciliation / cleanup
     * ownership) still owns them, via
     * {@code app.flexlb.inflight.cleanup.skipped.fenced.qps}.
     * <p>Scheduler-level metric tagged by role only. A persistently
     * non-zero value means
     * fence-held entries are accumulating past their TTL — the population the
     * hard age cap and the post-ACK audit eventually reclaim.
     *
     * @param count number of fence-held entries skipped in this cleanup cycle
     */
    public void reportInflightCleanupSkippedFenced(int count) {
        FlexMetricTags tags = FlexMetricTags.of("role", RoleType.PREFILL.name());
        monitor.report(INFLIGHT_CLEANUP_SKIPPED_FENCED_QPS, tags, count);
    }

    // ==================== Dispatch reconciliation metrics ====================

    /**
     * Report one dispatch-reconciliation fence lifecycle event via
     * {@code app.flexlb.dispatch.reconciliation.event.qps}.
     * <p>Events: start (uncertain ACK entered reconciliation), settled
     * (engine TOMBSTONED confirmed the fence), forced_terminal (fence
     * released without engine confirmation — natural alert point).
     * Metricizes the previously log-only event=dispatch_reconciliation_*
     * lines.
     *
     * @param event  start / settled / forced_terminal
     * @param reason event qualifier (uncertain_ack / engine_tombstoned /
     *               target_deregistered / failure_cap)
     */
    public void reportDispatchReconciliationEvent(String event, String reason) {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", SCHEDULER_ROLE,
                "event", event,
                "reason", reason);
        monitor.report(DISPATCH_RECONCILIATION_EVENT_QPS, tags, 1.0);
    }

    /**
     * Report the number of scheduler inflight entries currently holding the
     * dispatch-reconciliation fence via
     * {@code app.flexlb.dispatch.reconciliation.fence.size}.
     * <p>A fence population stuck above zero while dispatch QPS is zero is
     * the post-ACK ghost signature the TTL sweep cannot see (fenced entries
     * are skipped until the hard age cap).
     */
    public void reportDispatchReconciliationFenceSize(int size) {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", SCHEDULER_ROLE,
                "engineIp", SCHEDULER_ENGINE_IP);
        monitor.report(DISPATCH_RECONCILIATION_FENCE_SIZE, tags, size);
    }

    /**
     * Report post-ACK invisible inflight releases accumulated over the
     * caller's report window via
     * {@code app.flexlb.scheduler.inflight.audit.release.qps}, tagged by
     * release reason (post_ack_audit / decode_vanish_sync).
     * <p>Window-aggregated by the scheduler (LongAdder flush on the 2s
     * metrics tick) — never called per event on the WorkerStatus sync hot
     * path.
     */
    public void reportSchedulerInflightAuditRelease(String reason, long count) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(SCHEDULER_ENGINE_IP,
                "role", SCHEDULER_ROLE,
                "reason", reason);
        monitor.report(SCHEDULER_INFLIGHT_AUDIT_RELEASE_QPS, tags, count);
    }

    /**
     * Report the age of the oldest fence-held scheduler inflight entry
     * (preemption claim / dispatch reconciliation / cleanup ownership) via
     * {@code app.flexlb.scheduler.inflight.fenced.max.age.ms}; 0 when no
     * entry is fenced. Complements
     * {@link #reportDispatchReconciliationFenceSize}: the size gauge alone
     * cannot distinguish a healthy fence rotation from D/E-class entries
     * stuck behind their fences.
     */
    public void reportSchedulerInflightFencedMaxAgeMs(long ageMs) {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", SCHEDULER_ROLE,
                "engineIp", SCHEDULER_ENGINE_IP);
        monitor.report(SCHEDULER_INFLIGHT_FENCED_MAX_AGE_MS, tags, ageMs);
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
     * Report per-decode-worker hard KV cache reserved tokens (hard reservation that cannot be reclaimed)
     * via {@code flexlb.decode.inflight.hard.kv.reserved.tokens}.
     */
    public void reportDecodeInflightHardKvReserved(String engineIp, long kvReservedTokens) {
        FlexMetricTags tags = FlexMetricTags.ofEngine(engineIp,
                "role", RoleType.DECODE.name());
        monitor.report(DECODE_INFLIGHT_HARD_KV_RESERVED_TOKENS, tags, kvReservedTokens);
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
            monitor.prepare(ROUTING_QUEUE_WAIT_TIME_MS, tags);
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
}
