package org.flexlb.service.monitor;

import org.flexlb.balance.scheduler.TerminalReason;
import org.flexlb.constant.MetricConstant;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;

/**
 * Unified metric reporting helper for cross-path comparison.
 *
 * <p>Shared metrics (success / failure / timeout / cancel / TTFT / inflight size)
 * are reported with a {@code path} tag so that BATCH, QUEUE, and DIRECT paths
 * can be compared side-by-side in Grafana.
 *
 * <p>Adapted to the {@link FlexMonitor} interface which uses
 * {@code register()} + {@code report(metricName, FlexMetricTags, value)} rather
 * than convenience methods like {@code qps()} / {@code timer()} / {@code gauge()}.
 * Metric type is declared at registration time; the caller must invoke
 * {@link #register()} once before reporting.
 *
 * <p>Null-safe: all report methods return early if the wrapped monitor is null.
 */
public class FlexlbMetricHelper {

    private final FlexMonitor monitor;
    private final String path;

    /**
     * @param monitor the monitor to report through (may be null for no-op)
     * @param path    the scheduling path tag value ({@link MetricConstant#PATH_BATCH},
     *                {@link MetricConstant#PATH_QUEUE}, or {@link MetricConstant#PATH_DIRECT})
     */
    public FlexlbMetricHelper(FlexMonitor monitor, String path) {
        this.monitor = monitor;
        this.path = path;
    }

    /**
     * Register all unified metrics. Call once at startup (e.g. from a Spring
     * {@code @PostConstruct} method on the scheduler that owns this helper).
     */
    public void register() {
        if (monitor == null) {
            return;
        }
        monitor.register(MetricConstant.REQUEST_SUCCESS_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.REQUEST_FAILURE_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.REQUEST_TIMEOUT_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.REQUEST_CANCEL_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.REQUEST_TTFT_MS, FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.INFLIGHT_SIZE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        // Auto-TPM priority metrics
        monitor.register(MetricConstant.AUTO_TPM_REQUEST_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.AUTO_TPM_SCHEDULE_LATENCY_MS, FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.AUTO_TPM_NORMAL_PLACEMENT_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.AUTO_TPM_QUEUE_REJECT_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.AUTO_TPM_RUNNING_CANCEL_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.AUTO_TPM_PREEMPT_RATE_PER_MIN, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.AUTO_TPM_TTFT_MS, FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        monitor.register(MetricConstant.AUTO_TPM_DEADLINE_MISS_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
    }

    // ==================== Individual report methods ====================

    /**
     * Report a successful request.
     */
    public void reportSuccess(String role, String engineIp) {
        if (monitor == null) {
            return;
        }
        FlexMetricTags tags = FlexMetricTags.of(
                MetricConstant.TAG_ROLE, role,
                MetricConstant.TAG_ENGINE_IP, engineIp,
                MetricConstant.TAG_PATH, path);
        monitor.report(MetricConstant.REQUEST_SUCCESS_QPS, tags, 1.0);
    }

    /**
     * Report a failed request.
     *
     * @param code error code or {@code "UNKNOWN"} if not available
     */
    public void reportFailure(String role, String engineIp, String code) {
        if (monitor == null) {
            return;
        }
        FlexMetricTags tags = FlexMetricTags.of(
                MetricConstant.TAG_ROLE, role,
                MetricConstant.TAG_ENGINE_IP, engineIp,
                MetricConstant.TAG_PATH, path,
                MetricConstant.TAG_CODE, code != null ? code : "UNKNOWN");
        monitor.report(MetricConstant.REQUEST_FAILURE_QPS, tags, 1.0);
    }

    /**
     * Report a timed-out request.
     */
    public void reportTimeout(String role, String engineIp) {
        if (monitor == null) {
            return;
        }
        FlexMetricTags tags = FlexMetricTags.of(
                MetricConstant.TAG_ROLE, role,
                MetricConstant.TAG_ENGINE_IP, engineIp,
                MetricConstant.TAG_PATH, path);
        monitor.report(MetricConstant.REQUEST_TIMEOUT_QPS, tags, 1.0);
    }

    /**
     * Report a cancelled request.
     */
    public void reportCancel(String role, String engineIp) {
        if (monitor == null) {
            return;
        }
        FlexMetricTags tags = FlexMetricTags.of(
                MetricConstant.TAG_ROLE, role,
                MetricConstant.TAG_ENGINE_IP, engineIp,
                MetricConstant.TAG_PATH, path);
        monitor.report(MetricConstant.REQUEST_CANCEL_QPS, tags, 1.0);
    }

    /**
     * Report time-to-first-token for a request.
     *
     * @param ttftMs time to first token in milliseconds
     */
    public void reportTtft(String role, String engineIp, long ttftMs) {
        if (monitor == null) {
            return;
        }
        FlexMetricTags tags = FlexMetricTags.of(
                MetricConstant.TAG_ROLE, role,
                MetricConstant.TAG_ENGINE_IP, engineIp,
                MetricConstant.TAG_PATH, path);
        monitor.report(MetricConstant.REQUEST_TTFT_MS, tags, ttftMs);
    }

    /**
     * Report current inflight size for this path.
     *
     * @param size current inflight request count
     */
    public void reportInflightSize(String role, String engineIp, int size) {
        if (monitor == null) {
            return;
        }
        FlexMetricTags tags = FlexMetricTags.of(
                MetricConstant.TAG_ROLE, role,
                MetricConstant.TAG_ENGINE_IP, engineIp,
                MetricConstant.TAG_PATH, path);
        monitor.report(MetricConstant.INFLIGHT_SIZE, tags, size);
    }

    // ==================== Terminal reason dispatch ====================

    /**
     * Report a terminal-state transition for a request, dispatching to the
     * appropriate metric based on {@link TerminalReason}.
     *
     * @param reason   the terminal reason
     * @param role     engine role (PREFILL / DECODE / etc.)
     * @param engineIp engine IP address
     * @param code     error code for FAILED transitions (null defaults to "UNKNOWN")
     */
    public void reportTerminal(TerminalReason reason, String role, String engineIp, String code) {
        if (monitor == null || reason == null) {
            return;
        }
        switch (reason) {
            case COMPLETED -> reportSuccess(role, engineIp);
            case FAILED -> reportFailure(role, engineIp, code != null ? code : "UNKNOWN");
            case TIMED_OUT -> reportTimeout(role, engineIp);
            case CANCELLED -> reportCancel(role, engineIp);
        }
    }

    // ==================== Auto-TPM priority metrics ====================
    // D12: the 0 sentinel (no priority carried) never emits priority-tagged
    // metrics — every method below drops the report when priority <= 0.

    /**
     * Report a request arrival with its priority.
     */
    public void reportAutoTpmRequestCount(int priority) {
        if (monitor == null || priority <= 0) {
            return;
        }
        FlexMetricTags tags = FlexMetricTags.of(
                MetricConstant.TAG_PRIORITY, String.valueOf(priority),
                MetricConstant.TAG_PATH, path);
        monitor.report(MetricConstant.AUTO_TPM_REQUEST_COUNT, tags, 1.0);
    }

    /**
     * Report scheduling latency with priority dimension.
     */
    public void reportAutoTpmScheduleLatency(int priority, String result, long latencyMs) {
        if (monitor == null || priority <= 0) {
            return;
        }
        FlexMetricTags tags = FlexMetricTags.of(
                MetricConstant.TAG_PRIORITY, String.valueOf(priority),
                MetricConstant.TAG_RESULT, result,
                MetricConstant.TAG_PATH, path);
        monitor.report(MetricConstant.AUTO_TPM_SCHEDULE_LATENCY_MS, tags, latencyMs);
    }

    /**
     * Report a normal placement (successfully dispatched) with priority.
     */
    public void reportAutoTpmNormalPlacement(int priority) {
        if (monitor == null || priority <= 0) {
            return;
        }
        FlexMetricTags tags = FlexMetricTags.of(
                MetricConstant.TAG_PRIORITY, String.valueOf(priority),
                MetricConstant.TAG_PATH, path);
        monitor.report(MetricConstant.AUTO_TPM_NORMAL_PLACEMENT_COUNT, tags, 1.0);
    }

    /**
     * Report a queue reject event (victim yielded for incoming).
     */
    public void reportAutoTpmQueueReject(int victimPriority, int incomingPriority) {
        if (monitor == null || victimPriority <= 0 || incomingPriority <= 0) {
            return;
        }
        FlexMetricTags tags = FlexMetricTags.of(
                MetricConstant.TAG_VICTIM_PRIORITY, String.valueOf(victimPriority),
                MetricConstant.TAG_INCOMING_PRIORITY, String.valueOf(incomingPriority),
                MetricConstant.TAG_PATH, path);
        monitor.report(MetricConstant.AUTO_TPM_QUEUE_REJECT_COUNT, tags, 1.0);
    }

    /**
     * Report a running-decode cancel (preemption) attempt outcome (D10).
     *
     * @param result one of success/timeout/not_found/unsupported/rate_limited
     */
    public void reportAutoTpmRunningCancel(int victimPriority, int incomingPriority, String result) {
        if (monitor == null || victimPriority <= 0 || incomingPriority <= 0) {
            return;
        }
        FlexMetricTags tags = FlexMetricTags.of(
                MetricConstant.TAG_VICTIM_PRIORITY, String.valueOf(victimPriority),
                MetricConstant.TAG_INCOMING_PRIORITY, String.valueOf(incomingPriority),
                MetricConstant.TAG_RESULT, result,
                MetricConstant.TAG_PATH, path);
        monitor.report(MetricConstant.AUTO_TPM_RUNNING_CANCEL_COUNT, tags, 1.0);
    }

    /**
     * Report the current global preemption rate in the sliding 1-minute window.
     */
    public void reportAutoTpmPreemptRate(int currentPerMin) {
        if (monitor == null) {
            return;
        }
        FlexMetricTags tags = FlexMetricTags.of(MetricConstant.TAG_PATH, path);
        monitor.report(MetricConstant.AUTO_TPM_PREEMPT_RATE_PER_MIN, tags, currentPerMin);
    }

    /**
     * Report the scheduler-side TTFT approximation with priority dimension
     * (D10): submit arrival to engine enqueue ACK.
     */
    public void reportAutoTpmTtft(int priority, long ttftMs) {
        if (monitor == null || priority <= 0) {
            return;
        }
        FlexMetricTags tags = FlexMetricTags.of(
                MetricConstant.TAG_PRIORITY, String.valueOf(priority),
                MetricConstant.TAG_PATH, path);
        monitor.report(MetricConstant.AUTO_TPM_TTFT_MS, tags, ttftMs);
    }

    /**
     * Report a deadline miss (D10): the item was cleared on a queue-deadline
     * path (legacy expiry or yielded-queue-deadline rejection).
     */
    public void reportAutoTpmDeadlineMiss(int priority) {
        if (monitor == null || priority <= 0) {
            return;
        }
        FlexMetricTags tags = FlexMetricTags.of(
                MetricConstant.TAG_PRIORITY, String.valueOf(priority),
                MetricConstant.TAG_PATH, path);
        monitor.report(MetricConstant.AUTO_TPM_DEADLINE_MISS_COUNT, tags, 1.0);
    }
}
