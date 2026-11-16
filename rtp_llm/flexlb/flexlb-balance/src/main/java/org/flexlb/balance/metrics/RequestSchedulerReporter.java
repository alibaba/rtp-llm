package org.flexlb.balance.metrics;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.config.service.RoleType;
import org.flexlb.util.PriorityNormalizer;
import org.flexlb.util.metrics.FlexMetricTags;
import org.flexlb.util.metrics.FlexMetricType;
import org.flexlb.util.metrics.FlexMonitor;
import org.flexlb.util.metrics.FlexPriorityType;
import org.flexlb.util.metrics.SafeFlexMonitor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

import static org.flexlb.util.metrics.MetricConstant.AUTO_TPM_CANCEL_CONFIRM_COUNT;
import static org.flexlb.util.metrics.MetricConstant.AUTO_TPM_CANCEL_QPS;
import static org.flexlb.util.metrics.MetricConstant.AUTO_TPM_CANCEL_REQUEST_COUNT;
import static org.flexlb.util.metrics.MetricConstant.AUTO_TPM_CANCEL_TIMEOUT_COUNT;
import static org.flexlb.util.metrics.MetricConstant.AUTO_TPM_DECODE_ACCEPTED_COUNT;
import static org.flexlb.util.metrics.MetricConstant.AUTO_TPM_DECODE_ENGINE_LOAD;
import static org.flexlb.util.metrics.MetricConstant.AUTO_TPM_DECODE_RESERVED_COUNT;
import static org.flexlb.util.metrics.MetricConstant.AUTO_TPM_DECODE_RUNNING_COUNT;
import static org.flexlb.util.metrics.MetricConstant.AUTO_TPM_DECODE_SHADOW_KV_RESERVED;
import static org.flexlb.util.metrics.MetricConstant.AUTO_TPM_EVICTION_COMMIT_COUNT;
import static org.flexlb.util.metrics.MetricConstant.AUTO_TPM_EVICTION_PLAN_COUNT;
import static org.flexlb.util.metrics.MetricConstant.AUTO_TPM_PREFILL_QUEUE_DEPTH;
import static org.flexlb.util.metrics.MetricConstant.AUTO_TPM_PRIORITY_PREEMPT_COUNT;
import static org.flexlb.util.metrics.MetricConstant.AUTO_TPM_REQUEST_COUNT;
import static org.flexlb.util.metrics.MetricConstant.AUTO_TPM_SCHEDULE_LATENCY_MS;
import static org.flexlb.util.metrics.MetricConstant.AUTO_TPM_TTFT_MS;
import static org.flexlb.util.metrics.MetricConstant.AUTO_TPM_VICTIM_COUNT;
import static org.flexlb.util.metrics.MetricConstant.AUTO_TPM_VICTIM_KV_TOKENS;
import static org.flexlb.util.metrics.MetricConstant.ENGINE_BALANCING_MASTER_SELECT_DETAIL;
import static org.flexlb.util.metrics.MetricConstant.ENGINE_BALANCING_MASTER_WORKER_SELECT_DETAIL;

/**
 * Reports selected Workers, selection results and per-priority scheduling metrics,
 * including request volume, latency, preemption, cancellation and resource usage.
 */
@Slf4j
@Component
public class RequestSchedulerReporter {
    /**
     * Priority is a normalized level, so its single-tag set is shared per
     * level instead of being rebuilt on every report.
     */
    private static final FlexMetricTags[] PRIORITY_TAGS = buildPriorityTags();

    private final FlexMonitor monitor;

    @Autowired
    public RequestSchedulerReporter(FlexMonitor monitor) {
        this.monitor = SafeFlexMonitor.wrap(monitor);
    }

    @PostConstruct
    public void init() {
        monitor.register(
                ENGINE_BALANCING_MASTER_SELECT_DETAIL,
                FlexMetricType.QPS,
                FlexPriorityType.PRECISE);
        monitor.register(
                ENGINE_BALANCING_MASTER_WORKER_SELECT_DETAIL,
                FlexMetricType.QPS,
                FlexPriorityType.PRECISE);
        for (String metric :
                new String[] {
                    AUTO_TPM_REQUEST_COUNT,
                    AUTO_TPM_EVICTION_PLAN_COUNT,
                    AUTO_TPM_EVICTION_COMMIT_COUNT,
                    AUTO_TPM_VICTIM_COUNT,
                    AUTO_TPM_PRIORITY_PREEMPT_COUNT,
                    AUTO_TPM_CANCEL_REQUEST_COUNT,
                    AUTO_TPM_CANCEL_CONFIRM_COUNT,
                    AUTO_TPM_CANCEL_TIMEOUT_COUNT,
                    AUTO_TPM_CANCEL_QPS,
                }) {
            monitor.register(metric, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        }
        for (String metric :
                new String[] {
                    AUTO_TPM_SCHEDULE_LATENCY_MS, AUTO_TPM_TTFT_MS, AUTO_TPM_VICTIM_KV_TOKENS,
                }) {
            monitor.register(metric, FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        }
        for (String metric :
                new String[] {
                    AUTO_TPM_PREFILL_QUEUE_DEPTH,
                    AUTO_TPM_DECODE_RUNNING_COUNT,
                    AUTO_TPM_DECODE_ACCEPTED_COUNT,
                    AUTO_TPM_DECODE_ENGINE_LOAD,
                    AUTO_TPM_DECODE_RESERVED_COUNT,
                    AUTO_TPM_DECODE_SHADOW_KV_RESERVED,
                }) {
            monitor.register(metric, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        }
    }

    public void reportRequest(int priority) {
        monitor.report(AUTO_TPM_REQUEST_COUNT, priorityTags(priority), 1.0);
    }

    /**
     * Schedule response latency in milliseconds, tagged by priority and result.
     */
    public void reportScheduleLatency(int priority, String result, long latencyMs) {
        monitor.report(
                AUTO_TPM_SCHEDULE_LATENCY_MS,
                FlexMetricTags.of("priority", String.valueOf(priority), "result", result),
                latencyMs);
    }

    public void reportPrefillQueueDepth(String endpoint, int depth) {
        monitor.report(
                AUTO_TPM_PREFILL_QUEUE_DEPTH, FlexMetricTags.of("endpoint", endpoint), depth);
    }

    /**
     * Master arrival-to-response latency: a lower bound of actual Engine TTFT.
     */
    public void reportTtft(int priority, long latencyMs) {
        monitor.report(AUTO_TPM_TTFT_MS, priorityTags(priority), latencyMs);
    }

    /**
     * Reports eviction plan outcomes with incoming priority, eviction case and result tags.
     */
    public void reportPreemptionPlan(int priority, String evCase, String result) {
        monitor.report(
                AUTO_TPM_EVICTION_PLAN_COUNT,
                FlexMetricTags.of(
                        "priority", String.valueOf(priority), "case", evCase, "result", result),
                1.0);
    }

    public void reportDecodePreemption(
            String endpoint, org.flexlb.worker.endpoint.EndpointStats.DecodeCounts view) {
        FlexMetricTags tags = FlexMetricTags.of("endpoint", endpoint);
        monitor.report(AUTO_TPM_DECODE_RESERVED_COUNT, tags, view.inflight());
        monitor.report(AUTO_TPM_DECODE_SHADOW_KV_RESERVED, tags, view.hardKv());
        monitor.report(AUTO_TPM_DECODE_RUNNING_COUNT, tags, view.running());
        monitor.report(AUTO_TPM_DECODE_ACCEPTED_COUNT, tags, view.accepted());
        monitor.report(AUTO_TPM_DECODE_ENGINE_LOAD, tags, view.totalLoad());
    }

    /**
     * Format captured queue values; this method never reads or changes live scheduling state.
     */
    public static Map<String, Object> placementWait(
            String role,
            String group,
            String endpoint,
            int depth,
            int[] counts,
            String failureDetail,
            Map<String, Object> decision) {
        Map<String, Object> details = new LinkedHashMap<>();
        details.put(
                "cause",
                role
                        + " placement unavailable; "
                        + (failureDetail != null
                                ? failureDetail
                                : decision == null
                                        ? "no detailed decision evidence recorded"
                                        : decision.getOrDefault(
                                                "cause",
                                                decision.getOrDefault(
                                                        "reason", "no failure detail recorded"))));
        details.put("role", role);
        if (group != null) details.put("group", group);
        if (endpoint != null) details.put("endpoint", endpoint);
        details.put("capturedAtMs", System.currentTimeMillis());
        details.put("queueDepth", depth);
        details.put("priorityCounts", priorityCounts(counts));
        if (decision != null) details.put("decision", decision);
        return Collections.unmodifiableMap(details);
    }

    public void reportSelectionResult(RoleType role, boolean success, int code) {
        FlexMetricTags tags =
                FlexMetricTags.of(
                        "role",
                        role.name(),
                        "success",
                        String.valueOf(success),
                        "code",
                        String.valueOf(code));
        monitor.report(ENGINE_BALANCING_MASTER_SELECT_DETAIL, tags, 1.0);
    }

    public void reportSelectedWorker(
            RoleType role, String reason, String address, boolean success, int code) {
        monitor.report(
                ENGINE_BALANCING_MASTER_WORKER_SELECT_DETAIL,
                FlexMetricTags.of(
                        "role",
                        role.name(),
                        "reason",
                        reason == null ? "UNKNOWN" : reason,
                        "engineIp",
                        address,
                        "success",
                        String.valueOf(success),
                        "code",
                        String.valueOf(code)),
                1);
    }

    private static FlexMetricTags[] buildPriorityTags() {
        FlexMetricTags[] tags = new FlexMetricTags[PriorityNormalizer.MAX_PRIORITY + 1];
        for (int priority = 0; priority < tags.length; priority++) {
            tags[priority] = FlexMetricTags.of("priority", String.valueOf(priority));
        }
        return tags;
    }

    private static FlexMetricTags priorityTags(int priority) {
        return priority >= 0 && priority <= PriorityNormalizer.MAX_PRIORITY
                ? PRIORITY_TAGS[priority]
                : FlexMetricTags.of("priority", String.valueOf(priority));
    }

    private static Map<Integer, Integer> priorityCounts(int[] counts) {
        Map<Integer, Integer> values = new LinkedHashMap<>();
        for (int priority = 0; priority < counts.length; priority++) {
            if (counts[priority] > 0) values.put(priority, counts[priority]);
        }
        return Collections.unmodifiableMap(values);
    }
}
