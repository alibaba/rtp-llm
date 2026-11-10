package org.flexlb.service.monitor;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRequestView;
import org.flexlb.balance.endpoint.DecodeResources.DecodeRoutingView;
import org.flexlb.balance.endpoint.DecodeResources.ResourceSnapshot;
import org.flexlb.balance.endpoint.EndpointStats;
import org.flexlb.enums.DecodeTaskPhase;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.metric.SafeFlexMonitor;
import org.flexlb.util.Logger;
import org.flexlb.util.PriorityNormalizer;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.flexlb.constant.MetricConstant.AUTO_TPM_CANCEL_CONFIRM_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_CANCEL_QPS;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_CANCEL_REQUEST_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_CANCEL_TIMEOUT_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_DECODE_ACCEPTED_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_DECODE_ENGINE_LOAD;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_DECODE_RESERVED_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_DECODE_RUNNING_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_DECODE_SHADOW_KV_RESERVED;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_EVICTION_COMMIT_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_EVICTION_PLAN_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_PREFILL_QUEUE_DEPTH;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_PRIORITY_PREEMPT_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_REQUEST_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_SCHEDULE_LATENCY_MS;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_TTFT_MS;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_VICTIM_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_VICTIM_KV_TOKENS;

/**
 * Auto-TPM priority scheduling metrics reporter.
 *
 * <p>Observability for per-priority request volume, schedule latency,
 * placement, preemption, cancellation, and resource ownership.
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

    /** Schedule response latency in milliseconds, tagged by priority and result. */
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

    /** Master arrival-to-response latency: a lower bound of actual Engine TTFT. */
    public void reportTtft(int priority, long latencyMs) {
        monitor.report(AUTO_TPM_TTFT_MS, priorityTags(priority), latencyMs);
    }

    /** Report one snapshot: local reservations, accepted/running Engine tasks and Engine-facing load. */
    /** Read committed requestToPreempt facts; neither publication nor metric failure changes the attempt. */
    public void reportPreemptedRequests(
            String requestId,
            int incomingPriority,
            String endpoint,
            String preemptionReason,
            List<DecodeRequestView> requestsToPreempt,
            boolean cancelled) {
        for (var requestToPreempt : requestsToPreempt) {
            String stage =
                    !cancelled
                            ? "decode_reserved"
                            : requestToPreempt.phase() == DecodeTaskPhase.RUNNING
                                    ? "decode_running"
                                    : "decode_cancel";
            try {
                reportPreemptedRequest(
                        requestToPreempt.priority(), incomingPriority, stage, preemptionReason);
                reportPreemptedKvTokens(
                        requestToPreempt.priority(), stage, requestToPreempt.kvTokens());
                if (cancelled)
                    reportEngineCancel(CancelEvent.CONFIRM, endpoint, requestToPreempt.priority());
            } catch (Throwable failure) {
                Logger.warn(
                        "[decode-capacity] failed to report victim: request_id={}",
                        requestId,
                        failure);
            }
            if (!cancelled) {
                Logger.debug(
                        "[decode-capacity] decode victim preempted: victim_id={} victim_priority={}"
                                + " stage={} outcome={} kv_tokens={} incoming_id={} incoming_priority={} worker={}",
                        requestToPreempt.requestId(),
                        requestToPreempt.priority(),
                        stage,
                        "requeued",
                        requestToPreempt.kvTokens(),
                        requestId,
                        incomingPriority,
                        endpoint);
            }
        }
        reportPreemption(PreemptionEvent.COMMIT, incomingPriority, preemptionReason, "success");
    }

    /** Report the requestToPreempt and its priority-preemption event. */
    public void reportPreemptedRequest(
            int requestToPreemptPriority, int incomingPriority, String stage, String evCase) {
        monitor.report(
                AUTO_TPM_VICTIM_COUNT,
                FlexMetricTags.of(
                        "victim_priority",
                        String.valueOf(requestToPreemptPriority),
                        "incoming_priority",
                        String.valueOf(incomingPriority),
                        "stage",
                        stage,
                        "case",
                        evCase),
                1.0);
        monitor.report(AUTO_TPM_PRIORITY_PREEMPT_COUNT, FlexMetricTags.of("stage", stage), 1.0);
    }

    /** Hard KV tokens actually released by one requestToPreempt. */
    public void reportPreemptedKvTokens(int requestToPreemptPriority, String stage, long kvTokens) {
        monitor.report(
                AUTO_TPM_VICTIM_KV_TOKENS,
                FlexMetricTags.of(
                        "victim_priority",
                        String.valueOf(requestToPreemptPriority),
                        "stage",
                        stage),
                kvTokens);
    }

    public void reportEngineCancel(CancelEvent event, String endpoint, int priority) {
        monitor.report(
                event.metric,
                FlexMetricTags.of("endpoint", endpoint, "priority", String.valueOf(priority)),
                1.0);
    }

    /** Planning and commit outcomes share incoming priority, eviction case and result tags. */
    public void reportPreemption(
            PreemptionEvent event, int priority, String evCase, String result) {
        monitor.report(
                event.metric,
                FlexMetricTags.of(
                        "priority", String.valueOf(priority), "case", evCase, "result", result),
                1.0);
    }

    public void reportPreemptionCancelRequests(
            String endpoint, List<DecodeRequestView> requestsToPreempt) {
        for (var requestToPreempt : requestsToPreempt) {
            reportEngineCancel(CancelEvent.REQUEST, endpoint, requestToPreempt.priority());
            reportCancel(requestToPreempt.priority(), "PRIORITY_PREEMPTED");
        }
    }

    /** Count one Engine Cancel intent by requestToPreempt priority and cancellation reason. */
    public void reportCancel(int priority, String reason) {
        monitor.report(
                AUTO_TPM_CANCEL_QPS,
                FlexMetricTags.of("priority", String.valueOf(priority), "reason", reason),
                1.0);
    }

    public void reportDecodePreemption(String endpoint, ResourceSnapshot view) {
        FlexMetricTags tags = FlexMetricTags.of("endpoint", endpoint);
        monitor.report(AUTO_TPM_DECODE_RESERVED_COUNT, tags, view.reservedCount());
        monitor.report(AUTO_TPM_DECODE_SHADOW_KV_RESERVED, tags, view.routing().inflightHardKv());
        monitor.report(AUTO_TPM_DECODE_RUNNING_COUNT, tags, view.runningCount());
        monitor.report(AUTO_TPM_DECODE_ACCEPTED_COUNT, tags, view.acceptedCount());
        monitor.report(AUTO_TPM_DECODE_ENGINE_LOAD, tags, view.routing().engineLoad());
    }

    /** Format captured queue values; this method never reads or changes live scheduling state. */
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

    /**
     * Maps eviction planning and commit events to their exported counters.
     */
    public enum PreemptionEvent {
        PLAN(AUTO_TPM_EVICTION_PLAN_COUNT),
        COMMIT(AUTO_TPM_EVICTION_COMMIT_COUNT);

        private final String metric;

        PreemptionEvent(String metric) {
            this.metric = metric;
        }
    }

    /**
     * Maps cancellation requests, confirmations and timeouts to their exported counters.
     */
    public enum CancelEvent {
        /** One issued cancel, tagged with the requestToPreempt's priority. */
        REQUEST(AUTO_TPM_CANCEL_REQUEST_COUNT),
        /** Release confirmation, including after a timed-out wait; uses requestToPreempt priority. */
        CONFIRM(AUTO_TPM_CANCEL_CONFIRM_COUNT),
        /** Failed plan wait; uses incoming priority and does not release the requestToPreempt. */
        TIMEOUT(AUTO_TPM_CANCEL_TIMEOUT_COUNT);

        private final String metric;

        CancelEvent(String metric) {
            this.metric = metric;
        }
    }

    /**
     * Frozen decision evidence; the log representation is materialized only on failure.
     */
    @lombok.RequiredArgsConstructor(access = lombok.AccessLevel.PUBLIC)
    public static final class QueueWaitSnapshot {
        private final String endpoint;
        private final String cause;
        private final String headRequestId;
        private final long headEnqueueSeq;
        private final long capturedAtMs = System.currentTimeMillis();
        private final EndpointStats.QueueCounts counters;
        private final DecodeRoutingView decode;
        private volatile Map<String, Object> snapshotValues;

        public Map<String, Object> asMap() {
            Map<String, Object> cached = snapshotValues;
            if (cached != null) {
                return cached;
            }
            int[] requestCountsByPriority = counters.byPriority();
            int depth = java.util.Arrays.stream(requestCountsByPriority).sum();
            Map<String, Object> details = new LinkedHashMap<>();
            details.put("cause", cause);
            details.put("headRequestId", headRequestId);
            details.put("headEnqueueSeq", headEnqueueSeq);
            details.put("capturedAtMs", capturedAtMs);
            details.put("endpoint", endpoint);
            details.put("queueDepth", depth);
            details.put("queueVersion", counters.version());
            details.put("priorityCounts", priorityCounts(requestCountsByPriority));
            details.put("prefillRequests", counters.outstandingRequests());
            details.put("prefillBatchSlots", counters.batchSlots());
            if (decode != null) {
                details.put(
                        "decode",
                        Map.of(
                                "endpoint",
                                decode.address(),
                                "version",
                                decode.admissionVersion(),
                                "engineLoad",
                                decode.engineLoad(),
                                "totalLoad",
                                decode.totalLoad(),
                                "kvTotal",
                                decode.totalKv(),
                                "kvAvailable",
                                decode.placementUsage().hardKvAvailable()));
            }
            cached = Collections.unmodifiableMap(details);
            snapshotValues = cached;
            return cached;
        }
    }
}
