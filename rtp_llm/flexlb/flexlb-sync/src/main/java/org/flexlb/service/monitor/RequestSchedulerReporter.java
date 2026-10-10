package org.flexlb.service.monitor;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.balance.endpoint.DecodeResources.ResourceSnapshot;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.flexlb.util.PriorityNormalizer;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;

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
        this.monitor = monitor;
    }

    private static FlexMetricTags[] buildPriorityTags() {
        FlexMetricTags[] tags = new FlexMetricTags[
                PriorityNormalizer.MAX_PRIORITY + 1];
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

    @PostConstruct
    public void init() {
        monitor.register(AUTO_TPM_REQUEST_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_SCHEDULE_LATENCY_MS, FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_EVICTION_PLAN_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_EVICTION_COMMIT_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_VICTIM_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_PREFILL_QUEUE_DEPTH, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_TTFT_MS, FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_PRIORITY_PREEMPT_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_DECODE_RUNNING_COUNT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_DECODE_ACCEPTED_COUNT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_CANCEL_REQUEST_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_CANCEL_CONFIRM_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_CANCEL_TIMEOUT_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_CANCEL_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_DECODE_ENGINE_LOAD, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_VICTIM_KV_TOKENS, FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_DECODE_RESERVED_COUNT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_DECODE_SHADOW_KV_RESERVED, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        log.info("RequestSchedulerReporter initialized");
    }

    public void reportRequest(int priority) {
        monitor.report(AUTO_TPM_REQUEST_COUNT, priorityTags(priority), 1.0);
    }

    /** Schedule response latency in milliseconds, tagged by priority and result. */
    public void reportScheduleLatency(int priority, String result, long latencyMs) {
        monitor.report(AUTO_TPM_SCHEDULE_LATENCY_MS,
                FlexMetricTags.of("priority", String.valueOf(priority), "result", result), latencyMs);
    }

    public enum EvictionEvent {
        PLAN(AUTO_TPM_EVICTION_PLAN_COUNT),
        COMMIT(AUTO_TPM_EVICTION_COMMIT_COUNT);

        private final String metric;

        EvictionEvent(String metric) {
            this.metric = metric;
        }
    }

    /** Planning and commit outcomes share incoming priority, eviction case and result tags. */
    public void reportEviction(EvictionEvent event, int priority, String evCase, String result) {
        monitor.report(event.metric,
                FlexMetricTags.of("priority", String.valueOf(priority), "case", evCase, "result", result), 1.0);
    }

    /** Report the victim and its priority-preemption event. */
    public void reportVictim(int victimPriority, int incomingPriority, String stage, String evCase) {
        monitor.report(AUTO_TPM_VICTIM_COUNT,
                FlexMetricTags.of("victim_priority", String.valueOf(victimPriority),
                        "incoming_priority", String.valueOf(incomingPriority),
                        "stage", stage, "case", evCase), 1.0);
        monitor.report(AUTO_TPM_PRIORITY_PREEMPT_COUNT, FlexMetricTags.of("stage", stage), 1.0);
    }

    public void reportPrefillQueueDepth(String endpoint, int depth) {
        monitor.report(AUTO_TPM_PREFILL_QUEUE_DEPTH,
                FlexMetricTags.of("endpoint", endpoint), depth);
    }

    /** Hard KV tokens actually released by one victim. */
    public void reportVictimKvTokens(int victimPriority, String stage, long kvTokens) {
        monitor.report(AUTO_TPM_VICTIM_KV_TOKENS,
                FlexMetricTags.of("victim_priority", String.valueOf(victimPriority),
                        "stage", stage), kvTokens);
    }

    /** Master arrival-to-response latency: a lower bound of actual Engine TTFT. */
    public void reportTtft(int priority, long latencyMs) {
        monitor.report(AUTO_TPM_TTFT_MS,
                priorityTags(priority), latencyMs);
    }

    /** Count one Engine Cancel intent by victim priority and cancellation reason. */
    public void reportCancel(int priority, String reason) {
        monitor.report(AUTO_TPM_CANCEL_QPS,
                FlexMetricTags.of("priority", String.valueOf(priority), "reason", reason), 1.0);
    }

    public enum CancelEvent {
        /** One issued cancel, tagged with the victim's priority. */
        REQUEST(AUTO_TPM_CANCEL_REQUEST_COUNT),
        /** Release confirmation, including after a timed-out wait; uses victim priority. */
        CONFIRM(AUTO_TPM_CANCEL_CONFIRM_COUNT),
        /** Failed plan wait; uses incoming priority and does not release the victim. */
        TIMEOUT(AUTO_TPM_CANCEL_TIMEOUT_COUNT);

        private final String metric;

        CancelEvent(String metric) {
            this.metric = metric;
        }
    }

    public void reportEngineCancel(CancelEvent event, String endpoint, int priority) {
        monitor.report(event.metric,
                FlexMetricTags.of("endpoint", endpoint, "priority", String.valueOf(priority)), 1.0);
    }

    /** Report one snapshot: local reservations, accepted/running Engine tasks and Engine-facing load. */
    public void reportDecodeAdmission(String endpoint, ResourceSnapshot view) {
        FlexMetricTags tags = FlexMetricTags.of("endpoint", endpoint);
        monitor.report(AUTO_TPM_DECODE_RESERVED_COUNT, tags, view.reservedCount());
        monitor.report(AUTO_TPM_DECODE_SHADOW_KV_RESERVED, tags, view.routing().inflightHardKv());
        monitor.report(AUTO_TPM_DECODE_RUNNING_COUNT, tags, view.runningCount());
        monitor.report(AUTO_TPM_DECODE_ACCEPTED_COUNT, tags, view.acceptedCount());
        monitor.report(AUTO_TPM_DECODE_ENGINE_LOAD, tags, view.routing().engineLoad());
    }
}
