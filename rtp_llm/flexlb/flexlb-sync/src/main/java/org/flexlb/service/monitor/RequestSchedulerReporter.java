package org.flexlb.service.monitor;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
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
import static org.flexlb.constant.MetricConstant.AUTO_TPM_INFLIGHT_SETTLE_MISS;
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

    /** Highest normalized priority level; see {@code PriorityNormalizer}. */
    private static final int MAX_PRIORITY = 100;

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
        FlexMetricTags[] tags = new FlexMetricTags[MAX_PRIORITY + 1];
        for (int priority = 0; priority < tags.length; priority++) {
            tags[priority] = FlexMetricTags.of("priority", String.valueOf(priority));
        }
        return tags;
    }

    private static FlexMetricTags priorityTags(int priority) {
        return priority >= 0 && priority <= MAX_PRIORITY
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
        monitor.register(AUTO_TPM_INFLIGHT_SETTLE_MISS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_VICTIM_KV_TOKENS, FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_DECODE_RESERVED_COUNT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_DECODE_SHADOW_KV_RESERVED, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        log.info("RequestSchedulerReporter initialized");
    }

    /**
     * Report request arrival by normalized priority.
     */
    public void reportRequest(int priority) {
        monitor.report(AUTO_TPM_REQUEST_COUNT, priorityTags(priority), 1.0);
    }

    /**
     * Report end-to-end schedule latency via {@code auto_tpm.schedule.latency_ms}.
     *
     * @param priority  normalized priority
     * @param result    schedule result label (e.g. success / no_available_worker)
     * @param latencyMs milliseconds from request arrival to schedule response
     */
    public void reportScheduleLatency(int priority, String result, long latencyMs) {
        monitor.report(AUTO_TPM_SCHEDULE_LATENCY_MS,
                FlexMetricTags.of("priority", String.valueOf(priority), "result", result), latencyMs);
    }

    /**
     * Report an inflight settle miss via {@code auto_tpm.inflight_settle_miss.count}
     * (review P2-2): a finishYielded/PreemptedById found no inflight entry.
     *
     * @param kind settle kind ("yielded" / "preempted")
     */
    public void reportInflightSettleMiss(String kind) {
        monitor.report(AUTO_TPM_INFLIGHT_SETTLE_MISS,
                FlexMetricTags.of("kind", kind), 1.0);
    }

    /**
     * Report an eviction plan generation outcome via
     * {@code auto_tpm.eviction_plan.count}.
     *
     * @param priority incoming request priority
     * @param evCase   eviction case label (e.g. prefill_queue_full)
     * @param result   plan result label (feasible / infeasible)
     */
    public void reportEvictionPlan(int priority, String evCase, String result) {
        monitor.report(AUTO_TPM_EVICTION_PLAN_COUNT,
                FlexMetricTags.of("priority", String.valueOf(priority), "case", evCase, "result", result), 1.0);
    }

    /**
     * Report an eviction plan commit outcome via
     * {@code auto_tpm.eviction_commit.count}.
     *
     * @param priority incoming request priority
     * @param evCase   eviction case label (e.g. prefill_queue_full)
     * @param result   commit result label (success / version_mismatch / partial_failure)
     */
    public void reportEvictionCommit(int priority, String evCase, String result) {
        monitor.report(AUTO_TPM_EVICTION_COMMIT_COUNT,
                FlexMetricTags.of("priority", String.valueOf(priority), "case", evCase, "result", result), 1.0);
    }

    /**
     * Report one evicted victim via {@code auto_tpm.victim.count}.
     *
     * @param victimPriority   priority of the evicted request
     * @param incomingPriority priority of the request that displaced it
     * @param stage            victim scheduling stage (e.g. prefill_queued)
     * @param evCase           eviction case label (e.g. prefill_queue_full)
     */
    public void reportVictim(int victimPriority, int incomingPriority, String stage, String evCase) {
        monitor.report(AUTO_TPM_VICTIM_COUNT,
                FlexMetricTags.of("victim_priority", String.valueOf(victimPriority),
                        "incoming_priority", String.valueOf(incomingPriority),
                        "stage", stage, "case", evCase), 1.0);
    }

    /**
     * Report a prefill batcher queue depth via
     * {@code auto_tpm.prefill.queue_depth}.
     */
    public void reportPrefillQueueDepth(String endpoint, int depth) {
        monitor.report(AUTO_TPM_PREFILL_QUEUE_DEPTH,
                FlexMetricTags.of("endpoint", endpoint), depth);
    }

    /**
     * Report one victim's released hard KV tokens via
     * {@code auto_tpm.victim.kv_tokens} (design doc 19.2).
     *
     * @param victimPriority priority of the evicted request
     * @param stage          victim scheduling stage (e.g. decode_reserved)
     * @param kvTokens       hard KV tokens released by the eviction
     */
    public void reportVictimKvTokens(int victimPriority, String stage, long kvTokens) {
        monitor.report(AUTO_TPM_VICTIM_KV_TOKENS,
                FlexMetricTags.of("victim_priority", String.valueOf(victimPriority),
                        "stage", stage), kvTokens);
    }

    /**
     * Report a decode endpoint's shadow reservation count via
     * {@code auto_tpm.decode.reserved.count}.
     */
    public void reportDecodeReservedCount(String endpoint, int count) {
        monitor.report(AUTO_TPM_DECODE_RESERVED_COUNT,
                FlexMetricTags.of("endpoint", endpoint), count);
    }

    /**
     * Report a decode endpoint's shadow hard-KV reservation total via
     * {@code auto_tpm.decode.shadow_kv_reserved}.
     */
    public void reportDecodeShadowKvReserved(String endpoint, long kvTokens) {
        monitor.report(AUTO_TPM_DECODE_SHADOW_KV_RESERVED,
                FlexMetricTags.of("endpoint", endpoint), kvTokens);
    }

    /**
     * Report the TTFT approximation via {@code auto_tpm.ttft_ms} (§19.2).
     * Approximated as "request arrival → schedule completion" on the Master;
     * the engine-side first-token time is not observable here, so this is a
     * lower bound of the real TTFT.
     */
    public void reportTtft(int priority, long latencyMs) {
        monitor.report(AUTO_TPM_TTFT_MS,
                priorityTags(priority), latencyMs);
    }

    /**
     * Report one priority preemption via
     * {@code auto_tpm.priority_preempt.count} (§19.2).
     *
     * @param stage victim scheduling stage (prefill_queued / decode_reserved)
     */
    public void reportPriorityPreempt(String stage) {
        monitor.report(AUTO_TPM_PRIORITY_PREEMPT_COUNT,
                FlexMetricTags.of("stage", stage), 1.0);
    }

    /**
     * Report a decode endpoint's true running-layer request count via
     * {@code auto_tpm.decode.running.count} (Phase 5 layered view). Before
     * Phase 5 this gauge equalled the merged confirmed Engine-owned count
     * (accepted + running merged); it now counts only engine-reported
     * {@code RUNNING} tasks —
     * the accepted layer is reported separately via
     * {@link #reportDecodeAcceptedCount}.
     */
    public void reportDecodeRunningCount(String endpoint, int count) {
        monitor.report(AUTO_TPM_DECODE_RUNNING_COUNT,
                FlexMetricTags.of("endpoint", endpoint), count);
    }

    /**
     * Report a decode endpoint's accepted-not-running (engine KV-allocated)
     * request count via {@code auto_tpm.decode.accepted.count} (Phase 5
     * layered view). Together with the running gauge this splits the former
     * merged confirmed Engine-owned count without changing its total.
     */
    public void reportDecodeAcceptedCount(String endpoint, int count) {
        monitor.report(AUTO_TPM_DECODE_ACCEPTED_COUNT,
                FlexMetricTags.of("endpoint", endpoint), count);
    }

    /**
     * Report one cancel initiation via {@code auto_tpm.cancel.qps} — counted
     * once per cancel intent injected into the EngineCancelChannel, tagged
     * with the cancelled request's priority and the cancel reason
     * (PRIORITY_PREEMPTED / USER_CANCELLED / DEADLINE_EXCEEDED).
     *
     * @param priority normalized priority of the cancelled request
     * @param reason   cancel reason metric label
     */
    public void reportCancel(int priority, String reason) {
        monitor.report(AUTO_TPM_CANCEL_QPS,
                FlexMetricTags.of("priority", String.valueOf(priority), "reason", reason), 1.0);
    }

    /**
     * Report one engine cancel request issued via
     * {@code auto_tpm.cancel.request.count} (Phase 5 accepted eviction).
     *
     * @param victimPriority normalized priority of the cancelled victim
     */
    public void reportCancelRequest(String endpoint, int victimPriority) {
        monitor.report(AUTO_TPM_CANCEL_REQUEST_COUNT,
                FlexMetricTags.of("endpoint", endpoint,
                        "priority", String.valueOf(victimPriority)), 1.0);
    }

    /**
     * Report one cancel release confirmation via
     * {@code auto_tpm.cancel.confirm.count} — inside the commit wait window
     * or via the later WorkerStatus settle path, whichever happens.
     *
     * @param victimPriority normalized priority of the cancelled victim
     */
    public void reportCancelConfirm(String endpoint, int victimPriority) {
        monitor.report(AUTO_TPM_CANCEL_CONFIRM_COUNT,
                FlexMetricTags.of("endpoint", endpoint,
                        "priority", String.valueOf(victimPriority)), 1.0);
    }

    /**
     * Report one cancel wait-window timeout via
     * {@code auto_tpm.cancel.timeout.count}. The plan failed; the victim
     * stays CANCEL_REQUESTED until WorkerStatus settles it (a later settle
     * also counts one confirm, so confirms may exceed non-timed-out requests).
     *
     * @param incomingPriority normalized priority of the incoming request
     *                         whose eviction plan failed
     */
    public void reportCancelTimeout(String endpoint, int incomingPriority) {
        monitor.report(AUTO_TPM_CANCEL_TIMEOUT_COUNT,
                FlexMetricTags.of("endpoint", endpoint,
                        "priority", String.valueOf(incomingPriority)), 1.0);
    }

    /**
     * Report a decode endpoint's engine-facing load (dispatched, excludes
     * queued-phase shadow reservations) via {@code auto_tpm.decode.engine_load}
     * (N2 observability). Contrast against the reserved-count gauge to watch
     * root cause C (shadow saturation while the engine is idle).
     */
    public void reportDecodeEngineLoad(String endpoint, int load) {
        monitor.report(AUTO_TPM_DECODE_ENGINE_LOAD,
                FlexMetricTags.of("endpoint", endpoint), load);
    }
}
