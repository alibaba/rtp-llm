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
import static org.flexlb.constant.MetricConstant.AUTO_TPM_DEADLINE_MISS_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_EVICTION_COMMIT_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_EVICTION_PLAN_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_INFLIGHT_SETTLE_MISS;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_NORMAL_PLACEMENT_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_PLAN_AGE_MS;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_PLAN_CONFLICT_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_PREFILL_QUEUE_DEPTH;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_PRIORITY_PREEMPT_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_REQUEST_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_REQUEST_SLO_MS;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_SCHEDULE_LATENCY_MS;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_TTFT_MS;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_VICTIM_COUNT;
import static org.flexlb.constant.MetricConstant.AUTO_TPM_VICTIM_KV_TOKENS;

/**
 * Auto-TPM priority scheduling metrics reporter.
 *
 * <p>Phase 0 observability: per-priority request count, per-request SLO,
 * schedule latency by result, and normal-placement success count. Reported
 * for both the legacy path and the priority scheduler path so that enabling
 * {@code AUTO_TPM_ENABLED} can be compared against the baseline.
 */
@Slf4j
@Component
public class PrioritySchedulerReporter {

    private final FlexMonitor monitor;

    @Autowired
    public PrioritySchedulerReporter(FlexMonitor monitor) {
        this.monitor = monitor;
    }

    @PostConstruct
    public void init() {
        monitor.register(AUTO_TPM_REQUEST_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_REQUEST_SLO_MS, FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_SCHEDULE_LATENCY_MS, FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_NORMAL_PLACEMENT_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_EVICTION_PLAN_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_EVICTION_COMMIT_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_VICTIM_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_PLAN_CONFLICT_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_PREFILL_QUEUE_DEPTH, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_TTFT_MS, FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_DEADLINE_MISS_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_PRIORITY_PREEMPT_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_DECODE_RUNNING_COUNT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_DECODE_ACCEPTED_COUNT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_CANCEL_REQUEST_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_CANCEL_CONFIRM_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_CANCEL_TIMEOUT_COUNT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_CANCEL_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_PLAN_AGE_MS, FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_DECODE_ENGINE_LOAD, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(AUTO_TPM_INFLIGHT_SETTLE_MISS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        log.info("PrioritySchedulerReporter initialized (21 metrics)");
    }

    /**
     * Report request arrival with its per-request SLO via
     * {@code auto_tpm.request.count} and {@code auto_tpm.request.slo_ms}.
     *
     * @param priority     normalized priority (30/40/50/60/70)
     * @param seqBucket    SLO length-bucket label of the request seqLen
     * @param requestSloMs per-request SLO in ms
     */
    public void reportRequest(int priority, String seqBucket, long requestSloMs) {
        String priorityTag = String.valueOf(priority);
        monitor.report(AUTO_TPM_REQUEST_COUNT,
                FlexMetricTags.of("priority", priorityTag), 1.0);
        monitor.report(AUTO_TPM_REQUEST_SLO_MS,
                FlexMetricTags.of("priority", priorityTag, "seq_bucket", seqBucket), requestSloMs);
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
     * Report a successful normal P/D placement (no eviction involved) via
     * {@code auto_tpm.normal_placement.count}.
     */
    public void reportNormalPlacement(int priority) {
        monitor.report(AUTO_TPM_NORMAL_PLACEMENT_COUNT,
                FlexMetricTags.of("priority", String.valueOf(priority)), 1.0);
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
     * Report an optimistic-concurrency plan conflict via
     * {@code auto_tpm.plan_conflict.count}.
     *
     * @param conflictCase conflict case label (e.g. prefill_queue_version)
     */
    public void reportPlanConflict(String conflictCase) {
        monitor.report(AUTO_TPM_PLAN_CONFLICT_COUNT,
                FlexMetricTags.of("case", conflictCase), 1.0);
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
                FlexMetricTags.of("priority", String.valueOf(priority)), latencyMs);
    }

    /**
     * Report one deadline miss via {@code auto_tpm.deadline_miss.count}
     * (schedule completion exceeded the request deadlineMs, §19.2).
     */
    public void reportDeadlineMiss(int priority) {
        monitor.report(AUTO_TPM_DEADLINE_MISS_COUNT,
                FlexMetricTags.of("priority", String.valueOf(priority)), 1.0);
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
     * Phase 5 this gauge equalled confirmedRunningCount (accepted + running
     * merged); it now counts only engine-reported {@code RUNNING} tasks —
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
     * merged confirmedRunningCount without changing its total.
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
     * @param priority normalized priority of the cancelled request; 0 when
     *                 the request carried no Auto-TPM budget
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
     * @param victimPriority normalized priority of the cancelled victim; 0
     *                       when the settled item carried no Auto-TPM budget
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
     * Report the age of a committed plan (creation → commit success) via
     * {@code auto_tpm.plan_age_ms} (N3 observability). Quantifies how stale
     * a lockfree commit's plan view was when it landed.
     */
    public void reportPlanAge(int priority, long ageMs) {
        monitor.report(AUTO_TPM_PLAN_AGE_MS,
                FlexMetricTags.of("priority", String.valueOf(priority)), ageMs);
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
