package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/** Scheduler mode, diagnostics, and QUEUE-owned settings. */
@Getter
@Setter
public final class SchedulerConfig {

    public enum Type {
        DIRECT,
        QUEUE
    }

    private Type type = Type.QUEUE;
    /** Opt-in diagnostics; available in both scheduler modes. */
    private NettyMemoryReporterConfig nettyMemoryReporter = new NettyMemoryReporterConfig();
    /** Maximum time a request may remain owned by the QUEUE scheduler. */
    private long queueTimeoutMs = 3_600_000L;
    /** Maximum queue entries examined per candidate requested by the planner. */
    private double scanBudgetMultiplier = 2.0;
    private QueueOrderingConfig ordering = new QueueOrderingConfig();
    /** How queued requests form decision groups. */
    private DecisionPolicyConfig decision = new DecisionPolicyConfig();
    /** Optional age-based shedding of requests still waiting for Prefill placement. */
    private PrefillQueueSheddingConfig prefillQueueShedding;

    public static SchedulerConfig direct() {
        SchedulerConfig config = new SchedulerConfig();
        config.type = Type.DIRECT;
        return config;
    }

    public long resolveExpiresAtMs(long admissionTimeMs) {
        // Shedding never extends the queue deadline: its hard wait cap also
        // tightens the absolute scheduling deadline for queued requests.
        long timeoutMs = prefillQueueShedding == null
                ? queueTimeoutMs
                : Math.min(queueTimeoutMs, prefillQueueShedding.getMaxWaitMs());
        return admissionTimeMs > Long.MAX_VALUE - timeoutMs
                ? Long.MAX_VALUE
                : admissionTimeMs + timeoutMs;
    }
}
