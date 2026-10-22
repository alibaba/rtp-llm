package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/** Scheduler mode and QUEUE-owned settings. */
@Getter
@Setter
public final class SchedulerConfig {

    public enum Type {
        DIRECT,
        QUEUE
    }

    private Type type = Type.QUEUE;
    /** Maximum time a request may remain owned by the QUEUE scheduler. */
    private long queueTimeoutMs = 3_600_000L;
    /** Maximum queue entries examined per candidate requested by the planner. */
    private double scanBudgetMultiplier = 2.0;
    private PrefillQueueSheddingConfig prefillQueueShedding;
    private QueueOrderingConfig ordering = new QueueOrderingConfig();
    /** How queued requests form decision groups. */
    private DecisionPolicyConfig decision = new DecisionPolicyConfig();

    public long resolveExpiresAtMs(long admissionTimeMs) {
        long timeoutMs = prefillQueueShedding == null ? queueTimeoutMs
                : Math.min(queueTimeoutMs, prefillQueueShedding.getMaxWaitMs());
        return admissionTimeMs > Long.MAX_VALUE - timeoutMs
                ? Long.MAX_VALUE
                : admissionTimeMs + timeoutMs;
    }
}
