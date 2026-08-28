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
    private QueueOrderingConfig ordering = new QueueOrderingConfig();
    /** How queued requests form decision groups. */
    private DecisionPolicyConfig decision = new DecisionPolicyConfig();
    private QueueCapacityConfig capacity = new QueueCapacityConfig();
    private RequestLifecycleConfig lifecycle = new RequestLifecycleConfig();

    public static SchedulerConfig direct() {
        SchedulerConfig config = new SchedulerConfig();
        config.type = Type.DIRECT;
        return config;
    }

    public long resolveExpiresAtMs(long admissionTimeMs) {
        return admissionTimeMs > Long.MAX_VALUE - queueTimeoutMs
                ? Long.MAX_VALUE
                : admissionTimeMs + queueTimeoutMs;
    }
}
