package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public final class QueueSchedulerConfig implements SchedulerConfig {

    /** Maximum time a request may remain owned by the QUEUE scheduler. */
    private long queueTimeoutMs = 3_600_000L;
    private QueueOrderingConfig ordering = new FifoOrderingConfig();
    /** How queued requests form decision groups. */
    private DecisionPolicyConfig decision = new FixedWindowDecisionConfig();
    private QueueCapacityConfig capacity = new QueueCapacityConfig();
    private RequestLifecycleConfig lifecycle = new RequestLifecycleConfig();

    public long resolveExpiresAtMs(long admissionTimeMs) {
        return admissionTimeMs > Long.MAX_VALUE - queueTimeoutMs
                ? Long.MAX_VALUE
                : admissionTimeMs + queueTimeoutMs;
    }
}
