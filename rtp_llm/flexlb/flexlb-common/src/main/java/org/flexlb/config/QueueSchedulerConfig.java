package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public final class QueueSchedulerConfig implements SchedulerConfig {

    /** Maximum time a request may remain owned by the QUEUE scheduler. */
    private long queueTimeoutMs = 3_600_000L;
    private QueueOrderingConfig ordering = new FifoOrderingConfig();
    /**
     * How queued requests form decision groups. When omitted in schema v1,
     * FlexlbConfig maps the legacy dispatcher fields to an effective policy.
     */
    private DecisionPolicyConfig decision;
    private QueueCapacityConfig capacity = new QueueCapacityConfig();
    private RequestLifecycleConfig lifecycle = new RequestLifecycleConfig();

    public long resolveExpiresAtMs(long admissionTimeMs) {
        return admissionTimeMs > Long.MAX_VALUE - queueTimeoutMs
                ? Long.MAX_VALUE
                : admissionTimeMs + queueTimeoutMs;
    }
}
