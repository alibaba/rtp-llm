package org.flexlb.config.scheduler;

import lombok.Getter;
import lombok.Setter;

/**
 * Composes business-priority arbitration with a separate request ordering policy.
 */
@Getter
@Setter
public final class QueueOrderingConfig {
    private Priority priority = new Priority();
    private Policy policy = new Policy();

    public enum PolicyType {
        FIFO,
        PREFILL_COST
    }

    /**
     * Enables priority arbitration and its default preemption settings independently of request order.
     */
    @Getter
    @Setter
    public static final class Priority {
        private boolean enabled;
        private int defaultPriority = 50;
        private PreemptionConfig preemption;

        public void setEnabled(boolean enabled) {
            this.enabled = enabled;
            if (enabled && preemption == null) preemption = new PreemptionConfig();
            if (!enabled) preemption = null;
        }
    }

    /**
     * Selects FIFO or cached Prefill-cost ordering within each priority level.
     */
    @Getter
    @Setter
    public static final class Policy {
        private PolicyType type = PolicyType.FIFO;
    }
}
