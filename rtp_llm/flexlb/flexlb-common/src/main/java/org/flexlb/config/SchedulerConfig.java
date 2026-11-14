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
    /** Maximum queue entries examined per candidate requested by the planner. */
    private double scanBudgetMultiplier = 2.0;
    private QueueOrderingConfig ordering = new QueueOrderingConfig();
    /** How queued requests form decision groups. */
    private DecisionPolicyConfig decision = new DecisionPolicyConfig();

}
