package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public final class PriorityOrderingConfig implements QueueOrderingConfig {

    private int defaultPriority = 50;
    private PreemptionConfig preemption;
}
