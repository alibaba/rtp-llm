package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

import java.util.Optional;

@Getter
@Setter
public final class PriorityOrderingConfig implements QueueOrderingConfig {

    private int defaultPriority = 50;
    private PreemptionConfig preemption;

    @Override
    public Optional<PreemptionConfig> preemptionPolicy() {
        return Optional.ofNullable(preemption);
    }
}
