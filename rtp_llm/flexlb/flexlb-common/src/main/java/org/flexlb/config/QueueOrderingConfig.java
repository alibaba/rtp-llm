package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

import java.util.Optional;

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
@JsonSubTypes({
        @JsonSubTypes.Type(value = FifoOrderingConfig.class, name = "FIFO"),
        @JsonSubTypes.Type(value = PriorityOrderingConfig.class, name = "PRIORITY")
})
public sealed interface QueueOrderingConfig permits FifoOrderingConfig, PriorityOrderingConfig {

    /** Optional eviction capability supplied by this ordering policy. */
    default Optional<PreemptionConfig> preemptionPolicy() {
        return Optional.empty();
    }
}
