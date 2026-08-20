package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
@JsonSubTypes({
        @JsonSubTypes.Type(value = FifoOrderingConfig.class, name = "FIFO"),
        @JsonSubTypes.Type(value = PriorityOrderingConfig.class, name = "PRIORITY")
})
public sealed interface QueueOrderingConfig permits FifoOrderingConfig, PriorityOrderingConfig {
}
