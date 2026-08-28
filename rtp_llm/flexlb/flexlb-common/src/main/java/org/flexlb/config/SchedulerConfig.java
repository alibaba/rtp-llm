package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
@JsonSubTypes({
        @JsonSubTypes.Type(value = DirectSchedulerConfig.class, name = "DIRECT"),
        @JsonSubTypes.Type(value = QueueSchedulerConfig.class, name = "QUEUE"),
        @JsonSubTypes.Type(value = NaviBatchSchedulerConfig.class, name = "NAVI_BATCH")
})
public sealed interface SchedulerConfig
        permits DirectSchedulerConfig, QueueSchedulerConfig, NaviBatchSchedulerConfig {
}
