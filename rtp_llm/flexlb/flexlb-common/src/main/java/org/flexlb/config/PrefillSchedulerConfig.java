package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
@JsonSubTypes({
    @JsonSubTypes.Type(value = FlexlbPrefillSchedulerConfig.class, name = "FLEXLB"),
    @JsonSubTypes.Type(value = NaviSchedPrefillSchedulerConfig.class, name = "NAVI_SCHED")
})
public sealed interface PrefillSchedulerConfig
        permits FlexlbPrefillSchedulerConfig, NaviSchedPrefillSchedulerConfig {}
