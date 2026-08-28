package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;

/** Defines how requests owned by QUEUE are grouped into one decision. */
@JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
@JsonSubTypes({
        @JsonSubTypes.Type(value = SingleDecisionConfig.class, name = "SINGLE"),
        @JsonSubTypes.Type(value = FixedWindowDecisionConfig.class, name = "FIXED_WINDOW")
})
public sealed interface DecisionPolicyConfig
        permits SingleDecisionConfig, FixedWindowDecisionConfig {
}
