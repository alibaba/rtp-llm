package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/** Forms bounded request groups within a fixed collection window. */
@Getter
@Setter
public final class FixedWindowDecisionConfig implements DecisionPolicyConfig {

    private int maxRequests = 8;
    private long maxCollectionWaitMs = 300;

    /**
     * Prevents adding a request when the resulting group's predicted execution
     * time would exceed this value. An indivisible singleton may exceed it.
     */
    private Long maxPredictedExecutionMs;
}
