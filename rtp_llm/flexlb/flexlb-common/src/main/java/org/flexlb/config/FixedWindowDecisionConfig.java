package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/** Forms bounded request groups with a maximum collection window. */
@Getter
@Setter
public final class FixedWindowDecisionConfig implements DecisionPolicyConfig {

    /**
     * Upper bound for one decision group. Planning evaluates each growing
     * prefix because configured predictors are not required to be monotonic;
     * bounding the group is therefore part of the public CPU-cost contract.
     */
    public static final int MAX_REQUESTS = 1_024;

    private int maxRequests = 8;
    private long maxCollectionWaitMs = 300;

    /**
     * Prevents adding a request when the resulting group's predicted execution
     * time would exceed this value. Reaching the value releases the group
     * without waiting for the collection window. An indivisible singleton may
     * exceed it.
     */
    private Long maxPredictedExecutionMs;
}
