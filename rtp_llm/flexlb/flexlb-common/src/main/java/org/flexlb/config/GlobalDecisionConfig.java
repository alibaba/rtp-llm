package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/**
 * Collects requests before worker placement, independently of worker delivery groups.
 */
@Getter
@Setter
public final class GlobalDecisionConfig {

    public static final int DEFAULT_MAX_REQUESTS = 8;
    public static final long DEFAULT_MAX_COLLECTION_WAIT_MS = 5L;
    public static final int DEFAULT_MAX_PLAN_EVALUATIONS = 4096;

    public enum Type {
        SINGLE,
        FIXED_WINDOW
    }

    private Type type = Type.SINGLE;
    private int maxRequests = DEFAULT_MAX_REQUESTS;
    private long maxCollectionWaitMs = DEFAULT_MAX_COLLECTION_WAIT_MS;
    /**
     * Per-phase bound for completion search and each local plan-improvement pass.
     */
    private int maxPlanEvaluations = DEFAULT_MAX_PLAN_EVALUATIONS;
}
