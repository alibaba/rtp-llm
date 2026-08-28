package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/** Defines how requests owned by QUEUE are grouped into one decision. */
@Getter
@Setter
public final class DecisionPolicyConfig {

    public static final int MIN_REQUESTS = 1;
    public static final int DEFAULT_MAX_REQUESTS = 8;
    public static final long DEFAULT_MAX_COLLECTION_WAIT_MS = 300L;

    public enum Type {
        SINGLE,
        FIXED_WINDOW
    }

    private Type type = Type.FIXED_WINDOW;
    private int maxRequests = DEFAULT_MAX_REQUESTS;
    private long maxCollectionWaitMs = DEFAULT_MAX_COLLECTION_WAIT_MS;
    /** Optional execution-time cap for a fixed-window group. */
    private Long maxPredictedExecutionMs;

    /**
     * Request-group size for one fixed-window decision. BATCH uses the group
     * size to refill endpoint batch windows; NON_BATCH delivery capacity is
     * independently owned by
     * {@code dispatcher.maxInflightRequestsPerPrefillWorker}. A decision
     * always contains at least one request. SINGLE keeps a one-request
     * decision and does not use this target for look-ahead.
     */
    public int resolveMaxRequests() {
        return Math.max(MIN_REQUESTS, maxRequests);
    }

    public static DecisionPolicyConfig single() {
        DecisionPolicyConfig config = new DecisionPolicyConfig();
        config.type = Type.SINGLE;
        return config;
    }
}
