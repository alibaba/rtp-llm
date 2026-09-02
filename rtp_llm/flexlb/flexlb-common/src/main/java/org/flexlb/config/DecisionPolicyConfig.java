package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/** Defines how requests owned by QUEUE are grouped into one decision. */
@Getter
@Setter
public final class DecisionPolicyConfig {

    public static final int MAX_REQUESTS = 1_024;

    public enum Type {
        SINGLE,
        FIXED_WINDOW
    }

    private Type type = Type.FIXED_WINDOW;
    private int maxRequests = 8;
    private long maxCollectionWaitMs = 300;
    /** Optional execution-time cap for a fixed-window group. */
    private Long maxPredictedExecutionMs;

    public static DecisionPolicyConfig single() {
        DecisionPolicyConfig config = new DecisionPolicyConfig();
        config.type = Type.SINGLE;
        return config;
    }
}
